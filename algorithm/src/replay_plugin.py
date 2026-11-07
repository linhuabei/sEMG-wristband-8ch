# -*- coding: utf-8 -*-
r"""端到端回放测试：把 NinaPro 原始信号按“上位机的方式”喂给算法插件。

它复刻了 上位机/core/signal_processor.py 的流式链路
（20-450 Hz 带通 → 50 Hz 陷波 → 直流阻断 → 每 50 ms 出一个特征块），
然后把特征块按顺序丢给 plugins 里的 .py 插件，逐拍比较“插件输出”和“真实标签”。

这样一来，评估的就是**上位机里实际会跑的那条链路**，而不是训练脚本里的离线流程。

用上位机自带的解释器跑（那个环境只有 numpy/scipy/PyQt5，没有 torch）：
    上位机\.venv\Scripts\python.exe replay_plugin.py --subject 3 --duration 120
"""
import argparse
import importlib.util
import json
import sys
from pathlib import Path

import numpy as np
from scipy import signal

from config import APP_GESTURE_MAP, APP_GESTURE_NAMES, Config, PLUGIN_DIR
from ninapro import load_subject

# 上位机信号处理器参数（与 core/signal_processor.py 一致）
CHANNELS = 8


class StreamingSignalProcessor:
    """复刻上位机的流式滤波 + 50 ms 滑窗特征块。"""

    def __init__(self, fs=2000, channels=CHANNELS):
        self.fs = fs
        self.channels = channels
        nyq = fs / 2.0
        self.sos_bp = signal.butter(4, [20 / nyq, min(0.98, 450 / nyq)],
                                    btype="band", output="sos")
        b, a = signal.iirnotch(50 / nyq, 30)
        self.sos_notch = signal.tf2sos(b, a)
        self.zi_bp = np.tile(signal.sosfilt_zi(self.sos_bp)[..., None], (1, 1, channels))
        self.zi_notch = np.tile(signal.sosfilt_zi(self.sos_notch)[..., None], (1, 1, channels))
        self.zi_dc = np.zeros((1, channels))
        self.win = max(8, fs // 20)          # 50 ms
        self.buf = np.zeros((self.win, channels))
        self.idx = 0
        self.tick = 0

    def push(self, sample):
        """sample: (8,) -> 返回 50 ms 特征块（每 50 ms 一次），否则 None。"""
        x = np.asarray(sample, dtype=np.float64).reshape(1, -1)
        y, self.zi_bp = signal.sosfilt(self.sos_bp, x, axis=0, zi=self.zi_bp)
        y, self.zi_notch = signal.sosfilt(self.sos_notch, y, axis=0, zi=self.zi_notch)
        y, self.zi_dc = signal.lfilter([1.0, -1.0], [1.0, -0.995], y, axis=0, zi=self.zi_dc)
        y = np.clip(y, -8.0, 8.0)[0]
        self.buf[self.idx] = y
        self.idx = (self.idx + 1) % self.win
        self.tick += 1
        if self.tick >= self.win:
            self.tick = 0
            return np.roll(self.buf, -self.idx, axis=0).copy()
        return None


def load_plugin(path):
    spec = importlib.util.spec_from_file_location("replay_plugin_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["replay_plugin_mod"] = mod
    spec.loader.exec_module(mod)
    return mod


def true_gesture(label):
    """NinaPro 动作号 -> 上位机 6 种手势（不在这 6 类里的动作返回 None）。"""
    if label == 0:
        return 0
    for code, mov in APP_GESTURE_MAP.items():
        if mov == label:
            return code
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subject", type=int, default=3)
    ap.add_argument("--exercise", type=int, default=1)
    ap.add_argument("--duration", type=float, default=90.0, help="回放多少秒")
    ap.add_argument("--start", type=float, default=0.0, help="从第几秒开始")
    ap.add_argument("--plugin", default=str(PLUGIN_DIR / "手势识别_CNNTransformer.py"))
    ap.add_argument("--out", default=None, help="把逐拍结果存成 json")
    args = ap.parse_args()

    cfg = Config()
    cfg.data.exercise = args.exercise
    data = load_subject(args.subject, args.exercise, cfg.data)
    emg = data["emg"]
    labels = (data["restimulus"] if cfg.data.use_restimulus else data["stimulus"]).astype(int)

    fs = cfg.pre.fs
    a = int(args.start * fs)
    b = min(emg.shape[0], a + int(args.duration * fs))
    print(f"回放 S{args.subject} 练习{args.exercise} 第 {args.start:.0f}~{b / fs:.0f} s，"
          f"共 {b - a} 个采样点")

    plugin = load_plugin(args.plugin)
    if hasattr(plugin, "reset"):
        plugin.reset()
    sp = StreamingSignalProcessor(fs)

    is_angles = getattr(plugin, "TYPE", "gesture") == "angles"
    glove = None
    if is_angles and "glove" in data:
        keep = [c for c in cfg.data.angle_keep]
        glove = np.asarray(data["glove"], dtype=np.float64)[:, keep]
        glove = (glove - glove.mean(0)) / np.where(glove.std(0) < 1e-9, 1.0, glove.std(0))
    angle_records = []

    records = []
    for i in range(a, b):
        block = sp.push(emg[i, :8])
        if block is None:
            continue
        out = plugin.predict({"features": {"samples": block}})
        if is_angles:
            arr = np.asarray(out, dtype=np.float64).reshape(-1)
            if arr.size > 1 and glove is not None:
                angle_records.append((arr, glove[i]))
            continue
        if isinstance(out, (tuple, list)) and len(out) == 2:
            name, conf = str(out[0]), float(out[1])
        else:
            name, conf = "角度输出", 1.0
        gt = true_gesture(int(labels[i]))
        records.append({"t": round(i / fs, 3), "name": name, "conf": round(conf, 3),
                        "true_code": gt, "true_label": int(labels[i])})
        post = getattr(plugin, "LAST_POST", None)
        if post is not None:
            records[-1]["argmax_label"] = int(np.argmax(post))

    if is_angles:
        print(f"角度插件：{len(angle_records)} 次输出，每次 {len(angle_records[0][0]) if angle_records else 0} 路")
        if angle_records:
            pred = np.stack([r[0] for r in angle_records])
            true = np.stack([r[1] for r in angle_records])
            cors = []
            for c in range(pred.shape[1]):
                p, t = pred[:, c], true[:, c]
                if p.std() > 1e-9 and t.std() > 1e-9:
                    cors.append(float(np.corrcoef(p, t)[0, 1]))
            print("  输出范围: %.1f ~ %.1f（度）" % (pred.min(), pred.max()))
            print("  与真实关节角(标准化)的平均相关系数: %.3f（%d 路）"
                  % (float(np.mean(cors)), len(cors)))
        return

    used = [r for r in records if r["true_code"] is not None]
    hit = sum(1 for r in used
              if r["name"] == APP_GESTURE_NAMES.get(r["true_code"], "__"))
    acc = hit / len(used) if used else 0.0
    print(f"逐拍决策 {len(records)} 次；其中真实标签属于上位机 6 类手势的 {len(used)} 次")
    print(f"插件与真实手势一致率：{acc:.3f}")

    dbg = [r for r in records if "argmax_label" in r]
    n_cls = len(getattr(plugin, "LAST_POST", []) or [])
    if dbg and n_cls > 7:      # 只有 NinaPro 全动作模型才有“18 类 argmax”这个参考指标
        hit18 = sum(1 for r in dbg if r["argmax_label"] == r["true_label"])
        used18 = [r for r in dbg if r["true_label"] > 0]
        hit18m = sum(1 for r in used18 if r["argmax_label"] == r["true_label"])
        print(f"[参考] 插件内部 18 类 argmax 准确率：{hit18 / len(dbg):.3f}"
              f"（仅动作样本 {hit18m / max(1, len(used18)):.3f}，n={len(dbg)}）")

    codes = [0, 1, 2, 3, 4, 5, 6]
    cm = np.zeros((len(codes), len(codes)), dtype=int)
    for r in used:
        if r["name"] in APP_GESTURE_NAMES.values():
            pred = [c for c, n in APP_GESTURE_NAMES.items() if n == r["name"]][0]
        else:
            pred = 0                       # “无/其他”都算作休息
        cm[r["true_code"], pred] += 1
    print("混淆矩阵（行=真实，列=预测，顺序 休息/握拳/张开/OK/拇指/比耶/捏合）：")
    for row in cm:
        print("   " + " ".join(f"{v:5d}" for v in row))
    for c in codes:
        n = cm[c].sum()
        if n:
            print(f"   {APP_GESTURE_NAMES[c]:<6} 召回 {cm[c, c] / n:.3f}  (n={n})")

    if args.out:
        Path(args.out).write_text(json.dumps(records, ensure_ascii=False, indent=1),
                                  encoding="utf-8")
        print(f"逐拍结果已写入 {args.out}")


if __name__ == "__main__":
    main()

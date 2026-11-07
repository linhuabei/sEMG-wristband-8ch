# -*- coding: utf-8 -*-
r"""把「上位机录的 CSV」和「参考角度 CSV」按时间戳合并成自采数据格式 .npz。

上位机「交互训练」页存出来的 CSV 长这样（第一行表头）：
    timestamp,gesture_code,CH1,CH2,...,CH8
参考角度（数据手套 / 动捕 / 弯曲传感器）另存一份 CSV，需要有一列时间戳 + 若干角度列。

本脚本做的事：
  1. 读上位机 CSV 的 8 通道肌电和时间戳；
  2. 把参考角度按时间戳**线性插值**到肌电的每个采样点上；
  3. 自动推断采样率 fs、自动给每次重复编号 trial（同一个手势第几次出现）；
  4. 写成 custom_dataset.py 能直接读的 .npz。

用法：
    python merge_app_csv.py --app-csv ../training_data/xxx/training_xxx.csv \
        --ref-csv  D:/参考角度/手套.csv --out D:/自采数据/S01_ses1.npz --subject 1

先自测（造两份假 CSV，跑通整条链）：
    python merge_app_csv.py --self-test
"""
import argparse
import csv
from pathlib import Path

import numpy as np


def _read_csv(path):
    with open(path, "r", encoding="utf-8-sig", newline="") as fh:
        rdr = csv.reader(fh)
        head = [h.strip() for h in next(rdr)]
        rows = [r for r in rdr if r and any(x.strip() for x in r)]
    data = {name: np.array([float(r[i]) for r in rows]) for i, name in enumerate(head)}
    return head, data


def _pick(head, cands, what):
    for c in cands:
        for h in head:
            if h.lower() == c.lower():
                return h
    raise ValueError(f"在表头 {head} 里找不到{what}（候选：{cands}）")


def merge(app_csv, ref_csv, out_path, subject=1, session=1, ref_time_col=None,
          ref_angle_cols=None, fs=None, trial_mode="auto"):
    head_a, A = _read_csv(app_csv)
    t = A[_pick(head_a, ["timestamp", "time", "t"], "时间戳列")]
    code = A[_pick(head_a, ["gesture_code", "label", "code"], "标签列")]
    emg_cols = [h for h in head_a if h.lower().startswith("ch")]
    emg_cols.sort(key=lambda s: int("".join(ch for ch in s if ch.isdigit()) or 0))
    emg = np.stack([A[c] for c in emg_cols], axis=1).astype(np.float32)

    head_r, R = _read_csv(ref_csv)
    t_ref_name = ref_time_col or _pick(head_r, ["timestamp", "time", "t"], "参考时间戳列")
    t_ref = R[t_ref_name]
    if ref_angle_cols:
        ang_cols = [c.strip() for c in ref_angle_cols.split(",")]
    else:
        ang_cols = [h for h in head_r if h != t_ref_name]
    angles = np.stack([np.interp(t, t_ref, R[c]) for c in ang_cols], axis=1).astype(np.float32)

    if fs is None:
        dt = np.median(np.diff(t))
        fs = int(round(1.0 / dt)) if dt > 0 else 2000
    # trial：同一个手势**第几段连续出现** = 第几次重复
    # （上位机里一个手势会连续采几秒，所以按"连续段"计数，而不是按采样点）
    trial = np.zeros(len(code), dtype=np.int16)
    if trial_mode == "auto":
        seen = {}
        prev = None
        for i, c in enumerate(code):
            if c != prev:
                seen[c] = seen.get(c, 0) + 1
                prev = c
            trial[i] = min(seen[c], 127)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(out_path, emg=emg, fs=np.int32(fs), trial=trial,
             label=code.astype(np.int16), angles=angles,
             angle_names=np.array(ang_cols), subject=np.int16(subject),
             session=np.int16(session))
    print(f"合并完成 -> {out_path}：emg {emg.shape}，角度 {angles.shape}，"
          f"fs={fs} Hz，时长 {len(t) / fs:.1f} s，重复段 {np.unique(trial)}")
    return out_path


def self_test(tmp=None):
    """造两份假 CSV（肌电 + 参考角度）并跑一遍合并 + 读取。"""
    import tempfile
    d = Path(tmp) if tmp else Path(tempfile.mkdtemp(prefix="merge_selftest_"))
    d.mkdir(parents=True, exist_ok=True)
    fs, dur = 2000, 84.0
    T = int(fs * dur)
    t = np.arange(T) / fs + 1780000000.0
    # 造 6 轮，每轮依次做 0~6 号动作（每段 2 s），这样 trial 会出现 1..6
    一段 = T // (6 * 7)
    code = np.concatenate([np.repeat(np.arange(0, 7), 一段) for _ in range(6)])
    code = np.pad(code, (0, T - len(code)), mode="edge")
    emg = np.random.default_rng(0).normal(0, 2e-5, size=(T, 8))
    app_csv = d / "上位机训练记录.csv"
    with open(app_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["timestamp", "gesture_code"] + [f"CH{i}" for i in range(1, 9)])
        for i in range(T):
            w.writerow([f"{t[i]:.6f}", int(code[i])] + [f"{v:.8f}" for v in emg[i]])
    # 参考角度：时间戳略有偏差（模拟不同设备时钟）
    t_ref = t[::2] + 0.001
    angles = np.stack([np.sin(2 * np.pi * 0.5 * (t - t[0]) + k) for k in range(21)], axis=1)
    ref_csv = d / "参考角度.csv"
    with open(ref_csv, "w", encoding="utf-8", newline="") as fh:
        w = csv.writer(fh)
        names = [f"关节{k + 1}" for k in range(21)]
        w.writerow(["timestamp"] + names)
        for i in range(len(t_ref)):
            w.writerow([f"{t_ref[i]:.6f}"] + [f"{v:.6f}" for v in angles[i]])
    out = merge(app_csv, ref_csv, d / "S01_ses1.npz", subject=1)

    import custom_dataset
    rec = custom_dataset.load_recording(out)
    print(f"自测读取：emg {rec['emg'].shape}，角度 {rec['angles'].shape}，"
          f"trial {np.unique(rec['trial'])}，label {np.unique(rec['label'])}")
    print(f"自测通过（文件在 {d}）")
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--app-csv", default=None)
    ap.add_argument("--ref-csv", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--session", type=int, default=1)
    ap.add_argument("--ref-time-col", default=None)
    ap.add_argument("--ref-angle-cols", default=None, help="逗号分隔；不填则用参考文件里的全部非时间列")
    ap.add_argument("--fs", type=int, default=None)
    ap.add_argument("--trial-mode", default="auto", choices=["auto"])
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    if args.self_test:
        self_test()
        return
    if not (args.app_csv and args.ref_csv and args.out):
        ap.print_help()
        return
    merge(args.app_csv, args.ref_csv, args.out, args.subject, args.session,
          args.ref_time_col, args.ref_angle_cols, args.fs, args.trial_mode)


if __name__ == "__main__":
    main()

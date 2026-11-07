# -*- coding: utf-8 -*-
r"""自采数据集接入：让 train.py 直接吃你们自己采的数据。

数据格式（每条记录一个 .npz，或同样字段的 CSV）：

| 字段 | 形状 | 说明 |
| --- | --- | --- |
| `emg` | (T, 8) float32 | 8 通道肌电原始值（单位随意，程序会标准化） |
| `fs` | 标量 int | 采样率，例如 2000 |
| `trial` | (T,) int16 | **第几次重复**（用来划分训练/验证/测试，必须有） |
| `label` | (T,) int16 | 动作编号，0 = 休息；只做连续识别时可以全填 0 |
| `angles` | (T, K) float32 | 参考关节角（数据手套/动捕/弯曲传感器），连续识别的监督目标 |
| `angle_names` | (K,) str | 每路角度的名字，如 "食指MCP" |

约定：
* 时间轴已经对齐（肌电与参考角度是同一时钟采的）；
* 划分沿用 NinaPro 那套：`train_trials / val_trials / test_trials`（默认 1,3,4 / 6 / 2,5）；
* 采样率可以不是 2000 Hz，程序会按 `fs/2000` 自动重采样到模型的 200 点窗口。

自测（不需要真实数据）：
    python custom_dataset.py --self-test
"""
import argparse
import re
from pathlib import Path

import numpy as np

from config import Config
from dataset import SubjectSignals
from preprocess import causal_stream_filter, rolling_zscore


# ---------------------------------------------------------------- 读写
def load_recording(path):
    """读一条记录（.npz 或 .csv），返回 dict。"""
    path = Path(path)
    if path.suffix.lower() == ".npz":
        with np.load(path, allow_pickle=True) as z:
            d = {k: z[k] for k in z.files}
    elif path.suffix.lower() == ".csv":
        import csv
        with open(path, "r", encoding="utf-8-sig", newline="") as fh:
            rdr = csv.reader(fh)
            head = next(rdr)
            cols = {name.strip(): i for i, name in enumerate(head)}
            rows = [r for r in rdr if r]
        def col(name):
            return np.array([float(r[cols[name]]) for r in rows])
        emg_cols = [c for c in head if re.fullmatch(r"ch\d+", c.strip(), re.I)]
        ang_cols = [c for c in head if re.fullmatch(r"ang\d*.*", c.strip(), re.I)]
        d = {"emg": np.stack([col(c) for c in emg_cols], axis=1)}
        if ang_cols:
            d["angles"] = np.stack([col(c) for c in ang_cols], axis=1)
            d["angle_names"] = np.array(ang_cols)
        if "fs" in cols:
            d["fs"] = int(col("fs")[0])
        for k in ("trial", "label"):
            if k in cols:
                d[k] = col(k).astype(np.int16)
    else:
        raise ValueError(f"不认识的格式：{path}")

    if "fs" not in d:
        d["fs"] = 2000
    if "trial" not in d:
        raise ValueError(f"{path.name} 缺少 trial（第几次重复）字段，无法划分训练/测试")
    d["emg"] = np.asarray(d["emg"], dtype=np.float32)
    if d["emg"].ndim != 2 or d["emg"].shape[1] != 8:
        raise ValueError(f"{path.name}: emg 应该是 (T, 8)，实际 {d['emg'].shape}")
    return d


def list_recordings(cfg):
    """按 `custom_dir` 里的文件，按受试者号归类：{subject: [路径, ...]}。"""
    d = Path(cfg.data.custom_dir)
    if not d.is_dir():
        raise FileNotFoundError(f"自采数据目录不存在：{d}")
    pat = re.compile(cfg.data.custom_subject_regex)
    out = {}
    for f in sorted(list(d.glob(cfg.data.custom_pattern))):
        m = pat.search(f.stem)
        if not m:
            print(f"  [warn] 文件名 {f.name} 里没找到受试者号，已跳过"
                  f"（可用 --custom-subject-regex 调整）")
            continue
        out.setdefault(int(m.group(1)), []).append(f)
    return out


def write_synthetic(path, fs=2000, seconds=300.0, n_angles=21, seed=0):
    """造一条假数据，用来验证管线（不是真实数据）。"""
    rng = np.random.default_rng(seed)
    T = int(fs * seconds)
    trial = np.repeat(np.arange(1, 7), T // 6 + 1)[:T].astype(np.int16)
    label = np.zeros(T, dtype=np.int16)
    seg = T // 6
    for t in range(6):                      # 每次重复里塞 17 个动作段
        s0 = t * seg
        for m in range(1, 18):
            a = s0 + int((m - 0.5) * seg / 17)
            b = s0 + int(m * seg / 17)
            label[a:b] = m
    emg = rng.normal(0, 1e-5, size=(T, 8))
    drive = (label > 0).astype(float)
    for c in range(8):                      # 让肌电跟动作有点关系，便于自测出结果
        emg[:, c] += drive * (c + 1) * 2e-5 * np.sin(2 * np.pi * 60 * np.arange(T) / fs)
    angles = np.stack([0.5 * drive * np.sin(2 * np.pi * (0.2 * k + 0.3) * np.arange(T) / fs)
                       for k in range(n_angles)], axis=1).astype(np.float32)
    np.savez(path, emg=emg, fs=fs, trial=trial, label=label, angles=angles,
             angle_names=np.array([f"关节{k + 1}" for k in range(n_angles)]),
             subject=np.int16(1))
    return path


# ---------------------------------------------------------------- 转成管线要的结构
def prepare_custom(subject, cfg, verbose=False):
    """读一位受试者的全部记录 -> 与 NinaPro 一致的 SubjectSignals。"""
    files = list_recordings(cfg).get(subject, [])
    if not files:
        raise FileNotFoundError(f"自采数据里没有受试者 S{subject}")
    recs = [load_recording(f) for f in files]
    fs0 = int(recs[0]["fs"])
    emg = np.concatenate([r["emg"] for r in recs], axis=0).astype(np.float64)
    trial = np.concatenate([np.asarray(r["trial"]).ravel() for r in recs])
    label = np.concatenate([np.asarray(r.get("label", np.zeros(len(r["trial"])))).ravel()
                            for r in recs]).astype(np.int64)
    angles = None
    if all("angles" in r for r in recs):
        angles = np.concatenate([r["angles"] for r in recs], axis=0).astype(np.float64)

    # 采样率与模型不同就重采样到 2000 Hz（线性插值，够用；要更讲究可用 scipy.signal.resample_poly）
    if fs0 != cfg.pre.fs:
        n_new = int(round(len(emg) * cfg.pre.fs / fs0))
        idx = np.linspace(0, len(emg) - 1, n_new)
        i0 = np.floor(idx).astype(np.int64)
        i1 = np.minimum(i0 + 1, len(emg) - 1)
        w = (idx - i0)[:, None]
        emg = emg[i0] * (1 - w) + emg[i1] * w
        trial = np.round(np.interp(np.arange(n_new), np.linspace(0, n_new - 1, len(trial)),
                                   trial)).astype(np.int64)
        label = np.round(np.interp(np.arange(n_new), np.linspace(0, n_new - 1, len(label)),
                                   label)).astype(np.int64)
        if angles is not None:
            angles = angles[i0] * (1 - w) + angles[i1] * w

    # 和 NinaPro 同一条链路：因果滤波 + 因果滚动标准化
    emg = causal_stream_filter(emg, cfg.pre.fs, cfg.pre.bandpass, cfg.pre.notch,
                               cfg.pre.notch_q)
    emg = rolling_zscore(emg, int(cfg.pre.norm_window_s * cfg.pre.fs))

    angle_mean = angle_std = None
    if angles is not None:
        if cfg.data.angle_smooth_hz and cfg.data.angle_smooth_hz > 0:
            from scipy import signal as _sig
            sos = _sig.butter(2, cfg.data.angle_smooth_hz / (cfg.pre.fs / 2.0), output="sos")
            angles = _sig.sosfiltfilt(sos, angles, axis=0)
        train_mask = np.isin(trial, np.asarray(cfg.data.train_repetitions))
        if train_mask.sum() < 100:
            train_mask = np.ones(len(trial), dtype=bool)
        angle_mean = angles[train_mask].mean(axis=0)
        angle_std = angles[train_mask].std(axis=0)
        angle_std[angle_std < 1e-12] = 1.0
        angles = ((angles - angle_mean) / angle_std).astype(np.float32)

    if verbose:
        print(f"  [prep] S{subject}: {emg.shape[0]} 点，{len(files)} 条记录，"
              f"采样率 {fs0} Hz -> {cfg.pre.fs} Hz")
    return SubjectSignals(subject=subject, emg=emg, labels=label, reps=trial,
                          angles=angles, angle_mean=angle_mean, angle_std=angle_std)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--self-test", action="store_true", help="造一份假数据并验证整条链路")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()
    if not args.self_test:
        ap.print_help()
        return

    import tempfile
    out = Path(args.out_dir) if args.out_dir else Path(tempfile.mkdtemp(prefix="custom_selftest_"))
    out.mkdir(parents=True, exist_ok=True)
    for s in (1, 2):
        write_synthetic(out / f"S{s}_ses1.npz", seconds=360.0, seed=s)
    print(f"已生成 2 条假数据 -> {out}")

    cfg = Config()
    cfg.data.dataset = "custom"
    cfg.data.custom_dir = str(out)
    for s in (1, 2):
        sig = prepare_custom(s, cfg, verbose=True)
        print(f"    S{s}: emg {sig.emg.shape}，角度 {None if sig.angles is None else sig.angles.shape}，"
              f"重复段 {np.unique(sig.reps)}")
    print("\n下一步（真实数据同样跑法）：")
    print(f"  python train.py --dataset custom --data-dir \"{out}\" "
          f"--mode per-subject --subjects 1-2 --task angles --epochs 5")


if __name__ == "__main__":
    main()

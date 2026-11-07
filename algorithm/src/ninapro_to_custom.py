# -*- coding: utf-8 -*-
r"""把公开数据集的一位受试者导出成"自采数据格式"的 .npz。

两个用途：
  1. 给你当**格式样例**：看看自采数据应该长什么样（用 custom_dataset.py 直接读）；
  2. 做**用户迁移实验**：拿 A 的模型去适配 B，验证"微调能不能把跨人精度拉回来"。

用法：
    python ninapro_to_custom.py --subjects 1 --out-dir ../outputs/自采格式样例
    python ninapro_to_custom.py --subjects 1,2,3 --out-dir D:/自采数据
"""
import argparse
from pathlib import Path

import numpy as np

from config import Config, DB2_EX1_NAMES
from ninapro import load_subject


def convert(subject, exercise, out_path, cfg):
    d = load_subject(subject, exercise, cfg.data)
    emg = np.asarray(d["emg"], dtype=np.float32)                 # (T, 8) 原始值
    keep = cfg.data.angle_keep
    angles = np.asarray(d["glove"], dtype=np.float32)[:, keep]   # (T, 21) 参考角度
    trial = np.asarray(d["rerepetition"] if cfg.data.use_restimulus
                       else d["repetition"]).astype(np.int16)
    label = np.asarray(d["restimulus"] if cfg.data.use_restimulus
                       else d["stimulus"]).astype(np.int16)
    names = np.array([f"角度通道{c + 1}" for c in keep])          # 原始通道号（0 基 +1）
    np.savez(out_path, emg=emg, fs=np.int32(cfg.pre.fs), trial=trial, label=label,
             angles=angles, angle_names=names, subject=np.int16(subject),
             session=np.int16(1))
    print(f"  S{subject} -> {out_path.name}：emg {emg.shape}，角度 {angles.shape}，"
          f"重复段 {np.unique(trial)}，动作数 {len(np.unique(label)) - 1}")
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--subjects", default="1")
    ap.add_argument("--exercise", type=int, default=1)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--prefix", default="S")
    args = ap.parse_args()

    subs = []
    for chunk in args.subjects.split(","):
        if "-" in chunk:
            a, b = chunk.split("-")
            subs.extend(range(int(a), int(b) + 1))
        else:
            subs.append(int(chunk))

    cfg = Config()
    cfg.data.exercise = args.exercise
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"导出到 {out_dir}（格式见 自采数据接入规范.md 第三节）")
    for s in subs:
        convert(s, args.exercise, out_dir / f"{args.prefix}{s:02d}_ses1.npz", cfg)
    print("完成。可以直接跑：")
    print(f'  python train.py --dataset custom --data-dir "{out_dir}" '
          f"--mode per-subject --subjects {subs[0]} --task angles --epochs 3")


if __name__ == "__main__":
    main()

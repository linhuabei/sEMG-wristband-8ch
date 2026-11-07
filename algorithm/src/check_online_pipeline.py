# -*- coding: utf-8 -*-
r"""诊断脚本：比较“离线评估流程”和“上位机在线流程”的差异来自哪里。

同一段原始数据、同一个模型，只改预处理方式，看准确率变化：
  A 插件风格：流式滤波（因果 sosfilt + 直流阻断）+ 最近 6 s 滚动 z-score
  B 全局 z-score：流式滤波 + 用整段数据的均值方差标准化
  C 离线风格：零相位 filtfilt + 整段数据 z-score（= 训练/评估时的做法）

用法（torch 环境）：
    python check_online_pipeline.py --ckpt ../outputs/ex1_persubject_gesture/subjects/S1.pt --subject 1
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from config import Config
from model import CNNTransformer
from ninapro import load_subject
from preprocess import apply_filters
from replay_plugin import StreamingSignalProcessor
from train import cfg_to_dict            # noqa: F401  (保持接口一致)


def load_model(ckpt_path):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    d = ckpt["config"]
    cfg = Config()
    for name in ("data", "pre", "window", "model", "train"):
        obj = getattr(cfg, name)
        for k, v in d.get(name, {}).items():
            if hasattr(obj, k):
                setattr(obj, k, tuple(v) if isinstance(v, list) else v)
    model = CNNTransformer(cfg.model, in_channels=8, seq_len=cfg.window.length)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return cfg, model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--subject", type=int, default=1)
    ap.add_argument("--seconds", type=float, default=150.0)
    ap.add_argument("--warm-ms", type=int, default=6000, help="滚动标准化用多少毫秒历史")
    args = ap.parse_args()

    cfg, model = load_model(args.ckpt)
    fs = cfg.pre.fs
    data = load_subject(args.subject, cfg.data.exercise, cfg.data)
    emg = data["emg"][:, :8]
    labels = (data["restimulus"] if cfg.data.use_restimulus else data["stimulus"]).astype(int)
    n = int(args.seconds * fs)
    emg, labels = emg[:n], labels[:n]

    # ---- 流式滤波（上位机做法）----
    sp = StreamingSignalProcessor(fs)
    blocks, block_end = [], []
    for i in range(emg.shape[0]):
        b = sp.push(emg[i])
        if b is not None:
            blocks.append(b)
            block_end.append(i)
    blocks = np.asarray(blocks)             # (nb, 100, 8)
    block_end = np.asarray(block_end)
    nb, bl, ch = blocks.shape
    print(f"流式块 {blocks.shape}，每块 {bl} 点（{bl / fs * 1000:.0f} ms）")

    # 100 ms 窗 = 相邻两块拼起来；窗的标签取窗口末端采样点的标签
    W = cfg.window.length
    wins, win_end = [], []
    for k in range(1, nb):
        w = np.concatenate(blocks[k - 1:k + 1], axis=0)
        if w.shape[0] >= W:
            wins.append(w[-W:])
            win_end.append(block_end[k])
    wins = np.asarray(wins, dtype=np.float64)
    win_end = np.asarray(win_end)
    y = labels[win_end]
    keep = y >= 0
    print(f"窗口 {wins.shape}")

    # ---- 三种标准化 ----
    hist_blocks = max(2, args.warm_ms // int(bl / fs * 1000))
    variants = {}
    # A 滚动 6 s（插件做法）
    xa = np.empty_like(wins)
    for k in range(len(wins)):
        lo = max(0, k + 1 - hist_blocks)
        hist = blocks[lo:k + 1].reshape(-1, ch)
        mu, sd = hist.mean(0), hist.std(0)
        sd = np.where(sd < 1e-9, 1.0, sd)
        xa[k] = (wins[k] - mu) / sd
    variants["A 插件风格(滚动6s)"] = xa
    # B 全段统计量
    flat = blocks.reshape(-1, ch)
    mu, sd = flat.mean(0), flat.std(0)
    variants["B 全段z-score"] = (wins - mu) / np.where(sd < 1e-9, 1.0, sd)
    # C 离线：零相位滤波 + 全段统计量
    filt = apply_filters(emg, fs, cfg.pre.bandpass, cfg.pre.notch, cfg.pre.notch_q)
    mu2, sd2 = filt.mean(0), filt.std(0)
    xc = np.empty_like(wins)
    for k, e in enumerate(win_end):
        seg = filt[e - W + 1:e + 1]
        xc[k] = (seg - mu2) / np.where(sd2 < 1e-9, 1.0, sd2)
    variants["C 离线风格(filtfilt)"] = xc

    # ---- 推理 ----
    with torch.no_grad():
        for name, x in variants.items():
            preds = []
            for s in range(0, len(x), 512):
                xb = torch.from_numpy(x[s:s + 512].transpose(0, 2, 1).astype(np.float32))
                preds.append(model(xb)["logits"].argmax(1).numpy())
            p = np.concatenate(preds)
            acc = (p[keep] == y[keep]).mean()
            mv = y[keep] > 0
            accm = (p[keep][mv] == y[keep][mv]).mean()
            print(f"  {name}: 全部 {acc:.3f} | 仅动作 {accm:.3f}")


if __name__ == "__main__":
    main()

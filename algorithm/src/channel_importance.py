# -*- coding: utf-8 -*-
"""把黑箱拆开看之一：通道重要性（遮挡实验 occlusion test）。

做法：把某一路电极的输入整段置零（标准化后 0 = 该通道的均值，等价于"这一路没信号"），
其余通道不动，再看测试集指标掉多少。掉得越多，说明这一路越重要。

这是可解释性里最直白的一种"扰动实验"，不需要任何梯度。

用法：python channel_importance.py --ckpt ../outputs/ex1_persubject_app6/subjects/S1.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from config import Config, OUTPUT_DIR
from dataset import build_index, prepare_subject
from model import CNNTransformer
from recheck_clean import cfg_from_dict
from utils import classification_metrics, pick_device, regression_metrics


def make_x(sig, pos, W):
    """把索引展开成 (N, C, W) 张量。"""
    out = np.empty((len(pos), sig.emg.shape[1], W), dtype=np.float32)
    for i, (_, st) in enumerate(pos):
        out[i] = sig.emg[st:st + W].T
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUTPUT_DIR / "ex1_persubject_app6" / "subjects" / "S1.pt"))
    ap.add_argument("--subject", type=int, default=None, help="默认用 checkpoint 里的受试者")
    ap.add_argument("--batch", type=int, default=512)
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = cfg_from_dict(ckpt["config"])
    subject = args.subject or (ckpt.get("subjects") or [1])[0]
    sig = prepare_subject(subject, cfg)
    _, _, test_pos = build_index([sig], cfg)
    W = cfg.window.length
    X = make_x(sig, test_pos, W)
    y = np.array([int(sig.labels[st + W - 1]) for _, st in test_pos])
    print(f"受试者 S{subject}：测试窗口 {len(y)} 个，输入 {X.shape[1]} 通道 × {W} 点")

    device = pick_device("cpu")
    model = CNNTransformer(cfg.model, in_channels=X.shape[1], seq_len=W).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    pure_reg = cfg.train.gesture_loss_weight == 0

    ang_true = None
    if "angles_true" not in [] and cfg.model.multi_task:
        ang_true = np.stack([sig.angles[st:st + W][-max(1, W // 4):].mean(axis=0)
                             for _, st in test_pos])

    def run(x_in):
        logits, ang = [], []
        with torch.no_grad():
            for i in range(0, len(x_in), args.batch):
                out = model(torch.from_numpy(x_in[i:i + args.batch]).to(device))
                logits.append(out["logits"].cpu().numpy())
                if "angles" in out:
                    ang.append(out["angles"].cpu().numpy())
        return np.concatenate(logits), (np.concatenate(ang) if ang else None)

    def score(x_in):
        logits, ang = run(x_in)
        s = {}
        if not pure_reg:
            m = classification_metrics(y, logits.argmax(1), cfg.model.n_classes)
            s["accuracy"] = m["accuracy"]
            s["movement_acc"] = m["accuracy_movement_only"]
        if ang is not None and ang_true is not None:
            a = regression_metrics(ang_true, ang)
            s["r2"] = a["r2_mean"]
            s["r"] = a["pearson_mean"]
        return s

    base = score(X)
    print(f"基准（不遮挡）：{base}\n")
    print("遮挡单路通道后的变化：")
    rows = []
    for c in range(X.shape[1]):
        Xo = X.copy()
        Xo[:, c, :] = 0.0
        s = score(Xo)
        drop = {k: base[k] - s[k] for k in base}
        rows.append((c + 1, s, drop))
    rows.sort(key=lambda r: -(r[2].get("accuracy", r[2].get("r2", 0))))
    for ch, s, d in rows:
        txt = "  ".join(f"{k}: {s[k]:.4f} (Δ{d[k]:+.4f})" for k in base)
        print(f"  去掉通道 {ch}: {txt}")
    print("\n（Δ 为负 = 缺了这一路后变差，绝对值越大越重要）")


if __name__ == "__main__":
    main()

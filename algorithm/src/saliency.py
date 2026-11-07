# -*- coding: utf-8 -*-
"""把黑箱拆开看之二：输入显著性（gradient saliency，梯度归因）。

对每个测试窗口，计算"输出"对"输入 8×200 个数值"的偏导数 ∂输出/∂输入，
取绝对值再对所有窗口平均，得到一张 **8 通道 × 200 个时刻** 的敏感度地图：
颜色越深 = 模型在判断时越依赖"这一路电极、这几十毫秒"的肌电。

它回答的正是"权重是不是每个时刻一个固定值"这个问题：
敏感度是随输入变化的，而且高度集中在动作爆发的那一小段时间。

用法：python saliency.py --ckpt ../outputs/ex1_persubject_app6/subjects/S1.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from config import OUTPUT_DIR
from channel_importance import make_x
from dataset import build_index, prepare_subject
from model import CNNTransformer
from recheck_clean import cfg_from_dict
from utils import pick_device, setup_matplotlib_chinese


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUTPUT_DIR / "ex1_persubject_app6" / "subjects" / "S1.pt"))
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--batch", type=int, default=128)
    ap.add_argument("--out", default=None)
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

    device = pick_device("cpu")
    model = CNNTransformer(cfg.model, in_channels=X.shape[1], seq_len=W).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    pure_reg = cfg.train.gesture_loss_weight == 0

    sal = np.zeros((X.shape[1], W), dtype=np.float64)
    n_used = 0
    for i in range(0, len(X), args.batch):
        xb = torch.from_numpy(X[i:i + args.batch]).to(device).requires_grad_(True)
        out = model(xb)
        if pure_reg and "angles" in out:
            target = out["angles"].sum(dim=1)              # 回归：对所有关节角求和
        else:
            target = out["logits"].gather(
                1, torch.from_numpy(y[i:i + args.batch]).to(device).unsqueeze(1)).squeeze(1)
        model.zero_grad(set_to_none=True)
        target.sum().backward()
        g = xb.grad.detach().abs().cpu().numpy()           # (B, C, W)
        sal += g.sum(axis=0)
        n_used += g.shape[0]
    sal /= max(1, n_used)
    print(f"受试者 S{subject}：用 {n_used} 个测试窗口算敏感度（|∂输出/∂输入| 的平均）")
    print("各通道平均敏感度（越大越依赖这一路）：")
    order = np.argsort(sal.mean(axis=1))[::-1]
    for c in order:
        print(f"   通道 {c + 1}: {sal[c].mean():.5f}")
    # 敏感度在时间上有多集中？——前 30 名时刻占了多少比重
    flat = np.sort(sal.ravel())[::-1]
    print(f"   最高的 10% 的（通道,时刻）组合占了总敏感度的 "
          f"{flat[:int(0.1 * flat.size)].sum() / flat.sum():.1%}")
    prof = sal.mean(axis=0)
    edges = np.linspace(0, W, 5, dtype=int)
    seg = [prof[edges[i]:edges[i + 1]].sum() for i in range(4)]
    tot = sum(seg)
    print("   敏感度在时间上的分布（按 25 ms 四等分）：" +
          "  ".join(f"{20 * i}~{20 * (i + 1)}ms {s / tot:.1%}" for i, s in enumerate(seg)))

    font = setup_matplotlib_chinese()
    if font is None:
        print("[warn] 没找到中文字体，图中中文可能显示为方块")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(11, 3.2))
    im = ax.imshow(sal, aspect="auto", cmap="magma",
                   extent=[0, W / cfg.pre.fs * 1000, X.shape[1] + 0.5, 0.5])
    ax.set_xlabel("窗口内时间 (ms)，0 = 窗口起点，100 = 当前时刻")
    ax.set_ylabel("电极通道")
    ax.set_yticks(range(1, X.shape[1] + 1))
    ax.set_title(f"输入敏感度地图（S{subject}）：模型判断时最依赖哪些电极的哪几十毫秒")
    fig.colorbar(im, ax=ax, fraction=0.03, label="平均 |梯度|")
    fig.tight_layout()
    out_path = Path(args.out) if args.out else ckpt_path.parent.parent / "saliency.png"
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    print(f"图已保存：{out_path}")


if __name__ == "__main__":
    main()

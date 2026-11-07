# -*- coding: utf-8 -*-
r"""输入敏感度地图（正式版）：生成可直接放进论文的图 + 表。

原理：对测试集每个窗口计算 ∂输出/∂输入（输出对该窗口 8×200 个采样点的偏导），
取绝对值后对所有窗口求平均，得到"电极通道 × 窗内时刻"的敏感度地图。
颜色越深表示模型在判断时越依赖该通道、该时刻的肌电幅值。

输出（默认写到 ckpt 所在实验目录）：
    输入敏感度地图_XXX.png / .pdf / .svg   三面板正式图（300 dpi + 矢量）
    敏感度_原始矩阵_XXX.csv                8×200 完整矩阵（带表头）
    敏感度_汇总表_XXX.csv                  10 ms 分箱汇总表（带表头，可直接贴论文）

用法：
    python 输入敏感度地图.py --ckpt ../outputs/ex1_persubject_app6/subjects/S1.pt
    python 输入敏感度地图.py --ckpt ../outputs/ex1_persubject_angles/subjects/S3.pt --tag 连续关节角
"""
import argparse
import csv
from pathlib import Path

import numpy as np
import torch

from channel_importance import make_x
from config import OUTPUT_DIR
from dataset import build_index, prepare_subject
from model import CNNTransformer
from recheck_clean import cfg_from_dict
from utils import pick_device, setup_matplotlib_chinese


def 计算敏感度(ckpt, subject=None, batch=128):
    """返回 (敏感度矩阵 (C,W), 元信息 dict)。"""
    ckpt_path = Path(ckpt)
    ck = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = cfg_from_dict(ck["config"])
    subject = subject or (ck.get("subjects") or [1])[0]
    sig = prepare_subject(subject, cfg)
    _, _, test_pos = build_index([sig], cfg)
    W = cfg.window.length
    X = make_x(sig, test_pos, W)
    y = np.array([int(sig.labels[st + W - 1]) for _, st in test_pos])
    device = pick_device("cpu")
    model = CNNTransformer(cfg.model, in_channels=X.shape[1], seq_len=W).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    pure_reg = cfg.train.gesture_loss_weight == 0

    sal = np.zeros((X.shape[1], W), dtype=np.float64)
    for i in range(0, len(X), batch):
        xb = torch.from_numpy(X[i:i + batch]).to(device).requires_grad_(True)
        out = model(xb)
        if pure_reg and "angles" in out:
            target = out["angles"].sum(dim=1)          # 回归任务：21 路角度之和
        else:
            target = out["logits"].gather(
                1, torch.from_numpy(y[i:i + batch]).to(device).unsqueeze(1)).squeeze(1)
        model.zero_grad(set_to_none=True)
        target.sum().backward()
        sal += xb.grad.detach().abs().cpu().numpy().sum(axis=0)
    sal /= max(1, len(X))
    meta = {
        "受试者": subject,
        "样本数": int(len(X)),
        "通道数": int(X.shape[1]),
        "采样率_Hz": int(cfg.pre.fs),
        "窗长_ms": round(W / cfg.pre.fs * 1000, 1),
        "任务": "连续关节角回归" if pure_reg else f"{cfg.model.n_classes} 类手势分类",
        "模型": f"CNN-Transformer（{cfg.model.n_layers} 层编码器 / {cfg.model.n_heads} 头注意力）",
        "模型_EN": (f"CNN-Transformer ({cfg.model.n_layers} encoder layers, "
                    f"{cfg.model.n_heads} attention heads)"),
    }
    return sal, meta


def 导表(sal, meta, out_dir, tag):
    """导出两张 CSV：完整矩阵 + 10 ms 分箱汇总表。"""
    C, W = sal.shape
    ms_per_sample = meta["窗长_ms"] / W
    full = Path(out_dir) / f"敏感度_原始矩阵_{tag}.csv"
    with open(full, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["输入敏感度矩阵（平均 |∂输出/∂输入|）", f"模型={meta['模型']}",
                    f"任务={meta['任务']}", f"受试者=S{meta['受试者']}",
                    f"测试样本数={meta['样本数']}", f"窗长={meta['窗长_ms']} ms",
                    f"采样率={meta['采样率_Hz']} Hz"])
        w.writerow(["电极通道 / 窗内时刻(ms)"] + [f"{i * ms_per_sample:.2f}" for i in range(W)])
        for c in range(C):
            w.writerow([f"CH{c + 1}"] + [f"{v:.6g}" for v in sal[c]])

    bins = 10
    edges = np.linspace(0, W, bins + 1).astype(int)
    agg = np.stack([sal[:, edges[i]:edges[i + 1]].mean(axis=1) for i in range(bins)], axis=1)
    sm = Path(out_dir) / f"敏感度_汇总表_{tag}.csv"
    with open(sm, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"表：输入敏感度汇总（{meta['窗长_ms']:.0f} ms 窗口按 10 ms 分箱；"
                    f"单位：平均 |∂输出/∂输入|；模型：{meta['模型']}；"
                    f"任务：{meta['任务']}；受试者：S{meta['受试者']}；"
                    f"测试样本 n={meta['样本数']}）"])
        w.writerow(["电极通道"] + [f"{i * 10}~{(i + 1) * 10} ms" for i in range(bins)]
                   + ["行均值"])
        for c in range(C):
            w.writerow([f"CH{c + 1}"] + [f"{v:.6g}" for v in agg[c]]
                       + [f"{agg[c].mean():.6g}"])
        w.writerow(["列均值"] + [f"{v:.6g}" for v in agg.mean(axis=0)]
                   + [f"{agg.mean():.6g}"])
    return full, sm, agg


def 标签集(english: bool):
    """图内文字：中文版 / 英文版（投稿用）。"""
    if english:
        return {
            "x": "Relative time within window $t$ / ms",
            "y": "sEMG electrode channel",
            "cbar": "Mean absolute sensitivity $|\\partial y/\\partial x|$",
            "a": "(a) Input sensitivity map (channel × time)",
            "b_x": "Mean sensitivity", "b": "(b) Channel importance",
            "c_y": "Mean sensitivity", "c_x": "Time segment within window / ms",
            "c": "(c) Temporal distribution of sensitivity (channel-averaged)",
            "suptitle": "Input sensitivity map of the sEMG decoder",
            "sub": ("Model: {model}; Task: {task_en}; Subject: S{subj}; "
                    "Test samples $n$ = {n}; Window {w:.0f} ms @ {fs} Hz"),
            "task_en_map": {"连续关节角回归": "continuous joint-angle regression"},
        }
    return {
        "x": "窗内相对时间 $t$ / ms",
        "y": "表面肌电电极通道",
        "cbar": "平均绝对敏感度 $|\\partial y/\\partial x|$",
        "a": "(a) 输入敏感度地图（电极通道 × 窗内时刻）",
        "b_x": "平均敏感度", "b": "(b) 通道重要性",
        "c_y": "平均敏感度", "c_x": "窗内时间段 / ms（0 ms = 窗口起点，100 ms = 当前时刻）",
        "c": "(c) 敏感度的时间分布（各通道平均）",
        "suptitle": "图  输入敏感度地图：模型判读时依赖的电极通道与时段",
        "sub": ("（模型：{model}；任务：{task}；受试者：S{subj}；"
                "测试样本 $n$ = {n}；窗口 {w:.0f} ms @ {fs} Hz）"),
        "task_en_map": {},
    }


def 画图(sal, meta, agg, out_dir, tag, english=False, 版式="完整"):
    """版式：完整 = 热图+通道柱+时间曲线；双面板 = 热图+时间曲线；精简 = 只有热图。"""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    if not english and setup_matplotlib_chinese() is None:
        print("[warn] 未找到中文字体，图中中文可能显示为方块")
    if english:
        plt.rcParams["font.sans-serif"] = ["DejaVu Sans"]
    T = 标签集(english)
    plt.rcParams.update({"font.size": 10, "axes.titlesize": 11,
                         "axes.titleweight": "bold", "axes.linewidth": 0.8})

    C, W = sal.shape
    ch_mean = sal.mean(axis=1)
    ch_std = sal.std(axis=1)

    # 紧凑排版（三种版式：完整 / 双面板 / 精简）
    尺寸 = {"完整": (9.6, 4.6), "双面板": (9.2, 3.9), "精简": (8.6, 3.1)}[版式]
    fig = plt.figure(figsize=尺寸, layout="constrained")
    fig.get_layout_engine().set(w_pad=0.012, h_pad=0.012, wspace=0.008, hspace=0.008)
    ax_ch = ax_t = None
    if 版式 == "精简":
        gs = fig.add_gridspec(1, 2, width_ratios=[26, 1])
        ax_map = fig.add_subplot(gs[0, 0])
        ax_cb = fig.add_subplot(gs[0, 1])
    elif 版式 == "双面板":
        gs = fig.add_gridspec(2, 2, height_ratios=[3.0, 1.0], width_ratios=[26, 1])
        ax_map = fig.add_subplot(gs[0, 0])
        ax_cb = fig.add_subplot(gs[0, 1])
        ax_t = fig.add_subplot(gs[1, 0], sharex=ax_map)
    else:
        gs = fig.add_gridspec(2, 3, height_ratios=[3.1, 1.05],
                              width_ratios=[4.0, 0.85, 0.10])
        ax_map = fig.add_subplot(gs[0, 0])
        ax_ch = fig.add_subplot(gs[0, 1], sharey=ax_map)
        ax_cb = fig.add_subplot(gs[0, 2])
        ax_t = fig.add_subplot(gs[1, :], sharex=ax_map)

    # (a) 敏感度地图
    im = ax_map.imshow(sal, aspect="auto", cmap="magma", origin="upper",
                       extent=[0, meta["窗长_ms"], C + 0.5, 0.5])
    ax_map.set_yticks(range(1, C + 1))
    ax_map.set_yticklabels([f"CH{i}" for i in range(1, C + 1)], fontsize=8.5)
    ax_map.set_xticks(np.linspace(0, meta["窗长_ms"], 6))
    ax_map.tick_params(labelbottom=(版式 == "精简"), bottom=(版式 == "精简"))
    ax_map.set_ylabel(T["y"])
    ax_map.set_title(T["a"], fontsize=9.5, pad=3)
    cb = fig.colorbar(im, cax=ax_cb, orientation="vertical")
    cb.set_label(T["cbar"], fontsize=8)
    cb.ax.tick_params(labelsize=7.5)

    # (b) 通道重要性
    if ax_ch is not None:
        ypos = np.arange(1, C + 1)
        ax_ch.barh(ypos, ch_mean, xerr=ch_std, color="#b03a2e", alpha=0.88,
                   error_kw={"elinewidth": 0.8, "capsize": 2})
        ax_ch.invert_yaxis()
        ax_ch.tick_params(labelleft=False, left=False)
        ax_ch.set_xlabel(T["b_x"], fontsize=8.5)
        ax_ch.text(0.02, 0.97, T["b"], transform=ax_ch.transAxes, va="top", ha="left",
                   fontsize=8.5, fontweight="bold")
        ax_ch.grid(axis="x", alpha=0.3, linewidth=0.5)
        ax_ch.tick_params(labelsize=7.5)

    # (c) 时间分布（紧贴热图，共享横轴）
    if ax_t is not None:
        prof = agg.mean(axis=0)
        bin_ms = meta["窗长_ms"] / agg.shape[1]
        centers = np.arange(agg.shape[1]) * bin_ms + bin_ms / 2
        ax_t.bar(centers, prof, color="#2e6da4", alpha=0.85,
                 width=bin_ms * 0.7, align="center")
        for xc, v in zip(centers, prof):
            ax_t.annotate(f"{v:.3f}", (xc, v), textcoords="offset points", xytext=(0, 3),
                          ha="center", fontsize=6.5, color="#1f4e79")
        ax_t.set_ylim(0, float(prof.max()) * 1.32)
        ax_t.set_ylabel(T["c_y"], fontsize=8.5)
        ax_t.set_xlabel(T["x"], fontsize=8.5)
        ax_t.text(0.004, 0.94, T["c"], transform=ax_t.transAxes, va="top", ha="left",
                  fontsize=8.5, fontweight="bold")
        ax_t.grid(axis="y", alpha=0.3, linewidth=0.5)
        ax_t.tick_params(labelsize=7.5)

    任务 = meta["任务"]
    if english:
        任务 = T["task_en_map"].get(任务, f"{meta['通道数']}-class gesture classification")
    模型名 = meta["模型_EN"] if english else meta["模型"]
    副标题 = T["sub"].format(model=模型名, task=任务, task_en=任务,
                             subj=meta["受试者"], n=meta["样本数"],
                             w=meta["窗长_ms"], fs=meta["采样率_Hz"])
    # 标题 + 元信息合并成两行（元信息用较短的写法，避免单独占一行空白）
    if english:
        元信息 = (f"{模型名} | {任务} | Subject S{meta['受试者']} | "
                  f"$n$ = {meta['样本数']} | {meta['窗长_ms']:.0f} ms @ {meta['采样率_Hz']} Hz")
    else:
        元信息 = (f"{模型名} ｜ {任务} ｜ 受试者 S{meta['受试者']} ｜ "
                  f"$n$ = {meta['样本数']} ｜ {meta['窗长_ms']:.0f} ms @ {meta['采样率_Hz']} Hz")
    fig.suptitle(f"{T['suptitle']}\n{元信息}", fontsize=10.5, fontweight="bold")
    后缀 = ("_EN" if english else "") + ("" if 版式 == "完整" else f"_{版式}")
    png = Path(out_dir) / f"输入敏感度地图_{tag}{后缀}.png"
    fig.savefig(png, dpi=300, bbox_inches="tight", pad_inches=0.06)
    fig.savefig(Path(out_dir) / f"输入敏感度地图_{tag}{后缀}.pdf",
                bbox_inches="tight", pad_inches=0.06)
    fig.savefig(Path(out_dir) / f"输入敏感度地图_{tag}{后缀}.svg",
                bbox_inches="tight", pad_inches=0.06)
    plt.close(fig)
    return png


def main():
    ap = argparse.ArgumentParser(description="输入敏感度地图（正式版）")
    ap.add_argument("--ckpt", default=str(OUTPUT_DIR / "ex1_persubject_app6"
                                          / "subjects" / "S1.pt"))
    ap.add_argument("--subject", type=int, default=None)
    ap.add_argument("--tag", default=None, help="文件名后缀（默认按任务取）")
    ap.add_argument("--out-dir", default=None, help="默认写在 ckpt 上两级目录")
    args = ap.parse_args()

    ckpt = Path(args.ckpt)
    sal, meta = 计算敏感度(ckpt, args.subject)
    tag = args.tag or ("连续关节角" if "关节角" in meta["任务"] else "手势识别")
    out_dir = Path(args.out_dir) if args.out_dir else ckpt.parent.parent
    out_dir.mkdir(parents=True, exist_ok=True)

    full, sm, agg = 导表(sal, meta, out_dir, tag)
    图集 = {}
    for 版式 in ("完整", "双面板", "精简"):
        for en in (False, True):
            图集[(版式, en)] = 画图(sal, meta, agg, out_dir, tag, english=en, 版式=版式)
    png = 图集[("完整", False)]
    png_en = 图集[("完整", True)]

    print(f"\n模型：{meta['模型']}")
    print(f"任务：{meta['任务']}　受试者：S{meta['受试者']}　"
          f"测试样本 n={meta['样本数']}　窗口 {meta['窗长_ms']:.0f} ms")
    print("\n通道重要性排序（平均 |∂y/∂x|）：")
    for c in np.argsort(sal.mean(axis=1))[::-1]:
        print(f"    CH{c + 1}: {sal[c].mean():.5f}")
    prof = agg.mean(axis=0)
    print("\n时间分布（10 ms 分箱）：")
    for i, v in enumerate(prof):
        print(f"    {i * 10:>3}~{(i + 1) * 10:>3} ms : {v:.5f}（占 {v / prof.sum():.1%}）")
    print(f"\n图：{png}（另有 .pdf / .svg 矢量版）")
    print(f"    英文版：{png_en}")
    print("    另外两套紧凑版式：")
    for 版式 in ("双面板", "精简"):
        print(f"      {版式}：{图集[(版式, False)].name} / {图集[(版式, True)].name}")
    print(f"表：{full.name}（完整矩阵，8×200）、{sm.name}（10 ms 汇总表）")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""画 CNN-Transformer 预测模型的结构图 + 导出逐层表（论文用）。

输出（默认写到 算法/outputs/）：
    模型结构图.png / .pdf / .svg      竖向结构图（含每层张量形状与参数量）
    模型逐层表.csv                    逐层：层名 / 操作 / 输出形状 / 参数量

用法：
    python 画模型结构图.py                     # 默认画 7 类手势模型（上位机用）
    python 画模型结构图.py --classes 18 --multi-task   # 多任务（18 类 + 21 路角度）
"""
import argparse
import csv
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

from config import OUTPUT_DIR
from model import CNNTransformer, count_parameters
from config import ModelConfig
from utils import setup_matplotlib_chinese


def 逐层表():
    return [
        ("输入", "8 通道 sEMG，100 ms（200 点 @ 2 kHz）", "(B, 8, 200)", 0),
        ("卷积 1", "Conv1d(8→32, k=5, pad=2) + BN + ReLU + MaxPool(2)", "(B, 32, 100)", 1280 + 64),
        ("卷积 2", "Conv1d(32→64, k=5, pad=2) + BN + ReLU + MaxPool(2)", "(B, 64, 50)", 10240 + 128),
        ("位置编码", "正弦位置编码（固定，不参与训练）", "(B, 50, 64)", 0),
        ("编码器 ×2", "多头自注意力（8 头）+ 残差 + LayerNorm", "(B, 50, 64)", 33728 * 2 // 2),
        ("编码器 ×2", "深度可分离卷积前馈（64→128→64, k=3）+ 残差", "(B, 50, 64)", 33728 * 2 // 2),
        ("池化", "LayerNorm + 时间维全局平均池化 + Dropout(0.2)", "(B, 64)", 128),
    ]


def 画图(classes, multi_task, n_angles, out_path):
    if setup_matplotlib_chinese() is None:
        print("[warn] 未找到中文字体")
    plt.rcParams.update({"font.size": 10})
    cfg = ModelConfig()
    cfg.n_classes = classes
    cfg.multi_task = multi_task
    cfg.n_angles = n_angles
    model = CNNTransformer(cfg, in_channels=8, seq_len=200)
    n_param = count_parameters(model)

    fig, ax = plt.subplots(figsize=(8.8, 8.2), layout="constrained")
    ax.set_xlim(0.35, 8.95)
    ax.set_ylim(1.35, 14.3)
    ax.axis("off")

    def 方块(y, h, text, sub, color, x=0.6, w=6.4):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.06,rounding_size=0.12",
                                    linewidth=1.1, edgecolor="#333333", facecolor=color))
        ax.text(x + w / 2, y + h * 0.66, text, ha="center", va="center",
                fontsize=10.5, fontweight="bold", color="#111111")
        ax.text(x + w / 2, y + h * 0.27, sub, ha="center", va="center",
                fontsize=8.6, color="#333333")

    def 形状标注(y, text):
        ax.text(7.25, y, text, ha="left", va="center", fontsize=9,
                family="DejaVu Sans", color="#1f4e79")

    def 箭头(y_from, y_to, x=3.8):
        ax.add_patch(FancyArrowPatch((x, y_from), (x, y_to),
                                     arrowstyle="-|>", mutation_scale=13,
                                     linewidth=1.1, color="#555555"))

    方块(13.0, 1.0, "输入：8 通道表面肌电（sEMG）",
         "100 ms 窗口 = 200 点 @ 2 kHz，逐通道 z-score 标准化", "#e8f0fb")
    形状标注(13.5, "(B, 8, 200)")

    方块(11.4, 1.2, "卷积特征提取（1D CNN）",
         "Conv1d(8→32, k=5) + BN + ReLU + MaxPool(2)\n"
         "Conv1d(32→64, k=5) + BN + ReLU + MaxPool(2)", "#eaf7ee")
    形状标注(12.0, "(B, 32, 100) → (B, 64, 50)")

    方块(9.8, 0.9, "正弦位置编码",
         "固定编码（不训练），为 50 个时间片注入先后顺序", "#fdf6e3")
    形状标注(10.25, "(B, 50, 64)")

    方块(7.2, 2.2, "Transformer 编码器 × 2 层",
         "① 多头自注意力（8 头）：每个时间片与其余时间片、其余通道加权融合\n"
         "② 残差连接 + LayerNorm（Pre-LN，稳定深层训练）\n"
         "③ 深度可分离卷积前馈（64→128→64, k=3，轻量化）\n"
         "④ 残差连接 + LayerNorm", "#fdeceb")
    形状标注(8.3, "(B, 50, 64)")

    方块(5.6, 0.95, "全局平均池化 + Dropout",
         "时间维取平均，得到与序列长度无关的定长特征；Dropout 0.2 抑制过拟合", "#f2eefb")
    形状标注(6.07, "(B, 64)")

    if multi_task:
        方块(3.4, 1.35, "手势头：Linear(64→%d) + Softmax" % classes,
             "输出各手势概率", "#e8f0fb", x=0.6, w=3.0)
        方块(3.4, 1.35, "关节角头：Linear(64→%d)" % n_angles,
             "输出 %d 路连续关节角" % n_angles, "#eaf7ee", x=4.0, w=3.0)
        形状标注(4.07, "(B, %d)" % classes)
        ax.text(7.25, 3.35, "(B, %d)" % n_angles, ha="left", va="center",
                fontsize=9, family="DejaVu Sans", color="#1f4e79")
    else:
        方块(3.4, 1.35, "分类头：Linear(64→%d) + Softmax" % classes,
             "输出 %d 类手势的概率" % classes, "#e8f0fb")
        形状标注(4.07, "(B, %d)" % classes)

    方块(1.6, 1.0, "输出",
         ("手势类别概率 + %d 路关节角" % n_angles) if multi_task
         else ("%d 类手势概率（取最大即识别结果）" % classes), "#f5f5f5")
    形状标注(2.1, "(B, %d)" % (classes + n_angles if multi_task else classes))

    for y0, y1 in [(13.0, 12.6), (11.4, 10.7), (9.8, 9.4), (7.2, 6.55), (5.6, 4.75), (3.4, 2.6)]:
        箭头(y0, y1)

    ax.set_title(
        "CNN-Transformer 表面肌电预测模型结构\n"
        f"（8 通道 sEMG → {'%d 类手势 + %d 路关节角' % (classes, n_angles) if multi_task else '%d 类手势' % classes}；"
        f"可训练参数 {n_param:,} 个，约 {n_param * 4 / 1024:.0f} KB）",
        fontsize=12.5, fontweight="bold")
    fig.savefig(out_path, dpi=300, bbox_inches="tight", pad_inches=0.08)
    fig.savefig(Path(out_path).with_suffix(".pdf"), bbox_inches="tight", pad_inches=0.08)
    fig.savefig(Path(out_path).with_suffix(".svg"), bbox_inches="tight", pad_inches=0.08)
    plt.close(fig)
    return n_param


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--classes", type=int, default=7)
    ap.add_argument("--multi-task", action="store_true")
    ap.add_argument("--angles", type=int, default=21)
    ap.add_argument("--out-dir", default=str(OUTPUT_DIR))
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    tag = "多任务" if args.multi_task else f"{args.classes}类手势"
    png = out_dir / f"模型结构图_{tag}.png"
    n = 画图(args.classes, args.multi_task, args.angles, png)

    csv_path = out_dir / "模型逐层表.csv"
    with open(csv_path, "w", encoding="utf-8-sig", newline="") as fh:
        w = csv.writer(fh)
        w.writerow([f"CNN-Transformer 逐层结构（{tag}，可训练参数 {n:,} 个）"])
        w.writerow(["层 / 模块", "操作", "输出形状 (B=批大小)", "参数量"])
        for row in 逐层表():
            w.writerow([row[0], row[1], row[2], row[3]])
    print(f"结构图：{png}（另有 .pdf / .svg）")
    print(f"逐层表：{csv_path}")
    print(f"可训练参数：{n:,} 个")


if __name__ == "__main__":
    main()

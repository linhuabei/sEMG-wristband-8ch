# -*- coding: utf-8 -*-
"""画图 + 汇总指标：训练曲线、混淆矩阵、每位受试者准确率。

用法：
    python evaluate.py --run ../outputs/ex1_gesture
    python evaluate.py --run ../outputs/ex1_gesture --per-class-table
"""
import argparse
import json
from pathlib import Path

import numpy as np

from config import APP_GESTURE_NAMES, DB2_EX1_NAMES, OUTPUT_DIR
from utils import setup_matplotlib_chinese


def plot_history(history, out_path, title_prefix=""):
    import matplotlib.pyplot as plt
    # 支持两种格式：单模型(平铺 epoch 列表) / 逐受试者(每人一段曲线)
    if history and isinstance(history[0], dict) and "history" in history[0]:
        series = {f"S{h['subject']}": h["history"] for h in history}
    else:
        series = {"": history}
    multi = len(series) > 1

    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    specs = [("train_loss", "验证/训练损失"), ("val_acc", "准确率"),
             ("val_movement_acc", "仅动作样本准确率")]
    if history and history[0].get("val_angle_r2") is not None:
        specs = [("train_loss", "训练/验证损失"), ("val_acc", "手势准确率"),
                 ("val_angle_r2", "验证集关节角 R²")]
    for ax, (metric, title) in zip(axes, specs):
        for name, hist in series.items():
            if not hist or metric not in hist[0]:
                continue
            x = [h["epoch"] for h in hist]
            y = [h.get(metric, float("nan")) for h in hist]
            ax.plot(x, y, "-", lw=1.0 if multi else 1.8, alpha=0.55 if multi else 1.0,
                    label=name if multi else None)
        if multi:                        # 不同受试者 epoch 数不同，再画一条平均曲线
            max_ep = max(len(h) for h in series.values())
            xs, ys = [], []
            for e in range(max_ep):
                vals = [h[e][metric] for h in series.values()
                        if len(h) > e and metric in h[e]]
                if vals:
                    xs.append(e + 1)
                    ys.append(float(np.mean(vals)))
            ax.plot(xs, ys, "k-", lw=2.2, label="平均" if metric == specs[0][0] else None)
        ax.set_xlabel("训练轮次 epoch")
        ax.set_ylabel(title)
        ax.grid(alpha=0.3)
        ax.set_title(f"{title_prefix}{title}")
    axes[0].legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_confusion(cm, out_path, names=None, normalize=True):
    import matplotlib.pyplot as plt
    cm = np.asarray(cm, dtype=np.float64)
    if normalize:
        row = cm.sum(axis=1, keepdims=True)
        cm = cm / np.maximum(row, 1e-9)
    n = cm.shape[0]
    labels = [names.get(i, str(i)) if names else str(i) for i in range(n)]
    fig, ax = plt.subplots(figsize=(1.0 + 0.55 * n, 0.8 + 0.5 * n))
    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=cm.max() if cm.max() > 0 else 1)
    ax.set_xticks(range(n))
    ax.set_yticks(range(n))
    ax.set_xticklabels(labels, rotation=90, fontsize=7)
    ax.set_yticklabels(labels, fontsize=7)
    ax.set_xlabel("模型预测")
    ax.set_ylabel("真实标签")
    ax.set_title("测试集混淆矩阵（按行归一化）")
    fig.colorbar(im, ax=ax, fraction=0.03)
    fig.tight_layout()
    fig.savefig(out_path, dpi=170)
    plt.close(fig)
    return out_path


def plot_per_subject(per_subject, out_path):
    import matplotlib.pyplot as plt
    subs = [p["subject"] for p in per_subject]
    acc = [p["accuracy"] for p in per_subject]
    macc = [p["accuracy_movement_only"] for p in per_subject]
    x = np.arange(len(subs))
    fig, ax = plt.subplots(figsize=(max(7, 0.35 * len(subs)), 4))
    ax.bar(x - 0.2, acc, width=0.4, label="全部样本")
    ax.bar(x + 0.2, macc, width=0.4, label="仅动作样本")
    ax.axhline(np.mean(acc), color="r", ls="--", lw=1,
               label=f"平均 {np.mean(acc):.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels([f"S{s}" for s in subs], fontsize=8)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("准确率")
    ax.set_title("每位受试者的测试准确率（训练/测试按重复次数划分）")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)
    return out_path


def plot_angles(run_dir, out_path):
    """连续输出专用：① 时间跟踪曲线 ② 逐通道 R²/r 柱状图。"""
    import matplotlib.pyplot as plt
    run_dir = Path(run_dir)
    files = sorted(run_dir.glob("subjects/S*_angles.npz"))
    if not files:
        single = run_dir / "angles_test.npz"
        files = [single] if single.exists() else []
    if not files:
        return None
    yt = np.concatenate([np.load(f)["y_true"] for f in files])
    yp = np.concatenate([np.load(f)["y_pred"] for f in files])
    n_ch = yt.shape[1]
    r2 = [1 - ((yt[:, c] - yp[:, c]) ** 2).sum() /
          max(1e-9, ((yt[:, c] - yt[:, c].mean()) ** 2).sum()) for c in range(n_ch)]
    r = [float(np.corrcoef(yt[:, c], yp[:, c])[0, 1]) for c in range(n_ch)]
    labels = [f"#{c + 1 if c < 19 else c + 2}" for c in range(n_ch)]

    # ① 时间跟踪曲线：挑 R² 最好的 4 路
    best = list(np.argsort(r2)[::-1][:4])
    seg = slice(40 * 60, 40 * 90)          # 测试集 40 Hz 采样，取 60~90 s 一段
    t = np.arange(len(yt[seg])) / 40.0
    fig, axes = plt.subplots(len(best), 1, figsize=(11, 2.0 * len(best)), sharex=True)
    if len(best) == 1:
        axes = [axes]
    for ax, c in zip(axes, best):
        ax.plot(t, yt[seg, c], "k-", lw=1.4, label="真实关节角（Cyberglove）")
        ax.plot(t, yp[seg, c], "r-", lw=1.4, alpha=0.85, label="模型预测（8 通道 sEMG）")
        ax.set_ylabel(f"通道 {labels[c]}\n(标准化)", fontsize=8)
        ax.grid(alpha=0.3)
        ax.text(0.99, 0.05, f"r = {r[c]:.3f}", transform=ax.transAxes,
                ha="right", fontsize=9)
    axes[0].legend(fontsize=8, ncol=2)
    axes[-1].set_xlabel("测试集时间 (s)")
    fig.suptitle("连续关节角跟踪（测试集，窗口步长 25 ms）")
    fig.tight_layout()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)

    # ② 逐通道柱状图
    fig2, ax = plt.subplots(figsize=(11, 3.6))
    x = np.arange(n_ch)
    ax.bar(x - 0.2, r2, width=0.4, label="R²")
    ax.bar(x + 0.2, r, width=0.4, label="皮尔逊 r")
    ax.axhline(np.mean(r2), color="r", ls="--", lw=1, label=f"R² 平均 {np.mean(r2):.3f}")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=7)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Cyberglove 通道号（已排除第 20 路）")
    ax.set_ylabel("相关系数 / 决定系数")
    ax.set_title("每路关节角的连续预测精度")
    ax.grid(axis="y", alpha=0.3)
    ax.legend()
    fig2.tight_layout()
    p2 = Path(out_path).with_name("angle_per_channel.png")
    fig2.savefig(p2, dpi=160)
    plt.close(fig2)
    return out_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run", default=str(OUTPUT_DIR / "ex1_gesture"), help="结果目录")
    ap.add_argument("--per-class-table", action="store_true", help="打印每类指标")
    args = ap.parse_args()

    run_dir = Path(args.run)
    with open(run_dir / "metrics.json", "r", encoding="utf-8") as fh:
        m = json.load(fh)
    font = setup_matplotlib_chinese()
    if font is None:
        print("[warn] 没找到中文字体，图里中文可能显示为方块")

    name = run_dir.name
    plot_history(m["history"], run_dir / "curves.png", title_prefix=name + " ")
    n_cls = m["config"]["model"]["n_classes"]
    names = DB2_EX1_NAMES if n_cls == 18 else (APP_GESTURE_NAMES if n_cls == 7 else None)
    plot_confusion(m["confusion_matrix"], run_dir / "confusion_matrix.png", names=names)
    if m.get("per_subject"):
        plot_per_subject(m["per_subject"], run_dir / "per_subject.png")
    plot_angles(run_dir, run_dir / "angles_scatter.png")

    print(f"== {name} ==")
    print(f"准确率 {m['accuracy']:.4f} | 宏平均F1 {m['macro_f1']:.4f} | "
          f"平衡准确率 {m['balanced_accuracy']:.4f} | 仅动作 {m['accuracy_movement_only']:.4f}")
    print(f"受试者平均 {m.get('subject_accuracy_mean', float('nan')):.4f} ± "
          f"{m.get('subject_accuracy_std', float('nan')):.4f}")
    if "angles" in m:
        print(f"关节角: MAE {m['angles']['mae_mean']:.4f} (标准化单位), "
              f"R² {m['angles']['r2_mean']:.4f}")
    if args.per_class_table:
        print(f"{'类':>4} {'名称':<12} {'precision':>9} {'recall':>8} {'f1':>8} {'support':>8}")
        for row in m["per_class"]:
            nm = (names or {}).get(row["class"], "")
            print(f"{row['class']:>4} {nm:<12} {row['precision']:>9.3f} "
                  f"{row['recall']:>8.3f} {row['f1']:>8.3f} {row['support']:>8d}")
    print(f"图表已写入 {run_dir}")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""公共小工具：随机种子、指标计算、结果落盘、绘图字体。"""
import json
import random
from pathlib import Path

import numpy as np


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    import torch
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def pick_device(device: str = "auto"):
    import torch
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"
    if device == "cuda" and not torch.cuda.is_available():
        print("  [warn] 没有检测到 CUDA，自动改用 CPU")
        device = "cpu"
    return torch.device(device)


def confusion_matrix(y_true, y_pred, n_classes):
    cm = np.zeros((n_classes, n_classes), dtype=np.int64)
    np.add.at(cm, (np.asarray(y_true), np.asarray(y_pred)), 1)
    return cm


def classification_metrics(y_true, y_pred, n_classes=None) -> dict:
    """整体准确率 + 平衡准确率 + 宏平均 F1 + 每类 P/R/F1 + 混淆矩阵。"""
    y_true = np.asarray(y_true).astype(int)
    y_pred = np.asarray(y_pred).astype(int)
    if n_classes is None:
        n_classes = int(max(y_true.max(), y_pred.max())) + 1
    cm = confusion_matrix(y_true, y_pred, n_classes)

    per_class = []
    for c in range(n_classes):
        tp = cm[c, c]
        fp = cm[:, c].sum() - tp
        fn = cm[c, :].sum() - tp
        support = cm[c, :].sum()
        prec = tp / (tp + fp) if (tp + fp) else 0.0
        rec = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * prec * rec / (prec + rec) if (prec + rec) else 0.0
        per_class.append({"class": c, "precision": round(float(prec), 4),
                          "recall": round(float(rec), 4), "f1": round(float(f1), 4),
                          "support": int(support)})

    present = [m for m in per_class if m["support"] > 0]
    acc = float((y_true == y_pred).mean()) if y_true.size else 0.0
    macro_f1 = float(np.mean([m["f1"] for m in present])) if present else 0.0
    bal_acc = float(np.mean([m["recall"] for m in present])) if present else 0.0
    mv = y_true > 0
    move_acc = float((y_true[mv] == y_pred[mv]).mean()) if mv.any() else 0.0

    return {
        "accuracy": round(acc, 4),
        "balanced_accuracy": round(bal_acc, 4),
        "macro_f1": round(macro_f1, 4),
        "accuracy_movement_only": round(move_acc, 4),
        "n_samples": int(y_true.size),
        "per_class": per_class,
        "confusion_matrix": cm.tolist(),
    }


def save_json(obj, path):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        json.dump(obj, fh, ensure_ascii=False, indent=2)
    return path


def regression_metrics(y_true, y_pred):
    """连续关节角回归：逐通道的 MAE / RMSE / R² / 皮尔逊 r，以及它们的平均。

    R² 和 r 都是“逐通道算完再平均”，这样不会被量纲大的通道主导。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    mae = np.abs(y_true - y_pred).mean(axis=0)
    rmse = np.sqrt(((y_true - y_pred) ** 2).mean(axis=0))
    ss_res = ((y_true - y_pred) ** 2).sum(axis=0)
    ss_tot = ((y_true - y_true.mean(axis=0, keepdims=True)) ** 2).sum(axis=0)
    r2 = 1.0 - ss_res / np.maximum(ss_tot, 1e-12)
    r = np.zeros(y_true.shape[1])
    for c in range(y_true.shape[1]):
        a, b = y_true[:, c], y_pred[:, c]
        if a.std() > 1e-12 and b.std() > 1e-12:
            r[c] = float(np.corrcoef(a, b)[0, 1])
    return {
        "mae_mean": round(float(mae.mean()), 4),
        "rmse_mean": round(float(rmse.mean()), 4),
        "r2_mean": round(float(r2.mean()), 4),
        "pearson_mean": round(float(r.mean()), 4),
        "mae_per_channel": [round(float(v), 4) for v in mae],
        "rmse_per_channel": [round(float(v), 4) for v in rmse],
        "r2_per_channel": [round(float(v), 4) for v in r2],
        "pearson_per_channel": [round(float(v), 4) for v in r],
        # 显示用：目标标准化后 ±3σ 映射到 0~90°，所以 1σ ≈ 15°
        "mae_deg_mean": round(float(mae.mean()) * 15.0, 2),
        "mae_deg_per_channel": [round(float(v) * 15.0, 2) for v in mae],
    }


def setup_matplotlib_chinese():
    """让 matplotlib 能画中文（找不到中文字体就退回英文标签）。"""
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import font_manager
    candidates = ["Microsoft YaHei", "SimHei", "Noto Sans CJK SC",
                  "Source Han Sans SC", "DengXian", "Arial Unicode MS"]
    have = {f.name for f in font_manager.fontManager.ttflist}
    for name in candidates:
        if name in have:
            matplotlib.rcParams["font.sans-serif"] = [name]
            matplotlib.rcParams["axes.unicode_minus"] = False
            return name
    matplotlib.rcParams["axes.unicode_minus"] = False
    return None

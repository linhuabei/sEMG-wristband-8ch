# -*- coding: utf-8 -*-
"""把所有实验的结果汇总成一张表（用于写报告）。"""
import argparse
import json
from pathlib import Path

from config import OUTPUT_DIR


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=str(OUTPUT_DIR))
    ap.add_argument("--md", default=None, help="额外把表格存成 markdown")
    args = ap.parse_args()

    rows = []
    for mf in sorted(Path(args.root).glob("*/metrics.json")):
        try:
            with open(mf, "r", encoding="utf-8") as fh:
                m = json.load(fh)
        except Exception:                                   # noqa: BLE001
            continue
        if "accuracy" not in m:
            continue
        cfg = m.get("config", {})
        rows.append({
            "run": mf.parent.name,
            "模式": m.get("mode", "pooled"),
            "任务": ("连续角度" if cfg.get("train", {}).get("gesture_loss_weight", 1) == 0
                     else ("分类+角度" if cfg.get("model", {}).get("multi_task") else "离散手势")),
            "类别数": cfg.get("model", {}).get("n_classes", "?"),
            "预处理": f"{cfg.get('pre', {}).get('style', '?')}+"
                      f"{cfg.get('pre', {}).get('norm', '?')}",
            "准确率": m["accuracy"],
            "宏F1": m["macro_f1"],
            "仅动作": m["accuracy_movement_only"],
            "角度R²": m.get("angles", {}).get("r2_mean", ""),
            "角度r": m.get("angles", {}).get("pearson_mean", ""),
            "角度MAE°": m.get("angles", {}).get("mae_deg_mean", ""),
            "逐人均值": m.get("subject_accuracy_mean", ""),
            "逐人标准差": m.get("subject_accuracy_std", ""),
            "参数量": m.get("n_parameters", ""),
            "受试者": f"{len(m.get('subjects', []))} 人",
        })

    keys = ["run", "模式", "任务", "类别数", "预处理", "受试者", "准确率", "宏F1", "仅动作",
            "角度R²", "角度r", "角度MAE°", "逐人均值", "逐人标准差", "参数量"]
    widths = {k: max(len(k), *(len(str(r[k])) for r in rows)) if rows else len(k)
              for k in keys}
    line = "  ".join(k.ljust(widths[k]) for k in keys)
    print(line)
    print("-" * len(line))
    for r in rows:
        print("  ".join(str(r[k]).ljust(widths[k]) for k in keys))

    if args.md:
        with open(args.md, "w", encoding="utf-8") as fh:
            fh.write("| " + " | ".join(keys) + " |\n")
            fh.write("|" + "---|" * len(keys) + "\n")
            for r in rows:
                fh.write("| " + " | ".join(str(r[k]) for k in keys) + " |\n")
        print(f"\n已写入 {args.md}")


if __name__ == "__main__":
    main()

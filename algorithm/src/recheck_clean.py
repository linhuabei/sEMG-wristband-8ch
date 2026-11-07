# -*- coding: utf-8 -*-
"""用「严格划分」重新评估已经训练好的模型（不需要重训）。

严格划分 = 窗口内 200 个采样点必须全部落在同一个重复段里（require_same_repetition=True），
把跨重复边界的窗口（测试集里约 1.1%）全部剔掉，再算一遍指标，
用来量化"跨段窗口"到底有没有把成绩抬高。

用法：python recheck_clean.py --ckpt ../outputs/ex1_persubject_angles/subjects/S3.pt
"""
import argparse
from pathlib import Path

import numpy as np
import torch

from config import Config, OUTPUT_DIR
from dataset import build_index, prepare_subject
from model import CNNTransformer
from train import make_loader, evaluate
from utils import classification_metrics, pick_device, regression_metrics


def cfg_from_dict(d):
    cfg = Config()
    for name in ("data", "pre", "window", "model", "train"):
        obj = getattr(cfg, name)
        for k, v in d.get(name, {}).items():
            if hasattr(obj, k):
                setattr(obj, k, tuple(v) if isinstance(v, list) else v)
    return cfg


def run_once(ckpt_path, strict):
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = cfg_from_dict(ckpt["config"])
    cfg.window.require_same_repetition = strict
    subjects = ckpt.get("subjects") or cfg.data.subject_list[:1]
    device = pick_device("cpu")

    sigs, trs, vas, tes = [], [], [], []
    for s in subjects:
        sig = prepare_subject(s, cfg)
        tr, va, te = build_index([sig], cfg)
        sigs.append(sig)
        trs.append(tr)
        vas.append(va)
        tes.append(te)

    model = CNNTransformer(cfg.model, in_channels=len(cfg.data.emg_channels),
                           seq_len=cfg.window.length).to(device)
    model.load_state_dict(ckpt["state_dict"])
    dl = make_loader(sigs, np.concatenate(tes), cfg, train=False)
    res = evaluate(model, dl, device, cfg)

    out = {"n_windows": int(len(res["y_true"])),
           "n_train": int(sum(len(t) for t in trs)),
           "n_val": int(sum(len(v) for v in vas)),
           "_y_true": res["y_true"], "_y_pred": res["y_pred"]}
    if cfg.train.gesture_loss_weight > 0:
        m = classification_metrics(res["y_true"], res["y_pred"], cfg.model.n_classes)
        out.update({"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                    "accuracy_movement_only": m["accuracy_movement_only"]})
    if "angles_true" in res:
        pred = res["angles_pred"]
        calib = ckpt.get("calibration")
        if calib:
            pred = pred * np.asarray(calib["scale"]) + np.asarray(calib["bias"])
        a = regression_metrics(res["angles_true"], pred)
        out.update({"angle_r2": a["r2_mean"], "angle_r": a["pearson_mean"],
                    "angle_mae_deg": a["mae_deg_mean"]})
        out["_angles_true"] = res["angles_true"]
        out["_angles_pred"] = pred
    return out


def aggregate(rows, cfg_model, pure_reg):
    """把多位受试者的结果合并成一份汇总指标。"""
    y_true = np.concatenate([r["_y_true"] for r in rows])
    y_pred = np.concatenate([r["_y_pred"] for r in rows])
    out = {"n_windows": int(y_true.size)}
    if any("_angles_true" in r for r in rows):
        at = np.concatenate([r["_angles_true"] for r in rows])
        ap = np.concatenate([r["_angles_pred"] for r in rows])
        a = regression_metrics(at, ap)
        out.update({"angle_r2": a["r2_mean"], "angle_r": a["pearson_mean"],
                    "angle_mae_deg": a["mae_deg_mean"]})
    if not pure_reg:
        m = classification_metrics(y_true, y_pred, cfg_model.n_classes)
        out.update({"accuracy": m["accuracy"], "macro_f1": m["macro_f1"],
                    "accuracy_movement_only": m["accuracy_movement_only"]})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUTPUT_DIR / "ex1_persubject_angles" / "subjects" / "S3.pt"))
    ap.add_argument("--subjects", default=None, help="默认用 checkpoint 里记录的受试者")
    ap.add_argument("--run", default=None,
                    help="整个实验目录：对 subjects/S*.pt 逐个复查并汇总")
    args = ap.parse_args()

    if args.run:
        run_dir = Path(args.run)
        files = sorted((run_dir / "subjects").glob("S*.pt"))
        if not files:
            raise SystemExit(f"{run_dir} 下没有 subjects/S*.pt")
        cfg0 = cfg_from_dict(torch.load(files[0], map_location="cpu",
                                        weights_only=False)["config"])
        pure_reg = cfg0.train.gesture_loss_weight == 0
        print(f"实验目录 {run_dir.name}：共 {len(files)} 位受试者"
              f"（{'连续回归' if pure_reg else '分类'}任务）\n")
        res_loose, res_strict = [], []
        for f in files:
            a = run_once(f, strict=False)
            b = run_once(f, strict=True)
            res_loose.append(a)
            res_strict.append(b)
            key = "angle_r2" if pure_reg else "accuracy"
            print(f"  {f.stem:<5} 窗口 {a['n_windows']:6d} -> {b['n_windows']:6d}"
                  f" | {key}: {a[key]:.4f} -> {b[key]:.4f} (Δ {b[key] - a[key]:+.4f})")
        agg_l = aggregate(res_loose, cfg0.model, pure_reg)
        agg_s = aggregate(res_strict, cfg0.model, pure_reg)
        print(f"\n  汇总（{len(files)} 人合并）：")
        for k in [x for x in ("accuracy", "macro_f1", "accuracy_movement_only",
                              "angle_r2", "angle_r", "angle_mae_deg") if x in agg_l]:
            print(f"    {k:<24} {agg_l[k]:.4f}  ->  {agg_s[k]:.4f}"
                  f"   (Δ {agg_s[k] - agg_l[k]:+.4f})")
        return

    ckpt_path = Path(args.ckpt)
    if args.subjects:
        ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        subs = []
        for chunk in str(args.subjects).split(","):
            if "-" in chunk:
                a, b = chunk.split("-")
                subs.extend(range(int(a), int(b) + 1))
            else:
                subs.append(int(chunk))
        ckpt["subjects"] = subs
        torch.save(ckpt, ckpt_path)

    loose = run_once(ckpt_path, strict=False)
    strict = run_once(ckpt_path, strict=True)
    print(f"模型：{ckpt_path}")
    print(f"  窗口数：宽松 {loose['n_windows']} -> 严格 {strict['n_windows']}"
          f"（丢掉 {loose['n_windows'] - strict['n_windows']} 个跨段窗口）")
    print(f"  训练窗口：{loose['n_train']} -> {strict['n_train']}")
    keys = [k for k in ("accuracy", "macro_f1", "accuracy_movement_only",
                        "angle_r2", "angle_r", "angle_mae_deg") if k in loose]
    print("  指标（宽松 -> 严格）：")
    for k in keys:
        d = strict[k] - loose[k]
        print(f"    {k:<24} {loose[k]:.4f}  ->  {strict[k]:.4f}   (Δ {d:+.4f})")


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
"""训练 CNN-Transformer（NinaPro DB2）。

两种训练模式：
  per-subject  每人一个模型（NinaPro 论文的标准做法，精度最高）—— 默认
  pooled       所有人一起训练一个跨受试者通用模型

划分方式（两种模式一样）：
  第 1/3/4 次重复 → 训练   第 6 次重复 → 验证（早停）   第 2/5 次重复 → 测试
  注意是按“第几次重复”整段划分，绝不随机切窗口：相邻窗口有重叠，
  随机划分会让同一段动作同时进训练集和测试集，准确率会被严重高估。

用法（在 算法/src 下，用装了 torch 的 python 跑）：
    python train.py --mode per-subject --subjects 1-8 --epochs 25
    python train.py --mode pooled --subjects 1-19 --epochs 20
    python train.py --mode per-subject --subjects 1-3 --multi-task   # 同时回归关节角
    python train.py --subjects 1-2 --benchmark 20                    # 只测速
"""
import argparse
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from config import Config, OUTPUT_DIR
from dataset import WindowDataset, build_index, prepare_subject, split_summary
from model import CNNTransformer, count_parameters
from ninapro import ensure_cache
from utils import (classification_metrics, pick_device, regression_metrics,
                   save_json, set_seed)

_T0 = time.time()


def parse_subjects(text):
    """'1-19' / '1,2,5' / 'all'。"""
    if text is None or str(text).lower() in ("all", "*"):
        return None
    out = []
    for chunk in str(text).split(","):
        chunk = chunk.strip()
        if not chunk:
            continue
        if "-" in chunk:
            a, b = chunk.split("-")
            out.extend(range(int(a), int(b) + 1))
        else:
            out.append(int(chunk))
    return tuple(sorted(set(out)))


def build_config(args) -> Config:
    cfg = Config()
    if args.exclude_subjects is not None:
        cfg.data.exclude_subjects = tuple(
            int(x) for x in str(args.exclude_subjects).split(",") if x.strip() != "")
    subs = parse_subjects(args.subjects)
    if subs is not None:
        cfg.data.subjects = subs
    cfg.data.exercise = args.exercise
    if args.dataset == "custom":
        cfg.data.dataset = "custom"
        cfg.data.custom_dir = args.data_dir
        cfg.data.custom_pattern = args.data_pattern
        cfg.data.custom_subject_regex = args.custom_subject_regex
    cfg.pre.style = args.style
    cfg.pre.norm = args.norm
    cfg.pre.norm_window_s = args.norm_window
    cfg.window.length = args.window
    cfg.window.stride_train = args.stride_train
    cfg.window.stride_val = args.stride_val
    cfg.window.stride_test = args.stride_test
    if args.task == "gesture":
        cfg.model.multi_task = False
        cfg.train.gesture_loss_weight = 1.0
        cfg.train.angle_loss_weight = 0.0
    elif args.task == "angles":
        cfg.model.multi_task = True
        cfg.train.gesture_loss_weight = 0.0        # 纯连续回归
        cfg.train.angle_loss_weight = 1.0
    else:                                          # both
        cfg.model.multi_task = True
        cfg.train.gesture_loss_weight = 1.0
        cfg.train.angle_loss_weight = args.angle_loss_weight
    cfg.model.multi_task = cfg.model.multi_task or bool(args.multi_task)
    cfg.data.label_mode = args.label_mode
    cfg.model.n_classes = (7 if args.label_mode == "app6"
                           else {1: 18, 2: 24, 3: 10}[args.exercise])
    cfg.model.n_angles = 22 - len(cfg.data.angle_exclude_channels)
    cfg.train.epochs = args.epochs
    cfg.train.batch_size = args.batch_size
    cfg.train.lr = args.lr
    cfg.train.seed = args.seed
    cfg.train.device = args.device
    cfg.train.augment_time_flip = not args.no_augment
    cfg.train.augment_channel_shuffle = bool(args.channel_shuffle) and not args.no_augment
    cfg.train.class_weight = args.class_weight
    cfg.train.early_stop_patience = args.patience
    # 模型规模 / 训练窗口密度（用于"加大训练量"的对照实验）
    if args.d_model:
        cfg.model.d_model = args.d_model
        cfg.model.conv2_channels = args.d_model
    if args.conv1:
        cfg.model.conv1_channels = args.conv1
    if args.layers:
        cfg.model.n_layers = args.layers
    if args.heads:
        cfg.model.n_heads = args.heads
    if args.ff:
        cfg.model.dim_feedforward = args.ff
    cfg.data.angle_smooth_hz = args.angle_smooth
    cfg.window.angle_target = args.angle_target
    cfg.train.calibrate = not args.no_calibrate
    return cfg


def cfg_to_dict(cfg: Config):
    d = {}
    for name in ("data", "pre", "window", "model", "train"):
        obj = getattr(cfg, name)
        d[name] = {k: (list(v) if isinstance(v, tuple) else v)
                   for k, v in obj.__dict__.items()}
    return d


def make_loader(signals, pos, cfg, train):
    ds = WindowDataset(signals, pos, cfg, train=train, seed=cfg.train.seed)
    return DataLoader(ds, batch_size=cfg.train.batch_size if train else 256,
                      shuffle=train, num_workers=cfg.train.num_workers)


def run_epoch(model, loader, device, cfg, optimizer=None, criterion=None, collect=False):
    train = optimizer is not None
    model.train(train)
    total_loss, n_total, n_correct = 0.0, 0, 0
    ys, ps, angs, angp = [], [], [], []
    for batch in loader:
        x = batch["x"].to(device)
        y = batch["y"].to(device)
        with torch.set_grad_enabled(train):
            out = model(x)
            loss = torch.zeros((), device=device)
            if cfg.train.gesture_loss_weight > 0:
                loss = loss + cfg.train.gesture_loss_weight * criterion(out["logits"], y)
            if (cfg.model.multi_task and "angles" in batch
                    and cfg.train.angle_loss_weight > 0):
                ang = batch["angles"].to(device)
                loss = loss + cfg.train.angle_loss_weight * nn.functional.mse_loss(
                    out["angles"], ang)
            if train:
                optimizer.zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), 5.0)
                optimizer.step()
        total_loss += float(loss.item()) * y.numel()
        n_total += y.numel()
        n_correct += int((out["logits"].argmax(dim=1) == y).sum().item())
        if collect:
            ys.append(y.cpu().numpy())
            ps.append(out["logits"].argmax(dim=1).cpu().numpy())
            if cfg.model.multi_task and "angles" in batch:
                angs.append(batch["angles"].numpy())
                angp.append(out["angles"].cpu().numpy())
    res = [total_loss / max(1, n_total), n_correct / max(1, n_total)]
    if collect:
        res.append((np.concatenate(ys), np.concatenate(ps)) if ys else (None, None))
        res.append((np.concatenate(angs), np.concatenate(angp)) if angs else (None, None))
    return res


def evaluate(model, loader, device, cfg):
    model.eval()
    ys, ps, subs, angs, angs_pred = [], [], [], [], []
    with torch.no_grad():
        for batch in loader:
            out = model(batch["x"].to(device))
            ys.append(batch["y"].numpy())
            ps.append(out["logits"].argmax(dim=1).cpu().numpy())
            subs.append(batch["subject"].numpy())
            if cfg.model.multi_task and "angles" in batch:
                angs.append(batch["angles"].numpy())
                angs_pred.append(out["angles"].cpu().numpy())
    res = {"y_true": np.concatenate(ys) if ys else np.empty(0, dtype=int),
           "y_pred": np.concatenate(ps) if ps else np.empty(0, dtype=int),
           "subject": np.concatenate(subs) if subs else np.empty(0, dtype=int)}
    if angs:
        res["angles_true"] = np.concatenate(angs)
        res["angles_pred"] = np.concatenate(angs_pred)
    return res


def class_weight_tensor(counts, device):
    cnt = np.asarray(counts, dtype=np.float64)
    present = cnt > 0
    w = np.zeros_like(cnt)
    w[present] = cnt.sum() / (present.sum() * cnt[present])
    return torch.tensor(w, dtype=torch.float32, device=device)


def fit_affine(y_true, y_pred):
    """逐通道最小二乘仿射校准 y ≈ a·ŷ + b（只用验证集拟合，不碰测试集）。

    连续输出常有的问题：预测的幅度范围偏小（回归到均值）、不同通道偏置不同。
    用一个仿射映射就能把这类系统性误差修掉，是很划算的一步“个体校准”。
    """
    y_true = np.asarray(y_true, dtype=np.float64)
    y_pred = np.asarray(y_pred, dtype=np.float64)
    a = np.zeros(y_true.shape[1])
    b = np.zeros(y_true.shape[1])
    for c in range(y_true.shape[1]):
        x, y = y_pred[:, c], y_true[:, c]
        vx = x.var()
        if vx < 1e-12:
            a[c], b[c] = 0.0, float(y.mean())
        else:
            aa = float(((x - x.mean()) * (y - y.mean())).mean() / vx)
            a[c], b[c] = aa, float(y.mean() - aa * x.mean())
    return a, b


def train_one(cfg: Config, subjects, out_dir: Path, tag: str, quiet=False):
    """在一组受试者上训练一个模型并评估，返回指标字典（含 _y_true/_y_pred）。"""
    device = pick_device(cfg.train.device)
    signals = [prepare_subject(s, cfg) for s in subjects]
    train_pos, val_pos, test_pos = build_index(signals, cfg)
    counts = split_summary(signals, train_pos, test_pos, cfg)
    if not quiet:
        print(f"  [{tag}] 训练/验证/测试窗口 = {len(train_pos)}/{len(val_pos)}/"
              f"{len(test_pos)}；训练集休息类占 "
              f"{counts['train'][0] / max(1, counts['train'].sum()):.1%}")

    dl_train = make_loader(signals, train_pos, cfg, train=True)
    dl_val = make_loader(signals, val_pos, cfg, train=False)
    dl_test = make_loader(signals, test_pos, cfg, train=False)
    model = CNNTransformer(cfg.model, in_channels=len(cfg.data.emg_channels),
                           seq_len=cfg.window.length).to(device)
    criterion = nn.CrossEntropyLoss(
        label_smoothing=cfg.train.label_smoothing,
        weight=(class_weight_tensor(counts["train"], device)
                if cfg.train.class_weight == "balanced" else None))
    optimizer = torch.optim.AdamW(model.parameters(), lr=cfg.train.lr,
                                  weight_decay=cfg.train.weight_decay)
    scheduler = (torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=cfg.train.epochs)
                 if cfg.train.cosine_annealing else None)

    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"{tag}.pt"
    history, best_score, best_epoch, best_acc, bad = [], -1.0, -1, 0.0, 0
    for epoch in range(1, cfg.train.epochs + 1):
        t0 = time.time()
        tr_loss, tr_acc = run_epoch(model, dl_train, device, cfg, optimizer, criterion)
        va_loss, va_acc, (vy, vp), (vay, vap) = run_epoch(
            model, dl_val, device, cfg, None, criterion, collect=True)
        va = classification_metrics(vy, vp, cfg.model.n_classes)
        vang = regression_metrics(vay, vap) if vay is not None else None
        # 选模型的指标：连续任务看验证集 R²；分类任务在类别加权时看宏平均 F1，
        # 否则看准确率（准确率会被占一半的“休息”类主导）
        if cfg.train.gesture_loss_weight == 0 and vang is not None:
            score = vang["r2_mean"]
        elif cfg.train.class_weight == "balanced":
            score = va["macro_f1"]
        else:
            score = va_acc
        if scheduler is not None:
            scheduler.step()
        history.append({"epoch": epoch, "train_loss": round(tr_loss, 4),
                        "train_acc": round(tr_acc, 4), "val_loss": round(va_loss, 4),
                        "val_acc": round(va_acc, 4), "val_macro_f1": va["macro_f1"],
                        "val_movement_acc": va["accuracy_movement_only"],
                        "val_angle_r2": (vang or {}).get("r2_mean", None),
                        "val_angle_pearson": (vang or {}).get("pearson_mean", None),
                        "lr": round(optimizer.param_groups[0]["lr"], 6),
                        "seconds": round(time.time() - t0, 1)})
        flag = ""
        if score > best_score:
            best_score, best_epoch, best_acc, bad = score, epoch, va_acc, 0
            torch.save({"state_dict": model.state_dict(), "config": cfg_to_dict(cfg),
                        "subjects": [int(s) for s in subjects],
                        "val_score": best_score, "val_acc": best_acc,
                        "epoch": epoch}, best_path)
            flag = "  <- 保存最优"
        else:
            bad += 1
        if not quiet:
            print(f"      epoch {epoch:2d}/{cfg.train.epochs} "
                  f"train {tr_loss:.4f}/{tr_acc:.4f} | val {va_loss:.4f}/{va_acc:.4f} "
                  f"F1 {va['macro_f1']:.4f} mvAcc {va['accuracy_movement_only']:.4f}"
                  + (f" 角度R² {vang['r2_mean']:.3f} r {vang['pearson_mean']:.3f}"
                     if vang else "") + " | "
                  f"{history[-1]['seconds']}s{flag}")
        if bad >= cfg.train.early_stop_patience:
            if not quiet:
                print(f"      验证集连续 {cfg.train.early_stop_patience} 轮没提升，提前停止")
            break

    ckpt = torch.load(best_path, map_location=device, weights_only=False)
    model.load_state_dict(ckpt["state_dict"])

    # 连续输出：用验证集拟合逐通道仿射校准，并记录校准前后的指标
    calib, raw_angles = None, None
    if cfg.model.multi_task and cfg.train.angle_loss_weight > 0 and cfg.train.calibrate:
        rv = evaluate(model, dl_val, device, cfg)
        if "angles_true" in rv:
            a, b = fit_affine(rv["angles_true"], rv["angles_pred"])
            calib = {"scale": [round(float(v), 6) for v in a],
                     "bias": [round(float(v), 6) for v in b]}
            ckpt["calibration"] = calib
            torch.save(ckpt, best_path)
            raw_angles = regression_metrics(rv["angles_true"], rv["angles_pred"])

    res = evaluate(model, dl_test, device, cfg)
    if calib is not None and "angles_pred" in res:
        res["angles_pred"] = (res["angles_pred"] * np.asarray(calib["scale"])
                              + np.asarray(calib["bias"]))
    m = classification_metrics(res["y_true"], res["y_pred"], cfg.model.n_classes)
    m["_y_true"] = res["y_true"]
    m["_y_pred"] = res["y_pred"]
    m["_subject"] = res["subject"]
    m["history"] = history
    m["best_epoch"] = best_epoch
    m["best_val_acc"] = round(float(best_acc), 4)
    m["n_parameters"] = int(count_parameters(model))
    m["checkpoint"] = str(best_path)
    if "angles_true" in res:
        m["angles"] = regression_metrics(res["angles_true"], res["angles_pred"])
        if calib is not None:
            m["angles_val_before_calibration"] = raw_angles
            m["calibration"] = calib
        np.savez(out_dir / f"{tag}_angles.npz", y_true=res["angles_true"],
                 y_pred=res["angles_pred"], subject=res["subject"])
    return m


def subject_breakdown(y_true, y_pred, subject):
    out = []
    for s in np.unique(subject):
        m = subject == s
        mv = y_true[m] > 0
        out.append({"subject": int(s),
                    "accuracy": round(float((y_true[m] == y_pred[m]).mean()), 4),
                    "accuracy_movement_only": round(
                        float((y_true[m][mv] == y_pred[m][mv]).mean()) if mv.any() else 0.0, 4),
                    "n": int(m.sum())})
    return out


def report(m, out_dir):
    pure_reg = (m.get("config", {}).get("train", {}).get("gesture_loss_weight", 1.0) == 0)
    if not pure_reg:
        print(f"      测试集准确率 {m['accuracy']:.4f} | 宏平均F1 {m['macro_f1']:.4f} | "
              f"平衡准确率 {m['balanced_accuracy']:.4f} | "
              f"仅动作样本 {m['accuracy_movement_only']:.4f}")
    if "subject_accuracy_mean" in m and not pure_reg:
        print(f"      逐受试者准确率 {m['subject_accuracy_mean']:.4f} ± "
              f"{m['subject_accuracy_std']:.4f}"
              f"（{m['subject_accuracy_min']:.4f} ~ {m['subject_accuracy_max']:.4f}）")
    if "angles" in m:
        per_ch = (m["angles"].get("r2_per_channel")
                  or m.get("angles_per_channel_mean", {}).get("r2_per_channel") or [])
        print(f"      连续关节角：R² {m['angles']['r2_mean']:.4f}，"
              f"逐通道平均 r {m['angles']['pearson_mean']:.4f}，"
              f"MAE {m['angles']['mae_deg_mean']:.2f}°（{len(per_ch)} 路）")
    print(f"      结果目录 {out_dir}（总用时 {time.time() - _T0:.0f}s）")


def run_pooled(cfg, subjects, out_dir):
    print(f"      跨受试者统一模型（{len(subjects)} 人一起训练）")
    m = train_one(cfg, subjects, out_dir, "pooled")
    m["per_subject"] = subject_breakdown(m["_y_true"], m["_y_pred"], m["_subject"])
    m["mode"] = "pooled"
    m["subjects"] = [int(s) for s in subjects]
    m["config"] = cfg_to_dict(cfg)
    np.savez(out_dir / "predictions_test.npz", y_true=m.pop("_y_true"),
             y_pred=m.pop("_y_pred"), subject=m.pop("_subject"))
    save_json(m, out_dir / "metrics.json")
    report(m, out_dir)


def run_per_subject(cfg, subjects, out_dir):
    print(f"      逐受试者训练：每人一个模型，共 {len(subjects)} 个")
    sub_dir = out_dir / "subjects"
    rows = []
    for i, s in enumerate(subjects, 1):
        print(f"      --- [{i}/{len(subjects)}] 受试者 S{s} ---")
        m = train_one(cfg, [s], sub_dir, f"S{s}")
        rows.append((int(s), m))
        if cfg.train.gesture_loss_weight == 0 and "angles" in m:
            print(f"      S{s}: 关节角 R² {m['angles']['r2_mean']:.4f} | "
                  f"r {m['angles']['pearson_mean']:.4f} | "
                  f"MAE {m['angles']['mae_deg_mean']:.2f}° | 最优 epoch {m['best_epoch']}")
        else:
            print(f"      S{s}: 准确率 {m['accuracy']:.4f} | 宏F1 {m['macro_f1']:.4f} | "
                  f"仅动作 {m['accuracy_movement_only']:.4f} | 最优 epoch {m['best_epoch']}")

    y_true = np.concatenate([m["_y_true"] for _, m in rows])
    y_pred = np.concatenate([m["_y_pred"] for _, m in rows])
    subject = np.concatenate([m["_subject"] for _, m in rows])
    agg = classification_metrics(y_true, y_pred, cfg.model.n_classes)
    accs = np.array([m["accuracy"] for _, m in rows])
    maccs = np.array([m["accuracy_movement_only"] for _, m in rows])
    f1s = np.array([m["macro_f1"] for _, m in rows])
    # 连续任务按 R² 选“最好的那个受试者”，分类任务按准确率选
    if cfg.train.gesture_loss_weight == 0 and "angles" in rows[0][1]:
        best_i = int(np.argmax([m["angles"]["r2_mean"] for _, m in rows]))
    else:
        best_i = int(np.argmax(accs))
    agg.update({
        "mode": "per-subject",
        "subjects": [s for s, _ in rows],
        "subject_accuracy_mean": round(float(accs.mean()), 4),
        "subject_accuracy_std": round(float(accs.std()), 4),
        "subject_accuracy_min": round(float(accs.min()), 4),
        "subject_accuracy_max": round(float(accs.max()), 4),
        "subject_movement_accuracy_mean": round(float(maccs.mean()), 4),
        "subject_macro_f1_mean": round(float(f1s.mean()), 4),
        "per_subject": subject_breakdown(y_true, y_pred, subject),
        "history": [{"subject": s, "history": m["history"]} for s, m in rows],
        "best_subject": int(rows[best_i][0]),
        "n_parameters": rows[0][1].get("n_parameters"),
        "config": cfg_to_dict(cfg),
    })
    if "angles" in rows[0][1]:
        keys = ["mae_mean", "rmse_mean", "r2_mean", "pearson_mean", "mae_deg_mean"]
        agg["angles"] = {k: round(float(np.mean([m["angles"][k] for _, m in rows])), 4)
                         for k in keys}
        agg["angles_per_channel_mean"] = {
            key: [round(float(v), 4)
                  for v in np.array([m["angles"][key] for _, m in rows]).mean(axis=0)]
            for key in ("r2_per_channel", "pearson_per_channel", "mae_deg_per_channel")}
        agg["angles_per_subject"] = [{"subject": s, **m["angles"]} for s, m in rows]
    np.savez(out_dir / "predictions_test.npz", y_true=y_true, y_pred=y_pred,
             subject=subject)
    save_json(agg, out_dir / "metrics.json")
    import shutil
    shutil.copyfile(Path(rows[best_i][1]["checkpoint"]), out_dir / "best.pt")
    print(f"      把表现最好的 S{agg['best_subject']}（准确率 {accs[best_i]:.4f}）"
          f"的模型另存为 {out_dir / 'best.pt'}（供导出上位机插件）")
    report(agg, out_dir)


def main():
    ap = argparse.ArgumentParser(description="NinaPro DB2 + CNN-Transformer 训练")
    ap.add_argument("--mode", default="per-subject", choices=["per-subject", "pooled"])
    ap.add_argument("--subjects", default="1-8", help="受试者，如 1-8 / 1,3,5 / all")
    ap.add_argument("--exclude-subjects", default=None, help="额外排除的受试者号，如 20")
    ap.add_argument("--exercise", type=int, default=1, help="1=练习B(17动作) 2=练习C 3=练习D")
    ap.add_argument("--dataset", default="DB2", choices=["DB2", "custom"],
                    help="DB2=公开数据集；custom=自采数据（见 custom_dataset.py）")
    ap.add_argument("--data-dir", default="", help="自采数据目录")
    ap.add_argument("--data-pattern", default="*.npz", help="自采数据文件名匹配")
    ap.add_argument("--custom-subject-regex", default=r"[Ss](\d+)",
                    help="从文件名里抽受试者号的正则")
    ap.add_argument("--style", default="online", choices=["online", "offline"],
                    help="online=跟上位机一致的因果滤波（默认）；offline=零相位 filtfilt")
    ap.add_argument("--norm", default="rolling", choices=["rolling", "global"],
                    help="rolling=因果滚动标准化（=插件做法，默认）；global=整段标准化")
    ap.add_argument("--norm-window", type=float, default=6.0, help="滚动标准化窗口（秒）")
    ap.add_argument("--label-mode", default="ninapro", choices=["ninapro", "app6"],
                    help="ninapro=18 类；app6=映射成上位机的 6 种手势 + 休息（7 类）")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--batch-size", type=int, default=128)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--window", type=int, default=200, help="窗口长度（2 kHz 下 200 = 100 ms）")
    ap.add_argument("--stride-train", type=int, default=200)
    ap.add_argument("--stride-val", type=int, default=200)
    ap.add_argument("--stride-test", type=int, default=50)
    ap.add_argument("--multi-task", action="store_true", help="同时回归 Cyberglove 关节角")
    ap.add_argument("--task", default="gesture", choices=["gesture", "angles", "both"],
                    help="gesture=离散手势分类；angles=连续关节角回归；both=多任务")
    ap.add_argument("--angle-loss-weight", type=float, default=0.5,
                    help="--task both 时角度损失权重")
    ap.add_argument("--angle-smooth", type=float, default=5.0,
                    help="关节角目标低通截止频率(Hz)，0=不滤")
    ap.add_argument("--angle-target", default="tail", choices=["tail", "window"],
                    help="回归目标取窗口最后 25 ms(tail) 还是整窗平均")
    ap.add_argument("--no-calibrate", action="store_true",
                    help="关闭逐通道仿射校准（连续输出用验证集拟合，默认开）")
    ap.add_argument("--no-augment", action="store_true", help="关闭数据增强")
    ap.add_argument("--channel-shuffle", action="store_true", help="打开通道重排增强（默认关）")
    ap.add_argument("--class-weight", default="none", choices=["none", "balanced"],
                    help="balanced = 类别加权，压制“一直猜休息”")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--patience", type=int, default=6, help="早停耐心（验证指标连续多少轮不升）")
    ap.add_argument("--d-model", type=int, default=0, help="主干特征维度（默认 64；128 = 更大模型）")
    ap.add_argument("--conv1", type=int, default=0, help="第一层卷积输出通道（默认 32）")
    ap.add_argument("--layers", type=int, default=0, help="Transformer 层数（默认 2）")
    ap.add_argument("--heads", type=int, default=0, help="注意力头数（默认 8）")
    ap.add_argument("--ff", type=int, default=0, help="前馈隐藏维度（默认 128）")
    ap.add_argument("--device", default="auto")
    ap.add_argument("--threads", type=int, default=0)
    ap.add_argument("--name", default=None, help="结果目录名")
    ap.add_argument("--benchmark", type=int, default=0, help="只测速：跑 N 个 batch")
    args = ap.parse_args()

    cfg = build_config(args)
    set_seed(cfg.train.seed)
    if args.threads > 0:
        torch.set_num_threads(args.threads)

    subjects = cfg.data.subject_list
    tag = args.name or ("ex%d_%s_%s" % (
        cfg.data.exercise, args.mode.replace("-", ""), args.task)
        + ("" if cfg.data.label_mode == "ninapro" else "_app6"))
    out_dir = OUTPUT_DIR / tag
    out_dir.mkdir(parents=True, exist_ok=True)

    device = pick_device(cfg.train.device)
    if cfg.data.dataset == "custom":
        from custom_dataset import list_recordings
        recs = list_recordings(cfg)
        subjects = sorted(recs)
        cfg.data.subjects = tuple(subjects)
        print(f"[1/3] 自采数据目录 {Path(cfg.data.custom_dir)}："
              f"{len(subjects)} 位受试者，共 {sum(len(v) for v in recs.values())} 条记录")
        mx = 0                                  # 类别数按数据里的最大动作号自动确定
        for files in recs.values():
            for f in files:
                with np.load(f, allow_pickle=True) as z:
                    if "label" in z.files:
                        mx = max(mx, int(np.asarray(z["label"]).max()))
        cfg.model.n_classes = max(2, mx + 1)
        print(f"      识别类别数（自动）：{cfg.model.n_classes}（含 0 = 休息）")
    else:
        print(f"[1/3] 数据缓存（DB2 练习 {cfg.data.exercise}）...")
        ensure_cache(cfg.data, verbose=False)
    print(f"      受试者 {len(subjects)} 位：{subjects[0]}..{subjects[-1]}"
          f"（配置里排除 {list(cfg.data.exclude_subjects)} 号）")
    print(f"      sEMG 取通道 {list(cfg.data.emg_channels)}；关节角保留 "
          f"{len(cfg.data.angle_keep)} 路（排除 {list(cfg.data.angle_exclude_channels)} 号）")
    print(f"      模式 {args.mode}；设备 {device}；窗口 {cfg.window.length} 点；"
          f"步长 训练{cfg.window.stride_train}/验证{cfg.window.stride_val}/测试{cfg.window.stride_test}")

    if args.benchmark:
        print("[2/3] 测速 ...")
        signals = [prepare_subject(s, cfg) for s in subjects[:2]]
        tr, _, _ = build_index(signals, cfg)
        dl = make_loader(signals, tr, cfg, train=True)
        model = CNNTransformer(cfg.model, in_channels=len(cfg.data.emg_channels),
                               seq_len=cfg.window.length).to(device)
        criterion, opt = nn.CrossEntropyLoss(), torch.optim.AdamW(model.parameters(), lr=1e-3)
        it = iter(dl)
        t0 = time.time()
        for _ in range(args.benchmark):
            b = next(it)
            loss = criterion(model(b["x"].to(device))["logits"], b["y"].to(device))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            opt.step()
        dt = time.time() - t0
        print(f"      {args.benchmark} 个 batch 用时 {dt:.2f}s "
              f"({dt / args.benchmark * 1000:.0f} ms/batch)；"
              f"全量 {len(dl)} batch/epoch ≈ {dt / args.benchmark * len(dl):.1f}s/epoch")
        return

    print("[2/3] 训练 ...")
    if args.mode == "pooled":
        run_pooled(cfg, subjects, out_dir)
    else:
        run_per_subject(cfg, subjects, out_dir)


if __name__ == "__main__":
    main()

# -*- coding: utf-8 -*-
r"""用户迁移微调（个人化）：用上位机采的个人数据，把通用模型调成"你自己的模型"。

流程（对应上位机里的操作）：
  1. 上位机「交互训练」页按提示做几轮固定动作，存成 CSV（自带时间戳）；
  2. 参考角度（数据手套/动捕）另存一份 CSV，用 merge_app_csv.py 按时间戳合并；
  3. 本脚本读入预训练权重（.npz）+ 你的数据，微调 20 轮左右；
  4. 导出新的 .npz，丢回 plugins 目录，上位机重新导入插件即可用。

为什么要离线做：微调要**反向传播**，需要 PyTorch；上位机的运行环境只有 numpy。
所以"训练在上位机外面做，推理在上位机里面做"是这套架构的固定分工。

用法：
    python finetune.py --base ../plugins/模型权重_连续关节角.npz \
        --data-dir "../outputs/自采格式样例" --subjects 2 \
        --epochs 20 --lr 2e-4 --out ../outputs/迁移后_连续关节角.npz
"""
import argparse
import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from config import Config, OUTPUT_DIR
from dataset import build_index
from model import CNNTransformer
from numpy_infer import export_weights
from train import cfg_to_dict, make_loader, run_epoch, evaluate, fit_affine
from utils import (classification_metrics, pick_device, regression_metrics,
                   save_json, set_seed)


def cfg_from_npz(npz_path, data_dir, pattern, regex):
    """从权重文件里读回模型配置，并挂上自采数据。"""
    with np.load(npz_path, allow_pickle=False) as z:
        raw = z["__config__"]
    meta = json.loads(str(raw))
    cfg = Config()
    cfg.data.dataset = "custom"
    cfg.data.custom_dir = str(data_dir)
    cfg.data.custom_pattern = pattern
    cfg.data.custom_subject_regex = regex
    cfg.model.n_classes = int(meta["n_classes"])
    cfg.model.n_angles = int(meta["n_angles"])
    cfg.model.multi_task = bool(meta["multi_task"])
    cfg.model.d_model = int(meta["d_model"])
    cfg.model.n_heads = int(meta["n_heads"])
    cfg.model.n_layers = int(meta["n_layers"])
    cfg.window.length = int(meta["seq_len"])
    cfg.pre.fs = int(meta.get("fs", 2000))
    return cfg, meta


def load_weights(npz_path, model):
    with np.load(npz_path, allow_pickle=False) as z:
        sd = {k: torch.tensor(np.asarray(z[k])) for k in z.files if k != "__config__"}
    missing, unexpected = model.load_state_dict(sd, strict=False)
    if unexpected:
        print(f"  [warn] 权重里有 {len(unexpected)} 个用不上的张量：{unexpected[:3]}")
    return model


def freeze_front(model):
    for name in ("conv1", "bn1", "conv2", "bn2"):
        module = getattr(model, name, None)
        if module is not None:
            for p in module.parameters():
                p.requires_grad = False
    frozen = sum(1 for p in model.parameters() if not p.requires_grad)
    print(f"  已冻结卷积前端（{frozen} 组参数不更新），只训练 Transformer 与输出头")


def main():
    ap = argparse.ArgumentParser(description="用户迁移微调")
    ap.add_argument("--base", required=True, help="预训练权重 .npz（plugins 里那份）")
    ap.add_argument("--data-dir", required=True, help="你的自采数据目录")
    ap.add_argument("--eval-data-dir", default=None,
                    help="评估用目录（默认同 --data-dir）。做伪标签实验时，"
                         "训练用伪标签目录、评估用有真角度的目录")
    ap.add_argument("--subjects", default="1", help="用哪位受试者/用户的数据微调")
    ap.add_argument("--data-pattern", default="*.npz")
    ap.add_argument("--subject-regex", default=r"[Ss](\d+)")
    ap.add_argument("--epochs", type=int, default=20)
    ap.add_argument("--lr", type=float, default=2e-4)
    ap.add_argument("--batch-size", type=int, default=64)
    ap.add_argument("--patience", type=int, default=6)
    ap.add_argument("--freeze-front", action="store_true", help="冻结卷积前端（数据少时更稳）")
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--out", default=None, help="导出的新权重 .npz")
    ap.add_argument("--tag", default="迁移后")
    args = ap.parse_args()

    set_seed(args.seed)
    base_path = Path(args.base)
    cfg, meta = cfg_from_npz(base_path, args.data_dir, args.data_pattern, args.subject_regex)
    cfg.train.epochs = args.epochs
    cfg.train.lr = args.lr
    cfg.train.batch_size = args.batch_size
    cfg.train.early_stop_patience = args.patience
    cfg.train.angle_loss_weight = 1.0
    cfg.train.gesture_loss_weight = 0.0 if not cfg.model.multi_task else 1.0
    pure_reg = (not cfg.model.multi_task) or meta.get("multi_task") is False
    subjects = [int(x) for x in str(args.subjects).split(",")]
    device = pick_device("cpu")

    # ---- 数据 ----
    import copy
    from dataset import prepare_subject
    sigs = [prepare_subject(s, cfg, verbose=True) for s in subjects]
    tr, va, te = build_index(sigs, cfg)
    print(f"  个人数据窗口：训练 {len(tr)} / 验证 {len(va)} / 测试 {len(te)}")
    if len(tr) < 200:
        print("  [警告] 训练窗口太少，建议每人每个动作至少 5 次重复")
    # 评估数据（默认与训练相同；伪标签实验里换成有真角度的目录）
    cfg_eval = copy.deepcopy(cfg)
    if args.eval_data_dir:
        cfg_eval.data.custom_dir = args.eval_data_dir
        eval_sigs = [prepare_subject(s, cfg_eval) for s in subjects]
        eval_te = build_index(eval_sigs, cfg_eval)[2]
        print(f"  评估数据：{args.eval_data_dir}（测试窗口 {len(eval_te)}，真角度）")
    else:
        eval_sigs, eval_te = sigs, te

    def build_model():
        m = CNNTransformer(cfg.model, in_channels=8, seq_len=cfg.window.length).to(device)
        return load_weights(base_path, m)

    def report(tag, model, calib=None):
        res = evaluate(model, make_loader(eval_sigs, eval_te, cfg_eval, train=False),
                       device, cfg_eval)
        out = {}
        if "angles_true" in res:
            pred = res["angles_pred"]
            if calib:
                pred = pred * np.asarray(calib["scale"]) + np.asarray(calib["bias"])
            a = regression_metrics(res["angles_true"], pred)
            out.update({"R²": a["r2_mean"], "平均r": a["pearson_mean"],
                        "MAE°": a["mae_deg_mean"]})
        if not pure_reg:
            m = classification_metrics(res["y_true"], res["y_pred"], cfg.model.n_classes)
            out.update({"准确率": m["accuracy"], "宏F1": m["macro_f1"]})
        txt = "  ".join(f"{k} {v:.4f}" for k, v in out.items())
        print(f"  [{tag}] {txt}")
        return out, res

    print("\n[1/4] 微调前（通用模型直接用在这个人身上）")
    model = build_model()
    model.eval()
    before, _ = report("微调前", model)

    # ---- 微调 ----
    print(f"\n[2/4] 开始微调（{args.epochs} 轮，学习率 {args.lr}）")
    if args.freeze_front:
        freeze_front(model)
    crit = nn.CrossEntropyLoss()
    opt = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad],
                            lr=args.lr, weight_decay=1e-4)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs)
    dl_tr = make_loader(sigs, tr, cfg, train=True)
    dl_va = make_loader(sigs, va, cfg, train=False)
    best, best_state, bad = -9e9, None, 0
    for ep in range(1, args.epochs + 1):
        tl, ta = run_epoch(model, dl_tr, device, cfg, opt, crit)
        vl, va_acc, (vy, vp), (vay, vap) = run_epoch(model, dl_va, device, cfg,
                                                     None, crit, collect=True)
        score = (regression_metrics(vay, vap)["r2_mean"] if vay is not None else va_acc)
        sched.step()
        flag = ""
        if score > best:
            best, bad = score, 0
            best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            flag = "  <- 保存"
        else:
            bad += 1
        print(f"  第 {ep:2d} 轮 训练损失 {tl:.4f} | 验证 {score:.4f}{flag}")
        if bad >= args.patience:
            print("  验证集不再提升，提前停止")
            break
    if best_state:
        model.load_state_dict(best_state)

    # ---- 重新校准 + 评估 ----
    print("\n[3/4] 用这个人的验证集重新拟合仿射校准")
    rv = evaluate(model, dl_va, device, cfg)
    calib = None
    if "angles_true" in rv:
        a, b = fit_affine(rv["angles_true"], rv["angles_pred"])
        calib = {"scale": [round(float(v), 6) for v in a],
                 "bias": [round(float(v), 6) for v in b]}
    after, _ = report("微调后", model, calib)

    # ---- 导出 ----
    out_path = Path(args.out) if args.out else Path(
        OUTPUT_DIR) / f"{args.tag}_{base_path.stem.split('_')[-1]}.npz"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    new_meta = dict(meta)
    new_meta.update({"calibration": calib, "finetuned_subjects": subjects,
                     "finetuned_from": base_path.name})
    export_weights(model.state_dict(), out_path,
                   extra={"__config__": np.asarray(json.dumps(new_meta, ensure_ascii=False))})
    print(f"\n[4/4] 新权重已导出：{out_path}")
    print(f"  对比：微调前 {before} -> 微调后 {after}")
    summary = {"base": str(base_path), "out": str(out_path), "subjects": subjects,
               "epochs": args.epochs, "lr": args.lr, "before": before, "after": after,
               "calibration": calib}
    save_json(summary, out_path.with_suffix(".对比.json"))
    print("  把新 .npz 改名成插件里用的那份（例如 模型权重_连续关节角.npz），"
          "上位机重新导入插件即可。")


if __name__ == "__main__":
    main()

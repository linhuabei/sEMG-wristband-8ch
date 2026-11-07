# -*- coding: utf-8 -*-
"""把训练好的 best.pt 导出成纯 NumPy 权重（model_numpy.npz）并与 PyTorch 对数值校验。

用法：
    python export_numpy.py --ckpt ../outputs/ex1_gesture/best.pt
    python export_numpy.py --ckpt ../outputs/ex1_multitask/best.pt --copy-to-plugin angle
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np
import torch

from config import Config, OUTPUT_DIR, PLUGIN_DIR
from model import CNNTransformer
from numpy_infer import NumpyCNNTransformer, export_weights


def cfg_from_dict(d: dict) -> Config:
    cfg = Config()
    for name in ("data", "pre", "window", "model", "train"):
        obj = getattr(cfg, name)
        for k, v in d.get(name, {}).items():
            if hasattr(obj, k):
                setattr(obj, k, tuple(v) if isinstance(v, list) else v)
    return cfg


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", default=str(OUTPUT_DIR / "ex1_gesture" / "best.pt"))
    ap.add_argument("--out", default=None, help="输出 npz 路径（默认与 ckpt 同目录）")
    ap.add_argument("--copy-to-plugin", default=None,
                    choices=[None, "gesture", "angle"], help="额外复制到 plugins/ 供上位机使用")
    ap.add_argument("--tolerance", type=float, default=2e-4, help="与 torch 的最大允许误差")
    args = ap.parse_args()

    ckpt_path = Path(args.ckpt)
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    cfg = cfg_from_dict(ckpt["config"])
    n_chan = len(cfg.data.emg_channels)
    model = CNNTransformer(cfg.model, in_channels=n_chan, seq_len=cfg.window.length)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()

    out_path = Path(args.out) if args.out else ckpt_path.with_name("model_numpy.npz")
    extra = {"__config__": np.asarray(json.dumps({
        "n_channels": n_chan, "seq_len": cfg.window.length,
        "d_model": cfg.model.d_model, "n_heads": cfg.model.n_heads,
        "n_layers": cfg.model.n_layers, "n_classes": cfg.model.n_classes,
        "n_angles": cfg.model.n_angles, "multi_task": bool(cfg.model.multi_task),
        "emg_channels": list(cfg.data.emg_channels),
        "angle_keep_1based": [c + 1 for c in cfg.data.angle_keep],
        "fs": cfg.pre.fs,
        # 连续输出用的逐通道仿射校准（在验证集上拟合，插件里直接乘加）
        "calibration": ckpt.get("calibration"),
    }, ensure_ascii=False))}
    export_weights(ckpt["state_dict"], out_path, extra=extra)

    # ---- 数值校验：同一批输入，torch 与 numpy 前向结果应当一致 ----
    rng = np.random.default_rng(0)
    x = rng.normal(0, 1, size=(4, n_chan, cfg.window.length)).astype(np.float32)
    with torch.no_grad():
        ref = model(torch.from_numpy(x))
        ref_logits = ref["logits"].numpy()
        ref_angles = ref["angles"].numpy() if cfg.model.multi_task else None
    np_model = NumpyCNNTransformer(out_path)
    got = np_model.forward(x.astype(np.float64))
    diff_logits = float(np.abs(got["logits"] - ref_logits).max())
    print(f"导出: {out_path}")
    print(f"参数张量 {len(ckpt['state_dict'])} 个；logits 最大绝对误差 {diff_logits:.3e}")
    if ref_angles is not None:
        diff_angles = float(np.abs(got["angles"] - ref_angles).max())
        print(f"角度头最大绝对误差 {diff_angles:.3e}")
        if diff_angles > args.tolerance:
            raise SystemExit("与 PyTorch 数值不一致（角度头），请检查 numpy_infer.py")
    if diff_logits > args.tolerance:
        raise SystemExit("与 PyTorch 数值不一致（分类头），请检查 numpy_infer.py")

    if args.copy_to_plugin:
        # 插件目录里的中文文件名（插件按这个名字去找权重）
        name = {"gesture": "模型权重_手势.npz",
                "angle": "模型权重_连续关节角.npz"}[args.copy_to_plugin]
        PLUGIN_DIR.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(out_path, PLUGIN_DIR / name)
        engine_src = Path(__file__).resolve().with_name("numpy_infer.py")
        shutil.copyfile(engine_src, PLUGIN_DIR / "推理引擎_纯NumPy.py")
        print(f"已复制到插件目录: {PLUGIN_DIR / name}")
        print(f"           与引擎 {PLUGIN_DIR / '推理引擎_纯NumPy.py'}")


if __name__ == "__main__":
    main()

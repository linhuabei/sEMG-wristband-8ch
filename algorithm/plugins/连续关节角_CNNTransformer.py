# -*- coding: utf-8 -*-
"""上位机算法插件：CNN-Transformer 连续关节角输出（中文命名版）。

用法：上位机 ->「手势交互」页 ->「导入算法 .py」-> 选本文件。
界面上会每 50 ms 刷新一次各关节角度（连续输出，不是离散手势）。

同目录下需要：
    模型权重_连续关节角.npz    多任务/回归模型的权重
    推理引擎_纯NumPy.py        纯 numpy 前向实现

说明：预训练用的是 Cyberglove 未标定原始值，所以这里把输出线性映射到 0~90° 显示；
换成你们自己标定过的数据（自采数据集）训练后，输出就是真实角度。

上位机接口约定（不能改的英文名）：NAME / TYPE / predict(ctx)
"""
import importlib.util
import math
from collections import deque
from pathlib import Path

import numpy as np

NAME = "CNN-Transformer 连续关节角（中文命名版）"
TYPE = "angles"

标准化历史块数 = 600        # 30 s，与训练脚本 --norm-window 30 一致
预热块数 = 34               # 前 1.7 s 不输出
平滑系数 = 0.5              # 角度输出的指数平滑
角度下限 = 0.0              # 标准化值 -3σ 映射到 0°
角度上限 = 90.0             # 标准化值 +3σ 映射到 90°
截断倍数 = 3.0              # 标准化值截断在 ±3σ

_本目录 = Path(__file__).resolve().parent
_权重文件 = _本目录 / "模型权重_连续关节角.npz"
_引擎文件 = _本目录 / "推理引擎_纯NumPy.py"

_模型 = None
_引擎 = None
_错误 = ""
_数据块 = deque(maxlen=标准化历史块数)
_上次角度 = None
_窗口点数 = 200
_需要块数 = 2


def _加载模型():
    global _模型, _引擎, _错误, _窗口点数, _需要块数
    if _模型 is not None:
        return True
    if not _权重文件.exists() or not _引擎文件.exists():
        _错误 = "缺少权重或推理引擎文件"
        return False
    try:
        spec = importlib.util.spec_from_file_location("推理引擎", str(_引擎文件))
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _引擎 = mod
        _模型 = mod.NumpyCNNTransformer(str(_权重文件))
        _窗口点数 = int(_模型.cfg.get("seq_len", 200))
        _需要块数 = max(1, int(math.ceil(_窗口点数 / 100.0)))
        return True
    except Exception as exc:                       # noqa: BLE001
        _错误 = f"加载失败: {exc}"
        return False


def reset():
    """换人/重新采集时清空缓存。"""
    _数据块.clear()
    global _上次角度
    _上次角度 = None


def _重采样(序列, 目标长度):
    t = 序列.shape[0]
    if t == 目标长度:
        return 序列
    idx = np.linspace(0.0, t - 1.0, 目标长度)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, t - 1)
    w = (idx - i0)[:, None]
    return 序列[i0] * (1.0 - w) + 序列[i1] * w


def predict(ctx):
    """返回 21 路关节角（度）。数据不够时返回长度为 1 的占位数组。"""
    f = ctx.get("features", {}) if isinstance(ctx, dict) else {}
    samples = np.asarray(f.get("samples", []), dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 4:
        return np.zeros(1)
    if not _加载模型():
        return np.zeros(1)

    global _上次角度
    _数据块.append(samples)
    if len(_数据块) < max(预热块数, _需要块数):
        return np.zeros(1)

    hist = np.concatenate(_数据块, axis=0)
    mu = hist.mean(axis=0)
    sd = hist.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)

    need = _需要块数 * samples.shape[0]
    recent = np.concatenate(list(_数据块), axis=0)
    if recent.shape[0] < need:
        return np.zeros(1)
    win = _重采样(recent[-need:], _窗口点数)
    x = ((win - mu) / sd).T[None, :, :].astype(np.float64)

    out = _模型.forward(x)
    if "angles" not in out:
        return np.zeros(1)
    z = np.asarray(out["angles"][0], dtype=np.float64)
    校准 = _模型.cfg.get("calibration")
    if 校准:
        # 训练时在验证集上拟合的逐通道仿射校准（修正幅度偏小/通道偏置）
        z = z * np.asarray(校准["scale"], dtype=np.float64) \
            + np.asarray(校准["bias"], dtype=np.float64)
    z = np.clip(z, -截断倍数, 截断倍数)
    度 = (z + 截断倍数) / (2 * 截断倍数) * (角度上限 - 角度下限) + 角度下限
    _上次角度 = 度 if _上次角度 is None else (平滑系数 * _上次角度 + (1.0 - 平滑系数) * 度)
    return _上次角度

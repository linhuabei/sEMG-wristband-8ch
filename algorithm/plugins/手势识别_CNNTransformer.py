# -*- coding: utf-8 -*-
"""上位机算法插件：CNN-Transformer 手势识别"""
import importlib.util
import math
from collections import deque
from pathlib import Path

import numpy as np

NAME = "CNN-Transformer 手势识别（中文命名版）"
TYPE = "gesture"

# ---------------- 可调参数 ----------------
标准化历史块数 = 600      # 标准化统计量用最近多少个 50 ms 块（600 块 = 30 s），
                          # 必须和训练脚本 --norm-window 30 一致
预热块数 = 34             # 前 1.7 s 先不输出（训练时这段没有有效标签）
平滑系数 = 0.5            # 后验概率的指数平滑（越大越稳，越小越跟手）
最低置信度 = 0.40         # 低于它就报“无”

# NinaPro 动作号 -> 上位机 6 种手势
手势映射 = {1: 6, 2: 5, 3: 4, 4: 1, 5: 2, 6: 3}
手势名称 = {0: "休息", 1: "握拳", 2: "十指张开", 3: "OK手势",
            4: "比大拇指", 5: "比耶", 6: "捏合"}

_本目录 = Path(__file__).resolve().parent
_权重文件 = _本目录 / "模型权重_手势.npz"
_引擎文件 = _本目录 / "推理引擎_纯NumPy.py"

_模型 = None
_引擎 = None
_错误 = ""
_数据块 = deque(maxlen=标准化历史块数)
_后验 = None
最近一次后验 = None          # 供调试/二次开发查看 18 类概率
_窗口点数 = 200              # 模型输入长度（从权重里读，默认 200 点 = 100 ms）
_需要块数 = 2                # 需要攒几个 50 ms 块


def _加载模型():
    global _模型, _引擎, _错误, _窗口点数, _需要块数
    if _模型 is not None:
        return True
    if not _权重文件.exists():
        _错误 = f"缺少权重文件 {_权重文件.name}"
        return False
    if not _引擎文件.exists():
        _错误 = f"缺少推理引擎 {_引擎文件.name}"
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
    """清空缓存（换人/重新开始采集时调用，或重新导入本插件）。"""
    _数据块.clear()
    global _后验, 最近一次后验
    _后验 = None
    最近一次后验 = None


def _重采样(序列, 目标长度):
    """把 (T, C) 沿时间轴线性插值到目标长度。"""
    t = 序列.shape[0]
    if t == 目标长度:
        return 序列
    idx = np.linspace(0.0, t - 1.0, 目标长度)
    i0 = np.floor(idx).astype(np.int64)
    i1 = np.minimum(i0 + 1, t - 1)
    w = (idx - i0)[:, None]
    return 序列[i0] * (1.0 - w) + 序列[i1] * w


def predict(ctx):
    """上位机每个特征节拍（默认 50 ms）调用一次。"""
    f = ctx.get("features", {}) if isinstance(ctx, dict) else {}
    samples = np.asarray(f.get("samples", []), dtype=np.float64)
    if samples.ndim != 2 or samples.shape[0] < 4:
        return "无", 0.0
    if not _加载模型():
        return f"模型未就绪({_错误})", 0.0

    global _后验, 最近一次后验
    _数据块.append(samples)
    if len(_数据块) < max(预热块数, _需要块数):
        return "无", 0.0

    # 逐通道标准化：统计量取最近 30 s（和训练时一致）
    hist = np.concatenate(_数据块, axis=0)
    mu = hist.mean(axis=0)
    sd = hist.std(axis=0)
    sd = np.where(sd < 1e-9, 1.0, sd)

    need = _需要块数 * samples.shape[0]
    recent = np.concatenate(list(_数据块), axis=0)
    if recent.shape[0] < need:
        return "无", 0.0
    win = _重采样(recent[-need:], _窗口点数)
    x = ((win - mu) / sd).T[None, :, :].astype(np.float64)

    logits = _模型.forward(x)["logits"][0]
    p = np.asarray(_引擎.softmax(logits)).ravel()
    _后验 = p if _后验 is None else (平滑系数 * _后验 + (1.0 - 平滑系数) * p)
    最近一次后验 = _后验

    # 两种模型都能用：7 类模型直接取最大；18 类模型先按语义映射到 6 种手势
    if _后验.size <= 7:
        code = int(np.argmax(_后验))
        conf = float(_后验[code])
        other = 0.0
    else:
        app = {0: float(_后验[0])}
        for c, mov in 手势映射.items():
            if mov < _后验.size:
                app[c] = float(_后验[mov])
        other = max(0.0, 1.0 - sum(app.values()))
        code = max(app, key=app.get)
        conf = app[code]
    if code == 0 or conf < 最低置信度 or other > conf:
        return "无", float(conf)
    return 手势名称.get(code, "无"), float(min(0.99, conf))

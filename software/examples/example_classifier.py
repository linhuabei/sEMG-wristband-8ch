"""
示例：手势分类算法（RMS 阈值，不用训练也能跑）
TYPE = "gesture" 表示输出 (手势名, 置信度)

predict(ctx) 的 ctx 里有什么：
  ctx["features"]["samples"]  -> (窗口长度, 8) 的滤波后信号
  ctx["features"]["rms"]      -> 8 通道 RMS
  ctx["features"]["mav"]      -> 8 通道平均绝对值
  ctx["features"]["zc"]       -> 8 通道过零次数
  ctx["features"]["wl"]       -> 8 通道波形长度
"""
import numpy as np

NAME = "RMS 阈值示例（内置风格）"
TYPE = "gesture"  # gesture 或 angles


def predict(ctx):
    f = ctx["features"]
    rms = np.asarray(f["rms"], dtype=np.float64)
    peak = float(np.max(np.abs(rms))) if rms.size else 0.0

    if peak > 0.30:
        return "握拳", min(0.99, 0.55 + peak * 0.8)
    if peak > 0.15:
        return "张开", min(0.99, 0.55 + peak * 0.8)
    return "无", 0.5

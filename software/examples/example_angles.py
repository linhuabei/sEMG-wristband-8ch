"""
示例：连续角度回归算法（骨架）
TYPE = "angles" 表示输出关节角度数组，单位用度。
这里只是用 RMS 简单映射 6 个“关节”给你演示格式，
换成你的真实模型后，把 predict 里的内容替换掉即可。
"""
import numpy as np

NAME = "角度回归示例（骨架）"
TYPE = "angles"

JOINT_NAMES = ["肩", "肘", "腕", "指1", "指2", "指3"]


def predict(ctx):
    f = ctx["features"]
    rms = np.asarray(f["rms"], dtype=np.float64)
    act = float(np.max(rms)) if rms.size else 0.0
    # 占位：把活动强度映射成 0~90 度的“伪角度”
    return np.clip(act * 260.0, 0.0, 90.0) * np.ones(len(JOINT_NAMES))

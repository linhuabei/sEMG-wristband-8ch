# -*- coding: utf-8 -*-
"""信号预处理：带通 + 工频陷波 + 去均值 + 每通道标准化。

和上位机 core/signal_processor.py 里的流式滤波器（20-450 Hz 带通、50 Hz 陷波、
直流阻断）保持一致；离线这里用零相位 filtfilt，避免滤波引入相位偏移影响打标签。
"""
import numpy as np
from scipy import signal


def design_filters(fs, band=(20.0, 450.0), notch=50.0, q=30.0):
    """返回 (带通 SOS, 陷波 SOS)。"""
    nyq = fs / 2.0
    low, high = band[0] / nyq, min(0.98, band[1] / nyq)
    if low < high:
        sos_bp = signal.butter(4, [low, high], btype="band", output="sos")
    else:
        sos_bp = signal.butter(2, low, btype="high", output="sos")

    if 0 < notch < nyq:
        b, a = signal.iirnotch(notch / nyq, q)
        sos_notch = signal.tf2sos(b, a)
    else:
        sos_notch = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
    return sos_bp, sos_notch


def apply_filters(sig, fs, band=(20.0, 450.0), notch=50.0, q=30.0, detrend=True):
    """sig: (T, C) -> (T, C) float32。"""
    sig = np.asarray(sig, dtype=np.float64)
    sos_bp, sos_notch = design_filters(fs, band, notch, q)
    y = signal.sosfiltfilt(sos_bp, sig, axis=0)
    y = signal.sosfiltfilt(sos_notch, y, axis=0)
    if detrend:
        y = y - y.mean(axis=0, keepdims=True)
    return y.astype(np.float32)


class ChannelScaler:
    """每通道 z-score。统计量只在训练段上估计，避免测试信息泄漏。"""

    def __init__(self):
        self.mean = None
        self.std = None

    def fit(self, sig):
        sig = np.asarray(sig, dtype=np.float64)
        self.mean = sig.mean(axis=0)
        self.std = sig.std(axis=0)
        self.std[self.std < 1e-12] = 1.0
        return self

    def transform(self, sig):
        sig = np.asarray(sig, dtype=np.float64)
        return ((sig - self.mean) / self.std).astype(np.float32)

    def fit_transform(self, sig):
        self.fit(sig)
        return self.transform(sig)


def add_gaussian_noise(x, snr_db, rng):
    """按给定信噪比（dB）加高斯白噪声。x: (C, T) 或 (T, C)。"""
    power = float(np.mean(x ** 2))
    if power <= 0:
        return x
    noise_power = power / (10.0 ** (snr_db / 10.0))
    noise = rng.normal(0.0, np.sqrt(noise_power), size=x.shape).astype(x.dtype)
    return x + noise


# ======================================================================
# 部署一致性：完全复刻上位机 core/signal_processor.py 的流式链路
# ======================================================================
class StreamingFilter:
    """与上位机 signal_processor 完全一致的因果滤波（带通 + 陷波 + 直流阻断）。

    上位机是逐采样点处理的，这里用同样的初值（sosfilt_zi 不缩放，和上位机一样）
    分块处理，结果逐点等价但快得多。
    """

    def __init__(self, fs, band=(20.0, 450.0), notch=50.0, q=30.0, channels=8,
                 clip=8.0):
        self.fs = fs
        self.channels = channels
        self.clip = clip
        nyq = fs / 2.0
        low, high = band[0] / nyq, min(0.98, band[1] / nyq)
        if low < high:
            self.sos_bp = signal.butter(4, [low, high], btype="band", output="sos")
        else:
            self.sos_bp = signal.butter(2, low, btype="high", output="sos")
        if 0 < notch < nyq:
            b, a = signal.iirnotch(notch / nyq, q)
            self.sos_notch = signal.tf2sos(b, a)
        else:
            self.sos_notch = np.array([[1.0, 0.0, 0.0, 1.0, 0.0, 0.0]])
        self.reset()

    def reset(self):
        c = self.channels
        self.zi_bp = np.tile(signal.sosfilt_zi(self.sos_bp)[..., np.newaxis], (1, 1, c))
        self.zi_notch = np.tile(signal.sosfilt_zi(self.sos_notch)[..., np.newaxis], (1, 1, c))
        self.zi_dc = np.zeros((1, c))

    def process(self, x):
        """x: (n, C) -> 滤波后 (n, C)，状态在内部保留，可以连续分块调用。"""
        x = np.asarray(x, dtype=np.float64)
        one = x.ndim == 1
        if one:
            x = x.reshape(1, -1)
        y, self.zi_bp = signal.sosfilt(self.sos_bp, x, axis=0, zi=self.zi_bp)
        y, self.zi_notch = signal.sosfilt(self.sos_notch, y, axis=0, zi=self.zi_notch)
        y, self.zi_dc = signal.lfilter([1.0, -1.0], [1.0, -0.995], y, axis=0,
                                       zi=self.zi_dc)
        if self.clip:
            y = np.clip(y, -self.clip, self.clip)
        return y[0] if one else y


def causal_stream_filter(sig, fs, band=(20.0, 450.0), notch=50.0, q=30.0, clip=8.0):
    """整段信号按上位机方式因果滤波（等价于逐点流式处理）。"""
    return StreamingFilter(fs, band, notch, q, sig.shape[1], clip).process(sig)


def rolling_zscore(sig, win_samples):
    """因果滚动 z-score：每个时刻只用“当前及之前 win_samples 个点”的均值方差。

    这和上位机插件运行时做的标准化一致（插件用最近 6 s 的滑窗统计量），
    因此在训练里用它，模型看到的输入分布和实际部署时完全一样。
    """
    sig = np.asarray(sig, dtype=np.float64)
    n, c = sig.shape
    zeros = np.zeros((1, c))
    cs = np.cumsum(np.vstack([zeros, sig]), axis=0)
    cs2 = np.cumsum(np.vstack([zeros, sig ** 2]), axis=0)
    idx = np.arange(n) + 1
    lo = np.maximum(0, idx - int(win_samples))
    cnt = (idx - lo).reshape(-1, 1)
    mean = (cs[idx] - cs[lo]) / cnt
    var = (cs2[idx] - cs2[lo]) / cnt - mean ** 2
    var = np.maximum(var, 0.0)
    sd = np.sqrt(var)
    sd = np.where(sd < 1e-12, 1.0, sd)
    return ((sig - mean) / sd).astype(np.float32)

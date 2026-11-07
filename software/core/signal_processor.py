"""
sEMG 信号处理模块

输入：设备管理器按采样率逐点推送的 8 通道数据（形状 (8,) 或 (N, 8)）。
内部用带记忆的流式滤波器（带通 + 工频陷波 + DC 阻断），
并维护约 50ms 的滑窗来提取 RMS/MAV/ZC/WL 特征。
"""

import numpy as np
from scipy import signal
from PyQt5.QtCore import QObject, pyqtSignal


class SignalProcessor(QObject):
    """实时信号处理器"""

    # 信号定义
    filtered_data_ready = pyqtSignal(np.ndarray)  # 8通道滤波数据
    features_ready = pyqtSignal(dict)  # 特征字典

    def __init__(self, channels=8, sampling_rate=1000):
        super().__init__()
        self.channels = channels
        self.sampling_rate = sampling_rate

        # 滤波开关（由右侧「硬件控制」栏里的复选框控制，默认全开）
        self.bandpass_enabled = True
        self.notch_enabled = True

        self._design_filters()

        # 流式滤波器状态（按通道保存，供下一个采样点续用）
        zi_shape = self.sos_bp.shape[0]
        self._zi_bp = np.tile(
            signal.sosfilt_zi(self.sos_bp)[..., np.newaxis], (1, 1, channels)
        )
        zi_notch = signal.sosfilt_zi(self.sos_notch)
        self._zi_notch = np.tile(zi_notch[..., np.newaxis], (1, 1, channels))
        self._zi_dc = np.zeros((1, channels))  # 一阶 DC 阻断器状态

        # 特征滑窗（约 50ms）
        self.feature_window = max(8, sampling_rate // 20)
        self._window = np.zeros((self.feature_window, channels))
        self._windex = 0
        self._feature_tick = 0

    # ==================== 滤波器设计 ====================

    def _design_filters(self):
        """设计带通 (20-450 Hz)、50Hz 陷波，均输出 SOS 以保证数值稳定"""
        nyquist = self.sampling_rate / 2
        low = 20 / nyquist
        high = min(0.98, 450 / nyquist)
        if low < high:
            self.sos_bp = signal.butter(
                4, [low, high], btype='band', output='sos'
            )
        else:
            # 采样率过低时退化为高通保护
            self.sos_bp = signal.butter(2, low, btype='high', output='sos')

        notch_freq = 50 / nyquist
        if 0 < notch_freq < 0.5:
            b_notch, a_notch = signal.iirnotch(notch_freq, 30)
            self.sos_notch = signal.tf2sos(b_notch, a_notch)
        else:
            self.sos_notch = signal.butter(1, 0.99, btype='low', output='sos')

    # ==================== 处理 ====================

    def process_data(self, raw_data):
        """
        处理一帧(或一小块) sEMG 数据
        Args:
            raw_data: numpy array, shape (channels,) 或 (N, channels)
        """
        arr = np.asarray(raw_data, dtype=np.float64)
        if arr.ndim == 1:
            if arr.size != self.channels:
                raise ValueError(f"期望 {self.channels} 通道, 收到 {arr.size}")
            x = arr.reshape(1, -1)
        elif arr.ndim == 2:
            x = arr
        else:
            raise ValueError("数据维度必须是 1D 或 2D")
        if x.shape[1] < self.channels:
            raise ValueError(f"期望至少 {self.channels} 通道, 收到 {x.shape[1]}")
        x = x[:, :self.channels]

        # 1. 带通（可关：关了就是原始码值，只保留后面的直流阻断）
        if self.bandpass_enabled:
            y, self._zi_bp = signal.sosfilt(
                self.sos_bp, x, axis=0, zi=self._zi_bp
            )
        else:
            y = x
        # 2. 带阻 / 50Hz 工频陷波（可关）
        if self.notch_enabled:
            y, self._zi_notch = signal.sosfilt(
                self.sos_notch, y, axis=0, zi=self._zi_notch
            )
        # 3. 直流阻断: y[n] = x[n] - x[n-1] + 0.995*y[n-1]
        #    （这一级始终开启：它负责把基线压到 0，关掉波形会跑出屏幕）
        y, self._zi_dc = signal.lfilter(
            [1.0, -1.0], [1.0, -0.995], y, axis=0, zi=self._zi_dc
        )
        y = np.clip(y, -8.0, 8.0)

        filtered = y.astype(np.float32)
        if filtered.shape[0] == 1:
            filtered = filtered[0]

        # 送入监视/训练视图
        self.filtered_data_ready.emit(filtered)

        # 4. 滑窗特征：每个采样点都入窗，但约 50ms 才算一次特征
        sample = filtered if filtered.ndim == 1 else filtered[-1]
        self._window[self._windex] = sample
        self._windex = (self._windex + 1) % self.feature_window
        self._feature_tick += 1
        if self._feature_tick >= max(1, self.sampling_rate // 20):
            self._feature_tick = 0
            features = self._extract_features()
            self.features_ready.emit(features)

    def _extract_features(self):
        """从最近约 50ms 的滑窗提取每通道特征"""
        data = np.roll(self._window, -self._windex, axis=0).astype(np.float64)
        features = {}
        features['rms'] = np.sqrt(np.mean(data ** 2, axis=0))
        features['mav'] = np.mean(np.abs(data), axis=0)
        features['zc'] = np.sum(
            np.diff(np.signbit(data).astype(np.int8), axis=0) != 0, axis=0
        )
        features['wl'] = np.sum(np.abs(np.diff(data, axis=0)), axis=0)
        # 把最近窗口的滤波样本也带上，外部算法可能想要原始序列
        features['samples'] = data.astype(np.float32)
        return features

    def reset(self):
        """清空滤波器状态（重连 / 切换设备时调用）"""
        zi_shape = self.sos_bp.shape[0]
        self._zi_bp = np.tile(
            signal.sosfilt_zi(self.sos_bp)[..., np.newaxis], (1, 1, self.channels)
        )
        self._zi_notch = np.tile(
            signal.sosfilt_zi(self.sos_notch)[..., np.newaxis], (1, 1, self.channels)
        )
        self._zi_dc = np.zeros((1, self.channels))
        self._window[:] = 0.0
        self._windex = 0

    # ==================== 滤波开关 ====================

    def set_bandpass(self, enabled):
        """开关带通滤波器；重新打开时清一次状态，避免残留状态造成跳变"""
        enabled = bool(enabled)
        if enabled == self.bandpass_enabled:
            return
        self.bandpass_enabled = enabled
        if enabled:
            self._zi_bp = np.zeros_like(self._zi_bp)

    def set_notch(self, enabled):
        """开关 50Hz 带阻（陷波）滤波器"""
        enabled = bool(enabled)
        if enabled == self.notch_enabled:
            return
        self.notch_enabled = enabled
        if enabled:
            self._zi_notch = np.zeros_like(self._zi_notch)

    def filter_state(self):
        """返回 (带通开关, 带阻开关)"""
        return self.bandpass_enabled, self.notch_enabled

    def configure_sampling_rate(self, rate: int):
        """
        采样率变更（固件 EMG_CFG 上报 1000/2000Hz 时调用）。
        重新设计滤波器并清零状态，避免旧系数/旧状态残留。
        """
        rate = int(rate)
        if rate <= 0:
            return
        if rate == self.sampling_rate:
            return
        self.sampling_rate = rate
        self._design_filters()
        self.feature_window = max(8, rate // 20)
        self._window = np.zeros((self.feature_window, self.channels))
        self._windex = 0
        self._feature_tick = 0
        self.reset()

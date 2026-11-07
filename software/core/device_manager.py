"""
设备通信管理模块 - BLE版本（仅蓝牙）

数据来源：连接 ESP32 后固件自动开始 ADS1298 流式采集，
二进制帧在 ESP32Controller 中解析后经 emg_data_received 送达。

（注：早期版本内置过"测试模式"模拟波形，现已删除——数据必须来自真实硬件。）
"""
import asyncio
import re
import threading
import time

import numpy as np
from PyQt5.QtCore import QObject, QTimer, pyqtSignal

from core.emg_protocol import normalize_raw
from core.esp32_controller import ESP32Controller, ConnectionState

# 固件回的一行：IMU:AX=0.012,AY=-0.003,AZ=0.998,GX=0.12,GY=-0.34,GZ=0.05
_IMU_RE = re.compile(
    r"IMU:AX=(-?[\d.]+),AY=(-?[\d.]+),AZ=(-?[\d.]+),"
    r"GX=(-?[\d.]+),GY=(-?[\d.]+),GZ=(-?[\d.]+)"
)


class DeviceManager(QObject):
    """设备管理器 - 仅BLE"""

    connection_changed = pyqtSignal(bool)
    data_received = pyqtSignal(np.ndarray)
    devices_found = pyqtSignal(list)
    error_occurred = pyqtSignal(str)
    pin_status_updated = pyqtSignal(dict)
    esp_response = pyqtSignal(str)
    emg_config_changed = pyqtSignal(int, int, int)  # sps, bits, channels
    imu_sample = pyqtSignal(dict)      # ax/ay/az (g) + gx/gy/gz (dps) + t
    imu_unavailable = pyqtSignal(str)  # IMU 读不到时的原因

    def __init__(self):
        super().__init__()
        self.connected = False
        self.connection_type = None

        self.esp32 = ESP32Controller()
        self.esp32.connection_state_changed.connect(self._on_ble_state_changed)
        self.esp32.error_occurred.connect(self.error_occurred.emit)
        self.esp32.response_received.connect(self._on_text_response)
        self.esp32.emg_data_received.connect(self._on_emg_frame)
        self.esp32.emg_config_changed.connect(self.emg_config_changed)

        self.loop = None
        self._thread = None

        # 最新 EMG 数据缓冲区（归一化 float32）
        self._latest_emg_data = None

        # IMU 轮询（20 Hz，够画姿态了；读不到会自动停）
        self._imu_polling = False
        self._imu_poll_pending = False
        self._imu_timer = QTimer(self)
        self._imu_timer.setInterval(50)
        self._imu_timer.timeout.connect(self._poll_imu)

    # ==================== 扫描 ====================

    def scan_devices(self):
        """扫描 BLE 设备"""
        def scan_task():
            try:
                loop = asyncio.new_event_loop()
                ble_devices = loop.run_until_complete(self.esp32.scan_devices())
                self.devices_found.emit(ble_devices)
                loop.close()
            except Exception as e:
                self.error_occurred.emit(f"扫描失败: {e}")

        threading.Thread(target=scan_task, daemon=True).start()

    # ==================== 连接 ====================

    def connect_device(self, device_info):
        """连接 BLE 设备，成功后自动开启 ADS1298 流式采集。

        事件循环只建一次并常驻：断线后再点"连接"就是在这个循环里重连，
        避免反复新建线程/循环（旧循环里的客户端断开会把新连接的状态带偏）。
        """
        if self.loop is not None and self._thread is not None and self._thread.is_alive():
            asyncio.run_coroutine_threadsafe(
                self._connect_and_stream(device_info), self.loop
            )
            return

        self._thread = threading.Thread(
            target=self._run_loop, args=(device_info,), daemon=True
        )
        self._thread.start()

    def _run_loop(self, device_info):
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        self.loop = loop
        try:
            loop.run_until_complete(self._connect_and_stream(device_info))
            loop.run_forever()
        except Exception as e:
            self.error_occurred.emit(f"连接失败: {e}")
        finally:
            if self.loop is loop:
                self.loop = None

    async def _connect_and_stream(self, device_info):
        """连上设备并自动开始采集（首次连接和断线重连都走这里）"""
        try:
            success = await self.esp32.connect(device_info['address'])
            if not success:
                self.error_occurred.emit("连接失败，请确认设备已开机且处于有效范围内")
                return
            self.connected = True
            self.connection_type = 'ble'
            self.connection_changed.emit(True)
            # 自动开启采集（不阻塞界面）
            asyncio.create_task(self._auto_start_emg())
        except Exception as e:
            self.error_occurred.emit(f"连接失败: {e}")

    async def _auto_start_emg(self):
        """连接后自动开启 ADS1298 采集；老固件不支持时给出提示但不掉线"""
        try:
            await asyncio.sleep(0.3)
            response = await self.esp32.start_emg()
            self.esp_response.emit(response)
        except asyncio.CancelledError:
            pass
        except Exception as e:
            self.error_occurred.emit(f"自动开启采集失败: {e}")

    # ==================== 真实数据 ====================

    def _on_emg_frame(self, raw_frame):
        """BLE 通知解析出的原始帧 -> 归一化采样点，交给信号处理"""
        data = normalize_raw(raw_frame)
        self._latest_emg_data = data
        self.data_received.emit(data)

    # ==================== IMU ====================

    def set_imu_polling(self, enabled):
        """开关 IMU 周期读取（只在已连接时真正发命令）"""
        self._imu_polling = bool(enabled)
        if self._imu_polling and self.connected:
            self._imu_timer.start()
        else:
            self._imu_timer.stop()

    def _poll_imu(self):
        if not self.connected:
            self._imu_timer.stop()
            return
        self._imu_poll_pending = True
        self.send_esp_command("IMU")

    def _on_text_response(self, text):
        """文本应答统一入口：IMU 行先解析成数据，再决定要不要写命令台"""
        if text.startswith("IMU:"):
            match = _IMU_RE.search(text)
            if match:
                vals = [float(v) for v in match.groups()]
                self.imu_sample.emit({
                    "ax": vals[0], "ay": vals[1], "az": vals[2],
                    "gx": vals[3], "gy": vals[4], "gz": vals[5],
                    "t": time.time(),
                })
            if self._imu_poll_pending:
                self._imu_poll_pending = False
                return          # 轮询回来的不往命令台刷，否则日志会被刷爆
        elif text.startswith("ERROR: IMU"):
            self._imu_poll_pending = False
            self.set_imu_polling(False)
            self.imu_unavailable.emit(text)
        self.esp_response.emit(text)

    # ==================== 断开 ====================

    def disconnect_device(self):
        """停止采集并断开连接"""
        if self.connected and self.loop:
            async def _stop_and_close():
                try:
                    await asyncio.wait_for(self.esp32.stop_emg(), timeout=0.8)
                except Exception:
                    pass
                try:
                    await self.esp32.disconnect()
                except Exception:
                    pass
            try:
                asyncio.run_coroutine_threadsafe(_stop_and_close(), self.loop)
            except Exception:
                pass

        self.connected = False
        self.connection_changed.emit(False)

    # ==================== ESP32 命令接口 ====================

    def send_esp_command(self, cmd: str):
        """发送 ESP32 命令（设备详细页控制台）"""
        if self.loop:
            async def _send():
                try:
                    response = await self.esp32.send_command(cmd)
                    self.esp_response.emit(response)
                except Exception as e:
                    self.error_occurred.emit(str(e))
                    # 命令发不出去说明链路已经不在了，顺手把界面状态纠正过来
                    if not self.esp32.is_connected:
                        self._mark_disconnected()
            asyncio.run_coroutine_threadsafe(_send(), self.loop)
        else:
            self.error_occurred.emit(
                f"设备未连接，命令未发送: {cmd}"
            )
            self._mark_disconnected()

    def _mark_disconnected(self):
        """把"已连接"状态纠正为断开（幂等）"""
        if not self.connected:
            return
        self.connected = False
        self.connection_changed.emit(False)

    def set_pin(self, pin: int, value: int):
        self.send_esp_command(f"PIN{pin}={value}")

    def read_pin(self, pin: int):
        self.send_esp_command(f"READ{pin}")

    def read_all_status(self):
        self.send_esp_command("ALL")

    def scan_i2c(self):
        self.send_esp_command("I2C_SCAN")

    def spi_test(self):
        self.send_esp_command("SPI_TEST")

    def start_emg(self):
        self.send_esp_command("EMG_START")

    def stop_emg(self):
        self.send_esp_command("EMG_STOP")

    def ads_init(self):
        self.send_esp_command("ADS_INIT")

    def read_imu(self):
        self.send_esp_command("IMU")

    def set_emg_sps(self, sps: int):
        self.send_esp_command(f"EMG_SPS={int(sps)}")

    def led_on(self):
        self.set_pin(8, 1)

    def led_off(self):
        self.set_pin(8, 0)

    # ==================== 状态 ====================

    def _on_ble_state_changed(self, state):
        """BLE 连接状态变化"""
        if state == ConnectionState.CONNECTED:
            self.connected = True
        elif state in (ConnectionState.DISCONNECTED, ConnectionState.ERROR):
            self.connected = False
            self.connection_changed.emit(False)

"""
ESP32-S3 BLE 控制器模块

负责扫描 / 连接 / 文本命令，以及把固件下发的二进制 EMG 流
(协议见 core/emg_protocol.py) 解析成 8 通道采样帧。
"""
import asyncio
import logging
import os
import subprocess

import numpy as np
from PyQt5.QtCore import QObject, pyqtSignal

from core.emg_protocol import (EMG_MAGIC, EmgStreamParser, parse_cfg_line,
                               BITS_24, BITS_16)

SERVICE_UUID = "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
CHARACTERISTIC_UUID = "beb5483e-36e1-4688-b7f5-ea07361b26a8"
DEVICE_NAME = "Node"

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger("ESP32")


class ESP32Error(Exception):
    pass


class ConnectionState:
    DISCONNECTED = 0
    CONNECTING = 1
    CONNECTED = 2
    ERROR = 3


PIN_CONFIG = {
    1:  {"name": "IMU SDA",      "type": "I2C",  "direction": "inout"},
    2:  {"name": "ADS1298 MISO", "type": "SPI",  "direction": "in"},
    3:  {"name": "IMU SCL",      "type": "I2C",  "direction": "inout"},
    4:  {"name": "I2C2 SCL",     "type": "I2C",  "direction": "inout"},
    5:  {"name": "I2C2 SDA",     "type": "I2C",  "direction": "inout"},
    6:  {"name": "ADS1298 SCLK", "type": "SPI",  "direction": "out"},
    7:  {"name": "ADS1298 MOSI", "type": "SPI",  "direction": "out"},
    8:  {"name": "状态灯 LED",     "type": "GPIO", "direction": "out"},
    9:  {"name": "ADS1298 RESET","type": "GPIO", "direction": "out"},
    10: {"name": "ADS1298 CS",   "type": "GPIO", "direction": "out"},
    11: {"name": "ADS1298 DRDY", "type": "GPIO", "direction": "in"},
    12: {"name": "ADS1298 START","type": "GPIO", "direction": "out"},
    13: {"name": "ADS1298 PWDN", "type": "GPIO", "direction": "out"},
}


class ESP32Controller(QObject):
    """ESP32-S3 BLE 控制器"""

    connection_state_changed = pyqtSignal(int)
    response_received = pyqtSignal(str)
    emg_data_received = pyqtSignal(np.ndarray)  # int32 原始码值 (8,)
    emg_config_changed = pyqtSignal(int, int, int)  # sps, bits, channels
    pin_status_changed = pyqtSignal(dict)
    error_occurred = pyqtSignal(str)

    def __init__(self):
        super().__init__()
        self.client = None
        self.state = ConnectionState.DISCONNECTED
        self.notification_queue = asyncio.Queue()
        self._response_event = asyncio.Event()
        self._last_response = ""
        # 串行化命令：IMU 轮询和界面上的命令会同时发，不排队会拿串应答
        self._cmd_lock = asyncio.Lock()

        # EMG 流配置（固件 EMG_CFG 上报后更新）
        self.emg_sps = 1000
        self.emg_bits = BITS_24
        self.emg_channels = 8
        self._emg_parser = EmgStreamParser(BITS_24)

    # ==================== 扫描 ====================

    async def scan_devices(self) -> list:
        """扫描 BLE 设备"""
        devices = self._scan_windows()
        if not devices:
            devices = await self._scan_bleak_fallback()
        return devices

    def _scan_windows(self) -> list:
        """Windows 原生 BLE 扫描（PowerShell）"""
        ps_script = r"""
$ErrorActionPreference = 'SilentlyContinue'

[Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementWatcher, Windows.Devices.Bluetooth.Advertisement, ContentType = WindowsRuntime] | Out-Null
[Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementReceivedEventArgs, Windows.Devices.Bluetooth.Advertisement, ContentType = WindowsRuntime] | Out-Null

$watcher = [Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementWatcher]::new()
$results = @()

$handler = [Windows.Devices.Bluetooth.Advertisement.TypedEventHandler[Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementWatcher, Windows.Devices.Bluetooth.Advertisement.BluetoothLEAdvertisementReceivedEventArgs]]{
    param($sender, $eventArgs)
    $addr = $eventArgs.BluetoothAddress.ToString("X12")
    $name = $eventArgs.Advertisement.LocalName
    if (-not $name) { $name = "" }
    $rssi = $eventArgs.RawSignalStrengthInDBm
    $results += "$addr|$name|$rssi"
}

$watcher.add_Received($handler)
$watcher.Start()
Start-Sleep -Seconds 6
$watcher.Stop()

$results | Sort-Object -Unique | ForEach-Object { Write-Output $_ }
"""
        try:
            # 打包成 GUI exe 后父进程自己是没有控制台的，这时直接起 powershell
            # Windows 会给它新建一个控制台窗口（就是每次扫描闪的那下黑框）。
            # CREATE_NO_WINDOW 明确要求不建窗口；输出已经走管道，不需要控制台。
            no_window = getattr(subprocess, "CREATE_NO_WINDOW", 0) if os.name == "nt" else 0
            result = subprocess.run(
                ["powershell", "-NoProfile", "-ExecutionPolicy", "Bypass", "-Command", ps_script],
                capture_output=True, text=True, timeout=25, encoding='utf-8', errors='ignore',
                creationflags=no_window,
            )

            devices = []
            for line in result.stdout.strip().split('\n'):
                line = line.strip()
                if '|' in line:
                    parts = line.split('|')
                    if len(parts) >= 3:
                        addr = parts[0]
                        name = parts[1].strip() or f"未知 ({addr[-6:]})"
                        devices.append({
                            'name': name,
                            'address': addr,
                            'type': 'ble',
                        })

            logger.info(f"Windows扫描: 发现 {len(devices)} 个设备")
            for d in devices:
                logger.info(f"  {d['name']} - {d['address']}")
            return devices

        except Exception as e:
            logger.error(f"Windows扫描失败: {e}")
            return []

    async def _scan_bleak_fallback(self) -> list:
        """Bleak 回退扫描"""
        devices = []
        try:
            from bleak import BleakScanner
            scanner = BleakScanner()
            await scanner.start()
            await asyncio.sleep(5)
            await scanner.stop()

            for d in scanner.discovered_devices:
                name = d.name or f"未知 ({d.address[-6:]})"
                devices.append({'name': name, 'address': d.address, 'type': 'ble'})
        except Exception as e:
            logger.error(f"Bleak扫描失败: {e}")
        return devices

    # ==================== 连接 ====================

    async def connect(self, address: str) -> bool:
        try:
            from bleak import BleakClient
            self.state = ConnectionState.CONNECTING
            self.connection_state_changed.emit(self.state)

            # 上一次的客户端（比如刚断开那条链路）先收干净
            if self.client is not None:
                try:
                    await self.client.disconnect()
                except Exception:
                    pass

            # disconnected_callback：硬件关机 / 走远 / 蓝牙被占用时 bleak 会回调，
            # 界面才能立刻把状态刷成"未连接"（以前一直是"已连接"，是假状态）
            self.client = BleakClient(
                address, disconnected_callback=self._on_ble_disconnected
            )
            await self.client.connect()
            await self.client.start_notify(CHARACTERISTIC_UUID, self._notification_handler)

            self.state = ConnectionState.CONNECTED
            self.connection_state_changed.emit(self.state)
            logger.info(f"已连接: {address}")
            try:
                mtu = self.client.mtu_size
                logger.info(f"协商 ATT MTU: {mtu}")
            except Exception:
                pass
            return True
        except Exception as e:
            self.state = ConnectionState.ERROR
            self.error_occurred.emit(f"连接失败: {e}")
            return False

    def _on_ble_disconnected(self, client):
        """bleak 断开回调：把控制器状态和界面同步成"未连接" """
        if self.state == ConnectionState.DISCONNECTED:
            return
        self.state = ConnectionState.DISCONNECTED
        self.connection_state_changed.emit(self.state)
        self.error_occurred.emit(
            "设备连接已断开（硬件关机 / 超出范围 / 蓝牙被其他程序占用）"
        )

    async def disconnect(self):
        try:
            if self.client:
                await self.client.disconnect()
            self.state = ConnectionState.DISCONNECTED
        except Exception:
            pass

    # ==================== 通知分发 ====================

    def _notification_handler(self, sender, data):
        """BLE 通知回调：区分文本响应与二进制 EMG 流"""
        payload = bytes(data)
        if not payload:
            return

        # 二进制 EMG 数据
        if len(payload) >= 2 and payload[:2] == EMG_MAGIC:
            frames = self._emg_parser.feed(payload)
            for frame in frames:
                self.emg_data_received.emit(frame)
            return

        # 文本响应
        text = payload.decode('utf-8', errors='replace').strip()
        cfg = parse_cfg_line(text)
        if cfg:
            sps, bits, channels = cfg
            self.emg_sps, self.emg_bits, self.emg_channels = cfg
            self._emg_parser.reset(bits)
            self.emg_config_changed.emit(sps, bits, channels)
            logger.info(f"EMG 流配置: {sps}Hz / {bits}bit / {channels}ch")
        self._last_response = text
        self.response_received.emit(text)
        self._response_event.set()

    async def send_command(self, cmd: str, timeout: float = 2.0) -> str:
        if not self.client or not self.client.is_connected:
            raise ESP32Error("设备未连接")
        async with self._cmd_lock:
            self._response_event.clear()
            await self.client.write_gatt_char(
                CHARACTERISTIC_UUID, (cmd + "\n").encode()
            )
            try:
                await asyncio.wait_for(self._response_event.wait(), timeout)
                return self._last_response
            except asyncio.TimeoutError:
                raise ESP32Error(f"命令超时: {cmd}")

    # ==================== 文本命令 ====================

    async def set_pin(self, pin: int, value: int) -> str:
        return await self.send_command(f"PIN{pin}={value}")

    async def read_pin(self, pin: int) -> int:
        response = await self.send_command(f"READ{pin}")
        return int(response.split('=')[1])

    async def read_all_status(self) -> dict:
        return self._parse_status(await self.send_command("ALL"))

    async def scan_i2c(self) -> list:
        return self._parse_i2c(await self.send_command("I2C_SCAN"))

    async def spi_test(self) -> str:
        return await self.send_command("SPI_TEST")

    async def start_emg(self) -> str:
        """开启 ADS1298 流式采集；固件先回 EMG_CFG 再持续 notify"""
        return await self.send_command("EMG_START", timeout=3.0)

    async def stop_emg(self) -> str:
        return await self.send_command("EMG_STOP")

    async def ads_init(self) -> str:
        """重新初始化 ADS1298（复位寄存器）"""
        return await self.send_command("ADS_INIT", timeout=3.0)

    def _parse_status(self, response: str) -> dict:
        status = {}
        if response.startswith("STATUS:"):
            for part in response[7:].split(','):
                if '=' in part:
                    k, v = part.split('=')
                    status[k] = int(v)
        return status

    def _parse_i2c(self, response: str) -> list:
        devices = []
        if response.startswith("I2C_DEVICES:"):
            for addr in response[12:].split(','):
                if addr.strip():
                    devices.append(int(addr.strip(), 16))
        return devices

    async def led_on(self):
        return await self.set_pin(8, 1)

    async def led_off(self):
        return await self.set_pin(8, 0)

    @property
    def is_connected(self) -> bool:
        return self.state == ConnectionState.CONNECTED

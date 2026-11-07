# Node 8CH sEMG 固件（ESP32-S3 + ADS1298）

这是手环固件的**分层重构版**，替代原来把所有逻辑塞进一个 `.ino` 的写法。

> **验证状态**：已在 PlatformIO + espressif32 6.10.0（Arduino core 3.2.0）下编译通过，
> RAM 占用 13.8%、Flash 占用 28.3%；帧打包的 PC 侧单元测试通过。
>采样时序、BLE 吞吐和 IMU/ADS 行为需要上板复测。

BLE 设备名、Service/Characteristic UUID、EMG 帧格式和文本命令都与原固件保持一致，上位机不需要改。

## 分层结构

| 层 | 目录 | 职责 |
| --- | --- | --- |
| 应用层 | `Node_8CH_ADS1298/src/app/` | 启动流程、采集状态机、MTU 决策、连接/断开处理 |
| 驱动层 | `Node_8CH_ADS1298/src/drivers/` | ADS1298（SPI/寄存器/采样）、LSM6DSR（I2C）、GP08 状态灯（LEDC） |
| 协议层 | `Node_8CH_ADS1298/src/protocol/` | EMG 帧打包、BLE 传输、文本命令解析 |
| 诊断层 | `Node_8CH_ADS1298/src/diagnostics/` | I2C 扫描、特殊引脚读取 |
| 入口 | `Node_8CH_ADS1298/Node_8CH_ADS1298.ino` | Arduino IDE 和 PlatformIO 共用，只调用 `app::begin()` / `app::loop()` |

```text
firmware/
├── Node_8CH_ADS1298/
│   ├── Node_8CH_ADS1298.ino      # 入口（薄）
│   └── src/
│       ├── app/
│       │   ├── app.h / app.cpp   # 应用状态机
│       │   └── config.h          # 引脚、寄存器、帧长、灯语常量
│       ├── drivers/
│       │   ├── ads1298.h / .cpp
│       │   ├── imu_lsm6dsr.h / .cpp
│       │   └── status_led.h / .cpp
│       ├── protocol/
│       │   ├── emg_frame.h / .cpp    # 不依赖 Arduino，可 PC 侧测试
│       │   ├── ble_transport.h / .cpp
│       │   └── command_parser.h / .cpp
│       └── diagnostics/
│           └── diagnostics.h / .cpp
├── platformio.ini                # PlatformIO 构建
├── test/
│   ├── test_emg_frame.cpp        # 帧打包 host test
│   └── README.md
└── legacy/
    ├── ctrl_legacy_fw_ESP32S3.ino
    ├── Node_8CH_ADS1298_single_file.ino
    └── README.md
```

## 构建

### PlatformIO

```bash
cd firmware
pio run
```

### Arduino IDE

打开 `firmware/Node_8CH_ADS1298/Node_8CH_ADS1298.ino`，开发板选 **ESP32S3 Dev Module**。
`Node_8CH_ADS1298/src/` 下的 `.cpp` 会自动参与编译；不要再把 `.ino` 单独复制到别的文件夹，否则 Arduino IDE 找不到同级的 `src/`。

## 引脚

| GPIO | 功能 |
| --- | --- |
| 1 | IMU SDA |
| 2 | ADS1298 MISO |
| 3 | IMU SCL |
| 4 | 扩展 I2C SCL |
| 5 | 扩展 I2C SDA |
| 6 | ADS1298 SCLK |
| 7 | ADS1298 MOSI |
| 8 | 状态灯（LEDC PWM） |
| 9 | ADS1298 RESET# |
| 10 | ADS1298 CS |
| 11 | ADS1298 DRDY# |
| 12 | ADS1298 START |
| 13 | ADS1298 PWDN# |

## BLE 与帧格式

- 设备名：`Node_1_by_hzx`
- Service：`4fafc201-1fb5-459e-8fcc-c5c9c331914b`
- Characteristic：`beb5483e-36e1-4688-b7f5-ea07361b26a8`

每个采样点一帧：

```text
[0xAA][0x55][seq][CH1..CH8][xor]
```

- `seq`：1 字节循环序号；
- 24 bit：每通道 3 字节大端有符号码值，帧长 28；
- 16 bit：每通道 2 字节，帧长 20（MTU 太小时自动降级）；
- `xor`：除校验字节外整帧异或。

采集时按协商 MTU 攒若干帧一起 notify（24bit 最多 8 帧，16bit 降级最多 12 帧）；上位机按字节流解析，丢包可重同步。

## 文本命令

`PIN<n>=<v>`、`READ<n>`、`ALL`、`EMG_START`、`EMG_STOP`、`EMG_SPS=1000/2000`、`ADS_INIT`、`ADS_DBG`、`ADS_GAIN=1/2/3/4/6/8/12`、`ADS_TEST=0/1`、`IMU`、`I2C_SCAN`、`SPI_TEST`、`SPI_BUS`、`HELP`。

## 测试

帧打包不依赖 Arduino，可以在 PC 上单独跑：

```bash
g++ -std=c++17 -I src test/test_emg_frame.cpp src/protocol/emg_frame.cpp -o test_emg_frame
./test_emg_frame
```

当前测试覆盖：24bit 帧长/帧头/序号/XOR、8 帧批处理、16bit 帧长/XOR。
驱动层（ADS1298 / IMU / LED）目前没有 host 测试，需要硬件或 mock。

## 与旧单文件版本的关系

- `legacy/ctrl_legacy_fw_ESP32S3.ino`：更早的 legacy 单文件，仅作对照；
- `legacy/Node_8CH_ADS1298_single_file.ino`：重构前的当前单文件版本，保留用于对比和回退；
- `Node_8CH_ADS1298/src/` 是本次重构后的版本，行为按原单文件移植，接口/协议保持兼容。

## 已知限制

- 没有在真实手环上复测，采样时序、DRDY 轮询、BLE 吞吐、IMU 总线识别都需要上板确认；
- 驱动层没有单元测试/mock，目前只做了编译和 PC 侧协议测试；
- `ADS_DBG` 仍是阻塞 200 ms 的诊断命令，与原固件一致；
- 没有 OTA、低功耗管理、量产校准；
- 没有做长时间稳定性/内存泄漏测试。


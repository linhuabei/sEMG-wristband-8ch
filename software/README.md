# 八通道 sEMG 采集上位机

面向 **ESP32-S3 + ADS1298** 八通道表面肌电手环的 Windows 上位机。使用 PyQt5 构建界面，通过 BLE 接收 1000/2000 Hz 采样流，完成实时波形、数据采集、标定、IMU 姿态、设备调试和外部算法插件推理。

> 当前是论文/课程项目的**联调版本**：采集、显示、存储、标定链路可用；手势识别默认使用 RMS 阈值占位，真实模型通过「导入算法 .py」加载。详细限制见文末「已知限制」。
>
> 实际操作、界面说明和常见问题见 [USAGE.md](USAGE.md)。

## 功能

| 模块 | 内容 |
| --- | --- |
| 连接与监视 | BLE 扫描/连接、连接后自动开始采集、8 路实时波形、带通/陷波开关、1000/2000 Hz 切换、波形录制 CSV |
| 数据采集 | 采集 8 通道自采数据集，输出 `emg.csv` + `meta.json`；角度通道为预留接口 |
| 用户迁移 | 标定采集（6 动作 × N 轮、自动打标）、实时识别展示、外部算法导入、个人化迁移脚本生成 |
| 设备调试 | GPIO 读写、I2C/SPI 测试、IMU 读取、ADS1298 复位/启停、文本命令台 |
| IMU 姿态 | LSM6DSR 3D 方块 + 横滚/俯仰/航向 + 零位校准 |
| 信号处理 | 20–450 Hz 带通、50 Hz 陷波、直流阻断、50 ms 滑窗特征（RMS/MAV/ZC/WL） |

## 环境

- Windows 10/11（BLE 依赖系统蓝牙）
- Python 3.10+（本机在 3.14 上验证）
- 一块已烧录配套固件的 ESP32-S3 + ADS1298 采集板

## 快速开始

```powershell
cd <仓库根目录>
python -m venv .venv
.venv\Scripts\activate
pip install -r requirements.txt
python run.py
```

也可以直接双击 `start.bat`；需要看日志时：

```powershell
start.bat debug
```

启动前自检（关键文件、语法、导入、依赖、协议）：

```powershell
python run.py --check
# 或
start.bat check
```

## 使用流程

1. 给手环上电，确认状态灯亮；
2. 打开上位机 → 左侧「扫描」→ 选择 `Node_1_by_hzx` → 「连接」；
3. 连接成功后自动开始采集，左侧显示 8 路波形；
4. 右侧「控制」栏可开始/停止采集、切换采样率、开关滤波、录制波形、采集自采数据集、开始标定；
5. 「用户迁移」页按提示完成标定，或点击「导入算法 .py」加载训练好的模型。

> 连接后没有单独的「断开」按钮，关闭程序即断开。

## 目录结构

```text
.
├── main.py                     # 程序入口
├── run.py                      # 启动器（--check 运行完整性检查）
├── start.bat                   # 启动 / debug / check
├── check.py                    # 关键文件、语法、导入、依赖、协议自检
├── requirements.txt
├── requirements-dev.txt        # 打包用（PyInstaller）
├── core/
│   ├── esp32_controller.py     # BLE 扫描/连接/命令/EMG 流解析
│   ├── device_manager.py       # 设备状态、数据分发、IMU 轮询
│   ├── emg_protocol.py         # 二进制 EMG 协议 + 自检
│   ├── signal_processor.py     # 流式滤波 + 滑窗特征
│   ├── gesture_classifier.py   # 内置阈值分类器 / 外部算法调度
│   ├── algorithm_loader.py     # .py 算法插件加载
│   ├── training_plan.py        # 标定方案 JSON
│   └── paths.py                # 资源与数据目录
├── ui/
│   ├── main_window.py
│   ├── themes/pixel_theme.py
│   ├── views/                  # 监视 / 数据采集 / 用户迁移 / 实时识别 / 标定 / 设备调试
│   └── widgets/                # 右侧控制栏 / IMU 面板
├── examples/                   # 自定义算法插件示例
├── resources/                  # 字体、图标
├── plans/                      # 标定方案模板
├── README.md
└── USAGE.md                    # 详细使用说明
```

运行时自动创建：`captured_data/`、`training_data/`、`recordings/`（已在 `.gitignore` 中忽略）；`plans/` 只保留模板 `personal_calibration_plan.json`。

## 硬件与协议

- 设备名：`Node_1_by_hzx`（可在固件 `setup()` 修改）
- BLE Service：`4fafc201-1fb5-459e-8fcc-c5c9c331914b`
- BLE Characteristic：`beb5483e-36e1-4688-b7f5-ea07361b26a8`
- 默认 2000 SPS、24 bit；16 bit 模式由固件在 MTU 较小时自动降级

每个采样点一帧：

```text
[0xAA][0x55][seq][CH1..CH8][xor]
```

- `seq`：1 字节循环序号
- 24 bit：每通道 3 字节大端有符号码值，帧长 28 字节
- 16 bit：每通道 2 字节，帧长 20 字节
- `xor`：除校验字节外整帧异或

常用文本命令：`EMG_START`、`EMG_STOP`、`EMG_SPS=1000`、`ADS_INIT`、`ADS_GAIN=12`、`IMU`、`ALL`、`PIN8=1`、`READ11`、`I2C_SCAN`、`SPI_TEST`、`HELP`。

## 算法插件

在「用户迁移 → 实时识别」中点击「导入算法 .py」，选择一个满足以下约定的 Python 文件：

```python
NAME = "我的手势模型"
TYPE = "gesture"          # 或 "angles"

def predict(ctx):
    f = ctx["features"]   # rms / mav / zc / wl / samples(T, 8)
    return "握拳", 0.92    # TYPE="angles" 时返回角度数组
```

`ctx["features"]["samples"]` 是最近约 50 ms 的滤波后 8 通道数据。示例见 [`examples/example_classifier.py`](examples/example_classifier.py) 与 [`examples/example_angles.py`](examples/example_angles.py)。插件每 50 ms 调用一次，内部需要时序缓存的算法请自行维护状态。

## 数据格式

- `captured_data/capture_YYYYMMDD_HHMMSS/emg.csv`：`timestamp,ch1..ch8`
- 同目录 `meta.json`：采样率、通道数、样本数、时长、备注
- `training_data/session_YYYYMMDD_HHMMSS/training_*.csv`：`timestamp,gesture_code,CH1..CH8`
- `recordings/sEMG_YYYYMMDD_HHMMSS.csv`：`timestamp,CH1..CH8`

标签：`0=休息`、`1=握拳`、`2=十指张开`、`3=OK`、`4=比大拇指`、`5=比耶`、`6=捏合`。

## 打包 exe

```powershell
pip install -r requirements-dev.txt
pyinstaller --noconfirm --clean node_yushi.spec
# 产物：dist\node_yushi.exe
```

## 已知限制

- 内置分类器只是 RMS 阈值占位（握拳/张开/无），真实识别需导入算法插件；
- 本仓库不含训练好的模型权重，也不含同级的 `算法/plugins`；发布 exe 时请把插件和权重一起分发；
- 「个人化迁移」依赖同级的 `算法/src/个人化迁移.py` 和一台装了 PyTorch 的电脑，单独运行本仓库时会提示找不到脚本；
- 暂未实现机械手/舵机/力矩控制和串口输出，本程序只负责采集、显示、存储与识别；
- 「角度输出」目前是插件接口预留，视觉角度源尚未接入；
- 连接后没有断开按钮，需关闭程序断开；
- 只在 Windows + 本项目的 ESP32-S3 固件上验证过，其他平台/固件未测试。

## 许可证

暂未指定开源许可证。如需引用、复用或二次分发，请先联系作者。

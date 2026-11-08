# sEMG-wristband-8ch

基于 **ESP32-S3** 与 **ADS1298** 的低成本 8 通道表面肌电（sEMG）采集手环，面向前臂肌电信号采集与手势识别应用。

本项目包含**硬件设计**、**固件**以及 **上位机**，支持 8 通道 sEMG 实时采集、BLE 无线传输、IMU 姿态解算与数据可视化。

---

## 目录结构

```
sEMG-wristband-8ch/
├── firmware/          # ESP32-S3 固件（Arduino .ino）
├── hardware/          # 硬件设计（原理图、PCB、BOM）
│   └── README.md      # 硬件详细说明（引脚定义、电源、注意事项）
├── software/          # PyQt5 上位机（BLE 接收、波形显示、数据采集）
│   └── USAGE.md       # 上位机操作手册与常见问题
└── README.md          # 本文件
```

---

## 核心硬件

| 模块 | 型号 | 说明 |
| :--- | :--- | :--- |
| 主控 | ESP32-S3-ZERO | WiFi / BLE 无线通信，主控运算 |
| 模拟前端 | ADS1298IPAGR | 8 通道 24 位 ADC，内置 PGA，数据速率 250 SPS – 32 kSPS |
| 姿态传感器 | LSM6DSR | 6 轴 IMU（3 轴加速度 + 3 轴陀螺仪），IIC 接口 |
| 充电管理 | BQ24075RGTR | 锂电池充电与路径管理 |
| 稳压 | TPS7A2033PDBVR | 3.3 V LDO，为系统提供稳定电源 |

> ADS1298 每通道功耗约 0.75 mW，输入参考噪声低至 4 μVPP（G = 6），适合微弱生物电信号采集。本项目典型采样率 1000 / 2000 Hz，满足 sEMG 信号的频带需求。

硬件引脚定义、电源说明、调试注意事项等详细信息，请查阅 **[hardware/README.md](hardware/README.md)** 。

---

## 上位机功能

`software/` 目录提供一个基于 **PyQt5** 的 Windows 上位机，通过 **BLE** 接收 ESP32-S3 发送的 sEMG 数据流，主要功能包括：

- **实时采集与显示**：8 路波形实时绘制，支持 1000 / 2000 Hz 采样率切换，20–450 Hz 带通 + 50 Hz 陷波滤波
- **数据存储**：录制波形为 CSV，支持 8 通道自采数据集导出
- **标定与手势识别**：6 种手势标定采集、RMS 阈值分类器、外部算法 `.py` 插件导入
- **设备调试**：GPIO 读写、IIC / SPI 测试、IMU 读取、ADS1298 复位 / 启停
- **IMU 姿态**：LSM6DSR 3D 方块实时显示，横滚 / 俯仰 / 航向角，零位校准

详细操作流程、界面说明和常见问题，请查阅 **[software/USAGE.md](software/USAGE.md)** 。

---

## 步骤吧

### 固件烧录

1. 使用 Arduino IDE 打开 `firmware/ctrl_legacy_fw_ESP32S3.ino`。
2. 选择开发板 **ESP32S3 Dev Module**，端口选择对应的串口。
3. 编译并上传。

### 上位机运行

```bash
# 进入仓库根目录
python -m venv .venv
.venv\Scripts\activate

pip install -r software/requirements.txt
python software/run.py
```

也可以直接双击 `software/start.bat` 启动；如需查看日志，运行 `start.bat debug`。

启动前可执行自检，确认关键文件、依赖和通信协议完整：

```bash
python software/run.py --check
```

### 连接设备

1. 手环上电，确认状态指示灯亮起。
2. 打开上位机，点击 **「扫描」**，选择 `Node_1_by_hzx`。
3. 点击 **「连接」**，连接成功后自动开始采集，左侧显示 8 路波形。

> 连接后没有单独的「断开」按钮，关闭上位机程序即断开连接。

---

## 文档索引

| 文档 | 内容 |
| :--- | :--- |
| [hardware/README.md](hardware/README.md) | 硬件架构、ESP32 引脚映射表、电源说明、调试注意事项 |
| [software/USAGE.md](software/USAGE.md) | 上位机安装、界面说明、操作流程、常见问题 |

---

## 已知限制

- 手势识别默认使用 **RMS 阈值占位**，真实分类模型需通过「导入算法 .py」加载。
- 上位机为论文 / 课程项目的**联调版本**，采集、显示、存储、标定链路可用。
- 当前固件为单一 `.ino` 文件（`ctrl_legacy_fw_ESP32S3.ino`），尚未拆分为模块化工程。

## 说在最后
- 本项目除硬件内容外，有相当一**部分由AI辅助**完成
- 说实在还是有很多不足，测试因为考研所以暂且搁置，如果考上了有时间再玩玩吧，没考上我也没招。
- 祝好

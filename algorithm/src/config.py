# -*- coding: utf-8 -*-
"""算法配置：数据选择 / 预处理 / 滑窗 / 模型 / 训练超参数。

编号约定（全部用 1 基，跟 NinaPro 官方 README 一致）：
  * sEMG：DB2 一共 12 路电极。第 1-8 路是沿前臂“一圈”均匀分布的 8 个电极
    （腕肘之间），第 9-10 路贴在指浅屈肌/指总伸肌，第 11-12 路贴在肱二头肌/肱三头肌。
    本项目手环就是 8 通道，所以默认只取 1-8 路。
  * 受试者：DB2 共 40 位健全受试者，默认排除 20 号（exclude_subjects）。
  * 关节角：Cyberglove 共 22 路，默认排除 20 号（angle_exclude_channels）。
"""
from dataclasses import dataclass, field
from pathlib import Path
from typing import Tuple

# ---------------- 路径 ----------------
HERE = Path(__file__).resolve().parent
ALGO_DIR = HERE.parent                     # 算法/
REPO_DIR = ALGO_DIR.parent                 # 肌电手势检测/
NINAPRO_DIR = ALGO_DIR / "公开数据集" / "Ninapro"
DB2_DIR = NINAPRO_DIR / "DB2"
CACHE_DIR = ALGO_DIR / "_cache"            # 缓存目录（可随时删除后重新生成）
OUTPUT_DIR = ALGO_DIR / "outputs"          # 结果：曲线/混淆矩阵/指标/权重
PLUGIN_DIR = ALGO_DIR / "plugins"          # 上位机算法插件


# ---------------- 一、数据选择 ----------------
@dataclass
class DataConfig:
    """选哪些数据。"""

    dataset: str = "DB2"           # NinaPro DB2：12 通道、40 位健全受试者、2 kHz
    # 换成自己采的数据时：dataset="custom"，并指定目录与文件名规则
    custom_dir: str = ""           # 自采数据目录（每条记录一个 .npz / .csv）
    custom_pattern: str = "*.npz"
    custom_subject_regex: str = r"[Ss](\d+)"
    exercise: int = 1              # 1 = 练习 B（17 个手部基本动作，本文主实验）
                                   # 2 = 练习 C（抓握类，23 个），3 = 练习 D（力模式，9 个）
    emg_channels: Tuple[int, ...] = (1, 2, 3, 4, 5, 6, 7, 8)   # 取 sEMG 1-8 通道
    subjects: Tuple[int, ...] = tuple(range(1, 41))            # 全部 40 位
    exclude_subjects: Tuple[int, ...] = (20,)                  # 排除 20 号受试者
    use_restimulus: bool = True    # 用官方“重标注”标签（时间对齐更好）
    # label_mode:
    #   "ninapro" —— 18 类：0=休息 + 1-17 号 NinaPro 动作（练习 B 全集）
    #   "app6"    —— 7 类：0=休息 + 1-6 号上位机手势（把 NinaPro 动作按语义映射过去，
    #                对不上的动作直接丢掉），训练出来的模型可以直接插到上位机里
    label_mode: str = "ninapro"
    angle_source: str = "glove"    # 关节角来自 Cyberglove
    angle_exclude_channels: Tuple[int, ...] = (20,)            # 角度排除 20 号
    # 关节角目标低通（Hz）：真实手部姿态的变化基本在 5 Hz 以内，手套原始信号里
    # 的高频抖动是传感器噪声，肌电本来也预测不出来。0 = 不滤波。
    angle_smooth_hz: float = 5.0
    # 6 次重复这样分：1/3/4 训练，6 做验证（早停用），2/5 测试。
    # 全部按“第几次重复”划分，绝不随机切窗口 —— 相邻窗口 50% 重叠，
    # 随机划分会把同一段动作同时放进训练和测试，准确率会被明显高估。
    train_repetitions: Tuple[int, ...] = (1, 3, 4)              # 训练用重复
    val_repetitions: Tuple[int, ...] = (6,)                     # 验证用重复
    test_repetitions: Tuple[int, ...] = (2, 5)                  # 测试用重复

    @property
    def subject_list(self):
        drop = set(self.exclude_subjects)
        return [s for s in self.subjects if s not in drop]

    @property
    def emg_slice(self):
        """sEMG 通道号 -> python 的 0 基列索引。"""
        return [c - 1 for c in self.emg_channels]

    @property
    def angle_keep(self):
        """22 路角度里保留哪些（0 基索引）。"""
        drop = set(self.angle_exclude_channels)
        return [c - 1 for c in range(1, 23) if c not in drop]

    def cache_path(self, subject: int) -> Path:
        return CACHE_DIR / f"{self.dataset}_s{subject}_E{self.exercise}.npz"


# ---------------- 二、预处理 ----------------
@dataclass
class PreprocessConfig:
    """跟固件/上位机 signal_processor.py 对齐的滤波参数。"""

    fs: int = 2000                 # NinaPro 采样率
    bandpass: Tuple[float, float] = (20.0, 450.0)   # 20-450 Hz 巴特沃斯带通
    notch: float = 50.0            # 工频陷波
    notch_q: float = 30.0
    detrend: bool = True           # 去均值
    # style:
    #   "online"  —— 用上位机那套**因果**流式滤波（sosfilt + 直流阻断）。
    #                训练和部署预处理完全一致，这是默认值，也是能真正插到上位机里跑的前提。
    #   "offline" —— 零相位 filtfilt（离线研究常用，相位不偏移，指标更好看，
    #                但和上位机实时链路不一致，实机部署会有域偏移）。
    style: str = "online"
    norm: str = "rolling"          # "rolling" 因果滚动标准化（= 插件做法）/ "global" 整段标准化
    norm_window_s: float = 6.0     # 滚动标准化窗口（秒），要和插件里的参数一致


# ---------------- 三、滑窗 ----------------
@dataclass
class WindowConfig:
    """滑窗分段（对应项目书 2.3.4 节）。"""

    length: int = 200              # W=200 → 2000 Hz 下正好 100 ms
    stride_train: int = 200        # 训练步长 200（100 ms，窗口不重叠；CPU 上更省时间，
                                   # 相邻 50% 重叠窗口信息高度冗余，去掉后精度基本不变）
    stride_val: int = 200          # 验证步长（只用于早停，稀疏一点省时间）
    stride_test: int = 50          # 测试步长 50（25 ms，75% 重叠，和项目书一致）
    drop_unlabeled: bool = True    # 丢掉没有标签的采样点
    # 连续输出时，窗口的回归目标取哪一段：tail = 窗口最后 25 ms（近实时，推荐）
    angle_target: str = "tail"
    # 严格模式：只保留整段 200 点都落在同一个“重复段”里的窗口，
    # 彻底杜绝窗口跨重复边界（见 audit_split.py）。
    # 打开后每段训练数据大约少 2~3 个窗口，对指标无影响。
    require_same_repetition: bool = False


# ---------------- 四、模型 ----------------
@dataclass
class ModelConfig:
    """CNN-Transformer 结构（对应项目书 2.3.5 节）。"""

    conv1_channels: int = 32
    conv2_channels: int = 64       # = d_model
    kernel_size: int = 5
    d_model: int = 64
    n_heads: int = 8
    n_layers: int = 2
    dim_feedforward: int = 128     # 前馈用深度可分离卷积（轻量化）
    dropout: float = 0.2
    use_pos_encoding: bool = True
    n_classes: int = 18            # 练习 B：0=休息 + 1-17 号动作
    n_angles: int = 21             # 22 路关节角去掉第 20 路
    multi_task: bool = False       # True 时同时回归关节角


# ---------------- 五、训练 ----------------
@dataclass
class TrainConfig:
    epochs: int = 20
    batch_size: int = 128          # 项目书写 32；CPU 上放大批更快，精度相当
    lr: float = 1e-3
    weight_decay: float = 1e-4
    early_stop_patience: int = 5   # 验证集准确率连续 5 轮不升就停
    cosine_annealing: bool = True
    label_smoothing: float = 0.02
    class_weight: str = "none"     # "none" = 按真实类别先验（休息占一半）
                                   # "balanced" = 按类别数倒数加权，压制“一直猜休息”
    gesture_loss_weight: float = 1.0  # 分类损失权重（--task angles 时置 0 = 纯回归）
    angle_loss_weight: float = 0.2    # 角度回归损失权重（--task angles 时置 1）
    calibrate: bool = True            # 连续输出：用验证集拟合逐通道仿射校准
    augment_noise_snr_db: Tuple[float, float] = (10.0, 30.0)  # 高斯白噪声增强
    augment_time_flip: bool = True
    # 通道重排增强：项目书里写了，但实测会把“电极-通道”的对应关系打乱，
    # 8 通道模型反而学不动，默认关掉（需要时用 --channel-shuffle 打开）。
    augment_channel_shuffle: bool = False
    seed: int = 42
    num_workers: int = 0
    device: str = "auto"           # auto / cpu / cuda


@dataclass
class Config:
    data: DataConfig = field(default_factory=DataConfig)
    pre: PreprocessConfig = field(default_factory=PreprocessConfig)
    window: WindowConfig = field(default_factory=WindowConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    train: TrainConfig = field(default_factory=TrainConfig)


# 练习 B（DB2 的 17 个手部基本动作）的参考命名，用于混淆矩阵/报告可读。
DB2_EX1_NAMES = {
    0: "休息",
    1: "拇指竖起",
    2: "食指中指伸",
    3: "无名指小指屈",
    4: "拇指对指小指",
    5: "五指张开",
    6: "握拳",
    7: "食指伸出",
    8: "并指内收",
    9: "旋后-中指轴",
    10: "旋前-中指轴",
    11: "旋后-小指轴",
    12: "旋前-小指轴",
    13: "腕屈",
    14: "腕伸",
    15: "桡偏",
    16: "尺偏",
    17: "握拳腕伸",
}

# 上位机 6 种手势（0=休息）与 NinaPro 练习 B 动作的对应关系。
# 语义最接近的映射，用于把公开数据集训练出来的模型直接插到上位机上跑。
APP_GESTURE_MAP = {   # 上位机 code -> NinaPro 动作号
    1: 6,    # 握拳     <- 6 握拳
    2: 5,    # 十指张开 <- 5 五指张开
    3: 4,    # OK手势   <- 4 拇指对指（形态最接近）
    4: 1,    # 比大拇指 <- 1 拇指竖起
    5: 2,    # 比耶     <- 2 食指中指伸
    6: 3,    # 捏合     <- 3 无名指小指屈（近似）
}
APP_GESTURE_NAMES = {0: "休息", 1: "握拳", 2: "十指张开", 3: "OK手势",
                     4: "比大拇指", 5: "比耶", 6: "捏合"}

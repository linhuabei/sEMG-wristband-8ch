# -*- coding: utf-8 -*-
"""把 NinaPro 连续信号切成 CNN-Transformer 要的 (C, T) 样本。

流程（每位受试者各自一份）：
  1) 读缓存（8 路 sEMG + 标签 + 重复次数 + Cyberglove）
  2) 20-450 Hz 带通 + 50 Hz 陷波 + 去均值
  3) 每通道 z-score —— 统计量只用“训练重复”的数据估计（避免测试泄漏）
  4) 滑动窗口切段，窗口标签取窗口末端时刻的类别（项目书 2.3.4 节）
  5) 按“第几次重复”划分训练/测试，而不是随机划分窗口
     —— 重叠窗口如果随机划分，同一个动作会同时出现在训练和测试里，准确率会被高估

为了省内存，窗口不是预先展开的大数组，而是存 (受试者, 起始点) 索引，取样本时再切。
"""
from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np
import torch
from numpy.lib.stride_tricks import sliding_window_view
from torch.utils.data import Dataset

from config import APP_GESTURE_MAP, Config
from ninapro import load_subject
from preprocess import (ChannelScaler, add_gaussian_noise, apply_filters,
                        causal_stream_filter, rolling_zscore)


@dataclass
class SubjectSignals:
    """一位受试者预处理完的连续信号。"""

    subject: int
    emg: np.ndarray                 # (T, C) 已滤波 + 已标准化
    labels: np.ndarray              # (T,) 动作号，0 = 休息
    reps: np.ndarray                # (T,) 第几次重复
    angles: Optional[np.ndarray] = None   # (T, A) 已标准化的关节角
    angle_mean: Optional[np.ndarray] = None
    angle_std: Optional[np.ndarray] = None
    n_windows: int = 0


def prepare_subject(subject: int, cfg: Config, verbose: bool = False) -> SubjectSignals:
    """读一位受试者 -> 滤波 -> 标准化 -> 返回连续信号。"""
    if cfg.data.dataset == "custom":
        from custom_dataset import prepare_custom
        return prepare_custom(subject, cfg, verbose)
    data = load_subject(subject, cfg.data.exercise, cfg.data)
    raw = data["emg"]
    labels = (data["restimulus"] if cfg.data.use_restimulus else data["stimulus"]).astype(np.int64)
    reps = (data["rerepetition"] if cfg.data.use_restimulus else data["repetition"]).astype(np.int64)
    if cfg.data.label_mode == "app6":
        # NinaPro 动作号 -> 上位机 6 种手势（对不上的动作标 -1，后面会被丢掉）
        lut = np.full(int(labels.max()) + 2, -1, dtype=np.int64)
        lut[0] = 0
        for code, mov in APP_GESTURE_MAP.items():
            if 0 < mov < lut.size:
                lut[mov] = code
        labels = lut[labels]

    if cfg.pre.style == "online":
        # 和上位机 signal_processor.py 一模一样的因果滤波
        emg = causal_stream_filter(raw, cfg.pre.fs, cfg.pre.bandpass,
                                   cfg.pre.notch, cfg.pre.notch_q)
    else:
        emg = apply_filters(raw, cfg.pre.fs, cfg.pre.bandpass, cfg.pre.notch,
                            cfg.pre.notch_q, cfg.pre.detrend)

    train_mask = np.isin(reps, np.asarray(cfg.data.train_repetitions))
    if train_mask.sum() < 1000:            # 极端情况兜底
        train_mask = np.ones_like(train_mask, dtype=bool)
    if cfg.pre.norm == "rolling":
        # 因果滚动标准化：和插件的做法一致（插件用最近 6 s 的统计量）
        emg = rolling_zscore(emg, int(cfg.pre.norm_window_s * cfg.pre.fs))
        scaler = None
    else:
        scaler = ChannelScaler().fit(emg[train_mask])
        emg = scaler.transform(emg)

    angles = angle_mean = angle_std = None
    if cfg.model.multi_task and "glove" in data:
        glove = np.asarray(data["glove"], dtype=np.float64)[:, cfg.data.angle_keep]
        if cfg.data.angle_smooth_hz and cfg.data.angle_smooth_hz > 0:
            from scipy import signal as _sig
            sos = _sig.butter(2, cfg.data.angle_smooth_hz / (cfg.pre.fs / 2.0),
                              output="sos")
            glove = _sig.sosfiltfilt(sos, glove, axis=0)
        gmean = glove[train_mask].mean(axis=0)
        gstd = glove[train_mask].std(axis=0)
        gstd[gstd < 1e-12] = 1.0
        angles = ((glove - gmean) / gstd).astype(np.float32)
        angle_mean, angle_std = gmean.astype(np.float32), gstd.astype(np.float32)

    if verbose:
        print(f"  [prep] S{subject}: {emg.shape[0]} 点，滤波 {cfg.pre.style}，"
              f"标准化 {cfg.pre.norm}")

    return SubjectSignals(subject=subject, emg=emg, labels=labels, reps=reps,
                          angles=angles, angle_mean=angle_mean, angle_std=angle_std)


def window_index(sig: SubjectSignals, cfg: Config, reps, stride: int):
    """返回满足条件的窗口起始点数组。"""
    W = cfg.window.length
    T = sig.emg.shape[0]
    if T < W:
        return np.empty(0, dtype=np.int64)
    starts = np.arange(0, T - W + 1, stride, dtype=np.int64)
    end = starts + W - 1
    keep = np.isin(sig.reps[end], np.asarray(reps))
    if cfg.data.dataset == "custom":
        # 自采数据：标签 -1 表示"这段不算"（比如过渡段/没标的地方）
        keep &= sig.labels[end] >= 0
    if getattr(cfg.window, "require_same_repetition", False):
        # 严格模式：窗口内 200 个点必须全在同一个重复段里（不跨边界）
        keep &= (sig.reps[starts] == sig.reps[end])
    if cfg.window.drop_unlabeled:
        # 丢掉官方重标注里 rerepetition=0 的过渡段（没有有效标签）
        keep &= sig.reps[end] > 0
    if cfg.data.label_mode == "app6":
        # 丢掉不属于上位机 6 种手势的动作（映射后标签为 -1）
        keep &= sig.labels[end] >= 0
    return starts[keep]


def build_index(signals: List[SubjectSignals], cfg: Config):
    """把每人的训练/验证/测试窗口索引拼起来：[(受试者下标, 起始点), ...]"""
    train_pos, val_pos, test_pos = [], [], []
    for i, sig in enumerate(signals):
        for st in window_index(sig, cfg, cfg.data.train_repetitions, cfg.window.stride_train):
            train_pos.append((i, int(st)))
        for st in window_index(sig, cfg, cfg.data.val_repetitions, cfg.window.stride_val):
            val_pos.append((i, int(st)))
        for st in window_index(sig, cfg, cfg.data.test_repetitions, cfg.window.stride_test):
            test_pos.append((i, int(st)))
    def arr(p):
        return np.asarray(p, dtype=np.int64).reshape(-1, 2)
    return arr(train_pos), arr(val_pos), arr(test_pos)


class WindowDataset(Dataset):
    """按索引惰性切窗口。train=True 时做数据增强（噪声/时间翻转/通道重排）。"""

    def __init__(self, signals: List[SubjectSignals], positions: np.ndarray, cfg: Config,
                 train: bool = False, seed: int = 0):
        self.signals = signals
        self.positions = positions
        self.W = cfg.window.length
        self.train = train
        self.gen = torch.Generator().manual_seed(seed)
        self.tcfg = cfg.train
        self.multi_task = cfg.model.multi_task
        self.angle_target = getattr(cfg.window, "angle_target", "tail")

    @property
    def WINDOW(self):
        return self.W

    def __len__(self):
        return len(self.positions)

    def _window(self, idx):
        s, st = self.positions[idx]
        sig = self.signals[int(s)]
        end = int(st) + self.W
        return sig, int(st), end

    def __getitem__(self, idx):
        sig, st, end = self._window(idx)
        # (W, C) -> (C, W)
        x = np.ascontiguousarray(sig.emg[st:end].T)
        y = int(sig.labels[end - 1])
        item = {
            "x": x,
            "y": y,
            "subject": np.int16(sig.subject),
            "start": np.int64(st),
        }
        if self.multi_task and sig.angles is not None:
            # 连续输出：目标取窗口最后 25 ms 的平均（比整窗平均更接近“当前时刻”的姿态）
            tail = max(1, self.W // 4) if self.angle_target == "tail" else self.W
            item["angles"] = sig.angles[end - tail:end].mean(axis=0)

        if self.train:
            seed = int(torch.randint(0, 2 ** 31 - 1, (1,), generator=self.gen).item())
            rng = np.random.default_rng(seed)
            x = x.copy()
            # ① 10~30 dB 高斯白噪声
            lo, hi = self.tcfg.augment_noise_snr_db
            x = add_gaussian_noise(x, float(rng.uniform(lo, hi)), rng)
            # ② 时间轴翻转（左右颠倒，不改变频谱统计量）
            if self.tcfg.augment_time_flip and rng.random() < 0.5:
                x = x[:, ::-1]
            # ③ 通道重排（模拟电极位置/接触差异）
            if self.tcfg.augment_channel_shuffle and rng.random() < 0.5:
                perm = rng.permutation(x.shape[0])
                x = x[perm, :]
            item["x"] = np.ascontiguousarray(x)

        item["x"] = torch.from_numpy(np.ascontiguousarray(item["x"], dtype=np.float32))
        item["y"] = torch.tensor(y, dtype=torch.long)
        if "angles" in item:
            item["angles"] = torch.from_numpy(np.asarray(item["angles"], dtype=np.float32))
        return item


def split_summary(signals: List[SubjectSignals], train_pos, test_pos, cfg: Config):
    """统计每个类别的样本数，方便检查类别是否均衡。"""
    def counts(pos):
        c = np.zeros(cfg.model.n_classes, dtype=np.int64)
        for s, st in pos:
            lab = int(signals[int(s)].labels[int(st) + cfg.window.length - 1])
            if 0 <= lab < c.size:
                c[lab] += 1
        return c

    return {"train": counts(train_pos), "test": counts(test_pos)}

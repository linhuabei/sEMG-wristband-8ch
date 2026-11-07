# -*- coding: utf-8 -*-
"""CNN-Transformer 混合模型（对应项目书 2.3.5 节）。

输入:  (B, C=8, T=200)   8 通道、100 ms（2 kHz）
结构:
  ① 卷积特征提取  Conv1d(8→32, k=5) + BN + ReLU + MaxPool(2)
                  Conv1d(32→64, k=5) + BN + ReLU + MaxPool(2)   →  (B, 64, 50)
  ② 位置编码 + 2 层 Transformer 编码器（8 头自注意力 + 深度可分离卷积前馈 + 残差/层归一化）
  ③ 时间维全局平均池化 → Dropout → 全连接分类头（18 类：休息 + 17 个动作）
     （多任务时再加一个线性回归头输出 21 路关节角）

为什么要 CNN + Transformer：
  * 1D-CNN 的卷积核只覆盖局部几十毫秒，擅长抓“肌电爆发/波形局部形状”，参数少、好训练；
  * 自注意力让每个时间片都能直接看到其它时间片和其它通道，能建模“屈肌-伸肌”跨通道的
    协同发放关系，这是纯 CNN 感受野不容易覆盖的；
  * 两者结合后在相同参数量下比单纯 CNN/LSTM 更稳，且注意力可以并行算，推理延迟低。
"""
import math

import torch
import torch.nn as nn
import torch.nn.functional as F

from config import ModelConfig


class DepthwiseSeparableFFN(nn.Module):
    """沿时间轴的深度可分离卷积前馈（轻量化，替代标准 Transformer 的两层全连接）。"""

    def __init__(self, d_model: int, dim_ff: int, kernel: int = 3, dropout: float = 0.1):
        super().__init__()
        self.depthwise = nn.Conv1d(d_model, d_model, kernel,
                                   padding=kernel // 2, groups=d_model)
        self.pointwise1 = nn.Conv1d(d_model, dim_ff, 1)
        self.pointwise2 = nn.Conv1d(dim_ff, d_model, 1)
        self.drop = nn.Dropout(dropout)

    def forward(self, x):                    # x: (B, T, D)
        h = x.transpose(1, 2)                # (B, D, T)
        h = F.gelu(self.depthwise(h))
        h = F.gelu(self.pointwise1(h))
        h = self.pointwise2(h)
        return self.drop(h).transpose(1, 2)


class LiteEncoderLayer(nn.Module):
    """Pre-LN 的编码器层：多头自注意力 + 深度可分离卷积前馈 + 残差。"""

    def __init__(self, d_model: int, n_heads: int, dim_ff: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.attn = nn.MultiheadAttention(d_model, n_heads, dropout=dropout,
                                          batch_first=True)
        self.drop1 = nn.Dropout(dropout)
        self.norm2 = nn.LayerNorm(d_model)
        self.ffn = DepthwiseSeparableFFN(d_model, dim_ff, kernel=3, dropout=dropout)

    def forward(self, x):
        h = self.norm1(x)
        a, _ = self.attn(h, h, h, need_weights=False)
        x = x + self.drop1(a)
        x = x + self.ffn(self.norm2(x))
        return x


def sinusoidal_encoding(seq_len: int, d_model: int) -> torch.Tensor:
    """经典正弦位置编码 (1, T, D)。"""
    pos = torch.arange(seq_len, dtype=torch.float32).unsqueeze(1)
    div = torch.exp(torch.arange(0, d_model, 2, dtype=torch.float32)
                    * (-math.log(10000.0) / d_model))
    pe = torch.zeros(seq_len, d_model)
    pe[:, 0::2] = torch.sin(pos * div)
    pe[:, 1::2] = torch.cos(pos * div)
    return pe.unsqueeze(0)


class CNNTransformer(nn.Module):
    def __init__(self, cfg: ModelConfig, in_channels: int = 8, seq_len: int = 200):
        super().__init__()
        self.cfg = cfg
        self.multi_task = cfg.multi_task

        # ① 卷积特征提取
        self.conv1 = nn.Conv1d(in_channels, cfg.conv1_channels, cfg.kernel_size,
                               padding=cfg.kernel_size // 2, bias=False)
        self.bn1 = nn.BatchNorm1d(cfg.conv1_channels)
        self.pool1 = nn.MaxPool1d(2)
        self.conv2 = nn.Conv1d(cfg.conv1_channels, cfg.d_model, cfg.kernel_size,
                               padding=cfg.kernel_size // 2, bias=False)
        self.bn2 = nn.BatchNorm1d(cfg.d_model)
        self.pool2 = nn.MaxPool1d(2)

        # ② 位置编码 + Transformer 编码器
        seq2 = seq_len // 4
        if cfg.use_pos_encoding:
            self.register_buffer("pos_enc", sinusoidal_encoding(seq2, cfg.d_model),
                                 persistent=True)
        else:
            self.pos_enc = None
        self.encoder = nn.ModuleList([
            LiteEncoderLayer(cfg.d_model, cfg.n_heads, cfg.dim_feedforward, cfg.dropout)
            for _ in range(cfg.n_layers)
        ])
        self.norm = nn.LayerNorm(cfg.d_model)
        self.drop = nn.Dropout(cfg.dropout)

        # ③ 输出头
        self.head_gesture = nn.Linear(cfg.d_model, cfg.n_classes)
        if self.multi_task:
            self.head_angles = nn.Linear(cfg.d_model, cfg.n_angles)

    def features(self, x):                   # (B, C, T)
        h = self.pool1(F.relu(self.bn1(self.conv1(x))))
        h = self.pool2(F.relu(self.bn2(self.conv2(h))))
        h = h.transpose(1, 2)                # (B, T/4, D)
        if self.pos_enc is not None:
            h = h + self.pos_enc[:, :h.shape[1]]
        for layer in self.encoder:
            h = layer(h)
        return self.norm(h)

    def forward(self, x):
        h = self.features(x)
        z = self.drop(h.mean(dim=1))         # 全局平均池化
        out = {"logits": self.head_gesture(z)}
        if self.multi_task:
            out["angles"] = self.head_angles(z)
        return out


def count_parameters(model: nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

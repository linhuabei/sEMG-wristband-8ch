# -*- coding: utf-8 -*-
"""把训练好的 CNN-Transformer 用纯 NumPy 跑推理（不依赖 torch）。

为什么要有这个东西：
  上位机的运行环境（.venv）里只有 numpy/scipy/PyQt5，没有 torch。
  训练完把权重导出成 .npz，再用这份前向实现跑推理，插件就能直接在上位机里用，
  也方便以后移植到嵌入式端（每个算子都是卷积/矩阵乘/softmax，好量化）。

  * export_numpy.py 负责把 best.pt -> model_numpy.npz
  * 这个文件负责 npz -> 前向计算，数值与 PyTorch 版本一致（见 verify_numpy.py）
"""
import json

import numpy as np


def _erf(x):
    """math.erf 的向量化版本（gelu 精确版要用）。"""
    from math import erf
    return np.vectorize(erf, otypes=[np.float64])(x)


def gelu(x):
    arg = x / np.sqrt(2.0)
    try:
        from scipy.special import erf as _sci_erf
        return 0.5 * x * (1.0 + _sci_erf(arg))
    except ImportError:
        return 0.5 * x * (1.0 + _erf(arg))


def relu(x):
    return np.maximum(x, 0.0)


def softmax(x, axis=-1):
    x = x - np.max(x, axis=axis, keepdims=True)
    e = np.exp(x)
    return e / np.sum(e, axis=axis, keepdims=True)


def conv1d_same(x, weight, bias=None, pad=None):
    """x: (B, C, T)，weight: (O, C, K)，stride=1，padding 默认 K//2（保持长度）。"""
    B, C, T = x.shape
    O, _, K = weight.shape
    if pad is None:
        pad = K // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    out = np.zeros((B, O, T), dtype=np.float64)
    for k in range(K):
        out += np.einsum("oc,bct->bot", weight[:, :, k], xp[:, :, k:k + T],
                         optimize=True)
    if bias is not None:
        out += bias.reshape(1, -1, 1)
    return out


def conv1d_depthwise_same(x, weight, bias=None):
    """groups=C 的深度卷积。weight: (C, 1, K)。"""
    B, C, T = x.shape
    K = weight.shape[-1]
    pad = K // 2
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    out = np.zeros((B, C, T), dtype=np.float64)
    for k in range(K):
        out += weight[:, 0, k].reshape(1, C, 1) * xp[:, :, k:k + T]
    if bias is not None:
        out += bias.reshape(1, -1, 1)
    return out


def maxpool1d2(x):
    B, C, T = x.shape
    T2 = T // 2
    return x[:, :, :T2 * 2].reshape(B, C, T2, 2).max(axis=-1)


class NumpyCNNTransformer:
    """与 model.CNNTransformer 数值等价的前向实现。"""

    def __init__(self, npz_path):
        with np.load(npz_path, allow_pickle=False) as z:
            self.w = {k: np.asarray(z[k]) for k in z.files}
        self.cfg = {}
        if "__config__" in self.w:
            self.cfg = json.loads(str(self.w.pop("__config__")))
        self.n_layers = len({k.split(".")[1] for k in self.w
                             if k.startswith("encoder.")})
        self.multi_task = any(k.startswith("head_angles") for k in self.w)

    # ---------- 基础算子 ----------
    def _bn(self, x, p):
        g = self.w[f"{p}.weight"].reshape(1, -1, 1)
        b = self.w[f"{p}.bias"].reshape(1, -1, 1)
        m = self.w[f"{p}.running_mean"].reshape(1, -1, 1)
        v = self.w[f"{p}.running_var"].reshape(1, -1, 1)
        return (x - m) / np.sqrt(v + 1e-5) * g + b

    def _ln(self, x, p, eps=1e-5):
        mu = x.mean(axis=-1, keepdims=True)
        var = x.var(axis=-1, keepdims=True)
        h = (x - mu) / np.sqrt(var + eps)
        return h * self.w[f"{p}.weight"] + self.w[f"{p}.bias"]

    def _attention(self, x, p, n_heads):
        B, T, D = x.shape
        qkv = x @ self.w[f"{p}.in_proj_weight"].T + self.w[f"{p}.in_proj_bias"]
        q, k, v = np.split(qkv, 3, axis=-1)
        dh = D // n_heads
        def split(a):
            return a.reshape(B, T, n_heads, dh).transpose(0, 2, 1, 3)
        q, k, v = split(q), split(k), split(v)
        scores = q @ k.transpose(0, 1, 3, 2) / np.sqrt(dh)
        attn = softmax(scores, axis=-1)
        out = attn @ v                                  # (B, H, T, dh)
        out = out.transpose(0, 2, 1, 3).reshape(B, T, D)
        return out @ self.w[f"{p}.out_proj.weight"].T + self.w[f"{p}.out_proj.bias"]

    def _ffn(self, x, p):
        h = x.transpose(0, 2, 1)                        # (B, D, T)
        h = gelu(conv1d_depthwise_same(h, self.w[f"{p}.ffn.depthwise.weight"],
                                       self.w[f"{p}.ffn.depthwise.bias"]))
        h = gelu(conv1d_same(h, self.w[f"{p}.ffn.pointwise1.weight"],
                             self.w[f"{p}.ffn.pointwise1.bias"]))
        h = conv1d_same(h, self.w[f"{p}.ffn.pointwise2.weight"],
                        self.w[f"{p}.ffn.pointwise2.bias"])
        return h.transpose(0, 2, 1)

    # ---------- 整体前向 ----------
    def features(self, x):
        """x: (B, C, T) float -> (B, T//4, D)。"""
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 2:
            x = x[None, ...]
        h = conv1d_same(x, self.w["conv1.weight"])
        h = relu(self._bn(h, "bn1"))
        h = maxpool1d2(h)
        h = conv1d_same(h, self.w["conv2.weight"])
        h = relu(self._bn(h, "bn2"))
        h = maxpool1d2(h)
        h = h.transpose(0, 2, 1)
        if "pos_enc" in self.w:
            pe = self.w["pos_enc"]
            h = h + pe[:, :h.shape[1]]
        for i in range(self.n_layers):
            p = f"encoder.{i}"
            h = h + self._attention(self._ln(h, f"{p}.norm1"), f"{p}.attn",
                                    int(self.cfg.get("n_heads", 8)))
            h = h + self._ffn(self._ln(h, f"{p}.norm2"), p)
        return self._ln(h, "norm")

    def forward(self, x):
        h = self.features(x)
        z = h.mean(axis=1)
        logits = z @ self.w["head_gesture.weight"].T + self.w["head_gesture.bias"]
        out = {"logits": logits}
        if self.multi_task:
            out["angles"] = z @ self.w["head_angles.weight"].T + self.w["head_angles.bias"]
        return out

    def predict_proba(self, x):
        return softmax(self.forward(x)["logits"], axis=-1)


def export_weights(state_dict, out_path, extra=None):
    """torch state_dict -> npz（浮点存 float32，插件里计算会自动升到 float64）。"""
    payload = {k: v.detach().cpu().numpy().astype(np.float32)
               for k, v in state_dict.items()}
    if extra:
        for k, v in extra.items():
            payload[k] = v if isinstance(v, np.ndarray) else np.asarray(v)
    np.savez(out_path, **payload)
    return out_path

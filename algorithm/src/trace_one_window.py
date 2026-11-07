# -*- coding: utf-8 -*-
"""把「一次推理」从头到尾打印出来，看看到底有没有黑箱。

流程：从缓存里取一个真实的测试窗口（8 通道 × 200 点，标签 = 握拳），
一步一步算下去，每一层打印形状和数值；并且**手算一个卷积**去和程序结果对照。

只依赖 numpy（和上位机插件用的是同一个引擎），不需要 torch。

用法：python trace_one_window.py
"""
import importlib.util
import sys
from pathlib import Path

import numpy as np

from config import APP_GESTURE_NAMES, Config, PLUGIN_DIR
from dataset import build_index, prepare_subject
from preprocess import causal_stream_filter, rolling_zscore

np.set_printoptions(precision=4, suppress=True, linewidth=140)


def load_engine() -> object:
    path = PLUGIN_DIR / "推理引擎_纯NumPy.py"
    spec = importlib.util.spec_from_file_location("nina_cnn_numpy", str(path))
    mod = importlib.util.module_from_spec(spec)
    sys.modules["nina_cnn_numpy"] = mod
    spec.loader.exec_module(mod)
    return mod


def main():
    eng = load_engine()
    model = eng.NumpyCNNTransformer(str(PLUGIN_DIR / "模型权重_手势.npz"))

    # ---- 取一个真实窗口：受试者 1 的测试集里，标签为「握拳(1)」的第一个窗口 ----
    cfg = Config()
    sig = prepare_subject(1, cfg)
    _, _, test_pos = build_index([sig], cfg)
    W = cfg.window.length
    # test_pos 的每一行是 (受试者下标, 窗口起始点)，注意别把 0 当成起始点
    st = next(int(st) for _, st in test_pos
              if int(sig.labels[int(st) + W - 1]) == 6)      # NinaPro 6 号动作 = 握拳
    x = np.ascontiguousarray(sig.emg[st:st + W].T, dtype=np.float64)   # (8, 200)
    print("=" * 78)
    print(f"输入窗口：受试者 1，第 {st}~{st + W} 个采样点（100 ms），真实标签 = 握拳")
    print(f"形状 (通道, 时刻) = {x.shape}，也就是 {x.size} 个数字")
    print(f"每个通道前 6 个数（已标准化，0 = 该通道均值）：\n{x[:, :6]}")

    # ---- 第 1 层卷积 ----
    h = eng.conv1d_same(x[None], model.w["conv1.weight"])
    print("\n" + "=" * 78)
    print(f"[卷积 1] Conv1d(8→32, 核长 5)：{x.shape} -> {h.shape}，参数是 32×8×5 = 1280 个数")
    W1 = model.w["conv1.weight"]                     # (32, 8, 5)
    o, t = 0, 100                                    # 以「第 0 号核、第 100 个时刻」为例手算
    kern, seg = W1[o, 0, :], x[0, t - 2:t + 3]       # padding=2，所以取 t-2..t+2
    manual_one = float(np.dot(kern, seg))
    # 关键：卷积是"对所有输入通道求和"，所以要把 8 个通道各算一次再相加
    manual_all = float(sum(np.dot(W1[o, c, :], x[c, t - 2:t + 3])
                           for c in range(x.shape[0])))
    print(f"  手算第 0 号卷积核（位置 t={t}）：")
    print(f"    只算第 0 通道：核 {kern} × 输入 {seg} = {manual_one:.6f}")
    print(f"    8 个通道逐通道算完再相加        = {manual_all:.6f}")
    print(f"    程序算出来的同一个位置          = {float(h[0, o, t]):.6f}"
          f"   （差 {abs(manual_all - float(h[0, o, t])):.2e}）")
    print("  → 一个卷积核 = 5 个权重和 5 个输入逐项相乘再相加，再把 8 个通道的结果加总。")
    print("    32 个核同时对 8 个通道做这件事，就得到 32×200 个数。")

    # ---- 后面的层，逐步打印 ----
    h = eng.relu(model._bn(h, "bn1"))
    h = eng.maxpool1d2(h)
    print(f"\n[批归一化+ReLU+最大池化] {h.shape}（长度 200 → 100）")
    h = eng.conv1d_same(h, model.w["conv2.weight"])
    h = eng.relu(model._bn(h, "bn2"))
    h = eng.maxpool1d2(h)
    print(f"[卷积 2 + 同样处理] {h.shape}（32→64 通道，长度 100 → 50）")

    h = h.transpose(0, 2, 1)                                   # (1, 50, 64)
    h = h + model.w["pos_enc"][:, :h.shape[1]]
    print(f"[位置编码] {h.shape}（给 50 个时刻编号，让注意力知道先后）")
    for i in range(model.n_layers):
        p = f"encoder.{i}"
        a = model._attention(model._ln(h, f"{p}.norm1"), f"{p}.attn",
                             int(model.cfg.get("n_heads", 8)))
        h = h + a
        h = h + model._ffn(model._ln(h, f"{p}.norm2"), p)
        print(f"[Transformer 第 {i} 层] {h.shape}"
              f"（自注意力矩阵 50×50，看每个时刻该参考哪些时刻）")
    h = model._ln(h, "norm")

    z = h.mean(axis=1)                                          # 全局平均池化
    print(f"[全局平均池化] {h.shape} -> {z.shape}（50 个时刻取平均，得到 64 个数）")
    logits = z @ model.w["head_gesture.weight"].T + model.w["head_gesture.bias"]
    prob = eng.softmax(logits, axis=-1)[0]
    print(f"[输出层] 64 -> {logits.shape[1]} 个分数，再 Softmax 变成概率：")
    for i, p in enumerate(prob):
        print(f"    {APP_GESTURE_NAMES.get(i, i):<6} {p:6.2%}")
    print(f"  最终判断：{APP_GESTURE_NAMES[int(np.argmax(prob))]}（真值：握拳）")
    print("=" * 78)
    print("全过程只有：乘法、加法、取最大、指数、平均。没有随机数，没有查表。")


if __name__ == "__main__":
    main()

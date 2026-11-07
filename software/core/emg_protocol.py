"""
Node 御识 · 固件 <-> 上位机 EMG 二进制流协议

与 ESP32 固件 (硬件固件/Node_8CH_ADS1298/Node_8CH_ADS1298.ino) 保持一致。

帧格式（每采样点一帧）:
    [0xAA][0x55][seq][CH1..CH8 数据][xor]
    - seq     : 1 字节，0..255 循环
    - CHn 数据: 24bit 模式每通道 3 字节大端有符号; 16bit 模式 2 字节大端有符号
    - xor     : 除最后一字节外所有字节的异或校验

帧长度:  24bit 模式 = 2 + 1 + 8*3 + 1 = 28 字节
         16bit 模式 = 2 + 1 + 8*2 + 1 = 20 字节

传输层:
    - 固件把多帧拼接成一条 BLE notify，不跨 notify 拆帧。
    - 上位机按字节流解析，校验失败时逐字节滑动重同步，
      因此偶尔丢包/截断不会导致整条流瘫痪。
    - 采样配置由固件在开启前以文本形式上报:
      EMG_CFG:<采样率Hz>,<bit数>,<通道数>

本模块只依赖 numpy，可在无 GUI / 无 BLE 环境下自测。
"""
import re

import numpy as np

EMG_MAGIC = b"\xAA\x55"
EMG_CFG_RE = re.compile(r"^EMG_CFG:\s*(\d+)\s*,\s*(\d+)\s*,\s*(\d+)")

CHANNELS = 8
BITS_24 = 24
BITS_16 = 16
RAW_FULL_SCALE = 1 << 23  # 归一化显示时使用的满量程


def frame_length(bits: int) -> int:
    """返回指定 bit 模式下一帧的字节数"""
    if bits == BITS_24:
        return 2 + 1 + CHANNELS * 3 + 1
    if bits == BITS_16:
        return 2 + 1 + CHANNELS * 2 + 1
    raise ValueError(f"不支持的位深: {bits}")


def _xor_checksum(payload: bytes) -> int:
    x = 0
    for b in payload:
        x ^= b
    return x


def build_frame(samples, seq: int, bits: int = BITS_24) -> bytes:
    """
    把 8 通道采样点编码成一帧。
    samples: 长度为 8 的可迭代对象，值为 int（24bit 原始码值）。
    """
    if bits == BITS_24:
        raw = [int(v) & 0xFFFFFF for v in samples]
        data = b"".join(v.to_bytes(3, "big", signed=False) for v in raw)
    elif bits == BITS_16:
        raw = [(int(v) >> 8) & 0xFFFF for v in samples]
        data = b"".join(v.to_bytes(2, "big", signed=False) for v in raw)
    else:
        raise ValueError(f"不支持的位深: {bits}")

    head = bytes([0xAA, 0x55, seq & 0xFF])
    body = head + data
    return body + bytes([_xor_checksum(body)])


def decode_frame(frame: bytes, bits: int = BITS_24):
    """
    解码单帧。返回 int32 (8,) 原始码值数组（16bit 模式会左移 8 位对齐到 24bit 码值域），
    校验失败或长度不对返回 None。
    """
    need = frame_length(bits)
    if len(frame) != need or frame[:2] != EMG_MAGIC:
        return None
    if _xor_checksum(frame[:-1]) != frame[-1]:
        return None

    values = np.empty(CHANNELS, dtype=np.int32)
    if bits == BITS_24:
        for i in range(CHANNELS):
            values[i] = int.from_bytes(frame[3 + i * 3: 3 + (i + 1) * 3],
                                       "big", signed=True)
    elif bits == BITS_16:
        for i in range(CHANNELS):
            v = int.from_bytes(frame[3 + i * 2: 3 + (i + 1) * 2],
                               "big", signed=True)
            values[i] = v << 8  # 低 8 位是降采样时丢弃的，不参与码值对齐
    else:
        raise ValueError(f"不支持的位深: {bits}")
    return values


def normalize_raw(raw) -> np.ndarray:
    """24bit 码值 -> float32，范围约 ±1.0（满量程 = ±1）"""
    return np.asarray(raw, dtype=np.float32) / float(RAW_FULL_SCALE)


def parse_cfg_line(text: str):
    """解析固件上报的 EMG_CFG:1000,24,8，返回 (sps, bits, channels) 或 None"""
    m = EMG_CFG_RE.match(text.strip())
    if not m:
        return None
    return int(m.group(1)), int(m.group(2)), int(m.group(3))


class EmgStreamParser:
    """字节流 -> 采样帧。支持丢包/截断后的逐字节滑动重同步。"""

    def __init__(self, bits: int = BITS_24):
        self.bits = bits
        self.buffer = bytearray()

    @property
    def flen(self) -> int:
        return frame_length(self.bits)

    def reset(self, bits: int = None):
        if bits is not None:
            self.bits = bits
        self.buffer.clear()

    def feed(self, data: bytes):
        """喂入一段 notify 数据，返回解码出的 (N,8) 帧列表"""
        self.buffer.extend(data)
        out = []
        while len(self.buffer) >= self.flen:
            candidate = bytes(self.buffer[:self.flen])
            decoded = decode_frame(candidate, self.bits)
            if decoded is None:
                # 帧头或校验不符：丢弃 1 字节继续找同步
                del self.buffer[0]
                continue
            del self.buffer[:self.flen]
            out.append(decoded)
        return out


def self_test() -> bool:
    """编码<->解析往返 + 损坏重同步自测，check.py 调用"""
    rng = np.random.default_rng(20260906)
    seq = 0
    ok = True

    for bits in (BITS_24, BITS_16):
        flen = frame_length(bits)
        parser = EmgStreamParser(bits)
        payload = bytearray()
        n_frames = 100
        original = []
        for _ in range(n_frames):
            samples = rng.integers(-(1 << 22), (1 << 22), size=8)
            payload.extend(build_frame(samples, seq, bits))
            original.append(samples)
            seq = (seq + 1) & 0xFF

        # 正常流
        decoded = parser.feed(bytes(payload))
        if len(decoded) != n_frames:
            print(f"  [错误] {bits}bit 正常流帧数不符: {len(decoded)} != {n_frames}")
            ok = False
        else:
            expected_first = original[0] if bits == BITS_24 else \
                ((np.asarray(original[0]) >> 8) << 8)
            expected_last = original[-1] if bits == BITS_24 else \
                ((np.asarray(original[-1]) >> 8) << 8)
            if not np.array_equal(decoded[0], expected_first) or \
                    not np.array_equal(decoded[-1], expected_last):
                print(f"  [错误] {bits}bit 正常流首尾帧内容不符")
                ok = False

        # 对齐地丢掉一整帧，模拟 BLE 丢包（后续帧仍对齐）
        cut_at = 3 * flen
        dropped = bytes(payload[:cut_at]) + bytes(payload[cut_at + flen:])
        decoded2 = parser.feed(dropped)
        if len(decoded2) != n_frames - 1:
            print(f"  [错误] {bits}bit 丢包后帧数异常: {len(decoded2)}")
            ok = False
        else:
            expected_last = original[-1] if bits == BITS_24 else \
                ((np.asarray(original[-1]) >> 8) << 8)
            if not np.array_equal(decoded2[-1], expected_last):
                print(f"  [错误] {bits}bit 丢包后尾帧内容不符")
                ok = False

        # 通知尾部截断（半个帧），下一段从完整帧边界开始，应能恢复
        parser.reset(bits)
        torn = bytes(payload[:flen + 7])
        decoded3 = parser.feed(torn)
        decoded3.extend(parser.feed(bytes(payload[2 * flen:])))
        if len(decoded3) != n_frames - 1:
            print(f"  [错误] {bits}bit 尾部截断后帧数异常: {len(decoded3)}")
            ok = False

    # 配置行解析
    cfg = parse_cfg_line("EMG_CFG:1000,24,8")
    if cfg != (1000, 24, 8):
        print(f"  [错误] EMG_CFG 解析失败: {cfg}")
        ok = False
    if parse_cfg_line("EMG_STOPPED") is not None:
        print("  [错误] 非配置文本被误判为 EMG_CFG")
        ok = False
    return ok


if __name__ == "__main__":
    import sys
    try:
        sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    except Exception:
        pass
    print("EMG 协议自检:", "通过" if self_test() else "失败")

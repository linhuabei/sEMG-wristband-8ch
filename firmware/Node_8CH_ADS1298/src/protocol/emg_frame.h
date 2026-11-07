#pragma once

#include <stddef.h>
#include <stdint.h>

namespace emg {

// 协议层不直接依赖 BLE：发送函数由应用层注入，方便在 PC 上做单元测试。
using Sender = bool (*)(const uint8_t *data, size_t len);

void setSender(Sender sender);

void configure(bool bits24, uint8_t framesPerNotify);
void reset();       // 开始采集时清空序号和统计
void resetBatch();  // 停止采集时丢掉未发完的半包（保留累计统计）

void append(const int32_t ch[8]);
void flush();

size_t frameLength();
bool bits24();
uint8_t framesPerNotify();

uint32_t frameCount();
uint32_t packetCount();
uint32_t failCount();

}  // namespace emg

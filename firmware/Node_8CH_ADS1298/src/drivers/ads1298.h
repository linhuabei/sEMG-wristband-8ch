#pragma once

#include <Arduino.h>

#include "../app/config.h"

namespace ads {

// 初始化 GPIO / SPI / 寄存器，返回读到的 ID（正常应为 0x92）。
uint8_t initAll();

// RDATAC 模式下读一帧 8 通道 24bit 数据。
bool readChannels(int32_t ch[EMG_CHANNELS]);

// 按当前采样率 / 增益 / 测试信号状态重写寄存器。
void configure();

void startConversion();
void stopConversion();

void setSampleRate(uint16_t sps);
uint16_t sampleRate();

// 返回 false 表示增益不在 1/2/3/4/6/8/12 里。
bool setGainByValue(int gain);
uint8_t gainValue();

void setTestSignal(bool on);
bool testSignal();

uint8_t readRegister(uint8_t reg);
void writeRegister(uint8_t reg, uint8_t val);

// 诊断命令；streaming 只用于在 ADS_DBG 里回报当前是否在采集。
void debugReport(bool streaming);
void spiBusReport();
void spiTest();

}  // namespace ads

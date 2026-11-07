#pragma once

#include <Arduino.h>

namespace imu {

// 自动在 Wire（GP01/GP03）和 Wire1（GP04/GP05）上探测 LSM6DSR，
// 两个地址 0x6A / 0x6B 都试。找到后配置 416Hz / ±4g / ±500dps。
bool begin();

bool readRaw(int16_t raw[6]);

bool present();
uint8_t address();
uint8_t busId();

}  // namespace imu

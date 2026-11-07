#pragma once

namespace diag {

// READ 命令里对“特殊引脚”的语义读取：状态灯、IMU 在线、扩展 I2C、MISO。
int readSpecialPin(int pin);

// 扫描 GP01/GP03 和 GP04/GP05 两条 I2C 总线。
void scanI2C();

}  // namespace diag

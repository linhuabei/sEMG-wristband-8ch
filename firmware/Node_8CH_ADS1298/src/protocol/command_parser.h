#pragma once

#include <Arduino.h>

namespace cmd {

// 处理上位机通过 BLE 写入的一行文本命令。
void handle(const String &command);

}  // namespace cmd

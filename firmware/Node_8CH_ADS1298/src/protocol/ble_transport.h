#pragma once

#include <Arduino.h>

namespace ble {

using CommandHandler = void (*)(const String &cmd);
using ConnectionHandler = void (*)(bool connected);

void setHandlers(CommandHandler commandHandler,
                 ConnectionHandler connectionHandler);

void begin();

bool isConnected();
bool notify(const uint8_t *data, size_t len);
void sendText(const String &msg);

// 当前连接协商到的 ATT MTU；没连接时返回 23。
uint16_t peerMtu();

}  // namespace ble

#pragma once

#include <Arduino.h>

namespace app {

void begin();
void loop();

bool isStreaming();
void startStream();
void stopStream();

void onBleCommand(const String &command);
void onBleConnection(bool connected);

}  // namespace app

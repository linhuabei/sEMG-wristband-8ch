#pragma once

#include <Arduino.h>

namespace led {

enum Mode : uint8_t {
  Boot = 0,
  Idle = 1,
  Connected = 2,
  Streaming = 3,
  Manual = 4,
};

void begin();
void setMode(Mode mode);
void setManual(bool on);
bool manualOn();
void update();
int logicalState();
Mode mode();

}  // namespace led

#include "diagnostics.h"

#include <Arduino.h>
#include <Wire.h>

#include "../app/config.h"
#include "../drivers/imu_lsm6dsr.h"
#include "../drivers/status_led.h"
#include "../protocol/ble_transport.h"

namespace diag {
namespace {

bool i2cAnyDevice(TwoWire *bus) {
  for (uint8_t addr = 0x08; addr < 0x78; addr++) {
    bus->beginTransmission(addr);
    if (bus->endTransmission() == 0) return true;
  }
  return false;
}

String scanI2CBus(TwoWire *bus, const char *tag) {
  String devices = tag;
  int count = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    bus->beginTransmission(addr);
    if (bus->endTransmission() == 0) {
      if (count > 0) devices += ",";
      devices += "0x" + String(addr, HEX);
      count++;
    }
  }
  if (count == 0) devices += "none";
  return devices;
}

}  // namespace

int readSpecialPin(int pin) {
  if (pin == LED_PIN) return led::logicalState();
  if (pin == IMU_SDA || pin == IMU_SCL) return imu::present() ? 1 : 0;
  if (pin == I2C2_SCL || pin == I2C2_SDA) {
    Wire1.begin(I2C2_SDA, I2C2_SCL);
    return i2cAnyDevice(&Wire1) ? 1 : 0;
  }
  if (pin == SPI_MISO) return digitalRead(pin);
  return -1;
}

void scanI2C() {
  Wire.begin(IMU_SDA, IMU_SCL);
  Wire1.begin(I2C2_SDA, I2C2_SCL);
  ble::sendText(scanI2CBus(&Wire, "I2C_DEVICES:"));
  ble::sendText(scanI2CBus(&Wire1, "I2C2_DEVICES:"));
}

}  // namespace diag

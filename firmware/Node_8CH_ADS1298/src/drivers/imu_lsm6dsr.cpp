#include "imu_lsm6dsr.h"

#include <Wire.h>

#include "../app/config.h"

namespace imu {
namespace {

TwoWire *s_bus = &Wire;
uint8_t s_addr = 0;
uint8_t s_bus_id = 0;

bool writeReg(uint8_t reg, uint8_t val) {
  s_bus->beginTransmission(s_addr);
  s_bus->write(reg);
  s_bus->write(val);
  return s_bus->endTransmission() == 0;
}

bool readReg(uint8_t reg, uint8_t *val) {
  s_bus->beginTransmission(s_addr);
  s_bus->write(reg);
  if (s_bus->endTransmission(false) != 0) return false;
  if (s_bus->requestFrom((int)s_addr, 1) != 1) return false;
  *val = s_bus->read();
  return true;
}

}  // namespace

bool begin() {
  uint8_t id = 0;
  s_addr = 0;

  for (int bus = 0; bus < 2 && !s_addr; bus++) {
    s_bus = (bus == 0) ? &Wire : &Wire1;
    s_bus_id = (uint8_t)bus;
    for (int i = 0; i < 2; i++) {
      s_addr = (i == 0) ? LSM_ADDR_A : LSM_ADDR_B;
      if (readReg(LSM_WHO_AM_I, &id) && id == 0x6A) break;
      s_addr = 0;
    }
  }

  if (!s_addr) {
    s_bus = &Wire;
    s_bus_id = 0;
    Serial.println("[IMU] LSM6DSR not found on either bus (GP01/GP03 or GP04/GP05)");
    return false;
  }

  writeReg(LSM_CTRL3_C, 0x44);   // BDU + 地址自增
  writeReg(LSM_CTRL1_XL, 0x64);  // 416Hz ±4g
  writeReg(LSM_CTRL2_G, 0x64);   // 416Hz ±500dps

  Serial.printf("[IMU] LSM6DSR ok, addr=0x%02X, bus%d=%s, 416Hz ±4g / ±500dps\n",
                s_addr, s_bus_id,
                s_bus_id == 0 ? "GP01/GP03" : "GP04/GP05");
  return true;
}

bool readRaw(int16_t raw[6]) {
  if (!s_addr) return false;

  uint8_t buf[6];

  s_bus->beginTransmission(s_addr);
  s_bus->write(LSM_OUTX_L_G);
  if (s_bus->endTransmission(false) != 0) return false;
  if (s_bus->requestFrom((int)s_addr, 6) != 6) return false;
  for (int i = 0; i < 6; i++) buf[i] = s_bus->read();
  for (int i = 0; i < 3; i++) {
    raw[i] = (int16_t)((buf[i * 2 + 1] << 8) | buf[i * 2]);
  }

  s_bus->beginTransmission(s_addr);
  s_bus->write(LSM_OUTX_L_XL);
  if (s_bus->endTransmission(false) != 0) return false;
  if (s_bus->requestFrom((int)s_addr, 6) != 6) return false;
  for (int i = 0; i < 6; i++) buf[i] = s_bus->read();
  for (int i = 0; i < 3; i++) {
    raw[3 + i] = (int16_t)((buf[i * 2 + 1] << 8) | buf[i * 2]);
  }
  return true;
}

bool present() { return s_addr != 0; }
uint8_t address() { return s_addr; }
uint8_t busId() { return s_bus_id; }

}  // namespace imu

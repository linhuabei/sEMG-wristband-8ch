#include "command_parser.h"

#include "../app/app.h"
#include "../app/config.h"
#include "../diagnostics/diagnostics.h"
#include "../drivers/ads1298.h"
#include "../drivers/imu_lsm6dsr.h"
#include "../drivers/status_led.h"
#include "ble_transport.h"
#include "emg_frame.h"

namespace cmd {
namespace {

bool validGain(int gain) {
  return gain == 1 || gain == 2 || gain == 3 || gain == 4 || gain == 6 ||
         gain == 8 || gain == 12;
}

void handlePin(const String &cmd) {
  int pin_pos = cmd.indexOf('=');
  if (pin_pos <= 3) {
    ble::sendText("ERROR: bad format (PIN8=1)");
    return;
  }

  int pin = cmd.substring(3, pin_pos).toInt();
  int value = cmd.substring(pin_pos + 1).toInt();

  if (pin < 1 || pin > 48) {
    ble::sendText("ERROR: invalid pin");
    return;
  }
  if (pin == IMU_SDA || pin == IMU_SCL || pin == I2C2_SCL ||
      pin == I2C2_SDA || pin == SPI_MISO || pin == ADS_DRDY) {
    ble::sendText("ERROR: pin " + String(pin) + " read-only");
    return;
  }

  if (pin == LED_PIN) {
    led::setManual(value != 0);
  } else {
    pinMode(pin, OUTPUT);
    digitalWrite(pin, value ? HIGH : LOW);
  }
  ble::sendText("OK: PIN" + String(pin) + "=" + String(value));
}

void handleRead(const String &cmd) {
  int pin = cmd.substring(4).toInt();
  if (pin < 0 || pin > 48) {
    ble::sendText("ERROR: invalid pin");
    return;
  }

  if (pin == IMU_SDA || pin == IMU_SCL || pin == I2C2_SCL ||
      pin == I2C2_SDA || pin == SPI_MISO || pin == LED_PIN) {
    ble::sendText("READ" + String(pin) + "=" + String(diag::readSpecialPin(pin)));
  } else if (pin == ADS_RESET || pin == ADS_CS || pin == ADS_START ||
             pin == ADS_PWDN) {
    // 固件自己在驱动的 ADS 控制线，直接读，不改输入模式。
    ble::sendText("READ" + String(pin) + "=" + String(digitalRead(pin)));
  } else {
    pinMode(pin, INPUT_PULLUP);
    ble::sendText("READ" + String(pin) + "=" + String(digitalRead(pin)));
  }
}

void handleAll() {
  String response = "STATUS:";
  response += "LED=" + String(led::logicalState()) + ",";
  response += "LEDMODE=" + String((int)led::mode()) + ",";
  response += "DRDY=" + String(digitalRead(ADS_DRDY)) + ",";
  response += "RESET=" + String(digitalRead(ADS_RESET)) + ",";
  response += "START=" + String(digitalRead(ADS_START)) + ",";
  response += "PWDN=" + String(digitalRead(ADS_PWDN)) + ",";
  response += "EMG=" + String(app::isStreaming() ? 1 : 0) + ",";
  response += "TEST=" + String(ads::testSignal() ? 1 : 0) + ",";
  response += "GAIN=" + String((int)ads::gainValue()) + ",";
  response += "FRAMES=" + String(emg::frameCount()) + ",";
  response += "TXPKT=" + String(emg::packetCount()) + ",";
  response += "TXFAIL=" + String(emg::failCount());
  ble::sendText(response);
}

void handleImu() {
  if (!imu::present()) imu::begin();
  if (!imu::present()) {
    ble::sendText("ERROR: IMU not detected");
    return;
  }

  int16_t raw[6];
  if (!imu::readRaw(raw)) {
    ble::sendText("ERROR: IMU read failed");
    return;
  }

  float ax = raw[3] * (4.0f / 32768.0f);
  float ay = raw[4] * (4.0f / 32768.0f);
  float az = raw[5] * (4.0f / 32768.0f);
  float gx = raw[0] * (500.0f / 32768.0f);
  float gy = raw[1] * (500.0f / 32768.0f);
  float gz = raw[2] * (500.0f / 32768.0f);

  String r = "IMU:";
  r += "AX=" + String(ax, 3) + ",";
  r += "AY=" + String(ay, 3) + ",";
  r += "AZ=" + String(az, 3) + ",";
  r += "GX=" + String(gx, 2) + ",";
  r += "GY=" + String(gy, 2) + ",";
  r += "GZ=" + String(gz, 2);
  r += ",BUS" + String((int)imu::busId()) + "=" +
       (imu::busId() == 0 ? "GP01/GP03" : "GP04/GP05");
  ble::sendText(r);
}

void handleHelp() {
  String help = "Commands:\n";
  help += "PIN<n>=<v>  set pin output (e.g. PIN8=1)\n";
  help += "READ<n>     read pin level\n";
  help += "ALL         read status\n";
  help += "EMG_START   start ADS1298 streaming\n";
  help += "EMG_STOP    stop streaming\n";
  help += "EMG_SPS=n   set sample rate 1000 or 2000\n";
  help += "ADS_INIT    re-init ADS1298\n";
  help += "ADS_DBG     read ID/regs + count DRDY edges (is it converting?)\n";
  help += "ADS_GAIN=n  set PGA gain 1/2/3/4/6/8/12\n";
  help += "ADS_TEST=n  internal test signal on(1)/off(0)\n";
  help += "IMU         read LSM6DSR accel/gyro once (+ which I2C bus)\n";
  help += "I2C_SCAN    scan both I2C buses (GP01/03 + GP04/05)\n";
  help += "SPI_TEST    SPI loopback test\n";
  help += "SPI_BUS     test if MISO/DRDY lines are driven (chip alive?)\n";
  help += "HELP        show help";
  ble::sendText(help);
}

}  // namespace

void handle(const String &input) {
  String cmd = input;
  cmd.trim();

  // 允许 ADS INIT 这种手滑空格；PIN/READ 后面是数字，不动。
  if (!cmd.startsWith("PIN") && !cmd.startsWith("READ") &&
      cmd.indexOf(' ') >= 0) {
    cmd.replace(' ', '_');
  }

  if (cmd.startsWith("PIN")) {
    handlePin(cmd);
  } else if (cmd.startsWith("READ")) {
    handleRead(cmd);
  } else if (cmd.startsWith("ALL")) {
    handleAll();
  } else if (cmd.startsWith("EMG_START")) {
    app::startStream();
  } else if (cmd.startsWith("EMG_STOP")) {
    app::stopStream();
    ble::sendText("EMG_STOPPED");
  } else if (cmd.startsWith("EMG_SPS=")) {
    int sps = cmd.substring(8).toInt();
    if (sps != 1000 && sps != 2000) {
      ble::sendText("ERROR: SPS must be 1000 or 2000");
      return;
    }
    bool was_streaming = app::isStreaming();
    app::stopStream();
    ads::setSampleRate((uint16_t)sps);
    if (was_streaming) app::startStream();
    ble::sendText("OK: EMG_SPS=" + String(sps));
  } else if (cmd.startsWith("ADS_INIT")) {
    app::stopStream();
    uint8_t id = ads::initAll();
    ble::sendText("OK: ADS_INIT ID=0x" + String(id, HEX));
  } else if (cmd.startsWith("ADS_DBG")) {
    ads::debugReport(app::isStreaming());
  } else if (cmd.startsWith("ADS_GAIN=")) {
    int gain = cmd.substring(9).toInt();
    if (!validGain(gain)) {
      ble::sendText("ERROR: gain must be 1/2/3/4/6/8/12");
      return;
    }
    bool was_streaming = app::isStreaming();
    app::stopStream();
    ads::setGainByValue(gain);
    if (was_streaming) app::startStream();
    ble::sendText("OK: ADS_GAIN=" + String(gain));
  } else if (cmd.startsWith("ADS_TEST=")) {
    int on = cmd.substring(9).toInt();
    bool was_streaming = app::isStreaming();
    app::stopStream();
    ads::setTestSignal(on != 0);
    if (was_streaming) app::startStream();
    if (on != 0) {
      ble::sendText("OK: ADS_TEST=1 test signal on (gain=12x, ~2Hz square wave)");
    } else {
      ble::sendText("OK: ADS_TEST=0 back to electrode input");
    }
  } else if (cmd.startsWith("IMU")) {
    handleImu();
  } else if (cmd.startsWith("I2C_SCAN")) {
    diag::scanI2C();
  } else if (cmd.startsWith("SPI_TEST")) {
    ads::spiTest();
  } else if (cmd.startsWith("SPI_BUS")) {
    ads::spiBusReport();
  } else if (cmd.startsWith("HELP")) {
    handleHelp();
  } else {
    ble::sendText("ERROR: unknown command (HELP)");
  }
}

}  // namespace cmd

#include "app.h"

#include <Wire.h>

#include "../drivers/ads1298.h"
#include "../drivers/imu_lsm6dsr.h"
#include "../drivers/status_led.h"
#include "../protocol/ble_transport.h"
#include "../protocol/command_parser.h"
#include "../protocol/emg_frame.h"
#include "config.h"

namespace app {
namespace {

bool s_streaming = false;

}  // namespace

void begin() {
  Serial.begin(115200);
  delay(50);
  Serial.println();
  Serial.println("Node 8CH sEMG firmware (modular, ADS1298)");

  led::begin();

  Wire.begin(IMU_SDA, IMU_SCL);      // GP01/GP03
  Wire1.begin(I2C2_SDA, I2C2_SCL);   // GP04/GP05
  Serial.println("[I2C] initialized (GP01/03 + GP04/05)");

  imu::begin();
  ads::initAll();

  emg::setSender(ble::notify);
  ble::setHandlers(onBleCommand, onBleConnection);
  ble::begin();

  led::setMode(led::Idle);
  Serial.println("[LED] GP08 breath led: idle=slow dim / connected=mid / streaming=fast");
}

bool isStreaming() { return s_streaming; }

void startStream() {
  if (s_streaming) {
    ble::sendText("OK: EMG already running");
    return;
  }

  const uint16_t peer_mtu = ble::peerMtu();
  const int payload = (peer_mtu > 3) ? (int)(peer_mtu - 3) : 20;
  Serial.printf("[EMG] peer MTU=%u payload=%d\n", peer_mtu, payload);

  bool bits24;
  uint8_t frames_per_notify;

  if (payload >= EMG_24BIT_FRAME) {
    bits24 = true;
    int n = payload / EMG_24BIT_FRAME;
    frames_per_notify = (uint8_t)(n > EMG_MAX_BATCH ? EMG_MAX_BATCH : n);
  } else if (payload >= EMG_16BIT_FRAME) {
    bits24 = false;
    int n = payload / EMG_16BIT_FRAME;
    frames_per_notify = (uint8_t)(n > 12 ? 12 : n);
  } else {
    ble::sendText("ERROR: MTU too small for EMG");
    return;
  }

  emg::configure(bits24, frames_per_notify);
  emg::reset();

  String cfg = "EMG_CFG:" + String(ads::sampleRate()) + "," +
               String(bits24 ? 24 : 16) + "," + String(EMG_CHANNELS);
  ble::sendText(cfg);
  delay(10);

  ads::startConversion();
  s_streaming = true;
  led::setMode(led::Streaming);
  Serial.println("[EMG] streaming started: " + cfg);
}

void stopStream() {
  s_streaming = false;
  emg::resetBatch();
  ads::stopConversion();
  led::setMode(ble::isConnected() ? led::Connected : led::Idle);
}

void onBleCommand(const String &command) { cmd::handle(command); }

void onBleConnection(bool connected) {
  if (!connected) {
    stopStream();
    led::setMode(led::Idle);
    Serial.println("[APP] BLE disconnected, streaming stopped");
    return;
  }

  led::setMode(s_streaming ? led::Streaming : led::Connected);
  Serial.println("[APP] BLE connected");
}

void loop() {
  led::update();

  if (s_streaming && ble::isConnected()) {
    if (digitalRead(ADS_DRDY) == LOW) {
      int32_t channels[EMG_CHANNELS];
      if (ads::readChannels(channels)) emg::append(channels);
    } else {
      delayMicroseconds(80);
    }
  } else {
    delay(1);
  }
}

}  // namespace app

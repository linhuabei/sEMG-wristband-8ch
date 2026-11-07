#include "ble_transport.h"

#include <BLE2902.h>
#include <BLEDevice.h>
#include <BLEServer.h>
#include <BLEUtils.h>

#include <map>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#include "../app/config.h"

namespace ble {
namespace {

BLEServer *s_server = nullptr;
BLECharacteristic *s_characteristic = nullptr;
bool s_connected = false;
SemaphoreHandle_t s_tx_mutex = nullptr;
CommandHandler s_command_handler = nullptr;
ConnectionHandler s_connection_handler = nullptr;

class ServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *) override {
    s_connected = true;
    Serial.println("[BLE] device connected");
    if (s_connection_handler != nullptr) s_connection_handler(true);
  }

  void onDisconnect(BLEServer *server) override {
    s_connected = false;
    Serial.println("[BLE] device disconnected");
    if (s_connection_handler != nullptr) s_connection_handler(false);
    server->startAdvertising();
  }
};

class CharacteristicCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *characteristic) override {
    auto value = characteristic->getValue();
    if (value.length() == 0) return;

    String command(value.c_str());
    command.trim();
    Serial.print("[CMD] ");
    Serial.println(command);
    if (s_command_handler != nullptr) s_command_handler(command);
  }
};

}  // namespace

void setHandlers(CommandHandler commandHandler,
                 ConnectionHandler connectionHandler) {
  s_command_handler = commandHandler;
  s_connection_handler = connectionHandler;
}

void begin() {
  s_tx_mutex = xSemaphoreCreateMutex();

  BLEDevice::init(DEVICE_NAME);
  esp_err_t mtu_rc = BLEDevice::setMTU(BLE_LOCAL_MTU);
  Serial.printf("[BLE] local MTU set rc=%d\n", mtu_rc);

  s_server = BLEDevice::createServer();
  s_server->setCallbacks(new ServerCallbacks());

  BLEService *service = s_server->createService(SERVICE_UUID);
  s_characteristic = service->createCharacteristic(
      CHARACTERISTIC_UUID,
      BLECharacteristic::PROPERTY_READ |
          BLECharacteristic::PROPERTY_WRITE |
          BLECharacteristic::PROPERTY_NOTIFY |
          BLECharacteristic::PROPERTY_INDICATE);

  s_characteristic->setCallbacks(new CharacteristicCallbacks());
  s_characteristic->addDescriptor(new BLE2902());
  service->start();

  BLEAdvertising *advertising = s_server->getAdvertising();
  advertising->addServiceUUID(SERVICE_UUID);
  advertising->setScanResponse(true);
  advertising->setMinPreferred(0x06);
  advertising->setMaxPreferred(0x12);
  s_server->startAdvertising();

  Serial.printf("[BLE] advertising started, name=%s\n", DEVICE_NAME);
  Serial.printf("[BLE] UUID=%s\n", SERVICE_UUID);
}

bool isConnected() { return s_connected; }

bool notify(const uint8_t *data, size_t len) {
  if (!s_connected || s_characteristic == nullptr || len == 0) return false;
  if (s_tx_mutex == nullptr) return false;
  if (xSemaphoreTake(s_tx_mutex, pdMS_TO_TICKS(20)) != pdTRUE) return false;

  // BLE 库的 setValue 参数不是 const，这里只读数据，转换是安全的。
  s_characteristic->setValue((uint8_t *)data, len);
  s_characteristic->notify();
  xSemaphoreGive(s_tx_mutex);
  return true;
}

void sendText(const String &msg) {
  if (s_connected) notify((const uint8_t *)msg.c_str(), msg.length());
  Serial.println("[TX] " + msg);
}

uint16_t peerMtu() {
  uint16_t mtu = 23;
  if (s_server == nullptr) return mtu;

  std::map<uint16_t, conn_status_t> peers = s_server->getPeerDevices(false);
  for (auto &peer : peers) {
    if (peer.second.mtu > mtu) mtu = peer.second.mtu;
  }
  return mtu;
}

}  // namespace ble

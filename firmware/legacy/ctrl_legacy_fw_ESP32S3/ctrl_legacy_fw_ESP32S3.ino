/*
  这集卡了
  Node_1 sEMG 固件
  ESP32-S3 + ADS1298 + LSM6DSR
*/

#include <BLEDevice.h>
#include <BLEUtils.h>
#include <BLEServer.h>
#include <BLE2902.h>
#include <Wire.h>
#include <SPI.h>
#include <map>

#include "freertos/FreeRTOS.h"
#include "freertos/semphr.h"

#define BLE_SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define BLE_CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// 引脚
#define IMU_SDA_PIN     1
#define IMU_SCL_PIN     3
#define OLED_SCL_PIN    4
#define OLED_SDA_PIN    5
#define SPI_SCLK_PIN    6
#define SPI_MISO_PIN    2
#define SPI_MOSI_PIN    7
#define LED_PIN         8
#define ADS_RESET_PIN   9
#define ADS_CS_PIN      10
#define ADS_DRDY_PIN    11
#define ADS_START_PIN   12
#define ADS_PWDN_PIN    13

// ADS1298 命令
#define ADS_CMD_START   0x08
#define ADS_CMD_STOP    0x0A
#define ADS_CMD_RDATAC  0x10
#define ADS_CMD_SDATAC  0x11
#define ADS_CMD_RREG    0x20
#define ADS_CMD_WREG    0x40

// ADS1298 寄存器
#define ADS_REG_ID       0x00
#define ADS_REG_CONFIG1  0x01
#define ADS_REG_CONFIG2  0x02
#define ADS_REG_CONFIG3  0x03
#define ADS_REG_CHNSET   0x05

#define ADS_CONFIG1_2000SPS 0x84
#define ADS_CONFIG1_1000SPS 0x85
#define ADS_CONFIG2_NORMAL  0x40
#define ADS_CONFIG3_INTREF  0xC0

// LSM6DSR
#define LSM_ADDR_SA0_LOW   0x6A
#define LSM_ADDR_SA0_HIGH  0x6B
#define LSM_WHO_AM_I       0x0F
#define LSM_CTRL1_XL       0x10
#define LSM_CTRL2_G        0x11
#define LSM_CTRL3_C        0x12
#define LSM_OUTX_L_G       0x22
#define LSM_OUTX_L_XL      0x28

// 采样
#define DEFAULT_EMG_RATE     2000
#define EMG_CHANNEL_COUNT    8
#define FRAME_24BIT_LEN      28
#define FRAME_16BIT_LEN      20
#define MAX_FRAMES_PER_PACKET 8

BLECharacteristic *emgCharacteristic;
BLEServer *bleServer;
bool pcConnected = false;

bool isEmgRunning = false;
bool use24BitFrame = true;
uint16_t emgSampleRate = DEFAULT_EMG_RATE;
uint8_t framesPerPacket = MAX_FRAMES_PER_PACKET;
uint8_t framesInPacket = 0;
uint8_t frameSeq = 0;
uint8_t packetBuffer[12 * FRAME_16BIT_LEN];
uint8_t gainTableIndex = 0;
uint8_t imuI2cAddr = 0;

SemaphoreHandle_t bleSendMutex = NULL;

void sendBleData(const uint8_t *data, size_t len);
void sendText(const String &msg);
void adsSendCommand(uint8_t cmd);
void adsWriteRegister(uint8_t reg, uint8_t val);
uint8_t adsReadRegister(uint8_t reg);
void adsReset();
void configureAds();
uint8_t initAds();
bool readAdsChannels(int32_t ch[EMG_CHANNEL_COUNT]);
void sendEmgPacket();
void addEmgFrameToPacket(const int32_t ch[EMG_CHANNEL_COUNT]);
void startEmg();
void stopEmg();
bool imuWriteRegister(uint8_t reg, uint8_t val);
bool imuReadRegister(uint8_t reg, uint8_t *val);
void initImu();
bool readImuRaw(int16_t raw[6]);
void scanI2c();
void testSpi();
void setupI2c();
int readSpecialPin(int pin);
void handleCommand(String cmd);

// BLE 
void sendBleData(const uint8_t *data, size_t len) {
  if (!pcConnected || !emgCharacteristic || len == 0) return;

  if (xSemaphoreTake(bleSendMutex, pdMS_TO_TICKS(20)) != pdTRUE) return;
  emgCharacteristic->setValue(data, len);
  emgCharacteristic->notify();
  xSemaphoreGive(bleSendMutex);
}

void sendText(const String &msg) {
  if (pcConnected) {
    sendBleData((const uint8_t *)msg.c_str(), msg.length());
    Serial.println("[TX] " + msg);
  }
}

// ADS1298
void adsSendCommand(uint8_t cmd) {
  digitalWrite(ADS_CS_PIN, LOW);
  SPI.transfer(cmd);
  digitalWrite(ADS_CS_PIN, HIGH);
  delayMicroseconds(2);
}

void adsWriteRegister(uint8_t reg, uint8_t val) {
  digitalWrite(ADS_CS_PIN, LOW);
  SPI.transfer(ADS_CMD_WREG | reg);
  SPI.transfer(0x00);
  SPI.transfer(val);
  digitalWrite(ADS_CS_PIN, HIGH);
  delayMicroseconds(2);
}

uint8_t adsReadRegister(uint8_t reg) {
  digitalWrite(ADS_CS_PIN, LOW);
  SPI.transfer(ADS_CMD_RREG | reg);
  SPI.transfer(0x00);
  uint8_t val = SPI.transfer(0x00);
  digitalWrite(ADS_CS_PIN, HIGH);
  delayMicroseconds(2);
  return val;
}

void adsReset() {
  digitalWrite(ADS_RESET_PIN, LOW);
  delay(10);
  digitalWrite(ADS_RESET_PIN, HIGH);
  delay(100);
}

void configureAds() {
  adsSendCommand(ADS_CMD_SDATAC);
  delayMicroseconds(10);

  adsWriteRegister(ADS_REG_CONFIG1,
                   (emgSampleRate >= 2000) ? ADS_CONFIG1_2000SPS : ADS_CONFIG1_1000SPS);
  adsWriteRegister(ADS_REG_CONFIG2, ADS_CONFIG2_NORMAL);
  adsWriteRegister(ADS_REG_CONFIG3, ADS_CONFIG3_INTREF);

  for (int i = 0; i < EMG_CHANNEL_COUNT; i++) {
    adsWriteRegister(ADS_REG_CHNSET + i, gainTableIndex << 4);
  }

  delay(150);

  adsSendCommand(ADS_CMD_RDATAC);
  delayMicroseconds(10);
}

uint8_t initAds() {
  stopEmg();

  pinMode(ADS_PWDN_PIN, OUTPUT);
  digitalWrite(ADS_PWDN_PIN, HIGH);
  delay(20);

  pinMode(ADS_RESET_PIN, OUTPUT);
  pinMode(ADS_CS_PIN, OUTPUT);
  pinMode(ADS_DRDY_PIN, INPUT_PULLUP);
  pinMode(ADS_START_PIN, OUTPUT);
  digitalWrite(ADS_CS_PIN, HIGH);
  digitalWrite(ADS_START_PIN, LOW);
  digitalWrite(ADS_RESET_PIN, HIGH);

  SPI.begin(SPI_SCLK_PIN, SPI_MISO_PIN, SPI_MOSI_PIN, ADS_CS_PIN);
  SPI.setFrequency(1000000);
  SPI.setDataMode(SPI_MODE0);

  adsReset();
  configureAds();

  uint8_t id = adsReadRegister(ADS_REG_ID);
  uint8_t c1 = adsReadRegister(ADS_REG_CONFIG1);
  Serial.print("[ADS] ID=0x");
  Serial.print(id, HEX);
  Serial.print(" CONFIG1=0x");
  Serial.println(c1, HEX);

  if (id != 0x92) {
    Serial.println("[ADS] WARNING: ID != 0x92, check wiring!");
  }
  return id;
}

bool readAdsChannels(int32_t ch[EMG_CHANNEL_COUNT]) {
  uint32_t t0 = millis();
  while (digitalRead(ADS_DRDY_PIN) != LOW) {
    if (millis() - t0 > 4) return false;
  }

  uint8_t b[27];
  digitalWrite(ADS_CS_PIN, LOW);
  for (int i = 0; i < 27; i++) {
    b[i] = SPI.transfer(0x00);
  }
  digitalWrite(ADS_CS_PIN, HIGH);

  for (int i = 0; i < EMG_CHANNEL_COUNT; i++) {
    int32_t v = ((int32_t)b[3 + i * 3] << 16) |
                ((int32_t)b[4 + i * 3] << 8) |
                (int32_t)b[5 + i * 3];
    ch[i] = (v << 8) >> 8;
  }
  return true;
}

// EMG
void sendEmgPacket() {
  if (framesInPacket == 0) return;
  size_t frameLen = use24BitFrame ? FRAME_24BIT_LEN : FRAME_16BIT_LEN;
  sendBleData(packetBuffer, framesInPacket * frameLen);
  framesInPacket = 0;
}

void addEmgFrameToPacket(const int32_t ch[EMG_CHANNEL_COUNT]) {
  size_t frameLen = use24BitFrame ? FRAME_24BIT_LEN : FRAME_16BIT_LEN;
  uint8_t *p = packetBuffer + framesInPacket * frameLen;

  p[0] = 0xAA;
  p[1] = 0x55;
  p[2] = frameSeq++;

  for (int i = 0; i < EMG_CHANNEL_COUNT; i++) {
    if (use24BitFrame) {
      p[3 + i * 3] = (uint8_t)((ch[i] >> 16) & 0xFF);
      p[4 + i * 3] = (uint8_t)((ch[i] >> 8) & 0xFF);
      p[5 + i * 3] = (uint8_t)(ch[i] & 0xFF);
    } else {
      int16_t v16 = (int16_t)(ch[i] >> 8);
      p[3 + i * 2] = (uint8_t)((v16 >> 8) & 0xFF);
      p[4 + i * 2] = (uint8_t)(v16 & 0xFF);
    }
  }

  uint8_t x = 0;
  for (size_t i = 0; i < frameLen - 1; i++) x ^= p[i];
  p[frameLen - 1] = x;

  framesInPacket++;
  if (framesInPacket >= framesPerPacket) sendEmgPacket();
}

void startEmg() {
  if (isEmgRunning) {
    sendText("OK: EMG already running");
    return;
  }

  uint16_t peerMtu = 23;
  if (bleServer) {
    std::map<uint16_t, conn_status_t> peers = bleServer->getPeerDevices(false);
    for (std::map<uint16_t, conn_status_t>::iterator it = peers.begin();
         it != peers.end(); ++it) {
      if (it->second.mtu > peerMtu) peerMtu = it->second.mtu;
    }
  }
  int payload = (peerMtu > 3) ? (int)(peerMtu - 3) : 20;
  Serial.printf("[EMG] peer MTU=%d payload=%d\n", peerMtu, payload);

  if (payload >= FRAME_24BIT_LEN) {
    use24BitFrame = true;
    int n = payload / FRAME_24BIT_LEN;
    framesPerPacket = (n > MAX_FRAMES_PER_PACKET) ? MAX_FRAMES_PER_PACKET : (uint8_t)n;
  } else if (payload >= FRAME_16BIT_LEN) {
    use24BitFrame = false;
    int n = payload / FRAME_16BIT_LEN;
    framesPerPacket = (n > 12) ? 12 : (uint8_t)n;
  } else {
    sendText("ERROR: MTU too small for EMG");
    return;
  }

  framesInPacket = 0;
  frameSeq = 0;

  String cfg = "EMG_CFG:" + String(emgSampleRate) + "," +
               String(use24BitFrame ? 24 : 16) + "," +
               String(EMG_CHANNEL_COUNT);
  sendText(cfg);
  delay(10);

  digitalWrite(ADS_START_PIN, HIGH);
  isEmgRunning = true;
  Serial.println("[EMG] streaming started: " + cfg);
}

void stopEmg() {
  isEmgRunning = false;
  framesInPacket = 0;
  digitalWrite(ADS_START_PIN, LOW);
}

// IMU
bool imuWriteRegister(uint8_t reg, uint8_t val) {
  Wire.beginTransmission(imuI2cAddr);
  Wire.write(reg);
  Wire.write(val);
  return (Wire.endTransmission() == 0);
}

bool imuReadRegister(uint8_t reg, uint8_t *val) {
  Wire.beginTransmission(imuI2cAddr);
  Wire.write(reg);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)imuI2cAddr, 1) != 1) return false;
  *val = Wire.read();
  return true;
}

void initImu() {
  uint8_t id = 0;
  for (int i = 0; i < 2; i++) {
    imuI2cAddr = (i == 0) ? LSM_ADDR_SA0_LOW : LSM_ADDR_SA0_HIGH;
    if (imuReadRegister(LSM_WHO_AM_I, &id) && id == 0x6A) break;
    imuI2cAddr = 0;
  }
  if (!imuI2cAddr) {
    Serial.println("[IMU] LSM6DSR not found (pins 1/3)");
    return;
  }

  imuWriteRegister(LSM_CTRL3_C, 0x44);
  imuWriteRegister(LSM_CTRL1_XL, 0x64);
  imuWriteRegister(LSM_CTRL2_G, 0x64);

  Serial.print("[IMU] LSM6DSR ok, addr=0x");
  Serial.print(imuI2cAddr, HEX);
  Serial.println(" 416Hz ±4g / ±500dps");
}

bool readImuRaw(int16_t raw[6]) {
  uint8_t buf[6];

  Wire.beginTransmission(imuI2cAddr);
  Wire.write(LSM_OUTX_L_G);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)imuI2cAddr, 6) != 6) return false;
  for (int i = 0; i < 6; i++) buf[i] = Wire.read();
  for (int i = 0; i < 3; i++) {
    raw[i] = (int16_t)((buf[i * 2 + 1] << 8) | buf[i * 2]);
  }

  Wire.beginTransmission(imuI2cAddr);
  Wire.write(LSM_OUTX_L_XL);
  if (Wire.endTransmission(false) != 0) return false;
  if (Wire.requestFrom((int)imuI2cAddr, 6) != 6) return false;
  for (int i = 0; i < 6; i++) buf[i] = Wire.read();
  for (int i = 0; i < 3; i++) {
    raw[3 + i] = (int16_t)((buf[i * 2 + 1] << 8) | buf[i * 2]);
  }

  return true;
}

// 调试
void scanI2c() {
  Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN);
  String devices = "I2C_DEVICES:";
  int count = 0;
  for (uint8_t addr = 1; addr < 127; addr++) {
    Wire.beginTransmission(addr);
    if (Wire.endTransmission() == 0) {
      if (count > 0) devices += ",";
      devices += "0x" + String(addr, HEX);
      count++;
    }
  }
  if (count == 0) devices += "none";
  sendText(devices);
}

void testSpi() {
  SPI.begin(SPI_SCLK_PIN, SPI_MISO_PIN, SPI_MOSI_PIN, ADS_CS_PIN);
  SPI.setFrequency(1000000);
  SPI.setDataMode(SPI_MODE0);
  pinMode(ADS_CS_PIN, OUTPUT);
  digitalWrite(ADS_CS_PIN, HIGH);

  digitalWrite(ADS_CS_PIN, LOW);
  byte testData = 0xAA;
  byte result = SPI.transfer(testData);
  digitalWrite(ADS_CS_PIN, HIGH);

  sendText("SPI_TEST: TX=0x" + String(testData, HEX) +
           " RX=0x" + String(result, HEX));
}

void setupI2c() {
  Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN);
  Wire1.begin(OLED_SDA_PIN, OLED_SCL_PIN);
  Serial.println("[I2C] initialized");
}

int readSpecialPin(int pin) {
  if (pin == IMU_SDA_PIN || pin == IMU_SCL_PIN ||
      pin == OLED_SCL_PIN || pin == OLED_SDA_PIN) {
    Wire.begin(IMU_SDA_PIN, IMU_SCL_PIN);
    Wire.beginTransmission(0x68);
    return (Wire.endTransmission() == 0) ? 1 : 0;
  }
  if (pin == SPI_MISO_PIN) {
    return digitalRead(pin);
  }
  return -1;
}

// 命令
void handleCommand(String cmd) {
  cmd.trim();

  if (cmd.startsWith("PIN")) {
    int pinPos = cmd.indexOf('=');
    if (pinPos > 3) {
      int pin = cmd.substring(3, pinPos).toInt();
      int value = cmd.substring(pinPos + 1).toInt();

      if (pin == IMU_SDA_PIN || pin == IMU_SCL_PIN ||
          pin == OLED_SCL_PIN || pin == OLED_SDA_PIN ||
          pin == SPI_MISO_PIN || pin == ADS_DRDY_PIN) {
        sendText("ERROR: pin " + String(pin) + " read-only");
        return;
      }
      pinMode(pin, OUTPUT);
      digitalWrite(pin, value ? HIGH : LOW);
      sendText("OK: PIN" + String(pin) + "=" + String(value));
    } else {
      sendText("ERROR: bad format (PIN8=1)");
    }
  }
  else if (cmd.startsWith("READ")) {
    int pin = cmd.substring(4).toInt();
    if (pin >= 0 && pin <= 48) {
      if (pin == IMU_SDA_PIN || pin == IMU_SCL_PIN ||
          pin == OLED_SCL_PIN || pin == OLED_SDA_PIN ||
          pin == SPI_MISO_PIN) {
        sendText("READ" + String(pin) + "=" + String(readSpecialPin(pin)));
      } else {
        pinMode(pin, INPUT_PULLUP);
        sendText("READ" + String(pin) + "=" + String(digitalRead(pin)));
      }
    } else {
      sendText("ERROR: invalid pin");
    }
  }
  else if (cmd.startsWith("ALL")) {
    String response = "STATUS:";
    response += "LED=" + String(digitalRead(LED_PIN)) + ",";
    response += "DRDY=" + String(digitalRead(ADS_DRDY_PIN)) + ",";
    response += "RESET=" + String(digitalRead(ADS_RESET_PIN)) + ",";
    response += "START=" + String(digitalRead(ADS_START_PIN)) + ",";
    response += "PWDN=" + String(digitalRead(ADS_PWDN_PIN)) + ",";
    response += "EMG=" + String(isEmgRunning ? 1 : 0);
    sendText(response);
  }
  else if (cmd.startsWith("EMG_START")) {
    startEmg();
  }
  else if (cmd.startsWith("EMG_STOP")) {
    stopEmg();
    sendText("EMG_STOPPED");
  }
  else if (cmd.startsWith("EMG_SPS=")) {
    int sps = cmd.substring(8).toInt();
    if (sps != 1000 && sps != 2000) {
      sendText("ERROR: SPS must be 1000 or 2000");
      return;
    }
    bool wasRunning = isEmgRunning;
    stopEmg();
    emgSampleRate = (uint16_t)sps;
    configureAds();
    if (wasRunning) startEmg();
    sendText("OK: EMG_SPS=" + String(emgSampleRate));
  }
  else if (cmd.startsWith("ADS_INIT")) {
    uint8_t id = initAds();
    sendText("OK: ADS_INIT ID=0x" + String(id, HEX));
  }
  else if (cmd.startsWith("ADS_GAIN=")) {
    int gain = cmd.substring(9).toInt();
    const uint8_t table[] = {6, 1, 2, 3, 4, 8, 12};
    int idx = -1;
    for (int i = 0; i < 7; i++) {
      if (table[i] == gain) { idx = i; break; }
    }
    if (idx < 0) {
      sendText("ERROR: gain must be 1/2/3/4/6/8/12");
      return;
    }
    bool wasRunning = isEmgRunning;
    stopEmg();
    gainTableIndex = (uint8_t)idx;
    configureAds();
    if (wasRunning) startEmg();
    sendText("OK: ADS_GAIN=" + String(gain));
  }
  else if (cmd.startsWith("IMU")) {
    if (!imuI2cAddr) initImu();
    if (!imuI2cAddr) {
      sendText("ERROR: IMU not detected");
      return;
    }
    int16_t raw[6];
    if (!readImuRaw(raw)) {
      sendText("ERROR: IMU read failed");
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
    sendText(r);
  }
  else if (cmd.startsWith("I2C_SCAN")) {
    scanI2c();
  }
  else if (cmd.startsWith("SPI_TEST")) {
    testSpi();
  }
  else if (cmd.startsWith("HELP")) {
    String help = "Commands:\n";
    help += "PIN<n>=<v>  set pin output (e.g. PIN8=1)\n";
    help += "READ<n>     read pin level\n";
    help += "ALL         read status\n";
    help += "EMG_START   start ADS1298 streaming\n";
    help += "EMG_STOP    stop streaming\n";
    help += "EMG_SPS=n   set sample rate 1000 or 2000\n";
    help += "ADS_INIT    re-init ADS1298\n";
    help += "ADS_GAIN=n  set PGA gain 1/2/3/4/6/8/12\n";
    help += "IMU         read LSM6DSR accel/gyro once\n";
    help += "I2C_SCAN    scan I2C bus\n";
    help += "SPI_TEST    SPI loopback test\n";
    help += "HELP        show help";
    sendText(help);
  }
  else {
    sendText("ERROR: unknown command (HELP)");
  }
}

// 回调
class PcServerCallbacks : public BLEServerCallbacks {
  void onConnect(BLEServer *server) {
    pcConnected = true;
    digitalWrite(LED_PIN, HIGH);
    Serial.println("[BLE] 设备已连接");
  }

  void onDisconnect(BLEServer *server) {
    pcConnected = false;
    digitalWrite(LED_PIN, LOW);
    stopEmg();
    Serial.println("[BLE] 设备已断开");
    server->startAdvertising();
  }
};

class CommandCallbacks : public BLECharacteristicCallbacks {
  void onWrite(BLECharacteristic *characteristic) {
    String value = characteristic->getValue();
    if (value.length() > 0) {
      String cmd = String(value.c_str());
      cmd.trim();
      Serial.print("[CMD] ");
      Serial.println(cmd);
      handleCommand(cmd);
    }
  }
};

// setup / loop
void setup() {
  Serial.begin(115200);
  delay(50);
  Serial.println();
  Serial.println("Node 御识 8CH sEMG firmware (ADS1298)");

  bleSendMutex = xSemaphoreCreateMutex();

  pinMode(LED_PIN, OUTPUT);
  digitalWrite(LED_PIN, LOW);

  setupI2c();
  initImu();
  initAds();

  BLEDevice::init("Node_1_by_hzx");
  esp_err_t mtuRc = BLEDevice::setMTU(247);
  Serial.printf("[BLE] local MTU set rc=%d\n", mtuRc);

  bleServer = BLEDevice::createServer();
  bleServer->setCallbacks(new PcServerCallbacks());

  BLEService *bleService = bleServer->createService(BLE_SERVICE_UUID);
  emgCharacteristic = bleService->createCharacteristic(
    BLE_CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_WRITE |
    BLECharacteristic::PROPERTY_NOTIFY |
    BLECharacteristic::PROPERTY_INDICATE
  );

  emgCharacteristic->setCallbacks(new CommandCallbacks());
  emgCharacteristic->addDescriptor(new BLE2902());
  bleService->start();

  BLEAdvertising *bleAdvertising = bleServer->getAdvertising();
  bleAdvertising->addServiceUUID(BLE_SERVICE_UUID);
  bleAdvertising->setScanResponse(true);
  bleAdvertising->setMinPreferred(0x06);
  bleAdvertising->setMaxPreferred(0x12);
  bleServer->startAdvertising();

  Serial.println("[BLE] advertising started, name=Node_1_by_hzx");
  Serial.println("[BLE] UUID=" BLE_SERVICE_UUID);
}

void loop() {
  if (isEmgRunning && pcConnected) {
    if (digitalRead(ADS_DRDY_PIN) == LOW) {
      int32_t ch[EMG_CHANNEL_COUNT];
      if (readAdsChannels(ch)) {
        addEmgFrameToPacket(ch);
      }
    } else {
      delayMicroseconds(80);
    }
  } else {
    delay(1);
  }
}
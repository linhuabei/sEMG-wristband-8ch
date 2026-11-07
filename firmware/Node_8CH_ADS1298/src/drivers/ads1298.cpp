#include "ads1298.h"

#include <SPI.h>

#include "../protocol/ble_transport.h"

namespace ads {
namespace {

uint16_t s_sps = EMG_SPS_HZ;
uint8_t s_gain_index = 0;
uint8_t s_gain_backup = 0;
bool s_test_signal = false;

const uint8_t kGainTable[7] = {6, 1, 2, 3, 4, 8, 12};

void writeCommand(uint8_t cmd) {
  digitalWrite(ADS_CS, LOW);
  SPI.transfer(cmd);
  digitalWrite(ADS_CS, HIGH);
  delayMicroseconds(2);
}

void resetPulse() {
  digitalWrite(ADS_RESET, LOW);
  delay(10);
  digitalWrite(ADS_RESET, HIGH);
  delay(100);
}

void writeRegisterRaw(uint8_t reg, uint8_t val) {
  digitalWrite(ADS_CS, LOW);
  SPI.transfer(ADS_CMD_WREG | reg);
  SPI.transfer(0x00);
  SPI.transfer(val);
  digitalWrite(ADS_CS, HIGH);
  delayMicroseconds(2);
}

uint8_t readRegisterRaw(uint8_t reg) {
  digitalWrite(ADS_CS, LOW);
  SPI.transfer(ADS_CMD_RREG | reg);
  SPI.transfer(0x00);
  uint8_t val = SPI.transfer(0x00);
  digitalWrite(ADS_CS, HIGH);
  delayMicroseconds(2);
  return val;
}

}  // namespace

void writeRegister(uint8_t reg, uint8_t val) { writeRegisterRaw(reg, val); }

uint8_t readRegister(uint8_t reg) { return readRegisterRaw(reg); }

void configure() {
  writeCommand(ADS_CMD_SDATAC);
  delayMicroseconds(10);

  // 0x84 = 2000 SPS，0x85 = 1000 SPS（高分辨率模式）。
  writeRegisterRaw(ADS_REG_CONFIG1,
                   (s_sps >= 2000) ? ADS_CONFIG1_2000SPS : ADS_CONFIG1_1000SPS);
  writeRegisterRaw(ADS_REG_CONFIG2,
                   s_test_signal ? ADS_CONFIG2_TEST : ADS_CONFIG2_NORMAL);
  writeRegisterRaw(ADS_REG_CONFIG3, ADS_CONFIG3_INTREF);

  // CHnSET：bit6:4 增益，bit2:0 输入选择（000=电极，101=内部测试信号）。
  for (int i = 0; i < EMG_CHANNELS; i++) {
    writeRegisterRaw(ADS_REG_CHNSET + i,
                     (s_gain_index << 4) | (s_test_signal ? 0x05 : 0x00));
  }

  delay(150);  // 内部基准建立时间

  writeCommand(ADS_CMD_RDATAC);
  delayMicroseconds(10);
}

uint8_t initAll() {
  digitalWrite(ADS_START, LOW);

  pinMode(ADS_PWDN, OUTPUT);
  digitalWrite(ADS_PWDN, HIGH);
  delay(20);

  pinMode(ADS_RESET, OUTPUT);
  pinMode(ADS_CS, OUTPUT);
  pinMode(ADS_DRDY, INPUT_PULLUP);
  pinMode(ADS_START, OUTPUT);
  digitalWrite(ADS_CS, HIGH);
  digitalWrite(ADS_START, LOW);
  digitalWrite(ADS_RESET, HIGH);

  SPI.begin(SPI_SCLK, SPI_MISO, SPI_MOSI, ADS_CS);
  SPI.setFrequency(1000000);
  // ADS129x 使用 SPI 模式 1（CPOL=0, CPHA=1）。
  SPI.setDataMode(SPI_MODE1);

  resetPulse();
  configure();

  // RDATAC 模式下 RREG 会被忽略，读寄存器前先 SDATAC。
  writeCommand(ADS_CMD_SDATAC);
  delayMicroseconds(10);
  uint8_t id = readRegisterRaw(ADS_REG_ID);
  uint8_t c1 = readRegisterRaw(ADS_REG_CONFIG1);
  uint8_t c2 = readRegisterRaw(ADS_REG_CONFIG2);
  uint8_t c3 = readRegisterRaw(ADS_REG_CONFIG3);
  writeCommand(ADS_CMD_RDATAC);
  delayMicroseconds(10);

  Serial.print("[ADS] ID=0x");
  Serial.print(id, HEX);
  Serial.print(" CONFIG1=0x");
  Serial.print(c1, HEX);
  Serial.print(" CONFIG2=0x");
  Serial.print(c2, HEX);
  Serial.print(" CONFIG3=0x");
  Serial.println(c3, HEX);

  if (id != 0x92) {
    Serial.println("[ADS] WARNING: ID != 0x92, check wiring!");
  }
  return id;
}

bool readChannels(int32_t ch[EMG_CHANNELS]) {
  uint32_t t0 = millis();
  while (digitalRead(ADS_DRDY) != LOW) {
    if (millis() - t0 > 4) return false;
  }

  uint8_t b[27];
  digitalWrite(ADS_CS, LOW);
  for (int i = 0; i < 27; i++) b[i] = SPI.transfer(0x00);
  digitalWrite(ADS_CS, HIGH);

  for (int i = 0; i < EMG_CHANNELS; i++) {
    int32_t v = ((int32_t)b[3 + i * 3] << 16) |
                ((int32_t)b[4 + i * 3] << 8) |
                (int32_t)b[5 + i * 3];
    ch[i] = (v << 8) >> 8;  // 24bit 补码转 int32
  }
  return true;
}

void startConversion() { digitalWrite(ADS_START, HIGH); }
void stopConversion() { digitalWrite(ADS_START, LOW); }

void setSampleRate(uint16_t sps) {
  s_sps = sps;
  configure();
}

uint16_t sampleRate() { return s_sps; }

bool setGainByValue(int gain) {
  int idx = -1;
  for (int i = 0; i < 7; i++) {
    if (kGainTable[i] == gain) {
      idx = i;
      break;
    }
  }
  if (idx < 0) return false;
  s_gain_index = (uint8_t)idx;
  configure();
  return true;
}

uint8_t gainValue() { return kGainTable[s_gain_index]; }

void setTestSignal(bool on) {
  if (on) {
    s_gain_backup = s_gain_index;
    s_gain_index = 6;  // 12x：2 mV 测试信号不放大几乎看不见
  } else {
    s_gain_index = s_gain_backup;
  }
  s_test_signal = on;
  configure();
}

bool testSignal() { return s_test_signal; }

void debugReport(bool streaming) {
  writeCommand(ADS_CMD_SDATAC);
  delayMicroseconds(10);
  uint8_t id = readRegisterRaw(ADS_REG_ID);
  uint8_t c1 = readRegisterRaw(ADS_REG_CONFIG1);
  uint8_t c2 = readRegisterRaw(ADS_REG_CONFIG2);
  uint8_t c3 = readRegisterRaw(ADS_REG_CONFIG3);
  uint8_t ch1 = readRegisterRaw(ADS_REG_CHNSET);
  writeCommand(ADS_CMD_RDATAC);
  delayMicroseconds(10);

  uint32_t t0 = micros();
  uint32_t edges = 0;
  int last = digitalRead(ADS_DRDY);
  while (micros() - t0 < 200000UL) {
    int now = digitalRead(ADS_DRDY);
    if (last == HIGH && now == LOW) edges++;
    last = now;
  }
  uint32_t elapsed = micros() - t0;
  uint32_t rate = elapsed ? (uint32_t)((uint64_t)edges * 1000000ULL / elapsed) : 0;

  String r = "ADS_DBG:";
  r += "ID=0x" + String(id, HEX);
  r += ",C1=0x" + String(c1, HEX);
  r += ",C2=0x" + String(c2, HEX);
  r += ",C3=0x" + String(c3, HEX);
  r += ",CH1=0x" + String(ch1, HEX);
  r += ",DRDY_EDGES=" + String(edges);
  r += ",RATE=" + String(rate) + "Hz";
  r += ",DRDY_NOW=" + String(digitalRead(ADS_DRDY));
  r += ",EMG=" + String(streaming ? 1 : 0);
  r += ",START=" + String(digitalRead(ADS_START));
  r += ",PWDN=" + String(digitalRead(ADS_PWDN));
  r += ",RESET=" + String(digitalRead(ADS_RESET));
  ble::sendText(r);
}

void spiBusReport() {
  pinMode(ADS_CS, OUTPUT);
  digitalWrite(ADS_CS, HIGH);

  pinMode(SPI_MISO, INPUT_PULLUP);
  delayMicroseconds(100);
  int miso_pu = digitalRead(SPI_MISO);
  pinMode(SPI_MISO, INPUT_PULLDOWN);
  delayMicroseconds(100);
  int miso_pd = digitalRead(SPI_MISO);

  pinMode(ADS_DRDY, INPUT_PULLUP);
  delayMicroseconds(100);
  int drdy_pu = digitalRead(ADS_DRDY);
  pinMode(ADS_DRDY, INPUT_PULLDOWN);
  delayMicroseconds(100);
  int drdy_pd = digitalRead(ADS_DRDY);

  pinMode(SPI_MISO, INPUT);
  pinMode(ADS_DRDY, INPUT_PULLUP);

  String r = "SPI_BUS:";
  r += "MISO_PU=" + String(miso_pu);
  r += ",MISO_PD=" + String(miso_pd);
  r += ",DRDY_PU=" + String(drdy_pu);
  r += ",DRDY_PD=" + String(drdy_pd);
  r += ",CS=" + String(digitalRead(ADS_CS));
  r += ",SCLK=" + String(digitalRead(SPI_SCLK));
  r += ",MOSI=" + String(digitalRead(SPI_MOSI));
  ble::sendText(r);
}

void spiTest() {
  SPI.begin(SPI_SCLK, SPI_MISO, SPI_MOSI, ADS_CS);
  SPI.setFrequency(1000000);
  SPI.setDataMode(SPI_MODE1);
  pinMode(ADS_CS, OUTPUT);
  digitalWrite(ADS_CS, HIGH);

  digitalWrite(ADS_CS, LOW);
  uint8_t test_data = 0xAA;
  uint8_t result = SPI.transfer(test_data);
  digitalWrite(ADS_CS, HIGH);

  ble::sendText("SPI_TEST: TX=0x" + String(test_data, HEX) +
                " RX=0x" + String(result, HEX));
}

}  // namespace ads

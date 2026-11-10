/*
  Node_1 手环固件 —— ESP32-S3 + ADS1298 8 通道 sEMG
  适用板卡：2026-09 新采集板（丝印 GPxx 就是 GPIOxx）

  新板端口分配（下面 #define 一一对应；编号和上一版板卡完全相同）：
    GP01  IMU (IIC) SDA         GP02  ADS1298 (SPI) MISO
    GP03  IMU (IIC) SCL         GP04  扩展 (IIC) SCL   （旧板丝印写的是 IIC-OLED）
    GP05  扩展 (IIC) SDA        GP06  ADS1298 (SPI) SCLK
    GP07  ADS1298 (SPI) MOSI    GP08  MCU 状态灯
    GP09  ADS1298 RESET#        GP10  ADS1298 (SPI) CS
    GP11  ADS1298 DRDY#         GP12  ADS1298 START
    GP13  ADS1298 PWDN#

  跟着新板改了什么（引脚号本身没动，因为新板和旧板分配一致）：
    1) IMU 上电自动认总线：先试 GP01/GP03，读不到再试 GP04/GP05，串口会打印
       认到哪一条（旧板一直没确定 IMU 走的是 IIC1 还是 IIC2，现在不用猜了）；
    2) GP08 当 MCU 状态灯，做成呼吸灯（LEDC PWM，5kHz/10bit）：上电自检常亮、
       广播中 3s 慢呼吸（偏暗）、已连接 1.5s 中速呼吸、采集中 0.5s 快呼吸。
       上位机手动发 PIN8=1/0 会切到手动模式（常亮/常灭），等下一次连接/采集
       状态变化再交回自动灯语；
    3) GP04/GP05 由「OLED 专用」改称「扩展 IIC」，两条 I2C 都初始化好先备着。
       另加了一条 `ADS_DBG` 命令：读回 ID/寄存器 + 数 200ms 内 DRDY 下降沿，
       用来判断 ADS1298 到底在不在转换（FRAMES=0 时先发它）。

  硬件注意（2026-09-20 实测踩过的坑）：
    ADS1298 的 CLKSEL 是「时钟源选择」的静态配置脚，必须接 DVDD（=1，用芯片内部振荡器），
    千万不要把它和 CLK 引脚接在一起。悬空时时钟源不确定：芯片会时不时完全无响应，
    SPI 读回全 0（ID=0x00）、FRAMES=0，而且随触碰/角度变化，非常难查。
    用内部振荡器时 CLK 脚悬空（三态）即可；本固件 CONFIG1 的 CLK_EN 始终为 0，
    所以 CLK 脚不会输出时钟（若接成 CLKSEL=0 用外部时钟，CLK 必须喂 1.94~2.25MHz）。
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

// 服务/特征 UUID，跟上位机保持一致 上位机打包成exe文件再给你
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"

// - 引脚（新板丝印 GPxx = GPIOxx，顺序按新板端口表排）-
#define IMU_SDA    1    // GP01  IMU (IIC) SDA
#define SPI_MISO   2    // GP02  ADS1298 (SPI) MISO
#define IMU_SCL    3    // GP03  IMU (IIC) SCL
#define I2C2_SCL   4    // GP04  扩展 (IIC) SCL（旧板丝印 IIC-OLED）
#define I2C2_SDA   5    // GP05  扩展 (IIC) SDA（旧板丝印 IIC-OLED）
#define SPI_SCLK   6    // GP06  ADS1298 (SPI) SCLK
#define SPI_MOSI   7    // GP07  ADS1298 (SPI) MOSI
#define LED_PIN    8    // GP08  MCU 状态灯
#define ADS_RESET  9    // GP09  ADS1298 RESET#
#define ADS_CS     10   // GP10  ADS1298 (SPI) CS
#define ADS_DRDY   11   // GP11  ADS1298 DRDY#
#define ADS_START  12   // GP12  ADS1298 START
#define ADS_PWDN   13   // GP13  ADS1298 PWDN#

// 旧名字留着，老笔记/别的 sketch 里还在用
#define OLED_SCL   I2C2_SCL
#define OLED_SDA   I2C2_SDA

// - ADS1298 命令 
#define ADS_CMD_START   0x08
#define ADS_CMD_STOP    0x0A
#define ADS_CMD_RDATAC  0x10
#define ADS_CMD_SDATAC  0x11
#define ADS_CMD_RREG    0x20
#define ADS_CMD_WREG    0x40

// - ADS1298 寄存器 -
#define ADS_REG_ID       0x00
#define ADS_REG_CONFIG1  0x01
#define ADS_REG_CONFIG2  0x02
#define ADS_REG_CONFIG3  0x03
#define ADS_REG_CHNSET   0x05   // CHnSET = 0x05 + n

// CONFIG1 就两个值：DR 位 100/101 对应 HR 模式 2000/1000 SPS(我们测试尽可能用2000hz，不然黄家垅还得抽帧)
#define ADS_CONFIG1_2000SPS 0x84
#define ADS_CONFIG1_1000SPS 0x85
// CONFIG2：bit7:6 是保留位，固定写 01（所以基准值 0x40）；bit4 = INT_TEST
#define ADS_CONFIG2_NORMAL  0x40
// 内部测试信号：INT_TEST=1(bit4) + TEST_AMP=1(bit2, 2mV) + TEST_FREQ=01(bit0, f_CLK/2^20 ≈ 2Hz 方波)
#define ADS_CONFIG2_TEST    0x55
// CONFIG3：开内部基准（2.4V）。如果板子用外部基准，把这个改成 0x40
#define ADS_CONFIG3_INTREF  0xC0

// - LSM6DSR -
#define LSM_ADDR_A     0x6A
#define LSM_ADDR_B     0x6B   // SA0 拉高时的地址，自动试一下
#define LSM_WHO_AM_I   0x0F
#define LSM_CTRL1_XL   0x10   // 加速度 ODR/量程
#define LSM_CTRL2_G    0x11   // 陀螺仪 ODR/量程
#define LSM_CTRL3_C    0x12   // BDU + IF_INC
#define LSM_OUTX_L_G   0x22
#define LSM_OUTX_L_XL  0x28

// - 采样 -
#define EMG_SPS_HZ      2000   // 默认采样率
#define EMG_CHANNELS    8
#define EMG_24BIT_FRAME 28     // 2头 + 1seq + 8*3 + 1xor
#define EMG_16BIT_FRAME 20
#define EMG_MAX_BATCH   8      // 一包最多 8 帧(224B)，留点余量给 MTU-3

// 全局变量，基本都围着“电脑连不连、采没在采”打转
BLECharacteristic *pCharacteristic;   // 电脑订阅的那个特征，发数据
BLEServer *pServer;                   // 整个 BLE 服务，连接/广播
bool deviceConnected = false;         

bool emgStreaming = false;            // 现在是不是在给电脑推数据
bool emgBits24 = true;                // 每通道按 24bit 发还是 16bit 发（MTU 小会降）
uint16_t emgSps = EMG_SPS_HZ;         // 当前采样率，1000 或 2000
uint8_t emgFramesPerNotify = EMG_MAX_BATCH;   // 攒够几帧再发一包
uint8_t emgBatchCount = 0;            // 缓冲里已经攒了几帧
uint8_t emgSeq = 0;                   // 帧序号，电脑靠它看丢没丢帧
uint8_t emgFrameBuf[12 * EMG_16BIT_FRAME];    // 打包用的临时缓冲，240B 两种模式都够
uint8_t adsGainIndex = 0;             // PGA 增益档位，默认 6x
uint8_t imuAddr = 0;                  // IMU 的 I2C 地址，探测到了才非 0
TwoWire *imuBus = &Wire;              // IMU 挂在哪条 I2C 上，探测后确定
uint8_t imuBusId = 0;                 // 0 = Wire（GP01/GP03），1 = Wire1（GP04/GP05）

// GP08 MCU 状态灯：呼吸灯（LEDC 出 PWM），状态靠"呼吸快慢 + 亮度"区分
#define LED_BOOT       0              // 上电自检：常亮
#define LED_IDLE       1              // 广播中：慢呼吸（偏暗）
#define LED_CONNECTED  2              // 已连接、没采集：中速呼吸（满亮）
#define LED_STREAMING  3              // 采集中：快呼吸（满亮）
#define LED_MANUAL     4              // 上位机手动：常亮 / 常灭
#define LED_PWM_FREQ   5000           // PWM 5kHz，肉眼看不出闪
#define LED_PWM_BITS   10             // 10bit 占空比，暗的时候也顺滑
#define LED_PWM_MAX    ((1 << LED_PWM_BITS) - 1)
#define LED_PWM_CH     0              // 只有 core 2.x 用得上通道号
#define LED_ACTIVE_LOW 0              // 灯若接成"GP08 拉低才亮"（灌电流）就改成 1
// 三段呼吸节奏：周期(ms) + 峰值亮度(0~255)
#define LED_BR_SLOW_MS    3000
#define LED_BR_SLOW_PEAK  150         // 待机：慢 + 偏暗
#define LED_BR_MID_MS     1500
#define LED_BR_MID_PEAK   255         // 已连接：中速 + 满亮
#define LED_BR_FAST_MS    500
#define LED_BR_FAST_PEAK  255         // 采集中：快 + 满亮
uint8_t ledMode = LED_BOOT;
bool ledManualOn = false;             // 手动模式下是亮还是灭
uint32_t ledPhaseStart = 0;           // 当前灯语起点，换灯语就从最暗处重新起一段

const uint8_t adsGainTable[7] = {6, 1, 2, 3, 4, 8, 12};   // GAINn[2:0] 对应的倍数
uint8_t adsGainIndexBackup = 0;       // 开自检信号前的增益，关掉时还原
bool adsTestSignal = false;           // true = 8 个通道都改接内部测试信号（自检用）
uint32_t emgFrameCount = 0;           // 本次采集从 ADS 读到的样本帧数
uint32_t txPacketCount = 0;           // 成功发出去的 BLE 数据包数
uint32_t txFailCount = 0;             // 没能交给蓝牙栈的包数（未连接 / 抢不到发送锁）

SemaphoreHandle_t bleTxMutex = NULL;  // 发数据的锁，防命令回复和波形打架

// 函数声明（Arduino 个别版本不会自动生成，写全最保险）
bool sendNotify(const uint8_t *data, size_t len);
void sendResponse(const String &msg);
void adsWriteCmd(uint8_t cmd);
void adsWriteReg(uint8_t reg, uint8_t val);
uint8_t adsReadReg(uint8_t reg);
void adsResetPulse();
void adsConfigure();
uint8_t adsInitAll();
void adsDebugReport();
void adsSetTestSignal(bool on);
bool adsReadChannels(int32_t ch[EMG_CHANNELS]);
void flushEmgBatch();
void appendEmgFrame(const int32_t ch[EMG_CHANNELS]);
void startEmgStream();
void stopEmgStream();
bool imuWriteReg(uint8_t reg, uint8_t val);
bool imuReadReg(uint8_t reg, uint8_t *val);
void imuInit();
bool imuReadRaw(int16_t raw[6]);
void ledInit();
void ledWrite(uint16_t duty);
void ledSetMode(uint8_t mode);
void ledUpdate();
int ledLogicalState();
void scanI2C();
void spiBusReport();
void testSPI();
void setupI2C();
int readSpecialPin(int pin);
void handleCommand(String cmd);

// -
// BLE 发送
// -

// 不管命令回复还是数据帧，最后都从这儿发出去；顺手加锁防止两边抢着发
// 返回值 = 这包有没有成功交给蓝牙协议栈（未连接 / 拿不到发送锁时为 false）。
// 注意：ESP32 Arduino 核心自带的 BLE 库里 BLECharacteristic::notify() 返回 void，
// 拿不到"对端是否真的收到"这个结果，所以蓝牙拥塞造成的丢包在上位机按帧序号 seq 统计。
bool sendNotify(const uint8_t *data, size_t len) {
  if (!deviceConnected || !pCharacteristic || len == 0) return false;

  // 命令回复和 EMG 数据都走这个特征，不锁一下会串
  if (xSemaphoreTake(bleTxMutex, pdMS_TO_TICKS(20)) != pdTRUE) return false;
  pCharacteristic->setValue(data, len);
  pCharacteristic->notify();
  xSemaphoreGive(bleTxMutex);
  return true;
}

// 把一句话回复发给电脑，同时打印到串口方便调试
void sendResponse(const String &msg) {
  if (deviceConnected) {
    sendNotify((const uint8_t *)msg.c_str(), msg.length());
    Serial.println("[TX] " + msg);
  }
}

// -
// ADS1298 底层读写
// -

// 给 ADS1298 发一个单字节命令（开始/停止/切换模式那些）
void adsWriteCmd(uint8_t cmd) {
  digitalWrite(ADS_CS, LOW);
  SPI.transfer(cmd);
  digitalWrite(ADS_CS, HIGH);
  delayMicroseconds(2);
}

// 往 ADS1298 某个寄存器写一个字节
void adsWriteReg(uint8_t reg, uint8_t val) {
  // 一次只写一个寄存器，第三个字节就是值
  digitalWrite(ADS_CS, LOW);
  SPI.transfer(ADS_CMD_WREG | reg);
  SPI.transfer(0x00);
  SPI.transfer(val);
  digitalWrite(ADS_CS, HIGH);
  delayMicroseconds(2);
}

// 读 ADS1298 某个寄存器，返回的就是里边存的值
uint8_t adsReadReg(uint8_t reg) {
  digitalWrite(ADS_CS, LOW);
  SPI.transfer(ADS_CMD_RREG | reg);
  SPI.transfer(0x00);
  uint8_t val = SPI.transfer(0x00);
  digitalWrite(ADS_CS, HIGH);
  delayMicroseconds(2);
  return val;
}

// 上电/复位后等一会儿再操作，不然经常读不到 ID
void adsResetPulse() {
  digitalWrite(ADS_RESET, LOW);
  delay(10);
  digitalWrite(ADS_RESET, HIGH);
  delay(100);
}

// 写寄存器之前必须先离开 RDATAC，这个顺序折腾了我一晚上……
void adsConfigure() {
  adsWriteCmd(ADS_CMD_SDATAC);
  delayMicroseconds(10);

  adsWriteReg(ADS_REG_CONFIG1,
              (emgSps >= 2000) ? ADS_CONFIG1_2000SPS : ADS_CONFIG1_1000SPS);
  adsWriteReg(ADS_REG_CONFIG2, adsTestSignal ? ADS_CONFIG2_TEST : ADS_CONFIG2_NORMAL);
  adsWriteReg(ADS_REG_CONFIG3, ADS_CONFIG3_INTREF);

  // CHnSET：bit6:4 增益，bit2:0 输入选择（000=电极输入，101=内部测试信号）
  for (int i = 0; i < EMG_CHANNELS; i++) {
    adsWriteReg(ADS_REG_CHNSET + i,
                (adsGainIndex << 4) | (adsTestSignal ? 0x05 : 0x00));
  }

  // 内部基准要一点建立时间
  delay(150);

  adsWriteCmd(ADS_CMD_RDATAC);
  delayMicroseconds(10);
}

// 把 ADS1298 完整初始化一遍：供电、复位、写寄存器，最后读 ID 验证
uint8_t adsInitAll() {
  stopEmgStream();

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
  // ADS129x 要求 SPI 模式 1（CPOL=0, CPHA=1，见手册 Figure 1 下面的注）。
  // 之前写 MODE0，命令在错误的时钟沿被译码、DOUT 也在错误的边沿采样。
  SPI.setDataMode(SPI_MODE1);

  adsResetPulse();
  adsConfigure();

  // 读寄存器前必须先 SDATAC：RDATAC 模式下 RREG 会被芯片直接忽略
  // （手册 Table 15 注1），所以以前这里读到的 ID/CONFIG 都是垃圾值。
  adsWriteCmd(ADS_CMD_SDATAC);
  delayMicroseconds(10);
  uint8_t id = adsReadReg(ADS_REG_ID);
  uint8_t c1 = adsReadReg(ADS_REG_CONFIG1);
  uint8_t c2 = adsReadReg(ADS_REG_CONFIG2);
  uint8_t c3 = adsReadReg(ADS_REG_CONFIG3);
  adsWriteCmd(ADS_CMD_RDATAC);
  delayMicroseconds(10);

  Serial.print("[ADS] ID=0x");
  Serial.print(id, HEX);
  Serial.print(" CONFIG1=0x");
  Serial.print(c1, HEX);
  Serial.print(" CONFIG2=0x");
  Serial.print(c2, HEX);
  Serial.print(" CONFIG3=0x");
  Serial.println(c3, HEX);

  // 0x92 应该是 ADS1298，读到别的先查线
  if (id != 0x92) {
    Serial.println("[ADS] WARNING: ID != 0x92, check wiring!");
  }
  return id;
}

// 等 DRDY 拉低，然把 27 字节读回来
bool adsReadChannels(int32_t ch[EMG_CHANNELS]) {
  uint32_t t0 = millis();
  while (digitalRead(ADS_DRDY) != LOW) {
    if (millis() - t0 > 4) return false;   // 超了就跳过这一拍
  }

  // RDATAC 下第一组 3 字节是状态字，后面才是 8 个通道
  uint8_t b[27];
  digitalWrite(ADS_CS, LOW);
  for (int i = 0; i < 27; i++) {
    b[i] = SPI.transfer(0x00);
  }
  digitalWrite(ADS_CS, HIGH);

  for (int i = 0; i < EMG_CHANNELS; i++) {
    int32_t v = ((int32_t)b[3 + i * 3] << 16) |
                ((int32_t)b[4 + i * 3] << 8) |
                (int32_t)b[5 + i * 3];
    ch[i] = (v << 8) >> 8;   // 24bit 补码转 int32
  }
  return true;
}

// -
// EMG 数据流
// -

// 一条命令看清 ADS1298 到底在不在转换：读回 ID/寄存器，再在 200ms 窗口里数 DRDY 下降沿。
// 为什么不能只看一次 digitalRead：2000SPS 下 DRDY 的低脉冲只有几微秒，随机读一次几乎必然读到高。
void adsDebugReport() {
  // 读寄存器前必须先离开 RDATAC（手册 Table 15 注1），读完再切回去
  adsWriteCmd(ADS_CMD_SDATAC);
  delayMicroseconds(10);
  uint8_t id  = adsReadReg(ADS_REG_ID);
  uint8_t c1  = adsReadReg(ADS_REG_CONFIG1);
  uint8_t c2  = adsReadReg(ADS_REG_CONFIG2);
  uint8_t c3  = adsReadReg(ADS_REG_CONFIG3);
  uint8_t ch1 = adsReadReg(ADS_REG_CHNSET);   // 顺便验证写进去的增益有没有生效
  adsWriteCmd(ADS_CMD_RDATAC);
  delayMicroseconds(10);

  uint32_t t0 = micros();
  uint32_t edges = 0;
  int last = digitalRead(ADS_DRDY);
  while (micros() - t0 < 200000UL) {          // 200ms 窗口
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
  r += ",EMG=" + String(emgStreaming ? 1 : 0);
  r += ",START=" + String(digitalRead(ADS_START));
  r += ",PWDN=" + String(digitalRead(ADS_PWDN));
  r += ",RESET=" + String(digitalRead(ADS_RESET));
  sendResponse(r);
}

// 攒够一包的帧，一次性 notify 出去，省得一帧一包太碎
void flushEmgBatch() {
  if (emgBatchCount == 0) return;
  size_t frameLen = emgBits24 ? EMG_24BIT_FRAME : EMG_16BIT_FRAME;
  if (sendNotify(emgFrameBuf, emgBatchCount * frameLen)) {
    txPacketCount++;
  } else {
    txFailCount++;   // 没交出去（连接断开 / 抢不到发送锁）
  }
  emgBatchCount = 0;
}

// 把这一拍 8 通道的数据按协议塞进打包缓冲，满了就发
void appendEmgFrame(const int32_t ch[EMG_CHANNELS]) {
  size_t frameLen = emgBits24 ? EMG_24BIT_FRAME : EMG_16BIT_FRAME;
  uint8_t *p = emgFrameBuf + emgBatchCount * frameLen;

  p[0] = 0xAA;
  p[1] = 0x55;
  p[2] = emgSeq++;

  for (int i = 0; i < EMG_CHANNELS; i++) {
    if (emgBits24) {
      p[3 + i * 3] = (uint8_t)((ch[i] >> 16) & 0xFF);
      p[4 + i * 3] = (uint8_t)((ch[i] >> 8) & 0xFF);
      p[5 + i * 3] = (uint8_t)(ch[i] & 0xFF);
    } else {
      int16_t v16 = (int16_t)(ch[i] >> 8);
      p[3 + i * 2] = (uint8_t)((v16 >> 8) & 0xFF);
      p[4 + i * 2] = (uint8_t)(v16 & 0xFF);
    }
  }

  // 整帧除最后一位全做异或，丢坏帧
  uint8_t x = 0;
  for (size_t i = 0; i < frameLen - 1; i++) x ^= p[i];
  p[frameLen - 1] = x;

  emgFrameCount++;
  emgBatchCount++;
  if (emgBatchCount >= emgFramesPerNotify) flushEmgBatch();
}

// EMG_START 就进这儿：先看对端 MTU 定打包方式，再让 ADS 开始转
void startEmgStream() {
  if (emgStreaming) {
    sendResponse("OK: EMG already running");
    return;
  }

  // 看看对端 MTU 协商到多少，按能塞下的帧数打包
  uint16_t peerMtu = 23;
  if (pServer) {
    std::map<uint16_t, conn_status_t> peers = pServer->getPeerDevices(false);
    for (std::map<uint16_t, conn_status_t>::iterator it = peers.begin();
         it != peers.end(); ++it) {
      if (it->second.mtu > peerMtu) peerMtu = it->second.mtu;
    }
  }
  int payload = (peerMtu > 3) ? (int)(peerMtu - 3) : 20;
  Serial.printf("[EMG] peer MTU=%d payload=%d\n", peerMtu, payload);

  if (payload >= EMG_24BIT_FRAME) {
    emgBits24 = true;
    int n = payload / EMG_24BIT_FRAME;
    emgFramesPerNotify = (n > EMG_MAX_BATCH) ? EMG_MAX_BATCH : (uint8_t)n;
  } else if (payload >= EMG_16BIT_FRAME) {
    // MTU 只有 23 的奇葩情况，降到 16bit 保底
    emgBits24 = false;
    int n = payload / EMG_16BIT_FRAME;
    emgFramesPerNotify = (n > 12) ? 12 : (uint8_t)n;
  } else {
    sendResponse("ERROR: MTU too small for EMG");
    return;
  }

  emgBatchCount = 0;
  emgSeq = 0;
  emgFrameCount = 0;
  txPacketCount = 0;
  txFailCount = 0;

  // 先把配置告诉电脑，再让它收二进制，顺序不能反
  String cfg = "EMG_CFG:" + String(emgSps) + "," +
               String(emgBits24 ? 24 : 16) + "," + String(EMG_CHANNELS);
  sendResponse(cfg);
  delay(10);

  digitalWrite(ADS_START, HIGH);   // 开始连续转换
  emgStreaming = true;
  ledSetMode(LED_STREAMING);       // GP08：采集中快呼吸
  Serial.println("[EMG] streaming started: " + cfg);
}

// 停止采集：清标志、清缓冲，把 START 拉低让芯片歇着
void stopEmgStream() {
  emgStreaming = false;
  emgBatchCount = 0;
  digitalWrite(ADS_START, LOW);
  ledSetMode(deviceConnected ? LED_CONNECTED : LED_IDLE);
}

// 切内部测试信号：8 个通道都接内部 ~2Hz 方波，用来验证 ADC+SPI+蓝牙整条链路
void adsSetTestSignal(bool on) {
  bool wasStreaming = emgStreaming;
  stopEmgStream();
  if (on) {
    adsGainIndexBackup = adsGainIndex;
    adsGainIndex = 6;                 // 12x：测试信号只有 2mV，不放大在波形上几乎看不见
  } else {
    adsGainIndex = adsGainIndexBackup;
  }
  adsTestSignal = on;
  adsConfigure();
  if (wasStreaming) startEmgStream();
}

// -
// LSM6DSR IMU（I2C，慢一点没关系）
// -

// 往 LSM6DSR 寄存器写一个字节，成功返回 true（走 imuBus 那条总线）
bool imuWriteReg(uint8_t reg, uint8_t val) {
  imuBus->beginTransmission(imuAddr);
  imuBus->write(reg);
  imuBus->write(val);
  return (imuBus->endTransmission() == 0);
}

// 读 LSM6DSR 单个寄存器，读到了写进 val
bool imuReadReg(uint8_t reg, uint8_t *val) {
  imuBus->beginTransmission(imuAddr);
  imuBus->write(reg);
  if (imuBus->endTransmission(false) != 0) return false;
  if (imuBus->requestFrom((int)imuAddr, 1) != 1) return false;
  *val = imuBus->read();
  return true;
}

// 上电找 LSM6DSR：先试 GP01/GP03（新板 IMU 在这），读不到再试 GP04/GP05
// （旧板那条丝印写 IIC-OLED 的，万一硬件把 IMU 布到这边也能自己认出来）。
// 0x6A/0x6B 两个地址都试一遍，找到就把加速度和陀螺仪打开。
void imuInit() {
  uint8_t id = 0;
  imuAddr = 0;
  for (int bus = 0; bus < 2 && !imuAddr; bus++) {
    imuBus = (bus == 0) ? &Wire : &Wire1;   // 0 = GP01/GP03，1 = GP04/GP05
    imuBusId = (uint8_t)bus;
    for (int i = 0; i < 2; i++) {
      imuAddr = (i == 0) ? LSM_ADDR_A : LSM_ADDR_B;
      if (imuReadReg(LSM_WHO_AM_I, &id) && id == 0x6A) break;
      imuAddr = 0;
    }
  }
  if (!imuAddr) {
    imuBus = &Wire;                        // 都找不到就退回默认，别留野指针
    imuBusId = 0;
    Serial.println("[IMU] LSM6DSR not found on either bus (GP01/GP03 or GP04/GP05)");
    return;
  }

  imuWriteReg(LSM_CTRL3_C, 0x44);   // BDU + 地址自增
  imuWriteReg(LSM_CTRL1_XL, 0x64);  // 加速度 416Hz ±4g
  imuWriteReg(LSM_CTRL2_G, 0x64);   // 陀螺仪 416Hz ±500dps

  Serial.printf("[IMU] LSM6DSR ok, addr=0x%02X, bus%d=%s, 416Hz ±4g / ±500dps\n",
                imuAddr, imuBusId,
                imuBusId == 0 ? "GP01/GP03" : "GP04/GP05");
}

// 一口气把陀螺仪 + 加速度 6 个轴的原始值读回来，顺序 GX..AZ
bool imuReadRaw(int16_t raw[6]) {
  // 顺序固定：先陀螺仪(GX..GZ)再加速度(AX..AZ)
  uint8_t buf[6];
  imuBus->beginTransmission(imuAddr);
  imuBus->write(LSM_OUTX_L_G);
  if (imuBus->endTransmission(false) != 0) return false;
  if (imuBus->requestFrom((int)imuAddr, 6) != 6) return false;
  for (int i = 0; i < 6; i++) buf[i] = imuBus->read();
  for (int i = 0; i < 3; i++) {
    raw[i] = (int16_t)((buf[i * 2 + 1] << 8) | buf[i * 2]);
  }

  imuBus->beginTransmission(imuAddr);
  imuBus->write(LSM_OUTX_L_XL);
  if (imuBus->endTransmission(false) != 0) return false;
  if (imuBus->requestFrom((int)imuAddr, 6) != 6) return false;
  for (int i = 0; i < 6; i++) buf[i] = imuBus->read();
  for (int i = 0; i < 3; i++) {
    raw[3 + i] = (int16_t)((buf[i * 2 + 1] << 8) | buf[i * 2]);
  }
  return true;
}

// -
// 杂七杂八的调试命令
// -

// 总线上有没有器件应答（0x08~0x77 粗略扫一遍）
bool i2cAnyDevice(TwoWire *bus) {
  for (uint8_t addr = 0x08; addr < 0x78; addr++) {
    bus->beginTransmission(addr);
    if (bus->endTransmission() == 0) return true;
  }
  return false;
}

// 扫一条 I2C 总线上能应答的地址，返回类似 "0x6a,0x3c"，一个都没有就返回 "none"
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

// 两条 I2C 都扫一遍：先 GP01/GP03，再 GP04/GP05（新板那个扩展口）
void scanI2C() {
  Wire.begin(IMU_SDA, IMU_SCL);
  Wire1.begin(I2C2_SDA, I2C2_SCL);
  sendResponse(scanI2CBus(&Wire, "I2C_DEVICES:"));    // GP01/GP03
  sendResponse(scanI2CBus(&Wire1, "I2C2_DEVICES:"));  // GP04/GP05
}

// 看 MISO / DRDY 这两根"芯片输出"的线上到底有没有人在驱动——
// 用内部上拉和下拉各读一次：被外部拉死，还是没人管，一测就知道。
// 判读：
//   MISO_PU=0              → MISO 被拉到低（短路搭地/虚焊/未供电芯片的钳位），ID 恒 0x00 就有解释了
//   DRDY_PD=1              → 芯片在主动把 DRDY 拉高，说明芯片有电、还活着，问题在时钟(CLKSEL)/SPI/复位
//   DRDY_PD=0 且 MISO_PU=1 → 两条线都没人驱动，芯片那边基本没在工作（供电/焊接/断线）
void spiBusReport() {
  pinMode(ADS_CS, OUTPUT);
  digitalWrite(ADS_CS, HIGH);        // 芯片不被选中，此时它不该驱动 DOUT

  pinMode(SPI_MISO, INPUT_PULLUP);
  delayMicroseconds(100);
  int misoPu = digitalRead(SPI_MISO);
  pinMode(SPI_MISO, INPUT_PULLDOWN);
  delayMicroseconds(100);
  int misoPd = digitalRead(SPI_MISO);

  pinMode(ADS_DRDY, INPUT_PULLUP);
  delayMicroseconds(100);
  int drdyPu = digitalRead(ADS_DRDY);
  pinMode(ADS_DRDY, INPUT_PULLDOWN);
  delayMicroseconds(100);
  int drdyPd = digitalRead(ADS_DRDY);

  // 还原成固件平时用的模式（MISO 交回 SPI 外设，DRDY 回到上拉输入）
  pinMode(SPI_MISO, INPUT);
  pinMode(ADS_DRDY, INPUT_PULLUP);

  String r = "SPI_BUS:";
  r += "MISO_PU=" + String(misoPu);
  r += ",MISO_PD=" + String(misoPd);
  r += ",DRDY_PU=" + String(drdyPu);
  r += ",DRDY_PD=" + String(drdyPd);
  r += ",CS=" + String(digitalRead(ADS_CS));
  r += ",SCLK=" + String(digitalRead(SPI_SCLK));
  r += ",MOSI=" + String(digitalRead(SPI_MOSI));
  sendResponse(r);
}

// 发个字节看看能不能收回来，粗略判断 SPI 通不通
void testSPI() {
  // 纯属看看 SPI 通不通，正常情况 RX 是上一字节的回读
  SPI.begin(SPI_SCLK, SPI_MISO, SPI_MOSI, ADS_CS);
  SPI.setFrequency(1000000);
  SPI.setDataMode(SPI_MODE1);   // ADS129x 是 SPI 模式 1
  pinMode(ADS_CS, OUTPUT);
  digitalWrite(ADS_CS, HIGH);

  digitalWrite(ADS_CS, LOW);
  byte testData = 0xAA;
  byte result = SPI.transfer(testData);
  digitalWrite(ADS_CS, HIGH);

  sendResponse("SPI_TEST: TX=0x" + String(testData, HEX) +
               " RX=0x" + String(result, HEX));
}

// 两条 I2C 都初始化：GP01/GP03 固定给 IMU，GP04/GP05 是扩展口（旧板丝印 IIC-OLED，没挂器件）
void setupI2C() {
  Wire.begin(IMU_SDA, IMU_SCL);       // GP01/GP03
  Wire1.begin(I2C2_SDA, I2C2_SCL);    // GP04/GP05 扩展口，先备着
  Serial.println("[I2C] initialized (GP01/03 + GP04/05)");
}

// - MCU 状态灯（GP08）-

// 上电初始化 LEDC（core 3.x 和 2.x 的 ledc API 不一样，这里分开写）
void ledInit() {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(LED_PIN, LED_PWM_FREQ, LED_PWM_BITS);      // core 3.x：绑定引脚
#else
  ledcSetup(LED_PWM_CH, LED_PWM_FREQ, LED_PWM_BITS);    // core 2.x：先配通道
  ledcAttachPin(LED_PIN, LED_PWM_CH);                   //          再把引脚挂到通道上
#endif
  ledWrite(LED_PWM_MAX);                                // 上电先常亮
}

// 写占空比（两种核心的 ledcWrite 参数不一样，包一层省得每处都写 #if）
void ledWrite(uint16_t duty) {
  if (LED_ACTIVE_LOW) duty = LED_PWM_MAX - duty;        // 灌电流接法：亮度反一下
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(LED_PIN, duty);
#else
  ledcWrite(LED_PWM_CH, duty);
#endif
}

// 换灯语：连接/采集状态一变就调一次
void ledSetMode(uint8_t mode) {
  if (ledMode == mode) return;
  ledMode = mode;
  ledPhaseStart = millis();            // 新灯语从"最暗"重新起一段，眼睛能看出换了状态
}

// loop() 里每次都调，纯非阻塞；手动模式不插手
void ledUpdate() {
  if (ledMode == LED_MANUAL) return;

  uint32_t period;
  uint16_t peak;
  if (ledMode == LED_IDLE)            { period = LED_BR_SLOW_MS; peak = LED_BR_SLOW_PEAK; }
  else if (ledMode == LED_CONNECTED)  { period = LED_BR_MID_MS;  peak = LED_BR_MID_PEAK; }
  else if (ledMode == LED_STREAMING)  { period = LED_BR_FAST_MS; peak = LED_BR_FAST_PEAK; }
  else {                               // 上电自检：常亮
    ledWrite(LED_PWM_MAX);
    return;
  }

  // 三角波 0→255→0，再平方一下（人眼差不多是平方关系），才像"呼吸"而不是匀速来回
  uint32_t half = period / 2;
  uint32_t phase = (millis() - ledPhaseStart) % period;
  uint32_t tri = (phase < half) ? (phase * 255 / half)
                                : (255 - (phase - half) * 255 / half);
  uint32_t level = tri * tri / 255;                 // 0~255
  ledWrite((uint16_t)(level * peak / 255 * LED_PWM_MAX / 255));
}

// 报给上位机的"灯是不是亮着"的语义值：PWM 引脚的瞬时电平没法读（呼吸时会乱跳），
// 所以这里按状态给：待机呼吸算 0，自检/已连接/采集中算 1，手动模式按手动值
int ledLogicalState() {
  if (ledMode == LED_MANUAL) return ledManualOn ? 1 : 0;
  return (ledMode == LED_IDLE) ? 0 : 1;
}

// “读特殊引脚”其实就是假装在检测 IMU 在不在
int readSpecialPin(int pin) {
  if (pin == LED_PIN) {
    return ledLogicalState();          // 状态灯是 PWM 驱动的，读引脚没意义
  }
  if (pin == IMU_SDA || pin == IMU_SCL) {
    return imuAddr ? 1 : 0;            // IMU 认到了就算在线
  }
  if (pin == I2C2_SCL || pin == I2C2_SDA) {
    Wire1.begin(I2C2_SDA, I2C2_SCL);
    return i2cAnyDevice(&Wire1) ? 1 : 0;
  }
  if (pin == SPI_MISO) {
    return digitalRead(pin);
  }
  return -1;
}

// -
// 命令处理
// -

// 电脑发来的文本命令都在这里分流，认得的处理，不认得的回 ERROR
void handleCommand(String cmd) {
  cmd.trim();
  // 手滑打成空格也认：ADS INIT → ADS_INIT。PIN/READ 后面跟的是数字，不动它们，
  // 免得 "PIN 8=1" 被拼成 "PIN_8=1" 解析成 GPIO0。
  if (!cmd.startsWith("PIN") && !cmd.startsWith("READ") && cmd.indexOf(' ') >= 0) {
    cmd.replace(' ', '_');
  }

  if (cmd.startsWith("PIN")) {
    int pinPos = cmd.indexOf('=');
    if (pinPos > 3) {
      int pin = cmd.substring(3, pinPos).toInt();
      int value = cmd.substring(pinPos + 1).toInt();

      // 写之前先挡一下越界：GPIO0 是启动配置脚，写坏了会影响下次上电启动
      if (pin < 1 || pin > 48) {
        sendResponse("ERROR: invalid pin");
        return;
      }
      // 输入脚和总线脚别乱写，容易把传感器搞懵
      if (pin == IMU_SDA || pin == IMU_SCL || pin == I2C2_SCL ||
          pin == I2C2_SDA || pin == SPI_MISO || pin == ADS_DRDY) {
        sendResponse("ERROR: pin " + String(pin) + " read-only");
        return;
      }
      if (pin == LED_PIN) {
        // GP08 是 PWM 呼吸灯：别用 pinMode/digitalWrite，交给 LEDC。
        // 手动写过就先听手动的，等下次连接/采集状态变化再交回自动灯语。
        ledSetMode(LED_MANUAL);
        ledManualOn = (value != 0);
        ledWrite(ledManualOn ? LED_PWM_MAX : 0);
      } else {
        pinMode(pin, OUTPUT);
        digitalWrite(pin, value ? HIGH : LOW);
      }
      sendResponse("OK: PIN" + String(pin) + "=" + String(value));
    } else {
      sendResponse("ERROR: bad format (PIN8=1)");
    }
  }
  else if (cmd.startsWith("READ")) {
    int pin = cmd.substring(4).toInt();
    if (pin >= 0 && pin <= 48) {
      if (pin == IMU_SDA || pin == IMU_SCL || pin == I2C2_SCL ||
          pin == I2C2_SDA || pin == SPI_MISO || pin == LED_PIN) {
        sendResponse("READ" + String(pin) + "=" + String(readSpecialPin(pin)));
      } else if (pin == ADS_RESET || pin == ADS_CS ||
                 pin == ADS_START || pin == ADS_PWDN) {
        // 固件自己在驱动的 ADS 控制线：直接读，别改成输入
        // （改成 INPUT_PULLUP 之后 digitalWrite 也恢复不了输出，会出现"读一下 START 就不动了"）
        sendResponse("READ" + String(pin) + "=" + String(digitalRead(pin)));
      } else {
        pinMode(pin, INPUT_PULLUP);
        sendResponse("READ" + String(pin) + "=" + String(digitalRead(pin)));
      }
    } else {
      sendResponse("ERROR: invalid pin");
    }
  }
  else if (cmd.startsWith("ALL")) {
    String response = "STATUS:";
    response += "LED=" + String(ledLogicalState()) + ",";
    response += "LEDMODE=" + String(ledMode) + ",";
    response += "DRDY=" + String(digitalRead(ADS_DRDY)) + ",";
    response += "RESET=" + String(digitalRead(ADS_RESET)) + ",";
    response += "START=" + String(digitalRead(ADS_START)) + ",";
    response += "PWDN=" + String(digitalRead(ADS_PWDN)) + ",";
    response += "EMG=" + String(emgStreaming ? 1 : 0) + ",";
    response += "TEST=" + String(adsTestSignal ? 1 : 0) + ",";
    response += "GAIN=" + String(adsGainTable[adsGainIndex]) + ",";
    response += "FRAMES=" + String(emgFrameCount) + ",";
    response += "TXPKT=" + String(txPacketCount) + ",";
    response += "TXFAIL=" + String(txFailCount);
    sendResponse(response);
  }
  else if (cmd.startsWith("EMG_START")) {
    startEmgStream();
  }
  else if (cmd.startsWith("EMG_STOP")) {
    stopEmgStream();
    sendResponse("EMG_STOPPED");
  }
  else if (cmd.startsWith("EMG_SPS=")) {
    int sps = cmd.substring(8).toInt();
    if (sps != 1000 && sps != 2000) {
      sendResponse("ERROR: SPS must be 1000 or 2000");
      return;
    }
    bool wasStreaming = emgStreaming;
    stopEmgStream();
    emgSps = (uint16_t)sps;
    adsConfigure();   // 改完寄存器，采集中的话再自动开回来
    if (wasStreaming) startEmgStream();
    sendResponse("OK: EMG_SPS=" + String(emgSps));
  }
  else if (cmd.startsWith("ADS_INIT")) {
    uint8_t id = adsInitAll();
    sendResponse("OK: ADS_INIT ID=0x" + String(id, HEX));
  }
  else if (cmd.startsWith("ADS_DBG")) {
    adsDebugReport();
  }
  else if (cmd.startsWith("ADS_GAIN=")) {
    int gain = cmd.substring(9).toInt();
    int idx = -1;
    for (int i = 0; i < 7; i++) {
      if (adsGainTable[i] == gain) { idx = i; break; }
    }
    if (idx < 0) {
      sendResponse("ERROR: gain must be 1/2/3/4/6/8/12");
      return;
    }
    bool wasStreaming = emgStreaming;
    stopEmgStream();
    adsGainIndex = (uint8_t)idx;
    adsConfigure();
    if (wasStreaming) startEmgStream();
    sendResponse("OK: ADS_GAIN=" + String(gain));
  }
  else if (cmd.startsWith("ADS_TEST=")) {
    int on = cmd.substring(9).toInt();
    adsSetTestSignal(on != 0);
    if (adsTestSignal) {
      sendResponse("OK: ADS_TEST=1 test signal on (gain=12x, ~2Hz square wave)");
    } else {
      sendResponse("OK: ADS_TEST=0 back to electrode input");
    }
  }
  else if (cmd.startsWith("IMU")) {
    if (!imuAddr) imuInit();   // 开机没探测到的话，用之前再试一次
    if (!imuAddr) {
      sendResponse("ERROR: IMU not detected");
      return;
    }
    int16_t raw[6];
    if (!imuReadRaw(raw)) {
      sendResponse("ERROR: IMU read failed");
      return;
    }

    // 满量程换算：±4g / ±500dps，满码都是 ±32768
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
    r += ",BUS" + String(imuBusId) + "=" + (imuBusId == 0 ? "GP01/GP03" : "GP04/GP05");
    sendResponse(r);
  }
  else if (cmd.startsWith("I2C_SCAN")) {
    scanI2C();
  }
  else if (cmd.startsWith("SPI_TEST")) {
    testSPI();
  }
  else if (cmd.startsWith("SPI_BUS")) {
    spiBusReport();
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
    help += "ADS_DBG     read ID/regs + count DRDY edges (is it converting?)\n";
    help += "ADS_GAIN=n  set PGA gain 1/2/3/4/6/8/12\n";
    help += "ADS_TEST=n  internal test signal on(1)/off(0)\n";
    help += "IMU         read LSM6DSR accel/gyro once (+ which I2C bus)\n";
    help += "I2C_SCAN    scan both I2C buses (GP01/03 + GP04/05)\n";
    help += "SPI_TEST    SPI loopback test\n";
    help += "SPI_BUS     test if MISO/DRDY lines are driven (chip alive?)\n";
    help += "HELP        show help";
    sendResponse(help);
  }
  else {
    sendResponse("ERROR: unknown command (HELP)");
  }
}

// -
// BLE 回调
// -

class MyServerCallbacks : public BLEServerCallbacks {
    // 电脑连上来了：状态灯转常亮、标记已连接
  void onConnect(BLEServer *server) {
    deviceConnected = true;
    ledSetMode(emgStreaming ? LED_STREAMING : LED_CONNECTED);
    Serial.println("[BLE] 设备已连接");
  }

    // 电脑断了：停采集、状态灯转回慢闪、重新广播等它再连
  void onDisconnect(BLEServer *server) {
    deviceConnected = false;
    stopEmgStream();   // 断开就别让 ADS 空转了
    Serial.println("[BLE] 设备已断开");
    server->startAdvertising();
    ledSetMode(LED_IDLE);
  }
};

class MyCharacteristicCallbacks : public BLECharacteristicCallbacks {
    // 电脑往特征里写东西（命令）就跑到这儿
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

// -
// setup / loop
// -

// 上电只跑一次：串口、引脚、传感器、蓝牙全在这儿
void setup() {
  Serial.begin(115200);
  delay(50);
  Serial.println();
  Serial.println("Node 御识 8CH sEMG firmware (ADS1298)");

  bleTxMutex = xSemaphoreCreateMutex();

  ledInit();                        // GP08 MCU 状态灯：LEDC PWM，上电先常亮

  setupI2C();
  imuInit();
  adsInitAll();

  // 蓝牙名在这里改 ↓
  BLEDevice::init("Node_1_by_hzx");
  esp_err_t mtuRc = BLEDevice::setMTU(247);
  Serial.printf("[BLE] local MTU set rc=%d\n", mtuRc);

  pServer = BLEDevice::createServer();
  pServer->setCallbacks(new MyServerCallbacks());

  BLEService *pService = pServer->createService(SERVICE_UUID);
  pCharacteristic = pService->createCharacteristic(
    CHARACTERISTIC_UUID,
    BLECharacteristic::PROPERTY_READ |
    BLECharacteristic::PROPERTY_WRITE |
    BLECharacteristic::PROPERTY_NOTIFY |
    BLECharacteristic::PROPERTY_INDICATE
  );

  pCharacteristic->setCallbacks(new MyCharacteristicCallbacks());
  pCharacteristic->addDescriptor(new BLE2902());
  pService->start();

  BLEAdvertising *pAdvertising = pServer->getAdvertising();
  pAdvertising->addServiceUUID(SERVICE_UUID);
  pAdvertising->setScanResponse(true);
  pAdvertising->setMinPreferred(0x06);   // 连接间隔 7.5ms ~ 22.5ms
  pAdvertising->setMaxPreferred(0x12);
  pServer->startAdvertising();

  Serial.println("[BLE] advertising started, name=Node_1_by_hzx");
  Serial.println("[BLE] UUID=" SERVICE_UUID);
  ledSetMode(LED_IDLE);             // 灯语：广播中慢呼吸（偏暗）
  Serial.println("[LED] GP08 breath led: idle=slow dim / connected=mid / streaming=fast");
}

// 主循环：在采就盯着 DRDY 读数据打包，没在采就歇着
void loop() {
  ledUpdate();   // GP08 状态灯，非阻塞
  if (emgStreaming && deviceConnected) {
    if (digitalRead(ADS_DRDY) == LOW) {
      int32_t ch[EMG_CHANNELS];
      if (adsReadChannels(ch)) {
        appendEmgFrame(ch);
      }
    } else {
      delayMicroseconds(80);   // 没数据就稍微等等，别空转太狠
    }
  } else {
    delay(1);
  }
}

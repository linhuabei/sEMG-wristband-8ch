#pragma once

#include <stdint.h>

// ---------------------------------------------------------------------------
// BLE
// ---------------------------------------------------------------------------
#define SERVICE_UUID        "4fafc201-1fb5-459e-8fcc-c5c9c331914b"
#define CHARACTERISTIC_UUID "beb5483e-36e1-4688-b7f5-ea07361b26a8"
#define DEVICE_NAME         "Node_1_by_hzx"
#define BLE_LOCAL_MTU       247

// ---------------------------------------------------------------------------
// 引脚（新板丝印 GPxx = GPIOxx）
// ---------------------------------------------------------------------------
#define IMU_SDA    1
#define SPI_MISO   2
#define IMU_SCL    3
#define I2C2_SCL   4
#define I2C2_SDA   5
#define SPI_SCLK   6
#define SPI_MOSI   7
#define LED_PIN    8
#define ADS_RESET  9
#define ADS_CS     10
#define ADS_DRDY   11
#define ADS_START  12
#define ADS_PWDN   13

// 旧名字保留，兼容旧笔记
#define OLED_SCL   I2C2_SCL
#define OLED_SDA   I2C2_SDA

// ---------------------------------------------------------------------------
// ADS1298 命令 / 寄存器
// ---------------------------------------------------------------------------
#define ADS_CMD_START   0x08
#define ADS_CMD_STOP    0x0A
#define ADS_CMD_RDATAC  0x10
#define ADS_CMD_SDATAC  0x11
#define ADS_CMD_RREG    0x20
#define ADS_CMD_WREG    0x40

#define ADS_REG_ID       0x00
#define ADS_REG_CONFIG1  0x01
#define ADS_REG_CONFIG2  0x02
#define ADS_REG_CONFIG3  0x03
#define ADS_REG_CHNSET   0x05

#define ADS_CONFIG1_2000SPS 0x84
#define ADS_CONFIG1_1000SPS 0x85
#define ADS_CONFIG2_NORMAL  0x40
#define ADS_CONFIG2_TEST    0x55
#define ADS_CONFIG3_INTREF  0xC0

// ---------------------------------------------------------------------------
// LSM6DSR
// ---------------------------------------------------------------------------
#define LSM_ADDR_A     0x6A
#define LSM_ADDR_B     0x6B
#define LSM_WHO_AM_I   0x0F
#define LSM_CTRL1_XL   0x10
#define LSM_CTRL2_G    0x11
#define LSM_CTRL3_C    0x12
#define LSM_OUTX_L_G   0x22
#define LSM_OUTX_L_XL  0x28

// ---------------------------------------------------------------------------
// EMG 采样 / 帧格式
// ---------------------------------------------------------------------------
#define EMG_SPS_HZ       2000
#define EMG_CHANNELS     8
#define EMG_24BIT_FRAME  28   // 2 头 + 1 seq + 8*3 + 1 xor
#define EMG_16BIT_FRAME  20
#define EMG_MAX_BATCH    8

// ---------------------------------------------------------------------------
// 状态灯（GP08，LEDC PWM）
// ---------------------------------------------------------------------------
#define LED_PWM_FREQ   5000
#define LED_PWM_BITS   10
#define LED_PWM_MAX    ((1 << LED_PWM_BITS) - 1)
#define LED_PWM_CH     0
#define LED_ACTIVE_LOW 0

#define LED_BR_SLOW_MS    3000
#define LED_BR_SLOW_PEAK  150
#define LED_BR_MID_MS     1500
#define LED_BR_MID_PEAK   255
#define LED_BR_FAST_MS    500
#define LED_BR_FAST_PEAK  255

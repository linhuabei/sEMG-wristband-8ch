#include "status_led.h"

#include "../app/config.h"

namespace led {
namespace {

Mode s_mode = Boot;
bool s_manual_on = false;
uint32_t s_phase_start = 0;

void writeDuty(uint16_t duty) {
  if (LED_ACTIVE_LOW) duty = LED_PWM_MAX - duty;
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcWrite(LED_PIN, duty);
#else
  ledcWrite(LED_PWM_CH, duty);
#endif
}

}  // namespace

void begin() {
#if defined(ESP_ARDUINO_VERSION_MAJOR) && ESP_ARDUINO_VERSION_MAJOR >= 3
  ledcAttach(LED_PIN, LED_PWM_FREQ, LED_PWM_BITS);
#else
  ledcSetup(LED_PWM_CH, LED_PWM_FREQ, LED_PWM_BITS);
  ledcAttachPin(LED_PIN, LED_PWM_CH);
#endif
  writeDuty(LED_PWM_MAX);  // 上电先常亮
}

void setMode(Mode mode) {
  if (s_mode == mode) return;
  s_mode = mode;
  s_phase_start = millis();
}

void setManual(bool on) {
  s_mode = Manual;
  s_manual_on = on;
  writeDuty(on ? LED_PWM_MAX : 0);
}

bool manualOn() { return s_manual_on; }

void update() {
  if (s_mode == Manual) return;

  uint32_t period;
  uint16_t peak;
  if (s_mode == Idle) {
    period = LED_BR_SLOW_MS;
    peak = LED_BR_SLOW_PEAK;
  } else if (s_mode == Connected) {
    period = LED_BR_MID_MS;
    peak = LED_BR_MID_PEAK;
  } else if (s_mode == Streaming) {
    period = LED_BR_FAST_MS;
    peak = LED_BR_FAST_PEAK;
  } else {
    writeDuty(LED_PWM_MAX);  // 上电自检：常亮
    return;
  }

  uint32_t half = period / 2;
  uint32_t phase = (millis() - s_phase_start) % period;
  uint32_t tri = (phase < half) ? (phase * 255 / half)
                                : (255 - (phase - half) * 255 / half);
  uint32_t level = tri * tri / 255;
  writeDuty((uint16_t)(level * peak / 255 * LED_PWM_MAX / 255));
}

int logicalState() {
  if (s_mode == Manual) return s_manual_on ? 1 : 0;
  return (s_mode == Idle) ? 0 : 1;
}

Mode mode() { return s_mode; }

}  // namespace led

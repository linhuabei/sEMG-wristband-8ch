#include <cassert>
#include <cstdint>
#include <cstdio>
#include <cstring>

#include "../Node_8CH_ADS1298/src/protocol/emg_frame.h"

namespace {

uint8_t g_buffer[4096];
size_t g_length = 0;

bool capture(const uint8_t *data, size_t len) {
  assert(g_length + len <= sizeof(g_buffer));
  std::memcpy(g_buffer + g_length, data, len);
  g_length += len;
  return true;
}

uint8_t xor_of(const uint8_t *data, size_t len) {
  uint8_t x = 0;
  for (size_t i = 0; i < len; i++) x ^= data[i];
  return x;
}

}  // namespace

int main() {
  emg::setSender(capture);

  // 24bit：单帧 28 字节，seq / XOR 正确
  emg::configure(true, 8);
  emg::reset();
  int32_t channels[8] = {1, 2, 3, 4, 5, 6, 7, 8};

  g_length = 0;
  emg::append(channels);
  assert(g_length == 0);  // 还没攒满 8 帧
  emg::flush();
  assert(g_length == 28);
  assert(g_buffer[0] == 0xAA);
  assert(g_buffer[1] == 0x55);
  assert(g_buffer[2] == 0);
  assert(g_buffer[27] == xor_of(g_buffer, 27));

  // 8 帧一批：seq 0..7，一批 224 字节
  g_length = 0;
  emg::reset();
  for (int i = 0; i < 8; i++) emg::append(channels);
  assert(g_length == 8 * 28);
  assert(g_buffer[2] == 0);
  assert(g_buffer[28 + 2] == 1);
  assert(g_buffer[7 * 28 + 2] == 7);

  // 16bit：单帧 20 字节，XOR 正确
  emg::configure(false, 8);
  emg::reset();
  g_length = 0;
  emg::append(channels);
  emg::flush();
  assert(g_length == 20);
  assert(g_buffer[19] == xor_of(g_buffer, 19));

  std::printf("emg_frame tests passed\n");
  return 0;
}

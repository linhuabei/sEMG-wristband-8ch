#include "emg_frame.h"

#include "../app/config.h"

namespace emg {
namespace {

Sender s_sender = nullptr;
bool s_bits24 = true;
uint8_t s_frames_per_notify = EMG_MAX_BATCH;
uint8_t s_batch_count = 0;
uint8_t s_seq = 0;
uint8_t s_buffer[12 * EMG_16BIT_FRAME];

uint32_t s_frame_count = 0;
uint32_t s_packet_count = 0;
uint32_t s_fail_count = 0;

}  // namespace

void setSender(Sender sender) { s_sender = sender; }

void configure(bool bits24, uint8_t framesPerNotify) {
  s_bits24 = bits24;
  s_frames_per_notify = framesPerNotify;
  s_batch_count = 0;
}

void reset() {
  s_seq = 0;
  s_batch_count = 0;
  s_frame_count = 0;
  s_packet_count = 0;
  s_fail_count = 0;
}

void resetBatch() { s_batch_count = 0; }

size_t frameLength() { return s_bits24 ? EMG_24BIT_FRAME : EMG_16BIT_FRAME; }

void flush() {
  if (s_batch_count == 0) return;
  const size_t len = s_batch_count * frameLength();
  if (s_sender != nullptr && s_sender(s_buffer, len)) {
    s_packet_count++;
  } else {
    s_fail_count++;
  }
  s_batch_count = 0;
}

void append(const int32_t ch[8]) {
  const size_t frame_len = frameLength();
  uint8_t *p = s_buffer + s_batch_count * frame_len;

  p[0] = 0xAA;
  p[1] = 0x55;
  p[2] = s_seq++;

  for (int i = 0; i < EMG_CHANNELS; i++) {
    if (s_bits24) {
      p[3 + i * 3] = (uint8_t)((ch[i] >> 16) & 0xFF);
      p[4 + i * 3] = (uint8_t)((ch[i] >> 8) & 0xFF);
      p[5 + i * 3] = (uint8_t)(ch[i] & 0xFF);
    } else {
      const int16_t v16 = (int16_t)(ch[i] >> 8);
      p[3 + i * 2] = (uint8_t)((v16 >> 8) & 0xFF);
      p[4 + i * 2] = (uint8_t)(v16 & 0xFF);
    }
  }

  uint8_t x = 0;
  for (size_t i = 0; i < frame_len - 1; i++) x ^= p[i];
  p[frame_len - 1] = x;

  s_frame_count++;
  s_batch_count++;
  if (s_batch_count >= s_frames_per_notify) flush();
}

bool bits24() { return s_bits24; }
uint8_t framesPerNotify() { return s_frames_per_notify; }
uint32_t frameCount() { return s_frame_count; }
uint32_t packetCount() { return s_packet_count; }
uint32_t failCount() { return s_fail_count; }

}  // namespace emg

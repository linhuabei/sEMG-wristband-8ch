// Arduino IDE 入口：打开这个 .ino，src/ 下的模块会自动参与编译。
#include "src/app/app.h"

void setup() { app::begin(); }
void loop() { app::loop(); }

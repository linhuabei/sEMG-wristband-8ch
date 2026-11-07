# 帧打包单元测试（PC 侧，不依赖 Arduino）

`Node_8CH_ADS1298/src/protocol/emg_frame.cpp` 不包含 Arduino 头文件，只依赖 C++ 标准库，所以可以在 PC 上单独编译测试。

```bash
g++ -std=c++17 -I Node_8CH_ADS1298/src \
    test/test_emg_frame.cpp \
    Node_8CH_ADS1298/src/protocol/emg_frame.cpp \
    -o test_emg_frame
./test_emg_frame
```

没有 `g++` 时，可以只看 `test/test_emg_frame.cpp` 里的断言，或改用任意本机 C++ 编译器。

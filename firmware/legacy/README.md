# legacy 单文件固件

这里保留重构前的单文件版本，仅作对照和回退，不参与 PlatformIO 构建。

| 文件 | 说明 |
| --- | --- |
| `ctrl_legacy_fw_ESP32S3.ino` | 更早的 legacy 单文件固件 |
| `Node_8CH_ADS1298_single_file.ino` | 重构前的当前固件，所有逻辑在一个 `.ino` 里 |

当前维护的代码在 `../Node_8CH_ADS1298/src/`，入口是 `../Node_8CH_ADS1298/Node_8CH_ADS1298.ino`（Arduino IDE / PlatformIO 共用）。

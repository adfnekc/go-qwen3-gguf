# RMS 爆炸追踪工具

## 背景

在 NEOX RoPE 修复前，隐藏层状态 RMS 在第 2 层 FFN 处出现灾难性增长（0.535 → 222.858）。
修复后该问题消失，说明 RMS 爆炸由 RoPE 类型错误导致。

## 文件

| 文件 | 描述 |
|------|------|
| `debug_rms.go` | 完整 28 层 RMS 追踪（逐层、逐操作） |
| `debug_layer2.go` | 详细逐操作追踪第 0-2 层 |
| `debug_forward.go` | Forward() 方法输出测试 — top token 预测 |
| `debug_no_qknorm.go` | 禁用 QK 归一化测试（证明 QK norm 非根因） |

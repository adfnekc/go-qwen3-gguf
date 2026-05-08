# 逐层追踪工具

## 文件

| 文件 | 描述 |
|------|------|
| `debug_layers.go` | 单层前向传播测试 |
| `debug_layers2.go` | 扩展版层诊断（含 top-10 token 预测） |
| `debug_alllayers.go` | 所有层诊断追踪 |
| `debug_isolate.go` | 逐层隔离 RMS 爆炸位置 |
| `debug_embedding.go` | 嵌入查找和输出投影测试（无层） |
| `debug_compare.go` | Forward() 与手动实现的逐层对比 |

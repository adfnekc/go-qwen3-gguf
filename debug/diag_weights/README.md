# 权重诊断工具

## 文件

| 文件 | 描述 |
|------|------|
| `debug_norm.go` | 输出归一化权重统计和 token 嵌入值 |
| `debug_norms.go` | QK 归一化权重分析（逐层 attn_q_norm、attn_k_norm） |
| `debug_scales.go` | 逐层权重缩放分析 |
| `debug_rmsnorm.go` | RMSNorm 验证 — 对比手动计算和函数计算 |

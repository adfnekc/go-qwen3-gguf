# Qwen3 GGUF 推理引擎 — 调试程序

本目录包含开发纯 Go Qwen3 GGUF 推理引擎过程中的诊断工具。
模型为 **Qwen3-0.6B**（Q8_0 量化）。

## 模型架构参数

| 参数 | 值 |
|------|------|
| 层数 | 28 |
| 嵌入维度 | 1024 |
| 注意力头数 | 16 |
| KV 头数 | 8 |
| 头维度 | 128 |
| FFN 维度 | 3072 |
| RoPE 类型 | NEOX（半分割配对） |
| RoPE 频率基数 | 1,000,000 |
| QK 归一化 | 启用（逐头 RMSNorm） |
| 权重绑定 | 启用（output = tok_embd） |

## 目录结构

### 按主题分类的诊断工具

| 目录 | 描述 | 文件数 |
|------|------|--------|
| `diag_tok/` | Tokenizer 编码/解码测试 | 3 |
| `diag_meta/` | 模型元数据与 tensor 形状 | 3 |
| `diag_weights/` | 权重统计、归一化分析 | 4 |
| `diag_rms/` | RMS 爆炸追踪（RoPE 修复前） | 4 |
| `diag_layers/` | 逐层前向传播追踪 | 6 |

### 按调查编号的结构化调查

| 目录 | 调查主题 | 文件 |
|------|----------|------|
| `03_weight_verification/` | 权重正确性验证（反量化、统计、前向对比） | 4 |
| `04_chat_inference/` | KV cache 正确性、逐层对比 | 4 |
| `05_q8_dequant_verify/` | Python 交叉验证 Q8_0 反量化 | 2 |
| `06_rope_fix/` | NEOX RoPE 修复与 logits 对比 | 3 |

## 运行方式

```bash
go run debug/<dir>/<name>.go /path/to/model.gguf
```

## 已修复 Bug 历史

1. **MatMul 方向**：从 `MatMul` 改为 `MatMulTransposed`（匹配 ggml_mul_mat）

2. **嵌入布局**：从 `EmbeddingLookupDimFirst` 改为 `EmbeddingLookupTokenFirst`

3. **GGUF tensor 偏移**：mmap 加载缺失 `DataStart` 偏移，导致加载错误权重

4. **RoPE 类型（关键修复）**：使用 NORMAL（连续配对 (0,1), (2,3)...）但
   Qwen3 需要 NEOX（半分割配对 (0,64), (1,65)...）。这是乱码输出的根因。
   修复后模型生成文本从乱码变成正确的推理链。

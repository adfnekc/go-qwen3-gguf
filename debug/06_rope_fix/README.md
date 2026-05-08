# 调查 06：修复 NEOX RoPE

## 目的

修复 Qwen3 模型中的 RoPE 实现：从 NORMAL 模式改为 NEOX 模式。

## 背景

llama.cpp 中 Qwen3 使用 `LLAMA_ROPE_TYPE_NEOX` (=2)：

- **NORMAL**（我们之前的实现）：连续配对 (0,1), (2,3), (4,5), ...
- **NEOX**（Qwen3 正确模式）：半分割配对 (0, dim/2), (1, dim/2+1), ..., (dim/2-1, dim-1)

对 head_dim=128，NEOX 配对方式为 (0,64), (1,65), ..., (63,127)。

## 修改

| 文件 | 修改内容 |
|------|----------|
| `math/math.go` | 新增 `RoPEMode` 类型和 `RoPE_NORMAL` / `RoPE_NEOX` 常量；`RoPE` 函数新增 `mode` 参数 |
| `model/qwen3.go` | 所有 RoPE 调用改为 `llmmath.RoPE_NEOX` |
| `debug/04_chat_inference/layer_by_layer.go` | 同步修复 debug 工具中的 RoPE 调用 |

## 验证

### 方法

编写了 Go 和 C++ 两个版本的 logits dump 程序，在相同 prompt 下对比 llama.cpp 和 Go 实现的 logits：

```bash
go run debug/06_rope_fix/dump_logits.go <model>
cd llama.cpp_ref && g++ -std=c++14 -I include -I ggml/include -I common \
  -o ../debug/06_rope_fix/dump_logits ../debug/06_rope_fix/dump_logits.cpp \
  -L build/bin -lllama -lggml-base -lggml-cpu -lggml -lmtmd \
  -lpthread -lm -ldl -Wl,-rpath,build/bin
../debug/06_rope_fix/dump_logits <model>
```

### 结果

修复后两者 logits 完全一致：

| Step | llama.cpp top-1 | Go top-1 |
|------|-----------------|----------|
| After prompt | `<think>` (151667) 30.98 | `<think>` (151667) 31.12 |
| After `<think>` | `\n` (198) 32.40 | `\n` (198) 32.54 |

### 推理效果

修复前：输出乱码
修复后：输出正确的推理链

```
Input: "What is 2+2?"
Output:
<think>
Okay, so the question is "What is 2+2?" and I need to figure out the answer.
Well, when you add two 2s together, you're combining them. So, 2 plus 2 equals 4.
...
</think>
```

## 文件

| 文件 | 描述 |
|------|------|
| `dump_logits.go` | Go 实现的 logits dump 工具，输出两步推理的 top-20 logits |
| `dump_logits.cpp` | C++ (llama.cpp) 实现的 logits dump 工具，用于对比验证 |
| `README.md` | 本文件 |

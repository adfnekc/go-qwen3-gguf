# 重构验证工具

验证 6 层工程化改造的正确性。

## 验证内容

1. **配置加载** — 确认模型参数正确解析
2. **权重加载** — 确认所有 28 层权重加载成功
3. **前向传播** — 对 [44, 402] token 运行 forward，输出 top-10 logits
4. **文本生成** — `"hi" → <think>\nOkay, the` (temp=0, tokens=5)

## 运行

```bash
go run debug/07_refactoring_verify/main.go /path/to/Qwen3-0.6B-Q8_0.gguf
```

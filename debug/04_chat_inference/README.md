# 调查 04：聊天格式推理验证

## 目的

验证使用聊天模板格式（chat template）输入时，模型推理的正确性。
`infer.go` 使用聊天格式输入时，第一个生成 token 是 `<think>`（正确），
但后续生成卡在 `<think>\n\n<think>\n` 的循环中，无法产生连贯文本。

## 假设

第一个 Forward 调用正确预测了 `<think>` 作为第一个 token，
但后续调用中 KV cache 或自回归状态可能出错，导致模型退化。

## 测试

| 文件 | 描述 |
|------|------|
| `debug_chat.go` | 使用聊天格式 prompt，逐 token 推理并打印详细中间状态 |

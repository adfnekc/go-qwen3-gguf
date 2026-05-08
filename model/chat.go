package model

// Tokenizer is the interface needed for chat template formatting.
type Tokenizer interface {
	Encode(text string) []int
}

// ApplyQwen3ChatTemplate applies the Qwen3 chat template to a user input.
// Produces: <|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
func ApplyQwen3ChatTemplate(tok Tokenizer, userInput string) []int {
	imStart := 151644
	imEnd := 151645

	tokens := []int{imStart}
	tokens = append(tokens, tok.Encode("user\n")...)
	tokens = append(tokens, tok.Encode(userInput)...)
	tokens = append(tokens, imEnd)
	tokens = append(tokens, 198) // "\n"
	tokens = append(tokens, imStart)
	tokens = append(tokens, tok.Encode("assistant\n")...)
	return tokens
}

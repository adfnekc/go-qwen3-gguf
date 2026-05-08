package model

// Special token IDs for the Qwen3 chat template.
const (
	TokenIMStart = 151644
	TokenIMEnd   = 151645
	TokenNewline = 198
)

// Tokenizer is the interface needed for chat template formatting.
type Tokenizer interface {
	Encode(text string) []int
}

// ApplyQwen3ChatTemplate applies the Qwen3 chat template to a user input.
// Produces: <|im_start|>user\n{input}<|im_end|>\n<|im_start|>assistant\n
func ApplyQwen3ChatTemplate(tok Tokenizer, userInput string) []int {
	tokens := []int{TokenIMStart}
	tokens = append(tokens, tok.Encode("user\n")...)
	tokens = append(tokens, tok.Encode(userInput)...)
	tokens = append(tokens, TokenIMEnd)
	tokens = append(tokens, TokenNewline)
	tokens = append(tokens, TokenIMStart)
	tokens = append(tokens, tok.Encode("assistant\n")...)
	return tokens
}

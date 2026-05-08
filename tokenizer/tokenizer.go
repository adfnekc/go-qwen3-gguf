package tokenizer

import (
	"fmt"
	"regexp"
	"sort"
	"strings"
)

// Tokenizer is the interface for token encoding/decoding.
type Tokenizer interface {
	Encode(text string) []int
	Decode(tokens []int) string
	GetToken(id int) (string, bool)
	GetVocabSize() int
}

var byteToUnicode map[byte]rune
var unicodeToByte map[rune]byte

func init() {
	byteToUnicode = make(map[byte]rune)
	unicodeToByte = make(map[rune]byte)

	n := 0
	for b := 0; b < 256; b++ {
		if b >= 33 && b <= 126 {
			byteToUnicode[byte(b)] = rune(b)
			unicodeToByte[rune(b)] = byte(b)
		} else if b >= 161 && b <= 172 {
			byteToUnicode[byte(b)] = rune(b)
			unicodeToByte[rune(b)] = byte(b)
		} else if b >= 174 && b <= 255 {
			byteToUnicode[byte(b)] = rune(b)
			unicodeToByte[rune(b)] = byte(b)
		} else {
			byteToUnicode[byte(b)] = rune(256 + n)
			unicodeToByte[rune(256+n)] = byte(b)
			n++
		}
	}
}

func bytesToUnicode(bytes []byte) string {
	runes := make([]rune, len(bytes))
	for i, b := range bytes {
		runes[i] = byteToUnicode[b]
	}
	return string(runes)
}

func unicodeToBytes(s string) []byte {
	var result []byte
	for _, r := range s {
		if b, ok := unicodeToByte[r]; ok {
			result = append(result, b)
		} else {
			result = append(result, []byte(string(r))...)
		}
	}
	return result
}

type GGUFReader interface {
	GetMetadata(key string) (interface{}, bool)
	GetMetadataString(key string) (string, bool)
	GetMetadataInt(key string) (int64, bool)
}

// Pre-tokenization regex patterns for different architectures
var (
	qwen3PreTokenizePattern = regexp.MustCompile(
		`'s|'t|'re|'ve|'m|'ll|'d|` +
			`[^\r\n\p{L}\p{N}]?\p{L}+|` +
			`\p{N}|` +
			` ?[^\s\p{L}\p{N}]+[\r\n]*|` +
			`\s*[\r\n]+|` +
			`\s+`,
	)
)

type BPETokenizer struct {
	vocab            map[string]int
	invVocab         map[int]string
	merges           map[string]int
	mergeRanks       []pair
	specialTokens    map[int]string
	invSpecialTokens map[string]int
	byteEncoder      map[byte]string
	byteDecoder      map[string]byte
	addBosToken      bool
	addEosToken      bool
	pretokType       string // type of pretokenizer: "qwen2", "gpt2", etc.
}

type pair struct {
	first  string
	second string
	rank   int
}

func NewTokenizerFromGGUF(reader GGUFReader) (*BPETokenizer, error) {
	tok := &BPETokenizer{
		vocab:            make(map[string]int),
		invVocab:         make(map[int]string),
		merges:           make(map[string]int),
		specialTokens:    make(map[int]string),
		invSpecialTokens: make(map[string]int),
		byteEncoder:      make(map[byte]string),
		byteDecoder:      make(map[string]byte),
	}

	for b := 0; b < 256; b++ {
		tok.byteEncoder[byte(b)] = string([]byte{byte(b)})
		tok.byteDecoder[string([]byte{byte(b)})] = byte(b)
	}

	vocab, ok := reader.GetMetadata("tokenizer.ggml.tokens")
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens not found")
	}

	vocabList, ok := vocab.([]string)
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.tokens is not []string")
	}

	for i, token := range vocabList {
		tok.vocab[token] = i
		tok.invVocab[i] = token
	}

	merges, ok := reader.GetMetadata("tokenizer.ggml.merges")
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.merges not found")
	}

	mergeList, ok := merges.([]string)
	if !ok {
		return nil, fmt.Errorf("tokenizer.ggml.merges is not []string")
	}

	tok.mergeRanks = make([]pair, len(mergeList))
	for i, merge := range mergeList {
		parts := strings.SplitN(merge, " ", 2)
		if len(parts) == 2 {
			tok.merges[merge] = i
			tok.mergeRanks[i] = pair{parts[0], parts[1], i}
		}
	}

	if bosTok, ok := reader.GetMetadataInt("tokenizer.ggml.bos_token_id"); ok {
		tok.specialTokens[int(bosTok)] = "<|beginoftext|>"
		tok.invSpecialTokens["<|beginoftext|>"] = int(bosTok)
	}

	if eosTok, ok := reader.GetMetadataInt("tokenizer.ggml.eos_token_id"); ok {
		tok.specialTokens[int(eosTok)] = "<|endoftext|>"
		tok.invSpecialTokens["<|endoftext|>"] = int(eosTok)
	}

	// Read the pretokenization type from GGUF metadata
	if pretok, ok := reader.GetMetadataString("tokenizer.ggml.pre"); ok {
		tok.pretokType = pretok
	}

	return tok, nil
}

func (tok *BPETokenizer) Encode(text string) []int {
	var tokens []int

	words := tok.preTokenize(text)

	for _, word := range words {
		wordTokens := tok.bpeEncode(word)
		tokens = append(tokens, wordTokens...)
	}

	return tokens
}

func (tok *BPETokenizer) preTokenize(text string) []string {
	switch tok.pretokType {
	case "qwen2", "qwen3", "gpt2":
		matches := qwen3PreTokenizePattern.FindAllString(text, -1)
		if len(matches) > 0 {
			return matches
		}
	}
	return splitIntoWords(text)
}

func splitIntoWords(text string) []string {
	var words []string
	var currentWord []rune

	for _, r := range text {
		if isWordBoundary(r) {
			if len(currentWord) > 0 {
				words = append(words, string(currentWord))
				currentWord = nil
			}
			words = append(words, string(r))
		} else {
			currentWord = append(currentWord, r)
		}
	}

	if len(currentWord) > 0 {
		words = append(words, string(currentWord))
	}

	return words
}

func isWordBoundary(r rune) bool {
	if r <= 32 || r == 127 {
		return true
	}
	if r >= '!' && r <= '/' {
		return true
	}
	if r >= ':' && r <= '@' {
		return true
	}
	if r >= '[' && r <= '`' {
		return true
	}
	if r >= '{' && r <= '~' {
		return true
	}
	return false
}

func (tok *BPETokenizer) bpeEncode(word string) []int {
	if len(word) == 0 {
		return nil
	}

	if id, ok := tok.vocab[word]; ok {
		return []int{id}
	}

	var tokens []string
	for _, b := range []byte(word) {
		tokens = append(tokens, bytesToUnicode([]byte{b}))
	}

	if len(tokens) == 0 {
		return []int{0}
	}

	for {
		bestPair, _ := tok.findBestPair(tokens)
		if bestPair == nil {
			break
		}

		tokens = tok.mergePair(tokens, bestPair)
		if len(tokens) == 1 {
			break
		}
	}

	result := make([]int, 0, len(tokens))
	for _, t := range tokens {
		if id, ok := tok.vocab[t]; ok {
			result = append(result, id)
		} else {
			for _, b := range []byte(t) {
				id, ok := tok.vocab[bytesToUnicode([]byte{b})]
				if ok {
					result = append(result, id)
				} else {
					result = append(result, 0)
				}
			}
		}
	}

	if len(result) == 0 {
		return []int{0}
	}

	return result
}

func (tok *BPETokenizer) findBestPair(tokens []string) (*pair, int) {
	if len(tokens) < 2 {
		return nil, -1
	}

	var bestPair *pair
	bestRank := len(tok.mergeRanks) + 1

	for i := 0; i < len(tokens)-1; i++ {
		pairKey := tokens[i] + " " + tokens[i+1]
		if rank, ok := tok.merges[pairKey]; ok {
			if rank < bestRank {
				bestRank = rank
				bestPair = &pair{tokens[i], tokens[i+1], rank}
			}
		}
	}

	return bestPair, bestRank
}

func (tok *BPETokenizer) mergePair(tokens []string, p *pair) []string {
	if p == nil {
		return tokens
	}

	var result []string
	i := 0
	for i < len(tokens) {
		if i < len(tokens)-1 && tokens[i] == p.first && tokens[i+1] == p.second {
			result = append(result, p.first+p.second)
			i += 2
		} else {
			result = append(result, tokens[i])
			i++
		}
	}
	return result
}

func (tok *BPETokenizer) Decode(tokens []int) string {
	var textBytes []byte

	for _, tokenId := range tokens {
		if token, ok := tok.invVocab[tokenId]; ok {
			textBytes = append(textBytes, unicodeToBytes(token)...)
		}
	}

	return string(textBytes)
}

func (tok *BPETokenizer) GetVocabSize() int {
	return len(tok.vocab)
}

func (tok *BPETokenizer) GetToken(id int) (string, bool) {
	token, ok := tok.invVocab[id]
	if !ok {
		return "", false
	}
	return token, ok
}

func (tok *BPETokenizer) GetTokenId(token string) (int, bool) {
	id, ok := tok.vocab[token]
	return id, ok
}

func (tok *BPETokenizer) IsSpecialToken(id int) bool {
	_, ok := tok.specialTokens[id]
	return ok
}

func (tok *BPETokenizer) GetSpecialTokenName(id int) (string, bool) {
	name, ok := tok.specialTokens[id]
	return name, ok
}

func (tok *BPETokenizer) GetSpecialTokenId(name string) (int, bool) {
	id, ok := tok.invSpecialTokens[name]
	return id, ok
}

type SimpleTokenizer struct {
	vocab    map[string]int
	invVocab map[int]string
}

func NewSimpleTokenizer() *SimpleTokenizer {
	tok := &SimpleTokenizer{
		vocab:    make(map[string]int),
		invVocab: make(map[int]string),
	}

	for i := 0; i < 256; i++ {
		token := string([]byte{byte(i)})
		tok.vocab[token] = i
		tok.invVocab[i] = token
	}

	return tok
}

func (tok *SimpleTokenizer) Encode(text string) []int {
	tokens := make([]int, len(text))
	for i, b := range []byte(text) {
		tokens[i] = int(b)
	}
	return tokens
}

func (tok *SimpleTokenizer) Decode(tokens []int) string {
	bytes := make([]byte, len(tokens))
	for i, t := range tokens {
		bytes[i] = byte(t % 256)
	}
	return string(bytes)
}

func (tok *SimpleTokenizer) GetToken(id int) (string, bool) {
	token, ok := tok.invVocab[id]
	return token, ok
}

func (tok *SimpleTokenizer) GetVocabSize() int {
	return len(tok.vocab)
}

var _ Tokenizer = (*BPETokenizer)(nil)
var _ Tokenizer = (*SimpleTokenizer)(nil)

func sortMergesByRank(merges map[string]int) []pair {
	var pairs []pair
	for merge, rank := range merges {
		parts := strings.SplitN(merge, " ", 2)
		if len(parts) == 2 {
			pairs = append(pairs, pair{parts[0], parts[1], rank})
		}
	}

	sort.Slice(pairs, func(i, j int) bool {
		return pairs[i].rank < pairs[j].rank
	})

	return pairs
}

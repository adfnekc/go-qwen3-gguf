package tokenizer

import (
	"fmt"
	"sort"
	"strings"
)

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

type Tokenizer struct {
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
}

type pair struct {
	first  string
	second string
	rank   int
}

func NewTokenizerFromGGUF(reader GGUFReader) (*Tokenizer, error) {
	tok := &Tokenizer{
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

	return tok, nil
}

func (tok *Tokenizer) Encode(text string) []int {
	var tokens []int

	normalized := tok.preTokenize(text)

	words := splitIntoWords(normalized)

	for _, word := range words {
		wordTokens := tok.bpeEncode(word)
		tokens = append(tokens, wordTokens...)
	}

	return tokens
}

func (tok *Tokenizer) preTokenize(text string) string {
	return text
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

func (tok *Tokenizer) bpeEncode(word string) []int {
	if len(word) == 0 {
		return nil
	}

	if id, ok := tok.vocab[word]; ok {
		return []int{id}
	}

	unicodeStr := bytesToUnicode([]byte(word))

	var tokens []string
	for _, r := range unicodeStr {
		charStr := string(r)
		if _, ok := tok.vocab[charStr]; ok {
			tokens = append(tokens, charStr)
		} else {
			bytes := []byte(string(r))
			for _, b := range bytes {
				byteStr := string(byteToUnicode[b])
				if _, ok := tok.vocab[byteStr]; ok {
					tokens = append(tokens, byteStr)
				} else {
					tokens = append(tokens, string(byteToUnicode[0]))
				}
			}
		}
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
			bytes := unicodeToBytes(t)
			for _, b := range bytes {
				byteStr := string(byteToUnicode[b])
				if id, ok := tok.vocab[byteStr]; ok {
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

func (tok *Tokenizer) findBestPair(tokens []string) (*pair, int) {
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

func (tok *Tokenizer) mergePair(tokens []string, p *pair) []string {
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

func (tok *Tokenizer) Decode(tokens []int) string {
	var textBytes []byte

	for _, tokenId := range tokens {
		if token, ok := tok.invVocab[tokenId]; ok {
			bytes := unicodeToBytes(token)
			textBytes = append(textBytes, bytes...)
		}
	}

	return string(textBytes)
}

func (tok *Tokenizer) GetVocabSize() int {
	return len(tok.vocab)
}

func (tok *Tokenizer) GetToken(id int) (string, bool) {
	token, ok := tok.invVocab[id]
	if !ok {
		return "", false
	}
	return string(unicodeToBytes(token)), ok
}

func (tok *Tokenizer) GetTokenId(token string) (int, bool) {
	id, ok := tok.vocab[token]
	return id, ok
}

func (tok *Tokenizer) IsSpecialToken(id int) bool {
	_, ok := tok.specialTokens[id]
	return ok
}

func (tok *Tokenizer) GetSpecialTokenName(id int) (string, bool) {
	name, ok := tok.specialTokens[id]
	return name, ok
}

func (tok *Tokenizer) GetSpecialTokenId(name string) (int, bool) {
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

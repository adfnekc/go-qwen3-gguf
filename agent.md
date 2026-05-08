# Agent Guidelines

## Non-Negotiable Testing Requirements

**AFTER EVERY CODE CHANGE, YOU MUST COMPLETE THESE STEPS:**

1. ✅ Run unit tests
2. ✅ Run the actual command-line program and verify output is correct
3. ✅ Check for regressions

**MOCK TESTS ARE NOT SUFFICIENT. YOU MUST RUN THE REAL PROGRAM.**

## Mandatory Testing Checklist

### Phase 1: Unit Tests (Must Pass)
```bash
go test ./... -v
```

### Phase 2: Command-Line Verification (CRITICAL - Must Do)
**Example for this project:**
```bash
go run infer.go /mnt/d/model/Qwen3-0.6B-Q8_0.gguf "hi"
```

**What to verify:**
- Output is not garbage/random
- Token IDs are reasonable (not all zeros, not random noise)
- Decoded text makes sense
- No crashes or panics

### Phase 3: Regression Check
- Ensure previously working functionality still works
- Compare output before/after changes if possible

## Testing Requirements

After implementing any new feature or bug fix, ALWAYS test the code to verify it achieves the intended purpose.

### Testing Checklist:
1. **Unit Tests**: Write unit tests for new functionality
2. **Integration Tests**: Test the feature in the context of the full system
3. **Command-Line Test**: Run the actual program with real inputs (CRITICAL)
4. **Edge Cases**: Test boundary conditions and edge cases
5. **Regression Tests**: Ensure existing functionality still works

### Example from Bug Fix:
- Bug: All output token IDs were zeros
- Root cause: Temperature sampling used `Argmax` instead of proper categorical sampling
- Fix: Implemented `SampleCategorical` function and updated Generate method
- Test: Created `TestGenerateWithMockWeights` and `TestGenerateWithTemperature` to verify non-zero token generation
- **MISTAKE**: Did NOT run the actual `go run infer.go` command with real model

## Key Lessons

1. **Temperature Sampling**: When temperature > 0, use categorical sampling from the probability distribution, NOT argmax
2. **Debug Output**: Add debug prints to trace values through the computation
3. **Test Coverage**: Always test both greedy decoding (temperature=0) and sampling (temperature>0) modes
4. **Minimal Fix Principle**: When fixing a bug, make the minimal necessary change. Don't introduce additional changes that may break existing functionality.
5. **Real Model Testing**: Unit tests with mock weights are not sufficient. ALWAYS run the actual command-line program with real inputs.
6. **Consistency Check**: When modifying code, ensure consistency across related code paths. For example, if weight tying is used, ensure the same tensor layout assumptions are used for both embedding lookup and output projection.
7. **Command-Line First**: Before claiming work is done, run the actual program users will use. Mock tests can pass while real usage fails.
8. **RoPE Type Matters**: Different models use different RoPE types. Qwen3 uses NEOX (half-split pairing), NOT the standard NORMAL (consecutive pairing). Always check the reference implementation (llama.cpp) for the correct RoPE type.
9. **Compare Logits Directly**: When debugging inference issues, write a small C/C++ program using llama.cpp API to dump logits and compare element-by-element with your implementation. This is the most reliable way to find discrepancies.
10. **Isolate by Elimination**: When the root cause is unclear, systematically verify each component: dequantization → weights → KV cache → individual layer output → logits. This narrows down the search space.
11. **Chinese Documentation**: Debug directory READMEs should be written in Chinese. Use `debug/<nn>_<descriptive_name>/` structure for investigation records.

## Important Warnings

1. **Embedding Layout Issues**: Different model formats may use different tensor layouts:
   - Token-first: `[vocab_size, embedding_dim]` - token 0's embedding is first embeddingDim elements
   - Dim-first: `[embedding_dim, vocab_size]` - dim 0's values for all tokens is first vocabSize elements
   
   Always verify the expected layout before making changes.

2. **Regression Risks**: When fixing one bug, be careful not to introduce new bugs. Always run regression tests after making changes.

3. **MOCK != REAL**: Mock data in tests often uses simplified patterns that don't match real model weights. The real model may have:
   - Different tensor layouts
   - Different value ranges
   - Special tokens and edge cases
   - Quantization artifacts

## Systematic Debugging Process

1. **Identify the root cause** before making any changes
2. **Make minimal changes** to address only the identified root cause
3. **Test thoroughly**:
   - Unit tests (`go test ./...`)
   - **Command-line test with real inputs** (CRITICAL)
   - Edge cases
4. **Verify no regressions** - ensure existing functionality still works
5. **Compare output** - if possible, compare output before/after changes

## When to Be Extra Careful

- When modifying core algorithms (embedding lookup, matrix multiplication, sampling)
- When changing assumptions about data formats or layouts
- When the fix involves multiple components or files
- When the bug involves "random" or "unpredictable" behavior

## Signature of Incomplete Testing

❌ "All tests pass" - but only mock tests were run
❌ "The logic looks correct" - but no actual verification
❌ "It should work" - based on theory, not practice

✅ "I ran `go run infer.go ...` and the output is X" - verified with real usage
✅ "Before the fix, output was A; after the fix, output is B" - compared
✅ "Unit tests pass, and I manually tested with 3 different inputs" - comprehensive

## Memorize This

**NO CODE CHANGE IS COMPLETE UNTIL YOU HAVE:**
1. Written and run unit tests
2. **Run the actual command-line program**
3. Verified the output is correct and meaningful

**MOCK TESTS ARE A START, NOT AN END.**

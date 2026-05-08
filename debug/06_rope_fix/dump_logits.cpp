#include "llama.h"
#include <algorithm>
#include <cstdio>
#include <cstring>
#include <string>
#include <vector>
#include <cmath>

int main(int argc, char ** argv) {
    std::string model_path = "/mnt/d/model/Qwen3-0.6B-Q8_0.gguf";
    if (argc > 1) model_path = argv[1];

    ggml_backend_load_all();

    llama_model_params model_params = llama_model_default_params();
    model_params.n_gpu_layers = 0;
    llama_model * model = llama_model_load_from_file(model_path.c_str(), model_params);
    if (!model) { fprintf(stderr, "failed to load model\n"); return 1; }

    const llama_vocab * vocab = llama_model_get_vocab(model);

    // Build prompt: <|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n
    std::string prompt = "<|im_start|>user\nhi<|im_end|>\n<|im_start|>assistant\n";

    const int n_prompt = -llama_tokenize(vocab, prompt.c_str(), prompt.size(), NULL, 0, true, true);
    std::vector<llama_token> prompt_tokens(n_prompt);
    if (llama_tokenize(vocab, prompt.c_str(), prompt.size(), prompt_tokens.data(), prompt_tokens.size(), true, true) < 0) {
        fprintf(stderr, "tokenize failed\n"); return 1;
    }

    // Step 1: process prompt
    llama_context_params ctx_params = llama_context_default_params();
    ctx_params.n_ctx = n_prompt + 32;
    ctx_params.n_batch = n_prompt;
    llama_context * ctx = llama_init_from_model(model, ctx_params);
    if (!ctx) { fprintf(stderr, "context init failed\n"); return 1; }

    printf("Prompt: %s\n", prompt.c_str());
    printf("Prompt tokens: ");
    for (auto id : prompt_tokens) printf("%d ", id);
    printf("\n");
    printf("Prompt length: %d\n", n_prompt);

    // Step 1: process the prompt
    llama_batch batch = llama_batch_get_one(prompt_tokens.data(), prompt_tokens.size());
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "decode step 1 failed\n"); return 1;
    }

    // Get logits for the last token
    float * logits = llama_get_logits(ctx);
    int n_vocab = llama_vocab_n_tokens(vocab);

    printf("\n=== After prompt (step 1) ===\n");
    // Find top 20
    std::vector<std::pair<int, float>> scored;
    for (int i = 0; i < n_vocab; i++) {
        scored.push_back({i, logits[i]});
    }
    std::partial_sort(scored.begin(), scored.begin() + 20, scored.end(),
        [](const std::pair<int,float> & a, const std::pair<int,float> & b) { return a.second > b.second; });
    for (int i = 0; i < 20; i++) {
        char buf[128];
        int n = llama_token_to_piece(vocab, scored[i].first, buf, sizeof(buf), 0, true);
        std::string s(n > 0 ? buf : "", n > 0 ? n : 0);
        printf("  %5d %-25s %.6f\n", scored[i].first, s.c_str(), scored[i].second);
    }

    // Step 2: <think> token
    llama_token think_tok = 151667;
    batch = llama_batch_get_one(&think_tok, 1);
    if (llama_decode(ctx, batch)) {
        fprintf(stderr, "decode step 2 failed\n"); return 1;
    }

    logits = llama_get_logits(ctx);
    printf("\n=== After <think> (step 2) ===\n");
    scored.clear();
    for (int i = 0; i < n_vocab; i++) {
        scored.push_back({i, logits[i]});
    }
    std::partial_sort(scored.begin(), scored.begin() + 20, scored.end(),
        [](const std::pair<int,float> & a, const std::pair<int,float> & b) { return a.second > b.second; });
    for (int i = 0; i < 20; i++) {
        char buf[128];
        int n = llama_token_to_piece(vocab, scored[i].first, buf, sizeof(buf), 0, true);
        std::string s(n > 0 ? buf : "", n > 0 ? n : 0);
        printf("  %5d %-25s %.6f\n", scored[i].first, s.c_str(), scored[i].second);
    }

    llama_free(ctx);
    llama_model_free(model);
    return 0;
}

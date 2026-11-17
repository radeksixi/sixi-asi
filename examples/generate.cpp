// The engine as a library: load a model, ask one question, print the answer. No HTTP involved.
//
//   example_generate MODEL.gguf [--draft DRAFTER.gguf] "question"
//   example_generate --reference "question"     built-in test backend: runs anywhere, prints noise
//
// Build inside this repository (target `example_generate`), or against an installed copy:
//   find_package(sixi_asi REQUIRED)  /  target_link_libraries(app PRIVATE sixi_asi::sixi_asi)
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/server.hpp"

#include <iostream>
#include <memory>
#include <string>

int main(int argc, char** argv) {
    using namespace sixi_asi;
    std::string model, draft, question;
    bool reference = false;
    for (int i = 1; i < argc; ++i) {
        const std::string arg = argv[i];
        if (arg == "--reference") {
            reference = true;
        } else if (arg == "--draft" && i + 1 < argc) {
            draft = argv[++i];
        } else if (model.empty() && !reference && i + 1 < argc) {
            model = arg;
        } else {
            question = arg;
        }
    }
    if (question.empty() || (model.empty() && !reference)) {
        std::cerr << "usage: example_generate MODEL.gguf [--draft DRAFTER.gguf] \"question\"\n"
                     "       example_generate --reference \"question\"\n";
        return 2;
    }

    // 1. A backend: the model on the GPU through the llama.cpp library, or the deterministic reference backend
    std::shared_ptr<core::IBackend> backend;
    const uint32_t n_ctx = 8192;
    if (reference) {
        backend = std::make_shared<core::ReferenceHybridBackend>(core::ReferenceHybridBackend::Config{});
    } else {
        auto llama = std::make_shared<core::LlamaCppBackend>();
        core::LlamaCppBackend::LoadOptions opts;
        opts.model_path = model;
        opts.draft_model_path = draft;   // optional block drafter (DSpine / DSpark / DFlash2) for speculative decoding
        opts.n_ctx = n_ctx;
        if (Status st = llama->load_model(opts); !st.ok()) {
            std::cerr << "load failed: " << st.message << "\n";
            return 1;
        }
        backend = llama;
    }

    // 2. The engine: chat template, speculative decoding, prefix cache. start() would add the HTTP server.
    server::ServerConfig cfg;
    cfg.n_ctx = n_ctx;
    cfg.max_adaptive_k = 15;
    cfg.admission.admitted = true;
    cfg.manifest.model_name = reference ? "reference" : "example";
    server::SixiServer engine(cfg, backend);

    // 3. One request, as over the API
    server::ChatCompletionRequest req;
    req.messages.push_back(server::ChatMessage{});
    req.messages.back().role = "user";
    req.messages.back().content = question;
    req.budgets.max_tokens = reference ? 32 : 1024;
    req.enable_thinking = false;
    auto res = engine.handle_chat_completion(req);
    if (!res.ok()) {
        std::cerr << "request failed: " << res.message() << "\n";
        return 1;
    }
    std::cout << res->content << "\n";
    std::cerr << "[" << res->completion_tokens << " tokens, " << res->decode_tok_per_s << " tok/s, mean accepted length "
              << res->mean_acceptance_length << ", " << res->finish_reason << "]\n";
    return 0;
}

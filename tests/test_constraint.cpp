// Constrained decoding: row constraining, the backend decorator under speculative decoding (reference backend and
// a synthetic constraint), and the grammar engine on a real vocabulary (tokenizer-only load, no GPU; skipped when
// no model file is available: set SIXI_TEST_VOCAB_MODEL to a GGUF).
#include "sixi_asi/common.hpp"
#include "sixi_asi/constraint.hpp"
#include "sixi_asi/core_execution.hpp"
#include "sixi_asi/speculative.hpp"

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <map>
#include <memory>
#include <string>
#include <vector>

static int g_last_fail_line = 0;

#define SIXI_ASSERT(cond)                                                                     \
    do {                                                                                        \
        if (!(cond)) {                                                                          \
            g_last_fail_line = __LINE__;                                                        \
            std::cerr << "  [ASSERT FAILED] " << #cond << " at " << __FILE__ << ":" << __LINE__ \
                      << "\n";                                                                  \
            return false;                                                                       \
        }                                                                                       \
    } while (0)

#define RUN_TEST(fn)                                                           \
    do {                                                                       \
        g_last_fail_line = 0;                                                  \
        if (fn()) {                                                            \
            std::cout << "[PASS] " << #fn << "\n";                             \
            ++passed;                                                          \
        } else {                                                               \
            std::cerr << "[FAIL] " << #fn << " (line " << g_last_fail_line << ")\n"; \
            ++failed;                                                          \
        }                                                                      \
    } while (0)

namespace {
using namespace sixi_asi;
using namespace sixi_asi::core;
using namespace sixi_asi::speculative;

// Synthetic constraint: after n accepted tokens only tokens with id % modulus == n % modulus are allowed.
class ModConstraint final : public ITokenConstraint {
public:
    explicit ModConstraint(uint32_t modulus, uint64_t n = 0) : mod_(modulus), n_(n) {}
    bool allows(int32_t token) override { return token >= 0 && static_cast<uint64_t>(token) % mod_ == n_ % mod_; }
    Status accept(int32_t token) override {
        if (!allows(token)) {
            return Status::Error(ErrorCode::InvalidArgument, "refused");
        }
        ++n_;
        return Status::Ok();
    }
    Status mask_all(float* logits, uint64_t vocab_size) override {
        ++full_scans;
        for (uint64_t i = 0; i < vocab_size; ++i) {
            if (!allows(static_cast<int32_t>(i))) {
                logits[i] = kMaskedLogit;
            }
        }
        return Status::Ok();
    }
    std::unique_ptr<ITokenConstraint> clone() const override { return std::make_unique<ModConstraint>(mod_, n_); }
    uint64_t accepted() const { return n_; }
    uint64_t full_scans = 0;

private:
    uint32_t mod_;
    uint64_t n_;
};

// Allows a fixed set of token ids, whatever came before.
class SetConstraint final : public ITokenConstraint {
public:
    explicit SetConstraint(std::vector<int32_t> ids) : ids_(std::move(ids)) {}
    bool allows(int32_t token) override { return std::find(ids_.begin(), ids_.end(), token) != ids_.end(); }
    Status accept(int32_t token) override { return allows(token) ? Status::Ok() : Status::Error(ErrorCode::InvalidArgument, "refused"); }
    Status mask_all(float* logits, uint64_t vocab_size) override {
        ++full_scans;
        for (uint64_t i = 0; i < vocab_size; ++i) {
            if (!allows(static_cast<int32_t>(i))) {
                logits[i] = kMaskedLogit;
            }
        }
        return Status::Ok();
    }
    std::unique_ptr<ITokenConstraint> clone() const override { return std::make_unique<SetConstraint>(ids_); }
    uint64_t full_scans = 0;

private:
    std::vector<int32_t> ids_;
};

std::vector<float> ramp_row(uint64_t n) {
    // distinct logits, descending with the id: token 0 is best
    std::vector<float> row(n);
    for (uint64_t i = 0; i < n; ++i) {
        row[i] = 100.0f - 0.01f * static_cast<float>(i);
    }
    return row;
}

std::vector<int32_t> best_ids(const std::vector<float>& row, size_t k) {
    std::vector<int32_t> ids(row.size());
    for (size_t i = 0; i < ids.size(); ++i) {
        ids[i] = static_cast<int32_t>(i);
    }
    std::stable_sort(ids.begin(), ids.end(), [&](int32_t a, int32_t b) { return row[a] > row[b]; });
    ids.resize(std::min(k, ids.size()));
    return ids;
}

bool test_constrain_row() {
    // keep = 1, best token allowed: the row is untouched
    {
        ModConstraint c(3);
        std::vector<float> row = ramp_row(5000);
        const std::vector<float> before = row;
        bool best_refused = true;
        SIXI_ASSERT(c.constrain(row.data(), row.size(), 1, &best_refused).ok());
        SIXI_ASSERT(row == before);
        SIXI_ASSERT(!best_refused);
        SIXI_ASSERT(c.full_scans == 0);
    }
    // keep = 1, best token refused: the best allowed token wins, without a scan of the vocabulary
    {
        ModConstraint c(3, 2);  // allowed: id % 3 == 2
        std::vector<float> row = ramp_row(5000);
        bool best_refused = false;
        SIXI_ASSERT(c.constrain(row.data(), row.size(), 1, &best_refused).ok());
        SIXI_ASSERT(best_refused);
        SIXI_ASSERT(DeterministicSampler::sample_greedy_argmax(row.data(), row.size()) == 2);
        SIXI_ASSERT(c.full_scans == 0);
    }
    // keep = 4: the four best logits are the four best allowed tokens
    {
        ModConstraint c(3, 1);  // allowed: 1, 4, 7, 10, ...
        std::vector<float> row = ramp_row(5000);
        SIXI_ASSERT(c.constrain(row.data(), row.size(), 4).ok());
        SIXI_ASSERT(best_ids(row, 4) == (std::vector<int32_t>{1, 4, 7, 10}));
    }
    // keep = 0: every refused token is masked, allowed ones keep their logits
    {
        ModConstraint c(3, 1);
        std::vector<float> row = ramp_row(300);
        const std::vector<float> before = row;
        SIXI_ASSERT(c.constrain(row.data(), row.size(), 0).ok());
        for (size_t i = 0; i < row.size(); ++i) {
            SIXI_ASSERT(row[i] == ((i % 3 == 1) ? before[i] : kMaskedLogit));
        }
    }
    // The only allowed token ranks far below the search width: the whole vocabulary is scanned, greedy stays exact
    {
        SetConstraint c({4321});
        std::vector<float> row = ramp_row(5000);
        SIXI_ASSERT(c.constrain(row.data(), row.size(), 1).ok());
        SIXI_ASSERT(DeterministicSampler::sample_greedy_argmax(row.data(), row.size()) == 4321);
        SIXI_ASSERT(c.full_scans == 1);
    }
    // Fewer allowed tokens than `keep` among the best: they stay, everything else is masked
    {
        SetConstraint c({5, 9, 4000});
        std::vector<float> row = ramp_row(5000);
        SIXI_ASSERT(c.constrain(row.data(), row.size(), 20).ok());
        size_t finite = 0;
        for (float v : row) {
            finite += v > kMaskedLogit;
        }
        SIXI_ASSERT(finite == 2);
        SIXI_ASSERT(row[5] > kMaskedLogit && row[9] > kMaskedLogit);
    }
    // Vocabulary smaller than the search width
    {
        ModConstraint c(7, 6);
        std::vector<float> row = ramp_row(40);
        SIXI_ASSERT(c.constrain(row.data(), row.size(), 3).ok());
        SIXI_ASSERT(best_ids(row, 3) == (std::vector<int32_t>{6, 13, 20}));
    }
    // Nothing allowed: an error, not a silently unconstrained or arbitrary token
    {
        SetConstraint c({});
        std::vector<float> row = ramp_row(2000);
        SIXI_ASSERT(!c.constrain(row.data(), row.size(), 1).ok());
        std::vector<float> row2 = ramp_row(2000);
        SIXI_ASSERT(!c.constrain(row2.data(), row2.size(), 0).ok());
        std::vector<float> row3 = ramp_row(30);
        SIXI_ASSERT(!c.constrain(row3.data(), row3.size(), 5).ok());
    }
    return true;
}

ReferenceHybridBackend::Config ref_cfg(bool ring) {
    ReferenceHybridBackend::Config cfg;
    cfg.vocab_size = 256;
    cfg.hidden_size = 64;
    cfg.n_layers = 4;
    cfg.n_experts = 8;
    cfg.top_k = 2;
    cfg.gdn_dtype = GdnDType::FP32;
    cfg.weights_seed = 424242ULL;
    cfg.enable_fast_rollback_ring = ring;
    cfg.rollback_ring = 8;
    return cfg;
}

const std::vector<int32_t> kPrompt = {1, 7, 19, 31, 43};

// Constrained greedy decoding written out by hand: the reference every decorated run must reproduce. The
// constraint starts after `start_token` (-1: at once).
std::vector<int32_t> reference_constrained(size_t n_gen, uint32_t modulus, int32_t start_token) {
    ReferenceHybridBackend backend(ref_cfg(false));
    std::vector<float> logits = backend.forward_tokens(kPrompt).value();
    ModConstraint c(modulus);
    bool active = start_token < 0;
    std::vector<int32_t> out;
    for (size_t i = 0; i < n_gen; ++i) {
        int32_t best = -1;
        for (size_t t = 0; t < logits.size(); ++t) {
            if (active && !c.allows(static_cast<int32_t>(t))) {
                continue;
            }
            if (best < 0 || logits[t] > logits[static_cast<size_t>(best)]) {
                best = static_cast<int32_t>(t);
            }
        }
        out.push_back(best);
        if (active) {
            (void)c.accept(best);
        } else if (best == start_token) {
            active = true;
        }
        logits = backend.forward_tokens({best}).value();
    }
    return out;
}

struct Run {
    bool ok = false;
    std::vector<int32_t> tokens;
    ConstrainedBackend::Stats stats;
    uint64_t rollbacks = 0;
    size_t n_fed = 0;   // tokens generated before `tokens` was cut to the requested length (a step may overshoot)
};

Run run_decorated(size_t n_gen, uint32_t modulus, int32_t start_token, uint32_t k, bool ring, bool perturbed_drafter,
                  float temperature = 0.0f, uint64_t seed = 100ULL) {
    Run run;
    auto raw = std::make_shared<ReferenceHybridBackend>(ref_cfg(ring));
    auto init = raw->forward_tokens(kPrompt);
    if (!init.ok()) {
        return run;
    }
    std::vector<float> logits = init.value();

    ModConstraint origin(modulus);
    ConstrainedBackend::Options copts;
    copts.start_token = start_token;
    copts.keep = temperature > 0.0f ? 20u : 1u;
    auto backend = std::make_shared<ConstrainedBackend>(raw, origin, copts);
    if (!backend->constrain_current(logits).ok()) {
        return run;
    }

    SpeculativeConfig scfg;
    scfg.execution_mode = ExecutionMode::StrictReplay;
    scfg.draft_k = k;
    scfg.adaptive_k = false;
    scfg.enable_suffix_drafter = k > 0;
    scfg.neural_drafter = k > 0 ? NeuralDrafterType::MTP : NeuralDrafterType::None;
    std::shared_ptr<INeuralDrafter> drafter;
    if (k > 0) {
        ReferenceHybridBackend::Config dcfg = ref_cfg(false);
        if (perturbed_drafter) {
            dcfg.weights_seed += 1;
        }
        drafter = std::make_shared<ProgrammableNeuralDrafter>(NeuralDrafterType::MTP, std::make_shared<ReferenceHybridBackend>(dcfg), false);
    }
    SpeculativeDecoder decoder(scfg, backend, drafter);

    SamplingParams sp;
    sp.temperature = temperature;
    sp.top_k = 20;
    sp.top_p = 0.95f;
    sp.mode = ExecutionMode::StrictReplay;
    DeterministicRng rng(seed, 200ULL);
    std::vector<int32_t> context = kPrompt;
    while (context.size() - kPrompt.size() < n_gen) {
        auto step = decoder.step(context, logits, sp, rng);
        if (!step.ok()) {
            std::cerr << "  step failed: " << step.status().message << "\n";
            return run;
        }
    }
    run.tokens.assign(context.begin() + static_cast<std::ptrdiff_t>(kPrompt.size()), context.end());
    run.n_fed = run.tokens.size();
    run.tokens.resize(n_gen);
    run.stats = backend->stats();
    run.rollbacks = decoder.stats().rollbacks_executed;
    run.ok = true;
    return run;
}

bool test_decorator_greedy_ar_matches_reference() {
    const std::vector<int32_t> ref = reference_constrained(80, 3, -1);
    for (size_t i = 0; i < ref.size(); ++i) {
        SIXI_ASSERT(static_cast<size_t>(ref[i]) % 3 == i % 3);
    }
    for (bool ring : {false, true}) {
        Run run = run_decorated(80, 3, -1, 0, ring, false);
        SIXI_ASSERT(run.ok);
        SIXI_ASSERT(run.tokens == ref);
        SIXI_ASSERT(run.stats.rows >= 80);
        SIXI_ASSERT(run.stats.rows_changed > 0);  // the model alone does not follow a mod-3 pattern
    }
    return true;
}

bool test_decorator_speculative_equals_ar() {
    // Drafts come from an unconstrained drafter, so most of them are refused by the constraint: the decorator has
    // to stop at the refused token, the decoder to reject it, and the rollback to bring the constraint back.
    const std::vector<int32_t> ref = reference_constrained(96, 3, -1);
    // Positions where the model was overruled, counted without drafts for the same number of tokens: speculative
    // rows that are thrown away must not add to it, whatever the draft length
    std::map<size_t, uint64_t> overruled_by_len;
    auto overruled_for = [&](size_t n) {
        auto it = overruled_by_len.find(n);
        if (it == overruled_by_len.end()) {
            it = overruled_by_len.emplace(n, run_decorated(n, 3, -1, 0, true, false).stats.rows_changed).first;
        }
        return it->second;
    };
    SIXI_ASSERT(overruled_for(96) > 0 && overruled_for(96) <= 97);
    uint64_t refused_total = 0;
    for (bool ring : {true, false}) {
        for (uint32_t k : {1u, 2u, 4u, 6u}) {
            for (bool perturbed : {false, true}) {
                Run run = run_decorated(96, 3, -1, k, ring, perturbed);
                SIXI_ASSERT(run.ok);
                SIXI_ASSERT(run.tokens == ref);
                refused_total += run.stats.tokens_refused;
                const uint64_t overruled = overruled_for(run.n_fed);
                if (run.stats.rows_changed != overruled) {
                    std::cerr << "  k=" << k << " ring=" << ring << " n=" << run.n_fed << ": rows_changed " << run.stats.rows_changed
                              << " != " << overruled << "\n";
                }
                SIXI_ASSERT(run.stats.rows_changed == overruled);
            }
        }
    }
    SIXI_ASSERT(refused_total > 0);
    // A constraint the model follows by itself changes nothing and refuses nothing
    {
        const std::vector<int32_t> free_ref = reference_constrained(64, 1, -1);
        Run run = run_decorated(64, 1, -1, 4, true, false);
        SIXI_ASSERT(run.ok);
        SIXI_ASSERT(run.tokens == free_ref);
        SIXI_ASSERT(run.stats.rows_changed == 0 && run.stats.tokens_refused == 0);
    }
    return true;
}

bool test_decorator_start_token() {
    // The constraint begins after the first occurrence of a token the free-running model emits itself
    const std::vector<int32_t> free_ref = reference_constrained(12, 1, -1);
    const int32_t start = free_ref[5];
    size_t first = 0;
    while (free_ref[first] != start) {
        ++first;
    }
    const std::vector<int32_t> ref = reference_constrained(72, 3, start);
    for (size_t i = 0; i <= first; ++i) {
        SIXI_ASSERT(ref[i] == free_ref[i]);
    }
    for (size_t i = first + 1; i < ref.size(); ++i) {
        SIXI_ASSERT(static_cast<size_t>(ref[i]) % 3 == (i - first - 1) % 3);
    }
    for (bool ring : {true, false}) {
        for (uint32_t k : {0u, 2u, 4u, 6u}) {
            Run run = run_decorated(72, 3, start, k, ring, k == 4);
            SIXI_ASSERT(run.ok);
            SIXI_ASSERT(run.tokens == ref);
        }
    }
    return true;
}

bool test_decorator_sampled_tokens_allowed() {
    for (bool ring : {true, false}) {
        for (uint32_t k : {0u, 4u}) {
            Run a = run_decorated(120, 3, -1, k, ring, true, 0.8f, 7ULL);
            Run b = run_decorated(120, 3, -1, k, ring, true, 0.8f, 7ULL);
            Run c = run_decorated(120, 3, -1, k, ring, true, 0.8f, 8ULL);
            SIXI_ASSERT(a.ok && b.ok && c.ok);
            SIXI_ASSERT(a.tokens == b.tokens);   // same seed, same text
            SIXI_ASSERT(a.tokens != c.tokens);   // it does sample
            for (const Run* r : {&a, &c}) {
                for (size_t i = 0; i < r->tokens.size(); ++i) {
                    SIXI_ASSERT(static_cast<size_t>(r->tokens[i]) % 3 == i % 3);
                }
            }
        }
    }
    return true;
}

bool test_decorator_long_rollback() {
    // restore_state far behind the current position: the constraint is rebuilt from its checkpoints
    auto raw = std::make_shared<ReferenceHybridBackend>(ref_cfg(false));
    std::vector<float> logits = raw->forward_tokens(kPrompt).value();
    ModConstraint origin(5);
    auto backend = std::make_shared<ConstrainedBackend>(raw, origin, ConstrainedBackend::Options{});
    SIXI_ASSERT(backend->constrain_current(logits).ok());
    DeterministicRng rng(1ULL, 2ULL);

    auto generate = [&](size_t n, std::vector<float> lg) {
        std::vector<int32_t> out;
        for (size_t i = 0; i < n; ++i) {
            const int32_t t = DeterministicSampler::sample_greedy_argmax(lg.data(), lg.size());
            out.push_back(t);
            auto fw = backend->forward_tokens({t});
            if (!fw.ok()) {
                return std::vector<int32_t>{};
            }
            lg = fw.value();
        }
        return out;
    };

    const std::vector<int32_t> head = generate(37, logits);   // past one checkpoint (32)
    SIXI_ASSERT(head.size() == 37);
    auto ckpt = backend->checkpoint_state(rng);
    SIXI_ASSERT(ckpt.ok());

    // Reference continuation from a fresh, identical setup
    const std::vector<int32_t> ref = reference_constrained(37 + 110, 5, -1);
    SIXI_ASSERT(std::vector<int32_t>(ref.begin(), ref.begin() + 37) == head);

    std::vector<float> at37;
    {
        // constrained logits after `head`, from a second backend
        ReferenceHybridBackend twin(ref_cfg(false));
        std::vector<int32_t> all = kPrompt;
        all.insert(all.end(), head.begin(), head.end());
        at37 = twin.forward_tokens(all).value();
        ModConstraint c(5, 37);
        SIXI_ASSERT(c.constrain(at37.data(), at37.size(), 1).ok());
    }
    const std::vector<int32_t> tail1 = generate(110, at37);    // crosses three more checkpoints
    SIXI_ASSERT(tail1 == std::vector<int32_t>(ref.begin() + 37, ref.end()));
    SIXI_ASSERT(backend->restore_state(ckpt.value(), rng).ok());
    const std::vector<int32_t> tail2 = generate(110, at37);
    SIXI_ASSERT(tail2 == tail1);
    return true;
}

// ---- Real vocabulary and grammar (tokenizer-only load) -------------------------------------------------------

std::string vocab_model_path() {
    if (const char* env = std::getenv("SIXI_TEST_VOCAB_MODEL"); env != nullptr && env[0] != '\0') {
        return env;
    }
    const char* home = std::getenv("HOME");
    const std::string def = std::string(home != nullptr ? home : "") + "/models/qwen3.8-27b/Qwen3.8-27B-UD-Q4_K_XL-v3.1.gguf";
    return std::filesystem::exists(def) ? def : std::string();
}

std::shared_ptr<LlamaCppBackend> g_vocab;   // loaded once
bool g_vocab_tried = false;

std::shared_ptr<LlamaCppBackend> vocab_backend() {
    if (!g_vocab_tried) {
        g_vocab_tried = true;
        const std::string path = vocab_model_path();
        if (!path.empty()) {
            auto b = std::make_shared<LlamaCppBackend>();
            LlamaCppBackend::LoadOptions lo;
            lo.model_path = path;
            lo.vocab_only = true;
            Status st = b->load_model(lo);
            if (st.ok()) {
                g_vocab = b;
            } else {
                std::cerr << "  tokenizer-only load failed: " << st.message << "\n";
            }
        }
    }
    return g_vocab;
}

const char* kPersonSchema =
    R"({"type":"object","properties":{"name":{"type":"string"},"age":{"type":"integer"},"tags":{"type":"array","items":{"type":"string"}}},)"
    R"("required":["name","age","tags"],"additionalProperties":false})";

// The grammar the server builds: optional newlines (template glue after </think>) and then the schema's root
std::string wrapped(const std::string& gbnf) { return gbnf + "\nsixi-root ::= [\\n]* root\n"; }

bool test_grammar_engine() {
    auto b = vocab_backend();
    if (!b) {
        std::cout << "  (skipped: no model file for a tokenizer-only load)\n";
        return true;
    }
    SIXI_ASSERT(b->vocab_size() > 1000);
    auto ids = b->tokenize("Hello, world", false);
    SIXI_ASSERT(ids.ok() && !ids->empty());
    SIXI_ASSERT(b->detokenize(ids.value()).value() == "Hello, world");

    // schema -> grammar; a broken schema is the caller's error
    auto g = b->json_schema_to_grammar(kPersonSchema);
    if (!g.ok()) {
        std::cout << "  (skipped: " << g.status().message << ")\n";
        return true;
    }
    SIXI_ASSERT(g->find("root ::=") != std::string::npos);
    SIXI_ASSERT(b->json_schema_to_grammar("{\"type\": ").code() == ErrorCode::InvalidArgument);
    SIXI_ASSERT(b->json_schema_to_grammar(R"({"type":"string","pattern":"^(a|b"})").code() == ErrorCode::InvalidArgument);
    SIXI_ASSERT(!b->make_grammar_constraint("root ::= (", "root").ok());
    SIXI_ASSERT(!b->make_grammar_constraint(g.value(), "no-such-rule").ok());

    auto made = b->make_grammar_constraint(wrapped(g.value()), "sixi-root");
    SIXI_ASSERT(made.ok());
    std::unique_ptr<ITokenConstraint> c = std::move(made.value());
    const int32_t eos = b->eos_token_id();
    const int32_t hello = b->tokenize("Hello", false).value().front();
    const int32_t brace = b->tokenize("{", false).value().front();
    const int32_t nl = b->tokenize("\n", false).value().front();
    SIXI_ASSERT(!c->allows(hello));
    SIXI_ASSERT(c->allows(brace));
    SIXI_ASSERT(c->allows(nl));
    SIXI_ASSERT(!c->allows(eos));
    SIXI_ASSERT(!c->accept(hello).ok());   // refused, and the state survives it
    SIXI_ASSERT(c->allows(brace));

    // A valid document is accepted token by token; afterwards only end of generation is
    const std::string doc = "\n\n{\"name\": \"Ada Lovelace\", \"age\": 36, \"tags\": [\"math\", \"poetry\"]}";
    std::unique_ptr<ITokenConstraint> before_age;
    const auto doc_ids = b->tokenize(doc, false).value();
    for (size_t i = 0; i < doc_ids.size(); ++i) {
        if (i == 6) {
            before_age = c->clone();
        }
        SIXI_ASSERT(c->accept(doc_ids[i]).ok());
    }
    SIXI_ASSERT(c->allows(eos));
    SIXI_ASSERT(!c->allows(nl) && !c->allows(brace) && !c->allows(hello));

    // The clone is an independent state at token 6
    SIXI_ASSERT(before_age != nullptr);
    SIXI_ASSERT(!before_age->allows(eos));
    SIXI_ASSERT(before_age->accept(doc_ids[6]).ok());

    // A document that breaks the schema is refused at the offending token
    {
        auto c2 = b->make_grammar_constraint(wrapped(g.value()), "sixi-root").value();
        const auto bad = b->tokenize("{\"name\": \"Ada\", \"age\": \"old\"", false).value();
        size_t n_ok = 0;
        while (n_ok < bad.size() && c2->accept(bad[n_ok]).ok()) {
            ++n_ok;
        }
        SIXI_ASSERT(n_ok < bad.size());
        SIXI_ASSERT(b->detokenize(std::vector<int32_t>(bad.begin(), bad.begin() + static_cast<std::ptrdiff_t>(n_ok))).value().find("\"age\":") != std::string::npos);
    }

    // constrain(): the model's favourite is refused, the best allowed token takes its place; the whole-vocabulary
    // scan agrees with single checks
    {
        auto c3 = b->make_grammar_constraint(wrapped(g.value()), "sixi-root").value();
        const uint64_t nv = b->vocab_size();
        std::vector<float> row(nv, 0.0f);
        row[static_cast<size_t>(hello)] = 10.0f;
        row[static_cast<size_t>(brace)] = 5.0f;
        bool best_refused = false;
        SIXI_ASSERT(c3->constrain(row.data(), nv, 1, &best_refused).ok());
        SIXI_ASSERT(best_refused);
        SIXI_ASSERT(DeterministicSampler::sample_greedy_argmax(row.data(), nv) == brace);

        std::vector<float> all(nv, 1.0f);
        SIXI_ASSERT(c3->constrain(all.data(), nv, 0).ok());
        size_t allowed = 0;
        for (uint64_t t = 0; t < nv; ++t) {
            allowed += all[t] > kMaskedLogit;
        }
        SIXI_ASSERT(allowed > 0 && allowed < 200);
        for (uint64_t t = 0; t < nv; t += 97) {
            SIXI_ASSERT(c3->allows(static_cast<int32_t>(t)) == (all[t] > kMaskedLogit));
        }
        SIXI_ASSERT(all[static_cast<size_t>(brace)] > kMaskedLogit && all[static_cast<size_t>(hello)] == kMaskedLogit);
    }
    return true;
}

// A "model" over the real vocabulary that wants to say a fixed text after the prompt, whatever it is told, and
// would rather stop than say anything else. Its drafter proposes the same text. Position-only state, so rollback
// is trivial.
class ScriptedBackend final : public IBackend {
public:
    // `second`: the model's runner-up at each position (same indexing as `script`), e.g. the correct ending of a
    // text whose favourite ending is broken
    ScriptedBackend(std::shared_ptr<IBackend> vocab, std::vector<int32_t> script, bool fast_rollback, std::vector<int32_t> second = {})
        : vocab_(std::move(vocab)), script_(std::move(script)), second_(std::move(second)), fast_(fast_rollback) {}

    std::string backend_name() const override { return "scripted"; }
    uint64_t vocab_size() const override { return vocab_->vocab_size(); }
    int32_t eos_token_id() const override { return vocab_->eos_token_id(); }
    bool is_end_of_generation(int32_t t) const noexcept override { return vocab_->is_end_of_generation(t); }
    Result<std::vector<int32_t>> tokenize(const std::string& text, bool add_bos = true) override { return vocab_->tokenize(text, add_bos); }
    Result<std::string> detokenize(const std::vector<int32_t>& tokens) override { return vocab_->detokenize(tokens); }

    Status reset_state() override { tokens_.clear(); return Status::Ok(); }
    Result<BackendSequenceState> checkpoint_state(const DeterministicRng& rng) override {
        BackendSequenceState s;
        s.seq_len = tokens_.size();
        s.tokens = tokens_;
        s.rng_state = rng.checkpoint();
        return Result<BackendSequenceState>(std::move(s));
    }
    Status restore_state(const BackendSequenceState& s, DeterministicRng& rng) override {
        tokens_ = s.tokens;
        rng.restore(s.rng_state);
        return Status::Ok();
    }
    Result<std::vector<float>> forward_tokens(const std::vector<int32_t>& in, uint32_t = 512) override {
        tokens_.insert(tokens_.end(), in.begin(), in.end());
        return Result<std::vector<float>>(row(tokens_.size()));
    }
    Result<std::vector<std::vector<float>>> forward_sequence_all_logits(const std::vector<int32_t>& in) override {
        std::vector<std::vector<float>> rows;
        for (int32_t t : in) {
            tokens_.push_back(t);
            rows.push_back(row(tokens_.size()));
        }
        return Result<std::vector<std::vector<float>>>(std::move(rows));
    }
    std::vector<int32_t> context_tokens() const override { return tokens_; }
    bool supports_fast_prefix_rollback() const noexcept override { return fast_; }
    uint32_t fast_rollback_window() const noexcept override { return fast_ ? 8u : 0u; }
    uint64_t current_seq_len() const noexcept override { return tokens_.size(); }
    Status rollback_to_position(uint64_t n) override {
        if (n > tokens_.size()) {
            return Status::Error(ErrorCode::InvalidArgument, "rollback past the end");
        }
        tokens_.resize(n);
        return Status::Ok();
    }
    bool has_embedded_mtp_drafter() const noexcept override { return true; }
    uint32_t preferred_draft_k() const noexcept override { return 4; }
    Result<std::vector<int32_t>> draft_embedded_mtp(int32_t, uint32_t k) override {
        // the text that would follow the anchor if the model had its way
        std::vector<int32_t> out;
        const size_t next = tokens_.size() - prompt_len_ + 1;   // script index after the anchor
        for (size_t i = next; i < script_.size() && out.size() < k; ++i) {
            out.push_back(script_[i]);
        }
        if (out.empty()) {
            return Result<std::vector<int32_t>>(ErrorCode::BackendError, "nothing to draft");
        }
        return Result<std::vector<int32_t>>(std::move(out));
    }
    void set_prompt_len(size_t n) { prompt_len_ = n; }

private:
    std::vector<float> row(size_t len) const {
        std::vector<float> r(vocab_size(), 0.0f);
        const size_t idx = len - prompt_len_;
        // favourite: the next token of the text, then stopping; runner-up: the second text; stopping is never
        // worse than third
        const size_t eos = static_cast<size_t>(eos_token_id());
        if (idx < script_.size()) {
            r[static_cast<size_t>(script_[idx])] = 10.0f;
        } else {
            r[eos] = 10.0f;
        }
        if (idx < second_.size() && r[static_cast<size_t>(second_[idx])] < 7.0f) {
            r[static_cast<size_t>(second_[idx])] = 7.0f;
        }
        if (r[eos] < 5.0f) {
            r[eos] = 5.0f;
        }
        return r;
    }
    std::shared_ptr<IBackend> vocab_;
    std::vector<int32_t> script_;
    std::vector<int32_t> second_;
    std::vector<int32_t> tokens_;
    size_t prompt_len_ = 0;
    bool fast_;
};

struct Scripted {
    bool ok = false;
    std::string text;
    bool ended = false;   // stopped at end of generation (not at the token limit)
    ConstrainedBackend::Stats stats;
};

// schema: nullptr = unconstrained. second_text: the model's runner-up text (empty: none).
Scripted run_scripted(const std::string& model_text, const char* schema, bool thinking, uint32_t k, bool fast_rollback,
                      const std::string& second_text = std::string()) {
    Scripted out;
    auto vb = vocab_backend();
    const bool constrained = schema != nullptr;
    const std::vector<int32_t> prompt = vb->tokenize("Describe Ada as JSON.", false).value();
    auto raw = std::make_shared<ScriptedBackend>(
        vb, vb->tokenize(model_text, false).value(), fast_rollback,
        second_text.empty() ? std::vector<int32_t>{} : vb->tokenize(second_text, false).value());
    raw->set_prompt_len(prompt.size());
    std::vector<float> logits = raw->forward_tokens(prompt).value();

    std::shared_ptr<IBackend> backend = raw;
    std::shared_ptr<ConstrainedBackend> cb;
    if (constrained) {
        // as the server builds it: the schema's root, with newline glue in front only after reasoning
        auto g = vb->json_schema_to_grammar(schema);
        if (!g.ok()) {
            return out;
        }
        auto c = thinking ? vb->make_grammar_constraint(wrapped(g.value()), "sixi-root") : vb->make_grammar_constraint(g.value(), "root");
        if (!c.ok()) {
            return out;
        }
        ConstrainedBackend::Options o;
        if (thinking) {
            const auto close = vb->tokenize("</think>", false).value();
            if (close.size() != 1) {
                return out;
            }
            o.start_token = close.front();
        }
        cb = std::make_shared<ConstrainedBackend>(raw, *c.value(), o);
        if (!cb->constrain_current(logits).ok()) {
            return out;
        }
        backend = cb;
    }

    SpeculativeConfig scfg;
    scfg.draft_k = k;
    scfg.adaptive_k = false;
    scfg.enable_suffix_drafter = false;
    scfg.neural_drafter = k > 0 ? NeuralDrafterType::MTP : NeuralDrafterType::None;
    SpeculativeDecoder decoder(scfg, backend, nullptr);
    SamplingParams sp;
    sp.temperature = 0.0f;
    DeterministicRng rng(3ULL, 4ULL);
    std::vector<int32_t> context = prompt;
    std::vector<int32_t> gen;
    while (gen.size() < 200 && !out.ended) {
        auto step = decoder.step(context, logits, sp, rng);
        if (!step.ok()) {
            std::cerr << "  step failed: " << step.status().message << "\n";
            return out;
        }
        for (int32_t t : step->committed_tokens) {
            if (vb->is_end_of_generation(t)) {
                out.ended = true;
                break;
            }
            gen.push_back(t);
        }
    }
    out.text = vb->detokenize(gen).value();
    if (cb) {
        out.stats = cb->stats();
    }
    out.ok = true;
    return out;
}

bool test_scripted_broken_endings() {
    auto b = vocab_backend();
    if (!b || !b->json_schema_to_grammar(kPersonSchema).ok()) {
        std::cout << "  (skipped: no model file or no schema converter)\n";
        return true;
    }
    const std::string json = "{\"name\": \"Ada Lovelace\", \"age\": 36, \"tags\": [\"math\", \"poetry\"]}";
    struct Case {
        const char* what;
        std::string model_text;
        bool thinking;
        std::string expect;
    };
    const std::vector<Case> cases = {
        {"text after the closing brace", json + "\n\nI hope this helps! Let me know if", false, json},
        {"a second object after the first", json + "\n" + json, false, json},
        {"markdown fence after the object", json + "\n```\n", false, json},
        {"reasoning, then the object, then chatter", "The user wants JSON.\n</think>\n\n" + json + " Done.", true,
         "The user wants JSON.\n</think>\n\n" + json},
        {"already clean", json, false, json},
    };
    for (const Case& cs : cases) {
        // Without the constraint the model says all of it
        Scripted free_run = run_scripted(cs.model_text, nullptr, cs.thinking, 4, true);
        SIXI_ASSERT(free_run.ok && free_run.ended);
        SIXI_ASSERT(free_run.text == cs.model_text);
        for (bool fast : {true, false}) {
            for (uint32_t k : {0u, 1u, 4u, 7u}) {
                Scripted run = run_scripted(cs.model_text, kPersonSchema, cs.thinking, k, fast);
                if (!run.ok || !run.ended || run.text != cs.expect) {
                    std::cerr << "  case \"" << cs.what << "\" k=" << k << " fast=" << fast << ": got [" << run.text << "]\n";
                }
                SIXI_ASSERT(run.ok);
                SIXI_ASSERT(run.ended);
                SIXI_ASSERT(run.text == cs.expect);
            }
        }
    }
    // A model that never closes the object cannot end the answer: it runs to the token limit instead of stopping
    // with half a document (the server reports finish_reason "length" for it).
    {
        Scripted run = run_scripted("{\"name\": \"Ada Lovelace\", \"age\": 36, \"tags\": [\"math\"", kPersonSchema, false, 4, true);
        SIXI_ASSERT(run.ok);
        SIXI_ASSERT(!run.ended);
    }
    return true;
}


// ---- The shapes and failures reported by the agent that asked for this feature ------------------------------

const char* kUtteranceSchema =
    R"({"type":"object","properties":{"utterance":{"type":"string"}},"required":["utterance"],"additionalProperties":false})";

// Index of the first token of `text` the constraint refuses, or -1 when the whole text is accepted.
int first_refused(LlamaCppBackend& b, ITokenConstraint& c, const std::string& text) {
    const auto ids = b.tokenize(text, false).value();
    for (size_t i = 0; i < ids.size(); ++i) {
        if (!c.accept(ids[i]).ok()) {
            return static_cast<int>(i);
        }
    }
    return -1;
}

bool test_agent_shapes() {
    auto b = vocab_backend();
    if (!b || !b->json_schema_to_grammar(kUtteranceSchema).ok()) {
        std::cout << "  (skipped: no model file or no schema converter)\n";
        return true;
    }
    const int32_t eos = b->eos_token_id();
    auto fresh = [&](const std::string& schema) { return b->make_grammar_constraint(b->json_schema_to_grammar(schema).value(), "root").value(); };

    // Free text in five languages inside a string: typographic quotes, escaped quote and newline, any Unicode
    for (const char* doc : {
             R"({"utterance": "Zażółć gęślą jaźń — powiedział „dobrze” i wyszedł."})",
             R"({"utterance":"He said \"hello\"\nand left; 100% sure, {really} [yes] <ok> (fine)."})",
             R"({"utterance": "Größe: „Straße“ – schön, daß es klappt!"})",
             R"({"utterance": "« Ça va ? » — l’œuvre d’été, n’est-ce pas…"})",
             R"({"utterance": "Perché no? È così: «certo», più o meno."})",
             R"({"utterance": "日本語 and 🙂 and é"})",
             R"({ "utterance" : "" })"}) {
        auto c = fresh(kUtteranceSchema);
        const int at = first_refused(*b, *c, doc);
        if (at >= 0) {
            std::cerr << "  refused at token " << at << " of: " << doc << "\n";
        }
        SIXI_ASSERT(at < 0);
        SIXI_ASSERT(c->allows(eos));
        SIXI_ASSERT(!c->allows(b->tokenize("}", false).value().front()));
    }
    // Not JSON or not the schema: refused somewhere before the end
    for (const std::string& doc : std::vector<std::string>{
             std::string("{\"utterance\": \"line one\nline two\"}"),          // a raw newline inside a string
             std::string(R"({"utterance": "a", "extra": 1})"),
             std::string(R"({"utterance": 5})"),
             std::string(R"({"text": "a"})"),
             std::string(R"({})"),
             std::string(R"(["a"])"),
             std::string("{\"utterance\": \"a\")"),                          // the reported ending: ")
             std::string(R"({"utterance": "a">)")}) {                          // the reported ending: ">
        auto c = fresh(kUtteranceSchema);
        const int at = first_refused(*b, *c, doc);
        if (at < 0) {
            std::cerr << "  accepted, but must not be: " << doc << "\n";
        }
        SIXI_ASSERT(at >= 0);
    }

    // json_object: any object, and only an object
    {
        const char* any_object = R"({"type":"object"})";
        auto c = fresh(any_object);
        SIXI_ASSERT(first_refused(*b, *c, R"({"a": 1, "b": [true, null, {"c": "ż „x”"}], "d": -2.5e3})") < 0);
        SIXI_ASSERT(c->allows(eos));
        for (const char* doc : {R"([1, 2])", R"("text")", R"(42)", R"({"a": 1} {"b": 2})", "{\"a\": 1})", "Sure: {\"a\": 1}"}) {
            auto c2 = fresh(any_object);
            SIXI_ASSERT(first_refused(*b, *c2, doc) >= 0);
        }
    }

    // A string enum of 60 values: a listed value passes, anything else does not
    {
        std::string schema = R"({"type":"object","properties":{"intent":{"type":"string","enum":[)";
        for (int i = 0; i < 60; ++i) {
            schema += (i ? "," : "") + std::string("\"intent_") + (i < 10 ? "0" : "") + std::to_string(i) + "\"";
        }
        schema += R"(]}},"required":["intent"],"additionalProperties":false})";
        for (int i : {0, 9, 37, 59}) {
            auto c = fresh(schema);
            const std::string doc = std::string("{\"intent\": \"intent_") + (i < 10 ? "0" : "") + std::to_string(i) + "\"}";
            SIXI_ASSERT(first_refused(*b, *c, doc) < 0);
            SIXI_ASSERT(c->allows(eos));
        }
        for (const char* doc : {R"({"intent": "intent_60"})", R"({"intent": "other"})", R"({"intent": "intent_5"})", R"({"intent": "Intent_05"})"}) {
            auto c = fresh(schema);
            SIXI_ASSERT(first_refused(*b, *c, doc) >= 0);
        }
    }

    // Arrays of integers and arrays of objects
    {
        const char* schema =
            R"({"type":"object","properties":{"ids":{"type":"array","items":{"type":"integer"}},)"
            R"("items":{"type":"array","items":{"type":"object","properties":{"text":{"type":"string"},"lang":{"type":"string","enum":["pl","en","de","fr","it"]}},)"
            R"("required":["text","lang"],"additionalProperties":false}}},"required":["ids","items"],"additionalProperties":false})";
        auto c = fresh(schema);
        SIXI_ASSERT(first_refused(*b, *c, R"({"ids": [1, -2, 30], "items": [{"text": "cześć", "lang": "pl"}, {"text": "ciao", "lang": "it"}]})") < 0);
        SIXI_ASSERT(c->allows(eos));
        for (const char* doc : {R"({"ids": [1.5], "items": []})", R"({"ids": ["1"], "items": []})",
                                R"({"ids": [], "items": [{"text": "a", "lang": "es"}]})", R"({"ids": [], "items": [{"text": "a"}]})"}) {
            auto c2 = fresh(schema);
            SIXI_ASSERT(first_refused(*b, *c2, doc) >= 0);
        }
    }
    return true;
}

bool test_agent_reported_endings() {
    auto b = vocab_backend();
    if (!b || !b->json_schema_to_grammar(kUtteranceSchema).ok()) {
        std::cout << "  (skipped: no model file or no schema converter)\n";
        return true;
    }
    // The reported endings are single tokens in this vocabulary (") and "> where "} belongs). The model's favourite
    // is the broken one, its runner-up the right one: the answer is the right one, for the schema and for
    // json_object alike, with and without drafts. (The texts are chosen so that both tokenize alike up to the end.)
    const std::vector<std::pair<std::string, std::string>> cases = {
        {"{\"utterance\": \"Dzień dobry, „świecie”!\")", R"({"utterance": "Dzień dobry, „świecie”!"})"},
        {R"({"utterance": "Guten Tag, wie geht es dir">)", R"({"utterance": "Guten Tag, wie geht es dir"})"},
        {"{\"utterance\": \"Bonjour\")", R"({"utterance": "Bonjour"})"},
    };
    for (const auto& [wrong, right] : cases) {
        Scripted free_run = run_scripted(wrong, nullptr, false, 4, true, right);
        SIXI_ASSERT(free_run.ok && free_run.text == wrong);   // unconstrained: the broken ending
        for (const char* schema : {kUtteranceSchema, R"({"type":"object"})"}) {
            for (bool fast : {true, false}) {
                for (uint32_t k : {0u, 4u}) {
                    Scripted run = run_scripted(wrong, schema, false, k, fast, right);
                    if (!run.ok || !run.ended || run.text != right) {
                        std::cerr << "  k=" << k << " fast=" << fast << ": got [" << run.text << "] for [" << wrong << "]\n";
                    }
                    SIXI_ASSERT(run.ok && run.ended);
                    SIXI_ASSERT(run.text == right);
                    SIXI_ASSERT(run.stats.rows_changed >= 1);   // the client can see the model was overruled
                }
            }
        }
    }
    // A clean answer is left alone: nothing overruled
    {
        const std::string clean = R"({"utterance": "Buongiorno, come va?"})";
        Scripted run = run_scripted(clean, kUtteranceSchema, false, 4, true);
        SIXI_ASSERT(run.ok && run.ended && run.text == clean);
        SIXI_ASSERT(run.stats.rows_changed == 0);
    }
    return true;
}

}  // namespace

int main() {
    int passed = 0;
    int failed = 0;
    RUN_TEST(test_constrain_row);
    RUN_TEST(test_decorator_greedy_ar_matches_reference);
    RUN_TEST(test_decorator_speculative_equals_ar);
    RUN_TEST(test_decorator_start_token);
    RUN_TEST(test_decorator_sampled_tokens_allowed);
    RUN_TEST(test_decorator_long_rollback);
    RUN_TEST(test_grammar_engine);
    RUN_TEST(test_scripted_broken_endings);
    RUN_TEST(test_agent_shapes);
    RUN_TEST(test_agent_reported_endings);
    std::cout << passed << " passed, " << failed << " failed\n";
    return failed == 0 ? 0 : 1;
}

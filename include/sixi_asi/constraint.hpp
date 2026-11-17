#pragma once

#include "sixi_asi/common.hpp"
#include "sixi_asi/core_execution.hpp"
#include <cstdint>
#include <memory>
#include <string>
#include <vector>

// Constrained decoding: the answer must follow a grammar (JSON schema). The constraint lives on the host, one
// state per request, and is applied to the target's logits, so speculative decoding stays exact: a drafted token
// is accepted only when it is what the constrained target would have produced.

namespace sixi_asi::core {

// IBackend decorator that applies a constraint to every row of target logits and follows the backend through
// rollbacks. The speculative decoder and the server use it in place of the backend for one request.
//
// The inner backend must return real logits (device-side greedy sampling off): the decorator reports no
// device-side sampling, so callers take the logits path.
class ConstrainedBackend : public IBackend {
public:
    struct Options {
        int32_t start_token = -1;  // the constraint begins after this token (e.g. </think>); -1: from the first token
        uint32_t keep = 1;         // 1 for greedy, top_k when sampling without penalties, 0 = every allowed token
    };
    struct Stats {
        uint64_t rows = 0;           // logit rows constrained, speculative ones included
        uint64_t rows_changed = 0;   // positions of the kept sequence where the model's best token was refused
                                     // (rows of rejected drafts do not count: 0 = the model wrote it unaided)
        uint64_t tokens_refused = 0; // fed tokens the constraint refused (rejected drafts)
    };

    // `constraint`: the state before the first constrained token (cloned; the caller keeps its object)
    ConstrainedBackend(std::shared_ptr<IBackend> inner, const ITokenConstraint& constraint, Options opts);

    // Apply the constraint to logits the inner backend produced before the decorator existed (the last prompt row).
    Status constrain_current(std::vector<float>& logits);
    bool active() const noexcept { return active_; }
    // May the constrained text end here? (false before the constraint starts or while a refused token is held)
    bool can_end() { return live() && cur_->allows(inner_->eos_token_id()); }
    Stats stats() const;

    std::string backend_name() const override { return inner_->backend_name(); }
    uint64_t vocab_size() const override { return inner_->vocab_size(); }
    int32_t eos_token_id() const override { return inner_->eos_token_id(); }
    Result<std::vector<int32_t>> tokenize(const std::string& text, bool add_bos = true) override { return inner_->tokenize(text, add_bos); }
    Result<std::string> detokenize(const std::vector<int32_t>& tokens) override { return inner_->detokenize(tokens); }

    Status reset_state() override;
    Result<BackendSequenceState> checkpoint_state(const DeterministicRng& rng) override { return inner_->checkpoint_state(rng); }
    Status restore_state(const BackendSequenceState& checkpoint, DeterministicRng& rng) override;

    Result<std::vector<float>> forward_tokens(const std::vector<int32_t>& input_tokens, uint32_t prefill_chunk_size = 512) override;
    Result<std::vector<std::vector<float>>> forward_sequence_all_logits(const std::vector<int32_t>& input_tokens) override;

    bool is_end_of_generation(int32_t token_id) const noexcept override { return inner_->is_end_of_generation(token_id); }
    std::vector<int32_t> context_tokens() const override { return inner_->context_tokens(); }
    std::string model_chat_template() const override { return inner_->model_chat_template(); }
    void set_gpu_greedy_sampling_enabled(bool) noexcept override {}

    uint32_t device_snapshot_slots() const noexcept override { return inner_->device_snapshot_slots(); }
    Status device_snapshot_save(uint32_t slot) override { return inner_->device_snapshot_save(slot); }
    Status device_snapshot_restore(uint32_t slot) override;
    std::vector<int32_t> device_snapshot_tokens(uint32_t slot) const override { return inner_->device_snapshot_tokens(slot); }

    uint32_t stream_count() const noexcept override { return inner_->stream_count(); }
    void bind_stream(uint32_t stream) noexcept override { inner_->bind_stream(stream); }
    std::vector<int32_t> stream_context_tokens(uint32_t stream) const override { return inner_->stream_context_tokens(stream); }
    void begin_setup() noexcept override { inner_->begin_setup(); }
    void end_setup() noexcept override { inner_->end_setup(); }
    void begin_generation() noexcept override { inner_->begin_generation(); }
    void end_generation() noexcept override { inner_->end_generation(); }
    uint32_t generating_streams() const noexcept override { return inner_->generating_streams(); }
    uint32_t shared_draft_k() const noexcept override { return inner_->shared_draft_k(); }

    bool supports_fast_prefix_rollback() const noexcept override { return inner_->supports_fast_prefix_rollback(); }
    uint32_t fast_rollback_window() const noexcept override { return inner_->fast_rollback_window(); }
    uint64_t current_seq_len() const noexcept override { return inner_->current_seq_len(); }
    Status rollback_to_position(uint64_t target_seq_len) override;
    bool has_embedded_mtp_drafter() const noexcept override { return inner_->has_embedded_mtp_drafter(); }
    uint32_t preferred_draft_k() const noexcept override { return inner_->preferred_draft_k(); }
    uint32_t max_useful_draft_k() const noexcept override { return inner_->max_useful_draft_k(); }
    std::pair<uint32_t, uint32_t> slow_draft_k_band() const noexcept override { return inner_->slow_draft_k_band(); }
    Result<std::vector<int32_t>> draft_embedded_mtp(int32_t first_token, uint32_t k) override { return inner_->draft_embedded_mtp(first_token, k); }

    // Device-side sampling is not offered (see the class comment); the IBackend defaults of forward_tokens_greedy
    // and forward_sequence_greedy_argmax take the constrained logits.

private:
    // The constraint follows the tokens the backend holds after base_len_. A fed token the constraint refuses
    // (a rejected draft) stops it until a rollback removes that token.
    void feed(int32_t token);
    void truncate(uint64_t n_fed);
    bool live() const noexcept { return active_ && dead_at_ == kNone; }
    Status check_not_stuck() const;
    Status constrain_row(float* logits, uint64_t vocab_size, int32_t* best_token = nullptr);

    static constexpr uint64_t kNone = ~0ULL;
    static constexpr uint64_t kCheckpointEvery = 32;

    std::shared_ptr<IBackend> inner_;
    Options opts_;
    std::unique_ptr<ITokenConstraint> origin_;  // state before any token
    std::unique_ptr<ITokenConstraint> cur_;     // state after the accepted tokens
    std::vector<std::unique_ptr<ITokenConstraint>> checkpoints_;  // [i]: state after (i + 1) * kCheckpointEvery accepted tokens
    std::vector<int32_t> fed_;                  // tokens the backend holds after base_len_
    std::vector<uint8_t> changed_;              // [i]: the row after fed_[i] had its best token refused
    bool first_changed_ = false;                // the same for the row before any fed token (constrain_current)
    // Tokens a rollback removed, with their flags: a backend without fast rollback restores a checkpoint and feeds
    // the kept tokens again in one call, which yields only the last row; the flags of the others come from here.
    uint64_t stash_base_ = 0;
    std::vector<int32_t> stash_fed_;
    std::vector<uint8_t> stash_changed_;
    uint64_t base_len_ = 0;
    uint64_t active_from_ = kNone;              // index in fed_ of the first constrained token
    uint64_t accepted_ = 0;                     // tokens cur_ has accepted
    uint64_t dead_at_ = kNone;                  // index in fed_ of the refused token
    bool active_ = false;
    Stats stats_;
};

}  // namespace sixi_asi::core

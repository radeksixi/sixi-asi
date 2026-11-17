#include "sixi_asi/constraint.hpp"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <initializer_list>
#include <utility>

namespace sixi_asi::core {

namespace {

struct Ranked {
    int32_t id;
    float logit;
};

// Higher logit first; equal logits: lower id first (the tie-break of DeterministicSampler)
inline bool better(const Ranked& a, const Ranked& b) noexcept {
    return a.logit != b.logit ? a.logit > b.logit : a.id < b.id;
}

// The `width` best tokens of the row, best first.
std::vector<Ranked> top_tokens(const float* logits, uint64_t vocab_size, size_t width) {
    std::vector<Ranked> heap;
    width = std::min<size_t>(width, vocab_size);
    heap.reserve(width + 1);
    auto worst_on_top = [](const Ranked& a, const Ranked& b) { return better(a, b); };
    for (uint64_t i = 0; i < vocab_size; ++i) {
        const Ranked c{static_cast<int32_t>(i), logits[i]};
        if (heap.size() < width) {
            heap.push_back(c);
            std::push_heap(heap.begin(), heap.end(), worst_on_top);
        } else if (better(c, heap.front())) {
            std::pop_heap(heap.begin(), heap.end(), worst_on_top);
            heap.back() = c;
            std::push_heap(heap.begin(), heap.end(), worst_on_top);
        }
    }
    std::sort_heap(heap.begin(), heap.end(), worst_on_top);
    return heap;
}

}  // namespace

Status ITokenConstraint::constrain(float* logits, uint64_t vocab_size, uint32_t keep, bool* best_refused, int32_t* best_token) {
    if (best_refused != nullptr) {
        *best_refused = false;
    }
    if (best_token != nullptr) {
        *best_token = -1;
    }
    if (logits == nullptr || vocab_size == 0) {
        return Status::Error(ErrorCode::InvalidArgument, "constrain: empty logits row");
    }
    auto none_allowed = [] {
        return Status::Error(ErrorCode::BackendError, "constrained decoding: the constraint allows no token here");
    };
    if (keep == 0) {
        const int32_t best = DeterministicSampler::sample_greedy_argmax(logits, vocab_size);
        if (best_refused != nullptr) {
            *best_refused = !allows(best);
        }
        if (Status st = mask_all(logits, vocab_size); !st.ok()) {
            return st;
        }
        const int32_t left = DeterministicSampler::sample_greedy_argmax(logits, vocab_size);
        return logits[left] > kMaskedLogit ? Status::Ok() : none_allowed();
    }
    // Common case: the best token is allowed and nothing changes
    if (keep == 1) {
        const int32_t best = DeterministicSampler::sample_greedy_argmax(logits, vocab_size);
        if (allows(best)) {
            if (best_token != nullptr) {
                *best_token = best;
            }
            return Status::Ok();
        }
    }
    uint32_t found = 0;
    bool first = true;
    for (const size_t width : {static_cast<size_t>(std::max<uint32_t>(64u, 4u * keep)), static_cast<size_t>(kSearchWidth)}) {
        // The refused tokens met so far are masked, so the allowed ones found so far lead this list
        const std::vector<Ranked> top = top_tokens(logits, vocab_size, width);
        for (size_t i = found; i < top.size() && found < keep; ++i) {
            if (top[i].logit <= kMaskedLogit) {
                break;
            }
            if (allows(top[i].id)) {
                if (found == 0 && best_token != nullptr) {
                    *best_token = top[i].id;
                }
                ++found;
            } else {
                logits[top[i].id] = kMaskedLogit;
                if (first && best_refused != nullptr) {
                    *best_refused = true;
                }
            }
            first = false;
        }
        if (found >= keep) {
            return Status::Ok();
        }
        if (top.size() >= vocab_size) {
            // the whole vocabulary was tried: every refused token is masked
            return found > 0 ? Status::Ok() : none_allowed();
        }
    }
    if (found == 0) {
        if (Status st = mask_all(logits, vocab_size); !st.ok()) {
            return st;
        }
        const int32_t left = DeterministicSampler::sample_greedy_argmax(logits, vocab_size);
        if (best_token != nullptr) {
            *best_token = left;
        }
        return logits[left] > kMaskedLogit ? Status::Ok() : none_allowed();
    }
    // Fewer than `keep` allowed tokens among the best kSearchWidth: they lead the row now (every token tried and
    // refused is masked, every token not tried ranks below them). Keep them, mask the rest of the vocabulary.
    const std::vector<Ranked> kept = top_tokens(logits, vocab_size, found);
    std::fill(logits, logits + vocab_size, kMaskedLogit);
    for (const Ranked& r : kept) {
        logits[r.id] = r.logit;
    }
    return Status::Ok();
}

ConstrainedBackend::ConstrainedBackend(std::shared_ptr<IBackend> inner, const ITokenConstraint& constraint, Options opts)
    : inner_(std::move(inner)), opts_(opts), origin_(constraint.clone()) {
    cur_ = origin_->clone();
    base_len_ = inner_->current_seq_len();
    if (opts_.start_token < 0) {
        active_ = true;
        active_from_ = 0;
    }
}

ConstrainedBackend::Stats ConstrainedBackend::stats() const {
    // A row counts once the token it decided is in the sequence: the row after the last token is still pending
    // (when the document is complete the model may ask for a newline there, which is cut, not replaced).
    Stats out = stats_;
    out.rows_changed = (first_changed_ && !changed_.empty()) ? 1u : 0u;
    for (size_t i = 0; i + 1 < changed_.size(); ++i) {
        out.rows_changed += changed_[i];
    }
    return out;
}

void ConstrainedBackend::feed(int32_t token) {
    const uint64_t idx = fed_.size();
    fed_.push_back(token);
    changed_.push_back(0);
    if (idx >= stash_base_ && idx - stash_base_ < stash_fed_.size()) {
        if (stash_fed_[idx - stash_base_] == token) {
            changed_.back() = stash_changed_[idx - stash_base_];   // same prefix, same row
        } else {
            stash_fed_.resize(idx - stash_base_);
            stash_changed_.resize(idx - stash_base_);
        }
    }
    if (dead_at_ != kNone) {
        return;
    }
    if (!active_) {
        if (token == opts_.start_token) {
            active_ = true;
            active_from_ = idx + 1;
        }
        return;
    }
    if (cur_->allows(token) && cur_->accept(token).ok()) {
        ++accepted_;
        if (accepted_ % kCheckpointEvery == 0) {
            checkpoints_.push_back(cur_->clone());
        }
    } else {
        dead_at_ = idx;
        ++stats_.tokens_refused;
    }
}

void ConstrainedBackend::truncate(uint64_t n_fed) {
    if (n_fed >= fed_.size()) {
        return;
    }
    stash_base_ = n_fed;
    stash_fed_.assign(fed_.begin() + static_cast<std::ptrdiff_t>(n_fed), fed_.end());
    stash_changed_.assign(changed_.begin() + static_cast<std::ptrdiff_t>(n_fed), changed_.end());
    fed_.resize(n_fed);
    changed_.resize(n_fed);
    if (dead_at_ != kNone && dead_at_ >= n_fed) {
        dead_at_ = kNone;
    }
    if (!active_) {
        return;
    }
    if (opts_.start_token >= 0 && active_from_ > n_fed) {
        // the start token itself was removed
        active_ = false;
        active_from_ = kNone;
        accepted_ = 0;
        checkpoints_.clear();
        cur_ = origin_->clone();
        return;
    }
    if (dead_at_ != kNone) {
        return;  // still stopped at a refused token that is before n_fed
    }
    const uint64_t target = n_fed - active_from_;
    if (target >= accepted_) {
        return;
    }
    const uint64_t n_ckpt = target / kCheckpointEvery;
    checkpoints_.resize(n_ckpt);
    cur_ = (n_ckpt == 0 ? origin_ : checkpoints_.back())->clone();
    accepted_ = n_ckpt * kCheckpointEvery;
    for (uint64_t i = active_from_ + accepted_; i < n_fed; ++i) {
        (void)cur_->accept(fed_[i]);  // accepted before, so it is accepted again
        ++accepted_;
    }
}

Status ConstrainedBackend::check_not_stuck() const {
    if (dead_at_ != kNone) {
        return Status::Error(ErrorCode::BackendError,
                             "constrained decoding: a token the constraint refuses was kept in the sequence");
    }
    return Status::Ok();
}

Status ConstrainedBackend::constrain_row(float* logits, uint64_t vocab_size, int32_t* best_token) {
    if (!live()) {
        return Status::Ok();
    }
    ++stats_.rows;
    bool best_refused = false;
    static const bool debug = std::getenv("SIXI_CONSTRAINT_DEBUG") != nullptr;   // which choices were overruled, on stderr
    const int32_t wanted = debug ? DeterministicSampler::sample_greedy_argmax(logits, vocab_size) : -1;
    Status st = cur_->constrain(logits, vocab_size, opts_.keep, &best_refused, best_token);
    if (debug && best_refused) {
        const int32_t got = DeterministicSampler::sample_greedy_argmax(logits, vocab_size);
        auto piece = [&](int32_t t) { auto r = inner_->detokenize({t}); return r.ok() ? r.value() : std::string("?"); };
        std::fprintf(stderr, "constraint: after %zu tokens the model wanted %d [%s], allowed best is %d [%s]\n",
                     fed_.size(), wanted, piece(wanted).c_str(), got, piece(got).c_str());
    }
    // the row after the last fed token; a rollback that removes that token forgets it
    if (changed_.empty()) {
        first_changed_ = best_refused;
    } else {
        changed_.back() = best_refused ? 1 : 0;
    }
    return st;
}

Status ConstrainedBackend::constrain_current(std::vector<float>& logits) {
    if (logits.empty()) {
        return Status::Ok();
    }
    return constrain_row(logits.data(), logits.size());
}

Result<std::vector<float>> ConstrainedBackend::forward_tokens(const std::vector<int32_t>& input_tokens, uint32_t prefill_chunk_size) {
    if (Status st = check_not_stuck(); !st.ok()) {
        return Result<std::vector<float>>(st);
    }
    auto res = inner_->forward_tokens(input_tokens, prefill_chunk_size);
    if (!res.ok()) {
        return res;
    }
    for (const int32_t t : input_tokens) {
        feed(t);
    }
    if (Status st = constrain_row(res.value().data(), res.value().size()); !st.ok()) {
        return Result<std::vector<float>>(st);
    }
    return res;
}

Result<std::vector<std::vector<float>>> ConstrainedBackend::forward_sequence_all_logits(const std::vector<int32_t>& input_tokens) {
    using Rows = std::vector<std::vector<float>>;
    if (Status st = check_not_stuck(); !st.ok()) {
        return Result<Rows>(st);
    }
    auto res = inner_->forward_sequence_all_logits(input_tokens);
    if (!res.ok()) {
        return res;
    }
    Rows& rows = res.value();
    if (rows.size() != input_tokens.size()) {
        return Result<Rows>(ErrorCode::BackendError, "constrained decoding: backend returned a wrong number of logit rows");
    }
    // Greedy verification reads the rows in order and stops at the first one whose best token is not the next
    // input token (that draft is rejected); rows after a refused token are never used either. Those rows are left
    // as they are: constraining them would cost a pass over the vocabulary each, for nothing.
    bool unused = false;
    for (size_t i = 0; i < input_tokens.size(); ++i) {
        feed(input_tokens[i]);
        if (unused) {
            continue;
        }
        int32_t best = -1;
        if (Status st = constrain_row(rows[i].data(), rows[i].size(), &best); !st.ok()) {
            return Result<Rows>(st);
        }
        if (opts_.keep == 1 && live() && best >= 0 && i + 1 < input_tokens.size() && best != input_tokens[i + 1]) {
            unused = true;
        }
    }
    return res;
}

Status ConstrainedBackend::rollback_to_position(uint64_t target_seq_len) {
    Status st = inner_->rollback_to_position(target_seq_len);
    if (!st.ok()) {
        return st;
    }
    if (target_seq_len < base_len_) {
        base_len_ = target_seq_len;
        truncate(0);
    } else {
        truncate(target_seq_len - base_len_);
    }
    return st;
}

Status ConstrainedBackend::restore_state(const BackendSequenceState& checkpoint, DeterministicRng& rng) {
    Status st = inner_->restore_state(checkpoint, rng);
    if (!st.ok()) {
        return st;
    }
    const uint64_t len = inner_->current_seq_len();
    if (len < base_len_) {
        base_len_ = len;
        truncate(0);
    } else {
        truncate(len - base_len_);
    }
    return st;
}

Status ConstrainedBackend::reset_state() {
    Status st = inner_->reset_state();
    if (st.ok()) {
        base_len_ = inner_->current_seq_len();
        truncate(0);
    }
    return st;
}

Status ConstrainedBackend::device_snapshot_restore(uint32_t slot) {
    Status st = inner_->device_snapshot_restore(slot);
    if (st.ok()) {
        base_len_ = inner_->current_seq_len();
        truncate(0);
    }
    return st;
}

}  // namespace sixi_asi::core

#pragma once

#include "chefsi_layout.h"
#include "laplacian_opt.h"
#include "memory_pool.h"
#include "stencil.h"
#include <array>
#include <cassert>
#include <chrono>
#include <cmath>
#include <cstdio>
#include <cstdlib>
#include <cstring>
#include <limits>

namespace triple_head {
constexpr size_t candidates = 32;
constexpr size_t repeats = 10;
constexpr size_t stride_bytes = 64;
constexpr size_t slack_bytes = 2048;
constexpr size_t slack_elems = slack_bytes / sizeof(double);
constexpr size_t guard_elems = stride_bytes / sizeof(double);
constexpr double guard_value = 918273.625;
static_assert(candidates * stride_bytes == slack_bytes);

inline bool verbose_enabled() {
    const char* value = std::getenv("XLSDFT_TRIPLE_LOTTERY_VERBOSE");
    return value != nullptr && std::strcmp(value, "1") == 0;
}

struct Panels {
    std::array<double*, 3> raw{{nullptr, nullptr, nullptr}};
    std::array<double*, 3> head{{nullptr, nullptr, nullptr}};
    std::array<size_t, 3> offset{{0, 0, 0}};
    size_t elems = 0;
    bool enabled = false;
    bool usage_reported = false;

    void allocate(Memory_pool<double, Fast_memory>& pool, size_t n) {
        assert(!enabled && n > 0);
        elems = n;
        for (size_t b = 0; b < 3; ++b) {
            raw[b] = pool.allocate(elems + slack_elems);
            head[b] = raw[b];
            assert(reinterpret_cast<uintptr_t>(raw[b]) % stride_bytes == 0);
        }
        enabled = true;
        assert_distinct();
    }
    int index(const double* ptr) const {
        for (int b = 0; b < 3; ++b) if (ptr == head[b]) return b;
        return -1;
    }
    double* unused(const double* a, const double* b) const {
        assert(a != b && index(a) >= 0 && index(b) >= 0);
        for (double* p : head) if (p != a && p != b) return p;
        assert(false);
        return nullptr;
    }
    double* alternate(const double* a) const {
        assert(index(a) >= 0);
        for (double* p : head) if (p != a) return p;
        assert(false);
        return nullptr;
    }
    void set_offsets(const std::array<size_t, 3>& bytes) {
        for (size_t b = 0; b < 3; ++b) {
            assert(bytes[b] < slack_bytes && bytes[b] % stride_bytes == 0);
            offset[b] = bytes[b];
            head[b] = raw[b] + bytes[b] / sizeof(double);
        }
        assert_distinct();
    }
    void assert_distinct() const {
        for (size_t a = 0; a < 3; ++a) {
            assert(head[a] >= raw[a]);
            assert(head[a] + elems <= raw[a] + elems + slack_elems - guard_elems);
            for (size_t b = a + 1; b < 3; ++b) {
                const uintptr_t x = reinterpret_cast<uintptr_t>(head[a]);
                const uintptr_t y = reinterpret_cast<uintptr_t>(head[b]);
                assert(x + elems * sizeof(double) <= y || y + elems * sizeof(double) <= x);
            }
        }
    }
    void put_guards() const {
        for (auto p : raw)
            for (size_t i = 0; i < guard_elems; ++i)
                p[elems + slack_elems - guard_elems + i] = guard_value;
    }
    void check_guards() const {
        for (auto p : raw)
            for (size_t i = 0; i < guard_elems; ++i)
                assert(p[elems + slack_elems - guard_elems + i] == guard_value);
    }
    void seed() const {
        // Bounded nonzero synthetic values: no physical SCF values are needed.
        const size_t n = elems + slack_elems - guard_elems;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i) {
            const double value = 0.0001 * (int((i * 17 + 13) % 251) - 125);
            for (auto p : raw) p[i] = value;
        }
        put_guards();
    }
    void clear() const {
        const size_t n = elems + slack_elems - guard_elems;
        #pragma omp parallel for schedule(static)
        for (size_t i = 0; i < n; ++i)
            for (auto p : raw) p[i] = 0.0;
        put_guards();
    }
};

inline double measure(Panels& panels, const Vertices_3D& v, size_t nb,
                      const double* vloc, const Stencil<double>& stencil,
                      size_t phase) {
    panels.seed();
    const auto kernel = [&](size_t step) {
        const size_t in = (phase + step) % 3;
        const size_t out = (in + 1) % 3;
        laplacian_4d<16>(v.ni, v.nj, v.nk, nb, panels.head[out], panels.head[in],
                        vloc, stencil.get_D2_coeffs_x(), stencil.get_D2_coeffs_y(),
                        stencil.get_D2_coeffs_z(), stencil.get_D2_coef0());
    };
    kernel(2); // One untimed warm-up, identical for all candidates.
    const auto begin = std::chrono::steady_clock::now();
    for (size_t r = 0; r < repeats; ++r) kernel(r);
    const auto end = std::chrono::steady_clock::now();
    panels.check_guards();
    for (auto p : panels.head) assert(std::isfinite(p[0]));
    return std::chrono::duration<double, std::milli>(end - begin).count() / repeats;
}

inline void tune(Panels& panels, const Vertices_3D& vertices, size_t nb,
                 const Stencil<double>& original, Memory_pool<double, Fast_memory>& pool,
                 int rank, bool do_tune, size_t shape_count) {
    const auto start = std::chrono::steady_clock::now();
    const size_t mark = pool.mark();
    double* vloc = pool.allocate(vertices.get_size());
    #pragma omp parallel for schedule(static)
    for (size_t i = 0; i < vertices.get_size(); ++i) vloc[i] = 0.05;
    Stencil<double> stencil;
    stencil.deepcopy_mp(original, pool);
    stencil.coeffs_scale_self(-0.5);
    const bool verbose = verbose_enabled();
    if (verbose) {
        std::fprintf(stderr, "TRIPLE_LOTTERY_START rank=%d shape=%zux%zux%zu nb=%zu shape_count=%zu "
                     "payload_bytes=%zu slack_each=2048 timed_calls=10 warmup_calls=1 mode=%s\n",
                     rank, size_t(vertices.ni), size_t(vertices.nj), size_t(vertices.nk), nb, shape_count,
                     panels.elems * sizeof(double), do_tune ? "tune" : "fixed");
        std::fflush(stderr);
    }
    if (do_tune) {
        // One coordinate sweep; not a claim of exhaustive 32^3 optimization.
        for (size_t buffer = 0; buffer < 3; ++buffer) {
            auto trial = panels.offset;
            size_t best_offset = 0;
            double best = std::numeric_limits<double>::max();
            double worst = 0.0;
            for (size_t candidate = 0; candidate < candidates; ++candidate) {
                trial[buffer] = candidate * stride_bytes;
                panels.set_offsets(trial);
                const double ms = measure(panels, vertices, nb, vloc, stencil, buffer);
                assert(std::isfinite(ms) && ms > 0.0);
                if (ms < best) { best = ms; best_offset = trial[buffer]; }
                if (ms > worst) worst = ms;
                if (verbose) {
                    std::fprintf(stderr, "TRIPLE_LOTTERY_CAND rank=%d buffer=%zu offset=%zu avg_ms=%.6f\n",
                                 rank, buffer, trial[buffer], ms);
                }
            }
            trial[buffer] = best_offset;
            panels.set_offsets(trial);
            if (verbose) {
                std::fprintf(stderr, "TRIPLE_LOTTERY_PICK rank=%d buffer=%zu offset=%zu best_ms=%.6f worst_ms=%.6f\n",
                             rank, buffer, best_offset, best, worst);
                std::fflush(stderr);
            }
        }
        const auto chosen = panels.offset;
        const std::array<size_t, 3> zero{{0, 0, 0}};
        double base_sum = 0.0, chosen_sum = 0.0;
        // A-B-B-A recheck of the complete head tuple; each observation has 10 calls.
        for (int sample = 0; sample < 4; ++sample) {
            const bool candidate = sample == 1 || sample == 2;
            panels.set_offsets(candidate ? chosen : zero);
            const double ms = measure(panels, vertices, nb, vloc, stencil, 0);
            (candidate ? chosen_sum : base_sum) += ms;
        }
        const bool accepted = chosen_sum < base_sum;
        panels.set_offsets(accepted ? chosen : zero);
        if (verbose) {
            std::fprintf(stderr, "TRIPLE_LOTTERY_VERIFY rank=%d zero_ms=%.6f chosen_ms=%.6f accepted=%d\n",
                         rank, base_sum / 2.0, chosen_sum / 2.0, int(accepted));
        }
    }
    stencil.destructor_mp();
    pool.release(mark);
    // Synthetic data must never leak into the SCF state.
    panels.clear();
    panels.check_guards();
    const double elapsed = std::chrono::duration<double>(std::chrono::steady_clock::now() - start).count();
    if (verbose) {
        std::fprintf(stderr, "TRIPLE_LOTTERY_DONE rank=%d offsets=%zu,%zu,%zu elapsed_s=%.6f guard=PASS pool_floor=%zu\n",
                     rank, panels.offset[0], panels.offset[1], panels.offset[2], elapsed, mark);
    }
    if (rank == 0) {
        std::fprintf(stdout, "TRIPLE_LOTTERY_TIME rank=0 mode=%s elapsed_s=%.6f\n",
                     do_tune ? "tune" : "fixed", elapsed);
        std::fflush(stdout);
    }
    if (verbose) std::fflush(stderr);
}
} // namespace triple_head

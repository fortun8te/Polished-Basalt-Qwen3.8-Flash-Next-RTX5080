// tests/kernels/cpu/kq_avx512_test.cpp - parity test for kq512_q5_1_rows against kq256_rows across nt=1..8
// and both even (Flash-Next n_ff=640) and odd (n=96) block counts.
#include "basalt/kernels/cpu/kq_avx512.hpp"
#include "basalt/kernels/cpu/kq_avx2.hpp"
#include "basalt/kernels/cpu/expert_layout.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <cstring>
#include <random>
#include <vector>

namespace cpu = basalt::kernels::cpu;

namespace {

uint16_t f32_to_f16_bits(float f) {
    uint32_t x;
    std::memcpy(&x, &f, 4);
    const uint32_t sign = (x >> 16) & 0x8000u;
    const int32_t exp = ((int32_t) ((x >> 23) & 0xFFu)) - 127 + 15;
    const uint32_t mant = x & 0x007FFFFFu;
    if (exp <= 0) return (uint16_t) sign;
    if (exp >= 31) return (uint16_t) (sign | 0x7C00u);
    return (uint16_t) (sign | ((uint32_t) exp << 10) | (mant >> 13));
}

bool run_case(int n, int rows, int nt, uint32_t seed) {
    const int nb = n / QK5_1;
    const size_t row_bytes = (size_t) nb * sizeof(block_q5_1);
    std::mt19937 rng(seed);
    std::uniform_int_distribution<int> byte_dist(0, 255);
    std::uniform_int_distribution<int> i8_dist(-127, 127);
    std::uniform_real_distribution<float> scale_dist(-0.25f, 0.25f);

    std::vector<uint8_t> weights((size_t) rows * row_bytes);
    for (int r = 0; r < rows; ++r) {
        auto* b = (block_q5_1*) (weights.data() + (size_t) r * row_bytes);
        for (int ib = 0; ib < nb; ++ib) {
            const uint16_t d = f32_to_f16_bits(scale_dist(rng));
            const uint16_t m = f32_to_f16_bits(scale_dist(rng));
            std::memcpy((uint8_t*) &b[ib], &d, 2);
            std::memcpy((uint8_t*) &b[ib] + 2, &m, 2);
            for (int k = 0; k < 4; ++k) b[ib].qh[k] = (uint8_t) byte_dist(rng);
            for (int k = 0; k < 16; ++k) b[ib].qs[k] = (uint8_t) byte_dist(rng);
        }
    }

    std::vector<std::vector<block_q8_1>> acts((size_t) nt, std::vector<block_q8_1>((size_t) nb));
    std::vector<const void*> act_ptrs((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        for (int ib = 0; ib < nb; ++ib) {
            const uint16_t d = f32_to_f16_bits(scale_dist(rng));
            const uint16_t s = f32_to_f16_bits(scale_dist(rng));
            std::memcpy((uint8_t*) &acts[(size_t) t][(size_t) ib], &d, 2);
            std::memcpy((uint8_t*) &acts[(size_t) t][(size_t) ib] + 2, &s, 2);
            for (int k = 0; k < 32; ++k) acts[(size_t) t][(size_t) ib].qs[k] = (int8_t) i8_dist(rng);
        }
        act_ptrs[(size_t) t] = acts[(size_t) t].data();
    }

    std::vector<std::vector<float>> out256((size_t) nt, std::vector<float>((size_t) rows, 0.0f));
    std::vector<std::vector<float>> out512((size_t) nt, std::vector<float>((size_t) rows, 0.0f));
    std::vector<float*> p256((size_t) nt), p512((size_t) nt);
    for (int t = 0; t < nt; ++t) {
        p256[(size_t) t] = out256[(size_t) t].data();
        p512[(size_t) t] = out512[(size_t) t].data();
    }

    cpu::kq256_rows(7, weights.data(), row_bytes, n, act_ptrs.data(), nt, p256.data(), 0, rows);
    cpu::kq512_q5_1_rows(weights.data(), row_bytes, n, act_ptrs.data(), nt, p512.data(), 0, rows);

    double max_abs = 0.0, max_rel = 0.0;
    for (int t = 0; t < nt; ++t) {
        for (int r = 0; r < rows; ++r) {
            const double a = out256[(size_t) t][(size_t) r];
            const double b = out512[(size_t) t][(size_t) r];
            const double diff = std::fabs(a - b);
            const double rel = diff / (std::fabs(a) + 1e-6);
            if (diff > max_abs) max_abs = diff;
            if (rel > max_rel) max_rel = rel;
        }
    }
    if (max_rel > 1e-5) {
        std::fprintf(stderr, "FAIL n=%d rows=%d nt=%d max_abs=%.3e max_rel=%.3e\n", n, rows, nt, max_abs, max_rel);
        return false;
    }
    return true;
}

}  // namespace

int main() {
    if (!cpu::cpu_avx512_ok()) {
        std::printf("kq_avx512_test: skipped (CPU has no AVX-512 VNNI)\n");
        return 0;
    }
    const int nts[] = {1, 2, 3, 5, 8};
    const int ns[] = {640, 96, 32};
    uint32_t seed = 5080;
    for (int n : ns) {
        for (int nt : nts) {
            if (!run_case(n, 32, nt, seed++)) return 1;
        }
    }
    std::printf("kq_avx512_test: ok (n=640/96/32, nt=1/2/3/5/8)\n");
    return 0;
}

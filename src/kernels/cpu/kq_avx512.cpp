// src/kernels/cpu/kq_avx512.cpp - AVX-512 VNNI multi-token Q5_1 x Q8_1 dot product.
//
// In Qwen3.8-Flash-Next UD-IQ3_XXS packs, Q5_1 down rows are ~47% of every RAM-resident expert's bytes.
// Previously Q5_1 had no AVX-512 or VNNI kernel (falling back to AVX2 kq256_rows for nt >= 3 and ggml AVX2 vec_dot
// for nt <= 2).  With AVX-512 BW/VL/VNNI (gated by cpu_avx512_ok()):
//   - Two 32-weight Q5_1 blocks (64 weights) unpack in parallel into one ZMM register, using _mm512_maskz_set1_epi8
//     directly on the 64 packed high bits (qh0 | qh1 << 32) instead of byte-shuffle bit extraction.
//   - _mm512_dpbusd_epi32 computes all 16 int32 quad-sums in a single instruction with zero int16 saturation risk
//     (qx in [0..31], qy in [-128..127], max quad sum = 15,872).
//   - Handles all nt >= 1 so single-token and 2-token expert misses also take the 512-bit VNNI path.
#include "basalt/kernels/cpu/kq_avx512.hpp"

#define GGML_COMMON_DECL_CPP
#define GGML_COMMON_IMPL_CPP
#include "basalt/ggml/ggml_common.h"

#include <immintrin.h>

#include <cstring>

namespace basalt::kernels::cpu {
namespace {

constexpr int MAXT = 8;   // kVerifyMaxT

inline float h2f(ggml_half h) {
    uint16_t u;
    std::memcpy(&u, &h, 2);
    return _mm_cvtss_f32(_mm_cvtph_ps(_mm_cvtsi32_si128((int) u)));
}

inline float half_at(const void* block, int i) {
    ggml_half h;
    std::memcpy(&h, (const uint8_t*) block + 2 * i, 2);
    return h2f(h);
}

inline float hsum_float_8(const __m256 x) {
    __m128 res = _mm256_extractf128_ps(x, 1);
    res = _mm_add_ps(res, _mm256_castps256_ps128(x));
    res = _mm_add_ps(res, _mm_movehl_ps(res, res));
    res = _mm_add_ss(res, _mm_movehdup_ps(res));
    return _mm_cvtss_f32(res);
}

void q5_1_dot_avx512(const block_q5_1* x, int nb, const void* const* act, int nt, float* res) {
    const __m512i m4 = _mm512_set1_epi8(0x0F);
    const __m512i zero512 = _mm512_setzero_si512();
    __m512 acc[MAXT];
    float summs[MAXT];
    for (int t = 0; t < nt; ++t) {
        acc[t] = _mm512_setzero_ps();
        summs[t] = 0.0f;
    }

    int ib = 0;
    for (; ib + 1 < nb; ib += 2) {
        const float dx0 = half_at(&x[ib], 0);
        const float xm0 = half_at(&x[ib], 1);
        const float dx1 = half_at(&x[ib + 1], 0);
        const float xm1 = half_at(&x[ib + 1], 1);

        uint32_t qh0, qh1;
        std::memcpy(&qh0, x[ib].qh, sizeof(uint32_t));
        std::memcpy(&qh1, x[ib + 1].qh, sizeof(uint32_t));
        const __mmask64 hi_mask = (__mmask64) qh0 | ((__mmask64) qh1 << 32);

        const __m128i t0 = _mm_loadu_si128((const __m128i*) x[ib].qs);
        const __m128i t1 = _mm_loadu_si128((const __m128i*) x[ib + 1].qs);
        const __m256i n0 = _mm256_inserti128_si256(_mm256_castsi128_si256(t0), _mm_srli_epi16(t0, 4), 1);
        const __m256i n1 = _mm256_inserti128_si256(_mm256_castsi128_si256(t1), _mm_srli_epi16(t1, 4), 1);
        const __m512i qx = _mm512_or_si512(
            _mm512_and_si512(_mm512_inserti64x4(_mm512_castsi256_si512(n0), n1, 1), m4),
            _mm512_maskz_set1_epi8(hi_mask, 0x10));

        for (int t = 0; t < nt; ++t) {
            const block_q8_1* y0 = (const block_q8_1*) act[t] + ib;
            const block_q8_1* y1 = y0 + 1;
            summs[t] += xm0 * half_at(y0, 1) + xm1 * half_at(y1, 1);

            const __m256i qy0 = _mm256_loadu_si256((const __m256i*) y0->qs);
            const __m256i qy1 = _mm256_loadu_si256((const __m256i*) y1->qs);
            const __m512i qy = _mm512_inserti64x4(_mm512_castsi256_si512(qy0), qy1, 1);

            const __m512i dot32 = _mm512_dpbusd_epi32(zero512, qx, qy);
            const __m512 d_pair = _mm512_mask_blend_ps(
                (__mmask16) 0xFF00u,
                _mm512_set1_ps(dx0 * half_at(y0, 0)),
                _mm512_set1_ps(dx1 * half_at(y1, 0)));
            acc[t] = _mm512_fmadd_ps(_mm512_cvtepi32_ps(dot32), d_pair, acc[t]);
        }
    }

    if (ib < nb) {
        const float dx = half_at(&x[ib], 0);
        const float xm = half_at(&x[ib], 1);
        uint32_t qh0;
        std::memcpy(&qh0, x[ib].qh, sizeof(uint32_t));
        const __m128i t0 = _mm_loadu_si128((const __m128i*) x[ib].qs);
        const __m256i n0 = _mm256_inserti128_si256(_mm256_castsi128_si256(t0), _mm_srli_epi16(t0, 4), 1);
        const __m256i qx = _mm256_or_si256(
            _mm256_and_si256(n0, _mm256_set1_epi8(0x0F)),
            _mm256_maskz_set1_epi8((__mmask32) qh0, 0x10));
        const __m256i zero256 = _mm256_setzero_si256();

        for (int t = 0; t < nt; ++t) {
            const block_q8_1* y = (const block_q8_1*) act[t] + ib;
            summs[t] += xm * half_at(y, 1);
            const __m256i qy = _mm256_loadu_si256((const __m256i*) y->qs);
            const __m256i dot32 = _mm256_dpbusd_epi32(zero256, qx, qy);
            const __m256 prod = _mm256_mul_ps(_mm256_cvtepi32_ps(dot32), _mm256_set1_ps(dx * half_at(y, 0)));
            const __m512 prod512 = _mm512_maskz_mov_ps((__mmask16) 0x00FFu, _mm512_castps256_ps512(prod));
            acc[t] = _mm512_add_ps(acc[t], prod512);
        }
    }

    for (int t = 0; t < nt; ++t) {
        const __m256 hi256 = _mm256_castpd_ps(_mm512_extractf64x4_pd(_mm512_castps_pd(acc[t]), 1));
        const __m256 sum256 = _mm256_add_ps(_mm512_castps512_ps256(acc[t]), hi256);
        res[t] = hsum_float_8(sum256) + summs[t];
    }
}

}  // namespace

void kq512_q5_1_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                     int r0, int r1) {
    const int nb = n / QK5_1;
    for (int t0 = 0; t0 < nt; t0 += MAXT) {
        const int m = nt - t0 < MAXT ? nt - t0 : MAXT;
        float s[MAXT];
        for (int r = r0; r < r1; ++r) {
            q5_1_dot_avx512((const block_q5_1*) (w + (size_t) r * row_bytes), nb, act + t0, m, s);
            for (int t = 0; t < m; ++t) out[t0 + t][r] = s[t];
        }
    }
}

}  // namespace basalt::kernels::cpu

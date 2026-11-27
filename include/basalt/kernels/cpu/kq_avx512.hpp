// include/basalt/kernels/cpu/kq_avx512.hpp - AVX-512 VNNI multi-token dot products for Q5_1 down rows against Q8_1
// activations.  Processes two 32-weight blocks (64 weights) per iteration with _mm512_maskz_set1_epi8 and
// _mm512_dpbusd_epi32 (exact int32 dot against ggml-cpu's AVX2 vec_dot_q5_1_q8_1) for all group sizes nt >= 1.
#pragma once

#include <cstddef>
#include <cstdint>

namespace basalt::kernels::cpu {

/// Q5_1 down rows: out[t][r] = w_r . act[t], rows [r0, r1), for any nt >= 1.  Requires cpu_avx512_ok() (AVX-512
/// F/BW/VL/VNNI).
void kq512_q5_1_rows(const uint8_t* w, size_t row_bytes, int n, const void* const* act, int nt, float* const* out,
                     int r0, int r1);

}  // namespace basalt::kernels::cpu

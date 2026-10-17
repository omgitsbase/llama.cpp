// Quantizing weights on the GPU: the search ggml-quants.c runs per block, one thread to a block. The C code is
// moved across in the same arithmetic order, so the output is byte-identical; that is also why this file opts out
// of -use_fast_math (see CMakeLists), whose approximate reciprocal can move a scale by an ulp and flip a nibble.
//
// Unsupported types return 0 and the caller keeps its CPU path. The k-quants are carried only with an imatrix.
// q8_0 is left to the CPU, which quantizes it faster than the bus can carry it here.

#include "common.cuh"
#include "convert.cuh"
#include "quantize-weights.cuh"

#include <mutex>

// From ggml-quants.h, declared here rather than included: that header pulls
// ggml-common.h in its C flavour, which leaves common.cuh without the CUDA
// constants it needs.
extern "C" bool iq2xs_tables_impl(enum ggml_type type, const uint64_t ** grid, const int ** map,
                                  const uint16_t ** neighbours, int * grid_size, int * map_size,
                                  int * neighbours_size);
extern "C" void iq3xs_init_impl(int grid_size);
extern "C" bool iq3xs_tables_impl(int grid_size, const uint32_t ** grid, const int ** map,
                                  const uint16_t ** neighbours, int * grid_out, int * map_size,
                                  int * neighbours_size);

#define GROUP_MAX_EPS_D 1e-15f
#define GROUP_MAX_EPS_IQ3_XXS_D 1e-8f
#define GROUP_MAX_EPS_IQ2_S_D 1e-8f

// ------------------------------------------------------------ scalar helpers

static __device__ __forceinline__ int nearest_int_d(float fval) {
    const float val = fval + 12582912.f;
    return (__float_as_int(val) & 0x007fffff) - 0x00400000;
}

static __device__ __forceinline__ int best_index_int8_d(int n, const int8_t * val, float x) {
    if (x <= val[0]) {
        return 0;
    }
    if (x >= val[n - 1]) {
        return n - 1;
    }
    int ml = 0, mu = n - 1;
    while (mu - ml > 1) {
        const int mav = (ml + mu) / 2;
        if (x < val[mav]) {
            mu = mav;
        } else {
            ml = mav;
        }
    }
    return x - val[mu - 1] < val[mu] - x ? mu - 1 : mu;
}

static __device__ float make_qx_quants_d(int n, int nmax, const float * x, int8_t * L, int rmse_type,
                                         const float * qw) {
    float mx   = 0;
    float amax = 0;
    for (int i = 0; i < n; ++i) {
        const float ax = fabsf(x[i]);
        if (ax > amax) {
            amax = ax;
            mx   = x[i];
        }
    }
    if (amax < GROUP_MAX_EPS_D) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = -nmax / mx;
    if (rmse_type == 0) {
        for (int i = 0; i < n; ++i) {
            const int l = nearest_int_d(iscale * x[i]);
            L[i] = nmax + max(-nmax, min(nmax - 1, l));
        }
        return 1 / iscale;
    }
    bool return_early = false;
    if (rmse_type < 0) {
        rmse_type    = -rmse_type;
        return_early = true;
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_d(iscale * x[i]);
        l     = max(-nmax, min(nmax - 1, l));
        L[i]  = l + nmax;
        const float w = qw ? qw[i]
                           : rmse_type == 1 ? x[i] * x[i]
                           : rmse_type == 2 ? 1.0f
                           : rmse_type == 3 ? fabsf(x[i])
                                            : sqrtf(fabsf(x[i]));
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    float scale = suml2 ? sumlx / suml2 : 0.0f;
    if (return_early) {
        return suml2 > 0 ? 0.5f * (scale + 1 / iscale) : 1 / iscale;
    }
    float best = scale * sumlx;
    for (int is = -9; is <= 9; ++is) {
        if (is == 0) {
            continue;
        }
        iscale = -(nmax + 0.1f * is) / mx;
        sumlx = suml2 = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_d(iscale * x[i]);
            l     = max(-nmax, min(nmax - 1, l));
            const float w = qw ? qw[i]
                               : rmse_type == 1 ? x[i] * x[i]
                               : rmse_type == 2 ? 1.0f
                               : rmse_type == 3 ? fabsf(x[i])
                                                : sqrtf(fabsf(x[i]));
            sumlx += w * x[i] * l;
            suml2 += w * l * l;
        }
        if (suml2 > 0 && sumlx * sumlx > best * suml2) {
            for (int i = 0; i < n; ++i) {
                const int l = nearest_int_d(iscale * x[i]);
                L[i] = nmax + max(-nmax, min(nmax - 1, l));
            }
            scale = sumlx / suml2;
            best  = scale * sumlx;
        }
    }
    return scale;
}

static __device__ float make_qkx3_quants_d(int n, int nmax, const float * x, const float * weights, uint8_t * L,
                                           float * the_min, uint8_t * Laux, float rmin, float rdelta, int nstep,
                                           bool use_mad) {
    float mn    = x[0];
    float mx    = x[0];
    float sum_w = weights ? weights[0] : x[0] * x[0];
    float sum_x = sum_w * x[0];
    for (int i = 1; i < n; ++i) {
        if (x[i] < mn) {
            mn = x[i];
        }
        if (x[i] > mx) {
            mx = x[i];
        }
        const float w = weights ? weights[i] : x[i] * x[i];
        sum_w += w;
        sum_x += w * x[i];
    }
    if (mn > 0) {
        mn = 0;
    }
    if (mx <= mn) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        *the_min = -mn;
        return 0.f;
    }
    float iscale   = nmax / (mx - mn);
    float scale    = 1 / iscale;
    float best_mad = 0;
    for (int i = 0; i < n; ++i) {
        const int l = nearest_int_d(iscale * (x[i] - mn));
        L[i]        = max(0, min(nmax, l));
        float diff  = scale * L[i] + mn - x[i];
        diff        = use_mad ? fabsf(diff) : diff * diff;
        const float w = weights ? weights[i] : x[i] * x[i];
        best_mad += w * diff;
    }
    if (nstep < 1) {
        *the_min = -mn;
        return scale;
    }
    for (int is = 0; is <= nstep; ++is) {
        iscale = (rmin + rdelta * is + nmax) / (mx - mn);
        float sum_l = 0, sum_l2 = 0, sum_xl = 0;
        for (int i = 0; i < n; ++i) {
            int l   = nearest_int_d(iscale * (x[i] - mn));
            l       = max(0, min(nmax, l));
            Laux[i] = l;
            const float w = weights ? weights[i] : x[i] * x[i];
            sum_l  += w * l;
            sum_l2 += w * l * l;
            sum_xl += w * l * x[i];
        }
        const float D = sum_w * sum_l2 - sum_l * sum_l;
        if (D > 0) {
            float this_scale = (sum_w * sum_xl - sum_x * sum_l) / D;
            float this_min   = (sum_l2 * sum_x - sum_l * sum_xl) / D;
            if (this_min > 0) {
                this_min   = 0;
                this_scale = sum_xl / sum_l2;
            }
            float mad = 0;
            for (int i = 0; i < n; ++i) {
                float diff = this_scale * Laux[i] + this_min - x[i];
                diff       = use_mad ? fabsf(diff) : diff * diff;
                const float w = weights ? weights[i] : x[i] * x[i];
                mad += w * diff;
            }
            if (mad < best_mad) {
                for (int i = 0; i < n; ++i) {
                    L[i] = Laux[i];
                }
                best_mad = mad;
                scale    = this_scale;
                mn       = this_min;
            }
        }
    }
    *the_min = -mn;
    return scale;
}

static __device__ float make_qp_quants_d(int n, int nmax, const float * x, uint8_t * L, const float * quant_weights) {
    float mx = 0;
    for (int i = 0; i < n; ++i) {
        mx = fmaxf(mx, x[i]);
    }
    if (mx < GROUP_MAX_EPS_D) {
        for (int i = 0; i < n; ++i) {
            L[i] = 0;
        }
        return 0.f;
    }
    float iscale = nmax / mx;
    for (int i = 0; i < n; ++i) {
        L[i] = nearest_int_d(iscale * x[i]);
    }
    const float scale = 1 / iscale;
    float best_mse = 0;
    for (int i = 0; i < n; ++i) {
        const float diff = x[i] - scale * L[i];
        best_mse += quant_weights[i] * diff * diff;
    }
    for (int is = -4; is <= 4; ++is) {
        if (is == 0) {
            continue;
        }
        const float iscale_is = (0.1f * is + nmax) / mx;
        const float scale_is  = 1 / iscale_is;
        float       mse       = 0;
        for (int i = 0; i < n; ++i) {
            int l = nearest_int_d(iscale_is * x[i]);
            l     = min(nmax, l);
            const float diff = x[i] - scale_is * l;
            mse += quant_weights[i] * diff * diff;
        }
        if (mse < best_mse) {
            best_mse = mse;
            iscale   = iscale_is;
        }
    }
    float sumlx = 0;
    float suml2 = 0;
    for (int i = 0; i < n; ++i) {
        int l = nearest_int_d(iscale * x[i]);
        l     = min(nmax, l);
        L[i]  = l;
        const float w = quant_weights[i];
        sumlx += w * x[i] * l;
        suml2 += w * l * l;
    }
    for (int itry = 0; itry < 5; ++itry) {
        int n_changed = 0;
        for (int i = 0; i < n; ++i) {
            const float w   = quant_weights[i];
            float       slx = sumlx - w * x[i] * L[i];
            float       sl2 = suml2 - w * L[i] * L[i];
            if (slx > 0 && sl2 > 0) {
                int new_l = nearest_int_d(x[i] * sl2 / slx);
                new_l     = min(nmax, new_l);
                if (new_l != L[i]) {
                    slx += w * x[i] * new_l;
                    sl2 += w * new_l * new_l;
                    if (slx * slx * suml2 > sumlx * sumlx * sl2) {
                        L[i]  = new_l;
                        sumlx = slx;
                        suml2 = sl2;
                        ++n_changed;
                    }
                }
            }
        }
        if (!n_changed) {
            break;
        }
    }
    return suml2 > 0.0f ? sumlx / suml2 : 0.0f;
}

static __device__ __forceinline__ void get_scale_min_k4_d(int j, const uint8_t * q, uint8_t * d, uint8_t * m) {
    if (j < 4) {
        *d = q[j] & 63;
        *m = q[j + 4] & 63;
    } else {
        *d = (q[j + 4] & 0xF) | ((q[j - 4] >> 6) << 4);
        *m = (q[j + 4] >> 4) | ((q[j - 0] >> 6) << 4);
    }
}

// ------------------------------------------------------------------- q4_K

static __global__ void quantize_q4_K_kernel(const float * __restrict__ x, char * __restrict__ dst,
                                            const float * __restrict__ imatrix, int64_t nsb, int64_t sb_per_row,
                                            size_t row_size) {
    const int64_t isb = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (isb >= nsb) {
        return;
    }
    const int64_t row = isb / sb_per_row;
    const int64_t i   = isb % sb_per_row;

    x += isb * QK_K;
    block_q4_K * y = (block_q4_K *) (dst + row * row_size) + i;
    const float * quant_weights = imatrix + i * QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K / 32];
    uint8_t Lm[QK_K / 32];
    float   weights[32];
    float   sw[QK_K / 32];
    float   mins[QK_K / 32];
    float   scales[QK_K / 32];

    float sum_x2 = 0;
    for (int l = 0; l < QK_K; ++l) {
        sum_x2 += x[l] * x[l];
    }
    const float sigma2 = 2 * sum_x2 / QK_K;

    for (int j = 0; j < QK_K / 32; ++j) {
        const float * qw = quant_weights + 32 * j;
        for (int l = 0; l < 32; ++l) {
            weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
        }
        float sumw = 0;
        for (int l = 0; l < 32; ++l) {
            sumw += weights[l];
        }
        sw[j]     = sumw;
        scales[j] = make_qkx3_quants_d(32, 15, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    const float d_block = make_qp_quants_d(QK_K / 32, 63, scales, Ls, sw);
    const float m_block = make_qp_quants_d(QK_K / 32, 63, mins, Lm, sw);
    for (int j = 0; j < QK_K / 32; ++j) {
        const uint8_t ls = Ls[j];
        const uint8_t lm = Lm[j];
        if (j < 4) {
            y->scales[j]     = ls;
            y->scales[j + 4] = lm;
        } else {
            y->scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            y->scales[j - 4] |= ((ls >> 4) << 6);
            y->scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
    y->data.d    = __float2half(d_block);
    y->data.dmin = __float2half(m_block);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
        get_scale_min_k4_d(j, y->scales, &sc, &m);
        const float d = __half2float(y->data.d) * sc;
        if (!d) {
            continue;
        }
        const float dm = __half2float(y->data.dmin) * m;
        for (int ii = 0; ii < 32; ++ii) {
            int l = nearest_int_d((x[32 * j + ii] + dm) / d);
            l     = max(0, min(15, l));
            L[32 * j + ii] = l;
        }
    }
    uint8_t * q = y->qs;
    for (int j = 0; j < QK_K; j += 64) {
        for (int l = 0; l < 32; ++l) {
            q[l] = L[j + l] | (L[j + l + 32] << 4);
        }
        q += 32;
    }
}

// ------------------------------------------------------------------- q5_K

static __global__ void quantize_q5_K_kernel(const float * __restrict__ x, char * __restrict__ dst,
                                            const float * __restrict__ imatrix, int64_t nsb, int64_t sb_per_row,
                                            size_t row_size) {
    const int64_t isb = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (isb >= nsb) {
        return;
    }
    const int64_t row = isb / sb_per_row;
    const int64_t i   = isb % sb_per_row;

    x += isb * QK_K;
    block_q5_K * y = (block_q5_K *) (dst + row * row_size) + i;
    const float * quant_weights = imatrix + i * QK_K;

    uint8_t L[QK_K];
    uint8_t Laux[32];
    uint8_t Ls[QK_K / 32];
    uint8_t Lm[QK_K / 32];
    float   mins[QK_K / 32];
    float   scales[QK_K / 32];
    float   sw[QK_K / 32];
    float   weights[32];

    float sum_x2 = 0;
    for (int l = 0; l < QK_K; ++l) {
        sum_x2 += x[l] * x[l];
    }
    const float sigma2 = 2 * sum_x2 / QK_K;

    for (int j = 0; j < QK_K / 32; ++j) {
        const float * qw = quant_weights + 32 * j;
        for (int l = 0; l < 32; ++l) {
            weights[l] = qw[l] * sqrtf(sigma2 + x[32 * j + l] * x[32 * j + l]);
        }
        float sumw = 0;
        for (int l = 0; l < 32; ++l) {
            sumw += weights[l];
        }
        sw[j]     = sumw;
        scales[j] = make_qkx3_quants_d(32, 31, x + 32 * j, weights, L + 32 * j, &mins[j], Laux, -0.9f, 0.05f, 36, false);
    }

    const float d_block = make_qp_quants_d(QK_K / 32, 63, scales, Ls, sw);
    const float m_block = make_qp_quants_d(QK_K / 32, 63, mins, Lm, sw);
    for (int j = 0; j < QK_K / 32; ++j) {
        uint8_t ls = Ls[j];
        uint8_t lm = Lm[j];
        ls = min((uint8_t) 63, ls);
        lm = min((uint8_t) 63, lm);
        if (j < 4) {
            y->scales[j]     = ls;
            y->scales[j + 4] = lm;
        } else {
            y->scales[j + 4] = (ls & 0xF) | ((lm & 0xF) << 4);
            y->scales[j - 4] |= ((ls >> 4) << 6);
            y->scales[j - 0] |= ((lm >> 4) << 6);
        }
    }
    y->data.d    = __float2half(d_block);
    y->data.dmin = __float2half(m_block);

    uint8_t sc, m;
    for (int j = 0; j < QK_K / 32; ++j) {
        get_scale_min_k4_d(j, y->scales, &sc, &m);
        const float d = __half2float(y->data.d) * sc;
        if (!d) {
            continue;
        }
        const float dm = __half2float(y->data.dmin) * m;
        for (int ii = 0; ii < 32; ++ii) {
            int l = nearest_int_d((x[32 * j + ii] + dm) / d);
            l     = max(0, min(31, l));
            L[32 * j + ii] = l;
        }
    }

    uint8_t * qh = y->qh;
    uint8_t * ql = y->qs;
    for (int j = 0; j < QK_K / 8; ++j) {
        qh[j] = 0;
    }

    uint8_t m1 = 1, m2 = 2;
    for (int n = 0; n < QK_K; n += 64) {
        for (int j = 0; j < 32; ++j) {
            int l1 = L[n + j];
            if (l1 > 15) {
                l1 -= 16;
                qh[j] |= m1;
            }
            int l2 = L[n + j + 32];
            if (l2 > 15) {
                l2 -= 16;
                qh[j] |= m2;
            }
            ql[j] = l1 | (l2 << 4);
        }
        m1 <<= 2;
        m2 <<= 2;
        ql += 32;
    }
}

// ------------------------------------------------------------------- q6_K

static __global__ void quantize_q6_K_kernel(const float * __restrict__ x, char * __restrict__ dst,
                                            const float * __restrict__ imatrix, int64_t nsb, int64_t sb_per_row,
                                            size_t row_size) {
    const int64_t isb = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (isb >= nsb) {
        return;
    }
    const int64_t row = isb / sb_per_row;
    const int64_t i   = isb % sb_per_row;

    x += isb * QK_K;
    block_q6_K * y = (block_q6_K *) (dst + row * row_size) + i;
    const float * quant_weights = imatrix + i * QK_K;

    int8_t L[QK_K];
    float  scales[QK_K / 16];

    float max_scale     = 0;
    float max_abs_scale = 0;

    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const float scale = make_qx_quants_d(16, 32, x + 16 * ib, L + 16 * ib, 1, quant_weights + 16 * ib);
        scales[ib]        = scale;
        const float abs_scale = fabsf(scale);
        if (abs_scale > max_abs_scale) {
            max_abs_scale = abs_scale;
            max_scale     = scale;
        }
    }

    if (max_abs_scale < GROUP_MAX_EPS_D) {
        char * raw = (char *) y;
        for (int b = 0; b < (int) sizeof(block_q6_K); ++b) {
            raw[b] = 0;
        }
        y->d = __float2half(0.f);
        return;
    }

    const float iscale = -128.f / max_scale;
    y->d = __float2half(1 / iscale);
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        y->scales[ib] = min(127, nearest_int_d(iscale * scales[ib]));
    }

    for (int j = 0; j < QK_K / 16; ++j) {
        const float d = __half2float(y->d) * y->scales[j];
        if (!d) {
            continue;
        }
        for (int ii = 0; ii < 16; ++ii) {
            int l = nearest_int_d(x[16 * j + ii] / d);
            l     = max(-32, min(31, l));
            L[16 * j + ii] = l + 32;
        }
    }

    uint8_t * ql = y->ql;
    uint8_t * qh = y->qh;
    for (int j = 0; j < QK_K; j += 128) {
        for (int l = 0; l < 32; ++l) {
            const uint8_t q1 = L[j + l + 0] & 0xF;
            const uint8_t q2 = L[j + l + 32] & 0xF;
            const uint8_t q3 = L[j + l + 64] & 0xF;
            const uint8_t q4 = L[j + l + 96] & 0xF;
            ql[l + 0]  = q1 | (q3 << 4);
            ql[l + 32] = q2 | (q4 << 4);
            qh[l] = (L[j + l] >> 4) | ((L[j + l + 32] >> 4) << 2) | ((L[j + l + 64] >> 4) << 4) |
                    ((L[j + l + 96] >> 4) << 6);
        }
        ql += 64;
        qh += 32;
    }
}

// ------------------------------------------------------------ iq4_nl, iq4_xs

static __device__ void quantize_row_iq4_nl_impl_d(const int super_block_size, const int block_size, const float * x,
                                                  half * dh, uint8_t * q4, uint16_t * scales_h, uint8_t * scales_l,
                                                  float * scales, float * weight, uint8_t * L, const int8_t * values,
                                                  const float * quant_weights, const int ntry) {
    float sigma2 = 0;
    for (int j = 0; j < super_block_size; ++j) {
        sigma2 += x[j] * x[j];
    }
    sigma2 *= 2.f / super_block_size;

    for (int j = 0; j < super_block_size / 2; ++j) {
        q4[j] = 0;
    }
    dh[0] = __float2half(0.f);

    float max_scale = 0, amax_scale = 0;
    for (int ib = 0; ib < super_block_size / block_size; ++ib) {
        const float * xb = x + ib * block_size;
        uint8_t *     Lb = L + ib * block_size;
        if (quant_weights) {
            const float * qw = quant_weights + ib * block_size;
            for (int j = 0; j < block_size; ++j) {
                weight[j] = qw[j] * sqrtf(sigma2 + xb[j] * xb[j]);
            }
        } else {
            for (int j = 0; j < block_size; ++j) {
                weight[j] = xb[j] * xb[j];
            }
        }
        float amax = 0, mx = 0;
        for (int j = 0; j < block_size; ++j) {
            const float ax = fabsf(xb[j]);
            if (ax > amax) {
                amax = ax;
                mx   = xb[j];
            }
        }
        if (amax < GROUP_MAX_EPS_D) {
            scales[ib] = 0;
            continue;
        }
        float d  = ntry > 0 ? -mx / values[0] : mx / values[0];
        float id = 1 / d;
        float sumqx = 0, sumq2 = 0;
        for (int j = 0; j < block_size; ++j) {
            const float al = id * xb[j];
            const int   l  = best_index_int8_d(16, values, al);
            Lb[j] = l;
            const float q = values[l];
            const float w = weight[j];
            sumqx += w * q * xb[j];
            sumq2 += w * q * q;
        }
        d = sumq2 > 0 ? sumqx / sumq2 : 0.f;
        float best = d * sumqx;
        for (int itry = -ntry; itry <= ntry; ++itry) {
            id    = (itry + values[0]) / mx;
            sumqx = sumq2 = 0;
            for (int j = 0; j < block_size; ++j) {
                const float al = id * xb[j];
                const int   l  = best_index_int8_d(16, values, al);
                const float q  = values[l];
                const float w  = weight[j];
                sumqx += w * q * xb[j];
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                d    = sumqx / sumq2;
                best = d * sumqx;
            }
        }
        scales[ib] = d;
        const float abs_d = fabsf(d);
        if (abs_d > amax_scale) {
            amax_scale = abs_d;
            max_scale  = d;
        }
    }

    if (super_block_size / block_size > 1) {
        const int nb = super_block_size / block_size;
        for (int j = 0; j < (nb + 7) / 8; ++j) {
            scales_h[j] = 0;
        }
        const float d = -max_scale / 32;
        dh[0] = __float2half(d);
        const float id = d ? 1 / d : 0.f;
        for (int ib = 0; ib < super_block_size / block_size; ++ib) {
            int l = nearest_int_d(id * scales[ib]);
            l     = max(-32, min(31, l));
            const float dl  = d * l;
            const float idl = dl ? 1 / dl : 0.f;
            uint8_t *     Lb = L + ib * block_size;
            const float * xb = x + ib * block_size;
            for (int j = 0; j < block_size; ++j) {
                Lb[j] = best_index_int8_d(16, values, idl * xb[j]);
            }
            l += 32;
            const uint8_t l_l = l & 0xf;
            const uint8_t l_h = l >> 4;
            if (ib % 2 == 0) {
                scales_l[ib / 2] = l_l;
            } else {
                scales_l[ib / 2] |= (l_l << 4);
            }
            scales_h[ib / 8] |= (l_h << 2 * (ib % 8));
        }
    } else {
        dh[0] = __float2half(scales[0]);
        if (ntry > 0) {
            const float id = scales[0] ? 1 / scales[0] : 0;
            for (int j = 0; j < super_block_size; ++j) {
                L[j] = best_index_int8_d(16, values, id * x[j]);
            }
        }
    }

    for (int i = 0; i < super_block_size / 32; ++i) {
        for (int j = 0; j < 16; ++j) {
            q4[16 * i + j] = L[32 * i + j] | (L[32 * i + 16 + j] << 4);
        }
    }
}

static __global__ void quantize_iq4_nl_kernel(const float * __restrict__ x, char * __restrict__ dst,
                                              const float * __restrict__ imatrix, int64_t nsb, int64_t sb_per_row,
                                              size_t row_size) {
    const int64_t isb = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (isb >= nsb) {
        return;
    }
    const int64_t row = isb / sb_per_row;
    const int64_t ibl = isb % sb_per_row;

    block_iq4_nl * y = (block_iq4_nl *) (dst + row * row_size) + ibl;

    uint8_t  L[QK4_NL];
    float    weight[QK4_NL];
    float    scale;
    uint16_t unused_h;

    quantize_row_iq4_nl_impl_d(QK4_NL, 32, x + isb * QK4_NL, &y->d, y->qs, &unused_h, nullptr, &scale, weight, L,
                               kvalues_iq4nl, imatrix ? imatrix + QK4_NL * ibl : nullptr, 7);
}

static __global__ void quantize_iq4_xs_kernel(const float * __restrict__ x, char * __restrict__ dst,
                                              const float * __restrict__ imatrix, int64_t nsb, int64_t sb_per_row,
                                              size_t row_size) {
    const int64_t isb = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (isb >= nsb) {
        return;
    }
    const int64_t row = isb / sb_per_row;
    const int64_t ibl = isb % sb_per_row;

    block_iq4_xs * y = (block_iq4_xs *) (dst + row * row_size) + ibl;

    uint8_t L[QK_K];
    float   weight[32];
    float   scales[QK_K / 32];

    quantize_row_iq4_nl_impl_d(QK_K, 32, x + isb * QK_K, &y->d, y->qs, &y->scales_h, y->scales_l, scales, weight, L,
                               kvalues_iq4nl, imatrix ? imatrix + QK_K * ibl : nullptr, 7);
}

// ---------------------------------------------------------------- iq2_xxs

static __device__ int iq2_find_best_neighbour_d(const uint16_t * neighbours, const uint64_t * grid,
                                                const float * xval, const float * weight, float scale, int8_t * L) {
    const int num_neighbors = neighbours[0];
    float     best_d2       = FLT_MAX;
    int       grid_index    = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *) (grid + neighbours[j]);
        float          d2 = 0;
        for (int i = 0; i < 8; ++i) {
            const float q    = pg[i];
            const float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2    = d2;
            grid_index = neighbours[j];
        }
    }
    const int8_t * pg = (const int8_t *) (grid + grid_index);
    for (int i = 0; i < 8; ++i) {
        L[i] = (pg[i] - 1) / 2;
    }
    return grid_index;
}

static __global__ void quantize_iq2_xxs_kernel(const float * __restrict__ x, char * __restrict__ dst,
                                               const float * __restrict__ imatrix,
                                               const uint64_t * __restrict__ kgrid,
                                               const int * __restrict__ kmap,
                                               const uint16_t * __restrict__ kneigh, int64_t nsb, int64_t sb_per_row,
                                               size_t row_size) {
    const int64_t isb = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (isb >= nsb) {
        return;
    }
    const int64_t row = isb / sb_per_row;
    const int64_t ibl = isb % sb_per_row;

    const int kMaxQ = 3;

    block_iq2_xxs * y = (block_iq2_xxs *) (dst + row * row_size) + ibl;

    float    scales[QK_K / 32];
    float    weight[32];
    float    xval[32];
    int8_t   L[32];
    int8_t   Laux[32];
    float    waux[32];
    uint8_t  block_signs[4];
    uint32_t q2[2 * (QK_K / 32)];

    y->d = __float2half(0.f);
    for (int j = 0; j < 2 * (QK_K / 32); ++j) {
        q2[j] = 0;
    }

    float         max_scale = 0;
    const float * xbl       = x + isb * QK_K;
    float         sumx2     = 0;
    for (int i = 0; i < QK_K; ++i) {
        sumx2 += xbl[i] * xbl[i];
    }
    const float sigma2 = sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const float * xb = xbl + 32 * ib;
        const float * qw = imatrix + QK_K * ibl + 32 * ib;
        for (int i = 0; i < 32; ++i) {
            weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
        }
        for (int i = 0; i < 32; ++i) {
            waux[i] = sqrtf(weight[i]);
        }
        for (int k = 0; k < 4; ++k) {
            int     nflip = 0;
            uint8_t s     = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8 * k + i] >= 0) {
                    xval[8 * k + i] = xb[8 * k + i];
                } else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    ++nflip;
                    s |= (1 << i);
                }
            }
            if (nflip % 2) {
                int   imin = 0;
                float mn   = weight[8 * k + imin] * xb[8 * k + imin] * xb[8 * k + imin];
                for (int i = 1; i < 8; ++i) {
                    const float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                    if (ax < mn) {
                        mn   = ax;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= (1 << imin);
            }
            block_signs[k] = s & 127;
        }
        float mx = xval[0];
        for (int i = 1; i < 32; ++i) {
            mx = fmaxf(mx, xval[i]);
        }
        if (mx < GROUP_MAX_EPS_D) {
            scales[ib] = 0;
            for (int i = 0; i < 32; ++i) {
                L[i] = 0;
            }
            continue;
        }
        float       scale   = make_qp_quants_d(32, kMaxQ + 1, xval, (uint8_t *) L, weight);
        const float eff_max = scale * kMaxQ;
        if (eff_max <= 0) {
            scales[ib] = 0;
            for (int i = 0; i < 32; ++i) {
                L[i] = 0;
            }
            continue;
        }
        float best = 0;
        for (int is = -6; is <= 6; ++is) {
            const float id         = (2 * kMaxQ - 1 + is * 0.1f) / eff_max;
            const float this_scale = 1 / id;
            for (int k = 0; k < 4; ++k) {
                for (int i = 0; i < 8; ++i) {
                    const int l = nearest_int_d(0.5f * (id * xval[8 * k + i] - 1));
                    Laux[8 * k + i] = max(0, min(kMaxQ - 1, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    u |= (Laux[8 * k + i] << 2 * i);
                }
                int grid_index = kmap[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneigh - kmap[u] - 1;
                    grid_index = iq2_find_best_neighbour_d(neighbours, kgrid, xval + 8 * k, waux + 8 * k, this_scale,
                                                           Laux + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                const float w = weight[i];
                const float q = 2 * Laux[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best  = scale * sumqx;
                for (int i = 0; i < 32; ++i) {
                    L[i] = Laux[i];
                }
            }
        }
        if (scale > 0) {
            const float id = 1 / scale;
            for (int k = 0; k < 4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int_d(0.5f * (id * xval[8 * k + i] - 1));
                    l     = max(0, min(kMaxQ - 1, l));
                    u |= (l << 2 * i);
                }
                int grid_index = kmap[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneigh - kmap[u] - 1;
                    grid_index =
                        iq2_find_best_neighbour_d(neighbours, kgrid, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                }
                const int8_t * pg = (const int8_t *) (kgrid + grid_index);
                for (int i = 0; i < 8; ++i) {
                    L[8 * k + i] = (pg[i] - 1) / 2;
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                const float w = weight[i];
                const float q = 2 * L[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) {
                scale = sumqx / sumq2;
            }
        }
        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 4; ++k) {
                block_signs[k] = (~block_signs[k]) & 127;
            }
        }
        for (int k = 0; k < 4; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) {
                u |= (L[8 * k + i] << 2 * i);
            }
            const int grid_index = kmap[u];
            q2[2 * ib + 0] |= ((uint32_t) grid_index << 8 * k);
            q2[2 * ib + 1] |= (block_signs[k] << 7 * k);
        }
        scales[ib] = scale;
        max_scale  = fmaxf(max_scale, scale);
    }

    if (!max_scale) {
        for (int j = 0; j < QK_K / 4; ++j) {
            ((uint8_t *) y->qs)[j] = 0;
        }
        return;
    }

    const float d = max_scale / 31;
    y->d = __float2half(d);
    const float id = 1 / d;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        int l = nearest_int_d(0.5f * (id * scales[ib] - 1));
        l     = max(0, min(15, l));
        q2[2 * ib + 1] |= ((uint32_t) l << 28);
    }
    memcpy(y->qs, q2, QK_K / 4);
}

// ------------------------------------------------------------------ iq2_s

static __global__ void quantize_iq2_s_kernel(const float * __restrict__ x, char * __restrict__ dst,
                                             const float * __restrict__ imatrix,
                                             const uint64_t * __restrict__ kgrid,
                                             const int * __restrict__ kmap,
                                             const uint16_t * __restrict__ kneigh, int64_t nsb, int64_t sb_per_row,
                                             size_t row_size) {
    const int64_t isb = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (isb >= nsb) {
        return;
    }
    const int64_t row = isb / sb_per_row;
    const int64_t ibl = isb % sb_per_row;

    const int kMaxQ = 3;

    block_iq2_s * y = (block_iq2_s *) (dst + row * row_size) + ibl;

    float   scales[QK_K / 16];
    float   weight[16];
    float   xval[16];
    int8_t  L[16];
    int8_t  Laux[16];
    float   waux[16];
    bool    is_on_grid[2];
    bool    is_on_grid_aux[2];
    uint8_t block_signs[2];

    {
        char * raw = (char *) y;
        for (int b = 0; b < (int) sizeof(block_iq2_s); ++b) {
            raw[b] = 0;
        }
    }
    y->d = __float2half(0.f);

    float         max_scale = 0;
    const float * xbl       = x + isb * QK_K;
    float         sumx2     = 0;
    for (int i = 0; i < QK_K; ++i) {
        sumx2 += xbl[i] * xbl[i];
    }
    const float sigma2 = 2 * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 16; ++ib) {
        const float * xb = xbl + 16 * ib;
        const float * qw = imatrix + QK_K * ibl + 16 * ib;
        for (int i = 0; i < 16; ++i) {
            weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
        }
        for (int i = 0; i < 16; ++i) {
            waux[i] = sqrtf(weight[i]);
        }
        for (int k = 0; k < 2; ++k) {
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8 * k + i] >= 0) {
                    xval[8 * k + i] = xb[8 * k + i];
                } else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    s |= (1 << i);
                }
            }
            block_signs[k] = s;
        }
        float mx = xval[0];
        for (int i = 1; i < 16; ++i) {
            mx = fmaxf(mx, xval[i]);
        }
        for (int i = 0; i < 16; ++i) {
            L[i] = 0;
        }
        if (mx < GROUP_MAX_EPS_IQ2_S_D) {
            scales[ib] = 0;
            continue;
        }
        float best  = 0;
        float scale = mx / (2 * kMaxQ - 1);
        is_on_grid[0] = is_on_grid[1] = true;
        for (int is = -9; is <= 9; ++is) {
            const float id         = (2 * kMaxQ - 1 + is * 0.1f) / mx;
            const float this_scale = 1 / id;
            for (int k = 0; k < 2; ++k) {
                for (int i = 0; i < 8; ++i) {
                    const int l = nearest_int_d(0.5f * (id * xval[8 * k + i] - 1));
                    Laux[8 * k + i] = max(0, min(kMaxQ - 1, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    u |= (Laux[8 * k + i] << 2 * i);
                }
                int grid_index    = kmap[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t * neighbours = kneigh - kmap[u] - 1;
                    grid_index = iq2_find_best_neighbour_d(neighbours, kgrid, xval + 8 * k, waux + 8 * k, this_scale,
                                                           Laux + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                const float w = weight[i];
                const float q = 2 * Laux[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best  = scale * sumqx;
                for (int i = 0; i < 16; ++i) {
                    L[i] = Laux[i];
                }
                for (int k = 0; k < 2; ++k) {
                    is_on_grid[k] = is_on_grid_aux[k];
                }
            }
        }
        int n_not_ongrid = 0;
        for (int k = 0; k < 2; ++k) {
            if (!is_on_grid[k]) {
                ++n_not_ongrid;
            }
        }
        if (n_not_ongrid > 0 && scale > 0) {
            const float id = 1 / scale;
            for (int k = 0; k < 2; ++k) {
                if (is_on_grid[k]) {
                    continue;
                }
                uint16_t u = 0;
                for (int i = 0; i < 8; ++i) {
                    int l = nearest_int_d(0.5f * (id * xval[8 * k + i] - 1));
                    l     = max(0, min(kMaxQ - 1, l));
                    u |= (l << 2 * i);
                    L[8 * k + i] = l;
                }
                int grid_index = kmap[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneigh - kmap[u] - 1;
                    grid_index =
                        iq2_find_best_neighbour_d(neighbours, kgrid, xval + 8 * k, waux + 8 * k, scale, L + 8 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 16; ++i) {
                const float w = weight[i];
                const float q = 2 * L[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) {
                scale = sumqx / sumq2;
            }
        }
        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 2; ++k) {
                block_signs[k] = ~block_signs[k];
            }
        }
        for (int k = 0; k < 2; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 8; ++i) {
                u |= (L[8 * k + i] << 2 * i);
            }
            const int grid_index = kmap[u];
            const int i8         = 2 * ib + k;
            y->qs[i8] = grid_index & 255;
            y->qh[i8 / 4] |= ((grid_index >> 8) << 2 * (i8 % 4));
            y->qs[QK_K / 8 + i8] = block_signs[k];
        }
        scales[ib] = scale;
        max_scale  = fmaxf(max_scale, scale);
    }

    if (!max_scale) {
        return;
    }

    const float d = max_scale / 31;
    y->d = __float2half(d * 0.9875f);
    const float id = 1 / d;
    for (int ib = 0; ib < QK_K / 16; ++ib) {
        int l = nearest_int_d(0.5f * (id * scales[ib] - 1));
        l     = max(0, min(15, l));
        if (ib % 2 == 0) {
            y->scales[ib / 2] = l;
        } else {
            y->scales[ib / 2] |= (l << 4);
        }
    }
}

// ---------------------------------------------------------------- iq3_xxs

static __device__ int iq3_find_best_neighbour_d(const uint16_t * neighbours, const uint32_t * grid,
                                                const float * xval, const float * weight, float scale, int8_t * L) {
    const int num_neighbors = neighbours[0];
    float     best_d2       = FLT_MAX;
    int       grid_index    = -1;
    for (int j = 1; j <= num_neighbors; ++j) {
        const int8_t * pg = (const int8_t *) (grid + neighbours[j]);
        float          d2 = 0;
        for (int i = 0; i < 4; ++i) {
            const float q    = pg[i];
            const float diff = scale * q - xval[i];
            d2 += weight[i] * diff * diff;
        }
        if (d2 < best_d2) {
            best_d2    = d2;
            grid_index = neighbours[j];
        }
    }
    const int8_t * pg = (const int8_t *) (grid + grid_index);
    for (int i = 0; i < 4; ++i) {
        L[i] = (pg[i] - 1) / 2;
    }
    return grid_index;
}

static __global__ void quantize_iq3_xxs_kernel(const float * __restrict__ x, char * __restrict__ dst,
                                               const float * __restrict__ imatrix,
                                               const uint32_t * __restrict__ kgrid,
                                               const int * __restrict__ kmap,
                                               const uint16_t * __restrict__ kneigh, int64_t nsb, int64_t sb_per_row,
                                               size_t row_size) {
    const int64_t isb = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (isb >= nsb) {
        return;
    }
    const int64_t row = isb / sb_per_row;
    const int64_t ibl = isb % sb_per_row;

    const int kMaxQ = 8;

    block_iq3_xxs * y = (block_iq3_xxs *) (dst + row * row_size) + ibl;

    float    scales[QK_K / 32];
    float    weight[32];
    float    xval[32];
    int8_t   L[32];
    int8_t   Laux[32];
    float    waux[32];
    bool     is_on_grid[8];
    bool     is_on_grid_aux[8];
    uint8_t  block_signs[8];
    uint8_t  q3[3 * (QK_K / 8) + QK_K / 32];
    uint32_t * scales_and_signs = (uint32_t *) (q3 + QK_K / 4);

    y->d = __float2half(0.f);
    for (int j = 0; j < 3 * QK_K / 8 + QK_K / 32; ++j) {
        q3[j] = 0;
    }

    float         max_scale = 0;
    const float * xbl       = x + isb * QK_K;
    float         sumx2     = 0;
    for (int i = 0; i < QK_K; ++i) {
        sumx2 += xbl[i] * xbl[i];
    }
    const float sigma2 = 2 * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / 32; ++ib) {
        const float * xb = xbl + 32 * ib;
        const float * qw = imatrix + QK_K * ibl + 32 * ib;
        for (int i = 0; i < 32; ++i) {
            weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
        }
        for (int i = 0; i < 32; ++i) {
            waux[i] = sqrtf(weight[i]);
        }
        for (int k = 0; k < 4; ++k) {
            int     nflip = 0;
            uint8_t s     = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8 * k + i] >= 0) {
                    xval[8 * k + i] = xb[8 * k + i];
                } else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    ++nflip;
                    s |= (1 << i);
                }
            }
            if (nflip % 2) {
                int   imin = 0;
                float mn   = weight[8 * k + imin] * xb[8 * k + imin] * xb[8 * k + imin];
                for (int i = 1; i < 8; ++i) {
                    const float ax = weight[8 * k + i] * xb[8 * k + i] * xb[8 * k + i];
                    if (ax < mn) {
                        mn   = ax;
                        imin = i;
                    }
                }
                xval[8 * k + imin] = -xval[8 * k + imin];
                s ^= (1 << imin);
            }
            block_signs[k] = s & 127;
        }
        float mx = xval[0];
        for (int i = 1; i < 32; ++i) {
            mx = fmaxf(mx, xval[i]);
        }
        for (int i = 0; i < 32; ++i) {
            L[i] = 0;
        }
        if (mx < GROUP_MAX_EPS_IQ3_XXS_D) {
            scales[ib] = 0;
            continue;
        }
        float best  = 0;
        float scale = mx / (2 * kMaxQ - 1);
        for (int k = 0; k < 8; ++k) {
            is_on_grid[k] = true;
        }
        for (int is = -15; is <= 15; ++is) {
            const float id         = (2 * kMaxQ - 1 + is * 0.2f) / mx;
            const float this_scale = 1 / id;
            for (int k = 0; k < 8; ++k) {
                for (int i = 0; i < 4; ++i) {
                    const int l = nearest_int_d(0.5f * (id * xval[4 * k + i] - 1));
                    Laux[4 * k + i] = max(0, min(kMaxQ - 1, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) {
                    u |= (Laux[4 * k + i] << 3 * i);
                }
                int grid_index    = kmap[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t * neighbours = kneigh - kmap[u] - 1;
                    grid_index = iq3_find_best_neighbour_d(neighbours, kgrid, xval + 4 * k, waux + 4 * k, this_scale,
                                                           Laux + 4 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                const float w = weight[i];
                const float q = 2 * Laux[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best  = scale * sumqx;
                for (int i = 0; i < 32; ++i) {
                    L[i] = Laux[i];
                }
                for (int k = 0; k < 8; ++k) {
                    is_on_grid[k] = is_on_grid_aux[k];
                }
            }
        }
        int n_not_ongrid = 0;
        for (int k = 0; k < 8; ++k) {
            if (!is_on_grid[k]) {
                ++n_not_ongrid;
            }
        }
        if (n_not_ongrid > 0 && scale > 0) {
            const float id = 1 / scale;
            for (int k = 0; k < 8; ++k) {
                if (is_on_grid[k]) {
                    continue;
                }
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) {
                    int l = nearest_int_d(0.5f * (id * xval[4 * k + i] - 1));
                    l     = max(0, min(kMaxQ - 1, l));
                    u |= (l << 3 * i);
                }
                int grid_index = kmap[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneigh - kmap[u] - 1;
                    grid_index =
                        iq3_find_best_neighbour_d(neighbours, kgrid, xval + 4 * k, waux + 4 * k, scale, L + 4 * k);
                }
                const int8_t * pg = (const int8_t *) (kgrid + grid_index);
                for (int i = 0; i < 4; ++i) {
                    L[4 * k + i] = (pg[i] - 1) / 2;
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < 32; ++i) {
                const float w = weight[i];
                const float q = 2 * L[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) {
                scale = sumqx / sumq2;
            }
        }
        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < 4; ++k) {
                block_signs[k] = (~block_signs[k]) & 127;
            }
        }
        for (int k = 0; k < 8; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 4; ++i) {
                u |= (L[4 * k + i] << 3 * i);
            }
            q3[8 * ib + k] = kmap[u];
        }
        scales_and_signs[ib] =
            block_signs[0] | (block_signs[1] << 7) | (block_signs[2] << 14) | (block_signs[3] << 21);
        scales[ib] = scale;
        max_scale  = fmaxf(max_scale, scale);
    }

    const int quant_size = (int) (sizeof(block_iq3_xxs) - sizeof(half));
    if (!max_scale) {
        for (int j = 0; j < quant_size; ++j) {
            ((uint8_t *) y->qs)[j] = 0;
        }
        return;
    }

    const float d = max_scale / 31;
    y->d = __float2half(d * 1.0125f);
    const float id = 1 / d;
    for (int ib = 0; ib < QK_K / 32; ++ib) {
        int l = nearest_int_d(0.5f * (id * scales[ib] - 1));
        l     = max(0, min(15, l));
        scales_and_signs[ib] |= ((uint32_t) l << 28);
    }
    memcpy(y->qs, q3, quant_size);
}

// ------------------------------------------------------------------ iq3_s

static __global__ void quantize_iq3_s_kernel(const float * __restrict__ x, char * __restrict__ dst,
                                             const float * __restrict__ imatrix,
                                             const uint32_t * __restrict__ kgrid,
                                             const int * __restrict__ kmap,
                                             const uint16_t * __restrict__ kneigh, int64_t nsb, int64_t sb_per_row,
                                             size_t row_size) {
    const int64_t isb = blockIdx.x * (int64_t) blockDim.x + threadIdx.x;
    if (isb >= nsb) {
        return;
    }
    const int64_t row = isb / sb_per_row;
    const int64_t ibl = isb % sb_per_row;

    const int kMaxQ      = 8;
    const int block_size = 32;
    const int bs4        = block_size / 4;
    const int bs8        = block_size / 8;

    block_iq3_s * y = (block_iq3_s *) (dst + row * row_size) + ibl;

    float   scales[QK_K / 32];
    float   weight[32];
    float   xval[32];
    int8_t  L[32];
    int8_t  Laux[32];
    float   waux[32];
    bool    is_on_grid[8];
    bool    is_on_grid_aux[8];
    uint8_t block_signs[4];

    {
        char * raw = (char *) y;
        for (int b = 0; b < (int) sizeof(block_iq3_s); ++b) {
            raw[b] = 0;
        }
    }
    y->d = __float2half(0.f);

    uint8_t * qs    = y->qs;
    uint8_t * qh    = y->qh;
    uint8_t * signs = y->signs;

    float         max_scale = 0;
    const float * xbl       = x + isb * QK_K;
    float         sumx2     = 0;
    for (int i = 0; i < QK_K; ++i) {
        sumx2 += xbl[i] * xbl[i];
    }
    const float sigma2 = 2 * sumx2 / QK_K;

    for (int ib = 0; ib < QK_K / block_size; ++ib) {
        const float * xb = xbl + block_size * ib;
        const float * qw = imatrix + QK_K * ibl + block_size * ib;
        for (int i = 0; i < block_size; ++i) {
            weight[i] = qw[i] * sqrtf(sigma2 + xb[i] * xb[i]);
        }
        for (int i = 0; i < block_size; ++i) {
            waux[i] = sqrtf(weight[i]);
        }
        for (int k = 0; k < bs8; ++k) {
            uint8_t s = 0;
            for (int i = 0; i < 8; ++i) {
                if (xb[8 * k + i] >= 0) {
                    xval[8 * k + i] = xb[8 * k + i];
                } else {
                    xval[8 * k + i] = -xb[8 * k + i];
                    s |= (1 << i);
                }
            }
            block_signs[k] = s;
        }
        float mx = xval[0];
        for (int i = 1; i < block_size; ++i) {
            mx = fmaxf(mx, xval[i]);
        }
        for (int i = 0; i < block_size; ++i) {
            L[i] = 0;
        }
        if (!mx) {
            scales[ib] = 0;
            continue;
        }
        float best  = 0;
        float scale = mx / (2 * kMaxQ - 1);
        for (int k = 0; k < bs4; ++k) {
            is_on_grid[k] = false;
        }
        for (int is = -9; is <= 9; ++is) {
            const float id         = (2 * kMaxQ - 1 + is * 0.2f) / mx;
            const float this_scale = 1 / id;
            for (int k = 0; k < bs4; ++k) {
                for (int i = 0; i < 4; ++i) {
                    const int l = nearest_int_d(0.5f * (id * xval[4 * k + i] - 1));
                    Laux[4 * k + i] = max(0, min(kMaxQ - 1, l));
                }
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) {
                    u |= (Laux[4 * k + i] << 3 * i);
                }
                int grid_index    = kmap[u];
                is_on_grid_aux[k] = true;
                if (grid_index < 0) {
                    is_on_grid_aux[k] = false;
                    const uint16_t * neighbours = kneigh - kmap[u] - 1;
                    grid_index = iq3_find_best_neighbour_d(neighbours, kgrid, xval + 4 * k, waux + 4 * k, this_scale,
                                                           Laux + 4 * k);
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; ++i) {
                const float w = weight[i];
                const float q = 2 * Laux[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0 && sumqx * sumqx > best * sumq2) {
                scale = sumqx / sumq2;
                best  = scale * sumqx;
                for (int i = 0; i < block_size; ++i) {
                    L[i] = Laux[i];
                }
                for (int k = 0; k < bs4; ++k) {
                    is_on_grid[k] = is_on_grid_aux[k];
                }
            }
        }
        int n_not_ongrid = 0;
        for (int k = 0; k < bs4; ++k) {
            if (!is_on_grid[k]) {
                ++n_not_ongrid;
            }
        }
        if (n_not_ongrid > 0 && scale > 0) {
            const float id = 1 / scale;
            for (int k = 0; k < bs4; ++k) {
                uint16_t u = 0;
                for (int i = 0; i < 4; ++i) {
                    int l = nearest_int_d(0.5f * (id * xval[4 * k + i] - 1));
                    l     = max(0, min(kMaxQ - 1, l));
                    u |= (l << 3 * i);
                }
                int grid_index = kmap[u];
                if (grid_index < 0) {
                    const uint16_t * neighbours = kneigh - kmap[u] - 1;
                    grid_index =
                        iq3_find_best_neighbour_d(neighbours, kgrid, xval + 4 * k, waux + 4 * k, scale, L + 4 * k);
                }
                const int8_t * pg = (const int8_t *) (kgrid + grid_index);
                for (int i = 0; i < 4; ++i) {
                    L[4 * k + i] = (pg[i] - 1) / 2;
                }
            }
            float sumqx = 0, sumq2 = 0;
            for (int i = 0; i < block_size; ++i) {
                const float w = weight[i];
                const float q = 2 * L[i] + 1;
                sumqx += w * xval[i] * q;
                sumq2 += w * q * q;
            }
            if (sumq2 > 0) {
                scale = sumqx / sumq2;
            }
        }
        if (scale < 0) {
            scale = -scale;
            for (int k = 0; k < bs8; ++k) {
                block_signs[k] = ~block_signs[k];
            }
        }
        for (int k = 0; k < bs4; ++k) {
            uint16_t u = 0;
            for (int i = 0; i < 4; ++i) {
                u |= (L[4 * k + i] << 3 * i);
            }
            const int grid_index = kmap[u];
            qs[k] = grid_index & 255;
            qh[(ib * bs4 + k) / 8] |= ((grid_index >> 8) << ((ib * bs4 + k) % 8));
        }
        qs += bs4;
        for (int k = 0; k < bs8; ++k) {
            signs[k] = block_signs[k];
        }
        signs += bs8;
        scales[ib] = scale;
        max_scale  = fmaxf(max_scale, scale);
    }

    if (!max_scale) {
        return;
    }

    const float d = max_scale / 31;
    y->d = __float2half(d * 1.033f);
    const float id = 1 / d;
    for (int ib = 0; ib < QK_K / block_size; ib += 2) {
        int l1 = nearest_int_d(0.5f * (id * scales[ib + 0] - 1));
        l1     = max(0, min(15, l1));
        int l2 = nearest_int_d(0.5f * (id * scales[ib + 1] - 1));
        l2     = max(0, min(15, l2));
        y->scales[ib / 2] = l1 | (l2 << 4);
    }
}

// ------------------------------------------------------------------- host

// One allocation per process, grown as needed: a model is quantized as a few
// hundred calls and a cudaMalloc on every one of them costs more than the work.
// The host side is pinned, which is what lets the copies run at bus speed.
struct quant_scratch {
    void * d_src = nullptr;
    void * d_dst = nullptr;
    void * d_imat = nullptr;
    void * d_raw = nullptr;
    void * h_src = nullptr;
    void * h_dst = nullptr;
    size_t d_src_bytes = 0;
    size_t d_dst_bytes = 0;
    size_t d_imat_bytes = 0;
    size_t d_raw_bytes = 0;
    size_t h_src_bytes = 0;
    size_t h_dst_bytes = 0;
};

// ggml's own lattice tables, copied across once. Rebuilding them here would
// not do: the neighbour lists come out of a qsort, so only a copy gives the
// same answers.
struct lattice_tables {
    uint64_t * grid  = nullptr;
    int *      map   = nullptr;
    uint16_t * neigh = nullptr;
    bool       tried = false;
    bool       ok    = false;
};

// The same, for the D4 lattice the IQ3 types walk.
struct lattice3_tables {
    uint32_t * grid  = nullptr;
    int *      map   = nullptr;
    uint16_t * neigh = nullptr;
    bool       tried = false;
    bool       ok    = false;
};

static std::mutex      g_quant_mutex;
static quant_scratch   g_scratch;
static lattice_tables  g_lattice[GGML_TYPE_COUNT];
static lattice3_tables g_lattice3[2];

static bool fit_device(void ** p, size_t * have, size_t want) {
    if (*have >= want) {
        return true;
    }
    if (*p != nullptr) {
        cudaFree(*p);
    }
    if (cudaMalloc(p, want) != cudaSuccess) {
        *p    = nullptr;
        *have = 0;
        return false;
    }
    *have = want;
    return true;
}

static bool fit_host(void ** p, size_t * have, size_t want) {
    if (*have >= want) {
        return true;
    }
    if (*p != nullptr) {
        cudaFreeHost(*p);
    }
    if (cudaMallocHost(p, want) != cudaSuccess) {
        *p    = nullptr;
        *have = 0;
        return false;
    }
    *have = want;
    return true;
}

// Caller holds g_quant_mutex.
static const lattice_tables * lattice_for(ggml_type type) {
    lattice_tables & t = g_lattice[type];
    if (t.tried) {
        return t.ok ? &t : nullptr;
    }
    t.tried = true;

    ggml_quantize_init(type);
    const uint64_t * grid  = nullptr;
    const int *      map   = nullptr;
    const uint16_t * neigh = nullptr;
    int              ng = 0, nm = 0, nn = 0;
    if (!iq2xs_tables_impl(type, &grid, &map, &neigh, &ng, &nm, &nn)) {
        return nullptr;
    }
    const bool ok =
        cudaMalloc((void **) &t.grid, (size_t) ng * sizeof(uint64_t)) == cudaSuccess &&
        cudaMalloc((void **) &t.map, (size_t) nm * sizeof(int)) == cudaSuccess &&
        cudaMalloc((void **) &t.neigh, (size_t) nn * sizeof(uint16_t)) == cudaSuccess &&
        cudaMemcpy(t.grid, grid, (size_t) ng * sizeof(uint64_t), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(t.map, map, (size_t) nm * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(t.neigh, neigh, (size_t) nn * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) {
        cudaFree(t.grid);
        cudaFree(t.map);
        cudaFree(t.neigh);
        t.grid = nullptr;
        t.map = nullptr;
        t.neigh = nullptr;
        return nullptr;
    }
    t.ok = true;
    return &t;
}

// Block width this type needs here, and whether it is carried at all. The
// k-quants are the imatrix form only; iq4 runs either way.
static bool carried_here(int type, int64_t n_per_row, bool has_imatrix, int * blck_out) {
    int  blck          = 0;
    bool needs_imatrix = true;
    switch (type) {
        case GGML_TYPE_Q4_K:
        case GGML_TYPE_Q5_K:
        case GGML_TYPE_Q6_K:    blck = QK_K;   needs_imatrix = true;  break;
        case GGML_TYPE_IQ4_NL:  blck = QK4_NL; needs_imatrix = false; break;
        case GGML_TYPE_IQ4_XS:  blck = QK_K;   needs_imatrix = false; break;
        case GGML_TYPE_IQ2_XXS: blck = QK_K;   needs_imatrix = true;  break;
        case GGML_TYPE_IQ3_XXS: blck = QK_K;   needs_imatrix = true;  break;
        case GGML_TYPE_IQ2_S:   blck = QK_K;   needs_imatrix = true;  break;
        case GGML_TYPE_IQ3_S:   blck = QK_K;   needs_imatrix = true;  break;
        default: return false;
    }
    if (n_per_row <= 0 || n_per_row % blck != 0 || (needs_imatrix && !has_imatrix)) {
        return false;
    }
    if (blck_out != nullptr) {
        *blck_out = blck;
    }
    return true;
}

bool ggml_cuda_quantize_supported(int type, int64_t n_per_row, bool has_imatrix) {
    return carried_here(type, n_per_row, has_imatrix, nullptr);
}

// Caller holds g_quant_mutex.
static const lattice3_tables * lattice3_for(int grid_size) {
    lattice3_tables & t = g_lattice3[grid_size == 256 ? 0 : 1];
    if (t.tried) {
        return t.ok ? &t : nullptr;
    }
    t.tried = true;

    iq3xs_init_impl(grid_size);
    const uint32_t * grid  = nullptr;
    const int *      map   = nullptr;
    const uint16_t * neigh = nullptr;
    int              ng = 0, nm = 0, nn = 0;
    if (!iq3xs_tables_impl(grid_size, &grid, &map, &neigh, &ng, &nm, &nn)) {
        return nullptr;
    }
    const bool ok =
        cudaMalloc((void **) &t.grid, (size_t) ng * sizeof(uint32_t)) == cudaSuccess &&
        cudaMalloc((void **) &t.map, (size_t) nm * sizeof(int)) == cudaSuccess &&
        cudaMalloc((void **) &t.neigh, (size_t) nn * sizeof(uint16_t)) == cudaSuccess &&
        cudaMemcpy(t.grid, grid, (size_t) ng * sizeof(uint32_t), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(t.map, map, (size_t) nm * sizeof(int), cudaMemcpyHostToDevice) == cudaSuccess &&
        cudaMemcpy(t.neigh, neigh, (size_t) nn * sizeof(uint16_t), cudaMemcpyHostToDevice) == cudaSuccess;
    if (!ok) {
        cudaFree(t.grid);
        cudaFree(t.map);
        cudaFree(t.neigh);
        t.grid  = nullptr;
        t.map   = nullptr;
        t.neigh = nullptr;
        return nullptr;
    }
    t.ok = true;
    return &t;
}

// 64 to a CUDA block: a thread holds a 256-entry L plus its working arrays in
// local memory, and a wider block spills harder.
static bool launch_quantize(int type, const float * dx, char * dy, const float * d_imatrix,
                            const lattice_tables * lat, const lattice3_tables * lat3, int64_t nsb, int64_t spr,
                            size_t row_size) {
    const int threads = 64;
    const int grid    = (int) ((nsb + threads - 1) / threads);
    switch (type) {
        case GGML_TYPE_Q4_K:
            quantize_q4_K_kernel<<<grid, threads>>>(dx, dy, d_imatrix, nsb, spr, row_size);
            break;
        case GGML_TYPE_Q5_K:
            quantize_q5_K_kernel<<<grid, threads>>>(dx, dy, d_imatrix, nsb, spr, row_size);
            break;
        case GGML_TYPE_Q6_K:
            quantize_q6_K_kernel<<<grid, threads>>>(dx, dy, d_imatrix, nsb, spr, row_size);
            break;
        case GGML_TYPE_IQ4_NL:
            quantize_iq4_nl_kernel<<<grid, threads>>>(dx, dy, d_imatrix, nsb, spr, row_size);
            break;
        case GGML_TYPE_IQ4_XS:
            quantize_iq4_xs_kernel<<<grid, threads>>>(dx, dy, d_imatrix, nsb, spr, row_size);
            break;
        case GGML_TYPE_IQ2_XXS:
            quantize_iq2_xxs_kernel<<<grid, threads>>>(dx, dy, d_imatrix, lat->grid, lat->map, lat->neigh, nsb, spr,
                                                       row_size);
            break;
        case GGML_TYPE_IQ2_S:
            quantize_iq2_s_kernel<<<grid, threads>>>(dx, dy, d_imatrix, lat->grid, lat->map, lat->neigh, nsb, spr,
                                                     row_size);
            break;
        case GGML_TYPE_IQ3_XXS:
            quantize_iq3_xxs_kernel<<<grid, threads>>>(dx, dy, d_imatrix, lat3->grid, lat3->map, lat3->neigh, nsb, spr,
                                                       row_size);
            break;
        case GGML_TYPE_IQ3_S:
            quantize_iq3_s_kernel<<<grid, threads>>>(dx, dy, d_imatrix, lat3->grid, lat3->map, lat3->neigh, nsb, spr,
                                                     row_size);
            break;
        default:
            return false;
    }
    return cudaGetLastError() == cudaSuccess;
}

size_t ggml_cuda_requantize_chunk(int src_type, const void * src, int dst_type, void * dst, int64_t nrows,
                                  int64_t n_per_row, const float * imatrix) {
    int blck = 0;
    if (nrows <= 0 || !carried_here(dst_type, n_per_row, imatrix != nullptr, &blck)) {
        return 0;
    }
    const to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda((ggml_type) src_type);
    if (to_fp32 == nullptr) {
        return 0;
    }

    const size_t  src_row  = ggml_row_size((ggml_type) src_type, n_per_row);
    const size_t  row_size = ggml_row_size((ggml_type) dst_type, n_per_row);
    const int64_t n        = nrows * n_per_row;
    const size_t  raw_bytes = src_row * (size_t) nrows;
    const size_t  out_bytes = row_size * (size_t) nrows;

    std::lock_guard<std::mutex> lock(g_quant_mutex);

    const lattice_tables *  lat  = nullptr;
    const lattice3_tables * lat3 = nullptr;
    if (dst_type == GGML_TYPE_IQ2_XXS || dst_type == GGML_TYPE_IQ2_S) {
        lat = lattice_for((ggml_type) dst_type);
        if (lat == nullptr) {
            return 0;
        }
    }
    if (dst_type == GGML_TYPE_IQ3_XXS || dst_type == GGML_TYPE_IQ3_S) {
        lat3 = lattice3_for(dst_type == GGML_TYPE_IQ3_XXS ? 256 : 512);
        if (lat3 == nullptr) {
            return 0;
        }
    }

    if (!fit_device(&g_scratch.d_raw, &g_scratch.d_raw_bytes, raw_bytes) ||
        !fit_device(&g_scratch.d_src, &g_scratch.d_src_bytes, (size_t) n * sizeof(float)) ||
        !fit_device(&g_scratch.d_dst, &g_scratch.d_dst_bytes, out_bytes) ||
        !fit_host(&g_scratch.h_src, &g_scratch.h_src_bytes, raw_bytes) ||
        !fit_host(&g_scratch.h_dst, &g_scratch.h_dst_bytes, out_bytes)) {
        return 0;
    }

    float * d_imatrix = nullptr;
    if (imatrix != nullptr) {
        if (!fit_device(&g_scratch.d_imat, &g_scratch.d_imat_bytes, (size_t) n_per_row * sizeof(float))) {
            return 0;
        }
        d_imatrix = (float *) g_scratch.d_imat;
        if (cudaMemcpy(d_imatrix, imatrix, (size_t) n_per_row * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            return 0;
        }
    }

    memcpy(g_scratch.h_src, src, raw_bytes);
    bool ok = cudaMemcpy(g_scratch.d_raw, g_scratch.h_src, raw_bytes, cudaMemcpyHostToDevice) == cudaSuccess;

    if (ok) {
        to_fp32(g_scratch.d_raw, (float *) g_scratch.d_src, n, nullptr);
        ok = launch_quantize(dst_type, (const float *) g_scratch.d_src, (char *) g_scratch.d_dst, d_imatrix, lat, lat3,
                             n / blck, n_per_row / blck, row_size);
        ok = ok && cudaMemcpy(g_scratch.h_dst, g_scratch.d_dst, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }
    if (!ok) {
        return 0;
    }
    memcpy(dst, g_scratch.h_dst, out_bytes);
    return out_bytes;
}

size_t ggml_cuda_quantize_chunk(int type, const float * src, void * dst, int64_t start, int64_t nrows,
                                int64_t n_per_row, const float * imatrix) {
    int blck = 0;
    if (nrows <= 0 || !carried_here(type, n_per_row, imatrix != nullptr, &blck)) {
        return 0;
    }

    const size_t  row_size = ggml_row_size((ggml_type) type, n_per_row);
    const int64_t n        = nrows * n_per_row;
    const size_t  in_bytes = (size_t) n * sizeof(float);
    const size_t  out_bytes = row_size * (size_t) nrows;

    std::lock_guard<std::mutex> lock(g_quant_mutex);

    const lattice_tables * lat = nullptr;
    if (type == GGML_TYPE_IQ2_XXS || type == GGML_TYPE_IQ2_S) {
        lat = lattice_for((ggml_type) type);
        if (lat == nullptr) {
            return 0;
        }
    }
    const lattice3_tables * lat3 = nullptr;
    if (type == GGML_TYPE_IQ3_XXS || type == GGML_TYPE_IQ3_S) {
        lat3 = lattice3_for(type == GGML_TYPE_IQ3_XXS ? 256 : 512);
        if (lat3 == nullptr) {
            return 0;
        }
    }

    if (!fit_device(&g_scratch.d_src, &g_scratch.d_src_bytes, in_bytes) ||
        !fit_device(&g_scratch.d_dst, &g_scratch.d_dst_bytes, out_bytes) ||
        !fit_host(&g_scratch.h_src, &g_scratch.h_src_bytes, in_bytes) ||
        !fit_host(&g_scratch.h_dst, &g_scratch.h_dst_bytes, out_bytes)) {
        return 0;
    }

    // Kept rather than allocated per call: choosing bit widths asks for many
    // small chunks, and a cudaMalloc on each one costs more than the copy.
    float * d_imatrix = nullptr;
    if (imatrix != nullptr) {
        if (!fit_device(&g_scratch.d_imat, &g_scratch.d_imat_bytes, (size_t) n_per_row * sizeof(float))) {
            return 0;
        }
        d_imatrix = (float *) g_scratch.d_imat;
        if (cudaMemcpy(d_imatrix, imatrix, (size_t) n_per_row * sizeof(float), cudaMemcpyHostToDevice) !=
            cudaSuccess) {
            return 0;
        }
    }

    memcpy(g_scratch.h_src, src + start, in_bytes);
    bool ok = cudaMemcpy(g_scratch.d_src, g_scratch.h_src, in_bytes, cudaMemcpyHostToDevice) == cudaSuccess;

    if (ok) {
        const int64_t nsb = n / blck;
        const int64_t spr = n_per_row / blck;
        ok = launch_quantize(type, (const float *) g_scratch.d_src, (char *) g_scratch.d_dst, d_imatrix, lat, lat3,
                             nsb, spr, row_size);
        ok = ok && cudaMemcpy(g_scratch.h_dst, g_scratch.d_dst, out_bytes, cudaMemcpyDeviceToHost) == cudaSuccess;
    }

    if (!ok) {
        return 0;
    }

    memcpy((char *) dst + (size_t) (start / n_per_row) * row_size, g_scratch.h_dst, out_bytes);
    return out_bytes;
}

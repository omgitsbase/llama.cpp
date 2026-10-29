#include "getrows.cuh"

#include <climits>
#include <mutex>
#include <unordered_map>
#include "dequantize.cuh"
#include "convert.cuh"

template<int qk, int qr, dequantize_kernel_t dequantize_kernel, typename dst_t>
static __global__ void k_get_rows(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        for (int64_t i00 = 2*(blockIdx.y*blockDim.x + threadIdx.x); i00 < ne00; i00 += gridDim.y*blockDim.x) {
            // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
            const int i10 =  blockIdx.x;
            const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
            const int i11 =  dm.x;
            const int i12 =  dm.y;

            const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

            dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
            const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

            const int ib   =  i00/qk;      // block index
            const int iqs  = (i00%qk)/qr;  // quant index
            const int iybs = i00 - i00%qk; // dst block start index
            const int y_offset = qr == 1 ? 1 : qk/2;

            // dequantize
            float2 v;
            dequantize_kernel(src0_row, ib, iqs, v);

            dst_row[iybs + iqs + 0]        = ggml_cuda_cast<dst_t>(v.x);
            dst_row[iybs + iqs + y_offset] = ggml_cuda_cast<dst_t>(v.y);
        }
    }
}

template<typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static __global__ void k_get_rows_kq(
        const void * __restrict__ src0, const int32_t * __restrict__ src1, dst_t * __restrict__ dst,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_sync();
    const int64_t nsb = ne00/QK_K; // super-blocks per row
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm  = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const void * src0_row = (const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03;

        for (int64_t ib = blockIdx.y; ib < nsb; ib += gridDim.y) {
            dequantize_kq(src0_row, ib, dst_row + ib*QK_K, threadIdx.x);
        }
    }
}

template<typename src0_t, typename dst_t>
static __global__ void k_get_rows_float(
        const src0_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00, /*const int64_t ne01, const int64_t ne02, const int64_t ne03,*/
        /*const int64_t ne10,*/ const int64_t ne11, const uint3 ne12_fdv, /*const int64_t ne13,*/
        /*const size_t s0,*/ const size_t s1, const size_t s2, const size_t s3,
        /*const size_t nb00,*/ const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12/*, const size_t s13*/) {

    ggml_cuda_pdl_lc();
    const src0_t  * GGML_CUDA_RESTRICT src0 = src0_ptr;
    const int32_t * GGML_CUDA_RESTRICT src1 = src1_ptr;
    dst_t         * GGML_CUDA_RESTRICT dst  = dst_ptr;
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        // The x and y dimensions of the grid are swapped because the maximum allowed grid size for x is higher.
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1[i10*s10 + i11*s11 + i12*s12];

        dst_t * GGML_CUDA_RESTRICT dst_row = dst + i10*s1 + i11*s2 + i12*s3;
        const src0_t * GGML_CUDA_RESTRICT src0_row = (const src0_t *)((const char *) src0 + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i00 = blockIdx.y*blockDim.x + threadIdx.x; i00 < ne00; i00 += gridDim.y*blockDim.x) {
            dst_row[i00] = ggml_cuda_cast<dst_t>(src0_row[i00]);
        }
    }
}

template<typename dst_t>
static __global__ void k_get_rows_float_vec(
        const dst_t * src0_ptr, const int32_t * src1_ptr, dst_t * dst_ptr,
        const int64_t ne00v,
        const int64_t ne11, const uint3 ne12_fdv,
        const size_t s1, const size_t s2, const size_t s3,
        const size_t nb01, const size_t nb02, const size_t nb03,
        const size_t s10, const size_t s11, const size_t s12) {

    ggml_cuda_pdl_lc();
    ggml_cuda_pdl_sync();
    for (int64_t z = blockIdx.z; z < ne11*(int64_t)ne12_fdv.z; z += gridDim.z) {
        const int i10 = blockIdx.x;
        const uint2 dm = fast_div_modulo((uint32_t)z, ne12_fdv);
        const int i11 = dm.x;
        const int i12 = dm.y;

        const int i01 = src1_ptr[i10*s10 + i11*s11 + i12*s12];

        int4       * GGML_CUDA_RESTRICT dst_row  = (int4 *)      (dst_ptr + i10*s1 + i11*s2 + i12*s3);
        const int4 * GGML_CUDA_RESTRICT src0_row = (const int4 *)((const char *) src0_ptr + i01*nb01 + i11*nb02 + i12*nb03);

        for (int64_t i = blockIdx.y*blockDim.x + threadIdx.x; i < ne00v; i += gridDim.y*blockDim.x) {
            dst_row[i] = src0_row[i];
        }
    }
}

template<typename grad_t, typename dst_t>
static __global__ void k_get_rows_back_float(
        const grad_t * __restrict__ grad, const int32_t * __restrict__ rows, dst_t * __restrict__ dst,
        const int64_t ncols, const int64_t nrows_grad, const int64_t nrows_dst) {
    const int col = blockIdx.x*blockDim.x + threadIdx.x;

    if (col >= ncols) {
        return;
    }

    ggml_cuda_pdl_sync();

    // grid.y is clamped to the CUDA grid limit, so stride over the destination rows
    for (int64_t dst_row = blockIdx.y; dst_row < nrows_dst; dst_row += gridDim.y) {
        float sum = 0.0f;

        for (int64_t i = 0; i < nrows_grad; ++i) {
            if (rows[i] != dst_row) {
                continue;
            }
            sum += grad[i*ncols + col];
        }

        dst[dst_row*ncols + col] = sum;
    }
}

template<int qk, int qr, dequantize_kernel_t dq, typename dst_t>
static void get_rows_cuda_q(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);
    const int block_num_y = (ne00 + 2*CUDA_GET_ROWS_BLOCK_SIZE - 1) / (2*CUDA_GET_ROWS_BLOCK_SIZE);
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne00 % 2 == 0);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows<qk, qr, dq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<int block_dim, typename dst_t, dequantize_kq_t<dst_t> dequantize_kq>
static void get_rows_cuda_kq(
        const void * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    GGML_ASSERT(ne00 % QK_K == 0);
    const int64_t nsb = ne00/QK_K;

    const dim3 block_dims(block_dim, 1, 1);
    const dim3 block_nums(ne10, MIN(nsb, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    k_get_rows_kq<dst_t, dequantize_kq><<<block_nums, block_dims, 0, stream>>>(
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template<typename src0_t, typename dst_t>
static void get_rows_cuda_float(
        const src0_t * src0_d, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    const dim3 block_dims(CUDA_GET_ROWS_BLOCK_SIZE, 1, 1);

    // strides in elements
    // const size_t s0 = nb0 / sizeof(dst_t);
    const size_t s1 = nb1 / sizeof(dst_t);
    const size_t s2 = nb2 / sizeof(dst_t);
    const size_t s3 = nb3 / sizeof(dst_t);

    const size_t s10 = nb10 / sizeof(int32_t);
    const size_t s11 = nb11 / sizeof(int32_t);
    const size_t s12 = nb12 / sizeof(int32_t);
    // const size_t s13 = nb13 / sizeof(int32_t);

    GGML_ASSERT(ne12 > 0);
    GGML_ASSERT(ne11 <= std::numeric_limits<uint32_t>::max() / ne12);
    const uint3 ne12_fdv = init_fastdiv_values(ne12);

    if constexpr (std::is_same<src0_t, dst_t>::value) {
        constexpr int VEC = 16 / sizeof(dst_t);
        const int64_t ne00v = ne00 / VEC;
        const int64_t vec_block_num_y = (ne00v + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
        const bool enough_blocks = vec_block_num_y * ne10 * ne11 * ne12 >= 128;
        const bool can_vec = VEC > 1 && enough_blocks &&
            (ne00 % VEC == 0) &&
            (nb01 % 16 == 0) && (nb02 % 16 == 0) && (nb03 % 16 == 0) &&
            (nb1  % 16 == 0) && (nb2  % 16 == 0) && (nb3  % 16 == 0) &&
            (((uintptr_t) src0_d) % 16 == 0) && (((uintptr_t) dst_d) % 16 == 0);

        if (can_vec) {
            const int block_num_y = vec_block_num_y;
            const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));
            const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
            ggml_cuda_kernel_launch(k_get_rows_float_vec<dst_t>, launch_params,
                (const dst_t *) src0_d, src1_d, dst_d,
                ne00v, ne11, ne12_fdv,
                s1, s2, s3,
                nb01, nb02, nb03,
                s10, s11, s12);
            return;
        }
    }

    const int block_num_y = (ne00 + CUDA_GET_ROWS_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BLOCK_SIZE;
    const dim3 block_nums(ne10, MIN(block_num_y, UINT16_MAX), MIN(ne11*ne12, UINT16_MAX));

    const ggml_cuda_kernel_launch_params launch_params = ggml_cuda_kernel_launch_params{block_nums, block_dims, 0, stream};
    ggml_cuda_kernel_launch(k_get_rows_float<src0_t, dst_t>, launch_params,
        src0_d, src1_d, dst_d,
        ne00, /*ne01, ne02, ne03,*/
        /*ne10,*/ ne11, ne12_fdv, /*ne13,*/
        /* s0,*/ s1, s2, s3,
        /* nb00,*/ nb01, nb02, nb03,
        s10, s11, s12/*, s13*/);
}

template <typename dst_t>
static void ggml_cuda_get_rows_switch_src0_type(
        const void * src0_d, const ggml_type src0_type, const int32_t * src1_d, dst_t * dst_d,
        const int64_t ne00, const size_t nb01, const size_t nb02, const size_t nb03,
        const int64_t ne10, const int64_t ne11, const int64_t ne12, const size_t nb10, const size_t nb11, const size_t nb12,
        const size_t nb1, const size_t nb2, const size_t nb3,
        cudaStream_t stream) {
    switch (src0_type) {
        case GGML_TYPE_F16:
            get_rows_cuda_float((const half *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F32:
            get_rows_cuda_float((const float *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            get_rows_cuda_float((const int32_t *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            get_rows_cuda_float((const nv_bfloat16 *) src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q1_0:
            get_rows_cuda_q<QK1_0, QR1_0, dequantize_q1_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_0:
            get_rows_cuda_q<QK2_0, QR2_0, dequantize_q2_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_0:
            get_rows_cuda_q<QK4_0, QR4_0, dequantize_q4_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_1:
            get_rows_cuda_q<QK4_1, QR4_1, dequantize_q4_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_0:
            get_rows_cuda_q<QK5_0, QR5_0, dequantize_q5_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_1:
            get_rows_cuda_q<QK5_1, QR5_1, dequantize_q5_1>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q8_0:
            get_rows_cuda_q<QK8_0, QR8_0, dequantize_q8_0>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q2_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q2_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q3_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q3_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q4_K:
            get_rows_cuda_kq<32, dst_t, dequantize_q4_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q5_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q5_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_Q6_K:
            get_rows_cuda_kq<64, dst_t, dequantize_q6_K<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ2_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq2_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_XXS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_xxs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ3_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq3_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_S:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_s<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ1_M:
            get_rows_cuda_kq<32, dst_t, dequantize_iq1_m<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_NL:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_nl<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_IQ4_XS:
            get_rows_cuda_kq<32, dst_t, dequantize_iq4_xs<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_MXFP4:
            get_rows_cuda_kq<32, dst_t, dequantize_mxfp4<dst_t>>(src0_d, src1_d, dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported src0 type: %s\n", __func__, ggml_type_name(src0_type));
            break;
    }
}

void get_rows_cuda(
        const void * src0_d, ggml_type src0_type, const int32_t * src1_d, void * dst_d, ggml_type dst_type,
        int64_t ne00, size_t nb01, size_t nb02, size_t nb03,
        int64_t ne10, int64_t ne11, int64_t ne12, size_t nb10, size_t nb11, size_t nb12,
        size_t nb1, size_t nb2, size_t nb3,
        cudaStream_t stream) {
    switch (dst_type) {
        case GGML_TYPE_F32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (float *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_I32:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (int32_t *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (half *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_get_rows_switch_src0_type(src0_d, src0_type, src1_d, (nv_bfloat16 *) dst_d,
                ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
            break;
        default:
            GGML_ABORT("%s: unsupported dst type: %s\n", __func__, ggml_type_name(dst_type));
            break;
    }
}

bool ggml_cuda_buffer_is_mapped(ggml_backend_buffer_t buffer); // ggml-cuda.cu

// The rows a gather picks from a table in mapped host memory, copied over PCIe into a stage in VRAM as wide coalesced
// reads, many in flight, with the stage's row index beside them. Dequantizing in place reads each block's scales and
// quants as small dependent loads across the bus: Gemma 4's per-layer embedding table (Q6_K, 8.8 KB a token) came
// over at ~2.7 GB/s that way, 1.1 MB in 425 us for a 128-token verify.
// A row is read as whole aligned 128-byte lines (a warp's load is one line, not two halves of neighbours) and in chunks
// of STAGE_CHUNK words, a block each, so many lines are in flight across the bus at once.
static constexpr int STAGE_THREADS = 128;
static constexpr int STAGE_U       = 8;
static constexpr int STAGE_CHUNK   = STAGE_THREADS*STAGE_U;

// A VRAM cache of the rows such a gather picks: a generated token is one of a few thousand over a long run (1.1k
// distinct ids made 99% of 49k tokens of the bench replies), so a row that came over PCIe once is read from VRAM after.
// A row that misses claims a slot for its token (slot_of -2 while it is written: another row of the same token in the
// batch reads host memory), the stage copy writes the slot as well, and the slot is published once the copy is done.
// When the slots run out, rows keep coming from host memory. The rows are the table's bytes, so the gather's values are
// the same. GGML_CUDA_MAPPED_CACHE_MB sizes it (default 192, 0 off).
struct mapped_row_cache {
    uint32_t * rows    = nullptr;
    int32_t  * slot_of = nullptr; // per table row: its slot, -1 none, -2 being written
    int32_t  * n_used  = nullptr;
    int32_t    cap     = 0;
    int64_t    words   = 0;       // a slot's words, rows 128-byte aligned
};
static std::mutex mapped_cache_mtx;
static std::unordered_map<const void *, mapped_row_cache> mapped_caches;

// each gathered row's source: a slot to read (>= 0), a slot to write as it reads host memory (~slot), or host only
static __global__ void k_cache_plan(const int32_t * __restrict__ ids, const int64_t s10, const int64_t n, int32_t * __restrict__ slot_of,
        int32_t * __restrict__ n_used, const int32_t cap, int32_t * __restrict__ plan) {
    const int64_t r = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (r >= n) {
        return;
    }
    const int32_t tok = ids[r*s10];
    const int32_t s   = slot_of[tok];
    int32_t p = INT_MIN;
    if (s >= 0) {
        p = s;
    } else if (s == -1 && *(volatile int32_t *) n_used < cap && atomicCAS(&slot_of[tok], -1, -2) == -1) {
        const int32_t k = atomicAdd(n_used, 1);
        if (k < cap) {
            p = ~k;
        } else {
            slot_of[tok] = -1;
        }
    }
    plan[r] = p;
}

static __global__ void k_cache_publish(const int32_t * __restrict__ ids, const int64_t s10, const int64_t n, const int32_t * __restrict__ plan,
        int32_t * __restrict__ slot_of) {
    const int64_t r = (int64_t) blockIdx.x*blockDim.x + threadIdx.x;
    if (r < n && plan[r] < 0 && plan[r] != INT_MIN) {
        slot_of[ids[r*s10]] = ~plan[r];
    }
}

static __global__ void k_stage_rows(const char * __restrict__ src0, const int32_t * __restrict__ ids, const int64_t s10,
        const size_t nb01, const int64_t n_words, uint32_t * __restrict__ stage, int32_t * __restrict__ iota,
        const int32_t * __restrict__ plan, uint32_t * __restrict__ cache, const int64_t cache_words) {
    const int64_t r   = blockIdx.x;
    const int32_t p   = plan != nullptr ? plan[r] : INT_MIN;
    if (blockIdx.y == 0 && threadIdx.x == 0) {
        iota[r] = (int32_t) r;
    }
    if (p >= 0) {
        // a cached row, in VRAM and aligned
        const uint32_t * src = cache + (int64_t) p*cache_words;
        uint32_t       * dst = stage + r*n_words;
        const int64_t i0 = (int64_t) blockIdx.y*STAGE_CHUNK + threadIdx.x;
#pragma unroll
        for (int u = 0; u < STAGE_U; ++u) {
            const int64_t i = i0 + (int64_t) u*STAGE_THREADS;
            if (i < n_words) {
                dst[i] = src[i];
            }
        }
        return;
    }
    const int32_t i01 = ids[r*s10];
    const char * row = src0 + (int64_t) i01*nb01;
    // the row's first byte within its 128-byte line, in words
    const int64_t shift = (int64_t) (((uintptr_t) row & 127) / sizeof(uint32_t));
    const uint32_t * src = (const uint32_t *) (row - shift*sizeof(uint32_t));
    uint32_t       * dst = stage + r*n_words - shift;
    const int64_t i_end = n_words + shift;
    const int64_t i0 = (int64_t) blockIdx.y*STAGE_CHUNK + threadIdx.x;
    uint32_t v[STAGE_U];
#pragma unroll
    for (int u = 0; u < STAGE_U; ++u) {
        const int64_t i = i0 + (int64_t) u*STAGE_THREADS;
        v[u] = i < i_end ? src[i] : 0u;
    }
    uint32_t * cdst = p != INT_MIN ? cache + (int64_t) (~p)*cache_words - shift : nullptr;
#pragma unroll
    for (int u = 0; u < STAGE_U; ++u) {
        const int64_t i = i0 + (int64_t) u*STAGE_THREADS;
        if (i >= shift && i < i_end) {
            dst[i] = v[u];
            if (cdst != nullptr) {
                cdst[i] = v[u];
            }
        }
    }
}

// the cache of a mapped table's rows (data, n_rows of row_bytes), made on first use outside a capture, or nullptr
static const mapped_row_cache * mapped_cache_get(cudaStream_t stream, const void * data, const int64_t n_rows, const size_t row_bytes) {
    static const int64_t mb = getenv("GGML_CUDA_MAPPED_CACHE_MB") ? atoll(getenv("GGML_CUDA_MAPPED_CACHE_MB")) : 192;
    if (mb <= 0) {
        return nullptr;
    }
    std::lock_guard<std::mutex> lock(mapped_cache_mtx);
    auto it = mapped_caches.find(data);
    if (it != mapped_caches.end()) {
        return it->second.rows ? &it->second : nullptr;
    }
    cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
    CUDA_CHECK(cudaStreamIsCapturing(stream, &capture));
    if (capture != cudaStreamCaptureStatusNone) {
        return nullptr;
    }
    mapped_row_cache & c = mapped_caches[data];
    c.words = (int64_t) (((row_bytes + 127) / 128) * 128 / sizeof(uint32_t));
    c.cap   = (int32_t) std::min<int64_t>(n_rows, (mb << 20) / (c.words*(int64_t) sizeof(uint32_t)));
    if (c.cap <= 0 || n_rows > INT32_MAX) {
        c.cap = 0;
        return nullptr;
    }
    CUDA_CHECK(cudaMalloc(&c.rows, (size_t) c.cap*c.words*sizeof(uint32_t)));
    CUDA_CHECK(cudaMalloc(&c.slot_of, (size_t) n_rows*sizeof(int32_t)));
    CUDA_CHECK(cudaMalloc(&c.n_used, sizeof(int32_t)));
    CUDA_CHECK(cudaMemsetAsync(c.slot_of, 0xFF, (size_t) n_rows*sizeof(int32_t), stream));
    CUDA_CHECK(cudaMemsetAsync(c.n_used, 0, sizeof(int32_t), stream));
    GGML_LOG_INFO("%s: %d rows of a mapped table cached in VRAM (%.0f MB)\n", __func__, c.cap, c.cap*c.words*sizeof(uint32_t)/1e6);
    return &c;
}

void ggml_cuda_mapped_cache_forget(const void * base, const size_t size) {
    std::lock_guard<std::mutex> lock(mapped_cache_mtx);
    for (auto it = mapped_caches.begin(); it != mapped_caches.end();) {
        const char * p = (const char *) it->first;
        if (p >= (const char *) base && p < (const char *) base + size) {
            if (it->second.rows) {
                CUDA_CHECK(cudaFree(it->second.rows));
                CUDA_CHECK(cudaFree(it->second.slot_of));
                CUDA_CHECK(cudaFree(it->second.n_used));
            }
            it = mapped_caches.erase(it);
        } else {
            ++it;
        }
    }
}

void ggml_cuda_op_get_rows(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    cudaStream_t stream = ctx.stream();

    GGML_TENSOR_BINARY_OP_LOCALS

    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(ne13 == 1);

    GGML_ASSERT(src0->nb[0] == ggml_type_size(src0->type));
    GGML_ASSERT(src1->nb[0] == ggml_type_size(src1->type));
    GGML_ASSERT(dst->nb[0]  == ggml_type_size(dst->type));

    // a quantized table in mapped host memory: its picked rows come into VRAM first, then the same kernel dequantizes
    // them there (the same bytes, so the same values). GGML_CUDA_MAPPED_NO_STAGE=1 reads the table in place.
    static const bool no_stage = getenv("GGML_CUDA_MAPPED_NO_STAGE") != nullptr;
    const size_t row_bytes = ggml_row_size(src0->type, ne00);
    if (!no_stage && ggml_is_quantized(src0->type) && ggml_cuda_buffer_is_mapped(src0->buffer) &&
            ne02 == 1 && ne03 == 1 && ne11 == 1 && ne12 == 1 && ne10 > 0 &&
            row_bytes % sizeof(uint32_t) == 0 && nb01 % sizeof(uint32_t) == 0 && ((uintptr_t) src0->data) % sizeof(uint32_t) == 0) {
        const int64_t n_words = (int64_t) (row_bytes / sizeof(uint32_t));
        const int64_t s10     = (int64_t) (nb10 / sizeof(int32_t));
        ggml_cuda_pool_alloc<uint32_t> stage(ctx.pool(), (size_t) ne10*n_words);
        ggml_cuda_pool_alloc<int32_t>  iota (ctx.pool(), (size_t) ne10);
        ggml_cuda_pool_alloc<int32_t>  plan (ctx.pool());
        const mapped_row_cache * cache = mapped_cache_get(stream, src0->data, ne01, row_bytes);
        if (cache != nullptr) {
            plan.alloc((size_t) ne10);
            k_cache_plan<<<(unsigned int) ((ne10 + 127) / 128), 128, 0, stream>>>((const int32_t *) src1->data, s10, ne10, cache->slot_of,
                    cache->n_used, cache->cap, plan.get());
        }
        // the words of a row plus up to a line's worth before it
        const unsigned int n_chunks = (unsigned int) ((n_words + 32 + STAGE_CHUNK - 1) / STAGE_CHUNK);
        k_stage_rows<<<dim3((unsigned int) ne10, n_chunks, 1), STAGE_THREADS, 0, stream>>>((const char *) src0->data, (const int32_t *) src1->data,
                s10, nb01, n_words, stage.get(), iota.get(), cache ? plan.get() : nullptr, cache ? cache->rows : nullptr, cache ? cache->words : 0);
        CUDA_CHECK(cudaGetLastError());
        if (cache != nullptr) {
            k_cache_publish<<<(unsigned int) ((ne10 + 127) / 128), 128, 0, stream>>>((const int32_t *) src1->data, s10, ne10, plan.get(), cache->slot_of);
        }
        get_rows_cuda(stage.get(), src0->type, iota.get(), dst->data, dst->type,
            ne00, row_bytes, row_bytes*ne10, row_bytes*ne10, ne10, 1, 1,
            sizeof(int32_t), sizeof(int32_t)*ne10, sizeof(int32_t)*ne10, nb1, nb2, nb3, stream);
        return;
    }

    get_rows_cuda(src0->data, src0->type, (const int32_t *) src1->data, dst->data, dst->type,
        ne00, nb01, nb02, nb03, ne10, ne11, ne12, nb10, nb11, nb12, nb1, nb2, nb3, stream);
}

void ggml_cuda_op_get_rows_back(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0]; // gradients of forward pass output
    const ggml_tensor * src1 = dst->src[1]; // src1 in forward pass

    GGML_TENSOR_BINARY_OP_LOCALS

    const float   * src0_d = (const float   *) src0->data;
    const int32_t * src1_d = (const int32_t *) src1->data;
    float         * dst_d  = (float         *) dst->data;

    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(src0->type == GGML_TYPE_F32);
    GGML_ASSERT(src1->type == GGML_TYPE_I32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_ASSERT(ggml_is_contiguous(src0));
    GGML_ASSERT(ggml_is_contiguous(src1));
    GGML_ASSERT(ggml_is_contiguous(dst));

    GGML_ASSERT(ne02*ne03 == 1);
    GGML_ASSERT(ne12*ne13 == 1);
    GGML_ASSERT(ne2*ne3 == 1);

    const dim3 block_dims(CUDA_GET_ROWS_BACK_BLOCK_SIZE, 1, 1);
    const int block_num_x = (ne00 + CUDA_GET_ROWS_BACK_BLOCK_SIZE - 1) / CUDA_GET_ROWS_BACK_BLOCK_SIZE;
    const dim3 block_nums(block_num_x, MIN(ne1, (int64_t)UINT16_MAX), 1);

    k_get_rows_back_float<<<block_nums, block_dims, 0, stream>>>(src0_d, src1_d, dst_d, ne00, ne10, ne1);
}

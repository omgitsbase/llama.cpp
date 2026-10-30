#include "ggml-cuda.h"
#include "ggml-impl.h"
#include "ggml-backend-impl.h"

#include "ggml-cuda/allreduce.cuh"
#include "ggml-cuda/common.cuh"
#include "ggml-cuda/acc.cuh"
#include "ggml-cuda/add-id.cuh"
#include "ggml-cuda/arange.cuh"
#include "ggml-cuda/argmax.cuh"
#include "ggml-cuda/argsort.cuh"
#include "ggml-cuda/binbcast.cuh"
#include "ggml-cuda/clamp.cuh"
#include "ggml-cuda/col2im-1d.cuh"
#include "ggml-cuda/concat.cuh"
#include "ggml-cuda/conv-transpose-1d.cuh"
#include "ggml-cuda/conv2d.cuh"
#include "ggml-cuda/conv2d-dw.cuh"
#include "ggml-cuda/conv2d-transpose.cuh"
#include "ggml-cuda/convert.cuh"
#include "ggml-cuda/count-equal.cuh"
#include "ggml-cuda/cpy.cuh"
#include "ggml-cuda/cross-entropy-loss.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/diagmask.cuh"
#include "ggml-cuda/diag.cuh"
#include "ggml-cuda/fattn.cuh"
#include "ggml-cuda/fwht.cuh"
#include "ggml-cuda/getrows.cuh"
#include "ggml-cuda/im2col.cuh"
#include "ggml-cuda/mmf.cuh"
#include "ggml-cuda/mmq.cuh"
#include "ggml-cuda/fp4gemm.cuh"
#include "ggml-cuda/mmvf.cuh"
#include "ggml-cuda/mmvq.cuh"
#include "ggml-cuda/moe-weighted-reduction.cuh"
#include "ggml-cuda/norm.cuh"
#include "ggml-cuda/opt-step-adamw.cuh"
#include "ggml-cuda/opt-step-sgd.cuh"
#include "ggml-cuda/out-prod.cuh"
#include "ggml-cuda/pad.cuh"
#include "ggml-cuda/pool2d.cuh"
#include "ggml-cuda/pool1d.cuh"
#include "ggml-cuda/quantize.cuh"
#include "ggml-cuda/rope.cuh"
#include "ggml-cuda/roll.cuh"
#include "ggml-cuda/scale.cuh"
#include "ggml-cuda/snake.cuh"
#include "ggml-cuda/softcap.cuh"
#include "ggml-cuda/softmax.cuh"
#include "ggml-cuda/ssm-conv.cuh"
#include "ggml-cuda/ssm-scan.cuh"
#include "ggml-cuda/sum.cuh"
#include "ggml-cuda/sumrows.cuh"
#include "ggml-cuda/top-k.cuh"
#include "ggml-cuda/mean.cuh"
#include "ggml-cuda/tsembd.cuh"
#include "ggml-cuda/topk-moe.cuh"
#include "ggml-cuda/unary.cuh"
#include "ggml-cuda/upscale.cuh"
#include "ggml-cuda/wkv.cuh"
#include "ggml-cuda/gla.cuh"
#include "ggml-cuda/gated_delta_net.cuh"
#include "ggml-cuda/dsv4-hc.cuh"
#include "ggml-cuda/set.cuh"
#include "ggml-cuda/set-rows.cuh"
#include "ggml-cuda/pad_reflect_1d.cuh"
#include "ggml-cuda/solve_tri.cuh"
#include "ggml-cuda/tri.cuh"
#include "ggml-cuda/cumsum.cuh"
#include "ggml-cuda/fill.cuh"
#include "ggml-cuda/lightning-indexer.cuh"
#include "ggml.h"

#include <algorithm>
#include <array>
#include <atomic>
#include <charconv>
#include <cinttypes>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <cfloat>
#include <initializer_list>
#include <limits>
#include <map>
#include <memory>
#include <mutex>
#include <cstdarg>
#include <cstdio>
#include <cstdlib>
#include <string>
#include <unordered_set>
#include <vector>

static_assert(sizeof(half) == sizeof(ggml_fp16_t), "wrong fp16 size");

#define GGML_LOG_WARN_ONCE(str) \
    { static std::once_flag warn_flag; std::call_once(warn_flag, []() { GGML_LOG_WARN(str); }); }

[[noreturn]]
void ggml_cuda_error(const char * stmt, const char * func, const char * file, int line, const char * msg) {
    int id = -1; // in case cudaGetDevice fails
    (void)cudaGetDevice(&id);

    GGML_LOG_ERROR(GGML_CUDA_NAME " error: %s\n", msg);
    GGML_LOG_ERROR("  current device: %d, in function %s at %s:%d\n", id, func, file, line);
    GGML_LOG_ERROR("  %s\n", stmt);
    // abort with GGML_ABORT to get a stack trace
    GGML_ABORT(GGML_CUDA_NAME " error");
}

// map a (possibly virtual) device id to the physical CUDA device that backs it
static int ggml_cuda_get_physical_device(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_device;
}

// this is faster on Windows
// probably because the Windows CUDA libraries forget to make this check before invoking the drivers
void ggml_cuda_set_device(int device) {
    // translate the (possibly virtual) device id to the physical CUDA device that backs it
    const int physical_device = ggml_cuda_get_physical_device(device);

    int current_device;
    CUDA_CHECK(cudaGetDevice(&current_device));

    if (physical_device == current_device) {
        return;
    }

    CUDA_CHECK(cudaSetDevice(physical_device));
}

int ggml_cuda_get_device() {
    int id;
    CUDA_CHECK(cudaGetDevice(&id));
    return id;
}

static cudaError_t ggml_cuda_device_malloc(void ** ptr, size_t size, int device) {
    ggml_cuda_set_device(device);
    cudaError_t err;
    if (getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr) {
        err = cudaMallocManaged(ptr, size);
#if defined(GGML_USE_HIP)
        if (err == hipSuccess) {
            // hipMemAdviseSetCoarseGrain is an optional performance hint;
            // ignore errors (e.g. hipErrorInvalidValue on some APU/iGPU configs).
            (void)cudaMemAdvise(*ptr, size, hipMemAdviseSetCoarseGrain, device);
            (void)hipGetLastError(); // clear any error
        }

        // fall back to cudaMalloc if not supported (e.g. on Windows)
        if (err == hipErrorNotSupported) {
            static bool warned_unsupported = false;
            if (!warned_unsupported) {
                GGML_LOG_WARN("hipMallocManaged unsupported, falling back to hipMalloc.\n");
                warned_unsupported = true;
            }

            err = cudaMalloc(ptr, size);
        }
#endif // defined(GGML_USE_HIP)
    } else {
        err = cudaMalloc(ptr, size);
    }
    return err;
}

#if defined(GGML_USE_HIP)
static int ggml_cuda_parse_id(char devName[]) {
    // A list of possible Target IDs can be found under the rocclr/clr repo in device.cpp
    // these values are not stable so this is susceptible to breakage
    // https://github.com/ROCm/clr/blob/amd-staging/rocclr/device/device.cpp
    int archMajor = 0x0;
    int archMinor = 0x0;
    int archNum = GGML_CUDA_CC_OFFSET_AMD;
    int archLen = strlen(devName);
    char archName[archLen + 1];

    // strip leading 'gfx' while copying into our buffer
    if (archLen > 3) {
        strcpy(archName, &devName[3]);
        archLen -= 3;
    }

    // trim trailing :xnack- or :sramecc- statuses
    archLen = strcspn(archName, ":");
    archName[archLen] = '\0';

    // tease out the version information
    if (archLen > 8) {
        // versions labeled generic use '-' as delimiter
        // strip the trailing "-generic" then iterate through what remains
        if ((strstr(archName, "-generic"))) {
            archName[archLen - 8] = '\0';
            char * pch;
            if ((pch = strtok(archName, "-"))) {
                archMajor = (int)strtoul(pch, 0, 16);
                if ((pch = strtok(NULL, "-"))) {
                    archMinor = 0x10 * (int)strtoul(pch, 0, 16);
                }
            }
        }
    } else if (archLen >= 3) {
        // last two digits should be the minor * 0x10 + stepping
        archMinor = (int)strtoul(&archName[archLen - 2], 0, 16);
        archName[archLen - 2] = '\0';

        // only the major version remains
        archMajor = (int)strtoul(archName, 0, 16);
    }
    archNum += archMajor * 0x100;
    archNum += archMinor;

    return archNum;
}
#endif // defined(GGML_USE_HIP)

static ggml_cuda_device_info ggml_cuda_init() {
    ggml_cuda_device_info info = {};

    cudaError_t err = cudaGetDeviceCount(&info.physical_device_count);
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: failed to initialize " GGML_CUDA_NAME ": %s\n", __func__, cudaGetErrorString(err));
        return info;
    }

    GGML_ASSERT(info.physical_device_count <= GGML_CUDA_MAX_DEVICES);

    // by default expose exactly the physical devices; GGML_CUDA_DEVICES can request a different
    // number of (virtual) devices to emulate multi-GPU systems on a machine with fewer GPUs
    info.device_count = info.physical_device_count;

    const char * devices_env = getenv("GGML_CUDA_DEVICES");
    if (devices_env != nullptr && info.physical_device_count > 0) {
        const int requested = atoi(devices_env);
        if (requested > 0) {
            info.device_count = requested;
        } else {
            GGML_LOG_WARN("%s: ignoring invalid GGML_CUDA_DEVICES=\"%s\"\n", __func__, devices_env);
        }
    }

    if (info.device_count > GGML_CUDA_MAX_DEVICES) {
        GGML_LOG_WARN("%s: requested %d devices, clamping to GGML_CUDA_MAX_DEVICES=%d\n",
                      __func__, info.device_count, GGML_CUDA_MAX_DEVICES);
        info.device_count = GGML_CUDA_MAX_DEVICES;
    }

    // map each (virtual) device to a backing physical device (round-robin), assign each its index
    // among the (virtual) devices sharing that physical GPU, and store the per-physical share count
    int physical_share_count[GGML_CUDA_MAX_DEVICES] = {};
    GGML_ASSERT(info.device_count == 0 || info.physical_device_count > 0);
    for (int id = 0; id < info.device_count; ++id) {
        info.devices[id].physical_device = id % info.physical_device_count;
        info.devices[id].virtual_index  = physical_share_count[info.devices[id].physical_device]++;
    }

    int64_t total_vram = 0;
    for (int id = 0; id < info.physical_device_count; ++id) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, id));
        total_vram += prop.totalGlobalMem;
    }
    GGML_LOG_INFO("%s: found %d " GGML_CUDA_NAME " devices (Total VRAM: %zu MiB):\n",
                  __func__, info.physical_device_count, (size_t)(total_vram / (1024 * 1024)));
    if (info.device_count != info.physical_device_count) {
        GGML_LOG_INFO("%s: emulating %d virtual device(s) on %d physical device(s) (GGML_CUDA_DEVICES)\n",
                      __func__, info.device_count, info.physical_device_count);
    }
    total_vram = 0;

    std::vector<std::pair<int, std::string>> turing_devices_without_mma;
    for (int id = 0; id < info.device_count; ++id) {
        const int physical_id = info.devices[id].physical_device;

        int device_vmm = 0;

#if defined(GGML_USE_VMM)
        CUdevice device;
        CU_CHECK(cuDeviceGet(&device, physical_id));
        CU_CHECK(cuDeviceGetAttribute(&device_vmm, CU_DEVICE_ATTRIBUTE_VIRTUAL_MEMORY_MANAGEMENT_SUPPORTED, device));

        if (device_vmm) {
            CUmemAllocationProp alloc_prop = {};
            alloc_prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            alloc_prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            alloc_prop.location.id = physical_id;
            CU_CHECK(cuMemGetAllocationGranularity(&info.devices[id].vmm_granularity, &alloc_prop, CU_MEM_ALLOC_GRANULARITY_RECOMMENDED));
        }
#endif // defined(GGML_USE_VMM)
        info.devices[id].vmm = !!device_vmm;

        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, physical_id));

        // a virtual device owns only a share of its physical GPU's memory; report that share so the
        // logged per-device VRAM sums to the physical total above.
        GGML_ASSERT(physical_share_count[physical_id] > 0);
        info.devices[id].physical_share_count = physical_share_count[physical_id];
        const size_t device_vram = prop.totalGlobalMem / info.devices[id].physical_share_count;
        const size_t device_vram_mib = device_vram / (1024 * 1024);

        info.default_tensor_split[id] = total_vram;
        total_vram += device_vram;
        info.devices[id].integrated = false; // Temporarily disabled due to issues with corrupted output (e.g. #15034)
        info.devices[id].nsm        = prop.multiProcessorCount;
        info.devices[id].smpb       = prop.sharedMemPerBlock;
        info.devices[id].warp_size  = prop.warpSize;

#ifndef GGML_USE_MUSA
        int supports_coop_launch = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&supports_coop_launch, cudaDevAttrCooperativeLaunch, physical_id));
        info.devices[id].supports_cooperative_launch = !!supports_coop_launch;
#else
        info.devices[id].supports_cooperative_launch = false;
#endif // !(GGML_USE_MUSA)

#if defined(GGML_USE_HIP)
        info.devices[id].smpbo = prop.sharedMemPerBlock;

        info.devices[id].cc = ggml_cuda_parse_id(prop.gcnArchName);
        if ((info.devices[id].cc & 0xff00) == 0x0) {
            GGML_LOG_WARN("invalid architecture ID received for device %d %s: %s  cc %d.%d\n",
                            id, prop.name, prop.gcnArchName, prop.major, prop.minor);

            // Fallback to prop.major and prop.minor
            if (prop.major > 0) {
                info.devices[id].cc = GGML_CUDA_CC_OFFSET_AMD + prop.major * 0x100;
                info.devices[id].cc += prop.minor * 0x10;
            }
        }
        GGML_LOG_INFO("  Device %d: %s, %s (0x%x), VMM: %s, Wave Size: %d, VRAM: %zu MiB\n",
                      id, prop.name, prop.gcnArchName, info.devices[id].cc & 0xffff,
                      device_vmm ? "yes" : "no", prop.warpSize,
                      device_vram_mib);
#elif defined(GGML_USE_MUSA)
        // FIXME: Ensure compatibility with varying warp sizes across different MUSA archs.
        info.devices[id].warp_size = 32;
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = GGML_CUDA_CC_OFFSET_MTHREADS + prop.major * 0x100;
        info.devices[id].cc += prop.minor * 0x10;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
#else
        info.devices[id].smpbo = prop.sharedMemPerBlockOptin;
        info.devices[id].cc = 100*prop.major + 10*prop.minor;
        GGML_LOG_INFO("  Device %d: %s, compute capability %d.%d, VMM: %s, VRAM: %zu MiB\n",
                      id, prop.name, prop.major, prop.minor, device_vmm ? "yes" : "no",
                      device_vram_mib);
        std::string device_name(prop.name);
        if (device_name == "NVIDIA GeForce MX450") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name == "NVIDIA GeForce MX550") {
            turing_devices_without_mma.push_back({ id, device_name });
        } else if (device_name.substr(0, 21) == "NVIDIA GeForce GTX 16") {
            turing_devices_without_mma.push_back({ id, device_name });
        }

        // Temporary performance fix:
        // Setting device scheduling strategy for iGPUs with cc121 to "spinning" to avoid delays in cuda synchronize calls.
        // TODO: Check for future drivers the default scheduling strategy and
        // remove this call again when cudaDeviceScheduleSpin is default.
        if (prop.major == 12 && prop.minor == 1) {
            CUDA_CHECK(cudaSetDevice(physical_id));
            CUDA_CHECK(cudaSetDeviceFlags(cudaDeviceScheduleSpin));
        }

#endif  // defined(GGML_USE_HIP)
    }

    if (ggml_cuda_highest_compiled_arch(GGML_CUDA_CC_TURING) >= GGML_CUDA_CC_TURING && !turing_devices_without_mma.empty()) {
        GGML_LOG_INFO("The following devices will have suboptimal performance due to a lack of tensor cores:\n");
        for (size_t device_pos = 0; device_pos < turing_devices_without_mma.size(); device_pos++) {
            GGML_LOG_INFO(
                "  Device %d: %s\n", turing_devices_without_mma[device_pos].first, turing_devices_without_mma[device_pos].second.c_str());
        }
        GGML_LOG_INFO(
            "Consider compiling with CMAKE_CUDA_ARCHITECTURES=61-virtual;80-virtual and DGGML_CUDA_FORCE_MMQ to force the use of the Pascal code for Turing.\n");
    }

    for (int id = 0; id < info.device_count; ++id) {
        info.default_tensor_split[id] /= total_vram;
    }

    // configure logging to stdout
    // CUBLAS_CHECK(cublasLoggerConfigure(1, 1, 0, nullptr));

    if (getenv("GGML_CUDA_P2P") != nullptr) {
        for (int id = 0; id < info.physical_device_count; ++id) {
            CUDA_CHECK(cudaSetDevice(id));
            for (int id_other = 0; id_other < info.physical_device_count; ++id_other) {
                if (id == id_other) {
                    continue;
                }
                int can_access_peer;
                CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id, id_other));
                if (can_access_peer) {
                    CUDA_CHECK(cudaDeviceEnablePeerAccess(id_other, 0));
                }
            }
        }
    }

    return info;
}

const ggml_cuda_device_info & ggml_cuda_info() {
    static ggml_cuda_device_info info = ggml_cuda_init();
    return info;
}

// #define DEBUG_CUDA_MALLOC

// buffer pool for cuda (legacy)
struct ggml_cuda_pool_leg : public ggml_cuda_pool {
    static const int MAX_BUFFERS = 256;

    int device;
    struct ggml_cuda_buffer {
        void * ptr = nullptr;
        size_t size = 0;
    };

    ggml_cuda_buffer buffer_pool[MAX_BUFFERS] = {};
    size_t pool_size = 0;

    explicit ggml_cuda_pool_leg(int device) :
        device(device) {
    }

    ~ggml_cuda_pool_leg() {
        clear_pool();
        GGML_ASSERT(pool_size == 0);
    }

    void clear_pool() {
        ggml_cuda_set_device(device);
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer & b = buffer_pool[i];
            if (b.ptr != nullptr) {
                CUDA_CHECK(cudaFree(b.ptr));
                pool_size -= b.size;
                b.ptr  = nullptr;
                b.size = 0;
            }
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
#ifdef DEBUG_CUDA_MALLOC
        int nnz = 0;
        size_t max_size = 0;
#endif
        size_t best_diff = 1ull << 36;
        int ibest = -1;
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr != nullptr) {
#ifdef DEBUG_CUDA_MALLOC
                ++nnz;
                if (b.size > max_size) max_size = b.size;
#endif
                if (b.size >= size) {
                    size_t diff = b.size - size;
                    if (diff < best_diff) {
                        best_diff = diff;
                        ibest = i;
                        if (!best_diff) {
                            void * ptr = b.ptr;
                            *actual_size = b.size;
                            b.ptr = nullptr;
                            b.size = 0;
                            return ptr;
                        }
                    }
                }
            }
        }
        if (ibest >= 0) {
            ggml_cuda_buffer& b = buffer_pool[ibest];
            void * ptr = b.ptr;
            *actual_size = b.size;
            b.ptr = nullptr;
            b.size = 0;
            return ptr;
        }
        void * ptr;
        size_t look_ahead_size = (size_t) (1.05 * size);
        look_ahead_size = 256 * ((look_ahead_size + 255)/256);
        ggml_cuda_set_device(device);
        cudaError_t err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
        if (err == cudaErrorMemoryAllocation) {
            (void)cudaGetLastError();
            const size_t cached_bytes = pool_size;
            GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: alloc of %.2f MiB failed, flushing %.2f MiB of cached buffers and retrying\n",
                           device, look_ahead_size/1024.0/1024.0, cached_bytes/1024.0/1024.0);
            CUDA_CHECK(cudaDeviceSynchronize());
            clear_pool();
            err = ggml_cuda_device_malloc(&ptr, look_ahead_size, device);
            if (err == cudaSuccess) {
                GGML_LOG_DEBUG(GGML_CUDA_NAME " pool[%d]: retry succeeded\n", device);
            }
        }
        CUDA_CHECK(err);
        *actual_size = look_ahead_size;
        pool_size += look_ahead_size;
#ifdef DEBUG_CUDA_MALLOC
        GGML_LOG_INFO("%s[%d]: %d buffers, max_size = %u MB, pool_size = %u MB, requested %u MB\n", __func__, device, nnz,
                           (uint32_t)(max_size / 1024 / 1024), (uint32_t)(pool_size / 1024 / 1024), (uint32_t)(size / 1024 / 1024));
#endif
        return ptr;
    }

    void free(void * ptr, size_t size) override {
        for (int i = 0; i < MAX_BUFFERS; ++i) {
            ggml_cuda_buffer& b = buffer_pool[i];
            if (b.ptr == nullptr) {
                b.ptr = ptr;
                b.size = size;
                return;
            }
        }
        GGML_LOG_DEBUG(GGML_CUDA_NAME " buffer pool full, increase MAX_CUDA_BUFFERS\n");
        ggml_cuda_set_device(device);
        CUDA_CHECK(cudaFree(ptr));
        pool_size -= size;
    }
};

// pool with virtual memory
#if defined(GGML_USE_VMM)
struct ggml_cuda_pool_vmm : public ggml_cuda_pool {
    static const size_t CUDA_POOL_VMM_MAX_SIZE = 1ull << 35; // 32 GB

    int device;
    int physical_device;
    CUdeviceptr pool_addr = 0;
    size_t pool_used = 0;
    size_t pool_size = 0;
    size_t granularity;
#if defined(GGML_USE_HIP)
    std::vector<std::pair<CUdeviceptr, size_t>> mappings;
#endif

    explicit ggml_cuda_pool_vmm(int device) :
        device(device),
        physical_device(ggml_cuda_get_physical_device(device)),
        granularity(ggml_cuda_info().devices[device].vmm_granularity) {
    }

    ~ggml_cuda_pool_vmm() {
        if (pool_addr != 0) {
#if defined(GGML_USE_HIP)
            // Workaround for https://github.com/ROCm/ROCR-Runtime/issues/285
            for (std::pair<CUdeviceptr, size_t> & mapping : mappings) {
                CU_CHECK(cuMemUnmap(mapping.first, mapping.second));
            }
#else
            CU_CHECK(cuMemUnmap(pool_addr, pool_size));
#endif
            CU_CHECK(cuMemAddressFree(pool_addr, CUDA_POOL_VMM_MAX_SIZE));
        }
    }

    void * alloc(size_t size, size_t * actual_size) override {
        // round up the allocation size to the alignment to ensure that all allocations are aligned for all data types
        const size_t alignment = 128;
        size = alignment * ((size + alignment - 1) / alignment);

        size_t avail = pool_size - pool_used;

        if (size > avail) {
            // round up to the next multiple of the granularity
            size_t reserve_size = size - avail;
            reserve_size = granularity * ((reserve_size + granularity - 1) / granularity);

            GGML_ASSERT(pool_size + reserve_size <= CUDA_POOL_VMM_MAX_SIZE);

            // allocate more physical memory
            CUmemAllocationProp prop = {};
            prop.type = CU_MEM_ALLOCATION_TYPE_PINNED;
            prop.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
            prop.location.id = physical_device;
            CUmemGenericAllocationHandle handle;
            CU_CHECK(cuMemCreate(&handle, reserve_size, &prop, 0));

            // reserve virtual address space (if not already reserved)
            if (pool_addr == 0) {
                CU_CHECK(cuMemAddressReserve(&pool_addr, CUDA_POOL_VMM_MAX_SIZE, 0, 0, 0));
            }

            // map at the end of the pool
            CUdeviceptr start_ptr = (CUdeviceptr)((char *)(pool_addr) + pool_size);
            CU_CHECK(cuMemMap(start_ptr, reserve_size, 0, handle, 0));
#if defined(GGML_USE_HIP)
            mappings.push_back({start_ptr, reserve_size});
#endif

            // the memory allocation handle is no longer needed after mapping
            CU_CHECK(cuMemRelease(handle));

            // VMM Bug fix for P2P access if GGML_CUDA_P2P is set, or if NCCL build
            bool use_peer_access = getenv("GGML_CUDA_P2P") != nullptr;
#if defined(GGML_USE_NCCL)
            use_peer_access = true;
#endif // defined(GGML_USE_NCCL)

            if (use_peer_access) {
                // NCCL implicitly enables peer access (cudaDeviceEnablePeerAccess), and
                // GGML_CUDA_P2P enables it explicitly. Unlike cudaMalloc buffers, VMM
                // allocations do not become peer-accessible from that alone, so access
                // must be granted explicitly here. With virtual devices, grant access
                // on the backing *physical* devices (deduplicated, since several
                // virtual devices can map to the same physical GPU).
                std::vector<CUmemAccessDesc> access_descs;
                bool physical_seen[GGML_CUDA_MAX_DEVICES] = {};
                const int device_count = ggml_cuda_info().device_count;
                for (int id = 0; id < device_count; ++id) {
                    const int id_physical = ggml_cuda_get_physical_device(id);
                    if (id_physical != physical_device) {
                        int can_access_peer = 0;
                        CUDA_CHECK(cudaDeviceCanAccessPeer(&can_access_peer, id_physical, physical_device));
                        if (!can_access_peer) {
                            continue;
                        }
                    }
                    if (physical_seen[id_physical]) {
                        continue;
                    }
                    physical_seen[id_physical] = true;
                    CUmemAccessDesc access = {};
                    access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                    access.location.id = id_physical;
                    access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                    access_descs.push_back(access);
                }
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, access_descs.data(), access_descs.size()));
            } else {
                // set access for non P2P
                CUmemAccessDesc access = {};
                access.location.type = CU_MEM_LOCATION_TYPE_DEVICE;
                access.location.id = physical_device;
                access.flags = CU_MEM_ACCESS_FLAGS_PROT_READWRITE;
                CU_CHECK(cuMemSetAccess(start_ptr, reserve_size, &access, 1));
            }

            // add to the pool
            pool_size += reserve_size;

            //printf("cuda pool[%d]: size increased to %llu MB (reserved %llu MB)\n",
            //       device, (unsigned long long) (pool_size/1024/1024),
            //       (unsigned long long) (reserve_size/1024/1024));
        }

        GGML_ASSERT(pool_addr != 0);

        void * ptr = (void *) ((CUdeviceptr)((char *)(pool_addr) + pool_used));
        *actual_size = size;
        pool_used += size;

#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: allocated %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        return ptr;
    }

    void free(void * ptr, size_t size) override {
#ifdef DEBUG_CUDA_MALLOC
        printf("cuda pool[%d]: freed %llu bytes at %llx\n", device, (unsigned long long) size, ptr);
#endif

        pool_used -= size;

        // all deallocations must be in reverse order of the allocations
        GGML_ASSERT(ptr == (void *) ((char *)(pool_addr) + pool_used));
    }
};
#endif // defined(GGML_USE_VMM)

std::unique_ptr<ggml_cuda_pool> ggml_backend_cuda_context::new_pool_for_device(int                  device,
                                                                               [[maybe_unused]] int stream_no) {
#if defined(GGML_USE_VMM)
    if (ggml_cuda_info().devices[device].vmm) {
        return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_vmm(device));
    }
#endif // defined(GGML_USE_VMM)
    return std::unique_ptr<ggml_cuda_pool>(new ggml_cuda_pool_leg(device));
}

// destroying a cuBLAS handle while a graph is being captured in a different thread can result in a CUDA error
// this lock is used to ensure that no cuBLAS handle is destroyed while a graph is being captured

static std::mutex ggml_cuda_lock;
static std::condition_variable ggml_cuda_lock_cv;
static std::atomic<int> ggml_cuda_lock_counter;

ggml_backend_cuda_context::~ggml_backend_cuda_context() {
    std::unique_lock<std::mutex> lock(ggml_cuda_lock);
    ggml_cuda_lock_cv.wait(lock, []{ return ggml_cuda_lock_counter.load(std::memory_order_relaxed) == 0; });

    if (copy_event != nullptr) {
        CUDA_CHECK(cudaEventDestroy(copy_event));
    }
    for (int i = 0; i < GGML_CUDA_MAX_DEVICES; ++i) {
        for (int j = 0; j < GGML_CUDA_MAX_STREAMS; ++j) {
            if (streams[i][j] != nullptr) {
                CUDA_CHECK(cudaStreamDestroy(streams[i][j]));
            }
            if (cublas_handles[i][j] != nullptr) {
                CUBLAS_CHECK(cublasDestroy(cublas_handles[i][j]));
            }
            if (cublas_workspaces[i][j] != nullptr) {
                CUDA_CHECK(cudaFree(cublas_workspaces[i][j]));
            }
        }
    }
}


// cuda buffer

struct ggml_backend_cuda_buffer_context {
    int device;
    void * dev_ptr = nullptr;
    std::string name;

    ggml_backend_cuda_buffer_context(int device, void * dev_ptr) :
        device(device), dev_ptr(dev_ptr),
        name(GGML_CUDA_NAME + std::to_string(device)) {
    }

    ~ggml_backend_cuda_buffer_context() {
        CUDA_CHECK(cudaFree(dev_ptr));
    }
};

static void ggml_backend_cuda_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    ggml_cuda_fp4_gemm_forget(ctx->device, ctx->dev_ptr, buffer->size);
    delete ctx;
}

static bool ggml_backend_buffer_is_cuda(ggml_backend_buffer_t buffer) {
    return buffer->iface.free_buffer == ggml_backend_cuda_buffer_free_buffer;
}

static void * ggml_backend_cuda_buffer_get_base(ggml_backend_buffer_t buffer) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;
    return ctx->dev_ptr;
}

static enum ggml_status ggml_backend_cuda_buffer_init_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    if (tensor->view_src != NULL) {
        assert(tensor->view_src->buffer->buft == buffer->buft);
        return GGML_STATUS_SUCCESS;
    }

    if (ggml_is_quantized(tensor->type) && tensor->view_src == nullptr && ggml_backend_buffer_get_usage(buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        // initialize padding to 0 to avoid possible NaN values
        const size_t original_size = ggml_nbytes(tensor);
        const size_t padded_size = ggml_backend_buft_get_alloc_size(buffer->buft, tensor);

        if (padded_size > original_size) {
            ggml_cuda_set_device(ctx->device);
            CUDA_CHECK(cudaMemset((char *)tensor->data + original_size, 0, padded_size - original_size));
        }
    }
    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_buffer_memset_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, uint8_t value, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync((char *) tensor->data + offset, value, size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor(ggml_backend_buffer_t buffer, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_set_tensor_2d(ggml_backend_buffer_t buffer, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *) buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyHostToDevice, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static void ggml_backend_cuda_buffer_get_tensor_2d(ggml_backend_buffer_t buffer, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDeviceToHost, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static bool ggml_backend_cuda_buffer_cpy_tensor(ggml_backend_buffer_t buffer, const ggml_tensor * src, ggml_tensor * dst) {
    if (ggml_backend_buffer_is_cuda(src->buffer)) {
        ggml_backend_cuda_buffer_context * src_ctx = (ggml_backend_cuda_buffer_context *)src->buffer->context;
        ggml_backend_cuda_buffer_context * dst_ctx = (ggml_backend_cuda_buffer_context *)dst->buffer->context;
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(src_ctx->device);
        const int dst_physical = ggml_cuda_get_physical_device(dst_ctx->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(src), cudaMemcpyDeviceToDevice, cudaStreamPerThread));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(src), cudaStreamPerThread));
#endif
        }
        CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
        return true;
    }
    return false;

    GGML_UNUSED(buffer);
}

static void ggml_backend_cuda_buffer_clear(ggml_backend_buffer_t buffer, uint8_t value) {
    ggml_backend_cuda_buffer_context * ctx = (ggml_backend_cuda_buffer_context *)buffer->context;

    ggml_cuda_set_device(ctx->device);
    CUDA_CHECK(cudaMemsetAsync(ctx->dev_ptr, value, buffer->size, cudaStreamPerThread));
    CUDA_CHECK(cudaStreamSynchronize(cudaStreamPerThread));
}

static const ggml_backend_buffer_i ggml_backend_cuda_buffer_interface = {
    /* .free_buffer     = */ ggml_backend_cuda_buffer_free_buffer,
    /* .get_base        = */ ggml_backend_cuda_buffer_get_base,
    /* .init_tensor     = */ ggml_backend_cuda_buffer_init_tensor,
    /* .memset_tensor   = */ ggml_backend_cuda_buffer_memset_tensor,
    /* .set_tensor      = */ ggml_backend_cuda_buffer_set_tensor,
    /* .get_tensor      = */ ggml_backend_cuda_buffer_get_tensor,
    /* .set_tensor_2d   = */ ggml_backend_cuda_buffer_set_tensor_2d,
    /* .get_tensor_2d   = */ ggml_backend_cuda_buffer_get_tensor_2d,
    /* .cpy_tensor      = */ ggml_backend_cuda_buffer_cpy_tensor,
    /* .clear           = */ ggml_backend_cuda_buffer_clear,
    /* .reset           = */ NULL,
};

// cuda buffer type
struct ggml_backend_cuda_buffer_type_context {
    int device;
    std::string name;
};

static const char * ggml_backend_cuda_buffer_type_get_name(ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_buffer_type_context * ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    return ctx->name.c_str();
}

static bool ggml_backend_buft_is_cuda(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_buffer_type_get_name;
}

static ggml_backend_buffer_t ggml_backend_cuda_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)buft->context;

    ggml_cuda_set_device(buft_ctx->device);

    void * dev_ptr;
    cudaError_t err = ggml_cuda_device_malloc(&dev_ptr, size, buft_ctx->device);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_ERROR("%s: allocating %.2f MiB on device %d: cudaMalloc failed: %s\n", __func__, size / 1024.0 / 1024.0, buft_ctx->device, cudaGetErrorString(err));
        return nullptr;
    }

    ggml_backend_cuda_buffer_context * ctx = new ggml_backend_cuda_buffer_context(buft_ctx->device, dev_ptr);

    return ggml_backend_buffer_init(buft, ggml_backend_cuda_buffer_interface, ctx, size);
}

static size_t ggml_backend_cuda_buffer_type_get_alignment(ggml_backend_buffer_type_t buft) {
    return 128;

    GGML_UNUSED(buft);
}

static size_t ggml_backend_cuda_buffer_type_get_alloc_size(ggml_backend_buffer_type_t buft, const ggml_tensor * tensor) {
    ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *) buft->context;

    size_t size = tensor->op == GGML_OP_FLASH_ATTN_EXT
        ? ggml_cuda_flash_attn_ext_get_alloc_size(buft_ctx->device, tensor)
        : ggml_nbytes(tensor);
    int64_t ne0 = tensor->ne[0];

    // [TAG_ALLOC_SIZE_EXPAND]
    if (ggml_is_quantized(tensor->type)) {
        if (ne0 % MATRIX_ROW_PADDING != 0) {
            GGML_ASSERT(tensor->nb[0] == ggml_element_size(tensor));
            size += ggml_row_size(tensor->type, MATRIX_ROW_PADDING - ne0 % MATRIX_ROW_PADDING);
        }
    }

    return size;
}

// The largest single buffer to ask the driver for. Under WDDM a cudaMalloc
// past roughly half the card fails outright, however much is free: a context
// of a million tokens on a 96 GB card wants its 64 GiB cache in one piece and
// is refused while the card holds 12 GB. Capping the buffer size makes ggml
// lay the same tensors out across several buffers instead.
static size_t ggml_backend_cuda_buffer_type_get_max_size(ggml_backend_buffer_type_t buft) {
    static const size_t cap = [] {
        if (const char * env = getenv("GGML_CUDA_MAX_BUFFER_GB")) {
            const double gb = atof(env);
            return gb > 0 ? (size_t) (gb * (1ull << 30)) : SIZE_MAX;
        }
#ifdef _WIN32
        return (size_t) 16 << 30;
#else
        return SIZE_MAX;
#endif
    }();
    return cap;

    GGML_UNUSED(buft);
}

static const ggml_backend_buffer_type_i ggml_backend_cuda_buffer_type_interface = {
    /* .get_name         = */ ggml_backend_cuda_buffer_type_get_name,
    /* .alloc_buffer     = */ ggml_backend_cuda_buffer_type_alloc_buffer,
    /* .get_alignment    = */ ggml_backend_cuda_buffer_type_get_alignment,
    /* .get_max_size     = */ ggml_backend_cuda_buffer_type_get_max_size,
    /* .get_alloc_size   = */ ggml_backend_cuda_buffer_type_get_alloc_size,
    /* .is_host          = */ NULL,
};

ggml_backend_buffer_type_t ggml_backend_cuda_buffer_type(int device) {
    static std::mutex mutex;
    std::lock_guard<std::mutex> lock(mutex);

    if (device >= ggml_backend_cuda_get_device_count()) {
        return nullptr;
    }

    static ggml_backend_buffer_type ggml_backend_cuda_buffer_types[GGML_CUDA_MAX_DEVICES];

    static bool ggml_backend_cuda_buffer_type_initialized = false;

    if (!ggml_backend_cuda_buffer_type_initialized) {
        for (int i = 0; i < ggml_backend_cuda_get_device_count(); i++) {
            ggml_backend_cuda_buffer_types[i] = {
                /* .iface    = */ ggml_backend_cuda_buffer_type_interface,
                /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), i),
                /* .context  = */ new ggml_backend_cuda_buffer_type_context{i, GGML_CUDA_NAME + std::to_string(i)},
            };
        }
        ggml_backend_cuda_buffer_type_initialized = true;
    }

    return &ggml_backend_cuda_buffer_types[device];
}

// Communication context for multi-GPU AllReduce during tensor parallelism.
//
// Created once per meta backend instance.  Resources for the selected mode
// (NCCL communicators or the internal AllReduce pipeline) are initialised
// eagerly during comm_init so any init failure surfaces at startup rather
// than mid-run.
struct ggml_backend_cuda_comm_context {
    using try_allreduce_fn = bool(*)(ggml_backend_cuda_comm_context *, struct ggml_tensor **);

    std::vector<ggml_backend_t> backends;
    std::vector<int>            dev_ids;

    // Set by the init chain (comm_init_{nccl, internal, none}) to one of
    // try_allreduce_{nccl, internal, butterfly}.  nccl needs `comms`,
    // internal needs `ar_pipeline`, butterfly needs nothing.  Per-call
    // failures return false; the meta backend's generic implementation then
    // handles that call.
    try_allreduce_fn            try_allreduce = nullptr;

    ggml_cuda_ar_pipeline *     ar_pipeline = nullptr;

#ifdef GGML_USE_NCCL
    std::vector<ncclComm_t>     comms;
#endif // GGML_USE_NCCL

    ~ggml_backend_cuda_comm_context() {
#ifdef GGML_USE_NCCL
        for (ncclComm_t comm : comms) {
            NCCL_CHECK(ncclCommDestroy(comm));
        }
#endif // GGML_USE_NCCL
        ggml_cuda_ar_pipeline_free(ar_pipeline);
    }
};

#ifdef GGML_USE_NCCL
// AllReduce via NCCL. Reduces as FP32 for small tensors and BF16 for large
// tensors (bandwidth-bound), then converts back to FP32.
static bool ggml_backend_cuda_comm_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    const int64_t ne = ggml_nelements(tensors[0]);
    // FIXME the input of llm_graph_context::build_in_out_ids can produce a tensor with 0 elements if n_outputs == 0
    // This then causes a crash in this function
    if (ne == 0) {
        return true;
    }

    const size_t n_backends = comm_ctx->backends.size();

    for (size_t i = 0; i < n_backends; ++i) {
        GGML_ASSERT(tensors[i] != nullptr);
        GGML_ASSERT(ggml_nelements(tensors[i]) == ne);
        GGML_ASSERT(ggml_is_contiguously_allocated(tensors[i]));
    }

    // For small tensors, simply reduce them as FP32.
    // The following heuristic for how "small" a tensor should be is based on RTX 4090s connected via 16x PCIe 4.0.
    if ((n_backends <= 2 && ne < 32768) || (n_backends == 3 && ne < 131072) || (n_backends >= 4 && ne < 262144)) {
        for (size_t i = 0; i < n_backends; ++i) {
            if ((tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
                ggml_cuda_set_device(cuda_ctx->device);
                CUDA_CHECK(cudaMemsetAsync(tensors[i]->data, 0, ggml_nbytes(tensors[i]), cuda_ctx->stream()));
            }
        }
        NCCL_CHECK(ncclGroupStart());
        for (size_t i = 0; i < n_backends; ++i) {
            ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
            NCCL_CHECK(ncclAllReduce(tensors[i]->data, tensors[i]->data, ne, ncclFloat, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
        }
        NCCL_CHECK(ncclGroupEnd());
        return true;
    }

    // For large tensors it's faster to compress them to BF16 for the reduction:
    to_bf16_cuda_t to_bf16 = ggml_get_to_bf16_cuda(GGML_TYPE_F32);
    to_fp32_cuda_t to_fp32 = ggml_get_to_fp32_cuda(GGML_TYPE_BF16);

    ggml_cuda_pool_alloc<nv_bfloat16> tmp[GGML_CUDA_MAX_DEVICES];
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        tmp[i].pool = &cuda_ctx->pool();
        tmp[i].alloc(ne);

        ggml_cuda_set_device(cuda_ctx->device);
        if (tensors[i]->flags & GGML_TENSOR_FLAG_COMPUTE) {
            to_bf16(tensors[i]->data, tmp[i].get(), ne, cuda_ctx->stream());
        } else {
            CUDA_CHECK(cudaMemsetAsync(tmp[i].get(), 0, ne * sizeof(nv_bfloat16), cuda_ctx->stream()));
        }
        CUDA_CHECK(cudaGetLastError());
    }

    NCCL_CHECK(ncclGroupStart());
    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;
        NCCL_CHECK(ncclAllReduce(tmp[i].get(), tmp[i].get(), ne, ncclBfloat16, ncclSum, comm_ctx->comms[i], cuda_ctx->stream()));
    }
    NCCL_CHECK(ncclGroupEnd());

    for (size_t i = 0; i < n_backends; ++i) {
        ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) comm_ctx->backends[i]->context;

        ggml_cuda_set_device(cuda_ctx->device);
        to_fp32(tmp[i].get(), (float *) tensors[i]->data, ne, cuda_ctx->stream());
        CUDA_CHECK(cudaGetLastError());
    }

    return true;
}
#endif // GGML_USE_NCCL

// Run the internal AR pipeline.  Returns false on unsupported / failed input
// -- the caller decides whether to abort (env-forced) or fall back silently.
static bool ggml_backend_cuda_comm_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    GGML_ASSERT(comm_ctx->ar_pipeline != nullptr);

    const size_t n_backends = comm_ctx->backends.size();
    GGML_ASSERT(n_backends == 2);
    GGML_ASSERT(tensors[0] != nullptr);

    const int64_t   ne   = ggml_nelements(tensors[0]);
    const ggml_type type = tensors[0]->type;

    if (type != GGML_TYPE_F32 && type != GGML_TYPE_F16 && type != GGML_TYPE_BF16) {
        GGML_LOG_DEBUG("%s: internal unsupported: type=%d\n", __func__, (int) type);
        return false;
    }

    if (ne == 0) {
        return true;
    }

    for (size_t i = 0; i < n_backends; ++i) {
        if (tensors[i] == nullptr) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] is null\n", __func__, i);
            return false;
        }
        if (ggml_nelements(tensors[i]) != ne || tensors[i]->type != type) {
            GGML_LOG_ERROR("%s: internal failed: tensor[%zu] ne=%" PRId64 " type=%d expected ne=%" PRId64 " type=%d\n",
                           __func__, i, ggml_nelements(tensors[i]), (int) tensors[i]->type, ne, (int) type);
            return false;
        }
        if (!ggml_is_contiguously_allocated(tensors[i])) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] is not contiguously allocated: ne=%" PRId64 " nbytes=%zu packed=%zu type=%d\n",
                           __func__, i, ne, ggml_nbytes(tensors[i]),
                           (size_t) ne * ggml_type_size(type) / ggml_blck_size(type), (int) type);
            return false;
        }
        if (((uintptr_t) tensors[i]->data & 0xF) != 0) {
            GGML_LOG_DEBUG("%s: internal unsupported: tensor[%zu] data pointer is not 16-byte aligned: %p type=%d ne=%" PRId64 "\n",
                           __func__, i, tensors[i]->data, (int) type, ne);
            return false;
        }
        GGML_ASSERT((ggml_nbytes(tensors[i]) & 0xF) == 0);
    }

    return ggml_cuda_ar_allreduce(comm_ctx->ar_pipeline, comm_ctx->backends.data(), tensors);
}

// ---------------------------------------------------------------------------
// Per-call dispatch -- three variants, one per backend.  Each is set as
// comm_ctx->try_allreduce by the matching init step.  Per-call failure
// returns false; the meta backend's generic implementation handles that call.
// ---------------------------------------------------------------------------

#ifdef GGML_USE_NCCL
static bool ggml_backend_cuda_comm_try_allreduce_nccl(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_nccl(comm_ctx, tensors);
}
#endif // GGML_USE_NCCL

static bool ggml_backend_cuda_comm_try_allreduce_internal(
        ggml_backend_cuda_comm_context * comm_ctx, struct ggml_tensor ** tensors) {
    return ggml_backend_cuda_comm_allreduce_internal(comm_ctx, tensors);
}

static bool ggml_backend_cuda_comm_try_allreduce_butterfly(
        ggml_backend_cuda_comm_context *, struct ggml_tensor **) {
    return false;
}

static void ggml_backend_cuda_comm_free(void * comm_ctx_v) {
    if (comm_ctx_v == nullptr) {
        return;
    }
    delete static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
}

// ---------------------------------------------------------------------------
// Init -- chained nccl -> internal -> none.  Each step tries to bring up its
// resource; on failure it warns and recurses into the next step.
// ---------------------------------------------------------------------------
static void ggml_backend_cuda_comm_init_none(ggml_backend_cuda_comm_context * ret) {
    ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_butterfly;
}

static void ggml_backend_cuda_comm_init_internal(ggml_backend_cuda_comm_context * ret) {
    ret->ar_pipeline = ggml_cuda_ar_pipeline_init(ret->dev_ids.data(), ret->dev_ids.size());
    if (ret->ar_pipeline) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_internal;
        return;
    }

    // Clear sticky CUDA error from the failed init.
    (void) cudaGetLastError();
    GGML_LOG_WARN("internal AllReduce init failed (n_devices != 2?); "
                  "falling back to meta-backend butterfly\n");
    ggml_backend_cuda_comm_init_none(ret);
}

static void ggml_backend_cuda_comm_init_nccl(ggml_backend_cuda_comm_context * ret) {
#ifdef GGML_USE_NCCL
    // Disabling NCCL path when CUDA virtual devices are in use since NCCL requires one distinct physical GPU per rank.
    const ggml_cuda_device_info & info = ggml_cuda_info();
    if (info.device_count > info.physical_device_count) {
        GGML_LOG_WARN("NCCL disabled: virtual devices in use; "
                      "falling back to internal AllReduce\n");
        ggml_backend_cuda_comm_init_internal(ret);
        return;
    }

    const size_t n = ret->dev_ids.size();
    ret->comms.resize(n);
    ncclResult_t rc = ncclCommInitAll(ret->comms.data(), (int) n, ret->dev_ids.data());
    if (rc == ncclSuccess) {
        ret->try_allreduce = ggml_backend_cuda_comm_try_allreduce_nccl;
        return;
    }

    ret->comms.clear();
    GGML_LOG_WARN("NCCL init failed (%s); falling back to internal AllReduce\n",
                  ncclGetErrorString(rc));
#else // GGML_USE_NCCL
#ifndef GGML_USE_HIP
    GGML_LOG_WARN("NCCL not compiled in; falling back to internal AllReduce.  "
                  "Recompile with -DGGML_CUDA_NCCL=ON for best multi-GPU performance.\n");
#endif // !GGML_USE_HIP
#endif // GGML_USE_NCCL

    ggml_backend_cuda_comm_init_internal(ret);
}

// Top-level init.  Picks one of the three init paths based on
// GGML_CUDA_ALLREDUCE (or the platform default) and lets the chain handle
// any fallback.  Unrecognised env values warn and fall through to the
// platform default.
static void * ggml_backend_cuda_comm_init(ggml_backend_t * backends, size_t n_backends) {
    for (size_t i = 0; i < n_backends; i++) {
        if (!ggml_backend_is_cuda(backends[i])) {
            return nullptr;
        }
    }

    auto * ret = new ggml_backend_cuda_comm_context;
    ret->backends.assign(backends, backends + n_backends);
    ret->dev_ids.reserve(n_backends);
    for (size_t i = 0; i < n_backends; i++) {
        ret->dev_ids.push_back(static_cast<ggml_backend_cuda_context *>(backends[i]->context)->device);
    }

    const char * env = getenv("GGML_CUDA_ALLREDUCE");
    if (!env) {
        // Platform default: Linux uses NCCL, otherwise (generally Windows) internal
#if defined(__linux__)
        ggml_backend_cuda_comm_init_nccl(ret);
#else
        ggml_backend_cuda_comm_init_internal(ret);
#endif // defined(__linux__)
    } else {
        std::string env_str(env);
        if (env_str == "nccl") {
            ggml_backend_cuda_comm_init_nccl(ret);
        } else if (env_str == "internal") {
            ggml_backend_cuda_comm_init_internal(ret);
        } else if (env_str == "none") {
            ggml_backend_cuda_comm_init_none(ret);
        } else {
            GGML_LOG_WARN("unknown GGML_CUDA_ALLREDUCE value: %s\n", env);
            ggml_backend_cuda_comm_init_none(ret);
        }
    }

    return ret;
}

// Top-level dispatch -- calls the function pointer chosen by comm_init.
// Returns false to let the meta-backend's butterfly run.
static bool ggml_backend_cuda_comm_allreduce_tensor(void * comm_ctx_v, struct ggml_tensor ** tensors) {
    if (comm_ctx_v == nullptr) {
        return false;
    }
    auto * comm_ctx = static_cast<ggml_backend_cuda_comm_context *>(comm_ctx_v);
    return comm_ctx->try_allreduce(comm_ctx, tensors);
}

// host buffer type

static const char * ggml_backend_cuda_host_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Host";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_host(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
}

static void ggml_backend_cuda_host_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

void ggml_cuda_mapped_cache_forget(const void * base, size_t size); // getrows.cu

// a mapped table's rows cached in VRAM go with it
static void ggml_backend_cuda_mapped_buffer_free_buffer(ggml_backend_buffer_t buffer) {
    ggml_cuda_mapped_cache_forget(buffer->context, buffer->size);
    CUDA_CHECK(cudaFreeHost(buffer->context));
}

static void * ggml_cuda_host_malloc(size_t size) {
    if (getenv("GGML_CUDA_NO_PINNED") != nullptr) {
        return nullptr;
    }

    void * ptr = nullptr;
    cudaError_t err = cudaMallocHost((void **) &ptr, size);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
        GGML_LOG_DEBUG("%s: failed to allocate %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }

    return ptr;
}

static ggml_backend_buffer_t ggml_backend_cuda_host_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = ggml_cuda_host_malloc(size);

    if (ptr == nullptr) {
        // fallback to cpu buffer
        return ggml_backend_buft_alloc_buffer(ggml_backend_cpu_buffer_type(), size);
    }

    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_host_buffer_free_buffer;

    return buffer;
}

ggml_backend_buffer_type_t ggml_backend_cuda_host_buffer_type() {
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_host = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_host_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_host_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cpu_buffer_type()->iface.is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_host;
}

//static bool ggml_backend_buffer_is_cuda_host(ggml_backend_buffer_t buffer) {
//    return buffer->buft->iface.get_name == ggml_backend_cuda_host_buffer_type_name;
//}

// mapped buffer type: pinned host memory a kernel reads in place, for tables the device only gathers a few rows
// from (token and per-layer embeddings). They stay out of VRAM, yet a graph can look ids up that it produced on the
// device. The scheduler sees device memory, so the gathers run on the device; matrix products refuse it.

static const char * ggml_backend_cuda_mapped_buffer_type_name(ggml_backend_buffer_type_t buft) {
    return GGML_CUDA_NAME "_Mapped";

    GGML_UNUSED(buft);
}

static bool ggml_backend_buft_is_cuda_mapped(ggml_backend_buffer_type_t buft) {
    return buft->iface.get_name == ggml_backend_cuda_mapped_buffer_type_name;
}

// for the kernels: whether a tensor's memory is mapped host memory (getrows.cu stages such rows)
bool ggml_cuda_buffer_is_mapped(ggml_backend_buffer_t buffer) {
    return buffer != nullptr && buffer->buft != nullptr && ggml_backend_buft_is_cuda_mapped(buffer->buft);
}

static bool ggml_backend_cuda_mapped_buffer_type_is_host(ggml_backend_buffer_type_t buft) {
    return false;

    GGML_UNUSED(buft);
}

static ggml_backend_buffer_t ggml_backend_cuda_mapped_buffer_type_alloc_buffer(ggml_backend_buffer_type_t buft, size_t size) {
    void * ptr = nullptr;
    cudaError_t err = cudaHostAlloc(&ptr, std::max<size_t>(size, 1), cudaHostAllocMapped | cudaHostAllocPortable);
    if (err != cudaSuccess) {
        (void) cudaGetLastError();
        GGML_LOG_ERROR("%s: failed to allocate %.2f MiB of mapped pinned memory: %s\n", __func__, size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return nullptr;
    }
    // with unified addressing (checked when the type is offered) the host pointer is the device's address as well
    ggml_backend_buffer_t buffer = ggml_backend_cpu_buffer_from_ptr(ptr, size);
    buffer->buft = buft;
    buffer->iface.free_buffer = ggml_backend_cuda_mapped_buffer_free_buffer;
    return buffer;
}

// nullptr when the devices cannot read host memory in place, or GGML_CUDA_NO_MAPPED is set
ggml_backend_buffer_type_t ggml_backend_cuda_mapped_buffer_type() {
    static const bool usable = [] {
        if (getenv("GGML_CUDA_NO_MAPPED") != nullptr) {
            return false;
        }
        for (int id = 0; id < ggml_backend_cuda_get_device_count(); ++id) {
            cudaDeviceProp prop;
            if (cudaGetDeviceProperties(&prop, id) != cudaSuccess || !prop.unifiedAddressing || !prop.canMapHostMemory) {
                (void) cudaGetLastError();
                return false;
            }
        }
        return true;
    }();
    if (!usable) {
        return nullptr;
    }
    static struct ggml_backend_buffer_type ggml_backend_cuda_buffer_type_mapped = {
        /* .iface    = */ {
            /* .get_name         = */ ggml_backend_cuda_mapped_buffer_type_name,
            /* .alloc_buffer     = */ ggml_backend_cuda_mapped_buffer_type_alloc_buffer,
            /* .get_alignment    = */ ggml_backend_cpu_buffer_type()->iface.get_alignment,
            /* .get_max_size     = */ NULL, // defaults to SIZE_MAX
            /* .get_alloc_size   = */ ggml_backend_cpu_buffer_type()->iface.get_alloc_size,
            /* .is_host          = */ ggml_backend_cuda_mapped_buffer_type_is_host,
        },
        /* .device   = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), 0),
        /* .context  = */ nullptr,
    };

    return &ggml_backend_cuda_buffer_type_mapped;
}

static ggml_backend_buffer_type_t ggml_backend_cuda_dev_get_mapped_buffer_type(ggml_backend_dev_t dev) {
    return ggml_backend_cuda_mapped_buffer_type();

    GGML_UNUSED(dev);
}

/// kernels

typedef void (*ggml_cuda_op_mul_mat_t)(
    ggml_backend_cuda_context & ctx,
    const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst, const char * src0_dd_i, const float * src1_ddf_i,
    const char * src1_ddq_i, float * dst_dd_i, const int64_t row_low, const int64_t row_high, const int64_t src1_ncols,
    const int64_t src1_padded_row_size, cudaStream_t stream);

static __global__ void k_compute_batched_ptrs(
        const void * src0_as_f16, const void * src1_as_f16, char * dst,
        const void ** ptrs_src, void ** ptrs_dst,
        int64_t ne12, int64_t ne13,
        int64_t ne23,
        size_t  nb02, size_t  nb03,
        size_t  nb12, size_t  nb13,
        size_t  nbd2, size_t  nbd3,
        int64_t r2,   int64_t r3) {
    const int64_t i13 = blockIdx.x * blockDim.x + threadIdx.x;
    const int64_t i12 = blockIdx.y * blockDim.y + threadIdx.y;

    if (i13 >= ne13 || i12 >= ne12) {
        return;
    }

    const int64_t i03 = i13 / r3;
    const int64_t i02 = i12 / r2;

    ptrs_src[0*ne23 + i12 + i13*ne12] = (const char *) src0_as_f16 + i02*nb02 + i03*nb03;
    ptrs_src[1*ne23 + i12 + i13*ne12] = (const char *) src1_as_f16 + i12*nb12 + i13*nb13;
    ptrs_dst[0*ne23 + i12 + i13*ne12] = (      char *)         dst + i12*nbd2 + i13*nbd3;
}

// Type traits for mapping ggml types to CUDA/cuBLAS types
template<ggml_type T>
struct batched_mul_mat_traits;

template<>
struct batched_mul_mat_traits<GGML_TYPE_F32> {
    using cuda_type = float;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_32F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F32;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp32_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp32_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_BF16> {
    using cuda_type = nv_bfloat16;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_32F;
    static inline const cudaDataType_t data_type = CUDA_R_16BF;
    static inline const ggml_type ggml_type_val = GGML_TYPE_BF16;
    static inline const float alpha = 1.0f;
    static inline const float beta = 0.0f;
    static inline const void* get_alpha() { static const float val = alpha; return &val; }
    static inline const void* get_beta() { static const float val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_bf16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_bf16_nc_cuda(src_type); }
};

template<>
struct batched_mul_mat_traits<GGML_TYPE_F16> {
    using cuda_type = half;
    static inline const cublasComputeType_t compute_type = CUBLAS_COMPUTE_16F;
    static inline const cudaDataType_t data_type = CUDA_R_16F;
    static inline const ggml_type ggml_type_val = GGML_TYPE_F16;
    static inline const half alpha = 1.0;
    static inline const half beta = 0.0;
    static inline const void* get_alpha() { static const half val = alpha; return &val; }
    static inline const void* get_beta() { static const half val = beta; return &val; }
    static inline auto convert(ggml_type src_type) { return ggml_get_to_fp16_cuda(src_type); }
    static inline auto convert_nc(ggml_type src_type) { return ggml_get_to_fp16_nc_cuda(src_type); }
};

// BF16 copies of F32 activations for the BF16 GEMMs that read them (Gemma 3n/4's per-layer input gate and projection,
// whose weights a checkpoint keeps in BF16): the producer (the norm with its weight and residual, an elementwise chain)
// writes the copy as it writes the F32 rows, and ggml_cuda_mul_mat_cublas_impl takes it in place of its own conversion.
// Both round with __float2bfloat16, so the GEMM reads the same bytes. vLLM's activations are BF16 throughout. Two slots
// in turn: a copy is read by the GEMM right after its producer. GGML_CUDA_FUSE_OFF bit 57
static constexpr uint64_t GGML_CUDA_FUSE_BF16_SHADOW = 1ull << 57;

struct ggml_cuda_bf16_shadows {
    nv_bfloat16 * buf  = nullptr;
    size_t        slot = 0; // elements a slot holds
    int           next = 0;
    uint64_t      walk = 0; // graph walks of the context: a copy from an earlier one is not taken
    struct entry {
        const ggml_tensor * t    = nullptr;
        const void *        data = nullptr;
        int64_t             n    = 0;
        uint64_t            walk = 0;
        nv_bfloat16 *       bf16 = nullptr;
    } e[2];
};
static std::unordered_map<const ggml_backend_cuda_context *, ggml_cuda_bf16_shadows> g_bf16_shadows;

// the copy a producer of this walk wrote of src1, or nullptr
static const nv_bfloat16 * ggml_cuda_bf16_shadow_take(const ggml_backend_cuda_context & ctx, const ggml_tensor * src1) {
    const auto it = g_bf16_shadows.find(&ctx);
    if (it == g_bf16_shadows.end()) {
        return nullptr;
    }
    for (const auto & e : it->second.e) {
        if (e.t == src1 && e.data == src1->data && e.n == ggml_nelements(src1) && e.walk == it->second.walk && e.bf16 != nullptr) {
            return e.bf16;
        }
    }
    return nullptr;
}

template<ggml_type compute_type>
static void ggml_cuda_mul_mat_cublas_impl(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    using traits = batched_mul_mat_traits<compute_type>;
    using cuda_t = typename traits::cuda_type;

    GGML_ASSERT(ggml_is_contiguous(dst));

    // Byte offsets and tensor dimensions are currently used in an inconsistent way for dst.
    // As long as dst is contiguous this does not matter though.

    GGML_TENSOR_BINARY_OP_LOCALS

    const int64_t ne_dst = ggml_nelements(dst);
    cudaStream_t main_stream = ctx.stream();
    cublasHandle_t cublas_h = ctx.cublas_handle();

    const size_t src0_ts = ggml_type_size(src0->type);
    GGML_ASSERT(nb00 == src0_ts);
    int64_t s01 = nb01 / src0_ts;
    int64_t s02 = nb02 / src0_ts;
    int64_t s03 = nb03 / src0_ts;

    const size_t src1_ts = ggml_type_size(src1->type);
    GGML_ASSERT(nb10 == src1_ts);
    int64_t s11 = nb11 / src1_ts;
    int64_t s12 = nb12 / src1_ts;
    int64_t s13 = nb13 / src1_ts;

    float * dst_ddf = (float *) dst->data;

    const cuda_t * src0_ptr = nullptr;
    const cuda_t * src1_ptr = nullptr;

    ggml_cuda_pool_alloc<cuda_t> src0_alloc(ctx.pool());
    ggml_cuda_pool_alloc<cuda_t> src1_alloc(ctx.pool());

    bool is_src0_cont_2 = ggml_is_contiguous_2(src0);
    bool is_src1_cont_2 = ggml_is_contiguous_2(src1);

    if (src0->type == compute_type) {
        src0_ptr = (const cuda_t *) src0->data;
    } else {
        src0_alloc.alloc(ggml_nelements(src0));

        if (ggml_is_contiguously_allocated(src0)) {
            const auto convert_func = traits::convert(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ggml_nelements(src0), main_stream);
            const size_t src0_bs = ggml_blck_size(src0->type);
            s01 *= src0_bs;
            s02 *= src0_bs;
            s03 *= src0_bs;
        } else {
            const auto convert_func = traits::convert_nc(src0->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src0->data, src0_alloc.get(), ne00, ne01, ne02, ne03, s01, s02, s03, main_stream);
            s01 = ne00;
            s02 = ne01*s01;
            s03 = ne02*s02;
            is_src0_cont_2 = true;
        }
        src0_ptr = src0_alloc.get();
    }

    const nv_bfloat16 * src1_shadow = nullptr;
    if constexpr (compute_type == GGML_TYPE_BF16) {
        if (src1->type == GGML_TYPE_F32 && ggml_is_contiguous(src1)) {
            src1_shadow = ggml_cuda_bf16_shadow_take(ctx, src1);
        }
    }
    if (src1->type == compute_type) {
        src1_ptr = (const cuda_t *) src1->data;
    } else if (src1_shadow != nullptr) {
        src1_ptr = (const cuda_t *) src1_shadow;
    } else {
        src1_alloc.alloc(ggml_nelements(src1));

        if (ggml_is_contiguously_allocated(src1)) {
            const auto convert_func = traits::convert(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ggml_nelements(src1), main_stream);
            const size_t src1_bs = ggml_blck_size(src1->type);
            s11 *= src1_bs;
            s12 *= src1_bs;
            s13 *= src1_bs;
        } else {
            const auto convert_func = traits::convert_nc(src1->type);
            GGML_ASSERT(convert_func != nullptr);
            convert_func(src1->data, src1_alloc.get(), ne10, ne11, ne12, ne13, s11, s12, s13, main_stream);
            s11 = ne10;
            s12 = ne11*s11;
            s13 = ne12*s12;
            is_src1_cont_2 = true;
        }
        src1_ptr = src1_alloc.get();
    }

    ggml_cuda_pool_alloc<cuda_t> dst_temp(ctx.pool());
    char * dst_ptr;
    size_t nbd2 = dst->nb[2];
    size_t nbd3 = dst->nb[3];

    cublasComputeType_t cu_compute_type = traits::compute_type;
    cudaDataType_t cu_data_type = traits::data_type;
    cudaDataType_t cu_data_type_a = traits::data_type;
    cudaDataType_t cu_data_type_b = traits::data_type;
    const void * alpha = traits::get_alpha();
    const void * beta = traits::get_beta();

    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    bool prefer_f32_output = false;
    if (compute_type == GGML_TYPE_F16) {
        prefer_f32_output = cc == GGML_CUDA_CC_VOLTA || GGML_CUDA_CC_IS_RDNA4(cc) || GGML_CUDA_CC_IS_CDNA(cc);
    } else if (compute_type == GGML_TYPE_BF16) {
        prefer_f32_output = !GGML_CUDA_CC_IS_RDNA3(cc) && !GGML_CUDA_CC_IS_CDNA(cc);
    }

    if (prefer_f32_output) {
        dst_ptr = (char *) dst_ddf;
        cu_compute_type = batched_mul_mat_traits<GGML_TYPE_F32>::compute_type;
        cu_data_type = batched_mul_mat_traits<GGML_TYPE_F32>::data_type;
        alpha = batched_mul_mat_traits<GGML_TYPE_F32>::get_alpha();
        beta = batched_mul_mat_traits<GGML_TYPE_F32>::get_beta();
    } else {
        if constexpr (compute_type == GGML_TYPE_F32) {
            dst_ptr = (char *) dst_ddf;  // Direct F32 output
        } else {
            dst_ptr = (char *) dst_temp.alloc(ne_dst);
            nbd2 /= sizeof(float) / sizeof(cuda_t);
            nbd3 /= sizeof(float) / sizeof(cuda_t);
        }
    }

    GGML_ASSERT(ne12 % ne02 == 0);
    GGML_ASSERT(ne13 % ne03 == 0);

    // broadcast factors
    const int64_t r2 = ne12/ne02;
    const int64_t r3 = ne13/ne03;

    // Theoretically cublasGemmStridedBatchedEx would always work, even for a single matrix.
    // However, for some old NVIDIA and AMD GPUs the strided/Ex GEMM is much slower,
    //     probably because the internal kernel selection logic is suboptimal.
    if (compute_type == GGML_TYPE_F32 && ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasSgemm(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    (const float *) alpha, (const float *) src0_ptr, s01,
                                           (const float *) src1_ptr, s11,
                    (const float *) beta,  (float       *)  dst_ptr, ne0));
    } else if (ne12 == 1 && ne13 == 1) {
        CUBLAS_CHECK(
            cublasGemmEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                    ne01, ne11, ne10,
                    alpha, src0_ptr, cu_data_type_a, s01,
                           src1_ptr, cu_data_type_b, s11,
                    beta,   dst_ptr, cu_data_type,   ne0,
                    cu_compute_type,
                    CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else if (r2 == 1 && r3 == 1 && is_src0_cont_2 && is_src1_cont_2) {
        // with a [0, 2, 1, 3] perm. and ne02==1 the matrix strides need to be determined from dim 3:
        const int64_t sma = ne02 == 1 ? s03 : s02;
        const int64_t smb = ne12 == 1 ? s13 : s12;

        // there is no broadcast and src0, src1 are contiguous across dims 2, 3
        // use cublasGemmStridedBatchedEx
        CUBLAS_CHECK(
        cublasGemmStridedBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, src0_ptr, cu_data_type_a, s01, sma,     // strideA
                       src1_ptr, cu_data_type_b, s11, smb,     // strideB
                beta,   dst_ptr, cu_data_type,   ne0, ne1*ne0, // strideC
                ne12*ne13,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    } else {
        // use cublasGemmBatchedEx
        const int64_t ne23 = ne12*ne13;

        ggml_cuda_pool_alloc<const void *> ptrs_src(ctx.pool(), 2*ne23);
        ggml_cuda_pool_alloc<      void *> ptrs_dst(ctx.pool(), 1*ne23);

        const size_t src_type_size = sizeof(cuda_t);

        const int threads_x = 16;
        const int threads_y = 16;
        const dim3 block_dims(threads_x, threads_y);

        const dim3 grid_dims(
            (ne13 + threads_x - 1) / threads_x,
            (ne12 + threads_y - 1) / threads_y
        );
        k_compute_batched_ptrs<<<grid_dims, block_dims, 0, main_stream>>>(
                src0_ptr, src1_ptr, dst_ptr,
                ptrs_src.get(), ptrs_dst.get(),
                ne12, ne13,
                ne23,
                s02*src_type_size, s03*src_type_size,
                s12*src_type_size, s13*src_type_size,
                nbd2, nbd3,
                r2, r3);

        CUDA_CHECK(cudaGetLastError());

        CUBLAS_CHECK(
        cublasGemmBatchedEx(cublas_h, CUBLAS_OP_T, CUBLAS_OP_N,
                ne01, ne11, ne10,
                alpha, (const void **) (ptrs_src.get() + 0*ne23), cu_data_type_a, s01,
                       (const void **) (ptrs_src.get() + 1*ne23), cu_data_type_b, s11,
                beta,  (      void **) (ptrs_dst.get() + 0*ne23), cu_data_type,   ne0,
                ne23,
                cu_compute_type,
                CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Convert output back to F32 if needed
    if (cu_data_type != CUDA_R_32F) {
        const to_fp32_cuda_t to_fp32_cuda = ggml_get_to_fp32_cuda(traits::ggml_type_val);
        to_fp32_cuda(dst_temp.get(), dst_ddf, ne_dst, main_stream);
    }
}

static void ggml_cuda_mul_mat_cublas(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    ggml_type compute_type = src0->type;
    if (ggml_is_quantized(compute_type)) {
        compute_type = fast_fp16_hardware_available(cc) ? GGML_TYPE_F16 : GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc)) {
        compute_type = GGML_TYPE_F32;
    } else if (compute_type == GGML_TYPE_BF16 && !fast_bf16_hardware_available(cc)) {
        if (GGML_CUDA_CC_IS_AMD(cc) && src1->ne[1] > 32) {
            compute_type = GGML_TYPE_F32;
        }
        if (GGML_CUDA_CC_IS_NVIDIA(cc) && src1->ne[1] > (cc >= GGML_CUDA_CC_VOLTA ? 8 : 128)) {
            compute_type = GGML_TYPE_F32;
        }
    }
    if (dst->op_params[0] == GGML_PREC_F32) {
        compute_type = GGML_TYPE_F32;
    }

    const char * env_c = getenv("GGML_CUDA_CUBLAS_COMPUTE_TYPE");
    if (env_c != nullptr) {
        std::string env_cpp = env_c;
        for (char & c : env_cpp) {
            c = std::tolower(c);
        }
        if (env_cpp == "f32" || env_cpp == "fp32") {
            compute_type = GGML_TYPE_F32;
        } else if (env_cpp == "f16" || env_cpp == "fp16") {
            compute_type = GGML_TYPE_F16;
        } else if (env_cpp == "bf16") {
            compute_type = GGML_TYPE_BF16;
        } else if (env_cpp != "auto") {
            GGML_LOG_WARN("%s: unknown value for GGML_CUDA_CUBLAS_COMPUTE_TYPE: %s", __func__, env_cpp.c_str());
        }
    }

    switch (compute_type) {
        case GGML_TYPE_F32:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F32>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_BF16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_BF16>(ctx, src0, src1, dst);
            break;
        case GGML_TYPE_F16:
            ggml_cuda_mul_mat_cublas_impl<GGML_TYPE_F16>(ctx, src0, src1, dst);
            break;
        default:
            GGML_ABORT("fatal error");
    }
}

// the GLU ops the fused gate kernels apply, at the parameters they use
static bool ggml_cuda_fused_glu_ok(const ggml_tensor * glu) {
    switch (ggml_get_glu_op(glu)) {
        case GGML_GLU_OP_SWIGLU:
        case GGML_GLU_OP_GEGLU:
        case GGML_GLU_OP_SWIGLU_CLAMP: // the fused kernels read its limit from glu_limit
            return true;
        case GGML_GLU_OP_SWIGLU_OAI:
            return ggml_get_op_params_f32(glu, 2) == 1.702f && ggml_get_op_params_f32(glu, 3) == 7.0f;
        default:
            return false;
    }
}

static bool ggml_cuda_should_fuse_mul_mat(const ggml_tensor * ffn_up,
                                          const ggml_tensor * ffn_gate,
                                          const ggml_tensor * glu,
                                          const ggml_tensor * ffn_up_bias = nullptr,
                                          const ggml_tensor * ffn_gate_bias = nullptr,
                                          const ggml_tensor * ffn_up_scale = nullptr,
                                          const ggml_tensor * ffn_gate_scale = nullptr) {
    const bool has_bias = ffn_up_bias != nullptr || ffn_gate_bias != nullptr;
    const bool has_scale = ffn_up_scale != nullptr || ffn_gate_scale != nullptr;

    if (has_bias && (!ffn_up_bias || !ffn_gate_bias)) {
        return false;
    }
    if (has_scale && (!ffn_up_scale || !ffn_gate_scale)) {
        return false;
    }

    const bool is_mul_mat     = ffn_up->op == GGML_OP_MUL_MAT     && ffn_gate->op == GGML_OP_MUL_MAT     && glu->op == GGML_OP_GLU;
    const bool is_mul_mat_id  = ffn_up->op == GGML_OP_MUL_MAT_ID  && ffn_gate->op == GGML_OP_MUL_MAT_ID  && glu->op == GGML_OP_GLU;

    GGML_ASSERT(ffn_up && ffn_gate && glu);

    if (!is_mul_mat && !is_mul_mat_id) {
        return false;
    }

    const ggml_op expected_bias_op = is_mul_mat ? GGML_OP_ADD : GGML_OP_ADD_ID;
    const ggml_tensor * ffn_up_bias_src   = has_scale ? ffn_up_scale   : ffn_up;
    const ggml_tensor * ffn_gate_bias_src = has_scale ? ffn_gate_scale : ffn_gate;
    const ggml_tensor * ffn_up_out        = has_bias ? ffn_up_bias     : ffn_up_bias_src;
    const ggml_tensor * ffn_gate_out      = has_bias ? ffn_gate_bias   : ffn_gate_bias_src;

    if (glu->src[0] != ffn_gate_out || glu->src[1] != ffn_up_out) {
        return false;
    }

    if (has_scale) {
        if (ffn_up_scale->op != GGML_OP_MUL || ffn_gate_scale->op != GGML_OP_MUL) {
            return false;
        }
        const bool up_has_mm   = ffn_up_scale->src[0] == ffn_up || ffn_up_scale->src[1] == ffn_up;
        const bool gate_has_mm = ffn_gate_scale->src[0] == ffn_gate || ffn_gate_scale->src[1] == ffn_gate;
        if (!up_has_mm || !gate_has_mm) {
            return false;
        }
    }

    if (has_bias) {
        if (ffn_up_bias->op != expected_bias_op || ffn_gate_bias->op != expected_bias_op) {
            return false;
        }

        if (expected_bias_op == GGML_OP_ADD) {
            const bool up_has_mul   = ffn_up_bias->src[0] == ffn_up_bias_src || ffn_up_bias->src[1] == ffn_up_bias_src;
            const bool gate_has_mul = ffn_gate_bias->src[0] == ffn_gate_bias_src || ffn_gate_bias->src[1] == ffn_gate_bias_src;
            if (!up_has_mul || !gate_has_mul) {
                return false;
            }
        } else { // GGML_OP_ADD_ID
            if (ffn_up_bias->src[0] != ffn_up_bias_src || ffn_gate_bias->src[0] != ffn_gate_bias_src) {
                return false;
            }
            if (ffn_up_bias->src[2] != ffn_up->src[2] || ffn_gate_bias->src[2] != ffn_gate->src[2]) {
                return false;
            }
        }
    }

    if (ffn_up->src[0]->type != ffn_gate->src[0]->type || !ggml_are_same_shape(ffn_up->src[0], ffn_gate->src[0]) ||
        !ggml_are_same_stride(ffn_up->src[0], ffn_gate->src[0])) {
        return false;
    }

    if (ffn_up->src[1] != ffn_gate->src[1]) {
        return false;
    }

    if (is_mul_mat_id && ffn_up->src[2] != ffn_gate->src[2]) {
        return false;
    }

    if (!ggml_cuda_fused_glu_ok(glu)) {
        return false;
    }

    if (const bool swapped = ggml_get_op_params_i32(glu, 1); swapped) {
        return false;
    }

    return true;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_f(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    const bool is_mul_mat_id = tensor->op == GGML_OP_MUL_MAT_ID;

    bool use_mul_mat_vec_f =
        (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
        src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32;

    const int cc      = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    use_mul_mat_vec_f = use_mul_mat_vec_f && ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, is_mul_mat_id ? src1->ne[2] : src1->ne[1]);

    //we only support fusion for ncols_dst = 1
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] != 1) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] != 1) {
        return false;
    }


    return use_mul_mat_vec_f;
}

static bool ggml_cuda_should_fuse_mul_mat_vec_q(const ggml_tensor * tensor) {
    ggml_tensor *       src0 = tensor->src[0];
    ggml_tensor *       src1 = tensor->src[1];
    const ggml_tensor * dst  = tensor;

    if (src0->buffer == nullptr || src1->buffer == nullptr) {
        return false;
    }
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) &&
                                   src0->view_src;

    bool use_mul_mat_vec_q = ggml_is_quantized(src0->type) && !bad_padding_clear && src1->type == GGML_TYPE_F32 &&
                             dst->type == GGML_TYPE_F32 && src1->ne[1] <= MMVQ_MAX_BATCH_SIZE;

    // fusion is not universally faster on Pascal
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    if (cc <= GGML_CUDA_CC_PASCAL) {
        return false;
    }
    // the vector kernel carries the fusion for every batch it serves; the
    // expert kernel for several tokens does not
    if (tensor->op == GGML_OP_MUL_MAT && dst->ne[1] > MMVQ_MAX_BATCH_SIZE) {
        return false;
    }

    if (tensor->op == GGML_OP_MUL_MAT_ID && dst->ne[2] > get_mmvq_mmid_max_batch(src0->type, cc)) {
        return false;
    }

    return use_mul_mat_vec_q;
}

static bool ggml_cuda_tensors_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 && b0 && a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

static ggml_cuda_norm_fold * ggml_cuda_norm_fold_for(ggml_backend_cuda_context & ctx, const ggml_tensor * act) {
    if (ctx.norm_folds.empty()) {
        return nullptr;
    }
    const auto it = ctx.norm_folds.find(act);
    return it == ctx.norm_folds.end() || it->second->materialized ? nullptr : it->second.get();
}

// GGML_CUDA_FOLD_CHECK=1 (graphs off): after a fold's factors exist, the reference
// norm is run into a scratch buffer and compared bit for bit with scale*x*w
static void ggml_cuda_norm_fold_check(ggml_backend_cuda_context & ctx, ggml_cuda_norm_fold & fold) {
    static const bool on = getenv("GGML_CUDA_FOLD_CHECK") != nullptr;
    if (!on || !fold.scale_ready || fold.norm->op != GGML_OP_RMS_NORM) {
        return;
    }
    const int64_t n = ggml_nelements(fold.mul);
    ggml_cuda_pool_alloc<float> ours(ctx.pool(), n);
    ggml_cuda_pool_alloc<float> ref(ctx.pool(), n);
    rms_apply_cuda((const float *) fold.x->data, (const float *) fold.w->data, fold.scale, ours.get(), fold.mul->ne[0], fold.n_rows, ctx.stream());
    ggml_cuda_op_rms_norm_fused(ctx, fold.norm, fold.mul, ref.get());
    std::vector<float> a(n), b(n), s(fold.n_rows);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    CUDA_CHECK(cudaMemcpy(a.data(), ours.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(b.data(), ref.get(), n * sizeof(float), cudaMemcpyDeviceToHost));
    CUDA_CHECK(cudaMemcpy(s.data(), fold.scale, fold.n_rows * sizeof(float), cudaMemcpyDeviceToHost));
    int64_t bad = 0, first = -1;
    for (int64_t i = 0; i < n; ++i) {
        if (memcmp(&a[i], &b[i], sizeof(float)) != 0) {
            if (first < 0) { first = i; }
            bad++;
        }
    }
    // the fold's q8 against the plain quantization of the reference product
    int64_t bad_q8 = -1, first_q8 = -1;
    if (fold.q8_ready) {
        const ggml_tensor * m = fold.mul;
        ggml_cuda_pool_alloc<char> q8ref(ctx.pool(), fold.q8_bytes);
        quantize_row_q8_1_cuda(ref.get(), nullptr, q8ref.get(), GGML_TYPE_Q8_0, m->ne[0],
            m->nb[1] / sizeof(float), m->nb[2] / sizeof(float), m->nb[3] / sizeof(float),
            GGML_PAD(m->ne[0], MATRIX_ROW_PADDING), m->ne[1], m->ne[2], m->ne[3], ctx.stream());
        std::vector<char> qa(fold.q8_bytes), qb(fold.q8_bytes);
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        CUDA_CHECK(cudaMemcpy(qa.data(), fold.q8, fold.q8_bytes, cudaMemcpyDeviceToHost));
        CUDA_CHECK(cudaMemcpy(qb.data(), q8ref.get(), fold.q8_bytes, cudaMemcpyDeviceToHost));
        bad_q8 = 0;
        for (size_t i = 0; i < fold.q8_bytes; ++i) {
            if (qa[i] != qb[i]) {
                if (first_q8 < 0) { first_q8 = (int64_t) i; }
                bad_q8++;
            }
        }
    }
    fprintf(stderr, "[fold-check] %s: product %lld of %lld differ%s; q8 %lld bytes differ%s; scale[0]=%.9g\n", fold.norm->name,
            (long long) bad, (long long) n,
            first < 0 ? "" : (" first at " + std::to_string(first) + ": ours " + std::to_string(a[first]) + " ref " + std::to_string(b[first])).c_str(),
            (long long) bad_q8, first_q8 < 0 ? "" : (" first at byte " + std::to_string(first_q8)).c_str(), s[0]);
}

// the check right after a reader, while the norm's input is still live
static void ggml_cuda_norm_fold_check_at(ggml_backend_cuda_context & ctx, const ggml_tensor * act) {
    ggml_cuda_norm_fold * fold = ggml_cuda_norm_fold_for(ctx, act);
    if (fold) {
        ggml_cuda_norm_fold_check(ctx, *fold);
    }
}

// a reader that cannot quantize the norm on the way in gets the product written out, once
static void ggml_cuda_norm_fold_materialize(ggml_backend_cuda_context & ctx, const ggml_tensor * act) {
    ggml_cuda_norm_fold * fold = ggml_cuda_norm_fold_for(ctx, act);
    if (fold) {
        GGML_ASSERT(fold->norm->op == GGML_OP_RMS_NORM && "a gated fold's product is only ever quantized");
        ggml_cuda_op_rms_norm_fused(ctx, fold->norm, fold->mul);
        fold->materialized = true;
        ggml_cuda_region_publish(ctx);
    }
}

// a weight far past L2 (a vocabulary head) under several sequences' rows, or eight (a drafter's step over eight): the
// matvec reads it again for each few rows and reaches half the bandwidth, the FP4 GEMM reads it once
static bool ggml_cuda_fp4_big_weight(const ggml_tensor * src0, const ggml_tensor * src1, const ggml_tensor * dst, const int cc) {
    return src1->ne[1] >= 2 && ggml_nbytes(src0) >= (size_t) 128 << 20 && (ggml_mul_mat_get_n_seqs(dst) > 1 || src1->ne[1] >= 8) &&
        ggml_cuda_fp4_gemm_supported(src0, src1, dst, cc);
}

static void ggml_cuda_mul_mat(ggml_backend_cuda_context & ctx, const ggml_tensor * src0, const ggml_tensor * src1, ggml_tensor * dst) {
    GGML_TENSOR_BINARY_OP_LOCALS

    const int32_t hint = ggml_get_op_params_i32(dst, 1);
    if (hint == GGML_HINT_SRC0_IS_HADAMARD) {
        ggml_cuda_norm_fold_materialize(ctx, src1);
        if (ggml_cuda_op_fwht(ctx, src1, dst)) {
            return;
        }
    }

    // FP8 weights have kernels of their own (fp4gemm.cu); supports_op takes no other shapes of them
    if (src0->type == GGML_TYPE_F8_E4M3) {
        ggml_cuda_norm_fold_materialize(ctx, src1);
        GGML_ASSERT(ggml_cuda_fp8_mul_mat(ctx, src0, src1, dst, nullptr, nullptr));
        return;
    }

    // If src0 is a temporary compute buffer it may have some padding that needs to be cleared for mul_mat_vec_q or mul_mat_q.
    // But if src0 is also a view of another tensor then this cannot be done safely because it may overwrite valid tensor data.
    // Therefore, in such cases use cuBLAS.
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE
        && ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        ggml_cuda_norm_fold_materialize(ctx, src1);
        ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
        return;
    }

    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    if (ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11)) {
        // The custom F16 vector kernel can be used over batched cuBLAS GEMM.
        // But this is only faster for GPUs without tensor cores or with a thin src0 matrix (particularly KQV in attention)
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, dst, nullptr, ggml_cuda_norm_fold_for(ctx, src1)); ggml_cuda_norm_fold_check_at(ctx, src1);
        return;
    }
    // A transposed vector can still use MMVQ (i.e. ne01 == 1)
    if (ne01 == 1 && ne11 > MMVF_MAX_BATCH_SIZE && ne2 == 1 && ne3 == 1
            && src0->type == GGML_TYPE_F32
            && ggml_is_contiguous(src0) && ggml_is_contiguous(src1) && ggml_is_contiguous(dst)
            && ggml_cuda_should_use_mmvf(src1->type, cc, src1->ne, src1->nb, /*ne11 =*/ 1)) {
        ggml_tensor dst_vec = *dst;
        dst_vec.ne[0] = ne11;
        dst_vec.ne[1] = 1;
        dst_vec.nb[1] = dst_vec.nb[0]*ne11;
        dst_vec.nb[2] = dst_vec.nb[1];
        dst_vec.nb[3] = dst_vec.nb[1];
        ggml_cuda_norm_fold_materialize(ctx, src1);
        ggml_cuda_mul_mat_vec_f(ctx, src1, src0, nullptr, &dst_vec);
        return;
    }
    if (ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false)) {
        ggml_cuda_norm_fold_materialize(ctx, src1);
        ggml_cuda_mul_mat_f(ctx, src0, src1, nullptr, dst);
        return;
    }
    if (ggml_cuda_fp4_big_weight(src0, src1, dst, cc)) {
        ggml_cuda_norm_fold_materialize(ctx, src1);
        if (ggml_cuda_fp4_gemm(ctx, src0, src1, dst)) {
            return;
        }
    }
    if (ggml_cuda_should_use_mmvq(src0->type, cc, ne11)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, dst, nullptr, ggml_cuda_norm_fold_for(ctx, src1)); ggml_cuda_norm_fold_check_at(ctx, src1);
        return;
    }
    ggml_cuda_norm_fold_materialize(ctx, src1);
    // a small weight over several sequences' rows: its F32 copy through a TF32 GEMM (ggml_cuda_small_gemm)
    if (ggml_cuda_small_gemm_applies(src0, src1, dst) && ggml_cuda_small_gemm(ctx, src0, src1, dst)) {
        return;
    }
    // a step over several sequences: cuBLASLt's block-scaled FP4 GEMM for FP4 weights; one sequence keeps its kernels
    if ((ggml_mul_mat_get_n_seqs(dst) > 1 || ggml_cuda_fp4_gemm_forced()) && ggml_cuda_fp4_gemm_supported(src0, src1, dst, cc) &&
            ggml_cuda_fp4_gemm(ctx, src0, src1, dst)) {
        return;
    }
    if (ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        ggml_cuda_mul_mat_q(ctx, src0, src1, nullptr, dst);
        return;
    }
    ggml_cuda_mul_mat_cublas(ctx, src0, src1, dst);
}

// returns true when ggml_cuda_mul_mat_id takes the fallback path that requires stream synchronization
// [TAG_MUL_MAT_ID_CUDA_GRAPHS]
static bool ggml_cuda_mul_mat_id_needs_sync(const ggml_tensor * dst, const int cc) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];

    if (src1->type != GGML_TYPE_F32 || dst->type != GGML_TYPE_F32) {
        return true;
    }

    if (dst->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        if (ggml_is_quantized(src0->type)) {
            if (dst->ne[2] <= get_mmvq_mmid_max_batch(src0->type, cc)) {
                return false;
            }
        } else if (GGML_CUDA_CC_IS_AMD(cc)) {
            return false;
        }
    }

    if (ggml_cuda_should_use_mmq(src0->type, cc, src1->ne[2], /*n_experts=*/src0->ne[2])) {
        return false;
    }

    if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
        return false;
    }

    return true;
}

static void ggml_cuda_mul_mat_id(ggml_backend_cuda_context & ctx, ggml_tensor * dst) {
    const ggml_tensor * src0 = dst->src[0];
    const ggml_tensor * src1 = dst->src[1];
    const ggml_tensor * ids  = dst->src[2];

    GGML_ASSERT(src1->type == GGML_TYPE_F32);
    GGML_ASSERT(dst->type  == GGML_TYPE_F32);

    GGML_TENSOR_BINARY_OP_LOCALS

    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;

    // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
    if (src1->type == GGML_TYPE_F32 && dst->type == GGML_TYPE_F32) {
        static_assert(MMVQ_MAX_BATCH_SIZE == MMVF_MAX_BATCH_SIZE);
        if (ne2 <= MMVQ_MAX_BATCH_SIZE) {
            if (ggml_is_quantized(src0->type)) {
                const int mmvq_mmid_max = get_mmvq_mmid_max_batch(src0->type, cc);
                if (ne2 <= mmvq_mmid_max) {
                    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, dst, nullptr, ggml_cuda_norm_fold_for(ctx, src1)); ggml_cuda_norm_fold_check_at(ctx, src1);
                    return;
                }
            } else {
                if (GGML_CUDA_CC_IS_AMD(cc)) {
                    ggml_cuda_norm_fold_materialize(ctx, src1);
                    ggml_cuda_mul_mat_vec_f(ctx, src0, src1, ids, dst);
                    return;
                }
            }
        }

        ggml_cuda_norm_fold_materialize(ctx, src1);
        if (ggml_cuda_should_use_mmq(src0->type, cc, ne12, /*n_experts=*/ne02)) {
            ggml_cuda_mul_mat_q(ctx, src0, src1, ids, dst);
            return;
        }

        if (ggml_cuda_should_use_mmf(src0->type, cc, WARP_SIZE, src0->ne, src0->nb, src1->ne[2], /*mul_mat_id=*/true)) {
            ggml_cuda_mul_mat_f(ctx, src0, src1, ids, dst);
            return;
        }
    }

    // note: this path should not be reached when recording CUDA graphs, because it requires stream synchronization
    GGML_ASSERT(ggml_cuda_mul_mat_id_needs_sync(dst, cc));
    ggml_cuda_norm_fold_materialize(ctx, src1);
    cudaStream_t stream = ctx.stream();

    GGML_ASSERT(nb12 % nb11 == 0);
    GGML_ASSERT(nb2  % nb1  == 0);

    const ggml_type type_src1_sorted = (src0->type == GGML_TYPE_F16 && !fast_fp16_hardware_available(cc))
        || ggml_is_quantized(src0->type) ? GGML_TYPE_F32 : src0->type;
    const ggml_type type_dst_sorted  = GGML_TYPE_F32;
    const size_t ts_src1_sorted = ggml_type_size(type_src1_sorted);
    const size_t ts_dst_sorted  = ggml_type_size(type_dst_sorted);

    const int64_t n_expert_used = ids->ne[0];
    const int64_t ne_get_rows = ne12 * n_expert_used;

    std::vector<int32_t> ids_to_sorted_host;
    ids_to_sorted_host.reserve(2*ne_get_rows);
    std::vector<int32_t> ids_from_sorted_host(ne_get_rows);

    ggml_cuda_pool_alloc<int32_t> ids_buf_dev(ctx.pool(), 2*ne_get_rows);

    std::vector<int32_t> tokens_per_expert(ne02);

    ggml_cuda_pool_alloc<char> src1_sorted(ctx.pool(), ne12*n_expert_used*ne10*ts_src1_sorted);
    ggml_cuda_pool_alloc<char>  dst_sorted(ctx.pool(), ne2 *n_expert_used* ne0*ts_dst_sorted);

    std::vector<char> ids_host(ggml_nbytes(ids));
    CUDA_CHECK(cudaMemcpyAsync(ids_host.data(), ids->data, ggml_nbytes(ids), cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    for (int64_t i02 = 0; i02 < ne02; ++i02) { // expert matrices
        for (int64_t i12 = 0; i12 < ne12; ++i12) { // tokens
            for (int64_t iex = 0; iex < n_expert_used; ++iex) {
                const int32_t expert_to_use = *(const int32_t *)(ids_host.data() + i12*ids->nb[1] + iex*ids->nb[0]);
                assert(expert_to_use >= 0 && expert_to_use < ne02);
                if (expert_to_use == i02) {
                    ids_from_sorted_host[i12*n_expert_used + iex] = ids_to_sorted_host.size();
                    ids_to_sorted_host.push_back(i12*ne11 + iex % ne11);
                    tokens_per_expert[i02]++;
                    break;
                }
            }
        }
    }
    GGML_ASSERT(ids_to_sorted_host.size() == size_t(ne_get_rows));

    ids_to_sorted_host.insert(ids_to_sorted_host.end(), ids_from_sorted_host.begin(), ids_from_sorted_host.end());

    CUDA_CHECK(cudaMemcpyAsync(ids_buf_dev.ptr, ids_to_sorted_host.data(), 2*ne_get_rows*sizeof(int32_t), cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));

    const int32_t * ids_to_sorted   = ids_buf_dev.ptr + 0*ne_get_rows;
    const int32_t * ids_from_sorted = ids_buf_dev.ptr + 1*ne_get_rows;

    get_rows_cuda(src1->data, src1->type, ids_to_sorted, src1_sorted.ptr, type_src1_sorted,
        ne10, nb11, nb12, nb13,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, ne_get_rows*ne10*ts_src1_sorted, stream);
    CUDA_CHECK(cudaGetLastError());

    char * src1_data_cur = (char *) src1_sorted.ptr;
    char *  dst_data_cur = (char *)  dst_sorted.ptr;
    for (int64_t i02 = 0; i02 < ne02; ++i02) {
        if (tokens_per_expert[i02] == 0) {
            continue;
        }

        ggml_tensor src0_slice = *src0;
        src0_slice.ne[2]    = 1;
        src0_slice.nb[3]    = src0_slice.nb[2];
        src0_slice.op       = GGML_OP_VIEW;
        src0_slice.view_src = dst->src[0]; // non-const pointer to src0
        src0_slice.data     = (char *) src0->data + i02*nb02;

        ggml_tensor src1_slice;
        memset(&src1_slice, 0, sizeof(src1_slice));
        src1_slice.buffer = src1->buffer;
        src1_slice.type   = type_src1_sorted;
        src1_slice.ne[0]  = ne10;
        src1_slice.ne[1]  = tokens_per_expert[i02];
        src1_slice.ne[2]  = 1;
        src1_slice.ne[3]  = 1;
        src1_slice.nb[0]  = ts_src1_sorted;
        src1_slice.nb[1]  = src1_slice.ne[0] * src1_slice.nb[0];
        src1_slice.nb[2]  = src1_slice.ne[1] * src1_slice.nb[1];
        src1_slice.nb[3]  = src1_slice.ne[2] * src1_slice.nb[2];
        src1_slice.data   = src1_data_cur;

        ggml_tensor dst_slice;
        memset(&dst_slice, 0, sizeof(dst_slice));
        dst_slice.buffer = dst->buffer;
        dst_slice.type   = type_dst_sorted;
        dst_slice.ne[0]  = ne0;
        dst_slice.ne[1]  = tokens_per_expert[i02];
        dst_slice.ne[2]  = 1;
        dst_slice.ne[3]  = 1;
        dst_slice.nb[0]  = ts_dst_sorted;
        dst_slice.nb[1]  = dst_slice.ne[0] * dst_slice.nb[0];
        dst_slice.nb[2]  = dst_slice.ne[1] * dst_slice.nb[1];
        dst_slice.nb[3]  = dst_slice.ne[2] * dst_slice.nb[2];
        dst_slice.data   = dst_data_cur;

        ggml_cuda_mul_mat(ctx, &src0_slice, &src1_slice, &dst_slice);
        CUDA_CHECK(cudaGetLastError());

        src1_data_cur += src1_slice.nb[2];
        dst_data_cur  +=  dst_slice.nb[2];
    }

    get_rows_cuda(dst_sorted.ptr, type_dst_sorted, ids_from_sorted, dst->data, dst->type,
        ne0, ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted, ne_get_rows*ne0*ts_dst_sorted,
        ne_get_rows, 1, 1, sizeof(int32_t), ne_get_rows*sizeof(int32_t), ne_get_rows*sizeof(int32_t),
        nb1, nb2, nb3, stream);
}

static bool ggml_cuda_compute_forward(ggml_backend_cuda_context & ctx, struct ggml_tensor * dst) {
    switch (dst->op) {
        case GGML_OP_ARGMAX:
            ggml_cuda_argmax(ctx, dst);
            break;
        case GGML_OP_COUNT_EQUAL:
            ggml_cuda_count_equal(ctx, dst);
            break;
        case GGML_OP_REPEAT:
            ggml_cuda_op_repeat(ctx, dst);
            break;
        case GGML_OP_REPEAT_BACK:
            ggml_cuda_op_repeat_back(ctx, dst);
            break;
        case GGML_OP_GET_ROWS:
            ggml_cuda_op_get_rows(ctx, dst);
            break;
        case GGML_OP_GET_ROWS_BACK:
            ggml_cuda_op_get_rows_back(ctx, dst);
            break;
        case GGML_OP_SET_ROWS:
            ggml_cuda_op_set_rows(ctx, dst);
            break;
        case GGML_OP_SET:
            ggml_cuda_op_set(ctx, dst);
            break;
        case GGML_OP_DUP:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_CPY:
            ggml_cuda_cpy(ctx, dst->src[0], dst->src[1]);
            break;
        case GGML_OP_CONT:
            ggml_cuda_dup(ctx, dst);
            break;
        case GGML_OP_ADD:
        case GGML_OP_ADD1: // TODO: more efficient implementation
            ggml_cuda_op_add(ctx, dst);
            break;
        case GGML_OP_ADD_ID:
            ggml_cuda_op_add_id(ctx, dst);
            break;
        case GGML_OP_SUB:
            ggml_cuda_op_sub(ctx, dst);
            break;
        case GGML_OP_ACC:
            ggml_cuda_op_acc(ctx, dst);
            break;
        case GGML_OP_MUL:
            ggml_cuda_op_mul(ctx, dst);
            break;
        case GGML_OP_DIV:
            ggml_cuda_op_div(ctx, dst);
            break;
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(dst)) {
                case GGML_UNARY_OP_ABS:
                    ggml_cuda_op_abs(ctx, dst);
                    break;
                case GGML_UNARY_OP_SGN:
                    ggml_cuda_op_sgn(ctx, dst);
                    break;
                case GGML_UNARY_OP_NEG:
                    ggml_cuda_op_neg(ctx, dst);
                    break;
                case GGML_UNARY_OP_STEP:
                    ggml_cuda_op_step(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU:
                    ggml_cuda_op_gelu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SILU:
                    ggml_cuda_op_silu(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_ERF:
                    ggml_cuda_op_gelu_erf(ctx, dst);
                    break;
                case GGML_UNARY_OP_GELU_QUICK:
                    ggml_cuda_op_gelu_quick(ctx, dst);
                    break;
                case GGML_UNARY_OP_TANH:
                    ggml_cuda_op_tanh(ctx, dst);
                    break;
                case GGML_UNARY_OP_RELU:
                    ggml_cuda_op_relu(ctx, dst);
                    break;
                case GGML_UNARY_OP_SIGMOID:
                    ggml_cuda_op_sigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSIGMOID:
                    ggml_cuda_op_hardsigmoid(ctx, dst);
                    break;
                case GGML_UNARY_OP_HARDSWISH:
                    ggml_cuda_op_hardswish(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXP:
                    ggml_cuda_op_exp(ctx, dst);
                    break;
                case GGML_UNARY_OP_ELU:
                    ggml_cuda_op_elu(ctx, dst);
                    break;
                case GGML_UNARY_OP_XIELU:
                    ggml_cuda_op_xielu(ctx, dst);
                    break;
                case GGML_UNARY_OP_FLOOR:
                    ggml_cuda_op_floor(ctx, dst);
                    break;
                case GGML_UNARY_OP_CEIL:
                    ggml_cuda_op_ceil(ctx, dst);
                    break;
                case GGML_UNARY_OP_ROUND:
                    ggml_cuda_op_round(ctx, dst);
                    break;
                case GGML_UNARY_OP_TRUNC:
                    ggml_cuda_op_trunc(ctx, dst);
                    break;
                case GGML_UNARY_OP_EXPM1:
                    ggml_cuda_op_expm1(ctx, dst);
                    break;
                case GGML_UNARY_OP_SOFTPLUS:
                    ggml_cuda_op_softplus(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(dst)) {
                case GGML_GLU_OP_REGLU:
                    ggml_cuda_op_reglu(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU:
                    ggml_cuda_op_geglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU:
                    ggml_cuda_op_swiglu(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_OAI:
                    ggml_cuda_op_swiglu_oai(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_ERF:
                    ggml_cuda_op_geglu_erf(ctx, dst);
                    break;
                case GGML_GLU_OP_GEGLU_QUICK:
                    ggml_cuda_op_geglu_quick(ctx, dst);
                    break;
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    ggml_cuda_op_swiglu_clamp(ctx, dst);
                    break;
                default:
                    return false;
            }
            break;
        case GGML_OP_NORM:
            ggml_cuda_op_norm(ctx, dst);
            break;
        case GGML_OP_GROUP_NORM:
            ggml_cuda_op_group_norm(ctx, dst);
            break;
        case GGML_OP_L2_NORM:
            ggml_cuda_op_l2_norm(ctx, dst);
            break;
        case GGML_OP_CONCAT:
            ggml_cuda_op_concat(ctx, dst);
            break;
        case GGML_OP_UPSCALE:
            ggml_cuda_op_upscale(ctx, dst);
            break;
        case GGML_OP_PAD:
            ggml_cuda_op_pad(ctx, dst);
            break;
        case GGML_OP_PAD_REFLECT_1D:
            ggml_cuda_op_pad_reflect_1d(ctx, dst);
            break;
        case GGML_OP_ARANGE:
            ggml_cuda_op_arange(ctx, dst);
            break;
        case GGML_OP_TIMESTEP_EMBEDDING:
            ggml_cuda_op_timestep_embedding(ctx, dst);
            break;
        case GGML_OP_LEAKY_RELU:
            ggml_cuda_op_leaky_relu(ctx, dst);
            break;
        case GGML_OP_SILU_BACK:
            ggml_cuda_op_silu_back(ctx, dst);
            break;
        case GGML_OP_RMS_NORM:
            ggml_cuda_op_rms_norm(ctx, dst);
            break;
        case GGML_OP_RMS_NORM_BACK:
            ggml_cuda_op_rms_norm_back(ctx, dst);
            break;
        case GGML_OP_MUL_MAT:
            ggml_cuda_mul_mat(ctx, dst->src[0], dst->src[1], dst);
            break;
        case GGML_OP_MUL_MAT_ID:
            ggml_cuda_mul_mat_id(ctx, dst);
            break;
        case GGML_OP_OUT_PROD:
            ggml_cuda_out_prod(ctx, dst);
            break;
        case GGML_OP_SCALE:
            ggml_cuda_op_scale(ctx, dst);
            break;
        case GGML_OP_SQR:
            ggml_cuda_op_sqr(ctx, dst);
            break;
        case GGML_OP_SQRT:
            ggml_cuda_op_sqrt(ctx, dst);
            break;
        case GGML_OP_SIN:
            ggml_cuda_op_sin(ctx, dst);
            break;
        case GGML_OP_COS:
            ggml_cuda_op_cos(ctx, dst);
            break;
        case GGML_OP_CLAMP:
            ggml_cuda_op_clamp(ctx, dst);
            break;
        case GGML_OP_LOG:
            ggml_cuda_op_log(ctx, dst);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
                break;
        case GGML_OP_DIAG:
            ggml_cuda_op_diag(ctx, dst);
            break;
        case GGML_OP_DIAG_MASK_INF:
            ggml_cuda_op_diag_mask_inf(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX:
            ggml_cuda_op_soft_max(ctx, dst);
            break;
        case GGML_OP_SOFT_MAX_BACK:
            ggml_cuda_op_soft_max_back(ctx, dst);
            break;
        case GGML_OP_ROPE:
            ggml_cuda_op_rope(ctx, dst);
            break;
        case GGML_OP_ROPE_BACK:
            ggml_cuda_op_rope_back(ctx, dst);
            break;
        case GGML_OP_ROLL:
            ggml_cuda_op_roll(ctx, dst);
            break;
        case GGML_OP_IM2COL:
            ggml_cuda_op_im2col(ctx, dst);
            break;
        case GGML_OP_IM2COL_3D:
            ggml_cuda_op_im2col_3d(ctx, dst);
            break;
        case GGML_OP_CONV_2D:
            ggml_cuda_op_conv2d(ctx, dst);
            break;
        case GGML_OP_CONV_2D_DW:
            ggml_cuda_op_conv2d_dw(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_2D:
            ggml_cuda_conv_2d_transpose_p0(ctx, dst);
            break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            ggml_cuda_op_conv_transpose_1d(ctx,dst);
            break;
        case GGML_OP_COL2IM_1D:
            ggml_cuda_op_col2im_1d(ctx, dst);
            break;
        case GGML_OP_POOL_2D:
            ggml_cuda_op_pool2d(ctx, dst);
            break;
        case GGML_OP_POOL_1D:
            ggml_cuda_op_pool1d(ctx, dst);
            break;
        case GGML_OP_SUM:
            ggml_cuda_op_sum(ctx, dst);
            break;
        case GGML_OP_CUMSUM:
            ggml_cuda_op_cumsum(ctx, dst);
            break;
        case GGML_OP_SUM_ROWS:
            ggml_cuda_op_sum_rows(ctx, dst);
            break;
        case GGML_OP_MEAN:
            ggml_cuda_op_mean(ctx, dst);
            break;
        case GGML_OP_SSM_CONV:
            ggml_cuda_op_ssm_conv(ctx, dst);
            break;
        case GGML_OP_SSM_SCAN:
            ggml_cuda_op_ssm_scan(ctx, dst);
            break;
        case GGML_OP_TOP_K:
            ggml_cuda_op_top_k(ctx, dst);
            break;
        case GGML_OP_ARGSORT:
            ggml_cuda_op_argsort(ctx, dst);
            break;
        case GGML_OP_FLASH_ATTN_EXT:
            ggml_cuda_flash_attn_ext(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS:
            ggml_cuda_cross_entropy_loss(ctx, dst);
            break;
        case GGML_OP_TRI:
            ggml_cuda_op_tri(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV6:
            ggml_cuda_op_rwkv_wkv6(ctx, dst);
            break;
        case GGML_OP_GATED_LINEAR_ATTN:
            ggml_cuda_op_gated_linear_attn(ctx, dst);
            break;
        case GGML_OP_GATED_DELTA_NET:
            ggml_cuda_op_gated_delta_net(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_COMB:
            ggml_cuda_op_dsv4_hc_comb(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_PRE:
            ggml_cuda_op_dsv4_hc_pre(ctx, dst);
            break;
        case GGML_OP_DSV4_HC_POST:
            ggml_cuda_op_dsv4_hc_post(ctx, dst);
            break;
        case GGML_OP_RWKV_WKV7:
            ggml_cuda_op_rwkv_wkv7(ctx, dst);
            break;
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
            ggml_cuda_cross_entropy_loss_back(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_ADAMW:
            ggml_cuda_opt_step_adamw(ctx, dst);
            break;
        case GGML_OP_OPT_STEP_SGD:
            ggml_cuda_opt_step_sgd(ctx, dst);
            break;
        case GGML_OP_SOLVE_TRI:
            ggml_cuda_op_solve_tri(ctx, dst);
            break;
        case GGML_OP_FILL:
            ggml_cuda_op_fill(ctx, dst);
            break;
        case GGML_OP_LIGHTNING_INDEXER:
            ggml_cuda_lightning_indexer(ctx, dst);
            break;
        default:
            return false;
    }

    cudaError_t err = cudaGetLastError();
    if (err != cudaSuccess) {
        GGML_LOG_ERROR("%s: %s failed\n", __func__, ggml_op_desc(dst));
        CUDA_CHECK(err);
    }

    return true;
}

////////////////////////////////////////////////////////////////////////////////

// backend

static const char * ggml_backend_cuda_get_name(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    return cuda_ctx->name.c_str();
}

static void ggml_backend_cuda_free(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    delete cuda_ctx;
    delete backend;
}

static void ggml_backend_cuda_set_tensor_async(ggml_backend_t backend, ggml_tensor * tensor, const void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT((buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) || ggml_backend_buft_is_cuda_mapped(buf->buft)) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync((char *) tensor->data + offset, data, size, cudaMemcpyDefault, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT((buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) || ggml_backend_buft_is_cuda_mapped(buf->buft)) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDefault, cuda_ctx->stream()));
}

static void ggml_backend_cuda_set_tensor_2d_async(ggml_backend_t backend, struct ggml_tensor * tensor, const void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT((buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) || ggml_backend_buft_is_cuda_mapped(buf->buft)) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        (char *) tensor->data + offset, stride_tensor, data, stride_data, size, n_copies, cudaMemcpyDefault, cuda_ctx->stream()));
}

static void ggml_backend_cuda_get_tensor_2d_async(ggml_backend_t backend, const struct ggml_tensor * tensor, void * data,
        size_t offset, size_t size, size_t n_copies, size_t stride_tensor, size_t stride_data) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_backend_buffer_t buf = tensor->view_src ? tensor->view_src->buffer : tensor->buffer;

    GGML_ASSERT((buf->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) || ggml_backend_buft_is_cuda_mapped(buf->buft)) && "unsupported buffer type");

    CUDA_CHECK(cudaMemcpy2DAsync(
        data, stride_data, (const char *) tensor->data + offset, stride_tensor, size, n_copies, cudaMemcpyDefault, cuda_ctx->stream()));
}

static bool ggml_backend_cuda_cpy_tensor_async(ggml_backend_t backend_src, ggml_backend_t backend_dst, const ggml_tensor * src, ggml_tensor * dst) {
    ggml_backend_buffer_t buf_src = src->view_src ? src->view_src->buffer : src->buffer;
    ggml_backend_buffer_t buf_dst = dst->view_src ? dst->view_src->buffer : dst->buffer;

    if (!ggml_backend_is_cuda(backend_src) || !ggml_backend_is_cuda(backend_dst)) {
        return false;
    }

    if (!ggml_backend_buffer_is_cuda(buf_src) || !ggml_backend_buffer_is_cuda(buf_dst)) {
        return false;
    }

    // device -> device copy
    ggml_backend_cuda_context * cuda_ctx_src = (ggml_backend_cuda_context *) backend_src->context;
    ggml_backend_cuda_context * cuda_ctx_dst = (ggml_backend_cuda_context *) backend_dst->context;

    ggml_backend_cuda_buffer_context * buf_ctx_src = (ggml_backend_cuda_buffer_context *) buf_src->context;
    ggml_backend_cuda_buffer_context * buf_ctx_dst = (ggml_backend_cuda_buffer_context *) buf_dst->context;

    if (cuda_ctx_src->device != buf_ctx_src->device || cuda_ctx_dst->device != buf_ctx_dst->device) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: backend and buffer devices do not match\n", __func__);
#endif // NDEBUG
        return false;
    }

    if (backend_src != backend_dst) {
        // copy on src stream
        // compare the backing physical devices: distinct virtual devices may share one physical GPU,
        // in which case a same-device copy (not a peer copy) is required
        const int src_physical = ggml_cuda_get_physical_device(cuda_ctx_src->device);
        const int dst_physical = ggml_cuda_get_physical_device(cuda_ctx_dst->device);
        if (src_physical == dst_physical) {
            CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
        } else {
#ifdef GGML_CUDA_NO_PEER_COPY
            return false;
#else
            CUDA_CHECK(cudaMemcpyPeerAsync(dst->data, dst_physical, src->data, src_physical, ggml_nbytes(dst), cuda_ctx_src->stream()));
#endif // GGML_CUDA_NO_PEER_COPY
        }

        // record event on src stream after the copy
        if (!cuda_ctx_src->copy_event) {
            ggml_cuda_set_device(cuda_ctx_src->device);
            CUDA_CHECK(cudaEventCreateWithFlags(&cuda_ctx_src->copy_event, cudaEventDisableTiming));
        }

        CUDA_CHECK(cudaEventRecord(cuda_ctx_src->copy_event, cuda_ctx_src->stream()));

        // wait on dst stream for the copy to complete
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx_dst->stream(), cuda_ctx_src->copy_event, 0));
    } else {
        // src and dst are on the same backend
        CUDA_CHECK(cudaMemcpyAsync(dst->data, src->data, ggml_nbytes(dst), cudaMemcpyDeviceToDevice, cuda_ctx_src->stream()));
    }
    return true;
}

static void ggml_backend_cuda_synchronize(ggml_backend_t backend) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaStreamSynchronize(cuda_ctx->stream()));

    GGML_UNUSED(backend);
}

static bool ggml_cuda_is_view_or_noop(const ggml_tensor * t) {
    return ggml_is_empty(t) || t->op == GGML_OP_RESHAPE || t->op == GGML_OP_TRANSPOSE ||
           t->op == GGML_OP_VIEW || t->op == GGML_OP_PERMUTE || t->op == GGML_OP_NONE;
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_check_compability(ggml_cgraph * cgraph) {

    bool use_cuda_graph = true;
    // Loop over nodes in GGML graph to obtain info needed for CUDA graph

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_tensor * node = cgraph->nodes[i];

        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }

        // [TAG_MUL_MAT_ID_CUDA_GRAPHS]
        if (node->op == GGML_OP_MUL_MAT_ID) {
            const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
            if (ggml_cuda_mul_mat_id_needs_sync(node, cc)) {
                // the mul_mat_id fallback path synchronizes the stream, so we cannot use CUDA graphs
                // ref: https://github.com/ggml-org/llama.cpp/pull/18958
                use_cuda_graph = false;
#ifndef NDEBUG
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to unsupported node type\n", __func__);
#endif
            }
        }

        if (!use_cuda_graph) {
            break;
        }
    }

    return use_cuda_graph;
}

static const void * ggml_cuda_graph_get_key(ggml_cgraph * cgraph, const void ** shape_key) {
    // The key is the graph's shape, not its address: llama rebuilds a graph into fresh memory whenever its inputs
    // change, and a new key has to warm up node by node again. A collision only costs a re-capture, since node
    // properties are compared before replay.
    // Two keys: the shape (op, type and ne of every node) decides when a new graph is worth capturing; the shape with
    // every node's and source's address names the instance that replays. A verify, another slot's prompt chunk and
    // a draft are the same shape at other addresses: each keeps its own executable instead of capturing over the
    // last one's.
    uint64_t h = 0xcbf29ce484222325ULL ^ (uint64_t) cgraph->n_nodes;
    auto mix = [&h](uint64_t v) { h = (h ^ v) * 0x100000001b3ULL; };
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        mix((uint64_t) node->op);
        mix((uint64_t) node->type);
        for (int j = 0; j < GGML_MAX_DIMS; j++) {
            mix((uint64_t) node->ne[j]);
        }
    }
    if (shape_key) {
        *shape_key = (const void *) (uintptr_t) (h | 1);
    }
    for (int i = 0; i < cgraph->n_nodes; i++) {
        const ggml_tensor * node = cgraph->nodes[i];
        mix((uint64_t) (uintptr_t) node->data);
        for (int j = 0; j < GGML_MAX_SRC; j++) {
            mix(node->src[j] ? (uint64_t) (uintptr_t) node->src[j]->data : 0);
        }
    }
    return (const void *) (uintptr_t) (h | 1);
}

// A buffer of an op's own (the FP4 GEMM's scratch, the mask scans' results) that is freed and allocated again when it
// grows leaves every graph captured before pointing at freed memory: the context's captured graphs go before its next
// compute, and capture again from the new buffer.
static std::mutex g_cuda_graphs_stale_mtx;
static std::unordered_set<const ggml_backend_cuda_context *> g_cuda_graphs_stale;

void ggml_cuda_graphs_invalidate(ggml_backend_cuda_context & ctx) {
    std::lock_guard<std::mutex> lock(g_cuda_graphs_stale_mtx);
    g_cuda_graphs_stale.insert(&ctx);
}

static bool ggml_cuda_graphs_take_stale(const ggml_backend_cuda_context * ctx) {
    std::lock_guard<std::mutex> lock(g_cuda_graphs_stale_mtx);
    return g_cuda_graphs_stale.erase(ctx) > 0;
}

// calls per graph shape at any address, per backend context: a shape runs eagerly once before its instances are
// captured (kept here rather than in the context struct, whose header every kernel file includes)
struct ggml_cuda_shape_seen {
    const void * last_shape_key = nullptr;
    std::unordered_map<const void *, int> seen;
};

static ggml_cuda_shape_seen & ggml_cuda_shape_seen_of(ggml_backend_cuda_context * cuda_ctx) {
    static std::mutex mtx;
    static std::unordered_map<const ggml_backend_cuda_context *, ggml_cuda_shape_seen> m;
    std::lock_guard<std::mutex> lock(mtx);
    return m[cuda_ctx];
}

// hashing every node of a verify graph took longer than launching it; a graph the scheduler has not re-split
// keeps its uid, and with it the key it had
static const void * ggml_cuda_graph_key_for(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph) {
    if (cgraph->uid != 0 && cgraph->uid == cuda_ctx->last_uid) {
        return cuda_ctx->last_key;
    }
    const void * key = ggml_cuda_graph_get_key(cgraph, &ggml_cuda_shape_seen_of(cuda_ctx).last_shape_key);
    cuda_ctx->last_uid    = cgraph->uid;
    cuda_ctx->last_key    = key;
    cuda_ctx->last_compat = ggml_cuda_graph_check_compability(cgraph);
    return key;
}

static bool ggml_cuda_graph_update_required(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const void * graph_key) {
    bool res = false;

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    static const bool uid_trace = getenv("LLAMA_ROUND_TIMING") != nullptr;
    if (uid_trace && cgraph->n_nodes > 1000) {
        static int uid_hits = 0, uid_miss = 0, uid_zero = 0;
        if (cgraph->uid == 0) {
            uid_zero++;
        } else if (cgraph->uid == graph->uid) {
            uid_hits++;
        } else {
            uid_miss++;
        }
        if ((uid_hits + uid_miss + uid_zero) % 100 == 0) {
            GGML_LOG_INFO("%s: graph uid: %d reused, %d changed, %d absent\n", __func__, uid_hits, uid_miss, uid_zero);
            uid_hits = uid_miss = uid_zero = 0;
        }
    }
    if (cgraph->uid != 0 &&
        cgraph->uid == graph->uid) {
        GGML_LOG_DEBUG("CUDA Graph id %zu reused\n", cgraph->uid);
        GGML_ASSERT((int)graph->node_props.size() == cgraph->n_nodes);
        return false;
    }

    graph->uid = cgraph->uid;

    // Check if the graph size has changed
    if ((int)graph->node_props.size() != cgraph->n_nodes) {
        res = true;
        graph->node_props.resize(cgraph->n_nodes);
    }

    for (int i = 0; i < cgraph->n_nodes; i++) {
        ggml_cuda_graph::node_properties prop = {};
        memcpy(&prop.node, cgraph->nodes[i], sizeof(ggml_tensor));

        for (int j = 0; j < GGML_MAX_SRC; ++j) {
            if (cgraph->nodes[i]->src[j]) {
                prop.node_src_data_ptrs[j] = cgraph->nodes[i]->src[j]->data;
                memcpy(prop.node_src_ne[j], cgraph->nodes[i]->src[j]->ne, sizeof(prop.node_src_ne[j]));
                memcpy(prop.node_src_nb[j], cgraph->nodes[i]->src[j]->nb, sizeof(prop.node_src_nb[j]));
                prop.node_src_type[j] = cgraph->nodes[i]->src[j]->type;
            }
        }

        if (res || memcmp(&graph->node_props[i], &prop, sizeof(prop)) != 0) {
            graph->node_props[i] = prop;
            res = true;
        }
    }

    return res;
}

static void ggml_cuda_graph_update_executable(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

#if CUDART_VERSION >= 12000
    cudaGraphExecUpdateResultInfo result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &result_info);
#else
    cudaGraphNode_t errorNode;
    cudaGraphExecUpdateResult result_info;
    cudaError_t stat = cudaGraphExecUpdate(graph->instance, graph->graph, &errorNode, &result_info);
#endif // CUDART_VERSION >= 12000

    if (stat == cudaErrorGraphExecUpdateFailure) {
#ifndef NDEBUG
        GGML_LOG_DEBUG("%s: CUDA graph update failed\n", __func__);
#endif

        // The pre-existing graph exec cannot be updated due to violated constraints
        // so instead clear error and re-instantiate
        (void)cudaGetLastError();
        CUDA_CHECK(cudaGraphExecDestroy(graph->instance));
        graph->instance = nullptr;
        CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
    } else {
        GGML_ASSERT(stat == cudaSuccess);
    }
}
#endif // USE_CUDA_GRAPH

static bool ggml_cuda_fuse_multi_rope(const int mode) {
    return (mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE) && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_MROPE);
}

static bool ggml_cuda_should_fuse_rope_set_rows(const ggml_tensor * rope,
                                                const ggml_tensor * view,
                                                const ggml_tensor * set_rows) {

    if (rope->op != GGML_OP_ROPE || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    // ne3 not tested
    if (rope->src[0]->ne[3] != 1) {
        return false;
    }

    if (set_rows->type != GGML_TYPE_F32 && set_rows->type != GGML_TYPE_F16) {
        return false;
    }

    if (set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }

    // The view should flatten two dims of rope into one dim
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1]) {
        return false;
    }

    // the vision shader has no fusion code
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && !ggml_cuda_fuse_multi_rope(mode)) {
        return false;
    }

    return true;
}

// rope > reshape > mul_mat(rotation): one Hadamard rotation per head, at the widths the rotation kernel has
static bool ggml_cuda_should_fuse_rope_hadamard(const ggml_tensor * rope, const ggml_tensor * mm) {
    if (rope->op != GGML_OP_ROPE || mm->op != GGML_OP_MUL_MAT || rope->src[0]->ne[3] != 1) {
        return false;
    }

    const int64_t n = rope->ne[0];
    if (ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD || mm->src[0]->ne[0] != n || mm->src[0]->ne[1] != n ||
            n % 64 != 0 || n > 512 || mm->src[1]->ne[0] != n || mm->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(mm->src[1]) || !ggml_is_contiguous(mm) || !ggml_is_contiguous(rope) ||
            ggml_nelements(mm) != ggml_nelements(rope)) {
        return false;
    }

    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && !ggml_cuda_fuse_multi_rope(mode)) {
        return false;
    }

    return true;
}

// the K path of a quantized cache: the rotation, then reshape > view > set_rows(q8_0)
static bool ggml_cuda_should_fuse_rope_hadamard_set_rows(const ggml_tensor * rope,
                                                         const ggml_tensor * mm,
                                                         const ggml_tensor * view,
                                                         const ggml_tensor * set_rows) {
    if (!ggml_cuda_should_fuse_rope_hadamard(rope, mm) || view->op != GGML_OP_VIEW || set_rows->op != GGML_OP_SET_ROWS) {
        return false;
    }
    if (set_rows->type != GGML_TYPE_Q8_0 || set_rows->src[1]->type != GGML_TYPE_I64) {
        return false;
    }
    if (!ggml_is_contiguous(view) || view->ne[0] != rope->ne[0] * rope->ne[1] || view->ne[1] != rope->ne[2]) {
        return false;
    }

    return true;
}

// Two L2 norms whose inputs are views of the same tensor, laid out as one
// contiguous run of rows, with the outputs likewise adjacent. The delta-net
// q/k normalisation is this shape in every model that has it.
static bool ggml_cuda_should_fuse_l2_norm_pair(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->op != GGML_OP_L2_NORM || b->op != GGML_OP_L2_NORM) {
        return false;
    }
    if (a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * sa = a->src[0];
    const ggml_tensor * sb = b->src[0];
    if (!sa->view_src || sa->view_src != sb->view_src) {
        return false;
    }
    if (a->ne[0] != b->ne[0] || a->ne[2] != 1 || a->ne[3] != 1 || b->ne[2] != 1 || b->ne[3] != 1) {
        return false;
    }
    if (memcmp(a->op_params, b->op_params, sizeof(float)) != 0) {
        return false;
    }
    if (sa->nb[1] != sb->nb[1] || a->nb[1] != b->nb[1] || !ggml_is_contiguous_rows(a) || !ggml_is_contiguous_rows(b)) {
        return false;
    }
    // b's rows follow a's directly, in the source view and in the outputs
    const bool src_adjacent = (const char *) sb->data == (const char *) sa->data + a->ne[1]*sa->nb[1];
    const bool dst_adjacent = (const char *)  b->data == (const char *)  a->data + a->ne[1]* a->nb[1];
    return src_adjacent && dst_adjacent;
}

// Two L2 norms of one shape, eps and source strides, 3-D at most, contiguous outputs: q and k of a delta-net layer at
// any number of tokens. Neither may write what the other reads, since both run in one launch.
static bool ggml_cuda_should_fuse_l2_norm_pair2(const ggml_tensor * a, const ggml_tensor * b) {
    if (a->op != GGML_OP_L2_NORM || b->op != GGML_OP_L2_NORM || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32) {
        return false;
    }
    const ggml_tensor * sa = a->src[0];
    const ggml_tensor * sb = b->src[0];
    if (sa->type != GGML_TYPE_F32 || sb->type != GGML_TYPE_F32 || !ggml_are_same_shape(a, b) || a->ne[3] != 1 ||
            memcmp(a->op_params, b->op_params, sizeof(float)) != 0) {
        return false;
    }
    if (sa->nb[0] != sizeof(float) || sa->nb[1] != sb->nb[1] || sa->nb[2] != sb->nb[2] || !ggml_is_contiguous(a) || !ggml_is_contiguous(b)) {
        return false;
    }
    if (a->ne[1] > 65535 || a->ne[2] > 65535) {
        return false;
    }
    return !ggml_cuda_tensors_overlap(a, sb) && !ggml_cuda_tensors_overlap(b, sa) && !ggml_cuda_tensors_overlap(a, b);
}

// The mixture-of-experts combine is emitted as n_expert_used - 1 chained
// ADDs over 2D views of one [n_embd, n_expert_used, n_tokens] tensor, each
// view one expert row further along. That is one strided row-sum. Returns
// the number of ADD nodes in the chain (0 = no match) and the tensor whose
// rows are summed.
static int ggml_cuda_match_expert_sum(const ggml_cgraph * cgraph, int i, const ggml_tensor ** experts_out) {
    const ggml_tensor * add = cgraph->nodes[i];
    if (add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32) {
        return 0;
    }
    const ggml_tensor * v0 = add->src[0];
    const ggml_tensor * v1 = add->src[1];
    if (v0->op != GGML_OP_VIEW || v1->op != GGML_OP_VIEW || v0->view_src != v1->view_src) {
        return 0;
    }
    const ggml_tensor * experts = v0->view_src;
    if (experts->type != GGML_TYPE_F32 || !ggml_is_contiguous(experts) || experts->ne[3] != 1) {
        return 0;
    }
    const size_t stride = experts->nb[1];
    if (v0->view_offs != 0 || v1->view_offs != stride) {
        return 0;
    }
    // each view is one expert row across all tokens
    if (v0->ne[0] != experts->ne[0] || v0->ne[1] != experts->ne[2] || v0->nb[1] != experts->nb[2] ||
            !ggml_are_same_shape(v0, v1) || !ggml_are_same_shape(add, v0)) {
        return 0;
    }
    int n = 2; // rows summed so far
    int last = i;
    for (int j = i + 1; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * nxt = cgraph->nodes[j];
        if (nxt->op != GGML_OP_ADD || nxt->src[0] != cgraph->nodes[last]) {
            break;
        }
        const ggml_tensor * v = nxt->src[1];
        if (v->op != GGML_OP_VIEW || v->view_src != experts || v->view_offs != (size_t) n*stride ||
                !ggml_are_same_shape(v, v0) || v->nb[1] != v0->nb[1]) {
            break;
        }
        // the partial sum feeds only the next ADD
        if (ggml_node_get_use_count(cgraph, last) != 1 || (cgraph->nodes[last]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            break;
        }
        last = j;
        n++;
    }
    if (n != experts->ne[1]) {
        return 0; // a partial chain is not the full combine
    }
    *experts_out = experts;
    return last - i + 1;
}

// x [ne0, n_expert, n_tokens] times the router weights [1, n_expert, n_tokens], then the combine over its expert rows:
// the weighted sum in one launch, each product rounded before it is added, in the ADDs' order. Returns the nodes
// covered past i.
static constexpr uint64_t GGML_CUDA_FUSE_EXPERT_WSUM = 1ull << 41;

static int ggml_cuda_try_expert_wsum(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * mul = cgraph->nodes[i];
    if (mul->op != GGML_OP_MUL || mul->type != GGML_TYPE_F32 || (mul->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    const ggml_tensor * x = mul->src[0];
    const ggml_tensor * w = mul->src[1];
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || !ggml_are_same_shape(x, mul) || !ggml_is_contiguous(x) ||
            !ggml_is_contiguous(mul) || w->ne[0] != 1 || w->ne[1] != x->ne[1] || w->ne[2] != x->ne[2] || x->ne[3] != 1 || w->ne[3] != 1) {
        return 0;
    }
    int j = i + 1;
    while (j < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[j]) && cgraph->nodes[j]->op == GGML_OP_VIEW &&
            cgraph->nodes[j]->view_src == mul) {
        j++;
    }
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * experts = nullptr;
    const int n_adds = ggml_cuda_match_expert_sum(cgraph, j, &experts);
    if (n_adds == 0 || experts != mul) {
        return 0;
    }
    // the product is read only through the combine's views, each read once
    if (ggml_node_get_use_count(cgraph, i) != j - i - 1) {
        return 0;
    }
    for (int k = i + 1; k < j; ++k) {
        if (ggml_node_get_use_count(cgraph, k) != 1) {
            return 0;
        }
    }
    // the router weights die at the multiply in the graph's order, so the sum's output may have been given their
    // memory; one launch reads every token's weights while other blocks store, so neither input may lie under it
    ggml_tensor * dst = cgraph->nodes[j + n_adds - 1];
    if (ggml_cuda_tensors_overlap(dst, x) || ggml_cuda_tensors_overlap(dst, w)) {
        return 0;
    }
    ggml_cuda_op_sum_expert_rows_weighted(ctx, x, w, dst, (int) x->ne[1]);
    return j + n_adds - 1 - i;
}

// rms_norm > scale(const) > mul(weight): the scalar folds into the weight
// multiply, so the existing fused norm kernel serves with one more factor.
static bool ggml_cuda_should_fuse_rms_norm_scale_mul(const ggml_tensor * rms_norm, const ggml_tensor * scale, const ggml_tensor * mul) {
    if (rms_norm->op != GGML_OP_RMS_NORM || scale->op != GGML_OP_SCALE || mul->op != GGML_OP_MUL) {
        return false;
    }
    if (scale->src[0] != rms_norm || (mul->src[0] != scale && mul->src[1] != scale)) {
        return false;
    }
    if (rms_norm->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32) {
        return false;
    }
    float bias = 0.0f;
    memcpy(&bias, (const char *) scale->op_params + sizeof(float), sizeof(float));
    if (bias != 0.0f) {
        return false;
    }
    const ggml_tensor * w = mul->src[0] == scale ? mul->src[1] : mul->src[0];
    return w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_is_contiguous(rms_norm->src[0]) && ggml_is_contiguous(mul);
}

static bool ggml_cuda_should_fuse_rms_norm_mul_rope(const ggml_tensor * rms_norm,
                                                    const ggml_tensor * mul,
                                                    const ggml_tensor * rope) {
    if (rms_norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || rope->op != GGML_OP_ROPE) {
        return false;
    }

    if (rms_norm->src[0]->type != GGML_TYPE_F32 || rms_norm->type != GGML_TYPE_F32 ||
        mul->src[0]->type != GGML_TYPE_F32 || mul->src[1]->type != GGML_TYPE_F32 ||
        mul->type != GGML_TYPE_F32 || rope->type != GGML_TYPE_F32) {
        return false;
    }

    if (rope->src[0] != mul) {
        return false;
    }

    //if rms norm is the B operand, then we don't handle broadcast
    if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
        return false;
    }

    if (!ggml_are_same_shape(rms_norm, mul)) {
        return false;
    }

    //rms_norm kernel assumes contiguous rows
    if (!ggml_is_contiguous_rows(rms_norm->src[0]) ||
        !ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
        return false;
    }

    // the fused kernel handles every mode but vision
    const int mode = ((const int32_t *) rope->op_params)[2];
    if (mode != GGML_ROPE_TYPE_NORMAL && mode != GGML_ROPE_TYPE_NEOX && !ggml_cuda_fuse_multi_rope(mode)) {
        return false;
    }

    const int n_dims = ((const int32_t *) rope->op_params)[1];
    if (n_dims % 2 != 0 || rope->src[0]->ne[0] % 2 != 0) {
        return false;
    }

    // ggml_rope_set_offset is not yet supported in the fused kernel
    const int n_offs = ((const int32_t *) rope->op_params)[15];
    if (n_offs != 0) {
        return false;
    }

    return true;
}

// An elementwise node: f32, contiguous, one of the ops the chain kernel
// speaks, and for a binary op a second operand that broadcasts onto it.
static int ggml_cuda_chain_op_of(const ggml_tensor * n) {
    if (n->type != GGML_TYPE_F32 || !ggml_is_contiguous(n) || n->src[0] == nullptr || n->src[0]->type != GGML_TYPE_F32) {
        return 0;
    }
    switch (n->op) {
        case GGML_OP_ADD: case GGML_OP_SUB: case GGML_OP_MUL: case GGML_OP_DIV: {
            const ggml_tensor * b = n->src[1];
            if (b == nullptr || b->type != GGML_TYPE_F32 || !ggml_is_contiguous(b) || !ggml_are_same_shape(n, n->src[0])) {
                return 0;
            }
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (b->ne[d] != 1 && b->ne[d] != n->ne[d]) {
                    return 0;
                }
            }
            return n->op == GGML_OP_ADD ? GGML_CUDA_CHAIN_ADD : n->op == GGML_OP_SUB ? GGML_CUDA_CHAIN_SUB :
                   n->op == GGML_OP_MUL ? GGML_CUDA_CHAIN_MUL : GGML_CUDA_CHAIN_DIV;
        }
        case GGML_OP_SCALE: return ggml_are_same_shape(n, n->src[0]) ? GGML_CUDA_CHAIN_SCALE : 0;
        case GGML_OP_CLAMP: return ggml_are_same_shape(n, n->src[0]) ? GGML_CUDA_CHAIN_CLAMP : 0;
        case GGML_OP_SQR:   return ggml_are_same_shape(n, n->src[0]) ? GGML_CUDA_CHAIN_SQR : 0;
        case GGML_OP_SQRT:  return ggml_are_same_shape(n, n->src[0]) ? GGML_CUDA_CHAIN_SQRT : 0;
        case GGML_OP_LOG:   return ggml_are_same_shape(n, n->src[0]) ? GGML_CUDA_CHAIN_LOG : 0;
        case GGML_OP_SIN:   return ggml_are_same_shape(n, n->src[0]) ? GGML_CUDA_CHAIN_SIN : 0;
        case GGML_OP_COS:   return ggml_are_same_shape(n, n->src[0]) ? GGML_CUDA_CHAIN_COS : 0;
        case GGML_OP_UNARY: {
            const int u = (int) ggml_get_unary_op(n);
            return ggml_are_same_shape(n, n->src[0]) && ggml_cuda_chain_unary_supported(u) ? GGML_CUDA_CHAIN_UNARY + u : 0;
        }
        default:
            return 0;
    }
}

// The generic pass: a run of elementwise nodes where each consumes the one
// before it and nothing else does, collapsed into one launch. Views between
// them are skipped; anything else ends the run. Applies to every model.
// a BF16 copy of out, the F32 result of node i_last, for the BF16 GEMM after it over several rows that reads it (see
// ggml_cuda_bf16_shadows), or nullptr: none, or no room for it while a graph is captured
static nv_bfloat16 * ggml_cuda_bf16_shadow_for(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i_last, const ggml_tensor * out) {
    if ((ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_BF16_SHADOW) || out->type != GGML_TYPE_F32 || !ggml_is_contiguous(out) || out->ne[1] <= 8) {
        return nullptr;
    }
    const ggml_tensor * mm = nullptr;
    for (int j = i_last + 1; j < cgraph->n_nodes && j <= i_last + 4; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(t)) {
            continue;
        }
        if (t->op == GGML_OP_MUL_MAT && t->src[1] == out && t->src[0]->type == GGML_TYPE_BF16) {
            mm = t;
        }
        break;
    }
    if (mm == nullptr) {
        return nullptr;
    }
    ggml_cuda_bf16_shadows & sh = g_bf16_shadows[&ctx];
    const size_t n = (size_t) ggml_nelements(out);
    if (sh.slot < n) {
        cudaStreamCaptureStatus capture = cudaStreamCaptureStatusNone;
        CUDA_CHECK(cudaStreamIsCapturing(ctx.stream(), &capture));
        if (capture != cudaStreamCaptureStatusNone) {
            return nullptr;
        }
        CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
        if (sh.buf) {
            CUDA_CHECK(cudaFree(sh.buf));
        }
        sh.slot = std::max<size_t>(n + n/4, 1u << 20);
        CUDA_CHECK(cudaMalloc(&sh.buf, 2*sh.slot*sizeof(nv_bfloat16)));
        sh.e[0] = sh.e[1] = {};
        // graphs captured before write the freed buffer
        ggml_cuda_graphs_invalidate(ctx);
    }
    auto & e = sh.e[sh.next];
    e.t    = out;
    e.data = out->data;
    e.n    = (int64_t) n;
    e.walk = sh.walk;
    e.bf16 = sh.buf + sh.next*sh.slot;
    sh.next ^= 1;
    return e.bf16;
}

static int ggml_cuda_try_elt_chain(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    const ggml_tensor * first = cgraph->nodes[i];
    if (ggml_cuda_chain_op_of(first) == 0 || (first->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }
    if (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_ELT_CHAIN) {
        return 0;
    }
    ggml_cuda_chain_args a{};
    int n_args = 0;
    const ggml_tensor * operands[GGML_CUDA_CHAIN_MAX_ARGS] = {};
    auto strides_of = [&](const ggml_tensor * t, ggml_cuda_chain_arg & out) {
        out.data = (const float *) t->data;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            out.s[d] = t->ne[d] == 1 ? 0 : (int64_t) (t->nb[d] / sizeof(float));
        }
    };
    auto add_step = [&](const ggml_tensor * n, int op, const ggml_tensor * other) {
        ggml_cuda_chain_step & st = a.step[a.n_steps++];
        st.op  = op;
        st.arg = -1;
        st.p0  = 0.0f;
        st.p1  = 0.0f;
        if (other != nullptr) {
            st.arg = n_args;
            operands[n_args] = other;
            strides_of(other, a.arg[n_args++]);
        }
        if (n->op == GGML_OP_SCALE || n->op == GGML_OP_CLAMP) {
            memcpy(&st.p0, (const float *) n->op_params + 0, sizeof(float));
            memcpy(&st.p1, (const float *) n->op_params + 1, sizeof(float));
        }
    };
    // the first node reads x = src0 (and maybe src1)
    const ggml_tensor * x_t = first->src[0];
    strides_of(x_t, a.x);
    add_step(first, ggml_cuda_chain_op_of(first), first->src[1]);

    const ggml_tensor * shape = first;
    const ggml_tensor * prev  = first;
    int prev_idx = i;
    int last     = i;
    for (int j = i + 1; j < cgraph->n_nodes && a.n_steps < GGML_CUDA_CHAIN_MAX_STEPS && n_args < GGML_CUDA_CHAIN_MAX_ARGS; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n) || ggml_is_empty(n)) {
            continue;
        }
        int op = ggml_cuda_chain_op_of(n);
        // the chain may widen: a value computed over few elements can be read
        // by many, which is what a gate multiplied into the tensor it gates
        // looks like. Everything read so far is then read with a zero stride,
        // which the final shape check below confirms.
        // GGML_CUDA_CHAIN_NOGROW=1 holds the chain to one shape, which is what
        // it did before it could widen: for measuring the difference with one
        // binary rather than two
        static const bool no_grow = getenv("GGML_CUDA_CHAIN_NOGROW") != nullptr;
        bool fits = true;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            fits = fits && (shape->ne[d] == n->ne[d] || (!no_grow && shape->ne[d] == 1));
        }
        if (op == 0 || !fits) {
            break;
        }
        // a graph output (the logits after their softcap) ends the chain: the chain writes only its last node
        const bool is_output = (n->flags & GGML_TENSOR_FLAG_OUTPUT) != 0;
        const ggml_tensor * other = nullptr;
        if (n->src[0] == prev) {
            other = n->src[1];
        } else if (n->src[1] == prev && (n->op == GGML_OP_ADD || n->op == GGML_OP_MUL)) {
            other = n->src[0];
        } else if (n->src[1] == prev && n->op == GGML_OP_SUB) {
            op = GGML_CUDA_CHAIN_RSUB; other = n->src[0];
        } else if (n->src[1] == prev && n->op == GGML_OP_DIV) {
            op = GGML_CUDA_CHAIN_RDIV; other = n->src[0];
        } else {
            break;
        }
        if (other != nullptr && !ggml_are_same_shape(other, n)) {
            // the other operand must broadcast onto the chain's shape
            for (int d = 0; d < GGML_MAX_DIMS; ++d) {
                if (other->ne[d] != 1 && other->ne[d] != n->ne[d]) {
                    return 0;
                }
            }
        }
        // the previous value feeds only this node
        if (ggml_node_get_use_count(cgraph, prev_idx) != 1) {
            break;
        }
        add_step(n, op, other);
        shape    = n;
        prev     = n;
        prev_idx = j;
        last     = j;
        if (is_output) {
            break;
        }
    }
    if (a.n_steps < 2) {
        return 0;
    }
    const ggml_tensor * out = cgraph->nodes[last];
    a.dst = (float *) out->data;
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        a.ne[d] = out->ne[d];
    }
    // whatever the chain reads is indexed with the output's shape, so each of
    // its dimensions either matches or is one, which strides_of already turned
    // into a zero stride. A chain that never widened passes this trivially.
    auto under_out = [&](const ggml_tensor * t) {
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            if (t->ne[d] != 1 && t->ne[d] != a.ne[d]) {
                return false;
            }
        }
        return true;
    };
    if (!under_out(x_t)) {
        return 0;
    }
    for (int k = 0; k < n_args; ++k) {
        if (!under_out(operands[k])) {
            return 0;
        }
    }

    // a broadcast operand is read by many threads; it must not sit under the output
    const size_t out_bytes = ggml_nbytes(out);
    for (int k = 0; k < n_args; ++k) {
        const ggml_tensor * t = operands[k];
        bool broadcast = false;
        for (int d = 0; d < GGML_MAX_DIMS; ++d) {
            broadcast = broadcast || (t->ne[d] == 1 && a.ne[d] > 1);
        }
        const char * ob = (const char *) a.dst;
        const char * tb = (const char *) t->data;
        if (broadcast && tb < ob + out_bytes && ob < tb + ggml_nbytes(t)) {
            return 0;
        }
    }
    a.dst_bf16 = ggml_cuda_bf16_shadow_for(ctx, cgraph, last, out);
    ggml_cuda_op_elt_chain(ctx, a);
    return last - i;
}

// The logits' softcap (scale, tanh, scale) read by nothing but the greedy argmax and the gather of the chosen logit
// (llm_graph_context::build_sampling when no row's logits go to the host, so they are no graph output): the argmax
// caps each element as it reads it and writes the capped maximum as the gathered logit. The capped logits, a full
// pass over every row's vocabulary, are never stored. FUSE_OFF bit 55
static constexpr uint64_t GGML_CUDA_FUSE_SOFTCAP_ARGMAX = 1ull << 55;

struct ggml_cuda_softcap_argmax_nodes {
    const ggml_tensor *      x  = nullptr; // the logits before the cap
    ggml_tensor *            am = nullptr; // the argmax
    ggml_tensor *            gr = nullptr; // the gather of the chosen logit
    int                      i_gr = -1;
    ggml_cuda_softcap_params sc;
};

static bool ggml_cuda_match_softcap_argmax(const ggml_cgraph * cgraph, const int i, ggml_cuda_softcap_argmax_nodes & m) {
    if (i + 4 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * s0 = cgraph->nodes[i];
    const ggml_tensor * th = cgraph->nodes[i + 1];
    const ggml_tensor * s1 = cgraph->nodes[i + 2];
    if (s0->op != GGML_OP_SCALE || th->op != GGML_OP_UNARY || ggml_get_unary_op(th) != GGML_UNARY_OP_TANH ||
            s1->op != GGML_OP_SCALE || th->src[0] != s0 || s1->src[0] != th) {
        return 0;
    }
    const ggml_tensor * x = s0->src[0];
    if (x->type != GGML_TYPE_F32 || s1->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(s1) ||
            !ggml_are_same_shape(x, s1) || x->ne[0] < 16384 || x->ne[2] != 1 || x->ne[3] != 1 ||
            ((s0->flags | th->flags | s1->flags) & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, i + 1) != 1) {
        return false;
    }
    // next: the argmax of the capped logits, their reshape and the ids' reshape, the gather
    int i_am = -1, i_gr = -1;
    for (int j = i + 3; j < cgraph->n_nodes && j <= i + 7; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_ARGMAX && n->src[0] == s1 && i_am < 0) {
            i_am = j;
        } else if (n->op == GGML_OP_GET_ROWS && i_am >= 0) {
            i_gr = j;
            break;
        } else if (n->op != GGML_OP_RESHAPE) {
            return false;
        }
    }
    if (i_gr < 0) {
        return false;
    }
    ggml_tensor * am = cgraph->nodes[i_am];
    ggml_tensor * gr = cgraph->nodes[i_gr];
    const ggml_tensor * gl = gr->src[0];
    const ggml_tensor * gi = gr->src[1];
    const int64_t n_rows = x->ne[1];
    if (gl->op != GGML_OP_RESHAPE || gl->src[0] != s1 || gl->ne[0] != 1 || gl->ne[1] != x->ne[0] || gl->ne[2] != n_rows ||
            gi->op != GGML_OP_RESHAPE || gi->src[0] != am || gr->type != GGML_TYPE_F32 || !ggml_is_contiguous(gr) ||
            ggml_nelements(gr) != n_rows || am->type != GGML_TYPE_I32 || ggml_nelements(am) != n_rows) {
        return false;
    }
    // nothing else reads the capped logits, directly or through a view
    for (int j = i + 3; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n == am || n == gl) {
            continue;
        }
        if (n->view_src == s1) {
            return false;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == s1 || (n->src[s] && n->src[s] == gl && n != gr)) {
                return false;
            }
        }
    }
    memcpy(&m.sc.a0, (const float *) s0->op_params + 0, sizeof(float));
    memcpy(&m.sc.a1, (const float *) s0->op_params + 1, sizeof(float));
    memcpy(&m.sc.b0, (const float *) s1->op_params + 0, sizeof(float));
    memcpy(&m.sc.b1, (const float *) s1->op_params + 1, sizeof(float));
    m.x    = x;
    m.am   = am;
    m.gr   = gr;
    m.i_gr = i_gr;
    return true;
}

// The same softcap read by nothing but the batched sampler's top-k and the gather of its candidates' logits
// (build_sampling's batched form, its logits no graph output): the select caps each logit as it reads it, the gather
// caps the candidates, and the capped logits are never stored. FUSE_OFF bit 58
static constexpr uint64_t GGML_CUDA_FUSE_SOFTCAP_TOPK = 1ull << 58;

static int ggml_cuda_try_softcap_topk(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if ((ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_SOFTCAP_TOPK) || i + 5 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * s0 = cgraph->nodes[i];
    const ggml_tensor * th = cgraph->nodes[i + 1];
    const ggml_tensor * s1 = cgraph->nodes[i + 2];
    if (s0->op != GGML_OP_SCALE || th->op != GGML_OP_UNARY || ggml_get_unary_op(th) != GGML_UNARY_OP_TANH ||
            s1->op != GGML_OP_SCALE || th->src[0] != s0 || s1->src[0] != th) {
        return 0;
    }
    const ggml_tensor * x = s0->src[0];
    if (x->type != GGML_TYPE_F32 || s1->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_is_contiguous(s1) ||
            !ggml_are_same_shape(x, s1) || x->ne[2] != 1 || x->ne[3] != 1 ||
            ((s0->flags | th->flags | s1->flags) & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, i + 1) != 1) {
        return 0;
    }
    // next: the top-k of the capped logits, their reshape for the gather, the gather of the candidates
    int i_tk = -1, i_gr = -1;
    for (int j = i + 3; j < cgraph->n_nodes && j <= i + 8; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n->op == GGML_OP_TOP_K && n->src[0] == s1 && i_tk < 0) {
            i_tk = j;
        } else if (n->op == GGML_OP_GET_ROWS && i_tk >= 0) {
            i_gr = j;
            break;
        } else if (n->op != GGML_OP_RESHAPE) {
            return 0;
        }
    }
    if (i_gr < 0) {
        return 0;
    }
    ggml_tensor * tk = cgraph->nodes[i_tk];
    ggml_tensor * gr = cgraph->nodes[i_gr];
    const ggml_tensor * gl = gr->src[0];
    const int64_t ncols = x->ne[0], nrows = x->ne[1], k = tk->ne[0];
    if (gl->op != GGML_OP_RESHAPE || gl->src[0] != s1 || gl->ne[0] != 1 || gl->ne[1] != ncols || gl->ne[2] != nrows ||
            gr->src[1] != tk || tk->type != GGML_TYPE_I32 || !ggml_is_contiguous(tk) || tk->ne[1] != nrows ||
            gr->type != GGML_TYPE_F32 || !ggml_is_contiguous(gr) || ggml_nelements(gr) != k*nrows ||
            !ggml_cuda_top_k_softcap_supported(ncols, nrows, k)) {
        return 0;
    }
    // nothing else reads the capped logits, directly or through a view
    for (int j = i + 3; j < cgraph->n_nodes; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (n == tk || n == gl) {
            continue;
        }
        if (n->view_src == s1) {
            return 0;
        }
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            if (n->src[s] == s1 || (n->src[s] && n->src[s] == gl && n != gr)) {
                return 0;
            }
        }
    }
    ggml_cuda_softcap_params sc;
    memcpy(&sc.a0, (const float *) s0->op_params + 0, sizeof(float));
    memcpy(&sc.a1, (const float *) s0->op_params + 1, sizeof(float));
    memcpy(&sc.b0, (const float *) s1->op_params + 0, sizeof(float));
    memcpy(&sc.b1, (const float *) s1->op_params + 1, sizeof(float));
    ggml_cuda_top_k_softcap(ctx, (const float *) x->data, (int) ncols, (int) nrows, (int) k, sc, (int32_t *) tk->data, (float *) gr->data);
    return i_gr - i;
}

// Flash attention > reshape > scale (an FP8 cache's V scale on the output, llm_graph_context::build_attn_mha): the
// attention writes into the scale's tensor and applies the scale as it normalizes. FUSE_OFF bit 60
static constexpr uint64_t GGML_CUDA_FUSE_FA_SCALE = 1ull << 60;

static int ggml_cuda_try_fa_out_scale(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if ((ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_FA_SCALE) || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * fa = cgraph->nodes[i];
    const ggml_tensor * rs = cgraph->nodes[i + 1];
    ggml_tensor * sc = cgraph->nodes[i + 2];
    float s = 1.0f, b = 0.0f;
    memcpy(&s, (const float *) sc->op_params + 0, sizeof(float));
    memcpy(&b, (const float *) sc->op_params + 1, sizeof(float));
    if (rs->op != GGML_OP_RESHAPE || rs->src[0] != fa || sc->op != GGML_OP_SCALE || sc->src[0] != rs || b != 0.0f ||
            fa->type != GGML_TYPE_F32 || sc->type != GGML_TYPE_F32 || !ggml_is_contiguous(fa) || !ggml_is_contiguous(sc) ||
            ggml_nbytes(fa) != ggml_nbytes(sc) || ((fa->flags | rs->flags) & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, i + 1) != 1) {
        return 0;
    }
    void * data = fa->data;
    fa->data = sc->data;
    ggml_cuda_flash_attn_ext_scaled(ctx, fa, s);
    fa->data = data;
    return 2;
}

static int ggml_cuda_try_softcap_argmax(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    ggml_cuda_softcap_argmax_nodes m;
    if ((ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_SOFTCAP_ARGMAX) || !ggml_cuda_match_softcap_argmax(cgraph, i, m)) {
        return 0;
    }
    ggml_cuda_argmax_softcap(ctx, m.x, (int32_t *) m.am->data, (float *) m.gr->data, m.sc);
    return m.i_gr - i;
}


// match gated_delta_net + the strided cpy that scatters its state snapshots into the cache
// (slot i -> rollback group i, slot 0 newest), so the kernel can write them and skip the cpy.
static int ggml_cuda_try_gdn_cache_fusion(
        const ggml_cgraph * cgraph, int node_idx, ggml_cuda_gated_delta_net_fused_cache & fused_state_cpy) {
    const ggml_tensor * gdn = cgraph->nodes[node_idx];
    // the kernel skips the snapshot tail, so the gdn output must not be a graph output
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->type != GGML_TYPE_F32 ||
        (gdn->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return 0;
    }

    const ggml_tensor * src_v     = gdn->src[2];
    const int64_t       S_v       = src_v->ne[0];
    const int64_t       H         = src_v->ne[1];
    const int64_t       n_tokens  = src_v->ne[2];
    const int64_t       n_seqs    = src_v->ne[3];
    const int64_t       D         = S_v * S_v * H;
    const int64_t       K         = ggml_get_op_params_i32(gdn, 0); // snapshot slot count
    const int64_t       n_written = std::min<int64_t>(n_tokens, K); // newest n_written slots are written

    // snapshot tail starts right after the attention scores
    const size_t tail_off = ggml_row_size(GGML_TYPE_F32, S_v * H * n_tokens * n_seqs);

    // snapshot cpy is the first real node after the gdn (skip views/no-ops)
    const ggml_tensor * cpy  = nullptr;
    int                 skip = 0;
    for (int j = node_idx + 1; j < cgraph->n_nodes && cpy == nullptr; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(n)) {
            continue;
        }
        if (n->op != GGML_OP_CPY || (n->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
        cpy  = n;
        skip = j - node_idx;
    }
    if (cpy == nullptr) {
        return 0;
    }

    const ggml_tensor * src = cpy->src[0]; // view of the gdn snapshot tail
    const ggml_tensor * dst = cpy->src[1]; // cache view the kernel writes to

    // src must be this gdn's snapshot tail (contiguous, at the tail offset)
    if (src->op != GGML_OP_VIEW || src->view_src != gdn || src->view_offs != tail_off ||
        !ggml_is_contiguous(src)) {
        return 0;
    }

    // dst is the [D, n_seqs, n_written] cache view; require nb[1] == D (the per-seq stride the kernel
    // assumes). ggml_cpy pins src to the same element count.
    const std::array<int64_t, GGML_MAX_DIMS> expected_ne = { D, n_seqs, n_written, 1 };
    if (dst->op != GGML_OP_VIEW || (dst->type != GGML_TYPE_F32 && dst->type != GGML_TYPE_BF16) || dst->data == nullptr ||
        !std::equal(expected_ne.begin(), expected_ne.end(), dst->ne) ||
        dst->nb[0] != ggml_type_size(dst->type) || dst->nb[1] != (size_t) ggml_row_size(dst->type, D)) {
        return 0;
    }

    fused_state_cpy.data        = dst->data; // rollback group 0 (newest)
    fused_state_cpy.type        = dst->type;
    fused_state_cpy.slot_stride = K > 1 ? (int64_t) (dst->nb[2] / ggml_type_size(dst->type)) : 0;
    return skip;
}

static bool ggml_cuda_ranges_overlap(const void * a, size_t na, const void * b, size_t nb) {
    return (const char *) a < (const char *) b + nb && (const char *) b < (const char *) a + na;
}

// a single-row gather from a recurrent state cache whose only reader is a gated_delta_net, or the
// concat that opens a conv window: the reader takes the cache row in place and the gather is not launched
static bool ggml_cuda_try_state_fold(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int g) {
    static const bool disabled = (getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && atoi(getenv("GGML_CUDA_DISABLE_FUSION"))) ||
                                 (getenv("GGML_CUDA_GRAPH_OPT") != nullptr && atoi(getenv("GGML_CUDA_GRAPH_OPT")) == 1);
    if (disabled || (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_STATE_FOLD)) {
        return false;
    }

    const ggml_tensor * node = cgraph->nodes[g];
    const ggml_tensor * src  = node->src[0];
    const ggml_tensor * ids  = node->src[1];
    if (node->op != GGML_OP_GET_ROWS || node->type != GGML_TYPE_F32 || (src->type != GGML_TYPE_F32 && src->type != GGML_TYPE_BF16) ||
        ids->type != GGML_TYPE_I32 || (node->flags & GGML_TENSOR_FLAG_OUTPUT) || !ggml_is_contiguous(node) ||
        ggml_nelements(ids) < 1 || src->ne[2] != 1 || src->ne[3] != 1 || src->nb[0] != ggml_type_size(src->type) ||
        src->nb[1] % ggml_type_size(src->type) != 0 || src->data == nullptr || ids->data == nullptr) {
        return false;
    }
    const int64_t n_ids = ggml_nelements(ids);
    // the cache is only written by graph nodes, and the id is a leaf set before the graph starts
    const ggml_tensor * cache    = src->view_src ? src->view_src : src;
    const ggml_tensor * ids_base = ids->view_src ? ids->view_src : ids;
    if (cache->buffer == nullptr || ggml_backend_buffer_get_usage(cache->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
        ids_base->op != GGML_OP_NONE) {
        return false;
    }

    // the only reader, reached through offset-0 contiguous views each read once
    const ggml_tensor * state     = node;
    int                 state_idx = g;
    int                 c         = g + 1;
    for (; c < cgraph->n_nodes; ++c) {
        const ggml_tensor * n = cgraph->nodes[c];
        if (std::find(n->src, n->src + GGML_MAX_SRC, state) == n->src + GGML_MAX_SRC) {
            continue;
        }
        if (ggml_node_get_use_count(cgraph, state_idx) != 1) {
            return false;
        }
        if ((n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW) && n->view_offs == 0 && ggml_is_contiguous(n)) {
            state     = n;
            state_idx = c;
            continue;
        }
        break;
    }
    if (c >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * reader = cgraph->nodes[c];
    if (!(reader->flags & GGML_TENSOR_FLAG_COMPUTE) || ggml_is_empty(reader) || reader->type != GGML_TYPE_F32 ||
        ggml_nelements(state) != src->ne[0]*n_ids) {
        return false;
    }
    if (reader->op == GGML_OP_GATED_DELTA_NET) {
        // a sequence's row each: the kernel reads row ids[s] for sequence s
        if (reader->src[5] != state || reader->src[2]->ne[3] != n_ids) {
            return false;
        }
        for (int s = 0; s < 5; ++s) {
            if (reader->src[s] == state) {
                return false;
            }
        }
    } else if (reader->op == GGML_OP_CONCAT) {
        // a conv state heading its input window
        if (reader->src[0] != state || reader->src[1] == state || ggml_get_op_params_i32(reader, 0) == 3 || n_ids != 1 ||
                src->type != GGML_TYPE_F32) {
            return false;
        }
    } else {
        return false;
    }

    if (!ctx.state_writers_ready) {
        ctx.state_writers.clear();
        for (int k = 0; k < cgraph->n_nodes; ++k) {
            const ggml_tensor * n = cgraph->nodes[k];
            if (ggml_cuda_is_view_or_noop(n) || !(n->flags & GGML_TENSOR_FLAG_COMPUTE) || n->data == nullptr || n->buffer == nullptr ||
                ggml_backend_buffer_get_usage(n->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                continue;
            }
            ctx.state_writers.push_back({ k, (const char *) n->data, (const char *) n->data + ggml_nbytes(n) });
        }
        ctx.state_writers_ready = true;
    }

    // nothing may write the cache between the gather and its reader; the row is loaded before the wait
    // only when an earlier gated_delta_net, which exits after its own wait, orders all earlier work
    bool pre = false;
    for (int k = 0; k < g && !pre; ++k) {
        pre = cgraph->nodes[k]->op == GGML_OP_GATED_DELTA_NET;
    }
    for (const ggml_cuda_state_writer & w : ctx.state_writers) {
        if (w.node >= c) {
            break;
        }
        if (ggml_cuda_ranges_overlap(w.begin, w.end - w.begin, src->data, ggml_nbytes(src))) {
            if (w.node > g) {
                return false;
            }
            pre = false;
        }
    }

    // the id must outlive its last reader: no node up to the reader, the fused cache copy included,
    // may take its memory
    ggml_cuda_gated_delta_net_fused_cache unused;
    const int last = c + ggml_cuda_try_gdn_cache_fusion(cgraph, c, unused);
    for (int k = g + 1; k <= last; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (!ggml_cuda_is_view_or_noop(n) && n->data != nullptr &&
            ggml_cuda_ranges_overlap(n->data, ggml_nbytes(n), ids->data, n_ids*sizeof(int32_t))) {
            return false;
        }
    }

    ggml_cuda_state_fold & fold = ctx.state_folds[node];
    fold.base = src->data;
    fold.type = src->type;
    fold.ids  = (const int32_t *) ids->data;
    fold.row  = src->nb[1] / ggml_type_size(src->type);
    fold.pre  = pre && reader->op == GGML_OP_GATED_DELTA_NET;
#ifdef GGML_CUDA_DEBUG
    GGML_LOG_DEBUG("%s: %s read in place by %s%s\n", __func__, node->name, reader->name, fold.pre ? " before its wait" : "");
#endif
    return true;
}

static bool ggml_cuda_topk_moe_fusion(const struct ggml_cgraph * cgraph, int node_idx, ggml_cuda_topk_moe_args & args) {
    args.sigmoid         = false;
    args.sqrt_softplus   = false;
    args.softmax         = false;
    args.delayed_softmax = false;
    args.prob_bias       = false;
    args.norm            = false;

    const int      n_nodes = cgraph->n_nodes;
    ggml_tensor ** nodes   = cgraph->nodes;

    if (nodes[node_idx]->op == GGML_OP_SOFT_MAX) {
        args.softmax = true;
    }

    if (nodes[node_idx]->op == GGML_OP_UNARY) {
        const ggml_unary_op unary_op = ggml_get_unary_op(nodes[node_idx]);
        if (unary_op == GGML_UNARY_OP_SIGMOID) {
            args.sigmoid = true;
        } else if (unary_op == GGML_UNARY_OP_SOFTPLUS && node_idx + 1 < n_nodes &&
                   nodes[node_idx + 1]->op == GGML_OP_SQRT && nodes[node_idx + 1]->src[0] == nodes[node_idx]) {
            // sqrt(softplus(x)) scoring (DeepSeek-V4)
            args.sqrt_softplus = true;
            node_idx++;
        } else {
            return false;
        }
    }

    if (nodes[node_idx]->op == GGML_OP_ARGSORT) {
        args.delayed_softmax = true;
    }

    node_idx++;

    if (args.sigmoid || args.sqrt_softplus || args.softmax) {
        // SOFTMAX -> RESHAPE
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_RESHAPE ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx];
        node_idx++;

        if (node_idx >= n_nodes) {
            return false;
        }

        // src of bias add is the unreshaped probs (-2 instead of -1)
        if (nodes[node_idx]->op == GGML_OP_ADD && nodes[node_idx]->src[0] == nodes[node_idx - 2]) {
            args.prob_bias = true;
            node_idx++;
        }
        // RESHAPE/ADD -> ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_ARGSORT) {
            return false;
        }

        if (args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        } else if (!args.prob_bias && nodes[node_idx]->src[0] != nodes[node_idx - 2]) {
            return false;
        }

        node_idx++;

        // ARGSORT-> VIEW
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
                nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_GET_ROWS) {
            return false;
        }

        // GET_ROWS
        if (nodes[node_idx]->src[0] != probs_reshaped || nodes[node_idx]->src[1] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;
    } else if (args.delayed_softmax) {
        if (node_idx - 2 < 0) {
            return false;
        }
        ggml_tensor * probs_reshaped = nodes[node_idx - 2];

        // VIEW->ARGSORT
        if (node_idx >= n_nodes || nodes[node_idx]->op != GGML_OP_VIEW ||
            nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            return false;
        }
        node_idx++;

        // GET_ROWS
        if (node_idx >= n_nodes || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
                nodes[node_idx]->src[0] != probs_reshaped) {
            return false;
        }
        node_idx++;

        static const std::vector<ggml_op> remaining_ops = { GGML_OP_RESHAPE, GGML_OP_SOFT_MAX, GGML_OP_RESHAPE };

        for (const ggml_op op : remaining_ops) {
            if (node_idx >= n_nodes || nodes[node_idx]->op != op || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
                return false;
            }
            node_idx++;
        }
    }

    // At this point we can check for norm + scale. Everything is now at least valid till the norm
    if (node_idx >= n_nodes) {
        return true;
    }

    if (nodes[node_idx]->op == GGML_OP_RESHAPE) {
        //check RESHAPE->SUM_ROWS->CLAMP->DIV->RESHAPE
        static const std::vector<ggml_op> norm_ops = { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP };

        args.norm = true;
        for (const ggml_op op : norm_ops) {
            if (nodes[node_idx]->op == op && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
                node_idx++;
            } else {
                args.norm = false;
                return true;
            }
        }

        // DIV <- CLAMP, RESHAPE
        if (nodes[node_idx]->op != GGML_OP_DIV || nodes[node_idx]->src[1] != nodes[node_idx - 1] ||
            nodes[node_idx]->src[0] != nodes[node_idx - 3]) {
            args.norm = false;
            return true;
        }
        node_idx++;

        if (nodes[node_idx]->op != GGML_OP_RESHAPE || nodes[node_idx]->src[0] != nodes[node_idx - 1]) {
            args.norm = false;
            return true;
        }

        node_idx++;
    }

    if (nodes[node_idx]->op == GGML_OP_SCALE && nodes[node_idx]->src[0] == nodes[node_idx - 1]) {
        args.scale = true;
    }

    return true;
}

// returns whether the write (out) nodes overwrite the read nodes in operation
static bool ggml_cuda_check_fusion_memory_ranges(const ggml_cgraph * cgraph,
                                                 const int           node_idx,
                                                 const int           node_count,
                                                 const int *         out_nodes,
                                                 const int           out_count,
                                                 const bool          is_topk_moe = false) {
    auto nodes_overlap = [&](const ggml_tensor * a, const ggml_tensor * b) {
        const int64_t a_start = (int64_t) a->data;
        const int64_t a_end   = a_start + ggml_backend_buft_get_alloc_size(a->buffer->buft, a);

        const int64_t b_start = (int64_t) b->data;
        const int64_t b_end   = b_start + ggml_backend_buft_get_alloc_size(b->buffer->buft, b);

        if ((b_start <= a_start && a_start < b_end) || (a_start <= b_start && b_start < a_end)) {
            return true;
        }

        return false;
    };

    bool is_ok = true;
    // one block reads all logits before it writes, so logits may alias the out nodes
    const ggml_tensor * logits_may_alias = nullptr;
    if (is_topk_moe && ggml_nrows(cgraph->nodes[node_idx]) <= TOPK_MOE_ROWS_PER_BLOCK) {
        logits_may_alias = cgraph->nodes[node_idx]->src[0];
    }

    for (int i = 0; i < out_count; ++i) {
        const ggml_tensor * dst = cgraph->nodes[out_nodes[i]];

        for (int j = node_idx; j < node_idx + node_count; ++j) {
            // Loop over all srcs of all nodes in the fusion. If the src overlaps
            // the destination and the src is not an intermediate node that's being
            // elided, then disable fusion.

            for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
                const ggml_tensor * src = cgraph->nodes[j]->src[src_idx];

                if (!src || src->op == GGML_OP_NONE || src == logits_may_alias) {
                    continue;
                }

                if (nodes_overlap(dst, src)) {
                    bool found = false;

                    for (int k = node_idx; k < j; ++k) {
                        if (cgraph->nodes[k] == src) {
                            found = true;
                            break;
                        }
                    }

                    static const bool trace_fuse = getenv("GGML_CUDA_TRACE_FUSE") != nullptr;
                    if (!found && trace_fuse) {
                        fprintf(stderr, "[fuse] %s (%s) at %p+%zu sits under its input %s (%s) at %p+%zu\n", dst->name, ggml_op_name(dst->op),
                                dst->data, ggml_nbytes(dst), src->name, ggml_op_name(src->op), src->data, ggml_nbytes(src));
                    }
                    if (!found) {
                        is_ok = false;
                        break;
                    }
                }
            }
        }
    }

    return is_ok;
}

// The long form spans 2*k + 1 nodes. ggml_can_fuse_subgraph() accepts at most
// 31 nodes, so k <= 15; larger values use the per-operation path.
static constexpr int MOE_WEIGHTED_REDUCTION_MAX_EXPERTS = 15;

struct ggml_cuda_moe_weighted_reduction_match {
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    ggml_tensor *       dst          = nullptr;
    int                 node_count   = 0;
};

static bool ggml_cuda_match_moe_weighted_reduction(
        const ggml_cgraph * cgraph,
        int node_idx,
        ggml_cuda_moe_weighted_reduction_match & match) {
    const ggml_tensor * first = cgraph->nodes[node_idx];
    if (first->op != GGML_OP_MUL || first->type != GGML_TYPE_F32 || !ggml_is_contiguous(first)) {
        return false;
    }

    auto split_mul = [](const ggml_tensor * mul, const ggml_tensor *& full, const ggml_tensor *& broadcast) {
        auto is_weights = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) && tensor->ne[0] == 1 &&
                tensor->ne[1] == mul->ne[1] && tensor->ne[2] == mul->ne[2] && tensor->ne[3] == mul->ne[3];
        };
        auto is_experts = [mul](const ggml_tensor * tensor) {
            return tensor && tensor->type == GGML_TYPE_F32 && ggml_is_contiguous(tensor) &&
                ggml_are_same_shape(tensor, mul);
        };

        if (is_experts(mul->src[0]) && is_weights(mul->src[1])) {
            full      = mul->src[0];
            broadcast = mul->src[1];
            return true;
        }
        if (is_experts(mul->src[1]) && is_weights(mul->src[0])) {
            full      = mul->src[1];
            broadcast = mul->src[0];
            return true;
        }
        return false;
    };

    const ggml_tensor * weighted     = first;
    const ggml_tensor * experts      = nullptr;
    const ggml_tensor * expert_scale = nullptr;
    const ggml_tensor * weights      = nullptr;
    int                 mul_count    = 1;

    // Match both structural forms:
    //   (experts * expert_scale) * router_weight
    //   experts * router_weight
    // The matcher does not depend on the model or quantization type.
    if (node_idx + 1 < cgraph->n_nodes) {
        const ggml_tensor * second = cgraph->nodes[node_idx + 1];
        const ggml_tensor * scaled = nullptr;
        const ggml_tensor * route  = nullptr;
        const ggml_tensor * raw    = nullptr;
        const ggml_tensor * scale  = nullptr;
        if (second->op == GGML_OP_MUL && second->type == GGML_TYPE_F32 && ggml_is_contiguous(second) &&
                split_mul(second, scaled, route) && scaled == first && split_mul(first, raw, scale)) {
            weighted     = second;
            experts      = raw;
            expert_scale = scale;
            weights      = route;
            mul_count    = 2;
        }
    }

    if (experts == nullptr && !split_mul(first, experts, weights)) {
        return false;
    }

    const int     n_expert_used = (int) weighted->ne[1];
    const int64_t n_tokens      = weighted->ne[2] * weighted->ne[3];
    if (n_expert_used < 2 || n_expert_used > MOE_WEIGHTED_REDUCTION_MAX_EXPERTS) {
        return false;
    }

    const int node_count = 2 * n_expert_used + mul_count - 1;
    if (node_idx + node_count > cgraph->n_nodes) {
        return false;
    }

    std::vector<ggml_op> ops(node_count, GGML_OP_VIEW);
    ops[0] = GGML_OP_MUL;
    if (mul_count == 2) {
        ops[1] = GGML_OP_MUL;
    }
    std::vector<const ggml_tensor *> views;
    views.reserve(n_expert_used);
    const ggml_tensor * previous = nullptr;
    int n_adds = 0;
    for (int offset = mul_count; offset < node_count; ++offset) {
        const ggml_tensor * candidate = cgraph->nodes[node_idx + offset];
        ops[offset] = candidate->op;

        if (candidate->op == GGML_OP_VIEW) {
            const int expert = (int) views.size();
            if (expert >= n_expert_used || candidate->src[0] != weighted || candidate->view_src != weighted ||
                    candidate->type != GGML_TYPE_F32 || candidate->ne[0] != weighted->ne[0] ||
                    candidate->ne[1] != n_tokens || candidate->ne[2] != 1 || candidate->ne[3] != 1 ||
                    candidate->nb[0] != weighted->nb[0] || candidate->nb[1] != weighted->nb[2] ||
                    candidate->view_offs != (size_t) expert * weighted->nb[1]) {
                return false;
            }
            views.push_back(candidate);
            continue;
        }

        if (candidate->op != GGML_OP_ADD || views.size() < 2 || n_adds + 1 >= (int) views.size()) {
            return false;
        }
        const ggml_tensor * lhs = n_adds == 0 ? views[0] : previous;
        const ggml_tensor * rhs = views[n_adds + 1];
        if (candidate->src[0] != lhs || candidate->src[1] != rhs || candidate->type != GGML_TYPE_F32) {
            return false;
        }
        previous = candidate;
        ++n_adds;
    }

    if ((int) views.size() != n_expert_used || n_adds != n_expert_used - 1 || previous == nullptr) {
        return false;
    }
    if (!ggml_is_contiguous(previous) || previous->ne[0] != weighted->ne[0] ||
            previous->ne[1] != n_tokens || previous->ne[2] != 1 || previous->ne[3] != 1) {
        return false;
    }

    const int output_idx = node_idx + node_count - 1;
    if (!ggml_can_fuse_subgraph(cgraph, node_idx, node_count, ops.data(), &output_idx, 1)) {
        return false;
    }

    match.experts      = experts;
    match.expert_scale = expert_scale;
    match.weights      = weights;
    match.dst          = cgraph->nodes[output_idx];
    match.node_count   = node_count;
    return true;
}


static bool ggml_cuda_can_fuse(const struct ggml_cgraph *                cgraph,
                               int                                       node_idx,
                               std::initializer_list<enum ggml_op>       ops,
                               std::initializer_list<enum ggml_unary_op> unary_ops) {
#ifndef NDEBUG
    const size_t num_unary = std::count(ops.begin(), ops.end(), GGML_OP_UNARY);
    GGML_ASSERT(unary_ops.size() == num_unary);
#endif

    const auto is_equal = [](const std::initializer_list<enum ggml_op> & list1,
                             const std::initializer_list<enum ggml_op> & list2) {
        return std::equal(list1.begin(), list1.end(), list2.begin(), list2.end());
    };

    std::initializer_list<enum ggml_op> mul_mat_bias_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_MUL_MAT,    GGML_OP_ADD,    GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_id_bias_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_MUL_MAT_ID, GGML_OP_ADD_ID, GGML_OP_GLU };

    std::initializer_list<enum ggml_op> mul_mat_id_glu_ops = { GGML_OP_MUL_MAT_ID, GGML_OP_MUL_MAT_ID, GGML_OP_GLU };
    std::initializer_list<enum ggml_op> mul_mat_glu_ops    = { GGML_OP_MUL_MAT,    GGML_OP_MUL_MAT,    GGML_OP_GLU };

    if ((is_equal(mul_mat_bias_glu_ops, ops) || is_equal(mul_mat_id_bias_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * ffn_gate      = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_gate_bias = cgraph->nodes[node_idx + 1];
        const ggml_tensor * ffn_up        = cgraph->nodes[node_idx + 2];
        const ggml_tensor * ffn_up_bias   = cgraph->nodes[node_idx + 3];
        const ggml_tensor * glu           = cgraph->nodes[node_idx + 4];

        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu, ffn_up_bias, ffn_gate_bias)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if ((is_equal(mul_mat_id_glu_ops, ops) || is_equal(mul_mat_glu_ops, ops)) &&
        ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * ffn_gate = cgraph->nodes[node_idx];
        const ggml_tensor * ffn_up   = cgraph->nodes[node_idx + 1];
        const ggml_tensor * glu      = cgraph->nodes[node_idx + 2];

        static const bool trace_fuse = getenv("GGML_CUDA_TRACE_FUSE") != nullptr;
        if (ggml_cuda_should_fuse_mul_mat(ffn_up, ffn_gate, glu)) {
            int out_nodes[] = { node_idx + 2 };
            const bool ok = ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
            if (trace_fuse && !ok) {
                fprintf(stderr, "[fuse] %s: gate/up/glu not taken, an output sits under an input\n", glu->name);
            }
            return ok;
        } else if (trace_fuse) {
            fprintf(stderr, "[fuse] %s: gate/up/glu not taken by should_fuse_mul_mat\n", glu->name);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_ops          = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE };
    std::initializer_list<enum ggml_op> rms_norm_mul_rope_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_hadamard_set_rows_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rms_norm_mul_rope_hadamard_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 7 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * mm       = cgraph->nodes[node_idx + 4];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 6];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 7];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 1, 3}, {5, 0, 4}, {6, 0, 5}, {7, 0, 6}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_hadamard_set_rows(rope, mm, view, set_rows)) {
            int out_nodes[] = { node_idx + 7 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    std::initializer_list<enum ggml_op> rms_norm_mul_rope_hadamard_ops = { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_RESHAPE, GGML_OP_MUL_MAT };

    if (is_equal(rms_norm_mul_rope_hadamard_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * mm       = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 1, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_hadamard(rope, mm)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 4 })) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 3];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 4];

        if (ggml_check_edges(cgraph, node_idx, {{1, 0, 0}, {2, 0, 1}, {3, 0, 2}, {4, 0, 3}}) &&
            ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope) &&
            ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 4 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (is_equal(rms_norm_mul_rope_ops, ops) && ggml_can_fuse(cgraph, node_idx, ops)) {
        const ggml_tensor * rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor * mul      = cgraph->nodes[node_idx + 1];
        const ggml_tensor * rope     = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rms_norm_mul_rope(rms_norm, mul, rope)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
        return false;
    }

    std::initializer_list<enum ggml_op> rope_set_rows_ops = { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS };

    if (is_equal(rope_set_rows_ops, ops) && ggml_can_fuse_subgraph(cgraph, node_idx, ops, { node_idx + 2 })) {
        const ggml_tensor * rope     = cgraph->nodes[node_idx];
        const ggml_tensor * view     = cgraph->nodes[node_idx + 1];
        const ggml_tensor * set_rows = cgraph->nodes[node_idx + 2];

        if (ggml_cuda_should_fuse_rope_set_rows(rope, view, set_rows)) {
            int out_nodes[] = { node_idx + 2 };
            return ggml_cuda_check_fusion_memory_ranges(cgraph, node_idx, (int)ops.size(), out_nodes, 1);
        }
    }

    if (!ggml_can_fuse(cgraph, node_idx, ops)) {
        return false;
    }

    // rms_norm > mul(weight) > add(residual) > mul(a scalar): a layer's output scale folds into the fused norm
    if (ops.size() == 4 && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL && ops.begin()[2] == GGML_OP_ADD && ops.begin()[3] == GGML_OP_MUL) {
        if (!ggml_can_fuse(cgraph, node_idx, ops)) {
            return false;
        }
        const ggml_tensor * add  = cgraph->nodes[node_idx+2];
        const ggml_tensor * mul2 = cgraph->nodes[node_idx+3];
        const ggml_tensor * sc   = mul2->src[0] == add ? mul2->src[1] : mul2->src[1] == add ? mul2->src[0] : nullptr;
        if (sc == nullptr || ggml_nelements(sc) != 1 || sc->type != GGML_TYPE_F32 || mul2->type != GGML_TYPE_F32 || !ggml_are_same_shape(mul2, add)) {
            return false;
        }
        return ggml_cuda_can_fuse(cgraph, node_idx, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {});
    }
    if ((ops.size() == 2 || ops.size() == 3) && ops.begin()[0] == GGML_OP_RMS_NORM && ops.begin()[1] == GGML_OP_MUL) {
        const ggml_tensor *rms_norm = cgraph->nodes[node_idx];
        const ggml_tensor *mul      = cgraph->nodes[node_idx+1];
        const ggml_tensor *add      = nullptr;

        if (ops.size() == 3 && ops.begin()[2] == GGML_OP_ADD) {
            add = cgraph->nodes[node_idx+2];
        }

        GGML_ASSERT(rms_norm->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(rms_norm->type == GGML_TYPE_F32);

        //rms norm only supports F32
        if (mul->src[0]->type != GGML_TYPE_F32 ||
            mul->src[1]->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32) {
            return false;
        }

        if (add && (add->src[0]->type != GGML_TYPE_F32 ||
            add->src[1]->type != GGML_TYPE_F32 ||
            add->type != GGML_TYPE_F32) ) {
            return false;
        }

        //if rms norm is the B operand, then we don't handle broadcast
        if (rms_norm == mul->src[1] && !ggml_are_same_shape(mul->src[0], rms_norm)) {
            return false;
        }

        //rms_norm kernel assumes contiguous rows
        if (!ggml_is_contiguous_rows(mul->src[0]) || !ggml_is_contiguous_rows(mul->src[1])) {
            return false;
        }

        if (add && (!ggml_is_contiguous(add->src[0]) || !ggml_is_contiguous_rows(add->src[1]))) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_UNARY
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+1];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SSM_CONV && ops.begin()[1] == GGML_OP_ADD
     && ops.begin()[2] == GGML_OP_UNARY && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_SILU) {
        const ggml_tensor * ssm_conv = cgraph->nodes[node_idx];
        const ggml_tensor * add      = cgraph->nodes[node_idx+1];
        const ggml_tensor * silu     = cgraph->nodes[node_idx+2];
        if (ggml_get_unary_op(silu) != unary_ops.begin()[0]) {
            return false;
        }

        if (ssm_conv->type != GGML_TYPE_F32 || add->type != GGML_TYPE_F32 || silu->type != GGML_TYPE_F32) {
            return false;
        }

        // ADD must consume ssm_conv's output and broadcast a 1-D channel-wise bias.
        const ggml_tensor * bias = (add->src[0] == ssm_conv) ? add->src[1] : add->src[0];
        if (bias->type != GGML_TYPE_F32 || !ggml_is_contiguous(bias)) {
            return false;
        }
        if (ggml_nelements(bias) != ssm_conv->ne[0] || bias->ne[0] != ssm_conv->ne[0]) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_MUL
     && unary_ops.size() == 1 && (unary_ops.begin()[0] == GGML_UNARY_OP_SILU || unary_ops.begin()[0] == GGML_UNARY_OP_SIGMOID || unary_ops.begin()[0] == GGML_UNARY_OP_SOFTPLUS)) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * mul   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != unary_ops.begin()[0]) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != mul->type) {
            return false;
        }

        const ggml_tensor * other = (mul->src[0] == unary) ? mul->src[1] : mul->src[0];
        if (other->type != unary->type) {
            return false;
        }
        if (!ggml_is_contiguous_1(other) || !ggml_is_contiguous_1(unary->src[0]) || !ggml_are_same_shape(other, unary)) {
            return false;
        }

        return true;
    }

    if (ops.size() == 2 && ops.begin()[0] == GGML_OP_UNARY && ops.begin()[1] == GGML_OP_SQR
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_RELU) {
        const ggml_tensor * unary = cgraph->nodes[node_idx];
        const ggml_tensor * sqr   = cgraph->nodes[node_idx+1];

        if (ggml_get_unary_op(unary) != GGML_UNARY_OP_RELU) {
            return false;
        }

        if (unary->type != GGML_TYPE_F32 && unary->type != GGML_TYPE_F16) {
            return false;
        }

        if (unary->type != sqr->type) {
            return false;
        }

        if (!ggml_is_contiguous(unary->src[0])) {
            return false;
        }

        return true;
    }

    if (ops.size() == 3 && ops.begin()[0] == GGML_OP_SCALE && ops.begin()[1] == GGML_OP_UNARY && ops.begin()[2] == GGML_OP_SCALE
     && unary_ops.size() == 1 && unary_ops.begin()[0] == GGML_UNARY_OP_TANH) {
        const ggml_tensor *scale  = cgraph->nodes[node_idx];
        const ggml_tensor *tanh   = cgraph->nodes[node_idx+1];
        const ggml_tensor *scale2 = cgraph->nodes[node_idx+2];

        GGML_ASSERT(scale->src[0]->type == GGML_TYPE_F32);
        GGML_ASSERT(scale->type == GGML_TYPE_F32);

        if (ggml_get_unary_op(tanh) != GGML_UNARY_OP_TANH) {
            return false;
        }

        // Check for bias
        if (ggml_get_op_params_f32(scale, 1) != 0.0f || ggml_get_op_params_f32(scale2, 1) != 0.0f) {
            return false;
        }

        return true;
    }

    return false;
}

// try and fuse nodes and return the number of nodes to skip
// GGML_CUDA_FUSE_OFF (decimal or 0x) is a mask of ggml_cuda_fuse_path bits, GGML_CUDA_MMV_EXPERIMENTAL=0 adds the matvec ones
uint64_t ggml_cuda_fuse_off_mask() {
    static const uint64_t mask = [] {
        const char *   e   = getenv("GGML_CUDA_FUSE_OFF");
        const char *   x   = getenv("GGML_CUDA_MMV_EXPERIMENTAL");
        const uint64_t off = e ? std::strtoull(e, nullptr, 0) : 0;
        constexpr uint64_t mmv = GGML_CUDA_FUSE_MMV_PREFETCH | GGML_CUDA_FUSE_MMV_RELEASE | GGML_CUDA_FUSE_NORM_WEIGHT_EARLY |
                                GGML_CUDA_FUSE_MMV_ONE_ROW | GGML_CUDA_FUSE_MMV_LANE_GROUPS | GGML_CUDA_FUSE_MMV_IDS_EARLY;
        return off | (x && atoi(x) == 0 ? mmv : 0);
    }();
    return mask;
}

static bool ggml_cuda_ranges_overlap(const ggml_tensor * a, const ggml_tensor * b) {
    const char * a0 = (const char *) a->data;
    const char * b0 = (const char *) b->data;
    return a0 < b0 + ggml_nbytes(b) && b0 < a0 + ggml_nbytes(a);
}

// get_rows at i whose only reader is a copy into a same-typed, same-shaped destination: gather straight into it
static int ggml_cuda_try_get_rows_cpy(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    ggml_tensor * node = cgraph->nodes[i];
    int j = i + 1;
    while (j < cgraph->n_nodes && (ggml_is_empty(cgraph->nodes[j]) || cgraph->nodes[j]->op == GGML_OP_NONE ||
                                   cgraph->nodes[j]->op == GGML_OP_VIEW || cgraph->nodes[j]->op == GGML_OP_RESHAPE)) {
        j++;
    }
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * cpy = cgraph->nodes[j];
    const ggml_tensor * dst = cpy->src[1];
    if (cpy->op != GGML_OP_CPY || cpy->src[0] != node || dst == nullptr || dst->type != node->type ||
            !ggml_are_same_shape(node, dst) || dst->nb[0] != ggml_type_size(dst->type) ||
            ggml_node_get_use_count(cgraph, i) != 1 || (node->flags & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_cuda_ranges_overlap(dst, node->src[0]) || ggml_cuda_ranges_overlap(dst, node->src[1])) {
        return 0;
    }
    ggml_tensor gather = *node;
    gather.data  = dst->data;
    gather.nb[1] = dst->nb[1];
    gather.nb[2] = dst->nb[2];
    gather.nb[3] = dst->nb[3];
    ggml_cuda_op_get_rows(ctx, &gather);
    return j - i;
}

// a reader of the normed activation that will take a vector path: quantized
// weights through mmvq (experts included), float weights through mmvf on a
// plain 2-D activation
static bool ggml_cuda_norm_fold_reader_ok(const ggml_tensor * mm, const ggml_tensor * act, int cc) {
    if ((mm->op != GGML_OP_MUL_MAT && mm->op != GGML_OP_MUL_MAT_ID) || mm->src[1] != act || mm->src[0] == act || mm->type != GGML_TYPE_F32) {
        return false;
    }
    if (ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        return false;
    }
    const ggml_tensor * src0 = mm->src[0];
    if (src0->buffer == nullptr) {
        return false;
    }
    if (mm->op == GGML_OP_MUL_MAT_ID) {
        return !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_FOLD_Q8_READER) && ggml_is_quantized(src0->type) && ggml_cuda_should_fuse_mul_mat_vec_q(mm);
    }
    if (ggml_is_quantized(src0->type)) {
        return ggml_cuda_should_fuse_mul_mat_vec_q(mm) && ggml_cuda_should_use_mmvq(src0->type, cc, act->ne[1]) &&
               !ggml_cuda_fp4_big_weight(src0, act, mm, cc);
    }
    if (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_FOLD_F32_READER) {
        return false;
    }
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
                                   ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    return (src0->type == GGML_TYPE_F32 || src0->type == GGML_TYPE_F16 || src0->type == GGML_TYPE_BF16) &&
           !bad_padding_clear && act->ne[2] == 1 && act->ne[3] == 1 &&
           ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, act->ne[1]);
}

// rms_norm at i, its weight multiply at i+1: record the fold when every reader
// of the product is a quantized matvec, so neither node needs a launch
static bool ggml_cuda_try_norm_fold(ggml_backend_cuda_context * ctx, const ggml_cgraph * cgraph, int i) {
    // the readers share one quantized copy in stream order; forked streams would not see it
    static const bool multi_stream = getenv("GGML_CUDA_GRAPH_OPT") != nullptr && atoi(getenv("GGML_CUDA_GRAPH_OPT")) == 1;
    if (multi_stream || i + 2 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * norm = cgraph->nodes[i];
    ggml_tensor * mul  = cgraph->nodes[i + 1];
    if (mul->op != GGML_OP_MUL || mul->src[0] != norm) {
        return false;
    }
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul->src[1];
    if (norm->type != GGML_TYPE_F32 || mul->type != GGML_TYPE_F32 || x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32) {
        return false;
    }
    if (w->ne[0] != norm->ne[0] || ggml_nelements(w) != w->ne[0] || !ggml_is_contiguous(w)) {
        return false;
    }
    // readers take the norm's input through the product's strides
    if (!ggml_is_contiguous(x) || !ggml_is_contiguous(mul) || !ggml_are_same_shape(x, mul)) {
        return false;
    }
    if ((norm->flags | mul->flags) & GGML_TENSOR_FLAG_OUTPUT) {
        return false;
    }
    if (ggml_node_get_use_count(cgraph, i) != 1) {
        return false;
    }
    const int n_uses = ggml_node_get_use_count(cgraph, i + 1);
    if (n_uses < 1) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ctx->device].cc;
    // a plain view of the product (the experts read it reshaped) is a reader
    // when everything reading the view qualifies; it then shares the record
    std::vector<const ggml_tensor *> views;
    std::vector<const ggml_tensor *> outs;
    int  found         = 0;
    int  last          = i + 1;
    bool has_q8_reader = false;
    for (int j = i + 2; j < cgraph->n_nodes && found < n_uses; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        bool reads = false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            reads |= t->src[s] == mul;
        }
        if (!reads) {
            continue;
        }
        found++;
        last = j;
        if (ggml_cuda_norm_fold_reader_ok(t, mul, cc)) {
            has_q8_reader |= ggml_is_quantized(t->src[0]->type);
            outs.push_back(t);
            continue;
        }
        const bool plain_view = (t->op == GGML_OP_RESHAPE || t->op == GGML_OP_VIEW) && t->view_src == mul && t->view_offs == 0 &&
                                t->type == mul->type && t->ne[0] == mul->ne[0] && ggml_is_contiguous(t) &&
                                ggml_nelements(t) == ggml_nelements(mul);
        if (!plain_view || (t->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return false;
        }
        const int n_uses_v = ggml_node_get_use_count(cgraph, j);
        int found_v = 0;
        for (int k = j + 1; k < cgraph->n_nodes && found_v < n_uses_v; ++k) {
            const ggml_tensor * u = cgraph->nodes[k];
            bool reads_v = false;
            for (int s = 0; s < GGML_MAX_SRC; ++s) {
                reads_v |= u->src[s] == t;
            }
            if (!reads_v) {
                continue;
            }
            found_v++;
            last = std::max(last, k);
            if (!ggml_cuda_norm_fold_reader_ok(u, t, cc)) {
                return false;
            }
            has_q8_reader |= ggml_is_quantized(u->src[0]->type);
            outs.push_back(u);
        }
        if (found_v != n_uses_v || n_uses_v < 1) {
            return false;
        }
        views.push_back(t);
    }
    if (found != n_uses) {
        return false;
    }
    // a reader may fuse with the GLU, bias or scale node after it and write that output while it reads x
    int end = last;
    for (int k = last + 1; k < cgraph->n_nodes && k <= last + 4; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        bool reads = false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            reads |= t->src[s] && std::find(outs.begin(), outs.end(), t->src[s]) != outs.end();
        }
        if (reads) {
            outs.push_back(t);
            end = k;
        }
    }
    // the readers take x and w where the graph has them read the product: nothing up to the last
    // reader may write there. x and w are freed after the product unless still used or constant
    const ggml_tensor * w_root  = w->view_src ? w->view_src : w;
    const bool          w_const = w_root->buffer && ggml_backend_buffer_get_usage(w_root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    for (int j = i + 2; j <= end; ++j) {
        const ggml_tensor * t = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(t)) {
            continue;
        }
        if (ggml_cuda_tensors_overlap(t, x) || (!w_const && ggml_cuda_tensors_overlap(t, w))) {
            return false;
        }
    }

    auto fold = std::make_shared<ggml_cuda_norm_fold>();
    fold->norm     = norm;
    fold->mul      = mul;
    fold->x        = x;
    fold->w        = w;
    fold->n_rows   = ggml_nrows(mul);
    fold->q8_bytes = fold->n_rows * (GGML_PAD(mul->ne[0], MATRIX_ROW_PADDING) / QK8_1) * sizeof(block_q8_1);
    fold->has_q8_reader = has_q8_reader && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_FOLD_QUANTIZE);
    memcpy(&fold->eps, norm->op_params, sizeof(float));
    ctx->norm_folds[mul] = fold;
    for (const ggml_tensor * v : views) {
        ctx->norm_folds[v] = fold;
    }
    // GGML_CUDA_FOLD_TRACE_ALIAS=1: every other tensor whose bytes lie inside the product's buffer
    static const bool trace_alias = getenv("GGML_CUDA_FOLD_TRACE_ALIAS") != nullptr;
    if (trace_alias) {
        const char * lo = (const char *) mul->data;
        const char * hi = lo + ggml_nbytes(mul);
        for (int j = 0; j < cgraph->n_nodes; ++j) {
            const ggml_tensor * t = cgraph->nodes[j];
            const char * d = (const char *) t->data;
            if (t == mul || t == norm || !d || d >= hi || d + ggml_nbytes(t) <= lo || (t->view_src == mul)) {
                continue;
            }
            fprintf(stderr, "[fold-alias] %s: node %d %s (%s) at +%lld, %s\n", norm->name, j, ggml_op_name(t->op), t->name,
                    (long long) (d - lo), j < i ? "earlier" : "later");
        }
    }
    return true;
}

// GGML_CUDA_FOLD_DIAG=1: a recorded fold is written out at once, so every reader takes
// its usual path and only the split of the residual add from the norm remains to differ.
// GGML_CUDA_FOLD_DIAG2=1: written out as well, but the readers keep the fold path, which
// separates "a reader is wrong" from "something unseen reads the product's buffer"
static bool ggml_cuda_norm_fold_diag() {
    static const bool diag = getenv("GGML_CUDA_FOLD_DIAG") != nullptr;
    return diag;
}

static void ggml_cuda_norm_fold_diag2(ggml_backend_cuda_context & ctx, const ggml_tensor * act) {
    static const bool diag2 = getenv("GGML_CUDA_FOLD_DIAG2") != nullptr;
    // GGML_CUDA_FOLD_POISON=1 (graphs off): the unwritten product filled with NaN, so
    // whatever reads it shows up as the first node whose output turns NaN
    static const bool poison = getenv("GGML_CUDA_FOLD_POISON") != nullptr;
    ggml_cuda_norm_fold * fold = diag2 || poison ? ggml_cuda_norm_fold_for(ctx, act) : nullptr;
    if (!fold) {
        return;
    }
    // either write would clobber the norm input the readers still take
    if (ggml_cuda_tensors_overlap(fold->mul, fold->x) || ggml_cuda_tensors_overlap(fold->norm, fold->x)) {
        GGML_LOG_INFO("%s: %s shares bytes with the norm input, left unwritten\n", __func__, fold->norm->name);
        return;
    }
    if (diag2) {
        ggml_cuda_op_rms_norm_fused(ctx, fold->norm, fold->mul);
    }
    if (poison) {
        CUDA_CHECK(cudaMemsetAsync(fold->mul->data, 0xFF, ggml_nbytes(fold->mul), ctx.stream()));
        if (fold->norm->data != fold->mul->data) {
            CUDA_CHECK(cudaMemsetAsync(fold->norm->data, 0xFF, ggml_nbytes(fold->norm), ctx.stream()));
        }
    }
}

static bool ggml_cuda_norm_fold_poison() {
    static const bool poison = getenv("GGML_CUDA_FOLD_POISON") != nullptr;
    return poison;
}

// after a node ran: does its output hold NaN (the poisoned product leaking through)?
static void ggml_cuda_norm_fold_poison_scan(ggml_backend_cuda_context & ctx, const ggml_tensor * node, int i) {
    if (!ggml_cuda_norm_fold_poison() || ctx.norm_folds.empty() || node->type != GGML_TYPE_F32 || !node->data) {
        return;
    }
    const int64_t n = ggml_nelements(node);
    if (n > (1 << 22)) {
        return;
    }
    std::vector<float> h(n);
    CUDA_CHECK(cudaStreamSynchronize(ctx.stream()));
    CUDA_CHECK(cudaMemcpy(h.data(), node->data, n * sizeof(float), cudaMemcpyDeviceToHost));
    int64_t bad = 0;
    for (int64_t k = 0; k < n; ++k) {
        bad += h[k] != h[k];
    }
    if (bad) {
        fprintf(stderr, "[fold-poison] node %d %s (%s): %lld of %lld NaN; srcs:", i, ggml_op_name(node->op), node->name, (long long) bad, (long long) n);
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
            fprintf(stderr, " %s", node->src[s]->name);
        }
        fprintf(stderr, "\n");
    }
}


// A residual add whose result feeds the rms_norm and weight-multiply after it,
// which is how every transformer block starts. The add's output has two
// readers - the norm here and the next block's residual - so it cannot be
// treated as a dead intermediate the way a fused chain's middle is; the kernel
// writes it as well as the normalised result. Only the contiguous,
// same-shaped, F32 case is taken, which is the residual and nothing else.
static int ggml_cuda_try_add_rms_norm_mul(ggml_backend_cuda_context & ctx, ggml_cgraph * cgraph, int i) {
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * add  = cgraph->nodes[i];
    ggml_tensor * norm = cgraph->nodes[i + 1];
    ggml_tensor * mul  = cgraph->nodes[i + 2];

    if (add->op != GGML_OP_ADD || norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL) {
        return 0;
    }
    if (norm->src[0] != add) {
        return 0;
    }
    if (mul->src[0] != norm && mul->src[1] != norm) {
        return 0;
    }
    // the normalised value must go nowhere but the multiply, or skipping the
    // node would lose a reader
    if (ggml_node_get_use_count(cgraph, i + 1) != 1) {
        return 0;
    }

    ggml_tensor * a = add->src[0];
    ggml_tensor * b = add->src[1];
    ggml_tensor * w = mul->src[0] == norm ? mul->src[1] : mul->src[0];

    for (const ggml_tensor * t : { (const ggml_tensor *) add, (const ggml_tensor *) norm,
                                   (const ggml_tensor *) mul, (const ggml_tensor *) a,
                                   (const ggml_tensor *) b, (const ggml_tensor *) w }) {
        if (t->type != GGML_TYPE_F32 || !ggml_is_contiguous(t)) {
            return 0;
        }
    }
    // no broadcasting: the residual and the sum line up element for element,
    // and the weight is one row reused by every row
    if (!ggml_are_same_shape(add, a) || !ggml_are_same_shape(add, b) ||
        !ggml_are_same_shape(add, norm) || !ggml_are_same_shape(add, mul)) {
        return 0;
    }
    if (w->ne[0] != add->ne[0] || ggml_nrows(w) != 1) {
        return 0;
    }
    if (add->ne[0] > 4096) {
        return 0;
    }

    // the fold comes first: the add alone, the norm and multiply into its readers
    if (!(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_NORM_FOLD) && ggml_cuda_try_norm_fold(&ctx, cgraph, i + 1)) {
        GGML_ASSERT(ggml_cuda_compute_forward(ctx, add));
        if (ggml_cuda_norm_fold_diag()) {
            ggml_cuda_norm_fold_materialize(ctx, mul);
        }
        ggml_cuda_norm_fold_diag2(ctx, mul);
        return 2;
    }

    ggml_cuda_op_add_rms_norm_mul(ctx, add, norm, mul);
    return 2;
}

// the matvec epilogue, the gate/up pairs and the merged gate_up GLU run only with GGML_CUDA_MM_EPI=1
// on by default for one-column matvecs; GGML_CUDA_MM_EPI=0 turns the matvec epilogue and the gate/up pairs off
static bool ggml_cuda_mm_epi_enabled() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_MM_EPI");
        return e == nullptr || atoi(e) != 0;
    }();
    return on;
}

// the lean epilogue at one column: off until measured, GGML_CUDA_LEAN_T1=1 turns it on
static bool ggml_cuda_mm_epi_lean_t1() {
    static const bool on = [] {
        const char * e = getenv("GGML_CUDA_LEAN_T1");
        return e != nullptr && atoi(e) != 0;
    }();
    return on && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_MM_EPILOGUE_T1);
}

// where operand o, read at chain node n, sits over the matvec's output index: its offset must be
// row*s_row + col*s_col + channel*s_channel + sample*s_sample, with n a contiguous reshape of mm
static bool ggml_cuda_mm_epi_arg_of(const ggml_tensor * mm, const ggml_tensor * n, const ggml_tensor * o, ggml_cuda_mm_epi_arg & a) {
    if (o->type != GGML_TYPE_F32 || !ggml_is_contiguous(o)) {
        return false;
    }
    a.data = (const float *) o->data;
    if (ggml_are_same_shape(o, n)) {
        a.s_row     = 1;
        a.s_col     = (int) mm->ne[0];
        a.s_channel = (int) (mm->ne[0]*mm->ne[1]);
        a.s_sample  = (int) (mm->ne[0]*mm->ne[1]*mm->ne[2]);
        return ggml_nelements(o) < INT_MAX;
    }
    if (n->ne[0] != mm->ne[0]) {
        return false;
    }
    int64_t s[GGML_MAX_DIMS];
    for (int d = 0; d < GGML_MAX_DIMS; ++d) {
        s[d] = o->ne[d] == 1 ? 0 : (int64_t) (o->nb[d] / sizeof(float));
    }
    // output column q of the product is column q of n too; the operand's offset must be linear in (col, channel, sample)
    const int64_t n1 = n->ne[1];
    const int64_t n2 = n->ne[2];
    auto off = [&](const int64_t q) {
        return (q % n1)*s[1] + ((q / n1) % n2)*s[2] + (q / (n1*n2))*s[3];
    };
    const int64_t ne1 = mm->ne[1];
    const int64_t ne2 = mm->ne[2];
    const int64_t ne3 = mm->ne[3];
    if (ne1*ne2*ne3 > 65536) {
        return false;
    }
    const int64_t s_col     = ne1 > 1 ? off(1)       : 0;
    const int64_t s_channel = ne2 > 1 ? off(ne1)     : 0;
    const int64_t s_sample  = ne3 > 1 ? off(ne1*ne2) : 0;
    for (int64_t i3 = 0; i3 < ne3; ++i3) {
        for (int64_t i2 = 0; i2 < ne2; ++i2) {
            for (int64_t i1 = 0; i1 < ne1; ++i1) {
                if (off(i1 + ne1*(i2 + ne2*i3)) != i1*s_col + i2*s_channel + i3*s_sample) {
                    return false;
                }
            }
        }
    }
    if (ggml_nelements(o) >= INT_MAX) {
        return false;
    }
    a.s_row     = (int) s[0];
    a.s_col     = (int) s_col;
    a.s_channel = (int) s_channel;
    a.s_sample  = (int) s_sample;
    return true;
}

// the elementwise ops after the matvec at node i as an epilogue over its output: fills epi and the tensors it reads,
// and returns the last node taken (i when none)
static int ggml_cuda_mm_epi_collect(const ggml_cgraph * cgraph, const int i, ggml_cuda_mm_epilogue & epi, const ggml_tensor ** args) {
    const ggml_tensor * mm   = cgraph->nodes[i];
    const ggml_tensor * src1 = mm->src[1];
    const ggml_tensor * cur = mm;
    int cur_idx = i;
    int last    = i;
    for (int j = i + 1; j < cgraph->n_nodes && epi.n_steps < GGML_CUDA_MM_EPI_MAX_STEPS; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        bool reads_cur = false;
        for (int s = 0; s < GGML_MAX_SRC; ++s) {
            reads_cur = reads_cur || n->src[s] == cur;
        }
        if (!reads_cur) {
            if (ggml_cuda_is_view_or_noop(n) || ggml_is_empty(n)) {
                continue;
            }
            break;
        }
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 || ggml_is_empty(n) ||
            (cur->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, cur_idx) != 1) {
            break;
        }
        if (n->op == GGML_OP_RESHAPE || n->op == GGML_OP_VIEW) {
            if (n->src[0] != cur || n->view_offs != 0 || n->data != cur->data || !ggml_is_contiguous(n) ||
                ggml_nelements(n) != ggml_nelements(mm)) {
                break;
            }
            cur     = n;
            cur_idx = j;
            continue;
        }
        int op = ggml_cuda_chain_op_of(n);
        if (op == 0 || !ggml_are_same_shape(n, cur)) {
            break;
        }
        // a router's gating belongs to the top-k fusion, a gate/up product to the GLU paths
        if (n->op == GGML_OP_UNARY) {
            ggml_cuda_topk_moe_args topk_args;
            if (ggml_cuda_topk_moe_fusion(cgraph, j, topk_args)) {
                break;
            }
        }
        const ggml_tensor * other = nullptr;
        if (n->src[0] == cur) {
            other = n->src[1];
        } else if (n->op == GGML_OP_ADD || n->op == GGML_OP_MUL) {
            other = n->src[0];
        } else if (n->op == GGML_OP_SUB) {
            op    = GGML_CUDA_CHAIN_RSUB;
            other = n->src[0];
        } else if (n->op == GGML_OP_DIV) {
            op    = GGML_CUDA_CHAIN_RDIV;
            other = n->src[0];
        } else {
            break;
        }
        int arg = -1;
        if (other != nullptr) {
            if (other == cur || (other->op == GGML_OP_MUL_MAT && other->src[1] == src1)) {
                break;
            }
            for (int k = 0; k < epi.n_args; ++k) {
                if (args[k] == other) {
                    arg = k;
                }
            }
            if (arg < 0) {
                if (epi.n_args == GGML_CUDA_MM_EPI_MAX_ARGS || !ggml_cuda_mm_epi_arg_of(mm, n, other, epi.arg[epi.n_args])) {
                    break;
                }
                arg = epi.n_args;
                args[epi.n_args++] = other;
            }
        }
        ggml_cuda_chain_step & st = epi.step[epi.n_steps++];
        st.op  = op;
        st.arg = arg;
        st.p0  = 0.0f;
        st.p1  = 0.0f;
        if (n->op == GGML_OP_SCALE || n->op == GGML_OP_CLAMP) {
            memcpy(&st.p0, (const float *) n->op_params + 0, sizeof(float));
            memcpy(&st.p1, (const float *) n->op_params + 1, sizeof(float));
        }
        cur     = n;
        cur_idx = j;
        last    = j;
    }
    return last;
}

// A product over several columns that ggml_cuda_mul_mat sends to mul_mat_q, with an elementwise tail of no operands
// (the logits' softcap): the tail goes into the kernel's store, the separate pass over the result is gone (a verify's
// 128 rows of 262k logits read and written once more). GGML_CUDA_MMQ_EPI=0 off.
static int ggml_cuda_try_mmq_epilogue(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i, const bool dry_run) {
    static const bool off = getenv("GGML_CUDA_MMQ_EPI") && atoi(getenv("GGML_CUDA_MMQ_EPI")) == 0;
    const ggml_tensor * mm   = cgraph->nodes[i];
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    if (off || !ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_NVFP4 || src0->type == GGML_TYPE_MXFP4 ||
        mm->type != GGML_TYPE_F32 || src1->type != GGML_TYPE_F32 || (mm->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        !ggml_is_contiguous(mm) || mm->ne[2] != 1 || mm->ne[3] != 1 || src0->ne[2] != 1 || src0->ne[3] != 1 ||
        src0->buffer == nullptr || ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        return 0;
    }
    // the kernel ggml_cuda_mul_mat would take, in its order
    const int     cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int     warp_size = ggml_cuda_info().devices[ctx.device].warp_size;
    const int64_t ne11      = src1->ne[1];
    const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
        ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
    if (bad_padding_clear || ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, ne11) ||
        ggml_cuda_should_use_mmf(src0->type, cc, warp_size, src0->ne, src0->nb, ne11, /*mul_mat_id =*/ false) ||
        ggml_cuda_should_use_mmvq(src0->type, cc, ne11) || ggml_cuda_small_gemm_applies(src0, src1, mm) ||
        ((ggml_mul_mat_get_n_seqs(mm) > 1 || ggml_cuda_fp4_gemm_forced()) && ggml_cuda_fp4_gemm_supported(src0, src1, mm, cc)) ||
        !ggml_cuda_should_use_mmq(src0->type, cc, ne11, /*n_experts =*/ 0)) {
        return 0;
    }

    ggml_cuda_mm_epilogue epi{};
    const ggml_tensor * args[GGML_CUDA_MM_EPI_MAX_ARGS] = {};
    const int last = ggml_cuda_mm_epi_collect(cgraph, i, epi, args);
    // the softcap only: scale, tanh, scale
    if (epi.n_steps != 3 || epi.n_args > 0 || epi.step[0].op != GGML_CUDA_CHAIN_SCALE || epi.step[2].op != GGML_CUDA_CHAIN_SCALE ||
        epi.step[1].op != GGML_CUDA_CHAIN_UNARY + GGML_UNARY_OP_TANH) {
        return 0;
    }
    // the weight is read by other blocks while a tile is stored
    ggml_tensor * out = cgraph->nodes[last];
    const char * ob = (const char *) out->data;
    if ((const char *) src0->data < ob + ggml_nbytes(out) && ob < (const char *) src0->data + ggml_nbytes(src0)) {
        return 0;
    }
    if (dry_run) {
        return last - i;
    }

    ggml_cuda_norm_fold_materialize(ctx, src1);
    mmq_epi me{};
    me.n  = 1;
    me.a0 = epi.step[0].p0;
    me.a1 = epi.step[0].p1;
    me.b0 = epi.step[2].p0;
    me.b1 = epi.step[2].p1;
    ggml_tensor dst = *mm;
    dst.data = out->data;
    ggml_tensor * dsts[1] = { &dst };
    ggml_cuda_mul_mat_q_group(ctx, &src0, 1, src1, dsts, &me);
    return last - i;
}

static int ggml_cuda_try_mm_epilogue(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i, const bool dry_run = false) {
    const ggml_tensor * mm = cgraph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT || !ggml_cuda_mm_epi_enabled() || (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_MM_EPILOGUE)) {
        return 0;
    }
    if (mm->ne[1] > 1) {
        return ggml_cuda_try_mmq_epilogue(ctx, cgraph, i, dry_run);
    }
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    // a verify of several columns already has each row's loads well covered, and the epilogue's own cost it
    if (src0->type == GGML_TYPE_NVFP4 || mm->type != GGML_TYPE_F32 || (mm->flags & GGML_TENSOR_FLAG_OUTPUT) ||
        !ggml_is_contiguous(mm) || mm->ne[1] != 1 || src0->buffer == nullptr ||
        ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        return 0;
    }
    const int  cc    = ggml_cuda_info().devices[ctx.device].cc;
    const bool quant = ggml_is_quantized(src0->type);
    if (quant) {
        if ((mm->ne[1] == 1 && !ggml_cuda_mm_epi_lean_t1()) ||
            !ggml_cuda_should_fuse_mul_mat_vec_q(mm) || !ggml_cuda_should_use_mmvq(src0->type, cc, src1->ne[1]) ||
            ggml_cuda_fp4_big_weight(src0, src1, mm, cc)) {
            return 0;
        }
    } else {
        const bool bad_padding_clear = ggml_backend_buffer_get_usage(src0->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            ggml_nbytes(src0) != ggml_backend_buffer_get_alloc_size(src0->buffer, src0) && src0->view_src;
        if (mm->ne[1] != 1 || bad_padding_clear || src1->type != GGML_TYPE_F32 || !ggml_cuda_should_fuse_mul_mat_vec_f(mm)) {
            return 0;
        }
    }

    ggml_cuda_mm_epilogue epi{};
    const ggml_tensor * args[GGML_CUDA_MM_EPI_MAX_ARGS] = {};
    const int last = ggml_cuda_mm_epi_collect(cgraph, i, epi, args);
    if (epi.n_steps == 0) {
        return 0;
    }
    // an operand may be the output itself (an in-place residual, read and written by one thread) but must not
    // overlap it otherwise
    const ggml_tensor * out = cgraph->nodes[last];
    const char * ob = (const char *) out->data;
    const size_t on = ggml_nbytes(out);
    for (int k = 0; k < epi.n_args; ++k) {
        const ggml_tensor * t = args[k];
        const char * tb = (const char *) t->data;
        const bool exact = tb == ob && ggml_are_same_shape(t, out) && ggml_are_same_stride(t, out);
        if (!exact && tb < ob + on && ob < tb + ggml_nbytes(t)) {
            return 0;
        }
    }

    // the output may reuse memory freed after the product; the weight, and the float kernel's activation, are
    // still read by other blocks while it is stored
    ggml_cuda_norm_fold * fold = ggml_cuda_norm_fold_for(ctx, src1);
    for (const ggml_tensor * t : { src0, quant ? nullptr : (fold ? fold->x : src1) }) {
        if (t != nullptr && (const char *) t->data < ob + on && ob < (const char *) t->data + ggml_nbytes(t)) {
            return 0;
        }
    }
    if (dry_run) {
        return last - i;
    }

    ggml_tensor dst = *mm;
    dst.data = out->data;
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.epi = &epi;
    if (quant) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, &dst, &fusion, fold);
    } else {
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, &dst, &fusion, fold);
    }
    ggml_cuda_norm_fold_check_at(ctx, src1);
    return last - i;
}

// gate and up the fused gate kernel declines: the gate into scratch, then the up with its bias and the GLU as epilogue
static bool ggml_cuda_try_mm_glu_pair(ggml_backend_cuda_context & ctx, const ggml_tensor * gate_mm, const ggml_tensor * up_mm,
        const ggml_tensor * gate_b, const ggml_tensor * up_b, ggml_tensor * glu) {
    if (!ggml_cuda_mm_epi_enabled() || (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_GATE_UP)) {
        return false;
    }
    if (gate_mm->op != GGML_OP_MUL_MAT || up_mm->op != GGML_OP_MUL_MAT || gate_mm->src[1] != up_mm->src[1] ||
        !ggml_are_same_shape(gate_mm, up_mm) || !ggml_are_same_shape(up_mm, glu) || !ggml_is_contiguous(glu) ||
        glu->type != GGML_TYPE_F32 || ggml_get_op_params_i32(glu, 1) != 0) {
        return false;
    }
    switch (ggml_get_glu_op(glu)) {
        case GGML_GLU_OP_REGLU: case GGML_GLU_OP_GEGLU: case GGML_GLU_OP_SWIGLU: case GGML_GLU_OP_SWIGLU_OAI:
        case GGML_GLU_OP_GEGLU_ERF: case GGML_GLU_OP_GEGLU_QUICK:
            break;
        default:
            return false;
    }
    const ggml_tensor * src1 = up_mm->src[1];
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    for (const ggml_tensor * mm : { gate_mm, up_mm }) {
        const ggml_tensor * src0 = mm->src[0];
        if (!ggml_is_quantized(src0->type) || src0->type == GGML_TYPE_NVFP4 || src0->buffer == nullptr ||
            !ggml_is_contiguous(mm) || mm->ne[1] != 1 ||
            ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD ||
            !ggml_cuda_should_fuse_mul_mat_vec_q(mm) || !ggml_cuda_should_use_mmvq(src0->type, cc, src1->ne[1])) {
            return false;
        }
    }
    if (up_mm->ne[1] == 1 && !ggml_cuda_mm_epi_lean_t1()) {
        return false;
    }
    ggml_cuda_mm_epi_arg gb{};
    ggml_cuda_mm_epi_arg ub{};
    if ((gate_b && !ggml_cuda_mm_epi_arg_of(gate_mm, gate_mm, gate_b, gb)) || (up_b && !ggml_cuda_mm_epi_arg_of(up_mm, up_mm, up_b, ub))) {
        return false;
    }
    // a bias or weight under the output would be overwritten while other blocks still read it
    const char * ob = (const char *) glu->data;
    const size_t on = ggml_nbytes(glu);
    for (const ggml_tensor * t : std::initializer_list<const ggml_tensor *>{ gate_b, up_b, gate_mm->src[0], up_mm->src[0] }) {
        if (t && (const char *) t->data < ob + on && ob < (const char *) t->data + ggml_nbytes(t)) {
            return false;
        }
    }

    // the pool is freed in reverse order: the fold's buffer, which outlives this node, is taken first
    ggml_cuda_norm_fold * fold = ggml_cuda_norm_fold_for(ctx, src1);
    ggml_cuda_pool_alloc<char> q8(ctx.pool());
    const void * q8_pre = nullptr;
    if (fold != nullptr) {
        ggml_cuda_norm_fold_reserve(ctx, *fold);
    } else {
        const int64_t ne10_padded = GGML_PAD(src1->ne[0], MATRIX_ROW_PADDING);
        const size_t  ts          = ggml_type_size(src1->type);
        q8.alloc(src1->ne[3]*src1->ne[2]*src1->ne[1]*ne10_padded*sizeof(block_q8_1)/QK8_1);
        quantize_row_q8_1_cuda((const float *) src1->data, nullptr, q8.get(), up_mm->src[0]->type, src1->ne[0],
            src1->nb[1]/ts, src1->nb[2]/ts, src1->nb[3]/ts, ne10_padded, src1->ne[1], src1->ne[2], src1->ne[3], ctx.stream());
        q8_pre = q8.get();
    }

    ggml_cuda_pool_alloc<float> g(ctx.pool(), ggml_nelements(gate_mm));
    {
        ggml_tensor dst = *gate_mm;
        dst.data = g.get();
        ggml_cuda_mm_epilogue epi{};
        ggml_cuda_mm_fusion_args_host fusion{};
        if (gate_b) {
            epi.arg[0]  = gb;
            epi.n_args  = 1;
            epi.step[0] = { GGML_CUDA_CHAIN_ADD, 0, 0.0f, 0.0f };
            epi.n_steps = 1;
            fusion.epi  = &epi;
        }
        ggml_cuda_mul_mat_vec_q(ctx, gate_mm->src[0], src1, nullptr, &dst, gate_b ? &fusion : nullptr, fold, q8_pre);
    }
    {
        ggml_tensor dst = *up_mm;
        dst.data = glu->data;
        ggml_cuda_mm_epilogue epi{};
        if (up_b) {
            epi.arg[epi.n_args] = ub;
            epi.step[epi.n_steps++] = { GGML_CUDA_CHAIN_ADD, epi.n_args++, 0.0f, 0.0f };
        }
        ggml_cuda_mm_epi_arg & ga = epi.arg[epi.n_args];
        ga.data      = g.get();
        ga.s_row     = 1;
        ga.s_col     = (int) up_mm->ne[0];
        ga.s_channel = (int) (up_mm->ne[0]*up_mm->ne[1]);
        ga.s_sample  = (int) (up_mm->ne[0]*up_mm->ne[1]*up_mm->ne[2]);
        epi.step[epi.n_steps++] = { GGML_CUDA_CHAIN_GLU + (int) ggml_get_glu_op(glu), epi.n_args++,
                                    ggml_get_op_params_f32(glu, 2), ggml_get_op_params_f32(glu, 3) };
        ggml_cuda_mm_fusion_args_host fusion{};
        fusion.epi = &epi;
        ggml_cuda_mul_mat_vec_q(ctx, up_mm->src[0], src1, nullptr, &dst, &fusion, fold, q8_pre);
    }
    ggml_cuda_norm_fold_check_at(ctx, src1);
    return true;
}

// a merged gate_up weight and a GLU over the two halves of its product: the fused gate kernel over the two row halves
static int ggml_cuda_try_merged_glu(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    const ggml_tensor * gu = cgraph->nodes[i];
    if ((gu->op != GGML_OP_MUL_MAT && gu->op != GGML_OP_MUL_MAT_ID) || !ggml_cuda_mm_epi_enabled() || (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_GATE_UP) ||
        (gu->flags & GGML_TENSOR_FLAG_OUTPUT) || gu->type != GGML_TYPE_F32 || !ggml_is_contiguous(gu) || gu->ne[0] % 2 != 0) {
        return 0;
    }
    const ggml_tensor * w = gu->src[0];
    // a half-row shadow of a weight in a compute buffer would take the padding clear meant for the whole tensor
    if (!ggml_is_quantized(w->type) || w->type == GGML_TYPE_NVFP4 || w->buffer == nullptr ||
        ggml_backend_buffer_get_usage(w->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
        ggml_get_op_params_i32(gu, 1) == GGML_HINT_SRC0_IS_HADAMARD || !ggml_cuda_should_fuse_mul_mat_vec_q(gu)) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    if (gu->op == GGML_OP_MUL_MAT && !ggml_cuda_should_use_mmvq(w->type, cc, gu->src[1]->ne[1])) {
        return 0;
    }
    const int64_t f = gu->ne[0] / 2;

    const ggml_tensor * glu = nullptr;
    int glu_idx  = -1;
    int gate_half = 0;
    int n_views   = 0;
    const ggml_tensor * views[2] = {};
    for (int j = i + 1; j < cgraph->n_nodes && j <= i + 3; ++j) {
        const ggml_tensor * n = cgraph->nodes[j];
        if ((n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0 && n->op != GGML_OP_VIEW) {
            return 0;
        }
        if (n->op == GGML_OP_VIEW && n->src[0] == gu && n_views < 2) {
            views[n_views++] = n;
            continue;
        }
        if (n->op == GGML_OP_GLU) {
            glu     = n;
            glu_idx = j;
        }
        break;
    }
    if (glu == nullptr || glu->type != GGML_TYPE_F32 || !ggml_is_contiguous(glu)) {
        return 0;
    }
    if (n_views == 0) {
        if (glu->src[0] != gu || glu->src[1] != nullptr || ggml_node_get_use_count(cgraph, i) != 1) {
            return 0;
        }
        gate_half = ggml_get_op_params_i32(glu, 1) ? 1 : 0;
    } else {
        if (n_views != 2 || ggml_node_get_use_count(cgraph, i) != 2) {
            return 0;
        }
        for (const ggml_tensor * v : views) {
            if ((v != glu->src[0] && v != glu->src[1]) || v->ne[0] != f || v->nb[1] != gu->nb[1] || v->nb[2] != gu->nb[2] ||
                v->nb[3] != gu->nb[3] || !ggml_are_same_shape(v, glu) || (v->view_offs != 0 && v->view_offs != (size_t) f*gu->nb[0])) {
                return 0;
            }
        }
        if (glu->src[0]->view_offs == glu->src[1]->view_offs) {
            return 0;
        }
        gate_half = glu->src[0]->view_offs == 0 ? 0 : 1;
        for (int j = i + 1; j < glu_idx; ++j) {
            if (ggml_node_get_use_count(cgraph, j) != 1 || (cgraph->nodes[j]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
                return 0;
            }
        }
    }
    if (!ggml_cuda_fused_glu_ok(glu)) {
        return 0;
    }
    // the weight is read by other blocks while the output is stored
    const char * ob = (const char *) glu->data;
    const char * wb = (const char *) w->data;
    if (wb < ob + ggml_nbytes(glu) && ob < wb + ggml_nbytes(w)) {
        return 0;
    }

    ggml_tensor gate_w = *w;
    ggml_tensor up_w   = *w;
    gate_w.ne[1] = f;
    up_w.ne[1]   = f;
    gate_w.data = (char *) w->data + gate_half*f*w->nb[1];
    up_w.data   = (char *) w->data + (1 - gate_half)*f*w->nb[1];

    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.gate      = &gate_w;
    fusion.glu_op    = ggml_get_glu_op(glu);
    fusion.glu_limit = ggml_get_op_params_f32(glu, 3); // SWIGLU_CLAMP's limit; the other fused ops ignore it
    ggml_cuda_mul_mat_vec_q(ctx, &up_w, gu->src[1], gu->src[2], const_cast<ggml_tensor *>(glu), &fusion,
        ggml_cuda_norm_fold_for(ctx, gu->src[1]));
    ggml_cuda_norm_fold_check_at(ctx, gu->src[1]);
    return glu_idx - i;
}

// {gate, up, GLU} or {gate, bias, up, bias, GLU} over one activation that the fused gate kernel declined
static int ggml_cuda_try_glu_pair_at(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if (cgraph->nodes[i]->op != GGML_OP_MUL_MAT || !ggml_cuda_mm_epi_enabled() || (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_GATE_UP)) {
        return 0;
    }
    for (const bool with_bias : { false, true }) {
        const int n_ops = with_bias ? 5 : 3;
        const ggml_op ops_plain[] = { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU };
        const ggml_op ops_bias[]  = { GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_MUL_MAT, GGML_OP_ADD, GGML_OP_GLU };
        const int out_nodes[] = { i + n_ops - 1 };
        if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, with_bias ? ops_bias : ops_plain, out_nodes, 1)) {
            continue;
        }
        ggml_tensor * glu = cgraph->nodes[i + n_ops - 1];
        const ggml_tensor * gate_mm = nullptr;
        const ggml_tensor * up_mm   = nullptr;
        const ggml_tensor * gate_b  = nullptr;
        const ggml_tensor * up_b    = nullptr;
        if (!with_bias) {
            gate_mm = glu->src[0];
            up_mm   = glu->src[1];
        } else {
            // a bias node reads its matvec and one other tensor
            auto split = [&](const ggml_tensor * add, const ggml_tensor *& mm, const ggml_tensor *& b) {
                if (add == nullptr || add->op != GGML_OP_ADD) {
                    return false;
                }
                const bool lhs = add->src[0]->op == GGML_OP_MUL_MAT && (add->src[0] == cgraph->nodes[i] || add->src[0] == cgraph->nodes[i + 2]);
                mm = lhs ? add->src[0] : add->src[1];
                b  = lhs ? add->src[1] : add->src[0];
                // the fused path never writes the nodes in its range, so a bias must come from outside it
                for (int k = i; k < i + n_ops; ++k) {
                    if (b == cgraph->nodes[k] || b->view_src == cgraph->nodes[k]) {
                        return false;
                    }
                }
                return (mm == cgraph->nodes[i] || mm == cgraph->nodes[i + 2]) && ggml_are_same_shape(add, mm);
            };
            if (!split(glu->src[0], gate_mm, gate_b) || !split(glu->src[1], up_mm, up_b)) {
                continue;
            }
        }
        if (gate_mm == nullptr || up_mm == nullptr || gate_mm == up_mm ||
            (gate_mm != cgraph->nodes[i] && gate_mm != cgraph->nodes[i + (with_bias ? 2 : 1)])) {
            continue;
        }
        if (ggml_cuda_try_mm_glu_pair(ctx, gate_mm, up_mm, gate_b, up_b, glu)) {
            return n_ops - 1;
        }
    }
    return 0;
}

static bool ggml_cuda_mmvq_pair_member_ok(const ggml_tensor * mm, int cc) {
    if (mm->op != GGML_OP_MUL_MAT || !(mm->flags & GGML_TENSOR_FLAG_COMPUTE) || ggml_is_empty(mm) ||
            ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD) {
        return false;
    }
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    if (src1->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32 ||
            src1->nb[0] != sizeof(float) || !ggml_is_contiguous(src0) || !ggml_is_contiguous(mm) ||
            src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 || src1->ne[3] != 1) {
        return false;
    }
    if (src0->buffer == nullptr || ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS) {
        return false;
    }
    const int64_t ncols = src1->ne[1];
    return ncols >= 1 && ncols <= MMVQ_MAX_BATCH_SIZE && ggml_cuda_should_fuse_mul_mat_vec_q(mm) &&
           ggml_cuda_should_use_mmvq(src0->type, cc, ncols);
}

// two quantized matvecs over one activation, one after the other (views between them do nothing): one launch.
static int ggml_cuda_try_mmvq_pair(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool multi_stream = getenv("GGML_CUDA_GRAPH_OPT") != nullptr && atoi(getenv("GGML_CUDA_GRAPH_OPT")) == 1;
    if (multi_stream || (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_MMV_PAIRS)) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_tensor * a = cgraph->nodes[i];
    if (!ggml_cuda_mmvq_pair_member_ok(a, cc)) {
        return 0;
    }
    // up to four in a row over the same activation (q, k and v), each a wave of its own blocks in one launch
    const ggml_tensor * nodes[4] = { a };
    int n    = 1;
    int last = i;
    for (int j = i + 1; j < cgraph->n_nodes && n < 4; ++j) {
        const ggml_tensor * b = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(b)) {
            continue;
        }
        if (b->op != GGML_OP_MUL_MAT || b->src[1] != a->src[1] || !ggml_cuda_mmvq_pair_member_ok(b, cc) ||
                !ggml_cuda_mmvq_pair_ok(a, b, ctx.device)) {
            break;
        }
        // b's own bias, scale or gate/up fusion saves as much and keeps its epilogue
        if (ggml_can_fuse(cgraph, j, { GGML_OP_MUL_MAT, GGML_OP_ADD }) ||
                ggml_can_fuse(cgraph, j, { GGML_OP_MUL_MAT, GGML_OP_MUL }) ||
                ggml_cuda_can_fuse(cgraph, j, { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT, GGML_OP_GLU }, {}) ||
                ggml_cuda_try_mm_epilogue(ctx, cgraph, j, true) > 0) {
            break;
        }
        nodes[n++] = b;
        last = j;
    }
    if (n < 2) {
        return 0;
    }
    ggml_cuda_mul_mat_vec_q_group(ctx, nodes, n, ggml_cuda_norm_fold_for(ctx, a->src[1]));
    ggml_cuda_norm_fold_check_at(ctx, a->src[1]);
    return last - i;
}

// consecutive copies (views between them are nothing to launch) of 4-byte elements, where none reads or writes what an
// earlier one writes and none writes what an earlier one reads, run as one launch: the recurrent cache's per-token
// snapshot copies are four of these a layer. Returns the nodes the run covers past i.
static constexpr uint64_t GGML_CUDA_FUSE_CPY_RUN = 1ull << 40;

static int ggml_cuda_try_cpy_run(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    if (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_CPY_RUN) {
        return 0;
    }
    auto fits = [](const ggml_tensor * n) {
        const ggml_tensor * src = n->src[0];
        const ggml_tensor * dst = n->src[1];
        return n->op == GGML_OP_CPY && !ggml_is_empty(n) && src && dst && src->type == dst->type &&
               ggml_blck_size(src->type) == 1 && ggml_type_size(src->type) == 4 &&
               ggml_nelements(src) == ggml_nelements(dst) && ggml_nelements(src) < INT32_MAX && src->data && dst->data;
    };
    if (!fits(cgraph->nodes[i])) {
        return 0;
    }

    const ggml_tensor * nodes[GGML_CUDA_CPY_RUN_MAX];
    int n    = 0;
    int last = i;
    for (int j = i; j < cgraph->n_nodes && n < GGML_CUDA_CPY_RUN_MAX; ++j) {
        const ggml_tensor * node = cgraph->nodes[j];
        if (j > i && (ggml_cuda_is_view_or_noop(node) || ggml_is_empty(node))) {
            continue;
        }
        if (!fits(node)) {
            break;
        }
        bool independent = true;
        for (int k = 0; k < n && independent; ++k) {
            independent = !ggml_cuda_tensors_overlap(node->src[0], nodes[k]->src[1]) &&
                          !ggml_cuda_tensors_overlap(node->src[1], nodes[k]->src[1]) &&
                          !ggml_cuda_tensors_overlap(node->src[1], nodes[k]->src[0]);
        }
        if (!independent) {
            break;
        }
        nodes[n++] = node;
        last = j;
    }
    if (n < 2) {
        return 0;
    }

    // windows of one source's rows into contiguous destinations (the conv state's per-token snapshots): a row read once
    {
        const ggml_tensor * s0 = nodes[0]->src[0];
        bool win = s0->nb[0] == 4 && s0->ne[3] == 1 && s0->ne[0] <= 16 && s0->ne[1]*s0->ne[2] < INT32_MAX;
        const char * base = (const char *) s0->data;
        for (int k = 0; k < n && win; ++k) {
            const ggml_tensor * s = nodes[k]->src[0];
            win = s->ne[0] == s0->ne[0] && s->ne[1] == s0->ne[1] && s->ne[2] == s0->ne[2] && s->ne[3] == 1 &&
                  s->nb[0] == 4 && s->nb[1] == s0->nb[1] && s->nb[2] == s0->nb[2] && ggml_is_contiguous(nodes[k]->src[1]);
            base = std::min(base, (const char *) s->data);
        }
        for (int k = 0; k < n && win; ++k) {
            const int64_t d = (const char *) nodes[k]->src[0]->data - base;
            win = d % 4 == 0 && d/4 + nodes[k]->src[0]->ne[0] <= s0->nb[1]/4;
        }
        static const bool win_off = getenv("GGML_CUDA_CPY_WINDOWS") && atoi(getenv("GGML_CUDA_CPY_WINDOWS")) == 0;
        if (win && !win_off) {
            ggml_cuda_cpy_windows w{};
            w.base   = base;
            w.nb1    = s0->nb[1];
            w.nb2    = s0->nb[2];
            w.fd_ne1 = init_fastdiv_values(s0->ne[1]);
            w.n_rows = (uint32_t) (s0->ne[1]*s0->ne[2]);
            w.ne0    = (int) s0->ne[0];
            w.n      = n;
            for (int k = 0; k < n; ++k) {
                w.off[k] = (int) (((const char *) nodes[k]->src[0]->data - base)/4);
                w.dst[k] = (char *) nodes[k]->src[1]->data;
            }
            ggml_cuda_cpy_windows_4b(w, ctx.stream());
            return last - i;
        }
    }

    ggml_cuda_cpy_run run{};
    run.n = n;
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * src = nodes[k]->src[0];
        const ggml_tensor * dst = nodes[k]->src[1];
        ggml_cuda_cpy_run_item & it = run.item[k];
        it.src = (const char *) src->data;
        it.dst = (char *) dst->data;
        it.ne  = ggml_nelements(src);
        it.fd0[0] = init_fastdiv_values(src->ne[0]);
        it.fd0[1] = init_fastdiv_values(src->ne[0]*src->ne[1]);
        it.fd0[2] = init_fastdiv_values(src->ne[0]*src->ne[1]*src->ne[2]);
        it.fd1[0] = init_fastdiv_values(dst->ne[0]);
        it.fd1[1] = init_fastdiv_values(dst->ne[0]*dst->ne[1]);
        it.fd1[2] = init_fastdiv_values(dst->ne[0]*dst->ne[1]*dst->ne[2]);
        for (int d = 0; d < 4; ++d) {
            it.nb0[d] = src->nb[d];
            it.nb1[d] = dst->nb[d];
        }
        it.dst_contig = ggml_is_contiguous(dst);
    }
    ggml_cuda_cpy_run_4b(run, ctx.stream());
    return last - i;
}

// the two l2 norms of q and k that feed a delta-net recurrence and nothing else: the recurrence normalizes q and k as
// it loads them, so neither norm launches. Nothing up to the recurrence may write their sources.
static constexpr uint64_t GGML_CUDA_FUSE_GDN_QK_NORM = 1ull << 45;

static int ggml_cuda_try_gdn_qk_norm(const ggml_cgraph * cgraph, const int i) {
    if (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_GDN_QK_NORM) {
        return 0;
    }
    const ggml_tensor * a = cgraph->nodes[i];
    int j = i + 1;
    while (j < cgraph->n_nodes && (ggml_cuda_is_view_or_noop(cgraph->nodes[j]) || ggml_is_empty(cgraph->nodes[j]))) {
        j++;
    }
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * b = cgraph->nodes[j];
    if (a->op != GGML_OP_L2_NORM || b->op != GGML_OP_L2_NORM || a->type != GGML_TYPE_F32 || b->type != GGML_TYPE_F32 ||
            ((a->flags | b->flags) & GGML_TENSOR_FLAG_OUTPUT) || memcmp(a->op_params, b->op_params, sizeof(float)) != 0 ||
            ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, j) != 1) {
        return 0;
    }
    const ggml_tensor * sa = a->src[0];
    const ggml_tensor * sb = b->src[0];
    if (sa->type != GGML_TYPE_F32 || sb->type != GGML_TYPE_F32 || !ggml_are_same_shape(sa, a) || !ggml_are_same_shape(sb, b) ||
            !ggml_is_contiguous_rows(sa) || !ggml_is_contiguous_rows(sb) || !ggml_are_same_stride(sa, sb)) {
        return 0;
    }
    int g = -1;
    for (int k = j + 1; k < cgraph->n_nodes && g < 0; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (t->src[0] == a || t->src[1] == a || t->src[0] == b || t->src[1] == b) {
            g = k;
        }
    }
    if (g < 0) {
        return 0;
    }
    const ggml_tensor * gdn = cgraph->nodes[g];
    if (gdn->op != GGML_OP_GATED_DELTA_NET || gdn->src[0] != a || gdn->src[1] != b || gdn->src[2] == nullptr ||
            gdn->src[2]->ne[0] != a->ne[0]) {
        return 0;
    }
    for (int k = j + 1; k < g; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (!ggml_cuda_is_view_or_noop(t) && (ggml_cuda_tensors_overlap(t, sa) || ggml_cuda_tensors_overlap(t, sb))) {
            return 0;
        }
    }
    float eps;
    memcpy(&eps, a->op_params, sizeof(float));
    ggml_cuda_gated_delta_net_set_qk_norm(gdn, sa, sb, eps);
    return j - i;
}

// the gated norm in front of a quantized matvec, silu(z) * (rms_norm(x) * w): its product is quantized straight into
// the matvec's q8_1 input, which the matvec takes as a norm fold whose quantized copy is ready. The SiLU and the
// multiply never launch; when nothing between the norm and the SiLU writes x, neither does the norm.
static constexpr uint64_t GGML_CUDA_FUSE_GATED_FOLD = 1ull << 44;

struct ggml_cuda_gated_norm {
    const ggml_tensor * m;
    const ggml_tensor * x;
    const ggml_tensor * w;
    float               eps;
};

// the norms left for the gated quantize to do, until it runs
static thread_local std::vector<ggml_cuda_gated_norm> g_cuda_gated;

// the SiLU at i, the product with m right after it, maybe a plain reshape, and the one quantized matvec reading that
static bool ggml_cuda_match_gated(const ggml_cgraph * cgraph, const int i, const int cc, const ggml_tensor ** m_out, int * i_f) {
    if (i + 1 >= cgraph->n_nodes) {
        return false;
    }
    const ggml_tensor * s = cgraph->nodes[i];
    const ggml_tensor * p = cgraph->nodes[i + 1];
    if (s->op != GGML_OP_UNARY || ggml_get_unary_op(s) != GGML_UNARY_OP_SILU || s->type != GGML_TYPE_F32 || p->op != GGML_OP_MUL ||
            p->type != GGML_TYPE_F32 || (p->src[0] != s && p->src[1] != s) || ((s->flags | p->flags) & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const ggml_tensor * z = s->src[0];
    const ggml_tensor * m = p->src[0] == s ? p->src[1] : p->src[0];
    if (m == s || z->type != GGML_TYPE_F32 || m->type != GGML_TYPE_F32 || !ggml_is_contiguous(z) || !ggml_is_contiguous(m) ||
            !ggml_is_contiguous(p) || !ggml_are_same_shape(z, m) || !ggml_are_same_shape(p, m) ||
            ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, i + 1) != 1) {
        return false;
    }
    int f_idx = i + 1;
    const ggml_tensor * f = p;
    if (i + 2 < cgraph->n_nodes) {
        const ggml_tensor * r = cgraph->nodes[i + 2];
        if ((r->op == GGML_OP_RESHAPE || r->op == GGML_OP_VIEW) && r->view_src == p && r->view_offs == 0 && ggml_is_contiguous(r) &&
                ggml_nelements(r) == ggml_nelements(p) && !(r->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            if (ggml_node_get_use_count(cgraph, i + 2) != 1) {
                return false;
            }
            f     = r;
            f_idx = i + 2;
        }
    }
    if (f->ne[0] % MATRIX_ROW_PADDING != 0 || f->ne[2] != 1 || f->ne[3] != 1 || f->ne[1] > 65535) {
        return false;
    }
    const ggml_tensor * reader = nullptr;
    for (int k = f_idx + 1; k < cgraph->n_nodes && reader == nullptr; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        for (int q = 0; q < GGML_MAX_SRC; ++q) {
            if (t->src[q] == f) {
                reader = t;
            }
        }
    }
    if (reader == nullptr || reader->op != GGML_OP_MUL_MAT || !ggml_is_quantized(reader->src[0]->type) ||
            !ggml_cuda_norm_fold_reader_ok(reader, f, cc)) {
        return false;
    }
    *m_out = m;
    *i_f   = f_idx;
    return true;
}

// rms_norm at i and its weight multiply: left undone when the gated quantize after them will do it
static int ggml_cuda_try_gated_norm(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if ((ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_GATED_FOLD) || i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * norm = cgraph->nodes[i];
    const ggml_tensor * mul  = cgraph->nodes[i + 1];
    if (norm->op != GGML_OP_RMS_NORM || mul->op != GGML_OP_MUL || mul->src[0] != norm || norm->type != GGML_TYPE_F32 ||
            ((norm->flags | mul->flags) & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, i) != 1 ||
            ggml_node_get_use_count(cgraph, i + 1) != 1) {
        return 0;
    }
    const ggml_tensor * x = norm->src[0];
    const ggml_tensor * w = mul->src[1];
    // the norm's rows are the quantize's groups: whole q8_1 blocks, one warp's worth of rms_norm_f32<256> columns
    if (x->type != GGML_TYPE_F32 || w->type != GGML_TYPE_F32 || !ggml_is_contiguous(x) || !ggml_are_same_shape(x, mul) ||
            !ggml_is_contiguous(w) || ggml_nelements(w) != x->ne[0] || x->ne[0] % WARP_SIZE != 0 || x->ne[0] > 256) {
        return 0;
    }
    // the product's reader: the multiply with a SiLU just before it
    int ip = -1;
    for (int k = i + 2; k < cgraph->n_nodes && ip < 0; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (t->src[0] == mul || t->src[1] == mul) {
            ip = k;
        }
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_tensor * m = nullptr;
    int i_f = 0;
    if (ip < i + 3 || !ggml_cuda_match_gated(cgraph, ip - 1, cc, &m, &i_f) || m != mul || cgraph->nodes[i_f]->ne[0] % x->ne[0] != 0) {
        return 0;
    }
    // what runs until the SiLU leaves x and w alone and cannot take the SiLU into a launch of its own
    const ggml_tensor * w_root  = w->view_src ? w->view_src : w;
    const bool          w_const = w_root->buffer && ggml_backend_buffer_get_usage(w_root->buffer) == GGML_BACKEND_BUFFER_USAGE_WEIGHTS;
    for (int k = i + 2; k < ip - 1; ++k) {
        const ggml_tensor * t = cgraph->nodes[k];
        if (ggml_cuda_is_view_or_noop(t)) {
            continue;
        }
        if (ggml_cuda_tensors_overlap(t, x) || (!w_const && ggml_cuda_tensors_overlap(t, w))) {
            return 0;
        }
        if (t->op == GGML_OP_MUL_MAT && k + ggml_cuda_try_mm_epilogue(ctx, cgraph, k, true) >= ip - 1) {
            return 0;
        }
    }
    ggml_cuda_gated_norm g;
    g.m = mul;
    g.x = x;
    g.w = w;
    memcpy(&g.eps, norm->op_params, sizeof(float));
    g_cuda_gated.push_back(g);
    return 1;
}

// the SiLU at i: the gated product quantized for its matvec, which finds it as a ready norm fold
static int ggml_cuda_try_gated_fold(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_GATED_FOLD) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_tensor * m = nullptr;
    int i_f = 0;
    if (!ggml_cuda_match_gated(cgraph, i, cc, &m, &i_f)) {
        return 0;
    }
    const ggml_tensor * s = cgraph->nodes[i];
    const ggml_tensor * p = cgraph->nodes[i + 1];
    const ggml_tensor * f = cgraph->nodes[i_f];
    const ggml_cuda_gated_norm * g = nullptr;
    for (const ggml_cuda_gated_norm & c : g_cuda_gated) {
        g = c.m == m ? &c : g;
    }
    if (g == nullptr && ggml_cuda_norm_fold_for(ctx, m) != nullptr) {
        return 0; // m itself was never written
    }
    const int64_t ne0   = f->ne[0];
    const int64_t nrows = f->ne[1];
    auto fold = std::make_shared<ggml_cuda_norm_fold>();
    fold->norm          = const_cast<ggml_tensor *>(s);
    fold->mul           = const_cast<ggml_tensor *>(p);
    fold->x             = g ? g->x : m;
    fold->w             = g ? g->w : m;
    fold->eps           = g ? g->eps : 0.0f;
    fold->n_rows        = nrows;
    fold->q8_bytes      = nrows * (ne0 / QK8_1) * sizeof(block_q8_1);
    fold->has_q8_reader = true;
    ggml_cuda_norm_fold_reserve(ctx, *fold);
    quantize_q8_1_gated_cuda(g ? nullptr : (const float *) m->data, g ? (const float *) g->x->data : nullptr,
        g ? (const float *) g->w->data : nullptr, fold->eps, (const float *) s->src[0]->data, fold->q8, ne0, nrows,
        g ? (int) g->x->ne[0] : 256, ctx.stream());
    fold->q8_ready = true;
    ctx.norm_folds[f] = fold;
    if (g) {
        g_cuda_gated.erase(g_cuda_gated.begin() + (g - g_cuda_gated.data()));
    }
    return i_f - i;
}

// a delta-net layer's input projection through its convolution: the matvec, the transpose and concat that put each
// channel's state ahead of its tokens, the per-token state snapshot copies, and the convolution with its SiLU, as
// one matvec launch whose row lanes convolve and publish the snapshots. It launches at the matvec. The convolution
// nodes come after unrelated ones (a state gather), so they are taken as absorbed and skipped when the walk reaches
// them, and no node before them fuses, so no other launch can cover them.
static constexpr uint64_t GGML_CUDA_FUSE_CONV_PROJ = 1ull << 43;

static thread_local std::vector<const ggml_tensor *> g_cuda_absorbed;
static thread_local int                              g_cuda_no_fuse_until = 0;

static bool ggml_cuda_take_absorbed(const ggml_tensor * node) {
    for (size_t k = 0; k < g_cuda_absorbed.size(); ++k) {
        if (g_cuda_absorbed[k] == node) {
            g_cuda_absorbed.erase(g_cuda_absorbed.begin() + k);
            return true;
        }
    }
    return false;
}

// GGML_CUDA_TRACE_CONV=1: why a matvec was not taken with its convolution
static int ggml_cuda_conv_reject(const int line, const int i) {
    static const bool on = getenv("GGML_CUDA_TRACE_CONV") != nullptr;
    if (on) {
        fprintf(stderr, "[conv] node %d not taken at line %d\n", i, line);
    }
    return 0;
}

static int ggml_cuda_try_conv_proj(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_CONV_PROJ) {
        return ggml_cuda_conv_reject(__LINE__, i);
    }
    const ggml_tensor * mm = cgraph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT || mm->type != GGML_TYPE_F32 || (mm->flags & GGML_TENSOR_FLAG_OUTPUT) || !ggml_is_contiguous(mm)) {
        return ggml_cuda_conv_reject(__LINE__, i);
    }
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    const int     cc = ggml_cuda_info().devices[ctx.device].cc;
    const int64_t T  = mm->ne[1];
    const int64_t C  = mm->ne[0];
    if (src1->type != GGML_TYPE_F32 || src1->nb[0] != sizeof(float) || src0->ne[2] != 1 || src0->ne[3] != 1 || src1->ne[2] != 1 ||
            src1->ne[3] != 1 || src0->buffer == nullptr || ggml_backend_buffer_get_usage(src0->buffer) != GGML_BACKEND_BUFFER_USAGE_WEIGHTS ||
            !ggml_is_contiguous(src0) || ggml_get_op_params_i32(mm, 1) == GGML_HINT_SRC0_IS_HADAMARD ||
            !ggml_cuda_should_fuse_mul_mat_vec_q(mm) || !ggml_cuda_mmvq_conv_supported(src0->type, cc, T) ||
            ggml_node_get_use_count(cgraph, i) != 1) {
        return ggml_cuda_conv_reject(__LINE__, i);
    }

    // the concat: the first node after the matvec that is not a view, reading the transposed product after the state
    int ic = i + 1;
    while (ic < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[ic])) {
        ic++;
    }
    if (ic >= cgraph->n_nodes) {
        return ggml_cuda_conv_reject(__LINE__, i);
    }
    const ggml_tensor * cat = cgraph->nodes[ic];
    if (cat->op != GGML_OP_CONCAT || cat->type != GGML_TYPE_F32 || ggml_get_op_params_i32(cat, 0) != 0 || !ggml_is_contiguous(cat) ||
            cat->ne[0] != 3 + T || cat->ne[1] != C || cat->ne[2] != 1 || cat->ne[3] != 1) {
        return ggml_cuda_conv_reject(__LINE__, i);
    }
    const ggml_tensor * tr = cat->src[1];
    const ggml_tensor * st = cat->src[0];
    if (tr->op != GGML_OP_TRANSPOSE || tr->data != mm->data || tr->ne[0] != T || tr->ne[1] != C ||
            tr->nb[0] != (size_t) C*sizeof(float) || tr->nb[1] != sizeof(float)) {
        return ggml_cuda_conv_reject(__LINE__, i);
    }
    const ggml_tensor * rs = tr->src[0];
    if (rs != mm && !(rs->op == GGML_OP_RESHAPE && rs->src[0] == mm)) {
        return ggml_cuda_conv_reject(__LINE__, i);
    }
    for (int k = i + 1; k < ic; ++k) {
        if ((cgraph->nodes[k] == tr || cgraph->nodes[k] == rs) && ggml_node_get_use_count(cgraph, k) != 1) {
            return ggml_cuda_conv_reject(__LINE__, i);
        }
    }
    if (st->type != GGML_TYPE_F32 || st->ne[0] != 3 || st->ne[1] != C || st->ne[2] != 1 || st->ne[3] != 1 ||
            st->nb[0] != sizeof(float) || st->nb[1] != 3*sizeof(float)) {
        return ggml_cuda_conv_reject(__LINE__, i);
    }

    // after the concat: its snapshot copies, unrelated nodes, and the convolution
    ggml_cuda_mmvq_conv conv{};
    const ggml_tensor * snap_views[GGML_CUDA_CONV_SNAP_MAX] = {};
    int  view_idx[GGML_CUDA_CONV_SNAP_MAX];
    int  n_views   = 0;
    int  last_lead = ic;
    bool lead      = true;
    int  ik        = -1;
    std::vector<int> others;
    for (int k = ic + 1; k < cgraph->n_nodes; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (n->op == GGML_OP_SSM_CONV) {
            if (n->src[0] != cat) {
                return ggml_cuda_conv_reject(__LINE__, i);
            }
            ik = k;
            break;
        }
        if (n->op == GGML_OP_VIEW && n->view_src == cat) {
            if (n_views == GGML_CUDA_CONV_SNAP_MAX) {
                return ggml_cuda_conv_reject(__LINE__, i);
            }
            snap_views[n_views] = n;
            view_idx[n_views++] = k;
            if (lead) {
                last_lead = k;
            }
            continue;
        }
        if (n->op == GGML_OP_CPY && n->src[0]->op == GGML_OP_VIEW && n->src[0]->view_src == cat) {
            const ggml_tensor * v = n->src[0];
            const ggml_tensor * d = n->src[1];
            int vk = -1;
            for (int s = 0; s < n_views; ++s) {
                vk = snap_views[s] == v ? view_idx[s] : vk;
            }
            if (vk < 0 || ggml_node_get_use_count(cgraph, vk) != 1 || conv.n_snap == GGML_CUDA_CONV_SNAP_MAX ||
                    v->ne[0] != 3 || v->ne[1] != C || v->ne[2] != 1 || v->ne[3] != 1 || v->nb[0] != sizeof(float) ||
                    v->nb[1] != cat->nb[1] || v->view_offs % sizeof(float) != 0 || v->view_offs/sizeof(float) > (size_t) T ||
                    d->type != GGML_TYPE_F32 || !ggml_is_contiguous(d) || ggml_nelements(d) != 3*C || d->data == nullptr) {
                return ggml_cuda_conv_reject(__LINE__, i);
            }
            conv.snap[conv.n_snap]     = (float *) d->data;
            conv.snap_idx[conv.n_snap] = (int) (v->view_offs/sizeof(float));
            conv.n_snap++;
            if (lead) {
                last_lead = k;
            }
            continue;
        }
        if (ggml_cuda_is_view_or_noop(n) || ggml_is_empty(n) || (n->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
            if (lead) {
                last_lead = k;
            }
            continue;
        }
        lead = false;
        others.push_back(k);
    }
    if (ik < 0 || ggml_node_get_use_count(cgraph, ic) != n_views + 1) {
        return ggml_cuda_conv_reject(__LINE__, i);
    }
    const ggml_tensor * sc = cgraph->nodes[ik];
    const ggml_tensor * cw = sc->src[1];
    if (sc->type != GGML_TYPE_F32 || cw->type != GGML_TYPE_F32 || cw->ne[0] != 4 || cw->ne[1] != C || cw->nb[0] != sizeof(float)) {
        return ggml_cuda_conv_reject(__LINE__, i);
    }
    const ggml_tensor * add  = nullptr;
    const ggml_tensor * out  = sc;
    int                 last = ik;
    if (ggml_cuda_can_fuse(cgraph, ik, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        add  = cgraph->nodes[ik + 1];
        out  = cgraph->nodes[ik + 2];
        last = ik + 2;
        const ggml_tensor * bias = add->src[0] == sc ? add->src[1] : add->src[0];
        if (ggml_nelements(bias) != C || !ggml_is_contiguous(bias)) {
            return ggml_cuda_conv_reject(__LINE__, i);
        }
        conv.bias = (const float *) bias->data;
    } else if (ggml_cuda_can_fuse(cgraph, ik, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        out  = cgraph->nodes[ik + 1];
        last = ik + 1;
    }
    if (out->ne[0] != C || out->ne[1] != T || out->nb[0] != sizeof(float) || (out->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return ggml_cuda_conv_reject(__LINE__, i);
    }

    // the nodes that still run between the copies and the convolution neither read nor write what the launch writes
    for (const int k : others) {
        const ggml_tensor * n = cgraph->nodes[k];
        for (int s = -1; s < GGML_MAX_SRC; ++s) {
            const ggml_tensor * t = s < 0 ? n : n->src[s];
            if (t == nullptr) {
                continue;
            }
            if (ggml_cuda_tensors_overlap(t, out)) {
                return ggml_cuda_conv_reject(__LINE__, i);
            }
            for (int q = 0; q < conv.n_snap; ++q) {
                const char * b = (const char *) conv.snap[q];
                const char * e = b + 3*C*sizeof(float);
                if (t->data && (const char *) t->data < e && b < (const char *) t->data + ggml_nbytes(t)) {
                    return ggml_cuda_conv_reject(__LINE__, i);
                }
            }
        }
    }
    // a folded gather leaves the state in the cache, read in place as the concat would have read it; st itself is
    // then never written, and its memory may be the output's
    const ggml_cuda_state_fold * sfold = nullptr;
    if (!ctx.state_folds.empty()) {
        const auto it = ctx.state_folds.find(st->view_src ? st->view_src : st);
        sfold = it != ctx.state_folds.end() ? &it->second : nullptr;
    }
    // the output is written while the state, activation and taps are read by other lanes; the snapshots are distinct rows
    ggml_cuda_norm_fold * fold = ggml_cuda_norm_fold_for(ctx, src1);
    for (const ggml_tensor * t : { sfold ? nullptr : st, src1, cw, fold ? fold->x : nullptr, fold ? fold->w : nullptr }) {
        if (t != nullptr && ggml_cuda_tensors_overlap(t, out)) {
            return ggml_cuda_conv_reject(__LINE__, i);
        }
    }
    for (int q = 0; q < conv.n_snap; ++q) {
        const char * b = (const char *) conv.snap[q];
        if (b < (const char *) out->data + ggml_nbytes(out) && (const char *) out->data < b + 3*C*sizeof(float)) {
            return ggml_cuda_conv_reject(__LINE__, i);
        }
        for (int r = 0; r < q; ++r) {
            const char * o = (const char *) conv.snap[r];
            if (b < o + 3*C*sizeof(float) && o < b + 3*C*sizeof(float)) {
                return ggml_cuda_conv_reject(__LINE__, i);
            }
        }
    }

    conv.state_s1 = 3;
    conv.state    = (const float *) st->data;
    if (sfold) {
        GGML_ASSERT(sfold->type == GGML_TYPE_F32);
        conv.state     = (const float *) sfold->base;
        conv.state_ids = sfold->ids;
        conv.state_row = sfold->row;
        ctx.state_folds.erase(st->view_src ? st->view_src : st);
    }
    conv.w       = (const float *) cw->data;
    conv.w_s1    = (int) (cw->nb[1] / sizeof(float));
    conv.y       = (float *) out->data;
    conv.y_s1    = (int) (out->nb[1] / sizeof(float));
    conv.silu    = out != sc;
    conv.snap_s1 = 3;
    ggml_cuda_mul_mat_vec_q_conv(ctx, mm, fold, conv);
    ggml_cuda_norm_fold_check_at(ctx, src1);

    if (others.empty()) {
        return last - i;
    }
    for (int k = last_lead + 1; k <= last; ++k) {
        const ggml_tensor * n = cgraph->nodes[k];
        if (std::find(others.begin(), others.end(), k) == others.end() && !ggml_cuda_is_view_or_noop(n)) {
            g_cuda_absorbed.push_back(n);
        }
    }
    g_cuda_no_fuse_until = last;
    return last_lead - i;
}

// two float matvecs over one activation, each with its elementwise tail, as one launch: the gating projections of a
// delta-net layer (alpha with its bias, softplus and scale, beta with its sigmoid) are this at any token count. Each
// row sums as mul_mat_vec_f sums it, so the values are the separate launches' bits.
static constexpr uint64_t GGML_CUDA_FUSE_MMVF_PAIR = 1ull << 42;

static bool ggml_cuda_mmvf_pair_member(const ggml_tensor * mm, const int cc) {
    if (mm->op != GGML_OP_MUL_MAT || mm->type != GGML_TYPE_F32 || (mm->flags & GGML_TENSOR_FLAG_OUTPUT)) {
        return false;
    }
    const ggml_tensor * w = mm->src[0];
    const ggml_tensor * y = mm->src[1];
    return w->type == GGML_TYPE_F32 && y->type == GGML_TYPE_F32 && w->buffer != nullptr &&
           ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
           ggml_cuda_should_use_mmvf(w->type, cc, w->ne, w->nb, y->ne[1]) &&
           ggml_is_contiguous(mm) && ggml_is_contiguous(w) && y->nb[0] == sizeof(float) &&
           w->ne[2] == 1 && w->ne[3] == 1 && y->ne[2] == 1 && y->ne[3] == 1 &&
           mm->ne[1] >= 1 && mm->ne[1] <= 8 && w->ne[0] % 2 == 0 && (y->nb[1] / sizeof(float)) % 2 == 0 &&
           ggml_get_op_params_i32(mm, 1) != GGML_HINT_SRC0_IS_HADAMARD;
}

static int ggml_cuda_try_mmvf_pair_epi(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_MMVF_PAIR) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_tensor * ma = cgraph->nodes[i];
    if (!ggml_cuda_mmvf_pair_member(ma, cc)) {
        return 0;
    }
    ggml_cuda_mm_epilogue epi_a{};
    const ggml_tensor * args_a[GGML_CUDA_MM_EPI_MAX_ARGS] = {};
    const int last_a = ggml_cuda_mm_epi_collect(cgraph, i, epi_a, args_a);

    int j = last_a + 1;
    while (j < cgraph->n_nodes && (ggml_cuda_is_view_or_noop(cgraph->nodes[j]) || ggml_is_empty(cgraph->nodes[j]))) {
        j++;
    }
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * mb = cgraph->nodes[j];
    if (!ggml_cuda_mmvf_pair_member(mb, cc) || mb->src[1] != ma->src[1] || mb->src[0]->ne[0] != ma->src[0]->ne[0]) {
        return 0;
    }
    ggml_cuda_mm_epilogue epi_b{};
    const ggml_tensor * args_b[GGML_CUDA_MM_EPI_MAX_ARGS] = {};
    const int last_b = ggml_cuda_mm_epi_collect(cgraph, j, epi_b, args_b);

    // a gate and up product feeding a GLU belongs to the GLU paths
    int k_next = last_b + 1;
    while (k_next < cgraph->n_nodes && (ggml_cuda_is_view_or_noop(cgraph->nodes[k_next]) || ggml_is_empty(cgraph->nodes[k_next]))) {
        k_next++;
    }
    if (k_next < cgraph->n_nodes && cgraph->nodes[k_next]->op == GGML_OP_GLU) {
        return 0;
    }

    // the float reader of a folded norm reads its input with the factors applied, as mul_mat_vec_f would
    const ggml_tensor * src1 = ma->src[1];
    ggml_cuda_norm_fold * fold = ggml_cuda_norm_fold_for(ctx, src1);
    if (fold && (!fold->scale_ready || getenv("GGML_CUDA_FOLD_FLOAT_MATERIALIZE") != nullptr ||
            fold->x->type != GGML_TYPE_F32 || fold->w->type != GGML_TYPE_F32)) {
        return 0;
    }

    // both launches run at once: neither output may sit under anything either of them reads, or under the other
    const ggml_tensor * out_a = cgraph->nodes[last_a];
    const ggml_tensor * out_b = cgraph->nodes[last_b];
    const ggml_tensor * reads[4 + 2*GGML_CUDA_MM_EPI_MAX_ARGS] = { ma->src[0], mb->src[0], fold ? fold->x : src1, fold ? fold->w : nullptr };
    int n_reads = 4;
    for (int k = 0; k < epi_a.n_args; ++k) {
        reads[n_reads++] = args_a[k];
    }
    for (int k = 0; k < epi_b.n_args; ++k) {
        reads[n_reads++] = args_b[k];
    }
    for (const ggml_tensor * out : { out_a, out_b }) {
        for (int k = 0; k < n_reads; ++k) {
            if (reads[k] && ggml_cuda_tensors_overlap(out, reads[k])) {
                return 0;
            }
        }
    }
    if (ggml_cuda_tensors_overlap(out_a, out_b)) {
        return 0;
    }

    ggml_cuda_mul_mat_vec_f_pair_epi(ctx, ma->src[0], mb->src[0], src1, fold,
        (float *) out_a->data, (float *) out_b->data, epi_a, epi_b, (int) ma->ne[1]);
    ggml_cuda_norm_fold_check_at(ctx, src1);
    return last_b - i;
}

// a matvec, a plain reshape of it, and an add with a tensor of the reshape's shape (a residual after an output
// projection): the add rides the matvec as its per-column bias, as when the add follows the matvec directly
static constexpr uint64_t GGML_CUDA_FUSE_MM_RESHAPE_ADD = 1ull << 46;

static int ggml_cuda_try_mm_reshape_add(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if (i + 2 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * mm = cgraph->nodes[i];
    // an NVFP4 weight's own scale may sit between the product and the reshape; it rides the matvec as well
    const ggml_tensor * scale = nullptr;
    int                 k     = i + 1;
    if (mm->op == GGML_OP_MUL_MAT && mm->src[0]->type == GGML_TYPE_NVFP4 && i + 3 < cgraph->n_nodes) {
        const ggml_tensor * mul = cgraph->nodes[i + 1];
        if (mul->op == GGML_OP_MUL && mul->src[0] == mm && mul->src[1]->type == GGML_TYPE_F32 &&
                ggml_nelements(mul->src[1]) == 1 && ggml_is_contiguous(mul->src[1]) && mul->type == GGML_TYPE_F32 &&
                ggml_are_same_shape(mul, mm) && ggml_is_contiguous(mul) && !(mm->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                ggml_node_get_use_count(cgraph, i) == 1) {
            scale = mul->src[1];
            k     = i + 2;
        }
    }
    const ggml_tensor * prod = cgraph->nodes[k - 1];
    const ggml_tensor * rs   = cgraph->nodes[k];
    ggml_tensor *       add  = cgraph->nodes[k + 1];
    if (mm->op != GGML_OP_MUL_MAT || mm->type != GGML_TYPE_F32 || (rs->op != GGML_OP_RESHAPE && rs->op != GGML_OP_VIEW) ||
            rs->view_src != prod || rs->view_offs != 0 || add->op != GGML_OP_ADD || add->type != GGML_TYPE_F32 ||
            ((prod->flags | rs->flags) & GGML_TENSOR_FLAG_OUTPUT) || !ggml_is_contiguous(prod) || !ggml_is_contiguous(rs) ||
            !ggml_is_contiguous(add) || ggml_nelements(rs) != ggml_nelements(prod) || rs->ne[0] != prod->ne[0] ||
            ggml_node_get_use_count(cgraph, k - 1) != 1 || ggml_node_get_use_count(cgraph, k) != 1) {
        return 0;
    }
    const ggml_tensor * bias = add->src[0] == rs ? add->src[1] : (add->src[1] == rs ? add->src[0] : nullptr);
    if (bias == nullptr || bias == rs || !ggml_are_same_shape(add->src[0], add->src[1]) || bias->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(bias)) {
        return 0;
    }
    // the product is never written; the add's output may be the bias itself (each element read, then written, by one
    // lane) but must not otherwise overlap what the matvec reads
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    const bool exact = add->data == bias->data;
    if ((!exact && ggml_cuda_tensors_overlap(add, bias)) || ggml_cuda_tensors_overlap(add, src0)) {
        return 0;
    }
    ggml_cuda_norm_fold * fold = ggml_cuda_norm_fold_for(ctx, src1);
    if (ggml_cuda_tensors_overlap(add, fold ? fold->x : src1)) {
        return 0;
    }
    // the matvec writes the add's rows as its columns: a product over [K, 1, n] sequences (a recurrent layer's output
    // under several sequences) reshaped to [N, n] is n columns, read from src1 as columns only when it is contiguous
    const int64_t ncols = ggml_nrows(add);
    if (ncols > MMVQ_MAX_BATCH_SIZE || (mm->ne[2] * mm->ne[3] > 1 && !ggml_is_contiguous(src1))) {
        return 0;
    }
    ggml_cuda_mm_fusion_args_host fusion_data{};
    fusion_data.x_bias  = bias;
    fusion_data.x_scale = scale;
    if (!scale && ggml_cuda_should_fuse_mul_mat_vec_f(mm)) {
        ggml_cuda_mul_mat_vec_f(ctx, src0, src1, nullptr, add, &fusion_data, fold);
    } else if (ggml_cuda_should_fuse_mul_mat_vec_q(mm) && ggml_cuda_should_use_mmvq(src0->type, ggml_cuda_info().devices[ctx.device].cc, ncols) &&
               !ggml_cuda_fp4_big_weight(src0, src1, mm, ggml_cuda_info().devices[ctx.device].cc)) {
        ggml_cuda_mul_mat_vec_q(ctx, src0, src1, nullptr, add, &fusion_data, fold);
    } else {
        return 0;
    }
    ggml_cuda_norm_fold_check_at(ctx, src1);
    return k + 1 - i;
}

// a multi-token expert matvec and the multiply by its router weights after it: the weights ride the matvec's store, and
// the expert sum after them (a plain sum then) never reads the weights while other blocks store
static constexpr uint64_t GGML_CUDA_FUSE_MOE_ROUTER_WEIGHTS = 1ull << 47;

// NVFP4 per-expert scales carried by the multi-token expert kernel, where each was a repeat, a gather and a multiply
static constexpr uint64_t GGML_CUDA_FUSE_EXPERT_SCALE_MT = 1ull << 48;

// An NVFP4 expert matvec's second-level scale as the graph builds it: the per-expert scales reshaped, repeated
// over the tokens, picked by the same ids, then multiplied in. Returns the scales when nodes i..i+4 are that and
// nothing else reads the steps.
static const ggml_tensor * ggml_cuda_expert_scale_chain(const ggml_cgraph * cgraph, const int i) {
    if ((ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_EXPERT_SCALE_MT) || i + 4 >= cgraph->n_nodes) {
        return nullptr;
    }
    const ggml_tensor * mm      = cgraph->nodes[i];
    const ggml_tensor * reshape = cgraph->nodes[i + 1];
    const ggml_tensor * repeat  = cgraph->nodes[i + 2];
    const ggml_tensor * getrows = cgraph->nodes[i + 3];
    const ggml_tensor * mul     = cgraph->nodes[i + 4];
    if (mm->op != GGML_OP_MUL_MAT_ID || reshape->op != GGML_OP_RESHAPE || repeat->op != GGML_OP_REPEAT ||
            getrows->op != GGML_OP_GET_ROWS || mul->op != GGML_OP_MUL) {
        return nullptr;
    }
    if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm->src[2] ||
            !((mul->src[0] == mm && mul->src[1] == getrows) || (mul->src[0] == getrows && mul->src[1] == mm))) {
        return nullptr;
    }
    const ggml_tensor * scale = reshape->src[0];
    if (mm->src[0]->type != GGML_TYPE_NVFP4 || mul->type != GGML_TYPE_F32 || scale->type != GGML_TYPE_F32 ||
            !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm->src[0]->ne[2] || !ggml_are_same_shape(mul, mm)) {
        return nullptr;
    }
    for (int k = i; k <= i + 3; k++) {
        if ((cgraph->nodes[k]->flags & GGML_TENSOR_FLAG_OUTPUT) || ggml_node_get_use_count(cgraph, k) != 1) {
            return nullptr;
        }
    }
    return scale;
}

static __global__ void k_expert_scale(const float * __restrict__ x, float * dst, const float * __restrict__ scale,
        const int32_t * __restrict__ ids, const int64_t ids_s1, const int nrows, const int n_slots) {
    ggml_cuda_pdl_lc();
    const int r    = blockIdx.x*blockDim.x + threadIdx.x;
    const int slot = blockIdx.y;
    const int tok  = blockIdx.z;
    ggml_cuda_pdl_sync();
    if (r >= nrows) {
        return;
    }
    const int64_t i = ((int64_t) tok*n_slots + slot)*nrows + r;
    dst[i] = x[i]*scale[ids[tok*ids_s1 + slot]];
}

// The same chain over more tokens than the matvec takes (the MMQ path): the product as the graph runs it, then one
// pass for the repeat, the gather and the multiply, each slot's rows times its expert's scale
static int ggml_cuda_try_expert_scale_pass(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    static const bool off = getenv("GGML_CUDA_EXPERT_SCALE_PASS") && atoi(getenv("GGML_CUDA_EXPERT_SCALE_PASS")) == 0;
    ggml_tensor * mm = cgraph->nodes[i];
    if (off || mm->op != GGML_OP_MUL_MAT_ID || mm->ne[2] <= MMVQ_MAX_BATCH_SIZE) {
        return 0;
    }
    const ggml_tensor * scale = ggml_cuda_expert_scale_chain(cgraph, i);
    if (scale == nullptr) {
        return 0;
    }
    ggml_tensor *       mul = cgraph->nodes[i + 4];
    const ggml_tensor * ids = mm->src[2];
    if (!ggml_is_contiguous(mm) || !ggml_is_contiguous(mul) || ids->type != GGML_TYPE_I32 || ids->nb[0] != sizeof(int32_t) ||
            mm->ne[3] != 1 || mm->ne[0] > INT_MAX || mm->ne[2] > 65535) {
        return 0;
    }
    GGML_ASSERT(ggml_cuda_compute_forward(ctx, mm));
    ggml_cuda_norm_fold_poison_scan(ctx, mm, i);
    const dim3 grid((unsigned) ((mm->ne[0] + 255)/256), (unsigned) mm->ne[1], (unsigned) mm->ne[2]);
    const ggml_cuda_kernel_launch_params lp = ggml_cuda_kernel_launch_params(grid, 256, 0, ctx.stream());
    ggml_cuda_kernel_launch(k_expert_scale, lp, (const float *) mm->data, (float *) mul->data, (const float *) scale->data,
            (const int32_t *) ids->data, (int64_t) (ids->nb[1]/sizeof(int32_t)), (int) mm->ne[0], (int) mm->ne[1]);
    return 4;
}

static int ggml_cuda_try_moe_router_weights(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, const int i) {
    if ((ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_MOE_ROUTER_WEIGHTS) || i + 1 >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * mm    = cgraph->nodes[i];
    // the weights multiply the matvec, or the matvec with its expert scale applied
    const ggml_tensor * scale = ggml_cuda_expert_scale_chain(cgraph, i);
    const int           j     = scale ? i + 5 : i + 1;
    if (j >= cgraph->n_nodes) {
        return 0;
    }
    const ggml_tensor * prod = cgraph->nodes[j - 1];
    ggml_tensor *       mul  = cgraph->nodes[j];
    if (mm->op != GGML_OP_MUL_MAT_ID || mul->op != GGML_OP_MUL || mul->src[0] != prod || mm->type != GGML_TYPE_F32 ||
            mul->type != GGML_TYPE_F32 || ((mm->flags | prod->flags | mul->flags) & GGML_TENSOR_FLAG_OUTPUT) ||
            ggml_node_get_use_count(cgraph, j - 1) != 1 ||
            !ggml_are_same_shape(mm, mul) || !ggml_is_contiguous(mm) || !ggml_is_contiguous(mul)) {
        return 0;
    }
    const ggml_tensor * w    = mul->src[1];
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    const ggml_tensor * ids  = mm->src[2];
    // [1, n_used, n_tokens]: one weight per (slot, token), broadcast over the rows; the multi-token kernel only
    if (w->type != GGML_TYPE_F32 || w->ne[0] != 1 || w->ne[1] != mm->ne[1] || w->ne[2] != mm->ne[2] || w->ne[3] != 1 ||
            mm->ne[2] < 2 || !ggml_is_quantized(src0->type) || !ggml_cuda_should_fuse_mul_mat_vec_q(mm) ||
            w->nb[1] % sizeof(float) != 0 || w->nb[2] % sizeof(float) != 0) {
        return 0;
    }
    // the product lands where the multiply's output is; it may be the matvec's own, never the weights or the input
    if (ggml_cuda_tensors_overlap(mul, w) || ggml_cuda_tensors_overlap(mul, src1) || ggml_cuda_tensors_overlap(mul, ids)) {
        return 0;
    }
    ggml_cuda_mm_fusion_args_host fusion{};
    fusion.x_scale = scale;
    ggml_cuda_mmvq_set_router_weights((const float *) w->data, w->nb[1] / sizeof(float), w->nb[2] / sizeof(float));
    ggml_cuda_mul_mat_vec_q(ctx, src0, src1, ids, mul, scale ? &fusion : nullptr, ggml_cuda_norm_fold_for(ctx, src1));
    GGML_ASSERT(ggml_cuda_mmvq_router_weights_taken() && "the expert matvec did not take its router weights");
    ggml_cuda_norm_fold_check_at(ctx, src1);
    return j - i;
}

// a float gate and up pair under a few columns (a speculative verify batch) as one launch with its GLU, as one column is
static constexpr uint64_t GGML_CUDA_FUSE_MMVF_GATE_COLS = 1ull << 49;
// an FP4 gate and up group's GEMM writing its GEGLU (ggml_cuda_try_mmq_group)
static constexpr uint64_t GGML_CUDA_FUSE_FP4_GEGLU = 1ull << 50;
// the GEGLU's down projection quantized in the GEGLU kernel (bit 54 off)
static constexpr uint64_t GGML_CUDA_FUSE_FP4_DOWN_QUANT = 1ull << 54;

static bool ggml_cuda_should_fuse_mmvf_gate_cols(const ggml_tensor * mm) {
    if ((ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_MMVF_GATE_COLS) || mm->op != GGML_OP_MUL_MAT || mm->ne[1] < 2 ||
            mm->ne[1] > MMVF_MAX_BATCH_SIZE || mm->ne[2] != 1 || mm->ne[3] != 1) {
        return false;
    }
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    if ((src0->type != GGML_TYPE_F32 && src0->type != GGML_TYPE_F16 && src0->type != GGML_TYPE_BF16) ||
            src1->type != GGML_TYPE_F32 || mm->type != GGML_TYPE_F32) {
        return false;
    }
    const int cc = ggml_cuda_info().devices[ggml_cuda_get_device()].cc;
    return ggml_cuda_should_use_mmvf(src0->type, cc, src0->ne, src0->nb, src1->ne[1]);
}

// Quantized matmuls of several sequences' rows over the same activations, next to each other in the graph (q, k and
// v; gate and up), run as one launch: the activations quantized once and one grid over every matrix's row tiles,
// where the small ones would each hold the GPU for a few SMs' work and a fixup of their own. One sequence's rows keep
// their launches. Returns the nodes the group covers past i. GGML_CUDA_MMQ_NO_GROUP=1 turns it off.
static int ggml_cuda_try_mmq_group(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const bool off = getenv("GGML_CUDA_MMQ_NO_GROUP") != nullptr;
    if (off) {
        return 0;
    }
    const int cc        = ggml_cuda_info().devices[ctx.device].cc;
    const int warp_size = ggml_cuda_info().devices[ctx.device].warp_size;

    // the FP4 route (cuBLASLt) groups into one GEMM; the quantized route into one MMQ grid
    auto fp4_ok = [&](const ggml_tensor * t) {
        return t->op == GGML_OP_MUL_MAT && ggml_cuda_mmq_batched(t) && ggml_cuda_fp4_gemm_supported(t->src[0], t->src[1], t, cc) &&
            ggml_backend_buffer_get_usage(t->src[0]->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE;
    };
    auto member_ok = [&](const ggml_tensor * t) {
        if (t->op != GGML_OP_MUL_MAT || !ggml_cuda_mmq_batched(t)) {
            return false;
        }
        const ggml_tensor * w   = t->src[0];
        const ggml_tensor * act = t->src[1];
        return act->type == GGML_TYPE_F32 && t->type == GGML_TYPE_F32 && ggml_is_quantized(w->type) &&
            w->ne[2] == 1 && w->ne[3] == 1 && act->ne[2] == 1 && act->ne[3] == 1 &&
            w->nb[0] == ggml_type_size(w->type) && ggml_is_contiguous(w) && ggml_is_contiguous(t) &&
            ggml_backend_buffer_get_usage(w->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE &&
            !ggml_cuda_should_use_mmvf(w->type, cc, w->ne, w->nb, act->ne[1]) &&
            !ggml_cuda_should_use_mmf(w->type, cc, warp_size, w->ne, w->nb, act->ne[1], false) &&
            !ggml_cuda_should_use_mmvq(w->type, cc, act->ne[1]) &&
            !ggml_cuda_fp4_gemm_supported(w, act, t, cc) &&
            ggml_cuda_should_use_mmq(w->type, cc, act->ne[1], 0);
    };
    // a matmul with a bias or scale fused onto it keeps that (the gate/up GLU fusions are mat-vec kernels, which
    // several sequences' rows do not take)
    auto has_epilogue = [&](int j) {
        return ggml_can_fuse(cgraph, j, { GGML_OP_MUL_MAT, GGML_OP_ADD }) ||
               ggml_can_fuse(cgraph, j, { GGML_OP_MUL_MAT, GGML_OP_MUL });
    };

    // an FP4 member's per-tensor weight scale, the mul by one F32 value that is its only reader: the GEMM applies it
    // and the member's output is the mul's (vLLM folds the weight's global scale into the GEMM)
    auto fp4_scale_at = [&](int j) -> ggml_tensor * {
        if (!ggml_can_fuse(cgraph, j, { GGML_OP_MUL_MAT, GGML_OP_MUL })) {
            return nullptr;
        }
        ggml_tensor * mm  = cgraph->nodes[j];
        ggml_tensor * mul = cgraph->nodes[j + 1];
        const ggml_tensor * s = mul->src[0] == mm ? mul->src[1] : mul->src[0];
        if (s == nullptr || s->type != GGML_TYPE_F32 || ggml_nelements(s) != 1 || s->buffer == nullptr ||
                ggml_backend_buffer_get_usage(s->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
                mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || !ggml_are_same_shape(mul, mm)) {
            return nullptr;
        }
        return mul;
    };
    auto scale_data = [](const ggml_tensor * mul) -> const float * {
        return (const float *) (mul->src[0]->op == GGML_OP_MUL_MAT ? mul->src[1] : mul->src[0])->data;
    };

    ggml_tensor * a = cgraph->nodes[i];
    const bool fp4 = fp4_ok(a);
    ggml_tensor * a_mul = fp4 ? fp4_scale_at(i) : nullptr;
    if ((!fp4 && !member_ok(a)) || (has_epilogue(i) && !a_mul)) {
        return 0;
    }
    // small weights take their TF32 GEMM one by one (ggml_cuda_small_gemm)
    if (ggml_cuda_small_gemm_applies(a->src[0], a->src[1], a)) {
        return 0;
    }

    const ggml_tensor * mats[MMQ_GROUP_MAX]   = { a->src[0] };
    ggml_tensor *       outs[MMQ_GROUP_MAX]   = { a_mul ? a_mul : a };
    int                 out_at[MMQ_GROUP_MAX] = { a_mul ? i + 1 : i };
    const float *       w_s[MMQ_GROUP_MAX]    = { a_mul ? scale_data(a_mul) : nullptr };
    bool any_scale = a_mul != nullptr;
    int n    = 1;
    int last = out_at[0];
    for (int j = last + 1; j < cgraph->n_nodes && n < MMQ_GROUP_MAX; ++j) {
        ggml_tensor * b = cgraph->nodes[j];
        if (ggml_cuda_is_view_or_noop(b)) {
            continue;
        }
        ggml_tensor * b_mul = fp4 && fp4_ok(b) ? fp4_scale_at(j) : nullptr;
        if ((fp4 ? !fp4_ok(b) : !member_ok(b)) || b->src[1] != a->src[1] || b->src[0]->type != a->src[0]->type ||
                b->src[0]->ne[0] != a->src[0]->ne[0] || (has_epilogue(j) && !b_mul)) {
            break;
        }
        // the FP4 group's parts follow each other in 128-row scale tiles
        if (fp4 && mats[n - 1]->ne[1] % 128 != 0) {
            break;
        }
        mats[n]   = b->src[0];
        outs[n]   = b_mul ? b_mul : b;
        out_at[n] = b_mul ? j + 1 : j;
        w_s[n]    = b_mul ? scale_data(b_mul) : nullptr;
        any_scale = any_scale || b_mul != nullptr;
        n++;
        last = out_at[n - 1];
        j    = last;
    }
    if (n < 2) {
        return 0;
    }

    ggml_cuda_norm_fold_materialize(ctx, a->src[1]);
    if (fp4) {
        // a gate and up pair read only by the GEGLU right after them: the GEMM's epilogue writes gelu(gate)*up and the
        // GLU node is covered too (GGML_CUDA_FUSE_OFF bit 50 keeps them apart)
        if (n == 2 && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_FP4_GEGLU)) {
            int g = last + 1;
            while (g < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[g])) {
                g++;
            }
            ggml_tensor * glu = g < cgraph->n_nodes ? cgraph->nodes[g] : nullptr;
            const int gate_part = glu && glu->src[0] == outs[0] ? 0 : 1;
            if (glu && glu->op == GGML_OP_GLU && ggml_get_glu_op(glu) == GGML_GLU_OP_GEGLU && glu->src[1] != nullptr &&
                    glu->src[0] == outs[gate_part] && glu->src[1] == outs[1 - gate_part] &&
                    glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu) && glu->ne[0] == outs[0]->ne[0] &&
                    outs[0]->ne[0] == outs[1]->ne[0] && ggml_nrows(glu) == ggml_nrows(outs[0]) &&
                    ggml_node_has_n_uses(cgraph, out_at[0], 1) && ggml_node_has_n_uses(cgraph, out_at[1], 1)) {
                // the down projection reading only the GEGLU, through the FP4 GEMM: its activation is quantized here
                const ggml_tensor * down_w  = nullptr;
                const float *       down_ws = nullptr;
                int d = g + 1;
                while (d < cgraph->n_nodes && ggml_cuda_is_view_or_noop(cgraph->nodes[d])) {
                    d++;
                }
                ggml_tensor * dn = d < cgraph->n_nodes ? cgraph->nodes[d] : nullptr;
                if (dn && dn->op == GGML_OP_MUL_MAT && dn->src[1] == glu && dn->src[0]->type == GGML_TYPE_NVFP4 &&
                        ggml_node_has_n_uses(cgraph, g, 1) && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_FP4_DOWN_QUANT) &&
                        (ggml_mul_mat_get_n_seqs(dn) > 1 || ggml_cuda_fp4_gemm_forced()) &&
                        ggml_cuda_fp4_gemm_supported(dn->src[0], glu, dn, cc) && !ggml_cuda_should_use_mmvq(dn->src[0]->type, cc, glu->ne[1]) &&
                        !ggml_cuda_small_gemm_applies(dn->src[0], glu, dn)) {
                    down_w = dn->src[0];
                    ggml_tensor * dn_mul = fp4_scale_at(d);
                    down_ws = dn_mul ? scale_data(dn_mul) : nullptr;
                }
                if (!ggml_cuda_fp4_gemm_group_geglu(ctx, mats, a->src[1], gate_part, glu, any_scale ? w_s : nullptr, down_w, down_ws)) {
                    return 0;
                }
                return g - i;
            }
        }
        if (!ggml_cuda_fp4_gemm_group(ctx, mats, n, a->src[1], outs, any_scale ? w_s : nullptr)) {
            return 0;
        }
        return last - i;
    }
    ggml_cuda_mul_mat_q_group(ctx, mats, n, a->src[1], outs);
    return last - i;
}

// An FP4 GEMM of several sequences' rows and the mul by its weight's per-tensor scale right after it: the scale goes
// into the GEMM's alpha, as vLLM's alpha is the input scale times the weight scale. Returns the nodes covered past i.
static int ggml_cuda_try_fp4_scale(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    ggml_tensor * mm = cgraph->nodes[i];
    if (mm->op != GGML_OP_MUL_MAT || !ggml_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL })) {
        return 0;
    }
    const int cc = ggml_cuda_info().devices[ctx.device].cc;
    const ggml_tensor * src0 = mm->src[0];
    const ggml_tensor * src1 = mm->src[1];
    // the conditions under which ggml_cuda_mul_mat takes the FP4 GEMM
    if (!(ggml_mul_mat_get_n_seqs(mm) > 1 || ggml_cuda_fp4_gemm_forced()) || !ggml_cuda_fp4_gemm_supported(src0, src1, mm, cc) ||
            (ggml_cuda_should_use_mmvq(src0->type, cc, src1->ne[1]) && !ggml_cuda_fp4_big_weight(src0, src1, mm, cc)) ||
            ggml_cuda_small_gemm_applies(src0, src1, mm)) {
        return 0;
    }
    ggml_tensor * mul = cgraph->nodes[i + 1];
    const ggml_tensor * s = mul->src[0] == mm ? mul->src[1] : mul->src[0];
    if (s == nullptr || s->type != GGML_TYPE_F32 || ggml_nelements(s) != 1 || s->buffer == nullptr ||
            ggml_backend_buffer_get_usage(s->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
            mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || !ggml_are_same_shape(mul, mm)) {
        return 0;
    }
    ggml_cuda_norm_fold_materialize(ctx, src1);
    if (!ggml_cuda_fp4_gemm(ctx, src0, src1, mul, (const float *) s->data)) {
        return 0;
    }
    return 1;
}

// rms_norm > mul > add [> mul(scalar)] > rms_norm > mul, the second product read by a GEMM over several sequences'
// rows: the residual stream's norm and add and the next block's input norm in one launch (ggml_cuda_op_rms_norm_pair).
// One sequence's rows keep their launches (their quantized matvecs fold the norm). Returns the nodes covered past i.
// GGML_CUDA_FUSE_OFF bit 51 keeps them apart.
static constexpr uint64_t GGML_CUDA_FUSE_NORM_PAIR = 1ull << 51;

struct ggml_cuda_norm_pair_nodes {
    ggml_tensor * n0 = nullptr;
    ggml_tensor * m0 = nullptr;
    ggml_tensor * a0 = nullptr;
    ggml_tensor * sc = nullptr;
    ggml_tensor * n1 = nullptr;
    ggml_tensor * m1 = nullptr;
    ggml_tensor * mm = nullptr; // the GEMM reading m1
    int n_past = 0; // the nodes covered past the first
};

static bool ggml_cuda_match_rms_norm_pair(const ggml_cgraph * cgraph, int i, ggml_cuda_norm_pair_nodes & p) {
    if (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_NORM_PAIR) {
        return false;
    }
    if (i + 5 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * n0 = cgraph->nodes[i];
    ggml_tensor * m0 = cgraph->nodes[i + 1];
    ggml_tensor * a0 = cgraph->nodes[i + 2];
    if (n0->op != GGML_OP_RMS_NORM || m0->op != GGML_OP_MUL || a0->op != GGML_OP_ADD ||
            !(m0->src[0] == n0 || m0->src[1] == n0) || !(a0->src[0] == m0 || a0->src[1] == m0)) {
        return false;
    }
    int k = i + 3;
    ggml_tensor * sc  = nullptr;
    ggml_tensor * out = a0;
    {
        ggml_tensor * t = cgraph->nodes[k];
        const ggml_tensor * other = t->op != GGML_OP_MUL ? nullptr : t->src[0] == a0 ? t->src[1] : t->src[1] == a0 ? t->src[0] : nullptr;
        if (other && ggml_nelements(other) == 1 && other->type == GGML_TYPE_F32) {
            sc  = t;
            out = t;
            k++;
        }
    }
    if (k + 2 >= cgraph->n_nodes) {
        return false;
    }
    ggml_tensor * n1 = cgraph->nodes[k];
    ggml_tensor * m1 = cgraph->nodes[k + 1];
    ggml_tensor * mm = cgraph->nodes[k + 2];
    if (n1->op != GGML_OP_RMS_NORM || n1->src[0] != out || m1->op != GGML_OP_MUL || !(m1->src[0] == n1 || m1->src[1] == n1)) {
        return false;
    }
    // a GEMM over several sequences' rows reads the second product
    if (mm->op != GGML_OP_MUL_MAT || mm->src[1] != m1 || !ggml_cuda_mmq_batched(mm)) {
        return false;
    }
    // the norms and the first product feed only the next node; the add does too when the scale follows it
    if (ggml_node_get_use_count(cgraph, i) != 1 || ggml_node_get_use_count(cgraph, i + 1) != 1 ||
            (sc && ggml_node_get_use_count(cgraph, i + 2) != 1) || ggml_node_get_use_count(cgraph, k) != 1) {
        return false;
    }
    if ((n0->flags | m0->flags | n1->flags) & GGML_TENSOR_FLAG_OUTPUT || (sc && (a0->flags & GGML_TENSOR_FLAG_OUTPUT))) {
        return false;
    }
    const ggml_tensor * x   = n0->src[0];
    const ggml_tensor * w1  = m0->src[0] == n0 ? m0->src[1] : m0->src[0];
    const ggml_tensor * res = a0->src[0] == m0 ? a0->src[1] : a0->src[0];
    const ggml_tensor * w2  = m1->src[0] == n1 ? m1->src[1] : m1->src[0];
    const int64_t ncols = x->ne[0];
    const int64_t nrows = x->ne[1];
    auto rows_f32 = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->ne[0] == ncols && t->ne[1] == nrows && t->ne[2] == 1 && t->ne[3] == 1 &&
               t->nb[0] == sizeof(float);
    };
    auto weight = [&](const ggml_tensor * t) {
        return t->type == GGML_TYPE_F32 && t->ne[0] == ncols && ggml_nrows(t) == 1 && ggml_is_contiguous(t);
    };
    if (!rows_f32(x) || !rows_f32(res) || !rows_f32(out) || !rows_f32(m1) || !ggml_is_contiguous(out) || !ggml_is_contiguous(m1) ||
            !ggml_are_same_shape(m0, a0) || !weight(w1) || !weight(w2) || ncols > 8*1024) {
        return false;
    }
    p.n0 = n0;
    p.m0 = m0;
    p.a0 = a0;
    p.sc = sc;
    p.n1 = n1;
    p.m1 = m1;
    p.mm = mm;
    p.n_past = k + 1 - i;
    return true;
}

// An FP8 GEMM reading the pair's second product (q, k and v; an FP8 MLP's gate and up) takes its rows quantized by the
// pair, which has their |max| in hand: vLLM's compiled rms_norm with its per-token quant. GGML_CUDA_FUSE_OFF bit 56
static constexpr uint64_t GGML_CUDA_FUSE_NORM_FP8_QUANT = 1ull << 56;

static void ggml_cuda_norm_pair_q8(ggml_backend_cuda_context & ctx, const ggml_cuda_norm_pair_nodes & p, uint8_t ** q8, float ** q8_s,
        const bool may_grow = true) {
    *q8   = nullptr;
    *q8_s = nullptr;
    if ((ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_NORM_FP8_QUANT) || p.mm == nullptr || p.mm->src[0]->type != GGML_TYPE_F8_E4M3 ||
            !ggml_cuda_fp8_act_claim(ctx, p.m1, q8, q8_s, may_grow)) {
        *q8   = nullptr;
        *q8_s = nullptr;
    }
}

static int ggml_cuda_try_rms_norm_pair(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    ggml_cuda_norm_pair_nodes p;
    if (!ggml_cuda_match_rms_norm_pair(cgraph, i, p)) {
        return 0;
    }
    uint8_t * q8   = nullptr;
    float   * q8_s = nullptr;
    ggml_cuda_norm_pair_q8(ctx, p, &q8, &q8_s);
    ggml_cuda_op_rms_norm_pair(ctx, p.n0, p.m0, p.a0, p.sc, p.n1, p.m1, nullptr, nullptr, nullptr, q8, q8_s);
    return p.n_past;
}

// An FP8 projection whose scaled product starts a norm pair (the o projection, an FP8 MLP's down): the product stays
// raw and the pair applies its token and row scales as it reads it, fp8_scale_out's arithmetic, as vLLM's GEMM
// epilogue scales ahead of its fused norm. GGML_CUDA_FUSE_OFF bit 53 keeps them apart.
static constexpr uint64_t GGML_CUDA_FUSE_FP8_NORM = 1ull << 53;

// A fused FP8 q/k/v projection and everything up to its cache writes, in gemma4.cpp's node order: the GEMM, then one
// launch for its scales, the three heads' views, norms, norm weights and ropes and both SET_ROWS
// (ggml_cuda_qkv_norm_rope_store), as vLLM's compiled attention input runs it: 6 kernels a layer -> 2.
// Returns the nodes covered past i. GGML_CUDA_FUSE_OFF bit 52 keeps them apart.
static constexpr uint64_t GGML_CUDA_FUSE_QKV_EPI = 1ull << 52;

static int ggml_cuda_try_qkv_epilogue(ggml_backend_cuda_context & ctx, const ggml_cgraph * cgraph, int i) {
    static const ggml_op ops[] = {
        GGML_OP_MUL_MAT, GGML_OP_MUL,
        GGML_OP_VIEW, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE,
        GGML_OP_VIEW, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE,
        GGML_OP_VIEW, GGML_OP_CONT, GGML_OP_RESHAPE, GGML_OP_RMS_NORM,
    };
    constexpr int n_head_ops = (int) (sizeof(ops)/sizeof(ops[0]));
    if ((ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_QKV_EPI) || i + n_head_ops + 4 > cgraph->n_nodes) {
        return 0;
    }
    ggml_tensor * const * n = cgraph->nodes + i;
    for (int k = 0; k < n_head_ops; ++k) {
        if (n[k]->op != ops[k]) {
            return 0;
        }
    }
    // each cache write: a view of the head's rows, an FP8 cache's SCALE (llama_kv_cache::cpy_k), the cache's reshape
    // when it has a stream per sequence, SET_ROWS
    int at = n_head_ops;
    auto cache_write = [&](int & i_view, int & i_scale, int & i_set) {
        if (i + at + 1 >= cgraph->n_nodes || n[at]->op != GGML_OP_VIEW) {
            return false;
        }
        i_view = at++;
        if (i + at < cgraph->n_nodes && n[at]->op == GGML_OP_SCALE && n[at]->src[0] == n[i_view]) {
            i_scale = at++;
        }
        if (i + at < cgraph->n_nodes && n[at]->op == GGML_OP_RESHAPE) {
            at++;
        }
        if (i + at >= cgraph->n_nodes || n[at]->op != GGML_OP_SET_ROWS) {
            return false;
        }
        i_set = at++;
        return true;
    };
    int ik_view  = -1;
    int ik_scale = -1;
    int ik_set   = -1;
    int iv_view  = -1;
    int iv_scale = -1;
    int iv_set   = -1;
    if (!cache_write(ik_view, ik_scale, ik_set) || !cache_write(iv_view, iv_scale, iv_set)) {
        return 0;
    }
    const int n_ops = at;
    ggml_tensor * mm  = n[0];
    ggml_tensor * mul = n[1];
    if (mm->src[0]->type != GGML_TYPE_F8_E4M3 || (mul->src[0] != mm && mul->src[1] != mm)) {
        return 0;
    }
    const ggml_tensor * s = mul->src[0] == mm ? mul->src[1] : mul->src[0];
    const int64_t ncol  = mm->ne[0];
    const int64_t n_tok = mm->ne[1];
    if (mm->ne[2] != 1 || mm->ne[3] != 1 || mul->type != GGML_TYPE_F32 || !ggml_is_contiguous(mul) || !ggml_are_same_shape(mul, mm) ||
            s->type != GGML_TYPE_F32 || !ggml_is_contiguous(s) || s->buffer == nullptr ||
            ggml_backend_buffer_get_usage(s->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
        return 0;
    }
    const bool per_row = ggml_nelements(s) == ncol;
    if (!per_row && ggml_nelements(s) != 1) {
        return 0;
    }

    // the heads: a view of the product at its offset, its copy, its [hd, heads, tokens] shape and its norm
    const ggml_tensor * rq = n[4];
    const ggml_tensor * rk = n[10];
    const ggml_tensor * rv = n[16];
    const int hd  = (int) rq->ne[0];
    const int n_q = (int) rq->ne[1];
    const int n_k = (int) rk->ne[1];
    if (rk->ne[0] != hd || rv->ne[0] != hd || rv->ne[1] != n_k || rq->ne[2] != n_tok || rk->ne[2] != n_tok || rv->ne[2] != n_tok ||
            (int64_t) (n_q + 2*n_k)*hd != ncol || hd % 2 != 0 || hd >= 1024) {
        return 0;
    }
    auto head = [&](const int b, const int64_t off) {
        return n[b]->src[0] == mul && n[b]->view_offs == (size_t) off*sizeof(float) &&
               n[b + 1]->src[0] == n[b] && n[b + 2]->src[0] == n[b + 1] && n[b + 3]->src[0] == n[b + 2];
    };
    if (!head(2, 0) || !head(8, (int64_t) n_q*hd) || !head(14, (int64_t) (n_q + n_k)*hd)) {
        return 0;
    }
    auto weight = [&](const ggml_tensor * m, const ggml_tensor * norm) -> const ggml_tensor * {
        const ggml_tensor * w = m->src[0] == norm ? m->src[1] : m->src[1] == norm ? m->src[0] : nullptr;
        return w && w->type == GGML_TYPE_F32 && ggml_is_contiguous(w) && ggml_nelements(w) == hd ? w : nullptr;
    };
    const ggml_tensor * wq = weight(n[6], n[5]);
    const ggml_tensor * wk = weight(n[12], n[11]);
    if (!wq || !wk || n[7]->src[0] != n[6] || n[13]->src[0] != n[12] ||
            memcmp(n[7]->op_params, n[13]->op_params, 15*sizeof(int32_t)) != 0 ||
            n[7]->src[1] != n[13]->src[1] || n[7]->src[2] != n[13]->src[2] ||
            n[7]->type != GGML_TYPE_F32 || !ggml_is_contiguous(n[7])) {
        return 0;
    }
    const int mode = ((const int32_t *) n[7]->op_params)[2];
    if (mode == GGML_ROPE_TYPE_MROPE || mode == GGML_ROPE_TYPE_IMROPE || mode == GGML_ROPE_TYPE_VISION) {
        return 0;
    }

    // the cache writes: rows of k's rope and of v's norm into F16 caches
    const ggml_tensor * sk = n[ik_set];
    const ggml_tensor * sv = n[iv_set];
    // F16 caches, or FP8 ones each written through its SCALE
    const bool f8 = sk->type == GGML_TYPE_F8_E4M3;
    const size_t esz = f8 ? 1 : sizeof(half);
    const ggml_tensor * k_rows = ik_scale >= 0 ? n[ik_scale] : n[ik_view];
    const ggml_tensor * v_rows = iv_scale >= 0 ? n[iv_scale] : n[iv_view];
    if (n[ik_view]->src[0] != n[13] || sk->src[0] != k_rows || n[iv_view]->src[0] != n[17] || sv->src[0] != v_rows ||
            sk->type != sv->type || (sk->type != GGML_TYPE_F16 && !f8) || (f8 != (ik_scale >= 0)) || (f8 != (iv_scale >= 0)) ||
            sk->src[1]->type != GGML_TYPE_I64 || sv->src[1]->type != GGML_TYPE_I64 ||
            sk->ne[0] != (int64_t) n_k*hd || sv->ne[0] != (int64_t) n_k*hd || sk->src[1]->ne[0] != n_tok || sv->src[1]->ne[0] != n_tok ||
            sk->nb[0] != esz || sv->nb[0] != esz) {
        return 0;
    }
    for (const int k : { ik_view, ik_scale, iv_view, iv_scale }) {
        if (k >= 0 && ggml_node_get_use_count(cgraph, i + k) != 1) {
            return 0;
        }
    }
    float k_mul = 1.0f, k_add = 0.0f, v_mul = 1.0f, v_add = 0.0f;
    if (f8) {
        memcpy(&k_mul, (const float *) n[ik_scale]->op_params + 0, sizeof(float));
        memcpy(&k_add, (const float *) n[ik_scale]->op_params + 1, sizeof(float));
        memcpy(&v_mul, (const float *) n[iv_scale]->op_params + 0, sizeof(float));
        memcpy(&v_add, (const float *) n[iv_scale]->op_params + 1, sizeof(float));
    }
    // no other reader of what the launch does not write
    for (const int k : { 0, 2, 3, 4, 5, 6, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17 }) {
        if (ggml_node_get_use_count(cgraph, i + k) != 1) {
            return 0;
        }
    }
    if (ggml_node_get_use_count(cgraph, i + 1) != 3) {
        return 0;
    }
    for (int k = 0; k < n_ops; ++k) {
        if (k != 7 && k != ik_set && k != iv_set && (n[k]->flags & GGML_TENSOR_FLAG_OUTPUT)) {
            return 0;
        }
    }

    float * d = (float *) mul->data;
    const float * s_tok = nullptr;
    const float * s_row = per_row ? (const float *) s->data : nullptr;
    const float * s_t   = per_row ? nullptr : (const float *) s->data;
    ggml_cuda_norm_fold_materialize(ctx, mm->src[1]);
    if (!ggml_cuda_fp8_mul_mat_raw(ctx, mm->src[0], mm->src[1], d, &s_tok)) {
        // a few columns: the matvec writes the scaled product
        if (!ggml_cuda_fp8_mul_mat(ctx, mm->src[0], mm->src[1], mul, s_row, s_t)) {
            return 0;
        }
        s_row = nullptr;
        s_t   = nullptr;
    }
    ggml_cuda_qkv_norm_rope_store(ctx, d, ncol, s_tok, s_row, s_t, hd, n_q, n_k, n_tok, n[5], n[11], n[17],
            (const float *) wq->data, (const float *) wk->data, (float *) n[7]->data,
            sk->data, (const int64_t *) sk->src[1]->data, (int64_t) (sk->nb[1]/esz),
            sv->data, (const int64_t *) sv->src[1]->data, (int64_t) (sv->nb[1]/esz), n[7], f8, k_mul, k_add, v_mul, v_add);
    return n_ops - 1;
}

static int ggml_cuda_try_fuse(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, int i) {

    static bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (disable_fusion) {
        return 0;
    }

    ggml_tensor * node = cgraph->nodes[i];

    // an FP8 matmul's weight scales: the mul after it by one value or one per weight row (nullptr: neither)
    auto fp8_scales = [&](const ggml_tensor * mm, const ggml_tensor * mul, const float ** s_row, const float ** s_t) -> bool {
        const ggml_tensor * s = mul->src[0] == mm ? mul->src[1] : mul->src[0];
        if (s == nullptr || s->type != GGML_TYPE_F32 || !ggml_is_contiguous(s) || s->buffer == nullptr ||
                ggml_backend_buffer_get_usage(s->buffer) == GGML_BACKEND_BUFFER_USAGE_COMPUTE ||
                !ggml_are_same_shape(mul, mm) || !ggml_is_contiguous(mul)) {
            return false;
        }
        const bool per_row = ggml_nelements(s) == mm->src[0]->ne[1] && s->ne[0] == mm->src[0]->ne[1];
        if (!per_row && ggml_nelements(s) != 1) {
            return false;
        }
        *s_row = per_row ? (const float *) s->data : nullptr;
        *s_t   = per_row ? nullptr : (const float *) s->data;
        return true;
    };

    if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_F8_E4M3) {
        const int n_qkv = ggml_cuda_try_qkv_epilogue(*cuda_ctx, cgraph, i);
        if (n_qkv > 0) {
            return n_qkv;
        }
    }

    if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_F8_E4M3 && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_FP8_NORM) &&
            i + 2 < cgraph->n_nodes && ggml_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL })) {
        ggml_tensor * mul = cgraph->nodes[i + 1];
        const float * s_row = nullptr;
        const float * s_t   = nullptr;
        ggml_cuda_norm_pair_nodes p;
        if (cgraph->nodes[i + 2]->src[0] == mul && ggml_node_get_use_count(cgraph, i + 1) == 1 && !(mul->flags & GGML_TENSOR_FLAG_OUTPUT) &&
                fp8_scales(node, mul, &s_row, &s_t) && ggml_cuda_match_rms_norm_pair(cgraph, i + 2, p)) {
            ggml_cuda_norm_fold_materialize(*cuda_ctx, node->src[1]);
            const float * s_tok = nullptr;
            if (ggml_cuda_fp8_mul_mat_raw(*cuda_ctx, node->src[0], node->src[1], (float *) mul->data, &s_tok)) {
                // the claim may take the buffer the product's activations were quantized into: the product is queued
                // ahead of the pair, which writes it only after
                uint8_t * q8   = nullptr;
                float   * q8_s = nullptr;
                ggml_cuda_norm_pair_q8(*cuda_ctx, p, &q8, &q8_s, false);
                ggml_cuda_op_rms_norm_pair(*cuda_ctx, p.n0, p.m0, p.a0, p.sc, p.n1, p.m1, s_tok, s_row, s_t, q8, q8_s);
                return 2 + p.n_past;
            }
            // a few columns: the matvec writes the scaled product
            if (ggml_cuda_fp8_mul_mat(*cuda_ctx, node->src[0], node->src[1], mul, s_row, s_t)) {
                ggml_cuda_op_rms_norm_pair(*cuda_ctx, p.n0, p.m0, p.a0, p.sc, p.n1, p.m1);
                return 2 + p.n_past;
            }
        }
    }

    // an FP8 up and gate pair over one activation, each with its scales, read only by the GEGLU after them
    if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_F8_E4M3 && i + 4 < cgraph->n_nodes &&
            ggml_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL }) && ggml_can_fuse(cgraph, i + 2, { GGML_OP_MUL_MAT, GGML_OP_MUL })) {
        ggml_tensor * m0  = cgraph->nodes[i + 1];
        ggml_tensor * b   = cgraph->nodes[i + 2];
        ggml_tensor * m1  = cgraph->nodes[i + 3];
        ggml_tensor * glu = cgraph->nodes[i + 4];
        const float * s0_row = nullptr, * s0_t = nullptr, * s1_row = nullptr, * s1_t = nullptr;
        if (b->src[0]->type == GGML_TYPE_F8_E4M3 && b->src[1] == node->src[1] && glu->op == GGML_OP_GLU &&
                ggml_get_glu_op(glu) == GGML_GLU_OP_GEGLU && glu->src[1] != nullptr && ggml_get_op_params_i32(glu, 1) == 0 &&
                ((glu->src[0] == m0 && glu->src[1] == m1) || (glu->src[0] == m1 && glu->src[1] == m0)) &&
                ggml_node_get_use_count(cgraph, i + 1) == 1 && ggml_node_get_use_count(cgraph, i + 3) == 1 &&
                !((m0->flags | m1->flags) & GGML_TENSOR_FLAG_OUTPUT) &&
                glu->type == GGML_TYPE_F32 && ggml_is_contiguous(glu) &&
                fp8_scales(node, m0, &s0_row, &s0_t) && fp8_scales(b, m1, &s1_row, &s1_t)) {
            const bool gate_first = glu->src[0] == m0;
            // an FP8 down projection right after: the GEGLU quantizes its rows for it
            const ggml_tensor * dn = i + 5 < cgraph->n_nodes ? cgraph->nodes[i + 5] : nullptr;
            const bool down_fp8 = dn && dn->op == GGML_OP_MUL_MAT && dn->src[1] == glu && dn->src[0]->type == GGML_TYPE_F8_E4M3;
            ggml_cuda_norm_fold_materialize(*cuda_ctx, node->src[1]);
            if (ggml_cuda_fp8_gemm_geglu(*cuda_ctx, gate_first ? node->src[0] : b->src[0], gate_first ? b->src[0] : node->src[0],
                    node->src[1], glu, gate_first ? s0_row : s1_row, gate_first ? s1_row : s0_row,
                    gate_first ? s0_t : s1_t, gate_first ? s1_t : s0_t, down_fp8)) {
                return 4;
            }
        }
    }

    // an FP8 matmul and the mul by its weight's scales (a row each, or one) after it: the kernels apply them
    if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_F8_E4M3 && ggml_can_fuse(cgraph, i, { GGML_OP_MUL_MAT, GGML_OP_MUL })) {
        ggml_tensor * mul = cgraph->nodes[i + 1];
        const ggml_tensor * s = mul->src[0] == node ? mul->src[1] : mul->src[0];
        const bool per_row = s && ggml_nelements(s) == node->src[0]->ne[1] && s->ne[0] == node->src[0]->ne[1];
        if (s && s->type == GGML_TYPE_F32 && ggml_is_contiguous(s) && (per_row || ggml_nelements(s) == 1) && s->buffer &&
                ggml_backend_buffer_get_usage(s->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE && ggml_are_same_shape(mul, node) &&
                ggml_is_contiguous(mul)) {
            ggml_cuda_norm_fold_materialize(*cuda_ctx, node->src[1]);
            if (ggml_cuda_fp8_mul_mat(*cuda_ctx, node->src[0], node->src[1], mul,
                    per_row ? (const float *) s->data : nullptr, per_row ? nullptr : (const float *) s->data)) {
                return 1;
            }
        }
    }

    // a residual add folded into the norm that reads it
    if (node->op == GGML_OP_ADD && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_NORM_FOLD)) {
        const int nodes_to_skip = ggml_cuda_try_add_rms_norm_mul(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_GET_ROWS && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_GET_ROWS_CPY)) {
        const int nodes_to_skip = ggml_cuda_try_get_rows_cpy(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_MUL_MAT_ID) {
        int nodes_to_skip = ggml_cuda_try_moe_router_weights(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
        nodes_to_skip = ggml_cuda_try_expert_scale_pass(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    // an NVFP4 projection's scale and the residual after it, before the scale alone is taken below
    if (node->op == GGML_OP_MUL_MAT && node->src[0]->type == GGML_TYPE_NVFP4 &&
            !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_MM_RESHAPE_ADD)) {
        const int nodes_to_skip = ggml_cuda_try_mm_reshape_add(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_L2_NORM) {
        const int nodes_to_skip = ggml_cuda_try_gdn_qk_norm(cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_RMS_NORM) {
        const int nodes_to_skip = ggml_cuda_try_gated_norm(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_UNARY) {
        const int nodes_to_skip = ggml_cuda_try_gated_fold(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_MUL_MAT) {
        const int nodes_to_skip = ggml_cuda_try_conv_proj(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_MUL_MAT) {
        const int nodes_to_skip = ggml_cuda_try_mmvf_pair_epi(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_CPY) {
        const int nodes_to_skip = ggml_cuda_try_cpy_run(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_MUL) {
        ggml_cuda_moe_weighted_reduction_match match;
        if (ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
            const int output_idx = i + match.node_count - 1;
            if (ggml_cuda_check_fusion_memory_ranges(cgraph, i, match.node_count, &output_idx, 1)) {
                ggml_cuda_op_moe_weighted_reduction(
                    *cuda_ctx, match.experts, match.expert_scale, match.weights, match.dst);
                return match.node_count - 1;
            }
        }
    }

    // gated_delta_net -> cpy: scatter recurrent-state snapshots into the cache
    if (node->op == GGML_OP_GATED_DELTA_NET) {
        ggml_cuda_gated_delta_net_fused_cache fused_state_cpy;
        const int nodes_to_skip = ggml_cuda_try_gdn_cache_fusion(cgraph, i, fused_state_cpy);
        if (nodes_to_skip > 0) {
#ifdef GGML_CUDA_DEBUG
            GGML_LOG_INFO("%s: fused gated_delta_net snapshot copies for %s (skipped %d nodes)\n",
                          __func__, node->name, nodes_to_skip);
#endif
            ggml_cuda_op_gated_delta_net_fused_cache(*cuda_ctx, node, fused_state_cpy);
            return nodes_to_skip;
        }
    }




    if (node->op == GGML_OP_FLASH_ATTN_EXT) {
        const int nodes_to_skip = ggml_cuda_try_fa_out_scale(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_SCALE) {
        int nodes_to_skip = ggml_cuda_try_softcap_argmax(*cuda_ctx, cgraph, i);
        if (nodes_to_skip == 0) {
            nodes_to_skip = ggml_cuda_try_softcap_topk(*cuda_ctx, cgraph, i);
        }
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    // any run of elementwise ops, on every model
    if (ggml_cuda_chain_op_of(node) != 0) {
        const int nodes_to_skip = ggml_cuda_try_elt_chain(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    //topk-moe
    if (cgraph->nodes[i]->op == GGML_OP_UNARY || cgraph->nodes[i]->op == GGML_OP_SOFT_MAX ||
            cgraph->nodes[i]->op == GGML_OP_ARGSORT) {
        ggml_cuda_topk_moe_args args;
        const bool              can_fuse = ggml_cuda_topk_moe_fusion(cgraph, i, args);
        std::vector<ggml_op>    ops;

        if (can_fuse) {
            const ggml_tensor * logits  = node->src[0];
            ggml_tensor *       weights = nullptr;
            ggml_tensor *       ids     = nullptr;
            const ggml_tensor * bias    = nullptr;
            const ggml_tensor * clamp   = nullptr;
            const ggml_tensor * scale   = nullptr;

            if (!args.delayed_softmax) {
                int out_nodes[2];  // nodes which can't be elided

                if (args.sigmoid) {
                    ops.insert(ops.end(), { GGML_OP_UNARY });
                } else if (args.sqrt_softplus) {
                    ops.insert(ops.end(), { GGML_OP_UNARY, GGML_OP_SQRT });
                } else {
                    ops.insert(ops.end(), { GGML_OP_SOFT_MAX });
                }
                const int i_probs = i + (int) ops.size() - 1;  // last node of the gating activation

                if (args.prob_bias) {
                    bias = cgraph->nodes[i_probs + 2]->src[1];
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ADD, GGML_OP_ARGSORT, GGML_OP_VIEW,
                                            GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 4;
                } else {
                    ops.insert(ops.end(), { GGML_OP_RESHAPE, GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS });
                    out_nodes[0] = i_probs + 3;
                }
                ids = cgraph->nodes[out_nodes[0]];

                if (args.norm) {
                    ops.insert(ops.end(),
                               { GGML_OP_RESHAPE, GGML_OP_SUM_ROWS, GGML_OP_CLAMP, GGML_OP_DIV, GGML_OP_RESHAPE });
                    clamp = cgraph->nodes[i + ops.size() - 3];
                }
                if (args.scale) {
                    ops.insert(ops.end(), { GGML_OP_SCALE });
                    scale = cgraph->nodes[i + ops.size() - 1];
                }

                weights      = cgraph->nodes[i + ops.size() - 1];
                out_nodes[1] = i + ops.size() - 1;

                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(node, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            } else if (!args.norm && !args.prob_bias) {
                //special case gpt-oss, no norm, no bias.
                ops.insert(ops.end(), { GGML_OP_ARGSORT, GGML_OP_VIEW, GGML_OP_GET_ROWS, GGML_OP_RESHAPE,
                                        GGML_OP_SOFT_MAX, GGML_OP_RESHAPE });
                weights                     = cgraph->nodes[i + 5];
                ids                         = cgraph->nodes[i + 1];
                const ggml_tensor * softmax = cgraph->nodes[i + 4];

                int out_nodes[2] = { i + 1, i + 5 };
                if (ggml_can_fuse_subgraph(cgraph, i, ops.size(), ops.data(), out_nodes, 2) &&
                        ggml_cuda_should_use_topk_moe(softmax, logits, weights, ids) &&
                        ggml_cuda_check_fusion_memory_ranges(cgraph, i, ops.size(), out_nodes, 2, /*is_topk_moe=*/true)) {
                    ggml_cuda_op_topk_moe(*cuda_ctx, logits, weights, ids, clamp, scale, bias, args);
                    return ops.size() - 1;
                }
            }
        }
    }

    //RoPE + view + set-rows
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_tensor * rope     = cgraph->nodes[i];
        ggml_tensor * set_rows = cgraph->nodes[i + 2];

        ggml_cuda_op_rope_fused(*cuda_ctx, rope, set_rows);
        return 2;
    }

    // Snake activation: y = x + sin(a*x)^2 * inv_b
    // Naive 5-op decomposition emitted by frontends: mul -> sin -> sqr -> mul -> add
    if (ggml_can_fuse_subgraph(cgraph, i,
            { GGML_OP_MUL, GGML_OP_SIN, GGML_OP_SQR, GGML_OP_MUL, GGML_OP_ADD },
            { i + 4 })) {
        const ggml_tensor * mul0 = cgraph->nodes[i];
        const ggml_tensor * sqr  = cgraph->nodes[i + 2];
        const ggml_tensor * mul1 = cgraph->nodes[i + 3];
        ggml_tensor *       add  = cgraph->nodes[i + 4];

        // x carries the full activation shape, a is the broadcast operand
        const ggml_tensor * x = ggml_are_same_shape(mul0, mul0->src[0]) ? mul0->src[0] : mul0->src[1];
        const ggml_tensor * a = (x == mul0->src[0]) ? mul0->src[1] : mul0->src[0];

        // mul1 reads sqr and inv_b in either operand order
        const ggml_tensor * inv_b = (mul1->src[0] == sqr) ? mul1->src[1] : mul1->src[0];

        // closure check: the trailing add must read the same x as the leading mul
        const ggml_tensor * x_in_add = (add->src[0] == mul1) ? add->src[1] : add->src[0];

        // Kernel iterates over total = T * C, so x and add must be 2D and
        // a / inv_b must collapse to [1, C, 1, 1]. Higher dims are not handled.
        const bool dim_ok   = (x->ne[2]   == 1 && x->ne[3]   == 1) &&
                              (add->ne[2] == 1 && add->ne[3] == 1) &&
                              (a->ne[2]   == 1 && a->ne[3]   == 1);
        const bool shape_ok = ggml_are_same_shape(a, inv_b) && a->ne[0] == 1 && a->ne[1] == x->ne[1];

        // x is in the supported whitelist and every chain intermediate shares
        // x's type. launch_snake reads a and inv_b as const float *, so they
        // stay F32.
        const ggml_tensor * sin1 = cgraph->nodes[i + 1];
        const bool types_ok = (x->type == GGML_TYPE_F32 || x->type == GGML_TYPE_F16 || x->type == GGML_TYPE_BF16) &&
                              (a->type    == GGML_TYPE_F32) && (inv_b->type == GGML_TYPE_F32) &&
                              (mul0->type == x->type) && (sin1->type  == x->type) &&
                              (sqr->type  == x->type) && (mul1->type  == x->type) &&
                              (add->type  == x->type);

        // kernel reads x[idx] and a[c] / inv_b[c] linearly, so every operand is contiguous
        const bool contig_ok = ggml_is_contiguous(x) && ggml_is_contiguous(add) &&
                               ggml_is_contiguous(a) && ggml_is_contiguous(inv_b);

        if (types_ok && shape_ok && dim_ok && contig_ok && x_in_add == x) {
            ggml_cuda_op_snake_fused(*cuda_ctx, x, a, inv_b, add);
            return 4;
        }
    }

    // multi-(add or mul)
    if (node->op == GGML_OP_ADD || node->op == GGML_OP_MUL) {
        int     n_fuse = 0;
        ggml_op ops[8];
        std::fill(ops, ops + 8, node->op);

        for (; n_fuse <= 6; ++n_fuse) {
            if (!ggml_can_fuse(cgraph, i + n_fuse, ops + n_fuse, 2)) {
                break;
            }
            if (cgraph->nodes[i + n_fuse] != cgraph->nodes[i + n_fuse + 1]->src[0]) {
                break;
            }
            if (!ggml_are_same_layout(cgraph->nodes[i + n_fuse]->src[1], cgraph->nodes[i + n_fuse + 1]->src[1])) {
                break;
            }
        }

        n_fuse++;

        if (n_fuse > 1) {
            ggml_tensor fused_node;
            memcpy(&fused_node, node, sizeof(ggml_tensor));
            for (int j = 0; j < n_fuse - 1; ++j) {
                fused_node.src[j + 2] = cgraph->nodes[i + j + 1]->src[1];
            }
            fused_node.data = cgraph->nodes[i + n_fuse - 1]->data;
            if (node->op == GGML_OP_ADD) {
                ggml_cuda_op_fused_add(*cuda_ctx, &fused_node, n_fuse);
            } else {
                ggml_cuda_op_fused_mul(*cuda_ctx, &fused_node, n_fuse);
            }
            return n_fuse - 1;
        }
    }

    if (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) {
        const int nodes_to_skip = ggml_cuda_try_merged_glu(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    bool fused_mul_mat_vec = false;
    int  fused_node_count  = 0;

    auto get_mul_mat_scale = [](const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        const bool scale_lhs_mm = scale_node->src[0] == mm_node;
        const bool scale_rhs_mm = scale_node->src[1] == mm_node;
        if (!scale_lhs_mm && !scale_rhs_mm) {
            return nullptr;
        }

        const ggml_tensor * scale = scale_lhs_mm ? scale_node->src[1] : scale_node->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != 1 ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_mul_mat_id_scale = [](const ggml_tensor * reshape, const ggml_tensor * repeat, const ggml_tensor * getrows,
            const ggml_tensor * scale_node, const ggml_tensor * mm_node) -> const ggml_tensor * {
        if (repeat->src[0] != reshape || getrows->src[0] != repeat || getrows->src[1] != mm_node->src[2]) {
            return nullptr;
        }
        if (!((scale_node->src[0] == mm_node && scale_node->src[1] == getrows) ||
                (scale_node->src[0] == getrows && scale_node->src[1] == mm_node))) {
            return nullptr;
        }

        const ggml_tensor * scale = reshape->src[0];
        if (mm_node->src[0]->type != GGML_TYPE_NVFP4 || scale_node->type != GGML_TYPE_F32 ||
                scale->type != GGML_TYPE_F32 || !ggml_is_contiguous(scale) || ggml_nelements(scale) != mm_node->src[0]->ne[2] ||
                !ggml_are_same_shape(scale_node, mm_node)) {
            return nullptr;
        }

        return scale;
    };

    auto get_bias_tensor = [](const ggml_tensor * bias_node, const ggml_tensor * mul_node, ggml_op op_bias) -> const ggml_tensor * {
        if (op_bias == GGML_OP_ADD) {
            if (bias_node->src[0] == mul_node) {
                return bias_node->src[1];
            }
            if (bias_node->src[1] == mul_node) {
                return bias_node->src[0];
            }
            return nullptr;
        }
        GGML_ASSERT(op_bias == GGML_OP_ADD_ID);
        GGML_ASSERT(bias_node->src[0] == mul_node);
        return bias_node->src[1];
    };

    // gate + glu + up, with optional scale/bias on both lanes.
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (op == GGML_OP_MUL_MAT) {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 1;
                const int gate_bias_idx  = with_bias ? i + 2 : -1;
                const int up_idx         = with_bias ? i + 3 : i + 2;
                const int up_scale_idx   = up_idx + 1;
                const int up_bias_idx    = with_bias ? up_idx + 2 : -1;
                const int glu_idx        = with_bias ? up_idx + 3 : up_idx + 2;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[7];
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                    ops[3] = op;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                    ops[6] = GGML_OP_GLU;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = op;
                    ops[3] = GGML_OP_MUL;
                    ops[4] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 7 : 5;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_scale(gate_scale_n, gate_n);
                const ggml_tensor * up_scale   = get_mul_mat_scale(up_scale_n, up_n);
                // the multi-token expert kernel carries the gate alone, not the per-expert scales
                if (!gate_scale || !up_scale || up_n->ne[2] != 1) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;
                if (with_bias && (!ggml_are_same_shape(gate_out_n->src[0], gate_out_n->src[1]) ||
                        !ggml_are_same_shape(up_out_n->src[0], up_out_n->src[1]))) {
                    continue;
                }

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data, ggml_cuda_norm_fold_for(*cuda_ctx, src1)); ggml_cuda_norm_fold_check_at(*cuda_ctx, src1);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        } else {
            for (const bool with_bias : { false, true }) {
                const int gate_idx       = i;
                const int gate_scale_idx = i + 4;
                const int gate_bias_idx  = with_bias ? i + 5 : -1;
                const int up_idx         = with_bias ? i + 6 : i + 5;
                const int up_scale_idx   = up_idx + 4;
                const int up_bias_idx    = with_bias ? up_idx + 5 : -1;
                const int glu_idx        = with_bias ? up_idx + 6 : up_idx + 5;

                const int out_nodes[] = { glu_idx };
                ggml_op ops[13];
                if (with_bias) {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = bias_op;
                    ops[6]  = op;
                    ops[7]  = GGML_OP_RESHAPE;
                    ops[8]  = GGML_OP_REPEAT;
                    ops[9]  = GGML_OP_GET_ROWS;
                    ops[10] = GGML_OP_MUL;
                    ops[11] = bias_op;
                    ops[12] = GGML_OP_GLU;
                } else {
                    ops[0]  = op;
                    ops[1]  = GGML_OP_RESHAPE;
                    ops[2]  = GGML_OP_REPEAT;
                    ops[3]  = GGML_OP_GET_ROWS;
                    ops[4]  = GGML_OP_MUL;
                    ops[5]  = op;
                    ops[6]  = GGML_OP_RESHAPE;
                    ops[7]  = GGML_OP_REPEAT;
                    ops[8]  = GGML_OP_GET_ROWS;
                    ops[9]  = GGML_OP_MUL;
                    ops[10] = GGML_OP_GLU;
                }
                const int n_ops = with_bias ? 13 : 11;

                if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                        !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                    continue;
                }

                ggml_tensor * gate_n       = cgraph->nodes[gate_idx];
                ggml_tensor * gate_scale_n = cgraph->nodes[gate_scale_idx];
                ggml_tensor * gate_out_n   = with_bias ? cgraph->nodes[gate_bias_idx] : gate_scale_n;
                ggml_tensor * up_n         = cgraph->nodes[up_idx];
                ggml_tensor * up_scale_n   = cgraph->nodes[up_scale_idx];
                ggml_tensor * up_out_n     = with_bias ? cgraph->nodes[up_bias_idx] : up_scale_n;
                const ggml_tensor * glu = cgraph->nodes[glu_idx];

                if (!ggml_cuda_should_fuse_mul_mat(up_n, gate_n, glu,
                        with_bias ? up_out_n : nullptr, with_bias ? gate_out_n : nullptr, up_scale_n, gate_scale_n)) {
                    continue;
                }

                const ggml_tensor * gate_scale = get_mul_mat_id_scale(cgraph->nodes[gate_idx + 1], cgraph->nodes[gate_idx + 2],
                        cgraph->nodes[gate_idx + 3], gate_scale_n, gate_n);
                const ggml_tensor * up_scale = get_mul_mat_id_scale(cgraph->nodes[up_idx + 1], cgraph->nodes[up_idx + 2],
                        cgraph->nodes[up_idx + 3], up_scale_n, up_n);
                // the multi-token expert kernel takes no biases
                if (!gate_scale || !up_scale || ((with_bias || (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_EXPERT_SCALE_MT)) && up_n->ne[2] != 1)) {
                    continue;
                }

                const ggml_tensor * up_bias   = with_bias ? get_bias_tensor(up_out_n, up_scale_n, bias_op) : nullptr;
                const ggml_tensor * gate_bias = with_bias ? get_bias_tensor(gate_out_n, gate_scale_n, bias_op) : nullptr;

                const ggml_tensor * src0 = up_n->src[0];
                const ggml_tensor * src1 = up_n->src[1];
                const ggml_tensor * ids  = up_n->src[2];

                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate       = gate_n->src[0];
                fusion_data.x_bias     = up_bias;
                fusion_data.gate_bias  = gate_bias;
                fusion_data.x_scale    = up_scale;
                fusion_data.gate_scale = gate_scale;
                fusion_data.glu_op     = ggml_get_glu_op(glu);
                fusion_data.glu_limit  = ggml_get_op_params_f32(glu, 3);

                if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                    ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, cgraph->nodes[glu_idx], &fusion_data, ggml_cuda_norm_fold_for(*cuda_ctx, src1)); ggml_cuda_norm_fold_check_at(*cuda_ctx, src1);
                    fused_mul_mat_vec = true;
                    fused_node_count  = n_ops;
                    break;
                }
            }

            if (fused_mul_mat_vec) {
                break;
            }
        }

        if ((op != GGML_OP_MUL_MAT_ID || cgraph->nodes[i]->ne[2] == 1) &&
            ggml_cuda_can_fuse(cgraph, i, { op, bias_op, op, bias_op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu         = cgraph->nodes[i + 4];
            ggml_tensor * gate_bias_n = glu->src[0];
            ggml_tensor * up_bias_n   = glu->src[1];

            //we don't assume the order for {gate, up}. Instead infer it from the bias tensor
            ggml_tensor * gate_n = nullptr;
            ggml_tensor * up_n   = nullptr;

            if (gate_bias_n->src[0] == cgraph->nodes[i] || gate_bias_n->src[1] == cgraph->nodes[i]) {
                gate_n = cgraph->nodes[i];
                up_n   = cgraph->nodes[i + 2];
            } else if (gate_bias_n->src[0] == cgraph->nodes[i + 2] || gate_bias_n->src[1] == cgraph->nodes[i + 2]) {
                gate_n = cgraph->nodes[i + 2];
                up_n   = cgraph->nodes[i];
            } else {
                continue;
            }

            const ggml_tensor * up_bias_tensor   = get_bias_tensor(up_bias_n, up_n, bias_op);
            const ggml_tensor * gate_bias_tensor = get_bias_tensor(gate_bias_n, gate_n, bias_op);

            if (!up_bias_tensor || !gate_bias_tensor) {
                continue;
            }

            // we don't support repeating adds
            if (bias_op == GGML_OP_ADD && (!ggml_are_same_shape(gate_bias_n->src[0], gate_bias_n->src[1]) ||
                                           !ggml_are_same_shape(up_bias_n->src[0], up_bias_n->src[1]))) {
                continue;
            }

            const ggml_tensor * src0 = up_n->src[0];
            const ggml_tensor * src1 = up_n->src[1];
            const ggml_tensor * ids  = up_n->src[2];

            if (ggml_cuda_should_fuse_mul_mat_vec_f(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data, ggml_cuda_norm_fold_for(*cuda_ctx, src1)); ggml_cuda_norm_fold_check_at(*cuda_ctx, src1);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up_n)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate_n->src[0];
                fusion_data.x_bias    = up_bias_tensor;
                fusion_data.gate_bias = gate_bias_tensor;
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data, ggml_cuda_norm_fold_for(*cuda_ctx, src1)); ggml_cuda_norm_fold_check_at(*cuda_ctx, src1);
                fused_mul_mat_vec = true;
                fused_node_count  = 5;
                break;
            }
        } else if (ggml_cuda_can_fuse(cgraph, i, { op, op, GGML_OP_GLU }, {})) {
            ggml_tensor * glu  = cgraph->nodes[i + 2];
            ggml_tensor * gate = glu->src[0];
            ggml_tensor * up   = glu->src[1];

            bool ok = (gate == cgraph->nodes[i] && up == cgraph->nodes[i + 1]) ||
                      (gate == cgraph->nodes[i + 1] && up == cgraph->nodes[i]);

            if (!ok) {
                continue;
            }

            const ggml_tensor * src0 = up->src[0];
            const ggml_tensor * src1 = up->src[1];
            const ggml_tensor * ids  = up->src[2];

            const bool gate_cols = ggml_cuda_should_fuse_mmvf_gate_cols(up) && gate->src[1] == up->src[1] &&
                                   gate->src[0]->type == up->src[0]->type && ggml_are_same_shape(gate->src[0], up->src[0]) &&
                                   ggml_are_same_stride(gate->src[0], up->src[0]) && ggml_are_same_shape(gate, up);
            if (ggml_cuda_should_fuse_mul_mat_vec_f(up) || gate_cols) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, glu, &fusion_data, ggml_cuda_norm_fold_for(*cuda_ctx, src1)); ggml_cuda_norm_fold_check_at(*cuda_ctx, src1);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }

            if (ggml_cuda_should_fuse_mul_mat_vec_q(up)) {
                ggml_cuda_mm_fusion_args_host fusion_data{};
                fusion_data.gate      = gate->src[0];
                fusion_data.glu_op    = ggml_get_glu_op(glu);
                fusion_data.glu_limit = ggml_get_op_params_f32(glu, 3);

                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, glu, &fusion_data, ggml_cuda_norm_fold_for(*cuda_ctx, src1)); ggml_cuda_norm_fold_check_at(*cuda_ctx, src1);
                fused_mul_mat_vec = true;
                fused_node_count  = 3;
                break;
            }
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (node->op == GGML_OP_MUL_MAT) {
        const int nodes_to_skip = ggml_cuda_try_glu_pair_at(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    fused_mul_mat_vec = false;
    fused_node_count  = 0;

    // mul_mat + scale + optional bias
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        for (const bool with_bias : { false, true }) {
            const int n_ops = op == GGML_OP_MUL_MAT ? (with_bias ? 3 : 2) : (with_bias ? 6 : 5);
            const int out_nodes[] = { i + n_ops - 1 };
            ggml_op ops[6];
            if (op == GGML_OP_MUL_MAT) {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                    ops[2] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_MUL;
                }
            } else {
                if (with_bias) {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                    ops[5] = bias_op;
                } else {
                    ops[0] = op;
                    ops[1] = GGML_OP_RESHAPE;
                    ops[2] = GGML_OP_REPEAT;
                    ops[3] = GGML_OP_GET_ROWS;
                    ops[4] = GGML_OP_MUL;
                }
            }

            if (!ggml_can_fuse_subgraph(cgraph, i, n_ops, ops, out_nodes, 1) ||
                    !ggml_cuda_check_fusion_memory_ranges(cgraph, i, n_ops, out_nodes, 1)) {
                continue;
            }

            ggml_tensor * mm_node    = cgraph->nodes[i];
            ggml_tensor * scale_node = op == GGML_OP_MUL_MAT ? cgraph->nodes[i + 1] : cgraph->nodes[i + 4];
            ggml_tensor * out_node   = with_bias ? cgraph->nodes[i + n_ops - 1] : scale_node;

            const ggml_tensor * scale = nullptr;
            if (op == GGML_OP_MUL_MAT) {
                scale = get_mul_mat_scale(scale_node, mm_node);
            } else {
                scale = get_mul_mat_id_scale(cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3], scale_node, mm_node);
            }
            // the multi-token expert kernel takes no biases
            if (!scale || ((with_bias || (ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_EXPERT_SCALE_MT)) && op == GGML_OP_MUL_MAT_ID && mm_node->ne[2] != 1)) {
                continue;
            }

            const ggml_tensor * bias = with_bias ? get_bias_tensor(out_node, scale_node, bias_op) : nullptr;
            if (with_bias && !bias) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD && !ggml_are_same_shape(out_node->src[0], out_node->src[1])) {
                continue;
            }
            if (with_bias && bias_op == GGML_OP_ADD_ID && out_node->src[2] != mm_node->src[2]) {
                continue;
            }

            const ggml_tensor * src0 = mm_node->src[0];
            const ggml_tensor * src1 = mm_node->src[1];
            const ggml_tensor * ids  = mm_node->src[2];

            ggml_cuda_mm_fusion_args_host fusion_data{};
            fusion_data.x_bias  = bias;
            fusion_data.x_scale = scale;

            if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
                ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, out_node, &fusion_data, ggml_cuda_norm_fold_for(*cuda_ctx, src1)); ggml_cuda_norm_fold_check_at(*cuda_ctx, src1);
                fused_mul_mat_vec = true;
                fused_node_count  = n_ops;
                break;
            }
        }
        if (fused_mul_mat_vec) {
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (node->op == GGML_OP_MUL_MAT) {
        const int nodes_to_skip = ggml_cuda_try_mm_epilogue(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    if (node->op == GGML_OP_MUL_MAT && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_MM_RESHAPE_ADD)) {
        const int nodes_to_skip = ggml_cuda_try_mm_reshape_add(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
    }

    // mul_mat + add
    for (ggml_op op : { GGML_OP_MUL_MAT, GGML_OP_MUL_MAT_ID }) {
        const ggml_op bias_op = op == GGML_OP_MUL_MAT ? GGML_OP_ADD : GGML_OP_ADD_ID;

        if (!ggml_can_fuse(cgraph, i, { op, bias_op })) {
            continue;
        }

        ggml_tensor * mm_node   = cgraph->nodes[i];
        ggml_tensor * bias_node = cgraph->nodes[i + 1];

        ggml_tensor * bias_tensor = nullptr;
        if (bias_op == GGML_OP_ADD) {
            if (bias_node->src[0] == mm_node) {
                bias_tensor = bias_node->src[1];
            } else if (bias_node->src[1] == mm_node) {
                bias_tensor = bias_node->src[0];
            } else {
                continue;
            }
        } else {
            if (bias_node->src[0] != mm_node) {
                continue;
            }
            bias_tensor = bias_node->src[1];
        }

        const ggml_tensor * src0 = mm_node->src[0];
        const ggml_tensor * src1 = mm_node->src[1];
        const ggml_tensor * ids  = mm_node->src[2];

        if (bias_op == GGML_OP_ADD_ID && bias_node->src[2] != ids) {
            continue;
        }

        if (bias_op == GGML_OP_ADD && !ggml_are_same_shape(bias_node->src[0], bias_node->src[1])) {
            continue;
        }

        ggml_cuda_mm_fusion_args_host fusion_data{};
        fusion_data.x_bias = bias_tensor;

        if (ggml_cuda_should_fuse_mul_mat_vec_f(mm_node)) {
            ggml_cuda_mul_mat_vec_f(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data, ggml_cuda_norm_fold_for(*cuda_ctx, src1)); ggml_cuda_norm_fold_check_at(*cuda_ctx, src1);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }

        if (ggml_cuda_should_fuse_mul_mat_vec_q(mm_node)) {
            ggml_cuda_mul_mat_vec_q(*cuda_ctx, src0, src1, ids, bias_node, &fusion_data, ggml_cuda_norm_fold_for(*cuda_ctx, src1)); ggml_cuda_norm_fold_check_at(*cuda_ctx, src1);
            fused_mul_mat_vec = true;
            fused_node_count  = 2;
            break;
        }
    }

    if (fused_mul_mat_vec) {
        return fused_node_count - 1;
    }

    if (node->op == GGML_OP_MUL_MAT) {
        const int nodes_to_skip = ggml_cuda_try_mmvq_pair(*cuda_ctx, cgraph, i);
        if (nodes_to_skip > 0) {
            return nodes_to_skip;
        }
        const int n_group = ggml_cuda_try_mmq_group(*cuda_ctx, cgraph, i);
        if (n_group > 0) {
            return n_group;
        }
        const int n_scale = ggml_cuda_try_fp4_scale(*cuda_ctx, cgraph, i);
        if (n_scale > 0) {
            return n_scale;
        }
    }

    if (node->op == GGML_OP_RMS_NORM) {
        const int n_pair = ggml_cuda_try_rms_norm_pair(*cuda_ctx, cgraph, i);
        if (n_pair > 0) {
            return n_pair;
        }
    }

    // rms_norm > mul(weight) read only by quantized matvecs: neither is
    // launched, the matvecs apply the norm while quantizing their input
    if (node->op == GGML_OP_RMS_NORM && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_NORM_FOLD) && ggml_cuda_try_norm_fold(cuda_ctx, cgraph, i)) {
        if (ggml_cuda_norm_fold_diag()) {
            ggml_cuda_norm_fold_materialize(*cuda_ctx, cgraph->nodes[i + 1]);
        }
        ggml_cuda_norm_fold_diag2(*cuda_ctx, cgraph->nodes[i + 1]);
        return 1;
    }

    if (!(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_ROPE_HADAMARD) &&
            ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_RESHAPE, GGML_OP_MUL_MAT, GGML_OP_RESHAPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 7], cgraph->nodes[i + 4]);
        return 7;
    }

    if (!(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_ROPE_HADAMARD) &&
            ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_RESHAPE, GGML_OP_MUL_MAT }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr, cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE, GGML_OP_VIEW, GGML_OP_SET_ROWS }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 4]);
        return 4;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ROPE }, {})) {
        ggml_cuda_op_rms_norm_mul_rope_fused(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr);
        return 2;
    }

    if (!(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_NSCALE) &&
            ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], cgraph->nodes[i + 3],
                ggml_cuda_bf16_shadow_for(*cuda_ctx, cgraph, i + 3, cgraph->nodes[i + 3]));
        return 3;
    }
    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL, GGML_OP_ADD }, {})) {
        ggml_cuda_op_rms_norm_fused_add(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2], nullptr,
                ggml_cuda_bf16_shadow_for(*cuda_ctx, cgraph, i + 2, cgraph->nodes[i + 2]));
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_MUL }, {})) {
        ggml_cuda_op_rms_norm_fused(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    // rms_norm > scale > mul: a constant between the norm and its weight
    if (!((ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_NSCALE) || getenv("GGML_CUDA_FUSE_NSCALE") == nullptr) && ggml_can_fuse(cgraph, i, { GGML_OP_RMS_NORM, GGML_OP_SCALE, GGML_OP_MUL }) &&
            ggml_cuda_should_fuse_rms_norm_scale_mul(node, cgraph->nodes[i + 1], cgraph->nodes[i + 2])) {
        ggml_cuda_op_rms_norm_fused_scale(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    // the router-weight multiply of every expert's output and the combine after it, as one weighted sum
    if (node->op == GGML_OP_MUL && !(ggml_cuda_fuse_off_mask() & (GGML_CUDA_FUSE_EXPERT_SUM | GGML_CUDA_FUSE_EXPERT_WSUM))) {
        const int n_nodes = ggml_cuda_try_expert_wsum(*cuda_ctx, cgraph, i);
        if (n_nodes > 0) {
            return n_nodes;
        }
    }

    // the expert combine: a chain of ADDs over one tensor's rows, as one sum
    if (node->op == GGML_OP_ADD && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_EXPERT_SUM)) {
        const ggml_tensor * experts = nullptr;
        const int n_adds = ggml_cuda_match_expert_sum(cgraph, i, &experts);
        if (n_adds > 0) {
            ggml_cuda_op_sum_expert_rows(*cuda_ctx, experts, cgraph->nodes[i + n_adds - 1], (int) experts->ne[1]);
            return n_adds - 1;
        }
    }

    // two l2_norms over adjacent views of one tensor (q and k after the
    // conv in a delta-net layer): one launch covering both row ranges
    if (node->op == GGML_OP_L2_NORM && i + 1 < cgraph->n_nodes && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_L2_PAIR) &&
            ggml_cuda_should_fuse_l2_norm_pair(node, cgraph->nodes[i + 1])) {
        ggml_cuda_op_l2_norm_pair(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    // two l2_norms of one shape with views between them, as one launch over two sources
    if (node->op == GGML_OP_L2_NORM && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_L2_PAIR)) {
        int j = i + 1;
        while (j < cgraph->n_nodes && (ggml_cuda_is_view_or_noop(cgraph->nodes[j]) || ggml_is_empty(cgraph->nodes[j]))) {
            j++;
        }
        if (j < cgraph->n_nodes && ggml_cuda_should_fuse_l2_norm_pair2(node, cgraph->nodes[j])) {
            ggml_cuda_op_l2_norm_pair2(*cuda_ctx, node, cgraph->nodes[j]);
            return j - i;
        }
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_ADD, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, cgraph->nodes[i + 1], cgraph->nodes[i + 2]);
        return 2;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SSM_CONV, GGML_OP_UNARY }, { GGML_UNARY_OP_SILU })) {
        ggml_cuda_op_ssm_conv(*cuda_ctx, node, /*bias_add_node=*/ nullptr, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SILU }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SIGMOID }) ||
        ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_MUL }, { GGML_UNARY_OP_SOFTPLUS })) {
        ggml_cuda_op_unary_mul(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_UNARY, GGML_OP_SQR }, { GGML_UNARY_OP_RELU })) {
        ggml_cuda_op_relu_sqr(*cuda_ctx, node, cgraph->nodes[i + 1]);
        return 1;
    }

    if (ggml_cuda_can_fuse(cgraph, i, { GGML_OP_SCALE, GGML_OP_UNARY, GGML_OP_SCALE }, { GGML_UNARY_OP_TANH })) {
        ggml_cuda_op_softcap(*cuda_ctx, cgraph->nodes[i + 2], node);
        return 2;
    }

    return 0;
}


struct ggml_cuda_op_stat {
    double  ms = 0.0;
    int64_t n  = 0;
};

static int ggml_cuda_op_stats() {
    static const int v = [] { const char * e = getenv("GGML_CUDA_OP_STATS"); return e ? atoi(e) : 0; }();
    return v;
}

struct ggml_cuda_op_timer {
    std::vector<cudaEvent_t> events;
    std::vector<std::pair<std::string, int>> marks;
    int k = 0;

    void begin(cudaStream_t stream) {
        marks.clear();
        k = 0;
        record(stream);
    }
    void record(cudaStream_t stream) {
        if ((int) events.size() <= k) {
            cudaEvent_t e;
            CUDA_CHECK(cudaEventCreate(&e));
            events.push_back(e);
        }
        CUDA_CHECK(cudaEventRecord(events[k], stream));
        k++;
    }
    void mark(cudaStream_t stream, const ggml_tensor * node, int fused) {
        char key[320];
        const ggml_tensor * s0 = node->src[0];
        const ggml_tensor * s1 = node->src[1];
        snprintf(key, sizeof(key), "%s%s%d dst[%lld,%lld,%lld] s0:%s[%lld,%lld,%lld] s1:%s[%lld,%lld,%lld]",
                 fused ? "F" : "", ggml_op_desc(node), fused,
                 (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2],
                 s0 ? ggml_type_name(s0->type) : "-", s0 ? (long long) s0->ne[0] : 0, s0 ? (long long) s0->ne[1] : 0, s0 ? (long long) s0->ne[2] : 0,
                 s1 ? ggml_type_name(s1->type) : "-", s1 ? (long long) s1->ne[0] : 0, s1 ? (long long) s1->ne[1] : 0, s1 ? (long long) s1->ne[2] : 0);
        record(stream);
        marks.emplace_back(key, k - 1);
    }
    void flush(cudaStream_t stream, int bucket) {
        CUDA_CHECK(cudaStreamSynchronize(stream));
        static std::map<int, std::map<std::string, ggml_cuda_op_stat>> acc;
        static std::map<int, int> calls;
        auto & a = acc[bucket];
        for (const auto & [key, idx] : marks) {
            float ms = 0.0f;
            CUDA_CHECK(cudaEventElapsedTime(&ms, events[idx - 1], events[idx]));
            a[key].ms += ms;
            a[key].n  += 1;
        }
        const int c = ++calls[bucket];
        if (c % 50 != 0) {
            return;
        }
        std::vector<std::pair<std::string, ggml_cuda_op_stat>> rows(a.begin(), a.end());
        std::sort(rows.begin(), rows.end(), [](const auto & x, const auto & y) { return x.second.ms > y.second.ms; });
        double total = 0.0;
        for (const auto & r : rows) total += r.second.ms;
        fprintf(stderr, "[opstat] bucket=%d calls=%d total_ms_per_call=%.3f distinct=%zu" "\n", bucket, c, total / c, rows.size());
        for (size_t i = 0; i < rows.size() && i < 60; i++) {
            fprintf(stderr, "[opstat]   %8.3f ms/call %6.1f/call %6.2f us/op  %s" "\n",
                    rows[i].second.ms / c, (double) rows[i].second.n / c, 1000.0 * rows[i].second.ms / rows[i].second.n, rows[i].first.c_str());
        }
        a.clear();
    }
};

static ggml_cuda_op_timer & ggml_cuda_op_timer_get() {
    static ggml_cuda_op_timer t;
    return t;
}

// GGML_CUDA_SKIP_OPS=RMS_NORM,ADD,...: launch nothing for these op types. A fused
// run is checked on its head node only, so name every op of the group to drop
// it whole. The output is garbage; only the wall clock means anything.
static bool ggml_cuda_skip_op(const ggml_tensor * node) {
    static const std::vector<std::string> ops = [] {
        std::vector<std::string> v;
        const char * e = getenv("GGML_CUDA_SKIP_OPS");
        std::string s = e ? e : "";
        size_t i = 0;
        while (i < s.size()) {
            size_t j = s.find(',', i);
            if (j == std::string::npos) {
                j = s.size();
            }
            if (j > i) {
                v.push_back(s.substr(i, j - i));
            }
            i = j + 1;
        }
        return v;
    }();
    if (ops.empty()) {
        return false;
    }
    const char * name = node->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(node)) : ggml_op_name(node->op);
    for (const auto & o : ops) {
        // name:<prefix> drops the nodes whose tensor name starts with the prefix
        if (o == name || (o.rfind("name:", 0) == 0 && strncmp(node->name, o.c_str() + 5, o.size() - 5) == 0)) {
            return true;
        }
    }
    return false;
}

// concurrent regions, see ggml_cuda_find_concurrent_regions below
static const ggml_tensor * ggml_cuda_region_base(const ggml_tensor * t);
static bool                ggml_cuda_region_reads(const ggml_tensor * node, const ggml_tensor * t);
static cudaEvent_t         ggml_cuda_region_event(ggml_backend_cuda_context & ctx);

static void ggml_cuda_graph_evaluate_and_capture(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, const bool use_cuda_graph, const bool cuda_graph_update_required, const void * graph_key) {
    bool graph_evaluated_or_captured = false;

    // flag used to determine whether it is an integrated_gpu
    const bool integrated            = ggml_cuda_info().devices[cuda_ctx->device].integrated;

    ggml_cuda_stream_context & stream_ctx = cuda_ctx->stream_context();
    bool                         is_concurrent_event_active = false;
    ggml_cuda_concurrent_event * concurrent_event           = nullptr;
    bool                         should_launch_concurrent_events = false;

    const auto try_launch_concurrent_event = [&](const ggml_tensor * node) {
        if (stream_ctx.concurrent_events.find(node) != stream_ctx.concurrent_events.end()) {
            concurrent_event = &stream_ctx.concurrent_events[node];

            is_concurrent_event_active = true;

            GGML_LOG_DEBUG("Launching %d streams at %s\n", concurrent_event->n_streams, node->name);

            cudaStream_t main_stream = cuda_ctx->stream();  // this should be stream 0
            GGML_ASSERT(cuda_ctx->curr_stream_no == 0);
            CUDA_CHECK(cudaEventRecord(concurrent_event->fork_event, main_stream));

            for (int i = 1; i <= concurrent_event->n_streams; ++i) {
                cudaStream_t stream = cuda_ctx->stream(cuda_ctx->device, i);
                CUDA_CHECK(cudaStreamWaitEvent(stream, concurrent_event->fork_event));
            }
        }
    };

    while (!graph_evaluated_or_captured) {
        // Only perform the graph execution if CUDA graphs are not enabled, or we are capturing the graph.
        // With the use of CUDA graphs, the execution will be performed by the graph launch.
        if (!use_cuda_graph || cuda_graph_update_required) {
            [[maybe_unused]] int prev_i = 0;

            if (stream_ctx.concurrent_events.size() > 0) {
                static const bool trace_regions = getenv("GGML_CUDA_TRACE_REGIONS") != nullptr;
                should_launch_concurrent_events = true;
                for (const auto & [tensor, event] : stream_ctx.concurrent_events) {
                    const bool valid = event.is_valid();
                    if (!valid && trace_regions) {
                        fprintf(stderr, "[region] %s -> %s dropped: its branches overlap or read each other\n", tensor->name, event.join_node->name);
                    }
                    should_launch_concurrent_events = should_launch_concurrent_events && valid;
                }
            }

            if (should_launch_concurrent_events) {
                // Restore original node order within each concurrent region to enable fusion within streams

                std::unordered_map<const ggml_tensor *, int> node_to_idx;
                node_to_idx.reserve(cgraph->n_nodes);
                for (int i = 0; i < cgraph->n_nodes; ++i) {
                    node_to_idx[cgraph->nodes[i]] = i;
                }

                for (auto & [fork_node, event] : stream_ctx.concurrent_events) {
                    // Find positions of all nodes from this event in the current graph
                    std::vector<int> positions;
                    positions.reserve(event.original_order.size());

                    bool all_found = true;
                    for (const ggml_tensor * orig_node : event.original_order) {
                        auto it = node_to_idx.find(orig_node);
                        if (it != node_to_idx.end()) {
                            positions.push_back(it->second);
                        } else {
                            all_found = false;
                            break;
                        }
                    }

                    if (!all_found || positions.size() != event.original_order.size()) {
                        continue;
                    }

                    // Sort positions to get contiguous range
                    std::vector<int> sorted_positions = positions;
                    std::sort(sorted_positions.begin(), sorted_positions.end());

                    bool is_contiguous = true;
                    for (size_t i = 1; i < sorted_positions.size(); ++i) {
                        if (sorted_positions[i] != sorted_positions[i-1] + 1) {
                            is_contiguous = false;
                            break;
                        }
                    }

                    if (!is_contiguous) {
                        continue;
                    }

                    // Restore original order at the sorted positions
                    int start_pos = sorted_positions[0];
                    for (size_t i = 0; i < event.original_order.size(); ++i) {
                        cgraph->nodes[start_pos + i] = const_cast<ggml_tensor *>(event.original_order[i]);
                    }
                }
            } else {
                stream_ctx.concurrent_events.clear();
            }

            const bool op_stats = ggml_cuda_op_stats() && !use_cuda_graph;
            if (op_stats) { ggml_cuda_op_timer_get().begin(cuda_ctx->stream()); }
            cuda_ctx->active_region      = nullptr;
            cuda_ctx->region_events_used = 0;
            int region_join_idx          = cgraph->n_nodes;
            cuda_ctx->norm_folds.clear();
            cuda_ctx->norm_fold_order.clear();
            cuda_ctx->ids_ready = nullptr;
            cuda_ctx->state_folds.clear();
            cuda_ctx->state_writers_ready = false;
            g_cuda_absorbed.clear();
            g_cuda_no_fuse_until = 0;
            g_cuda_gated.clear();
            ggml_cuda_gated_delta_net_qk_norm_clear();
            {
                auto it = g_bf16_shadows.find(cuda_ctx);
                if (it != g_bf16_shadows.end()) {
                    it->second.walk++;
                }
            }
            for (int i = 0; i < cgraph->n_nodes; i++) {
                ggml_tensor * node = cgraph->nodes[i];
                // GGML_CUDA_TRACE_OPS=1: every op before its launch, for a synchronous fault hunt
                static const bool trace_ops = getenv("GGML_CUDA_TRACE_OPS") != nullptr;
                if (trace_ops) {
                    fprintf(stderr, "[op %d] %s %s data=%p src0=%p src1=%p" "\n", i, ggml_op_name(node->op), node->name, node->data,
                            node->src[0] ? node->src[0]->data : nullptr, node->src[1] ? node->src[1]->data : nullptr);
                    fflush(stderr);
                }
                if (!is_concurrent_event_active && i > 0 && !stream_ctx.concurrent_events.empty()) {
                    // a region forks at the last computed node before this one, whatever launch covered it
                    int p = i - 1;
                    while (p > 0 && ggml_cuda_is_view_or_noop(cgraph->nodes[p])) {
                        p--;
                    }
                    try_launch_concurrent_event(cgraph->nodes[p]);
                    if (is_concurrent_event_active) {
                        cuda_ctx->active_region = concurrent_event;
                        region_join_idx         = cgraph->n_nodes;
                        for (int k = i; k < cgraph->n_nodes; ++k) {
                            if (cgraph->nodes[k] == concurrent_event->join_node) {
                                region_join_idx = k;
                                break;
                            }
                        }
                    }
                }
                // the join: the main stream waits for every branch, and the region is over
                const auto join_region = [&]() {
                    cuda_ctx->curr_stream_no = 0;
                    for (int s = 1; s <= concurrent_event->n_streams; ++s) {
                        CUDA_CHECK(cudaEventRecord(concurrent_event->join_events[s - 1], cuda_ctx->stream(cuda_ctx->device, s)));
                        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), concurrent_event->join_events[s - 1]));
                    }
                    is_concurrent_event_active = false;
                    concurrent_event           = nullptr;
                    cuda_ctx->active_region    = nullptr;
                    region_join_idx            = cgraph->n_nodes;
                };
                if (is_concurrent_event_active) {
                    GGML_ASSERT(concurrent_event);

                    if (node == concurrent_event->join_node) {
                        join_region();
                    } else {
                        auto it = concurrent_event->stream_mapping.find(node);
                        GGML_ASSERT(it != concurrent_event->stream_mapping.end());
                        cuda_ctx->curr_stream_no = it->second;
                    }
                }

                prev_i = i;

                if (!g_cuda_absorbed.empty() && ggml_cuda_take_absorbed(node)) {
                    continue;
                }

                if (ggml_cuda_is_view_or_noop(node)) {
                    continue;
                }

                if ((node->flags & GGML_TENSOR_FLAG_COMPUTE) == 0) {
                    continue;
                }

                // a node with no elements has nothing to compute; recurrent
                // models emit several per layer when only one sequence is live
                if (ggml_is_empty(node) && !(ggml_cuda_fuse_off_mask() & GGML_CUDA_FUSE_EMPTY_SKIP)) {
                    continue;
                }

                if (ggml_cuda_skip_op(node)) {
                    continue;
                }

                // GGML_CUDA_FOLD_TRACE_READERS=1: any node that touches a folded product other
                // than through a matvec is named, since it would be reading an unwritten buffer
                static const bool trace_readers = getenv("GGML_CUDA_FOLD_TRACE_READERS") != nullptr;
                if (trace_readers && !cuda_ctx->norm_folds.empty()) {
                    for (int s = 0; s < GGML_MAX_SRC; ++s) {
                        const ggml_tensor * src = node->src[s];
                        if (!src) {
                            continue;
                        }
                        const ggml_tensor * base = src->view_src ? src->view_src : src;
                        for (const auto & kv : cuda_ctx->norm_folds) {
                            if ((src == kv.first || base == kv.second->mul || base == kv.second->norm) &&
                                    node->op != GGML_OP_MUL_MAT && node->op != GGML_OP_MUL_MAT_ID && node->op != GGML_OP_RESHAPE && node->op != GGML_OP_VIEW) {
                                fprintf(stderr, "[fold-reader] node %d %s (%s) reads fold %s through src%d %s\n", i, ggml_op_name(node->op), node->name,
                                        kv.second->norm->name, s, src->name);
                            }
                        }
                    }
                }

                if (node->op == GGML_OP_GET_ROWS && ggml_cuda_try_state_fold(*cuda_ctx, cgraph, i)) {
                    continue;
                }

                // every fusion fires as it would on one stream, so the bits stay what they were. A launch here may
                // cover the nodes after this one that each consume the one before (an elementwise tail, a norm
                // with its rope), or matvecs over this node's input (members, a gate with its up); whatever those
                // read from the other branches must be complete, so this stream waits for those branches first.
                if (is_concurrent_event_active) {
                    bool need[GGML_CUDA_MAX_STREAMS] = {};
                    const auto mark = [&](const ggml_tensor * t) {
                        auto it = concurrent_event->stream_mapping.find(t);
                        if (it != concurrent_event->stream_mapping.end() && it->second != cuda_ctx->curr_stream_no) {
                            need[it->second] = true;
                        }
                    };
                    const ggml_tensor * cur = node;
                    for (int j = i + 1, n_reach = 0; j < cgraph->n_nodes && n_reach < 16; ++j) {
                        const ggml_tensor * t = cgraph->nodes[j];
                        if (ggml_cuda_is_view_or_noop(t)) {
                            continue;
                        }
                        const bool consumes = ggml_cuda_region_reads(t, cur);
                        const bool sibling  = (t->op == GGML_OP_MUL_MAT || t->op == GGML_OP_MUL_MAT_ID) && t->op == node->op && t->src[1] == node->src[1];
                        if (!consumes && !sibling) {
                            break;
                        }
                        n_reach++;
                        if (j >= region_join_idx) {
                            for (int s = 1; s <= concurrent_event->n_streams; ++s) {
                                need[s] = s != cuda_ctx->curr_stream_no;
                            }
                            break;
                        }
                        mark(t);
                        for (int s = 0; s < GGML_MAX_SRC && t->src[s]; ++s) {
                            mark(t->src[s]);
                            mark(ggml_cuda_region_base(t->src[s]));
                        }
                        if (consumes) {
                            cur = t;
                        }
                    }
                    for (int s = 1; s <= concurrent_event->n_streams; ++s) {
                        if (need[s]) {
                            cudaEvent_t e = ggml_cuda_region_event(*cuda_ctx);
                            CUDA_CHECK(cudaEventRecord(e, cuda_ctx->stream(cuda_ctx->device, s)));
                            CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), e));
                        }
                    }
                }
                int nodes_to_skip = i < g_cuda_no_fuse_until ? 0 : ggml_cuda_try_fuse(cuda_ctx, cgraph, i);

                // GGML_CUDA_TRACE_LAUNCH=1: what the graph actually launches,
                // with fusion applied - a fused run prints its first op and how
                // many nodes it swallowed
                static const bool trace_launch = getenv("GGML_CUDA_TRACE_LAUNCH") != nullptr;
                if (trace_launch) {
                    fprintf(stderr, "[launch %d] %s %s fused=%d ne=%lld,%lld,%lld\n", i, ggml_op_name(node->op),
                            node->op == GGML_OP_UNARY ? ggml_unary_op_name(ggml_get_unary_op(node)) : node->name,
                            nodes_to_skip, (long long) node->ne[0], (long long) node->ne[1], (long long) node->ne[2]);
                }

                if (nodes_to_skip != 0) {
#ifdef GGML_CUDA_DEBUG
                    const int last_fused = i + nodes_to_skip;
                    GGML_LOG_INFO("nodes_fused: %d, first: %s (%s), last: %s (%s)\n",
                            nodes_to_skip + 1, ggml_op_name(node->op), node->name,
                            ggml_op_name(cgraph->nodes[last_fused]->op), cgraph->nodes[last_fused]->name);
#endif
                    if (op_stats) { ggml_cuda_op_timer_get().mark(cuda_ctx->stream(), node, nodes_to_skip + 1); }
                    ggml_cuda_norm_fold_poison_scan(*cuda_ctx, cgraph->nodes[i + nodes_to_skip], i + nodes_to_skip);
                    if (is_concurrent_event_active) {
                        if (i + nodes_to_skip >= region_join_idx) {
                            // the launch covered the join itself
                            join_region();
                        } else {
                            // a launch that covered nodes of other branches: their streams wait for it
                            for (int k = i + 1; k <= i + nodes_to_skip; ++k) {
                                auto it = concurrent_event->stream_mapping.find(cgraph->nodes[k]);
                                if (it != concurrent_event->stream_mapping.end() && it->second != cuda_ctx->curr_stream_no) {
                                    ggml_cuda_region_publish(*cuda_ctx);
                                    break;
                                }
                            }
                        }
                    }
                    i += nodes_to_skip;
                    continue;
                }
#ifndef NDEBUG
                // On integrated GPUs (APUs, e.g. RDNA3.5) the scheduler may place a
                // node's output on the host-visible buffer, which the compute path
                // handles. Allow that here, mirroring the src-tensor check below.
                assert(node->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                       (integrated && ggml_backend_buft_is_cuda_host(node->buffer->buft)));
                for (int j = 0; j < GGML_MAX_SRC; j++) {
                    if (node->src[j] != nullptr) {
                        assert(node->src[j]->buffer);
                        assert(node->src[j]->buffer->buft == ggml_backend_cuda_buffer_type(cuda_ctx->device) ||
                               (integrated && ggml_backend_buft_is_cuda_host(node->src[j]->buffer->buft)));
                    }
                }
#else
                GGML_UNUSED(integrated);
#endif  // NDEBUG

                bool ok = ggml_cuda_compute_forward(*cuda_ctx, node);
                if (!ok) {
                    GGML_LOG_ERROR("%s: op not supported %s (%s)\n", __func__, node->name, ggml_op_name(node->op));
                }
                GGML_ASSERT(ok);
                ggml_cuda_norm_fold_poison_scan(*cuda_ctx, node, i);
                if (op_stats) { ggml_cuda_op_timer_get().mark(cuda_ctx->stream(), node, 0); }
            }
            // GGML_CUDA_TRACE_FOLD=1: how many norms this walk left unlaunched
            static const bool trace_fold = getenv("GGML_CUDA_TRACE_FOLD") != nullptr;
            if (trace_fold && !cuda_ctx->norm_folds.empty()) {
                int folded = 0, materialized = 0;
                for (const auto & kv : cuda_ctx->norm_folds) {
                    if (kv.first == kv.second->mul) {
                        folded++;
                        materialized += kv.second->materialized;
                    }
                }
                fprintf(stderr, "[fold] %d norms folded, %d of them written out anyway, in a walk of %d nodes\n",
                        folded, materialized, cgraph->n_nodes);
            }
            for (auto it = cuda_ctx->norm_fold_order.rbegin(); it != cuda_ctx->norm_fold_order.rend(); ++it) {
                cuda_ctx->pool().free((*it)->q8, (*it)->q8_size);
            }
            cuda_ctx->norm_fold_order.clear();
            cuda_ctx->norm_folds.clear();
            cuda_ctx->ids_ready = nullptr;
            GGML_ASSERT(cuda_ctx->state_folds.empty() && "a recurrent state gather was folded into a reader that never ran");
            GGML_ASSERT(g_cuda_gated.empty() && "a gated norm was left for a quantize that never ran");
            GGML_ASSERT(ggml_cuda_gated_delta_net_qk_norm_pending() == 0 && "q/k norms were left for a recurrence that never ran");
            cuda_ctx->state_writers_ready = false;
            if (op_stats) { ggml_cuda_op_timer_get().flush(cuda_ctx->stream(), cgraph->n_nodes); }
        }

#ifdef USE_CUDA_GRAPH
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (use_cuda_graph && cuda_graph_update_required) { // End CUDA graph capture
            if (graph->graph != nullptr) {
                CUDA_CHECK(cudaGraphDestroy(graph->graph));
                graph->graph = nullptr;
            }

            CUDA_CHECK(cudaStreamEndCapture(cuda_ctx->stream(), &graph->graph));
            graph_evaluated_or_captured = true; // CUDA graph has been captured

            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            if (ggml_cuda_lock_counter.fetch_sub(1, std::memory_order_relaxed) == 1) {
                ggml_cuda_lock_cv.notify_all();
            }
        } else {
            graph_evaluated_or_captured = true; // ggml graph has been directly evaluated
        }
    }

    if (use_cuda_graph) {
        ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
        if (graph->instance == nullptr) { // Create executable graph from captured graph.
            CUDA_CHECK(cudaGraphInstantiate(&graph->instance, graph->graph, NULL, NULL, 0));
        }
        if (cuda_graph_update_required) { // Update graph executable
            ggml_cuda_graph_update_executable(cuda_ctx, graph_key);
        }
        // Launch graph
        CUDA_CHECK(cudaGraphLaunch(graph->instance, cuda_ctx->stream()));
#else
        GGML_UNUSED(graph_key);
        graph_evaluated_or_captured = true;
#endif  // USE_CUDA_GRAPH
    }
}

#ifdef USE_CUDA_GRAPH
static bool ggml_cuda_graph_set_enabled(ggml_backend_cuda_context * cuda_ctx, const void * graph_key) {
    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);

    if (graph->graph == nullptr) {
        if (ggml_cuda_info().devices[cuda_ctx->device].cc < GGML_CUDA_CC_VOLTA) {
            if (!graph->disable_due_to_gpu_arch) {
                GGML_LOG_DEBUG("%s: disabling CUDA graphs due to GPU architecture\n", __func__);
            }
            graph->disable_due_to_gpu_arch = true;
        }
    }

    return graph->is_enabled();
}
#endif // USE_CUDA_GRAPH


// Order the next graph on backend_dst after everything queued on backend_src,
// without the host waiting for either. Both graphs keep their own stream; the
// dependency is an event, and it is applied when the destination's graph is
// about to run, not when the caller asks for it. Asking earlier would put the
// wait on the stream before the scheduler synchronizes it to copy the graph's
// inputs, and that synchronize would then wait for the source's whole graph.
// one strided copy on the backend's stream: height rows of width bytes, host or device memory on either side
static void ggml_backend_cuda_copy_2d_async(ggml_backend_t backend, void * dst, size_t dpitch, const void * src, size_t spitch, size_t width, size_t height) {
    GGML_ASSERT(ggml_backend_is_cuda(backend));
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    CUDA_CHECK(cudaMemcpy2DAsync(dst, dpitch, src, spitch, width, height, cudaMemcpyDefault, cuda_ctx->stream()));
}

// A sequence's state copied out beside the decodes: the copies go on a stream of their own (one per context, made on
// first use) that waits for the main stream's work so far, so they read the cells the last decode wrote and the next
// decode's kernels do not wait for them. A call without a tensor records the event on that stream.
struct ggml_cuda_side_stream {
    cudaStream_t stream = nullptr;
    cudaEvent_t  order  = nullptr;
};

static ggml_cuda_side_stream & ggml_cuda_side_stream_of(ggml_backend_cuda_context * cuda_ctx) {
    static std::mutex mtx;
    static std::unordered_map<const ggml_backend_cuda_context *, ggml_cuda_side_stream> m;
    std::lock_guard<std::mutex> lock(mtx);
    auto & s = m[cuda_ctx];
    if (s.stream == nullptr) {
        ggml_cuda_set_device(cuda_ctx->device);
        CUDA_CHECK(cudaStreamCreateWithFlags(&s.stream, cudaStreamNonBlocking));
        CUDA_CHECK(cudaEventCreateWithFlags(&s.order, cudaEventDisableTiming));
    }
    return s;
}

static void ggml_backend_cuda_get_tensor_side_async(ggml_backend_t backend, const ggml_tensor * tensor, void * data, size_t offset, size_t size, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    auto & side = ggml_cuda_side_stream_of(cuda_ctx);
    if (tensor != nullptr) {
        CUDA_CHECK(cudaEventRecord(side.order, cuda_ctx->stream()));
        CUDA_CHECK(cudaStreamWaitEvent(side.stream, side.order, 0));
        CUDA_CHECK(cudaMemcpyAsync(data, (const char *) tensor->data + offset, size, cudaMemcpyDefault, side.stream));
    }
    if (event != nullptr) {
        CUDA_CHECK(cudaEventRecord((cudaEvent_t) event->context, side.stream));
    }
}

// copies one driver call issues (cudaMemcpyBatchAsync), each from the stream's position: a state save or a prefix copy
// is tens of rows per layer, and a call per row costs more host time than the copies take
static bool ggml_cuda_memcpy_batch(void * const * dsts, const void * const * srcs, const size_t * sizes, const size_t n, cudaStream_t stream) {
#if CUDART_VERSION >= 12080
    static const bool off = getenv("GGML_CUDA_NO_MEMCPY_BATCH") != nullptr;
    if (!off) {
        cudaMemcpyAttributes attr = {};
        attr.srcAccessOrder = cudaMemcpySrcAccessOrderStream;
        size_t idx = 0;
        if (cudaMemcpyBatchAsync(dsts, srcs, sizes, n, &attr, &idx, 1, stream) == cudaSuccess) {
            return true;
        }
        (void) cudaGetLastError();
    }
#endif
    for (size_t i = 0; i < n; ++i) {
        CUDA_CHECK(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDefault, stream));
    }
    return false;
}

// ggml_backend_cuda_get_tensor_side_async for n copies at once: one wait for the main stream, one batched copy
static void ggml_backend_cuda_get_tensors_side_async(ggml_backend_t backend, const size_t n, const ggml_tensor * const * tensors,
        void * const * dsts, const size_t * offsets, const size_t * sizes, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    auto & side = ggml_cuda_side_stream_of(cuda_ctx);
    if (n > 0) {
        CUDA_CHECK(cudaEventRecord(side.order, cuda_ctx->stream()));
        CUDA_CHECK(cudaStreamWaitEvent(side.stream, side.order, 0));
        std::vector<const void *> srcs(n);
        for (size_t i = 0; i < n; ++i) {
            srcs[i] = (const char *) tensors[i]->data + offsets[i];
        }
        ggml_cuda_memcpy_batch(dsts, srcs.data(), sizes, n, side.stream);
    }
    if (event != nullptr) {
        CUDA_CHECK(cudaEventRecord((cudaEvent_t) event->context, side.stream));
    }
}

// device-to-device copies as one kernel: the driver issues a batched D2D copy as one operation per copy (~5 us of host
// time each), the table here goes in the launch's parameters
#define GGML_CUDA_COPY_BATCH_MAX 128
struct ggml_cuda_copy_batch {
    const char * src[GGML_CUDA_COPY_BATCH_MAX];
    char *       dst[GGML_CUDA_COPY_BATCH_MAX];
    unsigned int n16[GGML_CUDA_COPY_BATCH_MAX]; // 16-byte words
};

static __global__ void k_copy_batch(const __grid_constant__ ggml_cuda_copy_batch b) {
    const int4 * src = (const int4 *) b.src[blockIdx.y];
    int4 *       dst = (int4 *) b.dst[blockIdx.y];
    const unsigned int n = b.n16[blockIdx.y];
    for (unsigned int i = blockIdx.x*blockDim.x + threadIdx.x; i < n; i += gridDim.x*blockDim.x) {
        dst[i] = src[i];
    }
}

static void ggml_backend_cuda_memcpy_batch_async(ggml_backend_t backend, const size_t n, void * const * dsts, const void * const * srcs,
        const size_t * sizes) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    cudaStream_t stream = cuda_ctx->stream();
    static const bool off = getenv("GGML_CUDA_NO_MEMCPY_BATCH") != nullptr;
    ggml_cuda_copy_batch b;
    int k = 0;
    auto flush = [&]() {
        if (k > 0) {
            k_copy_batch<<<dim3(8, k), 256, 0, stream>>>(b);
            CUDA_CHECK(cudaGetLastError());
            k = 0;
        }
    };
    for (size_t i = 0; i < n; ++i) {
        const bool aligned = ((uintptr_t) dsts[i] % 16) == 0 && ((uintptr_t) srcs[i] % 16) == 0 && sizes[i] % 16 == 0 && sizes[i]/16 <= UINT_MAX;
        if (off || !aligned) {
            CUDA_CHECK(cudaMemcpyAsync(dsts[i], srcs[i], sizes[i], cudaMemcpyDeviceToDevice, stream));
            continue;
        }
        b.src[k] = (const char *) srcs[i];
        b.dst[k] = (char *) dsts[i];
        b.n16[k] = (unsigned int) (sizes[i]/16);
        if (++k == GGML_CUDA_COPY_BATCH_MAX) {
            flush();
        }
    }
    flush();
}

static void ggml_backend_cuda_order_after(ggml_backend_t backend_dst, ggml_backend_t backend_src) {
    if (!ggml_backend_is_cuda(backend_dst) || !ggml_backend_is_cuda(backend_src) || backend_dst == backend_src) {
        return;
    }
    ggml_backend_cuda_context * dst = (ggml_backend_cuda_context *) backend_dst->context;
    ggml_backend_cuda_context * src = (ggml_backend_cuda_context *) backend_src->context;

    if (!src->order_event) {
        ggml_cuda_set_device(src->device);
        CUDA_CHECK(cudaEventCreateWithFlags(&src->order_event, cudaEventDisableTiming));
    }
    CUDA_CHECK(cudaEventRecord(src->order_event, src->stream()));
    dst->pending_order = src->order_event;
}



// LLAMA_ROUND_TIMING=1: host time inside the CUDA backend's graph compute
struct cuda_host_timing {
    bool on = getenv("LLAMA_ROUND_TIMING") != nullptr;
    struct acc { double check = 0, evaluate = 0; int n = 0; int graphs = 0;
                 double t_launch = 0, t_capture = 0, t_eager = 0; int n_capture = 0; int total = 0; };
    std::map<const void *, acc> by_backend;
    // why a call went eager: 1 disabled, 2 an op the capture cannot take,
    // 3 warming up a new key, 4 properties changed
    std::map<std::pair<const void *, int>, std::pair<int, double>> eager_why;
    std::map<std::pair<const void *, int>, int> eager_nodes;
    std::map<std::pair<const void *, int>, int> eager_seen;
    void add_eager(const void * key, int why, int n_nodes, double ms, int n_seen) {
        auto & e = eager_why[{key, why}];
        e.first++; e.second += ms;
        eager_nodes[{key, why}] = n_nodes;
        eager_seen[{key, why}]  = n_seen;
    }
    void report_eager(const void * key) {
        for (const auto & [k, v] : eager_why) {
            if (k.first != key) continue;
            fprintf(stderr, "[round] cuda %p   eager reason %d: %d calls, %.3f ms each, %d nodes, shape seen %d times" + 0,
                    key, k.second, v.first, v.second / v.first, eager_nodes[k], eager_seen[k]);
            fputc(10, stderr);
        }
    }
    // the three scans that run before a replay: the key, the compatibility
    // walk, and the property comparison
    struct chk { double key = 0, compat = 0, props = 0; int n = 0; };
    std::map<const void *, chk> checks;
    void add_check(const void * key, double a, double b, double c) {
        chk & k = checks[key];
        k.key += a; k.compat += b; k.props += c; k.n++;
        if (k.n % 100 == 0) {
            fprintf(stderr, "[round] cuda %p   check: key %.3f  compat %.3f  props %.3f ms per call\n",
                    key, k.key / k.n, k.compat / k.n, k.props / k.n);
            k = chk{};
        }
    }
    void add(const void * key, double a, double b, bool used_graph, bool captured) {
        acc & t = by_backend[key];
        t.check += a; t.evaluate += b; t.n++; t.graphs += used_graph ? 1 : 0;
        if (captured) { t.t_capture += b; t.n_capture++; }
        else if (used_graph) { t.t_launch += b; }
        else { t.t_eager += b; }
        if (t.n % 100 == 0) {
            const int n_launch = t.graphs - t.n_capture;
            const int n_eager  = t.n - t.graphs;
            fprintf(stderr, "[round] cuda %p calls %d-%d  check %.3f  evaluate-or-launch %.3f ms per call  cuda-graph %d/%d" + 0, key, t.total, t.total + t.n, t.check / t.n, t.evaluate / t.n, t.graphs, t.n);
            fputc(10, stderr);
            fprintf(stderr, "[round] cuda %p   launch  %d calls %.3f ms each | capture+instantiate %d calls %.3f ms each | eager %d calls %.3f ms each | per-call share: launch %.3f capture %.3f eager %.3f" + 0,
                    key,
                    n_launch,   n_launch   ? t.t_launch  / n_launch   : 0.0,
                    t.n_capture, t.n_capture ? t.t_capture / t.n_capture : 0.0,
                    n_eager,    n_eager    ? t.t_eager   / n_eager    : 0.0,
                    t.t_launch / t.n, t.t_capture / t.n, t.t_eager / t.n);
            fputc(10, stderr);
            report_eager(key);
            const int total = t.total + t.n;
            t = acc{};
            t.total = total;
            for (auto it = eager_why.begin(); it != eager_why.end(); ) {
                it = it->first.first == key ? eager_why.erase(it) : std::next(it);
            }
        }
    }
};
static cuda_host_timing & cuda_host_timer() { static cuda_host_timing t; return t; }
static int ggml_cuda_graph_stats() {
    static const int v = [] { const char * e = getenv("GGML_CUDA_GRAPH_STATS"); return e ? atoi(e) : 0; }();
    return v;
}

static enum ggml_status ggml_backend_cuda_graph_compute(ggml_backend_t backend, ggml_cgraph * cgraph) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    ggml_cuda_set_device(cuda_ctx->device);
#ifdef USE_CUDA_GRAPH
    if (ggml_cuda_graphs_take_stale(cuda_ctx)) {
        cuda_ctx->cuda_graphs.clear();
        cuda_ctx->last_uid = 0;
        cuda_ctx->last_key = nullptr;
    }
#endif
    ggml_cuda_fp4_gemm_graph_begin(*cuda_ctx);
    ggml_cuda_row_amax_graph_begin(*cuda_ctx);
    ggml_cuda_kv_max_graph_begin(*cuda_ctx);

    // a dependency another context asked for, applied before anything is
    // captured or launched, and after the scheduler's own input copies
    if (cuda_ctx->pending_order) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), cuda_ctx->pending_order, 0));
        cuda_ctx->pending_order = nullptr;
    }

    bool use_cuda_graph             = false;
    bool cuda_graph_update_required = false;
    const void * graph_key = nullptr;
    int  stats_why = 0;
    cudaEvent_t stats_e0 = nullptr, stats_e1 = nullptr;
    auto & cht = cuda_host_timer();
    const int64_t cht0 = cht.on ? ggml_time_us() : 0;
    if (ggml_cuda_graph_stats()) {
        CUDA_CHECK(cudaEventCreate(&stats_e0));
        CUDA_CHECK(cudaEventCreate(&stats_e1));
        CUDA_CHECK(cudaEventRecord(stats_e0, cuda_ctx->stream()));
    }

#ifdef USE_CUDA_GRAPH
    const int64_t cht_k0 = cht.on ? ggml_time_us() : 0;
    graph_key = ggml_cuda_graph_key_for(cuda_ctx, cgraph);
    const int64_t cht_key = cht.on ? ggml_time_us() : 0;

    ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);

    ggml_cuda_graph * graph = cuda_ctx->cuda_graph(graph_key);
    graph->n_seen++;
    const int n_shape_seen = [&] {
        auto & ss = ggml_cuda_shape_seen_of(cuda_ctx);
        return ++ss.seen[ss.last_shape_key];
    }();
    if (graph->is_enabled()) {
        const bool graph_compatible = cuda_ctx->last_compat;
        const int64_t cht_compat = cht.on ? ggml_time_us() : 0;
        if (graph_compatible) {
            const bool properties_changed = ggml_cuda_graph_update_required(cuda_ctx, cgraph, graph_key);
            if (cht.on) {
                cht.add_check(backend, (cht_key - cht_k0) / 1000.0,
                              (cht_compat - cht_key) / 1000.0,
                              (ggml_time_us() - cht_compat) / 1000.0);
            }

            // Evaluating node by node costs several times a capture, and a shape that only recurs now and then never
            // has two matching calls in a row. One eager call shows it is capturable; from the second on, capture.
            const bool seen_before = n_shape_seen > 1;  // calls with this shape at any address, this one counted

            if (!graph->warmup_complete) {
                if (!properties_changed || seen_before) {
                    graph->warmup_complete = true;
                    GGML_LOG_DEBUG("%s: CUDA graph warmup complete\n", __func__);
                    use_cuda_graph = true;
                    cuda_graph_update_required = true;
                } else {
                    stats_why = 3;
                }
                // else: the first calls of a new key - execute directly (use_cuda_graph stays false)
            } else {
                // Post-warmup: normal CUDA graph operation
                use_cuda_graph = true;
                if (properties_changed) {
                    // Properties changed - capture again over the new ones
                    cuda_graph_update_required = true;
                    stats_why = 4;
                    GGML_LOG_DEBUG("%s: CUDA graph re-capture\n", __func__);
                } else {
                    cuda_graph_update_required = graph->instance == nullptr;
                }
            }
        } else {
            stats_why = 2;
        }
    } else {
        stats_why = 1;
    }
#endif // USE_CUDA_GRAPH

    if (use_cuda_graph && cuda_graph_update_required) {
        // Start CUDA graph capture
        {
            std::lock_guard<std::mutex> lock(ggml_cuda_lock);
            ggml_cuda_lock_counter.fetch_add(1, std::memory_order_relaxed);
        }

        CUDA_CHECK(cudaStreamBeginCapture(cuda_ctx->stream(), cudaStreamCaptureModeRelaxed));
    }

    const int64_t cht1 = cht.on ? ggml_time_us() : 0;
    ggml_cuda_graph_evaluate_and_capture(cuda_ctx, cgraph, use_cuda_graph, cuda_graph_update_required, graph_key);
    if (cht.on) {
        const double ev_ms = (ggml_time_us() - cht1) / 1000.0;
        if (!use_cuda_graph) {
            int n_seen = 0;
#ifdef USE_CUDA_GRAPH
            if (graph_key) { n_seen = cuda_ctx->cuda_graph(graph_key)->n_seen; }
#endif
            cht.add_eager(backend, stats_why, cgraph->n_nodes, ev_ms, n_seen);
        }
        cht.add(backend, (cht1 - cht0) / 1000.0, ev_ms, use_cuda_graph, use_cuda_graph && cuda_graph_update_required);
    }

    if (stats_e0) {
        CUDA_CHECK(cudaEventRecord(stats_e1, cuda_ctx->stream()));
        CUDA_CHECK(cudaEventSynchronize(stats_e1));
        float ms = 0.0f;
        CUDA_CHECK(cudaEventElapsedTime(&ms, stats_e0, stats_e1));
        size_t gnodes = 0;
        int ops = 0;
        for (int i = 0; i < cgraph->n_nodes; i++) {
            if (!ggml_cuda_is_view_or_noop(cgraph->nodes[i])) ops++;
        }
#ifdef USE_CUDA_GRAPH
        if (use_cuda_graph) {
            ggml_cuda_graph * g = cuda_ctx->cuda_graph(graph_key);
            if (g->graph) { CUDA_CHECK(cudaGraphGetNodes(g->graph, nullptr, &gnodes)); }
        }
#endif
        fprintf(stderr, "[cgstat] nodes=%d ops=%d cuda_graph=%d update=%d why=%d gnodes=%zu ms=%.3f\n",
                cgraph->n_nodes, ops, (int) use_cuda_graph, (int) cuda_graph_update_required, stats_why, gnodes, ms);
        CUDA_CHECK(cudaEventDestroy(stats_e0));
        CUDA_CHECK(cudaEventDestroy(stats_e1));
    }

    return GGML_STATUS_SUCCESS;
}

static void ggml_backend_cuda_event_record(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    CUDA_CHECK(cudaEventRecord((cudaEvent_t)event->context, cuda_ctx->stream()));
}

static void ggml_backend_cuda_event_wait(ggml_backend_t backend, ggml_backend_event_t event) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *)backend->context;

    if (ggml_backend_is_cuda(backend)) {
        CUDA_CHECK(cudaStreamWaitEvent(cuda_ctx->stream(), (cudaEvent_t)event->context, 0));
    } else {
#if 0
        // untested
        auto wait_fn = [](void * user_data) {
            ggml_backend_event_t event = (ggml_backend_event_t)user_data;
            ggml_backend_event_synchronize(event);
        };

        CUDA_CHECK(cudaLaunchHostFunc(cuda_ctx->stream(), wait_fn, event));
#endif
        GGML_ABORT("fatal error");
    }
}

// Concurrent regions. Wherever one node feeds two or more chains of work that need nothing from each other until
// they meet again (q, k and v of an attention layer; the routed and the shared experts of a MoE layer; the gating
// projections beside the state projection of a delta-net layer), the chains run on their own streams between a
// fork and a join, inside the captured CUDA graph. Every kernel reads and writes what it did before; only the order
// between the chains is left to the GPU. Everything a region touches stays allocated until the join, so no chain can
// take over another's buffer, and no launch fuses across the join.
static constexpr int GGML_CUDA_REGION_MAX_BRANCHES = GGML_CUDA_MAX_STREAMS - 1;

static const ggml_tensor * ggml_cuda_region_base(const ggml_tensor * t) {
    return t->view_src ? t->view_src : t;
}

static bool ggml_cuda_region_reads(const ggml_tensor * node, const ggml_tensor * t) {
    for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
        if (node->src[s] == t || ggml_cuda_region_base(node->src[s]) == t) {
            return true;
        }
    }
    return false;
}

// a branch is worth a stream of its own when it carries a launch that streams real memory
static bool ggml_cuda_region_heavy(const ggml_tensor * n) {
    switch (n->op) {
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_FLASH_ATTN_EXT:
        case GGML_OP_GATED_DELTA_NET:
        case GGML_OP_SSM_CONV:
        case GGML_OP_SSM_SCAN:
        case GGML_OP_GET_ROWS:
        case GGML_OP_CONV_2D:
            return true;
        default:
            return false;
    }
}

static cudaEvent_t ggml_cuda_region_event(ggml_backend_cuda_context & ctx) {
    if (ctx.region_events_used == ctx.region_events.size()) {
        cudaEvent_t e = nullptr;
        CUDA_CHECK(cudaEventCreateWithFlags(&e, cudaEventDisableTiming));
        ctx.region_events.push_back(e);
    }
    return ctx.region_events[ctx.region_events_used++];
}

void ggml_cuda_region_publish(ggml_backend_cuda_context & ctx) {
    const ggml_cuda_concurrent_event * ev = ctx.active_region;
    if (ev == nullptr || ctx.curr_stream_no == 0) {
        return;
    }
    cudaEvent_t e = ggml_cuda_region_event(ctx);
    CUDA_CHECK(cudaEventRecord(e, ctx.stream()));
    for (int s = 1; s <= ev->n_streams; ++s) {
        if (s != ctx.curr_stream_no) {
            CUDA_CHECK(cudaStreamWaitEvent(ctx.stream(ctx.device, s), e));
        }
    }
}

static void ggml_cuda_find_concurrent_regions(ggml_backend_cuda_context * cuda_ctx, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    static const bool trace = getenv("GGML_CUDA_TRACE_REGIONS") != nullptr;
    const int n = cgraph->n_nodes;

    std::unordered_map<const ggml_tensor *, int> group; // a region node -> its group
    std::vector<int>  parent;                            // union-find over the groups
    std::vector<bool> seeded;                            // the group starts at a reader of the root
    auto find = [&](int g) {
        while (parent[g] != g) {
            parent[g] = parent[parent[g]];
            g = parent[g];
        }
        return g;
    };
    auto unite = [&](int a, int b) {
        a = find(a);
        b = find(b);
        if (a != b) {
            parent[b] = a;
            seeded[a] = seeded[a] || seeded[b];
        }
        return a;
    };

    // the compute nodes reading each tensor, directly or through a view, counted once: a root needs two. Searched for
    // each root, it made the finder quadratic in the graph, which a graph built at a new shape (a prompt step) paid
    // on the host each time
    std::unordered_map<const ggml_tensor *, int> n_readers_of;
    for (int k = 0; k < n; ++k) {
        const ggml_tensor * node = cgraph->nodes[k];
        if (ggml_cuda_is_view_or_noop(node)) {
            continue;
        }
        const ggml_tensor * seen[2*GGML_MAX_SRC];
        int n_seen = 0;
        for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
            const ggml_tensor * both[2] = { node->src[s], ggml_cuda_region_base(node->src[s]) };
            for (const ggml_tensor * t : both) {
                bool dup = false;
                for (int q = 0; q < n_seen && !dup; ++q) {
                    dup = seen[q] == t;
                }
                if (!dup) {
                    seen[n_seen++] = t;
                    n_readers_of[t]++;
                }
            }
        }
    }
    // a region spans at most this many nodes: the ones worth streams are a few launches wide (q, k and v; the routed
    // and shared experts; a layer beside its per-layer input), and a root whose readers meet no sooner is no region
    constexpr int max_span = 256;

    int n_regions = 0;
    for (int r = 0; r + 2 < n; ++r) {
        const ggml_tensor * root = cgraph->nodes[r];
        if (ggml_cuda_is_view_or_noop(root)) {
            continue;
        }
        const auto it_readers = n_readers_of.find(root);
        if (it_readers == n_readers_of.end() || it_readers->second < 2) {
            continue;
        }

        // forward from the root: a node joins the group of whatever region node it reads or writes into, and
        // merges the groups when it reads several; one that needs every group seeded by the root is the join; one
        // that needs nothing of the region floats until a reader takes it in
        group.clear();
        parent.clear();
        seeded.clear();
        // tensors from before the region and the groups that read them or wrote into them (a KV cache, a
        // recurrent state): a later write after a read, or any touch after a write, is a dependency the graph
        // edges do not show
        std::vector<std::pair<const ggml_tensor *, int>> touched_read;
        std::vector<std::pair<const ggml_tensor *, int>> touched_write;
        int join = -1;
        int prev = -1; // the last node before this one
        for (int k = r + 1; k < n && k <= r + max_span; ++k) {
            const ggml_tensor * node = cgraph->nodes[k];
            if (ggml_cuda_is_view_or_noop(node)) {
                continue;
            }
            int deps[4*GGML_MAX_SRC + 4];
            int n_deps   = 0;
            int n_seeded = 0;
            auto need_group = [&](int g) {
                g = find(g);
                for (int d = 0; d < n_deps; ++d) {
                    if (deps[d] == g) {
                        return;
                    }
                }
                if (n_deps == (int) (sizeof(deps)/sizeof(deps[0]))) {
                    return;
                }
                deps[n_deps++] = g;
                n_seeded += seeded[g] ? 1 : 0;
            };
            auto need = [&](const ggml_tensor * t) {
                auto it = group.find(t);
                if (it != group.end()) {
                    need_group(it->second);
                }
            };
            const ggml_tensor * wbase = node->view_src; // the node writes into this older tensor
            for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
                need(node->src[s]);
                need(ggml_cuda_region_base(node->src[s]));
                for (const auto & tw : touched_write) {
                    if (tw.first == ggml_cuda_region_base(node->src[s])) {
                        need_group(tw.second);
                    }
                }
            }
            if (wbase) {
                need(wbase);
                for (const auto & tw : touched_write) {
                    if (tw.first == wbase) {
                        need_group(tw.second);
                    }
                }
                for (const auto & tr : touched_read) {
                    if (tr.first == wbase) {
                        need_group(tr.second);
                    }
                }
            }
            if (n_seeded >= 2) {
                int n_live = 0;
                for (size_t g = 0; g < parent.size(); ++g) {
                    n_live += (seeded[g] && find((int) g) == (int) g) ? 1 : 0;
                }
                if (n_seeded == n_live) {
                    join = k;
                    break;
                }
            }
            const bool reads_root = ggml_cuda_region_reads(node, root);
            // the product before this one, looking past the scale its weights multiply it by (a per-tensor scale of
            // an NVFP4 or FP8 weight)
            const ggml_tensor * pm = prev >= 0 ? cgraph->nodes[prev] : nullptr;
            if (pm && pm->op == GGML_OP_MUL && pm->src[0] && (pm->src[0]->op == GGML_OP_MUL_MAT || pm->src[0]->op == GGML_OP_MUL_MAT_ID)) {
                pm = pm->src[0];
            }
            int g;
            if (n_deps == 0 && reads_root && pm && (node->op == GGML_OP_MUL_MAT || node->op == GGML_OP_MUL_MAT_ID) &&
                    pm->op == node->op && pm->src[1] == node->src[1] && pm->src[0]->type == node->src[0]->type && group.count(pm) &&
                    ggml_cuda_region_reads(pm, root)) {
                // the product right after one over the same input: the two launch as one (a grouped launch, a matvec
                // pair), so they are one branch
                g = find(group[pm]);
            } else if (n_deps == 0) {
                g = (int) parent.size();
                parent.push_back(g);
                seeded.push_back(reads_root);
            } else {
                g = deps[0];
                for (int d = 1; d < n_deps; ++d) {
                    g = unite(g, deps[d]);
                }
                seeded[g] = seeded[g] || reads_root;
            }
            group[node] = g;
            prev        = k;
            for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
                const ggml_tensor * base = ggml_cuda_region_base(node->src[s]);
                if (base != root && !group.count(base)) {
                    touched_read.emplace_back(base, g);
                }
            }
            if (wbase && !group.count(wbase)) {
                touched_write.emplace_back(wbase, g);
            }
        }
        if (join < 0) {
            continue;
        }
        ggml_tensor * join_node = cgraph->nodes[join];

        // a norm inside the region folds into its readers; a reader at or after the join would find nothing written
        bool ok = true;
        for (int k = r + 1; k < join && ok; ++k) {
            const ggml_tensor * node = cgraph->nodes[k];
            if (node->op != GGML_OP_RMS_NORM) {
                continue;
            }
            for (int m = join; m < n && ok; ++m) {
                ok = !ggml_cuda_region_reads(cgraph->nodes[m], node);
            }
        }
        if (!ok) {
            continue;
        }

        // the branches: the groups in order of first appearance; one not seeded by the root, without a heavy
        // launch, or past the stream count runs on the branch before it
        std::vector<int>  branch_of(parent.size(), 0);
        std::vector<int>  order;
        std::vector<bool> heavy;
        for (int k = r + 1; k < join; ++k) {
            const ggml_tensor * node = cgraph->nodes[k];
            if (ggml_cuda_is_view_or_noop(node)) {
                continue;
            }
            const int g = find(group[node]);
            if (branch_of[g] == 0) {
                order.push_back(g);
                heavy.push_back(false);
                branch_of[g] = (int) order.size();
            }
            heavy[branch_of[g] - 1] = heavy[branch_of[g] - 1] || ggml_cuda_region_heavy(node);
        }
        std::vector<int> merged(order.size(), 0);
        int n_branches = 0;
        for (size_t b = 0; b < order.size(); ++b) {
            if (n_branches == 0 || (seeded[order[b]] && heavy[b] && n_branches < GGML_CUDA_REGION_MAX_BRANCHES)) {
                n_branches++;
            }
            merged[b] = n_branches;
        }
        if (n_branches < 2) {
            continue;
        }
        auto branch_of_node = [&](const ggml_tensor * node) {
            auto it = group.find(node);
            return it == group.end() ? 0 : merged[branch_of[find(it->second)] - 1];
        };

        ggml_cuda_concurrent_event ev(n_branches);
        ev.fork_node = root;
        ev.join_node = join_node;
        std::vector<int> per_branch(n_branches + 1, 0);
        int last_branch = 1;
        for (int k = r + 1; k < join; ++k) {
            ggml_tensor * node = cgraph->nodes[k];
            int b = branch_of_node(node);
            if (b == 0) {
                // a view: of a region node, or of something older, read by one branch
                b = branch_of_node(ggml_cuda_region_base(node));
                for (int m = k + 1; m < join && b == 0; ++m) {
                    if (ggml_cuda_region_reads(cgraph->nodes[m], node)) {
                        b = branch_of_node(cgraph->nodes[m]);
                    }
                }
                if (b == 0) {
                    b = last_branch;
                }
            }
            ev.stream_mapping[node] = b;
            last_branch = b;
            if (ggml_cuda_is_view_or_noop(node)) {
                continue;
            }
            per_branch[b]++;
            // every compute tensor the region writes or reads stays allocated until the join
            params->add_alloc_dep(params->user_data, node, join_node);
            for (int s = 0; s < GGML_MAX_SRC && node->src[s]; ++s) {
                ggml_tensor * src = const_cast<ggml_tensor *>(ggml_cuda_region_base(node->src[s]));
                if (src->buffer != nullptr && ggml_backend_buffer_get_usage(src->buffer) != GGML_BACKEND_BUFFER_USAGE_COMPUTE) {
                    continue;
                }
                if (group.count(src) == 0) {
                    params->add_alloc_dep(params->user_data, src, join_node);
                }
            }
        }
        params->add_alloc_dep(params->user_data, cgraph->nodes[r], join_node);

        if (trace) {
            fprintf(stderr, "[region] %s -> %s: %d branches of", root->name, join_node->name, n_branches);
            for (int b = 1; b <= n_branches; ++b) {
                fprintf(stderr, " %d", per_branch[b]);
            }
            fprintf(stderr, " nodes\n");
        }
        GGML_ASSERT(cuda_ctx->stream_context().concurrent_events.find(root) == cuda_ctx->stream_context().concurrent_events.end());
        cuda_ctx->stream_context().concurrent_events.emplace(root, std::move(ev));
        n_regions++;
        r = join - 1;
    }
    if (trace) {
        fprintf(stderr, "[region] %d regions in a graph of %d nodes\n", n_regions, n);
    }
}

static void ggml_backend_cuda_graph_optimize(ggml_backend_t backend, ggml_cgraph * cgraph, ggml_backend_graph_optimize_params * params) {
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;

    static const bool disable_fusion = getenv("GGML_CUDA_DISABLE_FUSION") != nullptr && std::atoi(getenv("GGML_CUDA_DISABLE_FUSION"));
    if (!disable_fusion) {
        for (int i = 0; i < cgraph->n_nodes; ++i) {
            // the top-k gating fusion writes its weights and ids while it reads the logits and the elided
            // nodes between; it is taken only when the allocator placed none of its outputs over them, which
            // otherwise depends on the batch. Keeping the inputs until the weights makes it fire every time.
            const ggml_op op = cgraph->nodes[i]->op;
            if (op == GGML_OP_UNARY || op == GGML_OP_SOFT_MAX || op == GGML_OP_ARGSORT) {
                ggml_cuda_topk_moe_args args;
                if (ggml_cuda_topk_moe_fusion(cgraph, i, args)) {
                    int n_ops = 0;
                    if (!args.delayed_softmax) {
                        n_ops = (args.sqrt_softplus ? 2 : 1) + (args.prob_bias ? 5 : 4) + (args.norm ? 5 : 0) + (args.scale ? 1 : 0);
                    } else if (!args.norm && !args.prob_bias) {
                        n_ops = 6;
                    }
                    if (n_ops > 0 && i + n_ops <= cgraph->n_nodes) {
                        ggml_tensor * last = cgraph->nodes[i + n_ops - 1];
                        params->add_alloc_dep(params->user_data, cgraph->nodes[i]->src[0], last);
                        for (int k = i; k + 1 < i + n_ops; ++k) {
                            params->add_alloc_dep(params->user_data, cgraph->nodes[k], last);
                        }
                    }
                }
            }
            if (cgraph->nodes[i]->op != GGML_OP_MUL) {
                continue;
            }

            ggml_cuda_moe_weighted_reduction_match match;
            if (!ggml_cuda_match_moe_weighted_reduction(cgraph, i, match)) {
                continue;
            }

            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.experts), match.dst);
            params->add_alloc_dep(params->user_data, const_cast<ggml_tensor *>(match.weights), match.dst);
            if (match.expert_scale != nullptr) {
                params->add_alloc_dep(
                    params->user_data, const_cast<ggml_tensor *>(match.expert_scale), match.dst);
            }
            i += match.node_count - 1;
        }
    }

#ifdef USE_CUDA_GRAPH
    const void * graph_key = ggml_cuda_graph_get_key(cgraph, nullptr);
    const bool use_cuda_graph = ggml_cuda_graph_set_enabled(cuda_ctx, graph_key);
#else
    const bool use_cuda_graph = false;
    GGML_UNUSED(cuda_ctx);
    GGML_UNUSED(cgraph);
#endif

    // GGML_CUDA_GRAPH_OPT: unset or 0 keeps every graph on one stream, 1 is the attention-only pass below, 2 finds
    // every concurrent region (see ggml_cuda_find_concurrent_regions). Measured 2026-10-27 on the 35B MoE: the
    // regions take 0.2 ms off the GPU side of a 5 ms round and put 0.3 ms back on the host (branched graphs launch
    // slower and re-instantiate more often), so they stay opt-in.
    static const int graph_opt_mode = [] {
        const char * env = getenv("GGML_CUDA_GRAPH_OPT");
        return env == nullptr ? 0 : atoi(env);
    }();

    if (graph_opt_mode == 0) {
        return;
    }

    ggml_cuda_stream_context & stream_context = cuda_ctx->stream_context();
    stream_context.reset();

    if (!use_cuda_graph) {
        return;
    }

    if (graph_opt_mode != 1) {
        ggml_cuda_find_concurrent_regions(cuda_ctx, cgraph, params);
        return;
    }

    ggml_cuda_set_device(cuda_ctx->device);

    // number of out-degrees for a particular node
    std::unordered_map<const ggml_tensor *, int> fan_out;
    // reverse mapping of node to index in the cgraph
    std::unordered_map<const ggml_tensor *, int> node_indices;

    const auto & is_noop = [](const ggml_tensor * node) -> bool {
        return ggml_is_empty(node) || node->op == GGML_OP_NONE || node->op == GGML_OP_RESHAPE ||
               node->op == GGML_OP_TRANSPOSE || node->op == GGML_OP_VIEW || node->op == GGML_OP_PERMUTE;
    };

    const auto & depends_on = [](const ggml_tensor * dst, const ggml_tensor * src) -> bool {
        for (uint32_t s = 0; s < GGML_MAX_SRC; ++s) {
            if (dst->src[s] == src) {
                return true;
            }
        }
        // implicit dependency if they view the same tensor
        const ggml_tensor * dst2 = dst->view_src ? dst->view_src : dst;
        const ggml_tensor * src2 = src->view_src ? src->view_src : src;
        if (dst2 == src2) {
            return true;
        }
        return false;
    };

    for (int node_idx = 0; node_idx < cgraph->n_nodes; node_idx++) {
        const ggml_tensor * node = cgraph->nodes[node_idx];
        node_indices[node]       = node_idx;

        if (is_noop(node)) {
            continue;
        }
        for (int src_idx = 0; src_idx < GGML_MAX_SRC; ++src_idx) {
            const ggml_tensor * src = cgraph->nodes[node_idx]->src[src_idx];
            //TODO: check why nrows > 1 fails
            if (node && !is_noop(node) && ggml_nrows(node) <= 1) {
                fan_out[src] += 1;
            }
        }
    }

    // Target Q, K, V for concurrency
    // this is a more general way to find nodes which can be candidates for concurrency (although it has not been tested for anything else):
    // 1. find fan-out (fork) nodes where the same input is used at least N times (in QKV, it would be "attn-norm")
    // 2. find the join node, where 2 or more of the outputs are required (in QKV, this would "KQ" or "flash-attn")
    // 3. account for all branches from the fork to the join
    // 4. To extend lifetimes of the tensors, we interleave the branches (see below for more details)
    // 5. save the original cgraph and restore it in graph_compute, to enable fusion within streams
    // See discussion: https://github.com/ggml-org/llama.cpp/pull/16991#issuecomment-3522620030

    const int min_fan_out = 3;
    const int max_fan_out = 3;

    // store {fork_idx, join_idx}
    std::vector<std::pair<int, int>> concurrent_node_ranges;

    for (const auto & [root_node, count] : fan_out) {
        if (count >= min_fan_out && count <= max_fan_out) {
            const int root_node_idx = node_indices[root_node];

            // only optimize for attn_norm
            // TODO: make this more generic
            if (!strstr(root_node->name, "attn_norm")) {
                continue;
            }

            bool is_part_of_event = false;
            for (const auto & [start, end] : concurrent_node_ranges) {
                if (root_node_idx >= start && root_node_idx <= end) {
                    is_part_of_event = true;
                }
            }

            if (is_part_of_event) {
                continue;
            }

            std::vector<std::vector<const ggml_tensor *>> nodes_per_branch;
            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * node = cgraph->nodes[i];
                if (!is_noop(node) && depends_on(node, root_node)) {
                    nodes_per_branch.push_back({ node });
                }
            }

            GGML_ASSERT(nodes_per_branch.size() == (size_t) count);

            //find the join point
            const ggml_tensor * join_node = nullptr;

            const auto & belongs_to_branch = [&](const ggml_tensor *                      node,
                                                 const std::vector<const ggml_tensor *> & branch) -> bool {
                for (const ggml_tensor * n : branch) {
                    if (depends_on(node, n)) {
                        return true;
                    }
                }
                return false;
            };

            for (int i = root_node_idx + 1; i < cgraph->n_nodes; ++i) {
                const ggml_tensor * curr_node = cgraph->nodes[i];

                int num_joins = 0;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    if (belongs_to_branch(curr_node, nodes_per_branch[branch_idx])) {
                        num_joins++;
                    }
                }

                if (num_joins >= 2) {
                    join_node = curr_node;
                    break;
                }

                bool found_branch = false;
                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    std::vector<const ggml_tensor *> & branch_vec = nodes_per_branch[branch_idx];
                    if (belongs_to_branch(curr_node, branch_vec)) {
                        //continue accumulating
                        if (std::find(branch_vec.begin(), branch_vec.end(), curr_node) == branch_vec.end()) {
                            branch_vec.push_back(curr_node);
                        }
                        found_branch = true;
                    }
                }

                if (!found_branch && is_noop(curr_node)) {
                    // we can put it in any branch because it will be ignored
                    nodes_per_branch[0].push_back({ curr_node });
                }
            }

            if (join_node) {
                //Create ggml_cuda_concurrent_event
                ggml_cuda_concurrent_event concurrent_event(nodes_per_branch.size());
                concurrent_event.join_node = join_node;

                for (size_t branch_idx = 0; branch_idx < nodes_per_branch.size(); branch_idx++) {
                    for (const ggml_tensor * n : nodes_per_branch[branch_idx]) {
                        concurrent_event.stream_mapping[n] = branch_idx + 1;
                    }
                }

                int fork_node_idx = node_indices[root_node];
                int join_node_idx = node_indices[join_node];

                int       current_branch_idx = 0;
                int       current_node_idx   = fork_node_idx + 1;
                const int n_branches         = nodes_per_branch.size();

                int total_branch_nodes = 0;
                for (std::vector<const ggml_tensor *> branch_nodes : nodes_per_branch) {
                    total_branch_nodes += branch_nodes.size();
                }

                // there are other nodes in the middle which are unaccounted for
                // usually (cpy) nodes, then ignore this fork
                if (join_node_idx - fork_node_idx - 1 != total_branch_nodes) {
                    GGML_LOG_DEBUG(
                        "Skipping %s because the number of nodes in the middle is not equal to the total number of "
                        "branch nodes %d != %d\n",
                        root_node->name, join_node_idx - fork_node_idx - 1, total_branch_nodes);
                    continue;
                }

                // Save the original order of nodes in this region before interleaving
                // This is used later to restore grouping for fusion within streams
                concurrent_event.original_order.reserve(total_branch_nodes);
                for (int i = fork_node_idx + 1; i < join_node_idx; ++i) {
                    concurrent_event.original_order.push_back(cgraph->nodes[i]);
                }

                std::unordered_map<const ggml_tensor *, ggml_cuda_concurrent_event> & concurrent_events = cuda_ctx->stream_context().concurrent_events;
                GGML_ASSERT(concurrent_events.find(root_node) == concurrent_events.end());
                concurrent_events.emplace(root_node, std::move(concurrent_event));
                GGML_LOG_DEBUG("Adding stream at node %s %p\n", root_node->name, root_node);
                concurrent_node_ranges.emplace_back(fork_node_idx, join_node_idx);

                // interleave tensors to extend lifetimes so that ggml graph doesn't recycle them
                // example transformation:
                // [attn-norm, QMul, QNorm, QRope, KMul, KNorm, KRope, VMul, attn] ->
                // [attn-norm, QMul, KMul, VMul, QNorm, VNorm, QRope, KRope, attn]
                while (current_node_idx < join_node_idx) {
                    std::vector<const ggml_tensor *> & branch_nodes = nodes_per_branch[current_branch_idx];

                    bool has_node = false;
                    for (std::vector<const ggml_tensor *> branch_node : nodes_per_branch) {
                        has_node |= branch_node.size() > 0;
                    }

                    GGML_ASSERT(has_node);

                    if (branch_nodes.empty()) {
                        current_branch_idx = (current_branch_idx + 1) % n_branches;
                        continue;
                    }

                    cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                    current_node_idx++;
                    branch_nodes.erase(branch_nodes.begin());

                    // append all empty nodes
                    while (!branch_nodes.empty() && is_noop(branch_nodes.front())) {
                        cgraph->nodes[current_node_idx] = const_cast<ggml_tensor *>(branch_nodes.front());
                        current_node_idx++;
                        branch_nodes.erase(branch_nodes.begin());
                    }

                    current_branch_idx = (current_branch_idx + 1) % n_branches;
                }
            }
        }
    }
}

static const ggml_backend_i ggml_backend_cuda_interface = {
    /* .get_name                = */ ggml_backend_cuda_get_name,
    /* .free                    = */ ggml_backend_cuda_free,
    /* .set_tensor_async        = */ ggml_backend_cuda_set_tensor_async,
    /* .get_tensor_async        = */ ggml_backend_cuda_get_tensor_async,
    /* .set_tensor_2d_async     = */ ggml_backend_cuda_set_tensor_2d_async,
    /* .get_tensor_2d_async     = */ ggml_backend_cuda_get_tensor_2d_async,
    /* .cpy_tensor_async        = */ ggml_backend_cuda_cpy_tensor_async,
    /* .synchronize             = */ ggml_backend_cuda_synchronize,
    /* .graph_plan_create       = */ NULL,
    /* .graph_plan_free         = */ NULL,
    /* .graph_plan_update       = */ NULL,
    /* .graph_plan_compute      = */ NULL,
    /* .graph_compute           = */ ggml_backend_cuda_graph_compute,
    /* .event_record            = */ ggml_backend_cuda_event_record,
    /* .event_wait              = */ ggml_backend_cuda_event_wait,
    /* .graph_optimize          = */ ggml_backend_cuda_graph_optimize,
};

static ggml_guid_t ggml_backend_cuda_guid() {
    static ggml_guid guid = { 0x2c, 0xdd, 0xe8, 0x1c, 0x65, 0xb3, 0x65, 0x73, 0x6a, 0x12, 0x88, 0x61, 0x1c, 0xc9, 0xdc, 0x25 };
    return &guid;
}

bool ggml_backend_is_cuda(ggml_backend_t backend) {
    return backend != NULL && ggml_guid_matches(backend->guid, ggml_backend_cuda_guid());
}

int ggml_backend_cuda_get_device_count() {
    return ggml_cuda_info().device_count;
}

static std::string ggml_cuda_device_description(int device) {
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(device)));

    const ggml_cuda_device_info & info = ggml_cuda_info();
    std::string description = prop.name;
    if (info.device_count > info.physical_device_count) {
        description += " (dev p" + std::to_string(info.devices[device].physical_device) +
                       "/v" + std::to_string(info.devices[device].virtual_index) + ")";
    }
    return description;
}

void ggml_backend_cuda_get_device_description(int device, char * description, size_t description_size) {
    snprintf(description, description_size, "%s", ggml_cuda_device_description(device).c_str());
}

static int ggml_cuda_physical_device_share_count(int device) {
    const ggml_cuda_device_info & info = ggml_cuda_info();
    GGML_ASSERT(device >= 0 && device < info.device_count);
    return info.devices[device].physical_share_count;
}

void ggml_backend_cuda_get_device_memory(int device, size_t * free, size_t * total) {
    ggml_cuda_set_device(device);

    CUDA_CHECK(cudaMemGetInfo(free, total));

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(device);
    *free  /= share_count;
    *total /= share_count;
}

bool ggml_backend_cuda_register_host_buffer(void * buffer, size_t size) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return false;
    }

#if CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA) || defined(GGML_USE_HIP)
    cudaError_t err = cudaHostRegister(buffer, size, cudaHostRegisterPortable | cudaHostRegisterReadOnly);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();

        GGML_LOG_DEBUG("%s: failed to register %.2f MiB of pinned memory: %s\n", __func__,
                           size / 1024.0 / 1024.0, cudaGetErrorString(err));
        return false;
    }
    return true;
#else
    GGML_UNUSED(buffer);
    GGML_UNUSED(size);
    return false;
#endif // CUDART_VERSION >= 11010 || defined(GGML_USE_MUSA)
}

void ggml_backend_cuda_unregister_host_buffer(void * buffer) {
    if (getenv("GGML_CUDA_REGISTER_HOST") == nullptr) {
        return;
    }

    cudaError_t err = cudaHostUnregister(buffer);
    if (err != cudaSuccess) {
        // clear the error
        (void)cudaGetLastError();
    }
}


// backend device

struct ggml_backend_cuda_device_context {
    int device;
    std::string name;
    std::string description;
    std::string pci_bus_id;
    int op_offload_min_batch_size;
};

static const char * ggml_backend_cuda_device_get_name(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->name.c_str();
}

static const char * ggml_backend_cuda_device_get_description(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ctx->description.c_str();
}

#if defined(__linux__)
// Helper function to get available memory from /proc/meminfo for UMA systems
static bool ggml_backend_cuda_get_available_uma_memory(long * available_memory_kb, long * free_swap_kb) {
    FILE * meminfo_file = nullptr;
    // 2KB buffer for reading /proc/meminfo since it does not report size info, should be enough
    const size_t BUFFER_SIZE = 2048;
    auto file_buffer = std::make_unique<char[]>(BUFFER_SIZE);
    size_t bytes_read = 0;
    long huge_tlb_total_pages = -1;
    long huge_tlb_free_pages = -1;
    long huge_tlb_page_size = -1;

    if (available_memory_kb == nullptr || free_swap_kb == nullptr) {
        return false;
    }

    meminfo_file = fopen("/proc/meminfo", "r");
    if (meminfo_file == nullptr) {
        GGML_LOG_ERROR("%s: failed to open /proc/meminfo\n", __func__);
        return false;
    }

    // Read file into buffer
    bytes_read = fread(file_buffer.get(), 1, BUFFER_SIZE - 1, meminfo_file);
    fclose(meminfo_file);

    if (bytes_read == 0) {
        GGML_LOG_ERROR("%s: failed to read from /proc/meminfo\n", __func__);
        return false;
    }
    file_buffer[bytes_read] = '\0';

    *available_memory_kb = -1;
    *free_swap_kb = -1;

    // Parse the file buffer line by line
    char * line = file_buffer.get();
    char * line_next;
    while (line < file_buffer.get() + bytes_read) {
        // Find the end of the current line
        line_next = strchr(line, '\n');
        if (line_next != nullptr) {
            *line_next = '\0';
            line_next++;
        } else {
            line_next = file_buffer.get() + bytes_read;
        }

        long value;
        if (sscanf(line, "MemAvailable: %ld kB", &value) == 1) {
            *available_memory_kb = value;
        } else if (sscanf(line, "SwapFree: %ld kB", &value) == 1) {
            *free_swap_kb = value;
        } else if (sscanf(line, "HugePages_Total: %ld", &value) == 1) {
            huge_tlb_total_pages = value;
        } else if (sscanf(line, "HugePages_Free: %ld", &value) == 1) {
            huge_tlb_free_pages = value;
        } else if (sscanf(line, "Hugepagesize: %ld kB", &value) == 1) {
            huge_tlb_page_size = value;
        }

        line = line_next;
    }

    if (huge_tlb_total_pages != 0 && huge_tlb_total_pages != -1) {
        *available_memory_kb = huge_tlb_free_pages * huge_tlb_page_size;

        // Hugetlbfs pages are not swappable.
        *free_swap_kb = 0;
    }

    GGML_LOG_DEBUG("%s: final available_memory_kb: %ld\n", __func__, *available_memory_kb);
    return true;
}
#endif // defined(__linux__)

static void ggml_backend_cuda_device_get_memory(ggml_backend_dev_t dev, size_t * free, size_t * total) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    ggml_cuda_set_device(ctx->device);
    cudaError_t err = cudaMemGetInfo(free, total);
    if (err != cudaSuccess) {
        (void)cudaGetLastError();
        GGML_LOG_WARN("%s: cudaMemGetInfo failed (%s), returning 0/0\n", __func__, cudaGetErrorString(err));
        *free = 0;
        *total = 0;
        return;
    }

// ref: https://github.com/ggml-org/llama.cpp/pull/17368
#if defined(__linux__) && !defined(GGML_USE_HIP)
    // Check if this is a UMA (Unified Memory Architecture) system
    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    // Check if UMA is explicitly enabled via environment variable
    bool uma_env = getenv("GGML_CUDA_ENABLE_UNIFIED_MEMORY") != nullptr;
    bool is_uma = prop.integrated > 0 || uma_env;

    if (is_uma) {
        // For UMA systems (like DGX Spark), use system memory info
        long available_memory_kb = 0;
        long free_swap_kb = 0;

        if (ggml_backend_cuda_get_available_uma_memory(&available_memory_kb, &free_swap_kb) && available_memory_kb > 0) {
            *free = (size_t)available_memory_kb * 1024;
        } else {
            GGML_LOG_ERROR("%s: /proc/meminfo reading failed, using cudaMemGetInfo\n", __func__);
        }
    }
#endif // defined(__linux__) && !defined(GGML_USE_HIP)

    // virtual devices sharing one physical GPU share its memory pool; split it between them
    const int share_count = ggml_cuda_physical_device_share_count(ctx->device);
    *free  /= share_count;
    *total /= share_count;
}

static enum ggml_backend_dev_type ggml_backend_cuda_device_get_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *) dev->context;

    cudaDeviceProp prop;
    CUDA_CHECK(cudaGetDeviceProperties(&prop, ggml_cuda_get_physical_device(ctx->device)));

    return prop.integrated
        ? GGML_BACKEND_DEVICE_TYPE_IGPU
        : GGML_BACKEND_DEVICE_TYPE_GPU;
}

static void ggml_backend_cuda_device_get_props(ggml_backend_dev_t dev, ggml_backend_dev_props * props) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;

    props->name        = ggml_backend_cuda_device_get_name(dev);
    props->description = ggml_backend_cuda_device_get_description(dev);
    props->type        = ggml_backend_cuda_device_get_type(dev);
    props->device_id   = ctx->pci_bus_id.empty() ? nullptr : ctx->pci_bus_id.c_str();
    ggml_backend_cuda_device_get_memory(dev, &props->memory_free, &props->memory_total);

    bool host_buffer = getenv("GGML_CUDA_NO_PINNED") == nullptr;
#ifdef GGML_CUDA_NO_PEER_COPY
    bool events = false;
#else
    bool events = true;
#endif

    props->caps = {
        /* .async                 = */ true,
        /* .host_buffer           = */ host_buffer,
        /* .buffer_from_host_ptr  = */ false,
        /* .events                = */ events,
        /* .mmap_support          = */ props->type != GGML_BACKEND_DEVICE_TYPE_IGPU,
    };
}

static ggml_backend_t ggml_backend_cuda_device_init_backend(ggml_backend_dev_t dev, const char * params) {
    GGML_UNUSED(params);
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_init(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_buffer_type(ggml_backend_dev_t dev) {
    ggml_backend_cuda_device_context * ctx = (ggml_backend_cuda_device_context *)dev->context;
    return ggml_backend_cuda_buffer_type(ctx->device);
}

static ggml_backend_buffer_type_t ggml_backend_cuda_device_get_host_buffer_type(ggml_backend_dev_t dev) {
    GGML_UNUSED(dev);
    return ggml_backend_cuda_host_buffer_type();
}

// TODO: move these functions here
static bool ggml_backend_cuda_device_supports_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    // check if all the sources are allocated on this device
    for (int i = 0; i < GGML_MAX_SRC; i++) {
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda(op->src[i]->buffer->buft)) {
            ggml_backend_cuda_buffer_type_context * buft_ctx = (ggml_backend_cuda_buffer_type_context *)op->src[i]->buffer->buft->context;
            if (buft_ctx->device != dev_ctx->device) {
                return false;
            }
        }
        // a mapped source is read over the bus: fine for the rows a gather touches, never for a matrix product
        if (op->src[i] && op->src[i]->buffer && ggml_backend_buft_is_cuda_mapped(op->src[i]->buffer->buft) &&
                (op->op == GGML_OP_MUL_MAT || op->op == GGML_OP_MUL_MAT_ID || op->op == GGML_OP_OUT_PROD)) {
            return false;
        }
    }

    switch (op->op) {
        case GGML_OP_UNARY:
            switch (ggml_get_unary_op(op)) {
                case GGML_UNARY_OP_ABS:
                case GGML_UNARY_OP_SGN:
                case GGML_UNARY_OP_NEG:
                case GGML_UNARY_OP_STEP:
                case GGML_UNARY_OP_GELU:
                case GGML_UNARY_OP_SILU:
                case GGML_UNARY_OP_RELU:
                case GGML_UNARY_OP_SIGMOID:
                case GGML_UNARY_OP_HARDSIGMOID:
                case GGML_UNARY_OP_HARDSWISH:
                case GGML_UNARY_OP_GELU_ERF:
                case GGML_UNARY_OP_GELU_QUICK:
                case GGML_UNARY_OP_TANH:
                case GGML_UNARY_OP_EXP:
                case GGML_UNARY_OP_EXPM1:
                case GGML_UNARY_OP_SOFTPLUS:
                case GGML_UNARY_OP_ELU:
                case GGML_UNARY_OP_XIELU:
                case GGML_UNARY_OP_FLOOR:
                case GGML_UNARY_OP_CEIL:
                case GGML_UNARY_OP_ROUND:
                case GGML_UNARY_OP_TRUNC:
                    // TODO: should become:
                    //return ggml_is_contiguous_rows(op->src[0]);
                    return ggml_is_contiguous(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_GLU:
            switch (ggml_get_glu_op(op)) {
                case GGML_GLU_OP_REGLU:
                case GGML_GLU_OP_GEGLU:
                case GGML_GLU_OP_SWIGLU:
                case GGML_GLU_OP_SWIGLU_OAI:
                case GGML_GLU_OP_GEGLU_ERF:
                case GGML_GLU_OP_GEGLU_QUICK:
                case GGML_GLU_OP_SWIGLU_CLAMP:
                    return ggml_is_contiguous_1(op->src[0]);
                default:
                    return false;
            }
            break;
        case GGML_OP_MUL_MAT:
        case GGML_OP_MUL_MAT_ID:
            {
                struct ggml_tensor * a = op->src[0];
                struct ggml_tensor * b = op->src[1];
                if (a->nb[0] != ggml_element_size(a) || b->nb[0] != ggml_element_size(b)) {
                    return false; // TODO this could in principle be implemented though currently there is no use case.
                }
                if (b->type == GGML_TYPE_F16 && a->type != GGML_TYPE_F16) {
                    return false;
                }
#ifdef GGML_USE_MUSA
                const int cc = ggml_cuda_info().devices[dev_ctx->device].cc;
                if (b->ne[2]*b->ne[3] > 1 && !ggml_is_transposed(a) && !ggml_is_transposed(b)) {
                    if (GGML_CUDA_CC_IS_QY1(cc) && op->op == GGML_OP_MUL_MAT &&
                            a->type == GGML_TYPE_F16 && b->type == GGML_TYPE_F16) {
                        return false;
                    }
                    if (GGML_CUDA_CC_IS_QY2(cc) && op->op == GGML_OP_MUL_MAT_ID &&
                            a->type == GGML_TYPE_Q2_K && b->type == GGML_TYPE_F32) {
                        return false;
                    }
                }
#endif // GGML_USE_MUSA
                switch (a->type) {
                    case GGML_TYPE_F8_E4M3:
                        return op->op == GGML_OP_MUL_MAT && ggml_cuda_fp8_mul_mat_supported(a, b, op);
                    case GGML_TYPE_F32:
                    case GGML_TYPE_F16:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_MXFP4:
                    case GGML_TYPE_NVFP4:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_Q8_K:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_IQ4_XS:
                    case GGML_TYPE_BF16:
                        return true;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_OUT_PROD:
            return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32;
        case GGML_OP_GET_ROWS:
            {
                switch (op->src[0]->type) {
                    case GGML_TYPE_F16:
                    case GGML_TYPE_F32:
                    case GGML_TYPE_BF16:
                    case GGML_TYPE_I32:
                    case GGML_TYPE_Q1_0:
                    case GGML_TYPE_Q2_0:
                    case GGML_TYPE_Q4_0:
                    case GGML_TYPE_Q4_1:
                    case GGML_TYPE_Q5_0:
                    case GGML_TYPE_Q5_1:
                    case GGML_TYPE_Q8_0:
                    case GGML_TYPE_Q2_K:
                    case GGML_TYPE_Q3_K:
                    case GGML_TYPE_Q4_K:
                    case GGML_TYPE_Q5_K:
                    case GGML_TYPE_Q6_K:
                    case GGML_TYPE_IQ2_XXS:
                    case GGML_TYPE_IQ2_XS:
                    case GGML_TYPE_IQ2_S:
                    case GGML_TYPE_IQ3_XXS:
                    case GGML_TYPE_IQ3_S:
                    case GGML_TYPE_IQ1_S:
                    case GGML_TYPE_IQ1_M:
                    case GGML_TYPE_IQ4_XS:
                        return true;
                    case GGML_TYPE_IQ4_NL:
                    case GGML_TYPE_MXFP4:
                        // 32-value sub-blocks, the row size does not guarantee
                        // the QK_K super-blocks the get_rows kernel iterates on
                        return op->src[0]->ne[0] % QK_K == 0;
                    default:
                        return false;
                }
            } break;
        case GGML_OP_GET_ROWS_BACK:
            {
                return op->type == GGML_TYPE_F32 && op->src[0]->type == GGML_TYPE_F32 && op->ne[2] == 1 && op->ne[3] == 1;
            } break;
        case GGML_OP_SET_ROWS:
            {
                return (
                           (
                               (op->type == GGML_TYPE_F32 || op->type == GGML_TYPE_F16 || op->type == GGML_TYPE_BF16 ||
                               op->type == GGML_TYPE_Q4_0 || op->type == GGML_TYPE_Q4_1 || op->type == GGML_TYPE_Q5_0 ||
                               op->type == GGML_TYPE_Q5_1 || op->type == GGML_TYPE_Q8_0 || op->type == GGML_TYPE_IQ4_NL ||
                               op->type == GGML_TYPE_F8_E4M3) &&
                               op->src[0]->type == GGML_TYPE_F32
                           ) || (
                               op->type == GGML_TYPE_F16 && op->src[0]->type == GGML_TYPE_F16
                           )
                       ) &&
                       (op->src[1]->type == GGML_TYPE_I64 || op->src[1]->type == GGML_TYPE_I32);
            } break;
        case GGML_OP_SET:
            {
                const ggml_type t = op->type;
                return (t == GGML_TYPE_F32 || t == GGML_TYPE_I32) &&
                    t == op->src[0]->type &&
                    t == op->src[1]->type;
            } break;
        case GGML_OP_CPY:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if ((src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_BF16 || src0_type == GGML_TYPE_F16) &&
                    (src1_type == GGML_TYPE_F32 || src1_type == GGML_TYPE_BF16 || src1_type == GGML_TYPE_F16)
                ) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q8_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q8_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q4_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q4_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_0) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_0 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_Q5_1) {
                    return true;
                }
                if (src0_type == GGML_TYPE_Q5_1 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_IQ4_NL) {
                    return true;
                }
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                if (src0_type == GGML_TYPE_I32 && src1_type == GGML_TYPE_I32) {
                    return true;
                }
                if (src0_type == src1_type && ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1])) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_DUP:
                return true;
        case GGML_OP_ARGMAX:
        case GGML_OP_COUNT_EQUAL:
            {
                return true;
            } break;
        case GGML_OP_REPEAT:
            {
                // the CUDA REPEAT path only implements F32/F16; other types assert at runtime
                ggml_type src0_type = op->src[0]->type;
                return src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16;
            } break;
        case GGML_OP_REPEAT_BACK:
                return op->type == GGML_TYPE_F32 && (op->src[0]->ne[2]*op->src[0]->ne[3]) <= (1 << 15);
        case GGML_OP_CONCAT:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                const int32_t dim = op->op_params[0];
                return src0_type == src1_type &&
                       src0_type == op->type &&
                       (
                           (
                               ggml_is_quantized(src0_type) &&
                               (
                                   (
                                       dim == 3 &&
                                       ggml_is_contiguous(op->src[0]) &&
                                       ggml_is_contiguous(op->src[1])
                                   ) || (
                                       dim != 3 &&
                                       ggml_is_contiguous_to_3(op->src[0]) &&
                                       ggml_is_contiguous_to_3(op->src[1])
                                   )
                               ) &&
                               op->src[0]->ne[0] % ggml_blck_size(src0_type) == 0 &&
                               op->src[1]->ne[0] % ggml_blck_size(src0_type) == 0
                           ) || (
                               !ggml_is_quantized(src0_type) &&
                               ggml_blck_size(src0_type) == 1 &&
                               (
                                   ggml_type_size(src0_type) == 1 ||
                                   ggml_type_size(src0_type) == 2 ||
                                   ggml_type_size(src0_type) == 4 ||
                                   ggml_type_size(src0_type) == 8
                               )
                           )
                       );
            } break;
        case GGML_OP_CONV_TRANSPOSE_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                ggml_type src1_type = op->src[1]->type;
                if (src0_type == GGML_TYPE_F32 && src1_type == GGML_TYPE_F32) {
                    return true;
                }
                return false;
            } break;
        case GGML_OP_COL2IM_1D:
            {
                ggml_type src0_type = op->src[0]->type;
                return (src0_type == GGML_TYPE_F32 || src0_type == GGML_TYPE_F16 || src0_type == GGML_TYPE_BF16) &&
                    op->type == src0_type &&
                    ggml_is_contiguous(op->src[0]) &&
                    ggml_is_contiguous(op);
            } break;
        case GGML_OP_SILU_BACK:
            return ggml_is_contiguous(op->src[0]) && op->src[0]->type == GGML_TYPE_F32;
            break;
        case GGML_OP_NORM:
        case GGML_OP_RMS_NORM:
        case GGML_OP_L2_NORM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_RMS_NORM_BACK:
            return ggml_is_contiguous(op->src[0]);
            break;
        case GGML_OP_NONE:
        case GGML_OP_RESHAPE:
        case GGML_OP_VIEW:
        case GGML_OP_PERMUTE:
        case GGML_OP_TRANSPOSE:
        case GGML_OP_ADD_ID:
        case GGML_OP_ADD1:
        case GGML_OP_SCALE:
        case GGML_OP_SQR:
        case GGML_OP_SQRT:
        case GGML_OP_SIN:
        case GGML_OP_COS:
        case GGML_OP_CLAMP:
        case GGML_OP_LOG:
            return true;
        case GGML_OP_ADD:
        case GGML_OP_SUB:
        case GGML_OP_MUL:
        case GGML_OP_DIV:
            return (op->src[0]->type == GGML_TYPE_F32 || op->src[0]->type == GGML_TYPE_F16) &&
                   (op->src[1]->type == GGML_TYPE_F32 || op->src[1]->type == GGML_TYPE_F16) &&
                   (op->type         == GGML_TYPE_F32 || op->type         == GGML_TYPE_F16);
        case GGML_OP_SSM_SCAN: {
            const int32_t K = ggml_get_op_params_i32(op, 0);

            if (op->src[3]->ne[0] == 1) {
                // Mamba2
                // (kernel only supports (d_state == 128 || d_state == 256) && d_head % 16 == 0)
                return (op->src[0]->ne[0] == 128 || op->src[0]->ne[0] == 256) && op->src[0]->ne[1] % 16 == 0;
            } else {
                if (K > 1) {
                    return false;
                }

                // Mamba
                // (kernel only supports d_state == 16, d_head == 1, n_head % 128 == 0, n_group == 1)
                return op->src[0]->ne[0] == 16 && op->src[0]->ne[1] == 1 && op->src[0]->ne[2] % 128 == 0 && op->src[4]->ne[1] == 1;
            }
        }
        case GGML_OP_SSM_CONV: {
            // assumes d_inner % threads == 0
            return op->src[0]->ne[1] % 128 == 0;
        }
        case GGML_OP_CONT:
            return true;
        case GGML_OP_DIAG_MASK_INF:
            return true;
        case GGML_OP_SOFT_MAX:
            return true;
        case GGML_OP_SOFT_MAX_BACK: {
            float max_bias = 0.0f;
            memcpy(&max_bias, (const float *) op->op_params + 1, sizeof(float));
            return max_bias == 0.0f;
        }
        case GGML_OP_ROLL:
            if(op->src[0]->type == GGML_TYPE_F32 && ggml_is_contiguous(op->src[0])) {
                return true;
            }
            return false;
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK: {
            return op->src[0]->nb[0] == ggml_type_size(op->src[0]->type) && ggml_is_contiguous_2(op->src[0]);
        }
        case GGML_OP_IM2COL:
        case GGML_OP_IM2COL_3D:
        case GGML_OP_CONV_2D:
            return (ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]));
        case GGML_OP_CONV_2D_DW:
            return op->src[0]->type == GGML_TYPE_F32;
        case GGML_OP_CONV_TRANSPOSE_2D:
        case GGML_OP_POOL_1D:
        case GGML_OP_POOL_2D:
            return true;
        case GGML_OP_ACC:
            // TODO: extend support like so:
            //return ggml_is_contiguous_rows(op->src[0]) && ggml_is_contiguous_rows(op->src[1]);
            return ggml_is_contiguous(op->src[0]) && ggml_is_contiguous(op->src[1]);
        case GGML_OP_SUM:
            return ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_TOP_K:
#if defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
            return true;
#else
            return op->src[0]->ne[0] <= 1024;
#endif // defined(GGML_USE_HIP) || defined(GGML_CUDA_USE_CUB)
        case GGML_OP_ARGSORT:
#ifndef GGML_CUDA_USE_CUB
            return op->src[0]->ne[0] <= 1024;
#else
            return true;
#endif
        case GGML_OP_SUM_ROWS:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_MEAN:
            return op->src[0]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32 && ggml_is_contiguous_rows(op->src[0]);
        case GGML_OP_GROUP_NORM:
            return ggml_is_contiguous(op->src[0]);
        case GGML_OP_PAD:
            return true;
        case GGML_OP_UPSCALE:
        case GGML_OP_PAD_REFLECT_1D:
        case GGML_OP_ARANGE:
        case GGML_OP_TIMESTEP_EMBEDDING:
        case GGML_OP_LEAKY_RELU:
        case GGML_OP_RWKV_WKV6:
        case GGML_OP_GATED_LINEAR_ATTN:
        case GGML_OP_RWKV_WKV7:
            return true;
        case GGML_OP_GATED_DELTA_NET:
            //TODO: enable once MUSA compiler is solved https://github.com/ggml-org/llama.cpp/pull/19504#issuecomment-4018634327
#ifdef GGML_USE_MUSA
            return false;
#else
            return true;
#endif // GGML_USE_MUSA
        case GGML_OP_DSV4_HC_COMB:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_PRE:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_DSV4_HC_POST:
            return op->src[0]->type == GGML_TYPE_F32 && op->src[1]->type == GGML_TYPE_F32 &&
                op->src[2]->type == GGML_TYPE_F32 && (op->src[3] == nullptr || op->src[3]->type == GGML_TYPE_F32) &&
                op->type == GGML_TYPE_F32;
        case GGML_OP_FLASH_ATTN_EXT:
            return ggml_cuda_flash_attn_ext_supported(dev_ctx->device, op);
        case GGML_OP_CROSS_ENTROPY_LOSS:
        case GGML_OP_CROSS_ENTROPY_LOSS_BACK:
        case GGML_OP_OPT_STEP_ADAMW:
        case GGML_OP_OPT_STEP_SGD:
        case GGML_OP_FILL:
        case GGML_OP_CUMSUM:
        case GGML_OP_TRI:
        case GGML_OP_DIAG:
        case GGML_OP_SOLVE_TRI:
            return true;
        case GGML_OP_LIGHTNING_INDEXER:
            return ggml_cuda_lightning_indexer_supported(dev_ctx->device, op);

        default:
            return false;
    }
}

static bool ggml_backend_cuda_device_supports_buft(ggml_backend_dev_t dev, ggml_backend_buffer_type_t buft) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;
    const bool integrated = ggml_cuda_info().devices[dev_ctx->device].integrated;
    return (ggml_backend_buft_is_cuda(buft) && buft->device == dev) || (integrated && ggml_backend_buft_is_cuda_host(buft)) ||
           ggml_backend_buft_is_cuda_mapped(buft);
}

static int64_t get_op_batch_size(const ggml_tensor * op) {
    switch (op->op) {
        case GGML_OP_GET_ROWS:
            return 0;
        case GGML_OP_MUL_MAT:
            return op->ne[1];
        case GGML_OP_MUL_MAT_ID:
        case GGML_OP_ROPE:
        case GGML_OP_ROPE_BACK:
            return op->ne[2];
        default:
            return ggml_nrows(op);
    }
}

static bool ggml_backend_cuda_device_offload_op(ggml_backend_dev_t dev, const ggml_tensor * op) {
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *) dev->context;

    return get_op_batch_size(op) >= dev_ctx->op_offload_min_batch_size;
}

static ggml_backend_event_t ggml_backend_cuda_device_event_new(ggml_backend_dev_t dev) {
#ifdef GGML_CUDA_NO_PEER_COPY
    GGML_UNUSED(dev);
    return nullptr;
#else
    ggml_backend_cuda_device_context * dev_ctx = (ggml_backend_cuda_device_context *)dev->context;

    ggml_cuda_set_device(dev_ctx->device);

    cudaEvent_t event;
    CUDA_CHECK(cudaEventCreateWithFlags(&event, cudaEventDisableTiming));

    return new ggml_backend_event {
        /* .device  = */ dev,
        /* .context = */ event,
    };
#endif
}

static void ggml_backend_cuda_device_event_free(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);

    CUDA_CHECK(cudaEventDestroy((cudaEvent_t)event->context));
    delete event;
}

static void ggml_backend_cuda_device_event_synchronize(ggml_backend_dev_t dev, ggml_backend_event_t event) {
    GGML_UNUSED(dev);
    CUDA_CHECK(cudaEventSynchronize((cudaEvent_t)event->context));
}

static const ggml_backend_device_i ggml_backend_cuda_device_interface = {
    /* .get_name                = */ ggml_backend_cuda_device_get_name,
    /* .get_description         = */ ggml_backend_cuda_device_get_description,
    /* .get_memory              = */ ggml_backend_cuda_device_get_memory,
    /* .get_type                = */ ggml_backend_cuda_device_get_type,
    /* .get_props               = */ ggml_backend_cuda_device_get_props,
    /* .init_backend            = */ ggml_backend_cuda_device_init_backend,
    /* .get_buffer_type         = */ ggml_backend_cuda_device_get_buffer_type,
    /* .get_host_buffer_type    = */ ggml_backend_cuda_device_get_host_buffer_type,
    /* .buffer_from_host_ptr    = */ NULL,
    /* .supports_op             = */ ggml_backend_cuda_device_supports_op,
    /* .supports_buft           = */ ggml_backend_cuda_device_supports_buft,
    /* .offload_op              = */ ggml_backend_cuda_device_offload_op,
    /* .event_new               = */ ggml_backend_cuda_device_event_new,
    /* .event_free              = */ ggml_backend_cuda_device_event_free,
    /* .event_synchronize       = */ ggml_backend_cuda_device_event_synchronize,
};

// backend reg

struct ggml_backend_cuda_reg_context {
    std::vector<ggml_backend_dev_t> devices;
};

static const char * ggml_backend_cuda_reg_get_name(ggml_backend_reg_t reg) {
    GGML_UNUSED(reg);
    return GGML_CUDA_NAME;
}

static size_t ggml_backend_cuda_reg_get_device_count(ggml_backend_reg_t reg) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    return ctx->devices.size();
}

static ggml_backend_dev_t ggml_backend_cuda_reg_get_device(ggml_backend_reg_t reg, size_t index) {
    ggml_backend_cuda_reg_context * ctx = (ggml_backend_cuda_reg_context *)reg->context;
    GGML_ASSERT(index < ctx->devices.size());
    return ctx->devices[index];
}

static ggml_backend_feature * ggml_backend_cuda_get_features(ggml_backend_reg_t reg) {
    static std::vector<ggml_backend_feature> features = []() {
        std::vector<ggml_backend_feature> features;
    #define _STRINGIFY(...) #__VA_ARGS__
    #define STRINGIFY(...) _STRINGIFY(__VA_ARGS__)

    #ifdef __CUDA_ARCH_LIST__
        features.push_back({ "ARCHS", STRINGIFY(__CUDA_ARCH_LIST__) });
    #endif

    #ifdef GGML_CUDA_FORCE_MMQ
        features.push_back({ "FORCE_MMQ", "1" });
    #endif

    #ifdef GGML_CUDA_FORCE_CUBLAS
        features.push_back({ "FORCE_CUBLAS", "1" });
    #endif

    #ifndef GGML_USE_VMM
        features.push_back({ "NO_VMM", "1" });
    #endif

    #ifdef GGML_CUDA_NO_PEER_COPY
        features.push_back({ "NO_PEER_COPY", "1" });
    #endif

    #ifdef GGML_CUDA_USE_GRAPHS
        features.push_back({ "USE_GRAPHS", "1" });
    #endif

    #ifdef GGML_CUDA_FA_QUANTS
        features.push_back({ "FA_QUANTS", GGML_CUDA_FA_QUANTS });
    #endif

    {
        const auto & info = ggml_cuda_info();
        for (int id = 0; id < info.device_count; ++id) {
            if (blackwell_mma_available(info.devices[id].cc)) {
                features.push_back({ "BLACKWELL_NATIVE_FP4", "1"});
                break;
            }
        }
    }

    #undef _STRINGIFY
    #undef STRINGIFY

        features.push_back({ nullptr, nullptr });

        return features;
    }();

    return features.data();

    GGML_UNUSED(reg);
}

// The decode loop on the device. The body (a ggml graph whose inputs live on
// the device) is captured and instantiated the way every decode is, by the
// backend's own graph compute; that captured graph then sits as a child node
// inside a CUDA-graph while node, followed by a one-thread kernel that keeps
// the loop going while the counter is below n_iter and the token is not the
// stop id. The host launches once and waits. Every call of the backend's
// compute on the way there is a real iteration. Returns 2 when the loop ran
// on the device, 1 when the host had to step it, 0 on failure.
#if defined(USE_CUDA_GRAPH) && !defined(GGML_USE_HIP) && !defined(GGML_USE_MUSA) && CUDART_VERSION >= 12030
#define GGML_CUDA_LOOP_ON_DEVICE
static __global__ void ggml_cuda_loop_cond_kernel(cudaGraphConditionalHandle handle, const float * counter, const float n_iter, const int32_t * tok, const int32_t stop) {
    const unsigned keep = (*counter < n_iter) && (*tok != stop);
    cudaGraphSetConditional(handle, keep);
}
#endif

static int ggml_backend_cuda_loop(ggml_backend_t backend, ggml_cgraph * cgraph, const ggml_tensor * counter, int n_iter, const ggml_tensor * tok, int32_t stop) {
#ifndef GGML_CUDA_LOOP_ON_DEVICE
    GGML_UNUSED(counter); GGML_UNUSED(tok); GGML_UNUSED(stop);
    for (int i = 0; i < n_iter; ++i) {
        if (ggml_backend_cuda_graph_compute(backend, cgraph) != GGML_STATUS_SUCCESS) {
            return 0;
        }
    }
    return 1;
#else
    ggml_backend_cuda_context * cuda_ctx = (ggml_backend_cuda_context *) backend->context;
    ggml_cuda_set_device(cuda_ctx->device);
    cudaStream_t stream = cuda_ctx->stream();

    const void * key = ggml_cuda_graph_get_key(cgraph, nullptr);

    // the first iterations run through the backend until it has the body captured
    int done = 0;
    ggml_cuda_graph * g = nullptr;
    while (done < n_iter) {
        if (ggml_backend_cuda_graph_compute(backend, cgraph) != GGML_STATUS_SUCCESS) {
            return 0;
        }
        done++;
        g = cuda_ctx->cuda_graph(key);
        if (g && g->instance != nullptr && g->graph != nullptr) {
            break;
        }
        if (done >= 4) {
            break;
        }
    }
    CUDA_CHECK(cudaStreamSynchronize(stream));

    auto finish_on_host = [&]() -> int {
        for (; done < n_iter; ++done) {
            if (ggml_backend_cuda_graph_compute(backend, cgraph) != GGML_STATUS_SUCCESS) {
                return 0;
            }
        }
        CUDA_CHECK(cudaStreamSynchronize(stream));
        return 1;
    };

    if (done >= n_iter) {
        return 2;
    }
    if (!g || g->instance == nullptr || g->graph == nullptr) {
        GGML_LOG_INFO("%s: the body did not capture, the host steps it\n", __func__);
        return finish_on_host();
    }

    // the stop test reads what the last iteration left; a stop already hit
    // would loop once more, which the caller trims
    cudaGraph_t parent = nullptr;
    CUDA_CHECK(cudaGraphCreate(&parent, 0));

    cudaGraphConditionalHandle handle;
    if (cudaGraphConditionalHandleCreate(&handle, parent, 1, cudaGraphCondAssignDefault) != cudaSuccess) {
        (void) cudaGetLastError();
        CUDA_CHECK(cudaGraphDestroy(parent));
        GGML_LOG_INFO("%s: conditional graph nodes are not available here, the host steps it\n", __func__);
        return finish_on_host();
    }

    cudaGraphNodeParams np = {};
    np.type = cudaGraphNodeTypeConditional;
    np.conditional.handle = handle;
    np.conditional.type   = cudaGraphCondTypeWhile;
    np.conditional.size   = 1;
    cudaGraphNode_t cnode = nullptr;
    CUDA_CHECK(cudaGraphAddNode(&cnode, parent, nullptr, nullptr, 0, &np));
    cudaGraph_t body = np.conditional.phGraph_out[0];

    cudaGraphNode_t child = nullptr;
    CUDA_CHECK(cudaGraphAddChildGraphNode(&child, body, nullptr, 0, g->graph));

    const float   * counter_ptr = (const float *)   counter->data;
    const int32_t * tok_ptr     = (const int32_t *) tok->data;
    const float     n_iter_f    = (float) n_iter;
    const int32_t   stop_v      = stop;
    void * args[] = { (void *) &handle, (void *) &counter_ptr, (void *) &n_iter_f, (void *) &tok_ptr, (void *) &stop_v };

    cudaKernelNodeParams kp = {};
    kp.func           = (void *) ggml_cuda_loop_cond_kernel;
    kp.gridDim        = dim3(1, 1, 1);
    kp.blockDim       = dim3(1, 1, 1);
    kp.sharedMemBytes = 0;
    kp.kernelParams   = args;
    kp.extra          = nullptr;
    cudaGraphNode_t knode = nullptr;
    CUDA_CHECK(cudaGraphAddKernelNode(&knode, body, &child, 1, &kp));

    cudaGraphExec_t exec = nullptr;
    CUDA_CHECK(cudaGraphInstantiate(&exec, parent, 0));
    const int64_t t_launch = ggml_time_us();
    CUDA_CHECK(cudaGraphLaunch(exec, stream));
    CUDA_CHECK(cudaStreamSynchronize(stream));
    if (getenv("GGML_CUDA_LOOP_TIMING") != nullptr) {
        const double ms = (ggml_time_us() - t_launch) / 1000.0;
        GGML_LOG_INFO("%s: device loop: %d iterations in %.1f ms, %.3f ms each\n", __func__, n_iter - done, ms, ms / std::max(1, n_iter - done));
    }
    CUDA_CHECK(cudaGraphExecDestroy(exec));
    CUDA_CHECK(cudaGraphDestroy(parent));
    return 2;
#endif
}

// an NVFP4 weight's static input scale, from the checkpoint (see ggml_cuda_fp4_set_input_scale); s nullptr forgets it
static void ggml_backend_cuda_set_input_scale(const ggml_tensor * w, const ggml_tensor * s) {
    ggml_cuda_fp4_set_input_scale(w->data, s ? (const float *) s->data : nullptr);
}

static void * ggml_backend_cuda_reg_get_proc_address(ggml_backend_reg_t reg, const char * name) {
    GGML_UNUSED(reg);
    if (strcmp(name, "ggml_backend_cuda_loop") == 0) {
        return (void *)ggml_backend_cuda_loop;
    }
    if (strcmp(name, "ggml_backend_cuda_set_input_scale") == 0) {
        return (void *) ggml_backend_cuda_set_input_scale;
    }
    if (strcmp(name, "ggml_backend_cuda_order_after") == 0) {
        return (void *) ggml_backend_cuda_order_after;
    }
    if (strcmp(name, "ggml_backend_cuda_copy_2d_async") == 0) {
        return (void *) ggml_backend_cuda_copy_2d_async;
    }
    if (strcmp(name, "ggml_backend_cuda_get_tensor_side_async") == 0) {
        return (void *) ggml_backend_cuda_get_tensor_side_async;
    }
    if (strcmp(name, "ggml_backend_cuda_get_tensors_side_async") == 0) {
        return (void *) ggml_backend_cuda_get_tensors_side_async;
    }
    if (strcmp(name, "ggml_backend_cuda_memcpy_batch_async") == 0) {
        return (void *) ggml_backend_cuda_memcpy_batch_async;
    }
    if (strcmp(name, "ggml_backend_comm_init") == 0) {
        return (void *)ggml_backend_cuda_comm_init;
    }
    if (strcmp(name, "ggml_backend_comm_free") == 0) {
        return (void *)ggml_backend_cuda_comm_free;
    }
    if (strcmp(name, "ggml_backend_comm_allreduce_tensor") == 0) {
        return (void *)ggml_backend_cuda_comm_allreduce_tensor;
    }
    if (strcmp(name, "ggml_backend_dev_get_mapped_buffer_type") == 0) {
        return (void *)ggml_backend_cuda_dev_get_mapped_buffer_type;
    }
    if (strcmp(name, "ggml_backend_register_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_register_host_buffer;
    }
    if (strcmp(name, "ggml_backend_unregister_host_buffer") == 0) {
        return (void *)ggml_backend_cuda_unregister_host_buffer;
    }
    if (strcmp(name, "ggml_backend_get_features") == 0) {
        return (void *)ggml_backend_cuda_get_features;
    }
    return nullptr;
}

static const ggml_backend_reg_i ggml_backend_cuda_reg_interface = {
    /* .get_name          = */ ggml_backend_cuda_reg_get_name,
    /* .get_device_count  = */ ggml_backend_cuda_reg_get_device_count,
    /* .get_device        = */ ggml_backend_cuda_reg_get_device,
    /* .get_proc_address  = */ ggml_backend_cuda_reg_get_proc_address,
};

// backend registry
ggml_backend_reg_t ggml_backend_cuda_reg() {
    static ggml_backend_reg reg;
    static bool initialized = false;

    {
        static std::mutex mutex;
        std::lock_guard<std::mutex> lock(mutex);
        if (!initialized) {
            ggml_backend_cuda_reg_context * ctx = new ggml_backend_cuda_reg_context;
            const int min_batch_size = getenv("GGML_OP_OFFLOAD_MIN_BATCH") ? atoi(getenv("GGML_OP_OFFLOAD_MIN_BATCH")) : 32;

            const ggml_cuda_device_info & info = ggml_cuda_info();
            const bool virtual_devices = info.device_count > info.physical_device_count;

            for (int i = 0; i < info.device_count; i++) {
                const int physical_id = info.devices[i].physical_device;

                ggml_backend_cuda_device_context * dev_ctx = new ggml_backend_cuda_device_context;
                dev_ctx->device = i;
                dev_ctx->name = GGML_CUDA_NAME + std::to_string(i);
                dev_ctx->description = ggml_cuda_device_description(i);

                char pci_bus_id[32] = {};
                CUDA_CHECK(cudaDeviceGetPCIBusId(pci_bus_id, sizeof(pci_bus_id), physical_id));
                dev_ctx->pci_bus_id = pci_bus_id;
                if (virtual_devices) {
                    // make the pci bus id unique for virtual devices
                    dev_ctx->pci_bus_id += "-v" + std::to_string(i);
                }
                for (char & c : dev_ctx->pci_bus_id) {
                    c = std::tolower(c);
                }
                dev_ctx->op_offload_min_batch_size = min_batch_size;

                ggml_backend_dev_t dev = new ggml_backend_device {
                    /* .iface   = */ ggml_backend_cuda_device_interface,
                    /* .reg     = */ &reg,
                    /* .context = */ dev_ctx
                };
                ctx->devices.push_back(dev);
            }

            reg = ggml_backend_reg {
                /* .api_version = */ GGML_BACKEND_API_VERSION,
                /* .iface       = */ ggml_backend_cuda_reg_interface,
                /* .context     = */ ctx
            };
        }

        initialized = true;
    }

    return &reg;
}

ggml_backend_t ggml_backend_cuda_init(int device) {
    if (device < 0 || device >= ggml_backend_cuda_get_device_count()) {
        GGML_LOG_ERROR("%s: invalid device %d\n", __func__, device);
        return nullptr;
    }

    ggml_backend_cuda_context * ctx = new ggml_backend_cuda_context(device);
    if (ctx == nullptr) {
        GGML_LOG_ERROR("%s: failed to allocate context\n", __func__);
        return nullptr;
    }

    ggml_backend_t cuda_backend = new ggml_backend {
        /* .guid    = */ ggml_backend_cuda_guid(),
        /* .iface   = */ ggml_backend_cuda_interface,
        /* .device  = */ ggml_backend_reg_dev_get(ggml_backend_cuda_reg(), device),
        /* .context = */ ctx,
    };

    return cuda_backend;
}

GGML_BACKEND_DL_IMPL(ggml_backend_cuda_reg)

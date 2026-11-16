#include <hip/hip_runtime.h>
#include <stdint.h>

extern "C" __global__ void bw_read(const uint4* p, uint64_t n, uint32_t* out) {
    uint32_t acc = 0;
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x) {
        const uint4 v = p[i];
        acc ^= v.x ^ v.y ^ v.z ^ v.w;
    }
    if (acc == 0x9e3779b9u) out[0] = acc;
}

extern "C" __global__ void bw_copy(uint4* dst, const uint4* src, uint64_t n) {
    for (uint64_t i = blockIdx.x * (uint64_t)blockDim.x + threadIdx.x; i < n; i += (uint64_t)gridDim.x * blockDim.x)
        dst[i] = src[i];
}

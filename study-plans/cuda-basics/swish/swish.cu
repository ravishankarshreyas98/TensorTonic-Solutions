#include <cuda_runtime.h>
#include <math.h>

__global__ void swish_kernel(const float* input, float* output, int N) {
    // Write the float32 results into output; this kernel returns no value.
    int idx = blockDim.x*blockIdx.x + threadIdx.x;
    for (int i= idx; i < N; i+= gridDim.x*blockDim.x) {
        output[i] = input[i]/(1+ expf(-input[i]));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    swish_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}

#include <cuda_runtime.h>
#include <math.h>

__global__ void sigmoid_kernel(const float* input, float* output, int N) {
    // Store N element-wise results in output; this kernel returns no value.
    int idx = blockIdx.x*blockDim.x + threadIdx.x;
    for (int i = idx ; i < N ; i+= gridDim.x*blockDim.x) {
        //expf() and __expf() both are valid.. but the latter gives lesser number of overall instructions 
        //since we end up using the SFU. The disassembly shows the number of instr reduce from 25-->11 for the statement below. 
        output[i] = 1.0/(1.0+ __expf(-input[i]));
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    sigmoid_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}

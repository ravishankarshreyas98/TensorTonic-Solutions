#include <cuda_runtime.h>
#include <algorithm>

/*
    Nvidia T4 has 320GB/s HBM BW , 8.1 TFLOPS of single fp compute so ridge point is around 25.3
    For RELU - we load 8 bytes for 1 fmax op .. so we have AI
    as 0.125. which is memory bound 
    So what we need to do is 
    increase occupany (latency hiding for warps for hbm loads)
    coalesced requests

    clever compute tricks won't cause much difference, though it is cleaner to use fmax instead of using ternary operator which will cause branching 
    
*/
__global__ void relu_kernel(const float* input, float* output, int N) {
    // Store N element-wise results in output; this kernel returns no value.
    //This solution also avoids warp divergence
    int idx= blockDim.x*blockIdx.x + threadIdx.x;
    for(int i= idx; i < N; i+= blockDim.x*gridDim.x){
        //c++ std::max doesn't seem to work need to either
        //use fmaxf (float) or fmax (double)
        output[i] = fmaxf(input[i],0.0f);
    }
}

extern "C" void solve(const float* input, float* output, int N) {
    int threads = 256;
    int blocks = (N + threads - 1) / threads;
    relu_kernel<<<blocks, threads>>>(input, output, N);
    cudaDeviceSynchronize();
}

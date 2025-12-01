/**
 * Simple GPUFF-RCAPv3 Test Build
 * Minimal main file to test CUDA compilation
 */

#include <cuda_runtime.h>
#include <iostream>
#include <cstdlib>

// Simple CUDA kernel for testing
__global__ void testKernel(float* data, int n) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx < n) {
        data[idx] = data[idx] * 2.0f;
    }
}

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "GPUFF-RCAPv3 - Simple Build Test" << std::endl;
    std::cout << "========================================\n" << std::endl;

    // Check CUDA availability
    int deviceCount = 0;
    cudaError_t error = cudaGetDeviceCount(&deviceCount);

    if (error != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    if (deviceCount == 0) {
        std::cerr << "No CUDA-capable GPU found!" << std::endl;
        return 1;
    }

    std::cout << "Found " << deviceCount << " CUDA device(s)" << std::endl;

    // Get device properties
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, 0);
    std::cout << "Using GPU: " << prop.name << std::endl;
    std::cout << "Compute Capability: " << prop.major << "." << prop.minor << std::endl;
    std::cout << "Total Global Memory: " << prop.totalGlobalMem / (1024*1024) << " MB" << std::endl;

    // Test simple CUDA operation
    const int N = 1024;
    float *h_data = new float[N];
    float *d_data;

    // Initialize host data
    for (int i = 0; i < N; i++) {
        h_data[i] = static_cast<float>(i);
    }

    // Allocate device memory
    cudaMalloc((void**)&d_data, N * sizeof(float));

    // Copy data to device
    cudaMemcpy(d_data, h_data, N * sizeof(float), cudaMemcpyHostToDevice);

    // Launch kernel
    int blockSize = 256;
    int numBlocks = (N + blockSize - 1) / blockSize;
    testKernel<<<numBlocks, blockSize>>>(d_data, N);

    // Check for kernel launch errors
    error = cudaGetLastError();
    if (error != cudaSuccess) {
        std::cerr << "Kernel launch error: " << cudaGetErrorString(error) << std::endl;
        return 1;
    }

    // Copy result back to host
    cudaMemcpy(h_data, d_data, N * sizeof(float), cudaMemcpyDeviceToHost);

    // Verify results
    bool success = true;
    for (int i = 0; i < N; i++) {
        if (h_data[i] != i * 2.0f) {
            success = false;
            break;
        }
    }

    if (success) {
        std::cout << "\n[SUCCESS] CUDA kernel test passed!" << std::endl;
        std::cout << "Build and execution successful!" << std::endl;
    } else {
        std::cout << "\n[FAILED] CUDA kernel test failed!" << std::endl;
    }

    // Cleanup
    delete[] h_data;
    cudaFree(d_data);

    std::cout << "\n========================================" << std::endl;
    std::cout << "Test complete." << std::endl;
    std::cout << "========================================" << std::endl;

    return 0;
}
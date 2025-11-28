// Test program to verify include chain
#include <stdio.h>
#include <cuda_runtime.h>

// Include the main kernel file
#include "gpuff.cuh"

int main() {
    printf("Include chain test successful!\n");

    // Test that we can access functions from split files
    float test_sigma_h = Sigma_h_Pasquill_Gifford_cpu(3, 1000.0f);
    printf("Test Sigma_h calculation: %f\n", test_sigma_h);

    float test_sigma_z = Sigma_z_Pasquill_Gifford_cpu(3, 1000.0f);
    printf("Test Sigma_z calculation: %f\n", test_sigma_z);

    return 0;
}
// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel Implementation - Cloudshine Dose Calculations
// ====================================================================================
//
// File: gpuff_kernels_cloudshine.cuh
// Purpose: Cloudshine external dose calculation kernels
//
// This file contains CUDA kernels for:
//   - Cloudshine (gamma radiation from airborne radioactive material)
//   - External dose from radioactive plume
//   - 3D geometric factor calculations
//   - Protection factor applications for external exposure
//
// ====================================================================================

#ifndef GPUFF_KERNELS_CLOUDSHINE_CUH
#define GPUFF_KERNELS_CLOUDSHINE_CUH

#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <math.h>
#include "gpuff_struct.cuh"

// Device constants are defined in gpuff.cuh and accessible through the include chain
// No extern declarations needed here as they're already declared in the compilation unit

// ====================================================================================
// Cloudshine Dose Calculation Kernel
// ====================================================================================

/**
 * Compute cloudshine dose from radioactive plume
 *
 * This kernel calculates the external dose from cloudshine (gamma radiation from
 * airborne radioactive material). Each evacuee can receive cloudshine dose from
 * multiple puffs, and different weather scenarios (simIdx) are processed independently.
 *
 * Key features:
 * - No interaction between different weather scenarios (simIdx)
 * - One evacuee can be affected by multiple puffs
 * - Uses shared memory reduction for efficient summation
 * - Applies protection factors for shielding
 *
 * @param d_puffs_RCAP Puff centers containing radioactive concentrations
 * @param d_evacuees Evacuee array with position and dose tracking
 * @param d_exposure Exposure coefficients for each nuclide/organ
 * @param dPF Protection factors for different shelter types
 */
// __global__ void ComputeCloudshineDose(
//     Gpuff::Puffcenter_RCAP* __restrict__ d_puffs_RCAP,
//     Evacuee* __restrict__ d_evacuees,
//     const float* __restrict__ d_exposure,
//     const ProtectionFactors* __restrict__ dPF
// ) {
//     // Shared memory for reduction operations
//     extern __shared__ float sdata_cloudshine[];

//     // Grid organization:
//     // blockIdx.y: weather scenario index (simIdx)
//     // blockIdx.x: evacuee index within scenario
//     // threadIdx.x: puff index
//     int simIdx = blockIdx.y;
//     int evacueeIdx = blockIdx.x;
//     int puffIdx = threadIdx.x;

//     // TODO: Define cloudshine-specific parameters
//     const float CLOUDSHINE_HEIGHT_FACTOR = 1.0f;  // Placeholder for height correction
//     const float GAMMA_ATTENUATION_COEFF = 1.0f;   // Placeholder for air attenuation

//     // Check bounds to ensure we're within valid indices
//     if (simIdx < d_numSims && evacueeIdx < d_totalevacuees_per_Sim && puffIdx < d_totalpuff_per_Sim) {

//         // Initialize shared memory for this thread
//         sdata_cloudshine[threadIdx.x] = 0.0f;

//         // Load puff and evacuee data for this thread
//         Gpuff::Puffcenter_RCAP puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
//         Evacuee evacuee = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

//         // TODO: Calculate distance between evacuee and puff center
//         // Convert polar coordinates to Cartesian if needed
//         float cosTheta = cosf(evacuee.theta);
//         float sinTheta = sinf(evacuee.theta);
//         float evac_x = evacuee.r * cosTheta;
//         float evac_y = evacuee.r * sinTheta;

//         float dx = evac_x - puff.x;
//         float dy = evac_y - puff.y;
//         float dz = puff.z;  // Height of puff center above ground

//         // Calculate 3D distance for cloudshine
//         float distance_3d = sqrtf(dx * dx + dy * dy + dz * dz);

//         // TODO: Determine protection factor index based on evacuee status
//         int pfidx = 0;
//         if (evacuee.flag == 0) pfidx = 1;      // Sheltered
//         else if (evacuee.flag == 1) pfidx = 2; // Evacuating
//         else if (evacuee.flag == 2) pfidx = 0; // Normal

//         // Initialize dose accumulator for this puff
//         float puffCloudshineDose[MAX_ORGANS] = {0.0f};

//         // TODO: Check if puff is active and within influence range
//         if (puff.flag == 1 && distance_3d > 0.0f) {

//             // TODO: Calculate geometric factor for cloudshine
//             // This is different from inhalation - uses 3D point source approximation
//             float geometricFactor = 1.0f / (4.0f * PI * distance_3d * distance_3d);

//             // TODO: Calculate plume dispersion factor
//             // Simplified Gaussian plume model for cloudshine
//             float sigma_h = puff.sigma_h;
//             float sigma_z = puff.sigma_z;

//             // TODO: Loop through all nuclides
//             for (int nuclideIdx = 0; nuclideIdx < MAX_NUCLIDES; ++nuclideIdx) {
//                 float puffConc = puff.conc[nuclideIdx];

//                 if (puffConc > 0.0f) {
//                     // TODO: Loop through all organs for dose calculation
//                     #pragma unroll
//                     for (int organIdx = 0; organIdx < MAX_ORGANS; ++organIdx) {

//                         // TODO: Extract cloudshine coefficient from exposure data
//                         // Field 0: cloudshine coefficient
//                         float cloudshineCoeff = d_exposure[nuclideIdx * MAX_ORGANS * DATA_FIELDS +
//                                                           organIdx * DATA_FIELDS + 0];

//                         if (cloudshineCoeff > 0.0f) {
//                             // TODO: Calculate cloudshine dose
//                             // Dose = coefficient * concentration * time * geometric factor
//                             float dose = cloudshineCoeff * puffConc * d_dt * geometricFactor;

//                             // TODO: Apply protection factor for cloudshine (external exposure)
//                             if (dPF->pfactor[pfidx][0] > 0.0f) {
//                                 dose *= dPF->pfactor[pfidx][0];  // Cloudshine protection factor
//                             }

//                             // TODO: Apply air attenuation if needed
//                             // dose *= exp(-GAMMA_ATTENUATION_COEFF * distance_3d);

//                             puffCloudshineDose[organIdx] += dose;
//                         }
//                     }
//                 }
//             }
//         }

//         // TODO: Perform reduction across all puffs for each organ
//         for (int organIdx = 0; organIdx < MAX_ORGANS; ++organIdx) {

//             // Store this thread's contribution to shared memory
//             sdata_cloudshine[threadIdx.x] = puffCloudshineDose[organIdx];

//             __syncthreads();

//             // TODO: Parallel reduction to sum contributions from all puffs
//             for (unsigned int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
//                 if (threadIdx.x < stride) {
//                     sdata_cloudshine[threadIdx.x] += sdata_cloudshine[threadIdx.x + stride];
//                 }
//                 __syncthreads();
//             }

//             // TODO: Thread 0 writes the final sum to evacuee's dose
//             if (threadIdx.x == 0) {
//                 // Add to evacuee's cumulative cloudshine dose
//                 d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx].dose_cloudshines[organIdx] +=
//                     sdata_cloudshine[0];
//             }

//             __syncthreads();
//         }
//     }
// }
// Cloudshine dose kernel for RCAP
// 모델 근거: RASCAL 4 NUREG-1940 및 Supplement 1의 cloudshine 계산식


// 상수
__constant__ float kMu_air = 0.01f;   // 공기 선감쇠계수 [m^-1] 0.7 MeV 근사
__constant__ float kK_build = 1.4f;   // 빌드업 보정 계수 k
__constant__ float kInv720  = 1.0f / 720.0f;

// 포인트커널 테이블 메타 포함
struct DoseTables {
    // dp_point: [nuclide][Np] 행우선 플랫 배열, 단위는 (dose rate per unit activity)로 정렬
    const float* __restrict__ dp_point;
    // p 그리드
    const float* __restrict__ p_grid;   // 길이 Np, [m]
    int Np;                             // p 그리드 개수
    // 반무한 구름용 DCF_sic [nuclide], 단위는 (rem s^-1) per (Ci m^-3) 또는 문서 정의에 부합
    const float* __restrict__ df_sic;
};

// 720 대표점 좌표 오프셋
struct Puff720 {
    float3 pos_local[720]; // 퍼프 중심 기준
};

// 기존 구조체 정의를 사용 (gpuff_struct.cuh에 정의됨)
// Gpuff::Puffcenter_RCAP, ProtectionFactors, Evacuee는 이미 정의되어 있음

// 평면원천 10개 슬랩 높이 생성
__device__ inline void plane_slab_heights(float he, float sigma_z, float H_mix, float z_out[10]) {
    // 5개 분위수 계수 c, 각각 +, - 적용하여 총 10개
    const float c[5] = {0.127f, 0.385f, 0.674f, 1.037f, 1.645f};
    int idx = 0;
    #pragma unroll
    for (int s = 0; s < 5; ++s) {
        z_out[idx++] = he + c[s] * sigma_z;
        z_out[idx++] = he - c[s] * sigma_z;
    }
    // 지표면과 혼합층 상단 반사 처리
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        if (z_out[i] < 0.0f) z_out[i] = -z_out[i];
        if (H_mix > 0.0f && z_out[i] > H_mix) z_out[i] = 2.0f * H_mix - z_out[i];
    }
}

// 1차 선형보간 유틸
__device__ inline float lerp(float a, float b, float t){ return a + t * (b - a); }

// 포인트커널 보간기
// dp_point: [nuclide][Np], p_grid: 길이 Np
__device__ inline float lookup_point_kernel(const DoseTables& tbl, int nuclide, float p_meters) {
    // 범위 밖 클램프
    if (p_meters <= tbl.p_grid[0]) {
        return tbl.dp_point[nuclide * tbl.Np + 0];
    }
    if (p_meters >= tbl.p_grid[tbl.Np - 1]) {
        return tbl.dp_point[nuclide * tbl.Np + (tbl.Np - 1)];
    }
    // 이분 탐색
    int lo = 0, hi = tbl.Np - 1;
    while (hi - lo > 1) {
        int mid = (lo + hi) >> 1;
        if (tbl.p_grid[mid] > p_meters) hi = mid; else lo = mid;
    }
    float p0 = tbl.p_grid[lo];
    float p1 = tbl.p_grid[hi];
    float t  = (p_meters - p0) / fmaxf(p1 - p0, 1e-12f);
    const float* row = tbl.dp_point + nuclide * tbl.Np;
    return lerp(row[lo], row[hi], t);
}

// 소형 퍼프 포인트커널 720점 평균
__device__ inline float small_puff_point_kernel_avg(const Puff720& geom,
                                                    const float3& puffCenter,
                                                    const float3& receptor,
                                                    const DoseTables& tbl,
                                                    int nuclideIdx,
                                                    float sigma_y,
                                                    float sigma_z) {
    float sum = 0.0f;
    #pragma unroll 4
    for (int m = 0; m < 720; ++m) {
        // 정규화된 좌표를 실제 퍼프 크기로 스케일링
        float3 src = make_float3(puffCenter.x + geom.pos_local[m].x * sigma_y,
                                 puffCenter.y + geom.pos_local[m].y * sigma_y,
                                 puffCenter.z + geom.pos_local[m].z * sigma_z);
        float dx = receptor.x - src.x;
        float dy = receptor.y - src.y;
        float dz = receptor.z - src.z;
        float p  = sqrtf(dx*dx + dy*dy + dz*dz) + 1e-6f;
        sum += lookup_point_kernel(tbl, nuclideIdx, p);
    }
    return kInv720 * sum; // 대표점 평균
}

// 평면원천 10 슬랩 합산
// DCF_pn = DCF_sic / 241.2  적용
// lateral = exp(-0.5*(r/σ_y)^2) / (2π σ_y^2)
// 빌드업 합: Σ_i [ (1 + k μ z_i) exp(-μ z_i) ]
__device__ inline float plane_source_10slabs_sum(float he, float sigma_y, float sigma_z, float H_mix,
                                                 float r, const DoseTables& tbl, int nuclideIdx) {
    float z_s[10];
    plane_slab_heights(he, sigma_z, H_mix, z_s);

    const float DCF_pn = tbl.df_sic[nuclideIdx] / 241.2f;
    const float lateral = expf(-0.5f * (r / sigma_y) * (r / sigma_y)) / (2.0f * PI * sigma_y * sigma_y);

    float sum_build = 0.0f;
    #pragma unroll
    for (int i = 0; i < 10; ++i) {
        float z = fmaxf(z_s[i], 0.0f);
        sum_build += (1.0f + kK_build * kMu_air * z) * expf(-kMu_air * z);
    }
    // 슬랩별 활동분율 1/10
    return 0.1f * DCF_pn * lateral * sum_build;
}

// 반무한 구름 합산
// 균일혼합 근사: χ/Q = exp(-0.5*(r/σ_y)^2) / (2π σ_y^2 H)
// dose rate = Q_n * (χ/Q) * DCF_sic
__device__ inline float semi_infinite_dose_sum(float r, float sigma_y, float H_mix,
                                               const DoseTables& tbl, int nuclideIdx, float Qn) {
    float chi_over_Q = expf(-0.5f * (r / sigma_y) * (r / sigma_y)) /
                       (2.0f * PI * sigma_y * sigma_y * fmaxf(H_mix, 1e-6f));
    return Qn * chi_over_Q * tbl.df_sic[nuclideIdx];
}

__global__ void ComputeCloudshineDose(
    Gpuff::Puffcenter_RCAP* __restrict__ d_puffs_RCAP,
    Evacuee* __restrict__ d_evacuees,
    const ProtectionFactors* __restrict__ dPF,
    const DoseTables* __restrict__ tbl,
    const Puff720* __restrict__ geom720,
    int Nnucl,
    float build_height,   // 유효 방출고도
    float mix_height      // 혼합층 높이 H
) {
    extern __shared__ float sdata_cloudshine[]; // 크기: blockDim.x

    int simIdx     = blockIdx.y;
    int evacueeIdx = blockIdx.x;
    int puffIdx    = threadIdx.x;

    if (simIdx >= d_numSims || evacueeIdx >= d_totalevacuees_per_Sim || puffIdx >= d_totalpuff_per_Sim) return;

    // 공유메모리 초기화
    sdata_cloudshine[threadIdx.x] = 0.0f;

    const Gpuff::Puffcenter_RCAP puff = d_puffs_RCAP[simIdx * d_totalpuff_per_Sim + puffIdx];
    Evacuee evac = d_evacuees[simIdx * d_totalevacuees_per_Sim + evacueeIdx];

    // 수용체 좌표
    float ex = evac.r * cosf(evac.theta);
    float ey = evac.r * sinf(evac.theta);
    float ez = 1.0f; // 지상 1 m
    float3 receptor   = make_float3(ex, ey, ez);
    float3 puffCenter = make_float3(puff.x, puff.y, puff.z);

    // PF 선택
    int pfidx = 0;
    if      (evac.flag == 0) pfidx = 1; // Sheltered
    else if (evac.flag == 1) pfidx = 2; // Evacuating
    else                     pfidx = 0; // Normal
    float pf = dPF->pfactor[pfidx][0];

    const float sigma_y = puff.sigma_h;
    const float sigma_z = puff.sigma_z;

    // RASCAL 분기 기준에 맞춘 근사적 스위칭
    const bool small_puff = (sigma_y < 400.0f && sigma_z < 400.0f);
    const bool plane_src  = (sigma_y >= 400.0f && sigma_z < 400.0f);
    const bool semi_inf   = (sigma_z >= 400.0f);

    float dose_cloudshine = 0.0f;
    int current_mode = -1;  // Track which mode is being used

    // Debug output for first few threads
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x < 3) {
        printf("[CLOUDSHINE DEBUG] Thread %d - Puff %d: flag=%d, sigma_y=%.2f, sigma_z=%.2f\n",
               threadIdx.x, puffIdx, puff.flag, sigma_y, sigma_z);

        // Print first 3 nuclide concentrations
        float total_conc = 0.0f;
        for (int n = 0; n < 3; n++) {
            if (puff.conc[n] > 0) {
                printf("  conc[%d]=%.3e", n, puff.conc[n]);
                total_conc += puff.conc[n];
            }
        }
        if (total_conc > 0) printf(" (total=%.3e)\n", total_conc);
        else printf(" (all zero)\n");

        if (threadIdx.x == 0) {
            printf("[CLOUDSHINE DEBUG] Evacuee %d: r=%.2f, theta=%.2f, evac_flag=%d\n",
                   evacueeIdx, evac.r, evac.theta, evac.flag);
            printf("[CLOUDSHINE DEBUG] Parameters: d_dt=%.2f, pf=%.3f\n", d_dt, pf);
        }
    }

    if (puff.flag == 1) {
        if (small_puff) {
            current_mode = 0;  // Small puff mode
            float sum_all = 0.0f;
            for (int n = 0; n < Nnucl; ++n) {
                const float Qn = puff.conc[n]; // 활동도 [Ci]로 해석
                if (Qn <= 0.0f) continue;
                float dprime_avg = small_puff_point_kernel_avg(*geom720, puffCenter, receptor, *tbl, n, sigma_y, sigma_z);
                sum_all += Qn * dprime_avg;
            }
            dose_cloudshine = sum_all;
        } else if (plane_src) {
            current_mode = 1;  // Plane source mode
            // 지상 투영 거리 r
            const float dxg = ex - puff.x;
            const float dyg = ey - puff.y;
            const float r    = sqrtf(dxg*dxg + dyg*dyg);

            float sum_all = 0.0f;
            for (int n = 0; n < Nnucl; ++n) {
                const float Qn = puff.conc[n];
                if (Qn <= 0.0f) continue;
                float dsum = plane_source_10slabs_sum(build_height, sigma_y, sigma_z, mix_height, r, *tbl, n);
                sum_all += Qn * dsum;
            }
            dose_cloudshine = sum_all;
        } else if (semi_inf) {
            current_mode = 2;  // Semi-infinite mode
            const float dxg = ex - puff.x;
            const float dyg = ey - puff.y;
            const float r    = sqrtf(dxg*dxg + dyg*dyg);

            float sum_all = 0.0f;
            for (int n = 0; n < Nnucl; ++n) {
                const float Qn = puff.conc[n];
                if (Qn <= 0.0f) continue;
                sum_all += semi_infinite_dose_sum(r, sigma_y, mix_height, *tbl, n, Qn);
            }
            dose_cloudshine = sum_all;
        } else {
            current_mode = 1;  // Default to plane source mode for boundary cases
            // 경계 부근 보수적으로 평면원천 처리
            const float dxg = ex - puff.x;
            const float dyg = ey - puff.y;
            const float r    = sqrtf(dxg*dxg + dyg*dyg);

            float sum_all = 0.0f;
            for (int n = 0; n < Nnucl; ++n) {
                const float Qn = puff.conc[n];
                if (Qn <= 0.0f) continue;
                float dsum = plane_source_10slabs_sum(build_height, sigma_y, sigma_z, mix_height, r, *tbl, n);
                sum_all += Qn * dsum;
            }
            dose_cloudshine = sum_all;
        }

        // 외부피폭 PF 적용
        // 주의: PF 정의가 차폐배수(10, 100 등)라면 1.0f/PF를 곱해야 함
        if (pf > 0.0f) dose_cloudshine *= pf;

        // Apply time interval to convert dose rate to dose
        dose_cloudshine *= d_dt;

        // Debug output when dose is calculated
        if (dose_cloudshine > 0.0f && threadIdx.x == 0) {
            printf("[CLOUDSHINE CALC] Puff %d -> Evacuee %d: dose=%.3e, mode=%d\n",
                   puffIdx, evacueeIdx, dose_cloudshine, current_mode);
        }
    }

    // Debug output for dose calculation
    if (blockIdx.x == 0 && blockIdx.y == 0 && threadIdx.x == 0) {
        printf("[CLOUDSHINE DEBUG] Thread 0: dose=%.3e, mode=%d, pf=%.3f, dt=%.3f\n",
               dose_cloudshine, current_mode, pf, d_dt);
    }

    // 스레드 로컬 합산
    sdata_cloudshine[threadIdx.x] = dose_cloudshine;

    // Store mode in shared memory (use the second half for mode tracking)
    int* sdata_mode = (int*)(&sdata_cloudshine[blockDim.x]);
    sdata_mode[threadIdx.x] = (dose_cloudshine > 0.0f) ? current_mode : -1;
    __syncthreads();

    // 블록 내 병렬 감소
    for (unsigned int stride = blockDim.x >> 1; stride > 0; stride >>= 1) {
        if (threadIdx.x < stride) {
            sdata_cloudshine[threadIdx.x] += sdata_cloudshine[threadIdx.x + stride];
            // Take the mode from the thread with non-zero dose
            if (sdata_cloudshine[threadIdx.x + stride] > 0.0f && sdata_mode[threadIdx.x + stride] >= 0) {
                sdata_mode[threadIdx.x] = sdata_mode[threadIdx.x + stride];
            }
        }
        __syncthreads();
    }

    if (threadIdx.x == 0) {
        int evac_idx = simIdx * d_totalevacuees_per_Sim + evacueeIdx;
        Evacuee& evacuee = d_evacuees[evac_idx];

        // Update cumulative cloudshine dose for all 3 organs (assuming same dose for simplicity)
        // This matches the original cloudshine calculation
        if (sdata_cloudshine[0] > 0.0f) {
            evacuee.dose_cloudshines[0] += sdata_cloudshine[0];
            evacuee.dose_cloudshines[1] += sdata_cloudshine[0];
            evacuee.dose_cloudshines[2] += sdata_cloudshine[0];

            // Update tracking fields
            evacuee.dose_cloudshine_cumulative += sdata_cloudshine[0];
            evacuee.dose_cloudshine_instant = sdata_cloudshine[0];
            evacuee.cloudshine_mode = sdata_mode[0];
        }

        // Debug output for final values
        if (blockIdx.x == 0 && blockIdx.y == 0) {
            printf("[CLOUDSHINE FINAL] Evacuee %d: cumulative=%.3e, instant=%.3e, mode=%d, dose[0]=%.3e\n",
                   evacueeIdx, evacuee.dose_cloudshine_cumulative, evacuee.dose_cloudshine_instant,
                   evacuee.cloudshine_mode, evacuee.dose_cloudshines[0]);
        }
    }
}


#endif // GPUFF_KERNELS_CLOUDSHINE_CUH
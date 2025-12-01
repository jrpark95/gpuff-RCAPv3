// ====================================================================================
// GPUFF-RCAPv3 CUDA Kernel - Dispersion Calculations
// ====================================================================================
//
// File: gpuff_kernels_dispersion.cuh
// Purpose: GPU kernels for atmospheric dispersion parameter updates
//
// This file contains:
//   - Dispersion parameter (sigma) calculations
//   - Stability class determination
//   - Virtual distance updates
//   - Pasquill-Gifford and Briggs-McElroy-Pooler implementations
//
// ====================================================================================

#ifndef GPUFF_KERNELS_DISPERSION_CUH
#define GPUFF_KERNELS_DISPERSION_CUH

#include "gpuff_kernels_constants.cuh"

// ====================================================================================
// CUDA Kernels - Standard Dispersion Updates
// ====================================================================================

/**
 * Update puff dispersion parameters based on atmospheric stability
 *
 * Calculates atmospheric stability class from temperature gradient and
 * updates sigma_h and sigma_z using Pasquill-Gifford or Briggs-McElroy-Pooler formulas
 *
 * Thread organization: 1D grid, one thread per puff
 * Memory access: Irregular pattern due to meteorological field interpolation
 *
 * @param d_puffs Array of puff center data
 * @param device_meteorological_data_pres Pressure-level meteorological data
 * @param device_meteorological_data_unis Surface-level meteorological data
 * @param device_meteorological_data_etas Eta-coordinate meteorological data
 */
__global__ void puff_dispersion_update(
    Gpuff::Puffcenter* d_puffs,
    PresData* device_meteorological_data_pres,
    UnisData* device_meteorological_data_unis,
    EtasData* device_meteorological_data_etas)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    int xidx = int(p.x/1500.0);
    int yidx = int(p.y/1500.0);
    int zidx_uv = 1;
    int zidx_w = 1;

    for(int i=0; i<dimZ_etas-1; i++){
        if(p.z<d_etas_hgt_uv[i]){
            zidx_uv=i+1;
            break;
        }
    }

    for(int i=0; i<dimZ_etas-1; i++){
        if(p.z<d_etas_hgt_w[i]){
            zidx_w=i+1;
            break;
        }
    }

    if(zidx_uv<0) {
        printf("Invalid zidx_uv error.\n");
        zidx_uv = 1;
    }

    if(zidx_w<0) {
        printf("Invalid zidx_w error.\n");
        zidx_w = 1;
    }

    float x0 = p.x/1500.0-xidx;
    float y0 = p.y/1500.0-yidx;

    float x1 = 1-x0;
    float y1 = 1-y0;

    float xwind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].UGRD
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].UGRD
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].UGRD
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].UGRD;

    float ywind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].VGRD
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_uv].VGRD
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].VGRD
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_uv].VGRD;

    float zwind = x1*y1*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_w].DZDT
                    +x0*y1*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + yidx*(dimZ_etas) + zidx_w].DZDT
                    +x1*y0*device_meteorological_data_etas[xidx*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_w].DZDT
                    +x0*y0*device_meteorological_data_etas[(xidx+1)*(dimY)*(dimZ_etas) + (yidx+1)*(dimZ_etas) + zidx_w].DZDT;


    float vel = sqrt(xwind*xwind + ywind*ywind + zwind*zwind);

    //printf("zwind: %f, vel: %f ", zwind, vel);

    float t0 = x1*y1*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].TMP
                +x0*y1*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].TMP
                +x1*y0*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].TMP
                +x0*y0*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].TMP;

    float tu = x1*y1*device_meteorological_data_unis[xidx*(dimY) + yidx].T1P5
                +x0*y1*device_meteorological_data_unis[(xidx+1)*(dimY) + yidx].T1P5
                +x1*y0*device_meteorological_data_unis[xidx*(dimY) + (yidx+1)].T1P5
                +x0*y0*device_meteorological_data_unis[(xidx+1)*(dimY) + (yidx+1)].T1P5;

    float gph0 = x1*y1*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].HGT
                +x0*y1*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + yidx*(dimZ_pres)].HGT
                +x1*y0*device_meteorological_data_pres[xidx*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].HGT
                +x0*y0*device_meteorological_data_pres[(xidx+1)*(dimY)*(dimZ_pres) + (yidx+1)*(dimZ_pres)].HGT;

    float dtp100m = 100.0*(t0-tu)/(gph0-1.5);

    int PasquillCategory = 0;

    if(dtp100m < -1.9) PasquillCategory = 0;        // A: Extremely unstable
    else if(dtp100m < -1.7) PasquillCategory = 1;   // B: Moderately unstable
    else if(dtp100m < -1.5) PasquillCategory = 2;   // C: Slightly unstable
    else if(dtp100m < -0.5) PasquillCategory = 3;   // D: Neutral
    else if(dtp100m < 1.5) PasquillCategory = 4;    // E: Slightly stable
    else if(dtp100m < 4.0) PasquillCategory = 5;    // F: Moderately stable
    else PasquillCategory = 6;                      // G: Extremely stable

    float new_virtual_distance_h = NewtonRaphson_h(PasquillCategory, p.sigma_h, p.virtual_distance) + vel*d_dt;
    float new_virtual_distance_z = NewtonRaphson_z(PasquillCategory, p.sigma_z, p.virtual_distance) + vel*d_dt;

    if(d_isPG){
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
        p.sigma_z = Sigma_z_Pasquill_Gifford(PasquillCategory, new_virtual_distance_z);
    }
    else{
        //p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_z);
    }

    p.virtual_distance = new_virtual_distance_h;

}

/**
 * Validation version of dispersion update kernel
 * Uses fixed wind field and stability class for testing
 *
 * @param d_puffs Array of puff center data
 */
__global__ void puff_dispersion_update_val(Gpuff::Puffcenter* d_puffs)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    float xwind = 1.0f;
    float ywind = 0.0f;
    float zwind = 0.0f;

    float vel = sqrt(xwind*xwind + ywind*ywind + zwind+zwind);
    int PasquillCategory = 5;

    float new_virtual_distance_h = NewtonRaphson_h(PasquillCategory, p.sigma_h, p.virtual_distance) + vel*d_dt;
    float new_virtual_distance_z = NewtonRaphson_z(PasquillCategory, p.sigma_z, p.virtual_distance) + vel*d_dt;

    if(d_isPG){
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
        p.sigma_z = Sigma_z_Pasquill_Gifford(PasquillCategory, new_virtual_distance_z);
    }
    else{
        //p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_z);
    }

    p.virtual_distance = new_virtual_distance_h;

}

// ====================================================================================
// CUDA Kernels - RCAP Dispersion Updates
// ====================================================================================

/**
 * Update dispersion parameters for RCAP simulation
 * Uses puff-specific stability class and wind data
 *
 * @param d_puffs Array of puff center data
 * @param d_RCAP_windir Wind direction array
 * @param d_RCAP_winvel Wind velocity array
 * @param d_radi Radial distance array
 */
__global__ void puff_dispersion_update_RCAP(
    Gpuff::Puffcenter* d_puffs,
    float* d_RCAP_windir,
    float* d_RCAP_winvel,
    float* d_radi)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if(idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if(!p.flag) return;

    // int tt = floor((float)p.timeidx/(float)d_nop*(float)d_time_end/3600.0);

    // // float xwind = d_RCAP_winvel[timeidx]*cos(d_RCAP_windir[timeidx]);
    // // float ywind = d_RCAP_winvel[timeidx]*sin(d_RCAP_windir[timeidx]);

    // float xwind = d_RCAP_winvel[tt]*cos(d_RCAP_windir[tt]);
    // float ywind = d_RCAP_winvel[tt]*sin(d_RCAP_windir[tt]);

    float xwind = p.windvel*cos(p.windir);
    float ywind = p.windvel*sin(p.windir);

    float vel = sqrt(xwind*xwind + ywind*ywind);



    int PasquillCategory = p.stab-1;

    float new_virtual_distance_h = NewtonRaphson_h(PasquillCategory, p.sigma_h, p.virtual_distance) + vel*d_dt;
    //float new_virtual_distance_z = NewtonRaphson_z(PasquillCategory, p.sigma_z, p.virtual_distance) + vel*d_dt;

    if(d_isPG){
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Pasquill_Gifford(PasquillCategory, new_virtual_distance_z);
    }
    else{
        //p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_z);
    }

    p.virtual_distance = new_virtual_distance_h;

}

/**
 * Alternative RCAP dispersion update kernel
 * Simplified version without wind array dependency
 *
 * @param d_puffs Array of puff center data
 * @param d_RCAP_windir Wind direction array (unused)
 * @param d_RCAP_winvel Wind velocity array (unused)
 */
__global__ void puff_dispersion_update_RCAP2(
    Gpuff::Puffcenter* d_puffs,
    float* d_RCAP_windir,
    float* d_RCAP_winvel)
{

    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= d_nop) return;

    Gpuff::Puffcenter& p = d_puffs[idx];
    if (!p.flag) return;

    float xwind = p.windvel * cos(p.windir);
    float ywind = p.windvel * sin(p.windir);

    float vel = sqrt(xwind * xwind + ywind * ywind);

    int PasquillCategory = p.stab - 1;

    float new_virtual_distance_h = NewtonRaphson_h(PasquillCategory, p.sigma_h, p.virtual_distance) + vel * d_dt;
    //float new_virtual_distance_z = NewtonRaphson_z(PasquillCategory, p.sigma_z, p.virtual_distance) + vel*d_dt;

    if (d_isPG) {
        p.sigma_h = Sigma_h_Pasquill_Gifford(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Pasquill_Gifford(PasquillCategory, new_virtual_distance_z);
    }
    else {
        //p.sigma_h = Sigma_h_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_h);
        //p.sigma_z = Sigma_z_Briggs_McElroy_Pooler(PasquillCategory, new_virtual_distance_z);
    }

    p.virtual_distance = new_virtual_distance_h;

}

#endif // GPUFF_KERNELS_DISPERSION_CUH
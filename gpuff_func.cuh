
#include "gpuff.cuh"

Gpuff::Gpuff() 
    : device_meteorological_data_pres(nullptr), 
      device_meteorological_data_unis(nullptr), 
      device_meteorological_data_etas(nullptr){}

Gpuff::~Gpuff()
{
    if (device_meteorological_data_pres){
        cudaFree(device_meteorological_data_pres);
    }
    if (device_meteorological_data_unis){
        cudaFree(device_meteorological_data_unis);
    }
    if (device_meteorological_data_etas){
        cudaFree(device_meteorological_data_etas);
    }
    if (d_puffs){
        cudaFree(d_puffs);
    }
}


// void Gpuff::print_puffs() const {
//     std::ofstream outfile("output.txt");

//     if (!outfile.is_open()){
//         std::cerr << "Failed to open output.txt for writing!" << std::endl;
//         return;
//     }

//     outfile << "puffs Info:" << std::endl;
//     for (const auto& p : puffs){
//         outfile << "---------------------------------\n";
//         outfile << "x: " << p.x << ", y: " << p.y << ", z: " << p.z << "\n";
//         outfile << "Decay Constant: " << p.decay_const << "\n";
//         outfile << "Concentration: " << p.conc << "\n";
//         outfile << "Time Index: " << p.timeidx << "\n";
//         outfile << "Flag: " << p.flag << "\n";
//     }

//     outfile.close();
// }

void Gpuff::clock_start(){
    _clock0 = std::chrono::high_resolution_clock::now();
}

void Gpuff::clock_end(){
    _clock1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(_clock1 - _clock0);
    std::cout << "Elapsed time: " << duration.count()/1.0e6 << " seconds" << std::endl;
}

//void update_evac_velocity(const EvacuationData& EP, float currentTime) {
//
//    if (currentSpeedIndex >= EP.nSpeedPeriod - 1) {
//        return;
//    }
//
//    if (currentTime >= currentSpeedEndTime) {
//        std::cout << "currentSpeedIndex : " << currentSpeedIndex << std::endl;
//        std::cout << "EP.nSpeedPeriod - 1 : " << EP.nSpeedPeriod - 1 << std::endl;
//        std::cout << "currentSpeedEndTime : " << currentSpeedEndTime << std::endl;
//
//
//        if (currentSpeedIndex < EP.nSpeedPeriod - 1) {
//            currentSpeedIndex++;
//            currentSpeedEndTime += EP.durations[currentSpeedIndex - 1];
//            std::cout << "speed : " << EP.speeds[currentSpeedIndex] << std::endl;
//        }
//
//        float speed = EP.speeds[currentSpeedIndex];
//
//        for (auto& evacuee : evacuees) {
//            evacuee.speed = speed;
//        }
//
//        cudaMemcpy(d_evacuees, evacuees.data(), evacuees.size() * sizeof(Evacuee), cudaMemcpyHostToDevice);
//    }
//}

void update_evac_velocity(const EvacuationData& EP, float currentTime) {

    if (currentSpeedIndex > EP.nSpeedPeriod) {
        return;
    }

    if (currentTime >= currentSpeedEndTime) {
        if (currentSpeedIndex < EP.nSpeedPeriod - 1) {

            float speed = EP.speeds[currentSpeedIndex];
            currentSpeedEndTime = EP.alarmTime + EP.durations[currentSpeedIndex];
            currentSpeedIndex++;

            for (auto& evacuee : evacuees) {
                if (!evacuee.flag) evacuee.speed = 0.0f;
                evacuee.speed = speed;
            }

            cudaMemcpy(d_evacuees, evacuees.data(), evacuees.size() * sizeof(Evacuee), cudaMemcpyHostToDevice);

            //std::cout << "Speed updated to: " << speed << std::endl;
        }
        else if (currentSpeedIndex == EP.nSpeedPeriod - 1) {

            float speed = EP.speeds[currentSpeedIndex];
            currentSpeedEndTime = time_end;
            currentSpeedIndex++;

            for (auto& evacuee : evacuees) {
                evacuee.speed = speed;
            }

            cudaMemcpy(d_evacuees, evacuees.data(), evacuees.size() * sizeof(Evacuee), cudaMemcpyHostToDevice);

            //std::cout << "Final speed updated to: " << speed << std::endl;
        }
    }
}

void update_evac_velocity_cpu(const EvacuationData& EP, float currentTime) {

    if (currentSpeedIndex > EP.nSpeedPeriod) {
        return;
    }

    if (currentTime >= currentSpeedEndTime) {
        if (currentSpeedIndex < EP.nSpeedPeriod - 1) {

            float speed = EP.speeds[currentSpeedIndex];
            currentSpeedEndTime = EP.alarmTime + EP.durations[currentSpeedIndex];
            currentSpeedIndex++;

            for (auto& evacuee : evacuees) {
                if (!evacuee.flag) evacuee.speed = 0.0f;
                evacuee.speed = speed;
            }

            //cudaMemcpy(d_evacuees, evacuees.data(), evacuees.size() * sizeof(Evacuee), cudaMemcpyHostToDevice);

            //std::cout << "Speed updated to: " << speed << std::endl;
        }
        else if (currentSpeedIndex == EP.nSpeedPeriod - 1) {

            float speed = EP.speeds[currentSpeedIndex];
            currentSpeedEndTime = time_end;
            currentSpeedIndex++;

            for (auto& evacuee : evacuees) {
                evacuee.speed = speed;
            }

            //cudaMemcpy(d_evacuees, evacuees.data(), evacuees.size() * sizeof(Evacuee), cudaMemcpyHostToDevice);

            //std::cout << "Final speed updated to: " << speed << std::endl;
        }
    }
}



void Gpuff::print_puffs(){

    std::ofstream outfile("output.txt");

    outfile << std::left << std::setw(12) << "X" 
           << std::setw(12) << "Y" 
           << std::setw(12) << "Z" 
           << std::setw(17) << "decay_const" 
           << std::setw(17) << "source_conc"
           << std::setw(10) << "timeidx"
           << std::setw(10) << "flag" 
           << std::endl;
    outfile << std::string(110, '-') << std::endl;

    for(const auto& p : puffs){
        outfile << std::left << std::fixed << std::setprecision(2)
                << std::setw(12) << p.x 
                << std::setw(12) << p.y 
                << std::setw(12) << p.z;

        outfile << std::scientific
                << std::setw(17) << p.decay_const 
                << std::setw(17) << p.conc
                << std::setw(10) << p.timeidx 
                << std::setw(10) << p.flag 
                << std::endl;
    }
    outfile.close();

}


void Gpuff::allocate_and_copy_to_device(){

    cudaError_t err = cudaMalloc((void**)&d_puffs, puffs.size() * sizeof(Puffcenter));

    if (err != cudaSuccess){
        std::cerr << "Failed to allocate device memory for puffs: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }

    err = cudaMemcpy(d_puffs, puffs.data(), puffs.size() * sizeof(Puffcenter), cudaMemcpyHostToDevice);
    if (err != cudaSuccess){
        std::cerr << "Failed to copy puffs from host to device: " << cudaGetErrorString(err) << std::endl;
        exit(EXIT_FAILURE);
    }
}

void Gpuff::print_device_puffs_timeidx(){

    const int threadsPerBlock = 256; 
    const int blocks = (puffs.size() + threadsPerBlock - 1) / threadsPerBlock;

    print_timeidx<<<blocks, threadsPerBlock>>>(d_puffs);

    cudaDeviceSynchronize();

}

void Gpuff::time_update(){

    cudaError_t err = cudaGetLastError();
    
    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (puffs.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;
    

    while(currentTime <= time_end){

        activationRatio = currentTime / time_end;

        update_puff_flags<<<blocks, threadsPerBlock>>>
            (d_puffs, activationRatio);
        cudaDeviceSynchronize();

        move_puffs_by_wind<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        dry_deposition<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        wet_scavenging<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        radioactive_decay<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        puff_dispersion_update<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("CUDA error: %s, timestep = %d\n", cudaGetErrorString(err), timestep);

        currentTime += dt;
        timestep++;

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", currentTime);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));

            //puff_output_ASCII(timestep);
            puff_output_binary(timestep);
        }

    }

    // printf("-------------------------------------------------\n");
    // printf("Time : %f\n\tsec", currentTime);
    // printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));
    // printf("size = %d\n", puffs.size());
    // puff_output_ASCII(timestep);

}

void Gpuff::time_update_val(){

    cudaError_t err = cudaGetLastError();
    
    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (puffs.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;
    

    while(currentTime <= time_end){

        activationRatio = currentTime / time_end;

        update_puff_flags<<<blocks, threadsPerBlock>>>
            (d_puffs, activationRatio);
        cudaDeviceSynchronize();

        move_puffs_by_wind_val<<<blocks, threadsPerBlock>>>(d_puffs);
        cudaDeviceSynchronize();

        dry_deposition_val<<<blocks, threadsPerBlock>>>(d_puffs);
        cudaDeviceSynchronize();

        wet_scavenging_val<<<blocks, threadsPerBlock>>>(d_puffs);
        cudaDeviceSynchronize();

        radioactive_decay_val<<<blocks, threadsPerBlock>>>(d_puffs);
        cudaDeviceSynchronize();

        puff_dispersion_update_val<<<blocks, threadsPerBlock>>>(d_puffs);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("CUDA error: %s, timestep = %d\n", cudaGetErrorString(err), timestep);

        currentTime += dt;
        timestep++;

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", currentTime);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));

            //puff_output_ASCII(timestep);
            puff_output_binary(timestep);
        }

    }

}


void Gpuff::find_minmax(){

    cudaMalloc(&d_minX, sizeof(float));
    cudaMalloc(&d_minY, sizeof(float));
    cudaMalloc(&d_maxX, sizeof(float));
    cudaMalloc(&d_maxY, sizeof(float));

    float initial_min = FLT_MAX;
    float initial_max = -FLT_MAX;

    cudaMemcpy(d_minX, &initial_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_minY, &initial_min, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxX, &initial_max, sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_maxY, &initial_max, sizeof(float), cudaMemcpyHostToDevice);

    int threadsPerBlock = 256;
    int blocks = (puffs.size() + threadsPerBlock - 1) / threadsPerBlock;

    findMinMax<<<blocks, threadsPerBlock, 4 * threadsPerBlock * sizeof(float)>>>(d_puffs, d_minX, d_minY, d_maxX, d_maxY);

    cudaMemcpy(&minX, d_minX, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&minY, d_minY, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxX, d_maxX, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxY, d_maxY, sizeof(float), cudaMemcpyDeviceToHost);

    //std::cout << "Min X: " << minX << ", Max X: " << maxX << ", Min Y: " << minY << ", Max Y: " << maxY << std::endl;

    cudaFree(d_minX);
    cudaFree(d_minY);
    cudaFree(d_maxX);
    cudaFree(d_maxY);

}

// void Gpuff::conc_calc() {

//     RectangleGrid rect(minX, minY, maxX, maxY);

//     int ngrid = rect.rows * rect.cols;
    
//     RectangleGrid::GridPoint* d_grid;
//     float* d_concs;

//     cudaMalloc(&d_grid, ngrid * sizeof(RectangleGrid::GridPoint));
//     cudaMalloc(&d_concs, ngrid * sizeof(float));

//     float* h_concs = new float[ngrid];

//     cudaMemcpy(d_grid, rect.grid, ngrid * sizeof(RectangleGrid::GridPoint), cudaMemcpyHostToDevice);

//     int blockSize = 128;
//     dim3 threadsPerBlock(blockSize, blockSize);
//     dim3 numBlocks((ngrid + blockSize - 1) / blockSize, (nop + blockSize - 1) / blockSize);
    
//     accumulateConc<<<numBlocks, threadsPerBlock>>>
//         (d_puffs, d_grid, d_concs, ngrid);

//     cudaMemcpy(h_concs, d_concs, ngrid * sizeof(float), cudaMemcpyDeviceToHost);

//     grid_output_binary(rect, h_concs);

//     delete[] h_concs;
//     cudaFree(d_grid);
//     cudaFree(d_concs);

// }

void Gpuff::conc_calc(){

    RectangleGrid rect(minX, minY, maxX, maxY);

    int ngrid = rect.rows * rect.cols;
    
    RectangleGrid::GridPoint* d_grid;
    float* d_concs;

    cudaMalloc(&d_grid, ngrid * sizeof(RectangleGrid::GridPoint));
    cudaMalloc(&d_concs, ngrid * sizeof(float));

    float* h_concs = new float[ngrid];

    cudaMemcpy(d_grid, rect.grid, ngrid * sizeof(RectangleGrid::GridPoint), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int totalThreads = ngrid * nop;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    
    accumulate_conc<<<numBlocks, blockSize>>>
        (d_puffs, d_grid, d_concs, ngrid);

    cudaMemcpy(h_concs, d_concs, ngrid * sizeof(float), cudaMemcpyDeviceToHost);

    grid_output_binary(rect, h_concs);
    grid_output_csv(rect, h_concs);

    delete[] h_concs;
    cudaFree(d_grid);
    cudaFree(d_concs);

}

void Gpuff::conc_calc_val(){

    RectangleGrid rect(minX, minY, maxX, maxY);

    int ngrid = rect.rows * rect.cols * rect.zdim;
    
    RectangleGrid::GridPoint* d_grid;
    float* d_concs;

    cudaMalloc(&d_grid, ngrid * sizeof(RectangleGrid::GridPoint));
    cudaMalloc(&d_concs, ngrid * sizeof(float));

    float* h_concs = new float[ngrid];

    cudaMemcpy(d_grid, rect.grid, ngrid * sizeof(RectangleGrid::GridPoint), cudaMemcpyHostToDevice);

    int blockSize = 128;
    int totalThreads = ngrid * nop;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;
    
    accumulate_conc_val<<<numBlocks, blockSize>>>
        (d_puffs, d_grid, d_concs, ngrid);

    cudaMemcpy(h_concs, d_concs, ngrid * sizeof(float), cudaMemcpyDeviceToHost);

    grid_output_binary_val(rect, h_concs);
    grid_output_csv(rect, h_concs);

    delete[] h_concs;
    cudaFree(d_grid);
    cudaFree(d_concs);

}



void Gpuff::time_update_RCAP(){

    cudaError_t err = cudaGetLastError();
    
    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (puffs.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;

    int blockSize = 128;
    int totalThreads = 48 * nop;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;    

    while(currentTime <= time_end){

        activationRatio = currentTime / time_end;

        update_puff_flags<<<blocks, threadsPerBlock>>>
            (d_puffs, activationRatio);
        cudaDeviceSynchronize();

        time_inout_RCAP<<<blocks, threadsPerBlock>>>
        (d_puffs, d_RCAP_windir, d_RCAP_winvel, d_radi, currentTime, d_size, d_vdepo);
        cudaDeviceSynchronize();

        move_puffs_by_wind_RCAP<<<blocks, threadsPerBlock>>>
            (d_puffs, d_RCAP_windir, d_RCAP_winvel, d_radi);
        cudaDeviceSynchronize();

        // dry_deposition<<<blocks, threadsPerBlock>>>
        //     (d_puffs, 
        //     device_meteorological_data_pres,
        //     device_meteorological_data_unis,
        //     device_meteorological_data_etas);
        // cudaDeviceSynchronize();

        // wet_scavenging<<<blocks, threadsPerBlock>>>
        //     (d_puffs, 
        //     device_meteorological_data_pres,
        //     device_meteorological_data_unis,
        //     device_meteorological_data_etas);
        // cudaDeviceSynchronize();

        // radioactive_decay<<<blocks, threadsPerBlock>>>
        //     (d_puffs, 
        //     device_meteorological_data_pres,
        //     device_meteorological_data_unis,
        //     device_meteorological_data_etas);
        // cudaDeviceSynchronize();

        puff_dispersion_update_RCAP<<<blocks, threadsPerBlock>>>
            (d_puffs, d_RCAP_windir, d_RCAP_winvel, d_radi);
        cudaDeviceSynchronize();

        // accumulate_conc_RCAP<<<numBlocks, blockSize>>>(d_puffs, d_receptors);
        // cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("CUDA error: %s, timestep = %d\n", cudaGetErrorString(err), timestep);

        currentTime += dt;
        timestep++;

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", currentTime);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));

            //puff_output_ASCII(timestep);
            puff_output_binary(timestep);

            accumulate_conc_RCAP<<<numBlocks, blockSize>>>(d_puffs, d_receptors);
            cudaDeviceSynchronize();

            //receptor_output_binary_RCAP(timestep);
        }

    }

    for(int i=0; i<nop; i++){
        std::cout << std::endl;
        std::cout << "puff[" << i << "].tin: ";
        for(int j=0; j< RNUM; j++) std::cout << puffs[i].tin[j] << " ";
        std::cout << std::endl;

        std::cout << "puff[" << i << "].tout: ";
        for(int j=0; j< RNUM; j++) std::cout << puffs[i].tout[j] << " ";
        std::cout << std::endl;
        std::cout << std::endl;

        std::cout << "puff[" << i << "].conc" << std::endl;
        std::cout << "1_Xe      2_I       3_Cs      4_Te      5_Sr      6_Ru      7_La      8_Ce      9_Ba" << std::endl;
        for (int j = 0; j < NNUM; j++) std::cout << puffs[i].conc_arr[j] << " ";
        std::cout << std::endl;
        std::cout << std::endl;

        std::cout << "puff[" << i << "].fallout" << std::endl;
        std::cout << "           1_Xe      2_I       3_Cs      4_Te      5_Sr      6_Ru      7_La      8_Ce      9_Ba" << std::endl;
        for (int j = 0; j < RNUM; j++) {
            std::cout << "[Sector " << j + 1 << "] ";
            for (int k = 0; k < NNUM; k++) std::cout << std::scientific << std::setprecision(3) << puffs[i].fallout[k][j] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "puff[" << i << "].fd" << std::endl;
        std::cout << "           1_Xe      2_I       3_Cs      4_Te      5_Sr      6_Ru      7_La      8_Ce      9_Ba" << std::endl;
        for (int j = 0; j < RNUM; j++) {
            std::cout << "[Sector " << j + 1 << "] ";
            for (int k = 0; k < NNUM; k++) std::cout << std::scientific << puffs[i].fd[k][j] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "puff[" << i << "].fw" << std::endl;
        std::cout << "           1_Xe      2_I       3_Cs      4_Te      5_Sr      6_Ru      7_La      8_Ce      9_Ba" << std::endl;
        for (int j = 0; j < RNUM; j++) {
            std::cout << "[Sector " << j + 1 << "] ";
            for (int k = 0; k < NNUM; k++) std::cout << std::scientific << puffs[i].fw[k][j] << " ";
            std::cout << std::endl;
        }
        std::cout << std::endl;

        std::cout << "---------------------------------------------" << std::endl;

    }

}

void Gpuff::time_update_RCAP2(const SimulationControl& SC, const EvacuationData& EP, 
    const std::vector<RadioNuclideTransport>& RT, const std::vector<NuclideData>& ND, NuclideData* d_ND, const ProtectionFactors* dPF, const EvacuationData* dEP, int input_num) {

    cudaError_t err = cudaGetLastError();

    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks;

    int timestep = 0;

    cudaEvent_t start, stop;
    float timesum = 0;
    //plant_output_binary_RCAP(input_num, RT, ND);

    while (currentTime <= time_end) {


        blocks = (puffs_RCAP.size() + threadsPerBlock - 1) / threadsPerBlock;

        update_puff_flags2 << <blocks, threadsPerBlock >> >
            (d_puffs_RCAP, currentTime);
        cudaDeviceSynchronize();

        move_puffs_by_wind_RCAP2 << <numSims, totalpuff_per_Sim >> >
            (d_puffs_RCAP, d_Vdepo, d_particleSizeDistr, EP.EP_endRing, d_ground_deposit, 
                d_ND, d_radius, SC.numRad, SC.numTheta);
        cudaDeviceSynchronize();

        //blocks = (SC.numTheta * SC.numRad * MAX_NUCLIDES + threadsPerBlock - 1) / threadsPerBlock;
        //decayGroundDeposit << <blocks, threadsPerBlock >> > (d_ground_deposit, d_ND, SC.numTheta, SC.numRad);
        
        update_evac_velocity(EP, currentTime);

        //printf("currentTime = %f\n", currentTime);

        blocks = (evacuees.size() + threadsPerBlock - 1) / threadsPerBlock;

            evacuation_calculation_1D << <blocks, threadsPerBlock >> >
            ////evacuation_calculation_2D << <numSims, evacuees.size()/ numSims >> >
                (d_puffs_RCAP, d_dir, d_evacuees, d_radius, SC.numRad, SC.numTheta, 
                    evacuees.size(), EP.evaEndRing, EP.EP_endRing, d_ground_deposit, dEP, currentTime);
            cudaDeviceSynchronize();

        int numEvacuees = evacuees.size();

        //dim3 blockDim(totalpuff_per_Sim);

        //std::cout << totalpuff_per_Sim << " " << numSims << " " << numEvacuees << std::endl;

        //dim3 blockDim(totalpuff_per_Sim);
        //dim3 gridDim(numSims, numEvacuees/numSims);

        //////cudaEventCreate(&start);
        //////cudaEventCreate(&stop);  
        //////cudaEventRecord(start);

        //if (blockSize > 1024) blockSize = 1024;
        //blockSize = ((blockSize + 31) / 32) * 32;
         
        int blockSize = totalpuff_per_Sim;
        dim3 blockDim(blockSize);
        //dim3 gridDim(numSims, numEvacuees / numSims);
        dim3 gridDim(numEvacuees / numSims, numSims);  


        size_t sharedMemSize = sizeof(float) * blockSize * 2;
        //size_t sharedMemSize = sizeof(float) * blockSize;

        //printf("g = %d, b = %d, s = %d\n", numEvacuees / numSims, numSims, blockSize);

        ComputeExposureHmix << <gridDim, blockDim, sharedMemSize >> > (
            d_puffs_RCAP,
            d_evacuees, 
            d_exposure, 
            dPF
            );

        //ComputeExposureHmix_xy <<<gridDim, blockDim, sharedMemSize >> > (
        //    d_puffs_RCAP,
        //    d_evacuees,
        //    d_exposure,
        //    dPF
        //    );

        //ComputeExposureHmix_xy_single << <gridDim, blockDim, sharedMemSize >> > (
        //    d_puffs_RCAP,
        //    d_evacuees,
        //    d_exposure,
        //    dPF
        //    );

        //int totalEvacuees = d_totalevacuees_per_Sim * d_numSims;
        //int totalPuffs = d_totalpuff_per_Sim * d_numSims;

        //int totalThreads = totalEvacuees * totalPuffs;

        //int threadsPerBlock = 256;

        //int blocksPerGrid = (totalThreads + threadsPerBlock - 1) / threadsPerBlock;

        //size_t sharedMemSize = sizeof(float) * threadsPerBlock * 2;

        //ComputeExposureHmix_1D << <blocksPerGrid, threadsPerBlock, sharedMemSize >> > (
        //    d_puffs_RCAP,
        //    d_evacuees,
        //    d_exposure,
        //    dPF
        //    );



        //////cudaEventRecord(stop);
        //////cudaEventSynchronize(stop);

        //////float elapsedTime = getExecutionTime(start, stop);
        ////////std::cout << "Total execution time: " << elapsedTime << " ms" << std::endl;
        //////timesum += elapsedTime;

        //////cudaEventDestroy(start);
        //////cudaEventDestroy(stop);


        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("CUDA error: %s, timestep = %d\n", cudaGetErrorString(err), timestep);

        currentTime += dt;
        timestep++;

        if (timestep % freq_output == 0) {
            //printf("-------------------------------------------------\n");
            //printf("Time : %f\tsec\n", currentTime);
            //printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end / dt));

            puff_output_binary_RCAP(timestep);
            evac_output_binary_RCAP(timestep);
            //evac_output_binary_RCAP_xy(timestep);
            //evac_output_binary_RCAP_xy_single(timestep);

            cudaDeviceSynchronize();

        }

    }

    std::cout << "Total sum execution time: " << timesum << " ms" << std::endl;

}

void Gpuff::time_update_RCAP_cpu(const SimulationControl& SC, const EvacuationData& EP,
    const std::vector<RadioNuclideTransport>& RT, const std::vector<NuclideData>& ND, 
    NuclideData* d_ND, const ProtectionFactors* dPF, int input_num, EvacuationDirections ED, ProtectionFactors PF) {

    cudaError_t err = cudaGetLastError();

    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks;

    int timestep = 0;

    cudaEvent_t start, stop;
    float timesum = 0;
     
    while (currentTime <= time_end) {

        update_puff_flags2_cpu(currentTime, puffs_RCAP.size());
        move_puffs_by_wind_RCAP2_cpu
            (EP.EP_endRing, ND, SC.ir_distances, SC.numRad, SC.numTheta, puffs_RCAP.size());

        //for (int i = 0; i < nop; i++) { 
        //    printf("puffs_RCAP[%d].sigma_h = %e\n", i, puffs_RCAP[i].sigma_h);
        //}

        //update_evac_velocity_cpu(EP, currentTime);
        // 
        //evacuation_calculation_cpu
        //    (ED, evacuees, SC.ir_distances, SC.numRad, SC.numTheta,
        //        evacuees.size(), EP.evaEndRing, EP.EP_endRing);
         
        ComputeExposureHmix_cpu(evacuees, PF, numSims, 97, totalpuff_per_Sim);

        currentTime += dt;
        timestep++;

        if (timestep % freq_output == 0) {
            //printf("-------------------------------------------------\n");
            //printf("Time : %f\tsec\n", currentTime);
            //printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end / dt));

            puff_output_binary_RCAP_cpu(timestep);
            evac_output_binary_RCAP_cpu(timestep);

        }
    }
    std::cout << "Total sum execution time: " << timesum << " ms" << std::endl;

}

void Gpuff::time_update_polar(){

    cudaError_t err = cudaGetLastError();
    
    float currentTime = 0.0f;
    int threadsPerBlock = 256;
    int blocks = (puffs.size() + threadsPerBlock - 1) / threadsPerBlock;
    int timestep = 0;
    float activationRatio = 0.0;

    int blockSize = 128;
    int totalThreads = 48 * nop;
    int numBlocks = (totalThreads + blockSize - 1) / blockSize;  

    float dummys[48] = {0.};
    float dummy = 0.0f;

    printf("nop=%d\n", nop);

    while(currentTime <= time_end){

        activationRatio = currentTime / time_end;

        update_puff_flags<<<blocks, threadsPerBlock>>>
            (d_puffs, activationRatio);
        cudaDeviceSynchronize();

        move_puffs_by_wind_var<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas, currentTime);
        cudaDeviceSynchronize();

        // dry_deposition<<<blocks, threadsPerBlock>>>
        //     (d_puffs, 
        //     device_meteorological_data_pres,
        //     device_meteorological_data_unis,
        //     device_meteorological_data_etas);
        // cudaDeviceSynchronize();

        // wet_scavenging<<<blocks, threadsPerBlock>>>
        //     (d_puffs, 
        //     device_meteorological_data_pres,
        //     device_meteorological_data_unis,
        //     device_meteorological_data_etas);
        // cudaDeviceSynchronize();

        // radioactive_decay<<<blocks, threadsPerBlock>>>
        //     (d_puffs, 
        //     device_meteorological_data_pres,
        //     device_meteorological_data_unis,
        //     device_meteorological_data_etas);
        // cudaDeviceSynchronize();

        puff_dispersion_update<<<blocks, threadsPerBlock>>>
            (d_puffs, 
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess) 
            printf("CUDA error: %s, timestep = %d\n", cudaGetErrorString(err), timestep);

        currentTime += dt;
        timestep++;

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", currentTime);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));

            //puff_output_ASCII(timestep);
            puff_output_binary(timestep);

            accumulate_conc_RCAP<<<numBlocks, blockSize>>>(d_puffs, d_receptors);
            cudaDeviceSynchronize();

            receptor_output_binary_RCAP(timestep);

            cudaMemcpy(receptors.data(), d_receptors, receptors.size() * sizeof(receptors_RCAP), cudaMemcpyDeviceToHost);
            dummy = receptors[0].conc;//14
            //cudaMemcpyFromSymbol(&dummy, d_receptors[15].conc, sizeof(float));
            con1.push_back(dummy);
            dummy = receptors[16].conc;//30
            //cudaMemcpyFromSymbol(&dummy, d_receptors[31].conc, sizeof(float));
            con2.push_back(dummy);
            dummy = receptors[32].conc;//46
            //cudaMemcpyFromSymbol(&dummy, d_receptors[47].conc, sizeof(float));
            con3.push_back(dummy);
        }

    }
    //for (float element : con1) std::cout << element << std::endl;



    std::ofstream outFile1("output1.txt");
    for (float element : con1) outFile1 << element << std::endl;
    outFile1.close();

    std::ofstream outFile2("output2.txt");
    for (float element : con2) outFile2 << element << std::endl;
    outFile2.close();

    std::ofstream outFile3("output3.txt");
    for (float element : con3) outFile3 << element << std::endl;
    outFile3.close();



    // printf("-------------------------------------------------\n");
    // printf("Time : %f\n\tsec", currentTime);
    // printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));
    // printf("size = %d\n", puffs.size());
    // puff_output_ASCII(timestep);

}

void Gpuff::allocate_and_copy_puffs_RCAP_to_device() {
    size_t size = puffs_RCAP.size() * sizeof(Puffcenter_RCAP);
    cudaMalloc(&d_puffs_RCAP, size);
    cudaMemcpy(d_puffs_RCAP, puffs_RCAP.data(), size, cudaMemcpyHostToDevice);
}
void Gpuff::allocate_and_copy_evacuees_to_device() {
    cudaMalloc(&d_evacuees, evacuees.size() * sizeof(Evacuee));
    cudaMemcpy(d_evacuees, evacuees.data(), evacuees.size() * sizeof(Evacuee), cudaMemcpyHostToDevice);
}
void Gpuff::allocate_and_copy_radius_to_device(SimulationControl SC) {
    cudaMalloc(&d_radius, SC.numRad * sizeof(float));
    cudaMemcpy(d_radius, SC.ir_distances, SC.numRad * sizeof(float), cudaMemcpyHostToDevice);
}

void Gpuff::free_puffs_RCAP_device_memory() {
    if (d_puffs_RCAP != nullptr) {
        cudaFree(d_puffs_RCAP);
        d_puffs_RCAP = nullptr;
    }
}

void Gpuff::health_effect(std::vector<Evacuee>& evacuees, HealthEffect HE) {

    std::vector<ResultLine> results;
    results.reserve(evacuees.size() * 14);

    float risk_deterministic = 0.0f;
    float risk_stochastic_cf = 0.0f;
    float risk_stochastic_ci = 0.0f;
    float dose = 0.0f;

    float H = 0.0;
    float npeople = 0.0;


    for (int evaIdx = 0; evaIdx < (int)evacuees.size(); evaIdx++)
    {
        int metVal = evacuees[evaIdx].met_idx;
        int radVal = evacuees[evaIdx].prev_rad_idx;
        int thetaVal = evacuees[evaIdx].prev_theta_idx;
        float pop = evacuees[evaIdx].population;

        //----------------------------------------------
        // (A) 결정적 영향(Derministic Effect) 계산
        //----------------------------------------------

        // 1) HematopoieticSyndrome (RED_MARR: 3)
        {
            int oidx = 3;
            int nidx = 0;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.threshold_AF[nidx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nidx], HE.beta_f[nidx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "deterministic",
                        "HematopoieticSyndrome",
                        npeople
                        });
                }
            }
        }

        // 2) PulmonarySyndrome (LUNGS: 2)
        {
            int oidx = 2;
            int nidx = 1;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.threshold_AF[nidx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nidx], HE.beta_f[nidx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "deterministic",
                        "PulmonarySyndrome",
                        npeople
                        });
                }
            }
        }

        // 3) Prodromal_Vomit (STOMACH: 0)
        {
            int oidx = 0;
            int nidx = 0;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.threshold_AF[nidx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nidx], HE.beta_f[nidx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "deterministic",
                        "Prodromal_Vomit",
                        npeople
                        });
                }
            }
        }

        // 4) Diarrhea (STOMACH: 0)
        {
            int oidx = 0;
            int nidx = 1;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.threshold_AF[nidx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nidx], HE.beta_f[nidx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "deterministic",
                        "Diarrhea",
                        npeople
                        });
                }
            }
        }

        // 5) Pneumonitis (LUNGS: 2)
        {
            int oidx = 2;
            int nidx = 2;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.threshold_AF[nidx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nidx], HE.beta_f[nidx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "deterministic",
                        "Pneumonitis",
                        npeople
                        });
                }
            }
        }

        // 6) Thyroiditis (THYROIDH: 11)
        {
            int oidx = 11;
            int nidx = 5;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.threshold_AF[nidx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nidx], HE.beta_f[nidx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "deterministic",
                        "Thyroiditis",
                        npeople
                        });
                }
            }
        }

        // 7) Hypothyrodism (THYROIDH: 11)
        {
            int oidx = 11;
            int nidx = 6;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.threshold_AF[nidx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nidx], HE.beta_f[nidx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "deterministic",
                        "Hypothyrodism",
                        npeople
                        });
                }
            }
        }

        //----------------------------------------------
        // (B) 확률적 영향(Stochastic Effect) 계산
        //  - _cf: 명목 "암 발생" 위험도 계수
        //  - _ci: 명목 "암 사망" 위험도 계수
        //----------------------------------------------

        // 8) Leukemia (RED_MARR: 3)
        {
            int oidx = 3;
            int nidx = 0;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.LNT_threshold[nidx]) {
                risk_stochastic_cf = HE.cf_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);
                risk_stochastic_ci = HE.ci_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nidx]) {
                    risk_stochastic_cf /= HE.ddrf[nidx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nidx]) {
                    risk_stochastic_ci /= HE.ddrf[nidx];
                }

                float npeople_cf = HE.sus_frac[nidx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nidx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_cf",
                        "Leukemia_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_ci",
                        "Leukemia_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 9) Bone (BONE_SUR: 6)
        {
            int oidx = 6;
            int nidx = 1;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.LNT_threshold[nidx]) {
                risk_stochastic_cf = HE.cf_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);
                risk_stochastic_ci = HE.ci_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nidx]) {
                    risk_stochastic_cf /= HE.ddrf[nidx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nidx]) {
                    risk_stochastic_ci /= HE.ddrf[nidx];
                }

                float npeople_cf = HE.sus_frac[nidx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nidx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_cf",
                        "BoneCancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_ci",
                        "BoneCancer_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 10) Breast (BREAST: 7)
        {
            int oidx = 7;
            int nidx = 2;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.LNT_threshold[nidx]) {
                risk_stochastic_cf = HE.cf_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);
                risk_stochastic_ci = HE.ci_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nidx]) {
                    risk_stochastic_cf /= HE.ddrf[nidx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nidx]) {
                    risk_stochastic_ci /= HE.ddrf[nidx];
                }

                float npeople_cf = HE.sus_frac[nidx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nidx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_cf",
                        "BreastCancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_ci",
                        "BreastCancer_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 11) Lung (LUNGS: 2)
        {
            int oidx = 2;
            int nidx = 3;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.LNT_threshold[nidx]) {
                risk_stochastic_cf = HE.cf_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);
                risk_stochastic_ci = HE.ci_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nidx]) {
                    risk_stochastic_cf /= HE.ddrf[nidx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nidx]) {
                    risk_stochastic_ci /= HE.ddrf[nidx];
                }

                float npeople_cf = HE.sus_frac[nidx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nidx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_cf",
                        "LungCancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_ci",
                        "LungCancer_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 12) Thyroid (THYROIDH: 11)
        {
            int oidx = 11;
            int nidx = 4;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.LNT_threshold[nidx]) {
                risk_stochastic_cf = HE.cf_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);
                risk_stochastic_ci = HE.ci_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nidx]) {
                    risk_stochastic_cf /= HE.ddrf[nidx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nidx]) {
                    risk_stochastic_ci /= HE.ddrf[nidx];
                }

                float npeople_cf = HE.sus_frac[nidx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nidx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_cf",
                        "ThyroidCancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_ci",
                        "ThyroidCancer_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 13) GI (LOWER_LI: 5)
        {
            int oidx = 5;
            int nidx = 5;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.LNT_threshold[nidx]) {
                risk_stochastic_cf = HE.cf_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);
                risk_stochastic_ci = HE.ci_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nidx]) {
                    risk_stochastic_cf /= HE.ddrf[nidx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nidx]) {
                    risk_stochastic_ci /= HE.ddrf[nidx];
                }

                float npeople_cf = HE.sus_frac[nidx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nidx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_cf",
                        "GICancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_ci",
                        "GICancer_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 14) Other (EDEWBODY: 10)
        {
            int oidx = 10;
            int nidx = 6;
            dose = evacuees[evaIdx].dose_inhalations[oidx]
                + evacuees[evaIdx].dose_cloudshines[oidx];
            if (dose > HE.LNT_threshold[nidx]) {
                risk_stochastic_cf = HE.cf_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);
                risk_stochastic_ci = HE.ci_risk[nidx]
                    * dose * (HE.dos_a[nidx] + HE.dos_b[nidx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nidx]) {
                    risk_stochastic_cf /= HE.ddrf[nidx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nidx]) {
                    risk_stochastic_ci /= HE.ddrf[nidx];
                }

                float npeople_cf = HE.sus_frac[nidx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nidx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_cf",
                        "OtherCancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        metVal, radVal, thetaVal,
                        "stochastic_ci",
                        "OtherCancer_mortality",
                        npeople_ci
                        });
                }
            }
        }

    } // end of evacuee loop

    std::map<int, std::map<std::string, float>> sumMap;

    for (auto& rec : results) {
        sumMap[rec.metVal][rec.effect_name] += rec.npeople;
    }

    std::ofstream ofs("health_effect_output.csv");

    ofs << "met_idx,rad_idx,theta_idx,effect_type,effect_name,npeople,npeople_fraction" << std::endl;

    for (auto& rec : results) {
        float totalN = sumMap[rec.metVal][rec.effect_name];
        float fraction = 0.0f;
        if (totalN > 1.0e-12f) {
            fraction = rec.npeople / totalN;
        }

        ofs << rec.metVal << ","
            << rec.radVal << ","
            << rec.thetaVal << ","
            << rec.effect_type << ","
            << rec.effect_name << ","
            << rec.npeople << ","
            << fraction
            << std::endl;
    }

    ofs.close();

}
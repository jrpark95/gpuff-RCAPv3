
#include "gpuff.cuh"
#include <cmath>
#include <algorithm>  // For std::min, std::max
#include <fstream>    // For file output in debug functions
#include <iomanip>    // For output formatting

// ============================================================================
// Constructor & Destructor
// ============================================================================

/**
 * Default constructor for Gpuff class
 * Initializes all GPU device pointers to nullptr to ensure safe memory management
 */
Gpuff::Gpuff()
    : device_meteorological_data_pres(nullptr),
      device_meteorological_data_unis(nullptr),
      device_meteorological_data_etas(nullptr){}

/**
 * Destructor for Gpuff class
 * Releases all GPU device memory allocated during execution
 * Memory cleanup order:
 *   1. Meteorological data arrays (pres, unis, etas)
 *   2. Puff data structures
 */
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


// ============================================================================
// Timing Utilities
// ============================================================================

/**
 * Starts the high-resolution performance timer
 * Used for benchmarking simulation execution time
 */
void Gpuff::clock_start(){
    _clock0 = std::chrono::high_resolution_clock::now();
}

/**
 * Stops the performance timer and prints elapsed time
 * Calculates duration from clock_start() and outputs to console
 * Time resolution: microseconds, output in seconds
 */
void Gpuff::clock_end(){
    _clock1 = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(_clock1 - _clock0);
    std::cout << "Elapsed time: " << duration.count()/1.0e6 << " seconds" << std::endl;
}

// ============================================================================
// Evacuation Velocity Management
// ============================================================================

/**
 * Updates evacuee velocity based on time-dependent speed profiles (GPU version)
 *
 * This function manages evacuee speed transitions during the simulation.
 * Speed changes occur at predefined time intervals based on evacuation scenario.
 *
 * @param EP EvacuationData structure containing speed profiles and timing
 * @param currentTime Current simulation time (seconds)
 *
 * Host-Device Memory Flow:
 *   1. Check if speed update is needed based on time
 *   2. Update evacuee speeds in host memory
 *   3. Copy updated evacuees to device via cudaMemcpy
 *
 * Speed Periods:
 *   - Intermediate periods: Update speed and advance to next period
 *   - Final period: Set speed and extend to simulation end time
 */
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
        }
        else if (currentSpeedIndex == EP.nSpeedPeriod - 1) {

            float speed = EP.speeds[currentSpeedIndex];
            currentSpeedEndTime = time_end;
            currentSpeedIndex++;

            for (auto& evacuee : evacuees) {
                evacuee.speed = speed;
            }

            cudaMemcpy(d_evacuees, evacuees.data(), evacuees.size() * sizeof(Evacuee), cudaMemcpyHostToDevice);
        }
    }
}

/**
 * Updates evacuee velocity based on time-dependent speed profiles (CPU version)
 *
 * Same functionality as update_evac_velocity() but for CPU-only execution.
 * No GPU memory transfer occurs in this version.
 *
 * @param EP EvacuationData structure containing speed profiles and timing
 * @param currentTime Current simulation time (seconds)
 */
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
        }
        else if (currentSpeedIndex == EP.nSpeedPeriod - 1) {

            float speed = EP.speeds[currentSpeedIndex];
            currentSpeedEndTime = time_end;
            currentSpeedIndex++;

            for (auto& evacuee : evacuees) {
                evacuee.speed = speed;
            }
        }
    }
}
// ============================================================================
// Output & Debugging Functions
// ============================================================================

/**
 * Prints puff data to text file for debugging
 *
 * Outputs formatted puff information to "output.txt"
 * Execution: CPU-side operation on host data
 *
 * Output Format:
 *   - Position coordinates (x, y, z) in fixed-point notation
 *   - Decay constant and concentration in scientific notation
 *   - Time index and activation flag
 */
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
// ============================================================================
// GPU Memory Management - Basic Puff Operations
// ============================================================================

/**
 * Allocates GPU memory and copies puff data from host to device
 *
 * Memory Flow:
 *   1. Allocate device memory for puff array
 *   2. Copy puff data from host (puffs vector) to device (d_puffs)
 *
 * Memory Ownership:
 *   - Device pointer d_puffs is managed by class
 *   - Must be freed in destructor or via free_puffs_device_memory()
 *
 * Error Handling:
 *   - Exits program if allocation or copy fails
 *   - Prints descriptive error message before exit
 */
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

/**
 * Prints puff time indices from device memory (debugging utility)
 *
 * CUDA Kernel Configuration:
 *   - Threads per block: 256
 *   - Blocks: Calculated to cover all puffs
 *   - Synchronization: Required after kernel launch
 *
 * Execution: GPU kernel launched from host
 */
void Gpuff::print_device_puffs_timeidx(){

    const int threads_per_block = 256;
    const int blocks = (puffs.size() + threads_per_block - 1) / threads_per_block;

    print_timeidx_kernel<<<blocks, threads_per_block>>>(d_puffs);

    cudaDeviceSynchronize();

}

// ============================================================================
// Main Simulation Loop - Standard Atmospheric Dispersion
// ============================================================================

/**
 * Main time-stepping loop for atmospheric dispersion simulation
 *
 * Executes sequential GPU kernels for puff transport and physics.
 * Each timestep advances the simulation by dt seconds.
 *
 * Simulation Pipeline (per timestep):
 *   1. update_puff_flags        - Activate puffs based on release schedule
 *   2. move_puffs_by_wind       - Advect puffs using wind field
 *   3. dry_deposition           - Apply dry deposition to ground
 *   4. wet_scavenging           - Apply precipitation scavenging
 *   5. radioactive_decay        - Apply radioactive decay
 *   6. puff_dispersion_update   - Update dispersion parameters (sigma)
 *
 * CUDA Kernel Configuration:
 *   - Threads per block: 256
 *   - Blocks: Calculated to cover all puffs
 *   - Synchronization: Required after each kernel
 *
 * Output:
 *   - Periodic binary output at freq_output intervals
 *   - Progress information printed to console
 */
void Gpuff::time_update(){

    cudaError_t err = cudaGetLastError();

    float current_time = 0.0f;
    int threads_per_block = 256;
    int blocks = (puffs.size() + threads_per_block - 1) / threads_per_block;
    int timestep = 0;
    float activation_ratio = 0.0;


    while(current_time <= time_end){

        activation_ratio = current_time / time_end;

        update_puff_flags_kernel<<<blocks, threads_per_block>>>
            (d_puffs, activation_ratio);
        cudaDeviceSynchronize();

        move_puffs_by_wind_kernel<<<blocks, threads_per_block>>>
            (d_puffs,
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        dry_deposition_kernel<<<blocks, threads_per_block>>>
            (d_puffs,
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        wet_scavenging_kernel<<<blocks, threads_per_block>>>
            (d_puffs,
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        radioactive_decay_kernel<<<blocks, threads_per_block>>>
            (d_puffs,
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        puff_dispersion_update<<<blocks, threads_per_block>>>
            (d_puffs,
            device_meteorological_data_pres,
            device_meteorological_data_unis,
            device_meteorological_data_etas);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("CUDA error: %s, timestep = %d\n", cudaGetErrorString(err), timestep);

        current_time += dt;
        timestep++;

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", current_time);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));

            puff_output_binary(timestep);
        }

    }

}

/**
 * Validation version of time-stepping loop
 *
 * Uses simplified validation kernels (_val suffix) for testing/verification.
 * These kernels may use hardcoded or simplified parameters for validation purposes.
 *
 * Simulation Pipeline (per timestep):
 *   1. update_puff_flags           - Activate puffs
 *   2. move_puffs_by_wind_val      - Wind advection (validation)
 *   3. dry_deposition_val          - Dry deposition (validation)
 *   4. wet_scavenging_val          - Wet scavenging (validation)
 *   5. radioactive_decay_val       - Radioactive decay (validation)
 *   6. puff_dispersion_update_val  - Dispersion update (validation)
 */
void Gpuff::time_update_val(){

    cudaError_t err = cudaGetLastError();

    float current_time = 0.0f;
    int threads_per_block = 256;
    int blocks = (puffs.size() + threads_per_block - 1) / threads_per_block;
    int timestep = 0;
    float activation_ratio = 0.0;


    while(current_time <= time_end){

        activation_ratio = current_time / time_end;

        update_puff_flags_kernel<<<blocks, threads_per_block>>>
            (d_puffs, activation_ratio);
        cudaDeviceSynchronize();

        move_puffs_by_wind_val<<<blocks, threads_per_block>>>(d_puffs);
        cudaDeviceSynchronize();

        dry_deposition_val<<<blocks, threads_per_block>>>(d_puffs);
        cudaDeviceSynchronize();

        wet_scavenging_val<<<blocks, threads_per_block>>>(d_puffs);
        cudaDeviceSynchronize();

        radioactive_decay_val<<<blocks, threads_per_block>>>(d_puffs);
        cudaDeviceSynchronize();

        puff_dispersion_update_val<<<blocks, threads_per_block>>>(d_puffs);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("CUDA error: %s, timestep = %d\n", cudaGetErrorString(err), timestep);

        current_time += dt;
        timestep++;

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", current_time);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));

            puff_output_binary(timestep);
        }

    }

}
// ============================================================================
// Spatial Extent Calculation
// ============================================================================

/**
 * Finds spatial extent of puff cloud for grid generation
 *
 * Computes min/max X and Y coordinates across all puffs using GPU reduction.
 * Results are used to define concentration grid boundaries.
 *
 * Memory Flow:
 *   1. Allocate device memory for min/max values
 *   2. Initialize with extreme values (FLT_MAX, -FLT_MAX)
 *   3. Launch reduction kernel with shared memory
 *   4. Copy results back to host
 *   5. Free temporary device memory
 *
 * CUDA Kernel Configuration:
 *   - Threads per block: 256
 *   - Shared memory: 4 * threads_per_block * sizeof(float)
 *   - Performs parallel reduction on device
 *
 * Output:
 *   - Updates member variables: minX, minY, maxX, maxY
 */
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

    int threads_per_block = 256;
    int blocks = (puffs.size() + threads_per_block - 1) / threads_per_block;

    findMinMax<<<blocks, threads_per_block, 4 * threads_per_block * sizeof(float)>>>(d_puffs, d_minX, d_minY, d_maxX, d_maxY);

    cudaMemcpy(&minX, d_minX, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&minY, d_minY, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxX, d_maxX, sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(&maxY, d_maxY, sizeof(float), cudaMemcpyDeviceToHost);

    cudaFree(d_minX);
    cudaFree(d_minY);
    cudaFree(d_maxX);
    cudaFree(d_maxY);

}

// ============================================================================
// Concentration Grid Calculation
// ============================================================================

/**
 * Calculates concentrations on a rectangular grid
 *
 * Creates a regular grid covering the puff cloud extent and accumulates
 * concentration from all puffs at each grid point using Gaussian distribution.
 *
 * Memory Flow:
 *   1. Create RectangleGrid structure based on min/max coordinates
 *   2. Allocate device memory for grid points and concentrations
 *   3. Copy grid structure to device
 *   4. Launch accumulation kernel (puff-grid interaction)
 *   5. Copy concentration results back to host
 *   6. Output to binary and CSV files
 *   7. Free temporary device memory
 *
 * CUDA Kernel Configuration:
 *   - Block size: 128 threads
 *   - Total threads: ngrid * nop (grid points * number of puffs)
 *   - Blocks: Calculated to cover all threads
 *
 * Output:
 *   - Binary output: grid_output_binary()
 *   - CSV output: grid_output_csv()
 */
void Gpuff::conc_calc(){

    RectangleGrid rect(minX, minY, maxX, maxY);

    int ngrid = rect.rows * rect.cols;

    RectangleGrid::GridPoint* d_grid;
    float* d_concs;

    cudaMalloc(&d_grid, ngrid * sizeof(RectangleGrid::GridPoint));
    cudaMalloc(&d_concs, ngrid * sizeof(float));

    float* h_concs = new float[ngrid];

    cudaMemcpy(d_grid, rect.grid, ngrid * sizeof(RectangleGrid::GridPoint), cudaMemcpyHostToDevice);

    int block_size = 128;
    int total_threads = ngrid * nop;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    accumulate_conc<<<num_blocks, block_size>>>
        (d_puffs, d_grid, d_concs, ngrid);

    cudaMemcpy(h_concs, d_concs, ngrid * sizeof(float), cudaMemcpyDeviceToHost);

    grid_output_binary(rect, h_concs);
    grid_output_csv(rect, h_concs);

    delete[] h_concs;
    cudaFree(d_grid);
    cudaFree(d_concs);

}

/**
 * Calculates concentrations on 3D rectangular grid (validation version)
 *
 * Same as conc_calc() but with 3D grid including vertical dimension.
 * Uses validation kernel for concentration accumulation.
 *
 * Grid Dimensions:
 *   - Horizontal: rect.rows * rect.cols
 *   - Vertical: rect.zdim
 *   - Total: rows * cols * zdim
 */
void Gpuff::conc_calc_val(){

    RectangleGrid rect(minX, minY, maxX, maxY);

    int ngrid = rect.rows * rect.cols * rect.zdim;

    RectangleGrid::GridPoint* d_grid;
    float* d_concs;

    cudaMalloc(&d_grid, ngrid * sizeof(RectangleGrid::GridPoint));
    cudaMalloc(&d_concs, ngrid * sizeof(float));

    float* h_concs = new float[ngrid];

    cudaMemcpy(d_grid, rect.grid, ngrid * sizeof(RectangleGrid::GridPoint), cudaMemcpyHostToDevice);

    int block_size = 128;
    int total_threads = ngrid * nop;
    int num_blocks = (total_threads + block_size - 1) / block_size;

    accumulate_conc_val<<<num_blocks, block_size>>>
        (d_puffs, d_grid, d_concs, ngrid);

    cudaMemcpy(h_concs, d_concs, ngrid * sizeof(float), cudaMemcpyDeviceToHost);

    grid_output_binary_val(rect, h_concs);
    grid_output_csv(rect, h_concs);

    delete[] h_concs;
    cudaFree(d_grid);
    cudaFree(d_concs);

}
// ============================================================================
// RCAP Simulation - Reactor Consequence Analysis Package
// ============================================================================

/**
 * Time-stepping loop for RCAP reactor accident simulation
 *
 * Specialized version for nuclear reactor accident consequence analysis.
 * Uses polar coordinate system for radionuclide transport and receptor calculations.
 *
 * RCAP-Specific Features:
 *   - Polar coordinate wind field (direction and velocity)
 *   - Multi-nuclide tracking with deposition
 *   - Receptor point concentration accumulation
 *   - Time-dependent source term activation
 *
 * Simulation Pipeline (per timestep):
 *   1. update_puff_flags           - Activate puffs based on release
 *   2. time_inout_RCAP             - Track puff entry/exit times at receptors
 *   3. move_puffs_by_wind_RCAP     - Advect in polar coordinates
 *   4. puff_dispersion_update_RCAP - Update dispersion parameters
 *
 * CUDA Kernel Configuration:
 *   - Puff kernels: 256 threads per block
 *   - Receptor kernels: 128 threads per block
 */
void Gpuff::time_update_RCAP(){

    cudaError_t err = cudaGetLastError();

    float current_time = 0.0f;
    int threads_per_block = 256;
    int blocks = (puffs.size() + threads_per_block - 1) / threads_per_block;
    int timestep = 0;
    float activation_ratio = 0.0;

    int block_size = 128;
    int total_threads = 48 * nop;
    int num_blocks = (total_threads + block_size - 1) / block_size;    

    while(current_time <= time_end){

        activation_ratio = current_time / time_end;

        update_puff_flags_kernel<<<blocks, threads_per_block>>>
            (d_puffs, activation_ratio);
        cudaDeviceSynchronize();

        time_inout_RCAP<<<blocks, threads_per_block>>>
        (d_puffs, d_RCAP_windir, d_RCAP_winvel, d_radi, current_time, d_size, d_vdepo);
        cudaDeviceSynchronize();

        move_puffs_by_wind_RCAP<<<blocks, threads_per_block>>>
            (d_puffs, d_RCAP_windir, d_RCAP_winvel, d_radi);
        cudaDeviceSynchronize();

        puff_dispersion_update_RCAP<<<blocks, threads_per_block>>>
            (d_puffs, d_RCAP_windir, d_RCAP_winvel, d_radi);
        cudaDeviceSynchronize();

        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("CUDA error: %s, timestep = %d\n", cudaGetErrorString(err), timestep);

        current_time += dt;
        timestep++;

        if(timestep % freq_output==0){
            printf("-------------------------------------------------\n");
            printf("Time : %f\tsec\n", current_time);
            printf("Time steps : \t%d of \t%d\n", timestep, (int)(time_end/dt));

            puff_output_binary(timestep);

            accumulate_conc_RCAP<<<num_blocks, block_size>>>(d_puffs, d_receptors);
            cudaDeviceSynchronize();
        }

    }

    // Debug output: Print puff deposition data for verification
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

// ============================================================================
// Puff720 Geometry Initialization Functions
// ============================================================================

// 표준정규 CDF 역함수 Φ^{-1}(p) 근사
inline double inv_norm_cdf(double p) {
    static const double a1=-3.969683028665376e+01,a2=2.209460984245205e+02,a3=-2.759285104469687e+02;
    static const double a4= 1.383577518672690e+02,a5=-3.066479806614716e+01,a6= 2.506628277459239e+00;
    static const double b1=-5.447609879822406e+01,b2=1.615858368580409e+02,b3=-1.556989798598866e+02;
    static const double b4= 6.680131188771972e+01,b5=-1.328068155288572e+01;
    static const double c1=-7.784894002430293e-03,c2=-3.223964580411365e-01,c3=-2.400758277161838e+00;
    static const double c4=-2.549732539343734e+00,c5= 4.374664141464968e+00,c6= 2.938163982698783e+00;
    static const double d1= 7.784695709041462e-03,d2= 3.224671290700398e-01,d3= 2.445134137142996e+00,d4= 3.754408661907416e+00;

    const double plow  = 0.02425;
    const double phigh = 1.0 - plow;

    double q, r, x;
    if (p < plow) {
        q = std::sqrt(-2.0 * std::log(p));
        x = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
            ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
        return -x;
    } else if (p > phigh) {
        q = std::sqrt(-2.0 * std::log(1.0 - p));
        x = (((((c1*q + c2)*q + c3)*q + c4)*q + c5)*q + c6) /
            ((((d1*q + d2)*q + d3)*q + d4)*q + 1.0);
        return x;
    } else {
        q = p - 0.5;
        r = q * q;
        x = (((((a1*r + a2)*r + a3)*r + a4)*r + a5)*r + a6) * q /
            (((((b1*r + b2)*r + b3)*r + b4)*r + b5)*r + 1.0);
        return x;
    }
}

// 수평 반경 r의 6개 등질량 셸 중심값
// Rayleigh CDF F_R(r)=1-exp(-r^2/2) 를 6등분, 각 셸 중앙 분위수 p_c=(j-0.5)/6
// σ_y=1 참조에서 r_c=sqrt(-2 ln(1-p_c))
inline void build_radial_centers_6(std::vector<double>& r_centers) {
    r_centers.resize(6);
    for (int j = 1; j <= 6; ++j) {
        const double p_c = (static_cast<double>(j) - 0.5) / 6.0;
        const double r_c = std::sqrt(-2.0 * std::log(std::max(1e-12, 1.0 - p_c)));
        r_centers[j - 1] = r_c; // σ_y=1 기준
    }
}

// 수직 z의 10개 등질량 레이어 중심값
// 표준정규 CDF Φ(z) 10등분, 각 레이어 중앙 분위수 p_c=(i-0.5)/10
// z_c=Φ^{-1}(p_c), σ_z=1 기준
inline void build_vertical_centers_10(std::vector<double>& z_centers) {
    z_centers.resize(10);
    for (int i = 1; i <= 10; ++i) {
        const double p_c = (static_cast<double>(i) - 0.5) / 10.0;
        const double z_c = inv_norm_cdf(std::min(1.0 - 1e-12, std::max(1e-12, p_c)));
        z_centers[i - 1] = z_c; // σ_z=1 기준
    }
}

// 방위각 12등분
inline void build_theta_12(std::vector<double>& thetas) {
    thetas.resize(12);
    for (int j = 0; j < 12; ++j) thetas[j] = 2.0 * PI * static_cast<double>(j) / 12.0;
}

// 6 × 10 × 12 = 720 정규화 좌표 생성
inline void build_geom720_normalized(Puff720& geom) {
    std::vector<double> r6, z10, th12;
    build_radial_centers_6(r6);
    build_vertical_centers_10(z10);
    build_theta_12(th12);

    int idx = 0;
    for (int ir = 0; ir < 6; ++ir) {
        const double r = r6[ir];
        for (int iz = 0; iz < 10; ++iz) {
            const double z = z10[iz];
            for (int it = 0; it < 12; ++it) {
                const double th = th12[it];
                const double x = r * std::cos(th);
                const double y = r * std::sin(th);
                geom.pos_local[idx++] = float3{static_cast<float>(x),
                                               static_cast<float>(y),
                                               static_cast<float>(z)};
            }
        }
    }
    std::cout << "[CLOUDSHINE:001] Initialized Puff720 geometry with 6×10×12 points" << std::endl;
}

// ============================================================================
// Cloudshine Table Initialization Functions
// ============================================================================

// Constants for table building
static constexpr float kMu_air_host  = 0.01f;   // Air attenuation coefficient [1/m]
static constexpr float kK_build_host = 1.4f;    // Build-up factor coefficient
static constexpr float kTwo13_host   = 2.13e6f; // Unit conversion factor

// Create logarithmically spaced distance grid
inline std::vector<float> make_p_grid(int Np, float p_min_m = 0.1f, float p_max_m = 5.0e4f) {
    std::vector<float> p(Np);
    float log_min = std::log(p_min_m);
    float log_max = std::log(p_max_m);
    for (int i = 0; i < Np; ++i) {
        float t = static_cast<float>(i) / static_cast<float>(Np - 1);
        p[i] = std::exp(log_min + t * (log_max - log_min));
    }
    // Avoid singularity at first point
    if (p[0] < 1e-3f) p[0] = 1e-3f;
    return p;
}

// Simplified attenuation function A(rho) = B(μ,rho) * exp(-μ rho) with B = 1 + k * μ * rho
inline float A_of_rho(float rho) {
    float mu = kMu_air_host;
    float B  = 1.0f + kK_build_host * mu * rho;
    return B * std::exp(-mu * rho);
}

// Point kernel dose rate: D′p(rho) = 2.13e6 / (4π rho^2) * A(rho)
inline float Dp_point_of_rho(float rho) {
    float geom = kTwo13_host / (4.0f * PI * rho * rho);
    return geom * A_of_rho(rho);
}

// Debug output function for cloudshine tables
inline void output_cloudshine_tables_to_file(
    const std::vector<float>& h_p,
    const std::vector<float>& h_dp,
    const std::vector<float>& h_df,
    int Nnucl, int Np,
    const std::string& filename = "cloudshine_tables_debug.txt"
) {
    std::ofstream outfile(filename);

    if (!outfile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << " for writing." << std::endl;
        return;
    }

    outfile << "========================================\n";
    outfile << "CLOUDSHINE DOSE TABLES DEBUG OUTPUT\n";
    outfile << "========================================\n\n";

    outfile << "Configuration:\n";
    outfile << "  Number of nuclides: " << Nnucl << "\n";
    outfile << "  Number of distance points: " << Np << "\n";
    outfile << "  Distance range: " << h_p[0] << " - " << h_p[Np-1] << " meters\n\n";

    // Output p_grid (distance grid)
    outfile << "----------------------------------------\n";
    outfile << "DISTANCE GRID (p_grid) [meters]\n";
    outfile << "----------------------------------------\n";
    outfile << std::scientific << std::setprecision(6);

    for (int i = 0; i < Np; ++i) {
        outfile << "p[" << std::setw(3) << i << "] = " << h_p[i] << " m";
        if (i > 0) {
            float ratio = h_p[i] / h_p[i-1];
            outfile << "  (ratio: " << std::fixed << std::setprecision(3) << ratio << ")";
        }
        outfile << "\n";
    }

    // Output dp_point table (point kernel dose rates)
    outfile << "\n----------------------------------------\n";
    outfile << "POINT KERNEL TABLE (dp_point) [(rem/h)/Ci]\n";
    outfile << "----------------------------------------\n";
    outfile << "Note: All nuclides have same values (Cs-137 approximation)\n\n";
    outfile << std::scientific << std::setprecision(6);

    // Show values for first nuclide only (since all are the same)
    outfile << "Nuclide 0 (representative for all " << Nnucl << " nuclides):\n";
    outfile << std::setw(10) << "Distance" << std::setw(20) << "Dose Rate" << std::setw(20) << "Attenuation\n";
    outfile << std::setw(10) << "[m]" << std::setw(20) << "[(rem/h)/Ci]" << std::setw(20) << "[relative]\n";

    float dp_1m = h_dp[0];  // Find value at ~1m for normalization
    for (int i = 0; i < Np; ++i) {
        if (std::abs(h_p[i] - 1.0f) < 0.1f) {
            dp_1m = h_dp[i];
            break;
        }
    }

    for (int i = 0; i < Np; ++i) {
        outfile << std::setw(10) << h_p[i]
                << std::setw(20) << h_dp[i]
                << std::setw(20) << h_dp[i] / dp_1m << "\n";
    }

    // Output df_sic (semi-infinite cloud DCF)
    outfile << "\n----------------------------------------\n";
    outfile << "SEMI-INFINITE CLOUD DCF (df_sic)\n";
    outfile << "----------------------------------------\n";
    outfile << "Note: All nuclides have same value (Cs-137 approximation)\n\n";

    outfile << "df_sic value for all nuclides: " << h_df[0] << " [(rem/s)/(Ci/m^3)]\n";

    // Physical validation checks
    outfile << "\n----------------------------------------\n";
    outfile << "PHYSICAL VALIDATION CHECKS\n";
    outfile << "----------------------------------------\n";

    // Check 1: Monotonic decrease with distance
    bool monotonic = true;
    for (int i = 1; i < Np; ++i) {
        if (h_dp[i] > h_dp[i-1]) {
            monotonic = false;
            break;
        }
    }
    outfile << "1. Monotonic decrease with distance: " << (monotonic ? "PASS" : "FAIL") << "\n";

    // Check 2: Approximate 1/r^2 behavior at close distances
    float r1 = h_p[5];
    float r2 = h_p[10];
    float dp1 = h_dp[5];
    float dp2 = h_dp[10];
    float expected_ratio = (r1 * r1) / (r2 * r2);
    float actual_ratio = dp1 / dp2;
    float deviation = std::abs(actual_ratio - expected_ratio) / expected_ratio;

    outfile << "2. 1/r^2 behavior check (r1=" << r1 << "m, r2=" << r2 << "m):\n";
    outfile << "   Expected ratio: " << expected_ratio << "\n";
    outfile << "   Actual ratio: " << actual_ratio << "\n";
    outfile << "   Deviation: " << deviation * 100 << "%\n";

    // Check 3: Attenuation at large distances
    float atten_1km = h_dp[Np * 3/4] / dp_1m;  // Attenuation at ~1km
    float atten_10km = h_dp[Np - 1] / dp_1m;   // Attenuation at max distance

    outfile << "3. Attenuation factors:\n";
    outfile << "   At ~1 km: " << atten_1km << "\n";
    outfile << "   At " << h_p[Np-1]/1000 << " km: " << atten_10km << "\n";

    outfile << "\n========================================\n";
    outfile << "END OF DEBUG OUTPUT\n";
    outfile << "========================================\n";

    outfile.close();

    std::cout << "[CLOUDSHINE:DEBUG] Table debug output written to " << filename << std::endl;
}

// Build cloudshine tables on device
// All nuclides assumed to have same properties as Cs-137
inline void build_cloudshine_tables_on_device(
    int Nnucl, int Np,
    float** d_p_grid_out,
    float** d_dp_point_out,
    float** d_df_sic_out,
    DoseTables& h_tbl,
    cudaStream_t stream = 0,
    bool debug_output = false  // Toggle for debug output
) {
    // 1. Create p_grid
    std::vector<float> h_p = make_p_grid(Np, 0.1f, 5.0e4f);

    // 2. Create dp_point: same values for all nuclides (Cs-137 approximation)
    std::vector<float> h_dp(Nnucl * Np);
    for (int i = 0; i < Np; ++i) {
        float rho = h_p[i];
        float dp  = Dp_point_of_rho(rho); // [(rem/h)/Ci]
        for (int n = 0; n < Nnucl; ++n) {
            h_dp[n * Np + i] = dp;
        }
    }

    // 3. Create df_sic (semi-infinite cloud dose conversion factors)
    // Set df_sic = D′p(1 m) * 241.2 so that DCF_pn ≈ D′p(1 m)
    float dp_at_1m = Dp_point_of_rho(1.0f);
    std::vector<float> h_df(Nnucl, dp_at_1m * 241.2f);

    // Debug output if enabled
    if (debug_output) {
        output_cloudshine_tables_to_file(h_p, h_dp, h_df, Nnucl, Np);
    }

    // 4. Allocate device memory
    float* d_p_grid   = nullptr;
    float* d_dp_point = nullptr;
    float* d_df_sic   = nullptr;

    cudaMalloc(&d_p_grid,   sizeof(float) * Np);
    cudaMalloc(&d_dp_point, sizeof(float) * Nnucl * Np);
    cudaMalloc(&d_df_sic,   sizeof(float) * Nnucl);

    // 5. Copy to device
    cudaMemcpyAsync(d_p_grid,   h_p.data(),  sizeof(float) * Np,         cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_dp_point, h_dp.data(), sizeof(float) * Nnucl * Np, cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_df_sic,   h_df.data(), sizeof(float) * Nnucl,      cudaMemcpyHostToDevice, stream);

    // 6. Fill host-side DoseTables structure
    h_tbl.p_grid   = d_p_grid;
    h_tbl.dp_point = d_dp_point;
    h_tbl.df_sic   = d_df_sic;
    h_tbl.Np       = Np;

    // Output pointers for cleanup later
    if (d_p_grid_out)   *d_p_grid_out   = d_p_grid;
    if (d_dp_point_out) *d_dp_point_out = d_dp_point;
    if (d_df_sic_out)   *d_df_sic_out   = d_df_sic;

    std::cout << "[CLOUDSHINE:002] Initialized dose tables - Np=" << Np
              << ", Nnucl=" << Nnucl
              << ", p_range=[" << h_p[0] << "," << h_p[Np-1] << "]m" << std::endl;
}

// Free cloudshine tables from device memory
inline void free_cloudshine_tables_on_device(float* d_p_grid, float* d_dp_point, float* d_df_sic) {
    if (d_p_grid)   cudaFree(d_p_grid);
    if (d_dp_point) cudaFree(d_dp_point);
    if (d_df_sic)   cudaFree(d_df_sic);
}

/**
 * Main RCAP simulation loop with evacuation and exposure calculation
 *
 * This is the primary simulation engine for nuclear accident consequence analysis.
 * Combines atmospheric dispersion, ground deposition, evacuation dynamics, and
 * radiation exposure calculations in a coupled simulation.
 *
 * @param SC SimulationControl - Grid configuration (radial/angular)
 * @param EP EvacuationData - Evacuation scenarios and protection factors
 * @param RT RadioNuclideTransport - Radionuclide transport parameters
 * @param ND NuclideData - Decay chains and dose coefficients
 * @param d_ND Device pointer to nuclide data
 * @param dPF Device pointer to protection factors
 * @param dEP Device pointer to evacuation data
 * @param input_num Simulation input case number
 *
 * Simulation Pipeline (per timestep):
 *   1. update_puff_flags2              - Activate puffs based on release time
 *   2. move_puffs_by_wind_RCAP2        - Transport with deposition
 *   3. update_evac_velocity            - Update evacuee speeds
 *   4. evacuation_calculation_1D       - Move evacuees, track doses
 *   5. ComputeExposureHmix             - Calculate radiation exposure
 *
 * CUDA Kernel Configuration:
 *   - Puff transport: 2D grid (numSims x totalpuff_per_Sim)
 *   - Evacuation: 1D grid with 256 threads per block
 *   - Exposure: 2D grid (numEvacuees/numSims x numSims)
 *   - Shared memory: Used for exposure reduction
 *
 * Memory Flow:
 *   - Puff data, evacuees, and ground deposition remain on device
 *   - Periodic output via binary files
 *   - Evacuee velocities updated via host-device transfers
 *
 * Output:
 *   - Puff positions and concentrations
 *   - Evacuee positions and cumulative doses
 *   - Ground deposition fields
 */
void Gpuff::time_update_RCAP2(const SimulationControl& SC, const EvacuationData& EP,
    const std::vector<RadioNuclideTransport>& RT, const std::vector<NuclideData>& ND, NuclideData* d_ND, const ProtectionFactors* dPF, const EvacuationData* dEP, int input_num, const WeatherSamplingData& WD) {

    std::cout << "\n[TIME_UPDATE:001] ==== ENTERING time_update_RCAP2() ====" << std::endl;

    cudaError_t err = cudaGetLastError();

    float current_time = 0.0f;
    int threads_per_block = 256;
    int blocks;

    int timestep = 0;

    cudaEvent_t start, stop;
    float timesum = 0;

    // Allocate device memory for cloudshine kernel structures (outside the loop)
    DoseTables h_tbl = {};  // Will be properly initialized with table data

    // Initialize cloudshine dose tables
    float* d_p_grid = nullptr;
    float* d_dp_point = nullptr;
    float* d_df_sic = nullptr;

    const int Nnucl_tbl = MAX_NUCLIDES;  // Use MAX_NUCLIDES for consistency
    const int Np = 200;  // Number of distance grid points (can be adjusted for performance)

    // Toggle for debug output - set to true to enable table output to file
    const bool ENABLE_TABLE_DEBUG_OUTPUT = true;  // Change to false to disable output

    build_cloudshine_tables_on_device(
        Nnucl_tbl, Np,
        &d_p_grid, &d_dp_point, &d_df_sic,
        h_tbl,
        0,  // default stream
        ENABLE_TABLE_DEBUG_OUTPUT  // Enable/disable debug output
    );

    // Initialize Puff720 geometry with 720 representative points
    Puff720 h_geom720 = {};
    build_geom720_normalized(h_geom720);  // Fill with normalized coordinates

    DoseTables* d_tbl;
    Puff720* d_geom720;
    cudaMalloc(&d_tbl, sizeof(DoseTables));
    cudaMalloc(&d_geom720, sizeof(Puff720));

    // Copy structures to device (now h_tbl has valid pointers)
    cudaMemcpy(d_tbl, &h_tbl, sizeof(DoseTables), cudaMemcpyHostToDevice);
    cudaMemcpy(d_geom720, &h_geom720, sizeof(Puff720), cudaMemcpyHostToDevice);

    std::cout << "[TIME_UPDATE:002] Starting simulation loop (time_end=" << time_end << ")" << std::endl;
    while (current_time <= time_end) {


        blocks = (puffs_RCAP.size() + threads_per_block - 1) / threads_per_block;

        update_puff_flags2_kernel << <blocks, threads_per_block >> >
            (d_puffs_RCAP, current_time);
        cudaDeviceSynchronize();

        move_puffs_by_wind_RCAP2 << <numSims, totalpuff_per_Sim >> >
            (d_puffs_RCAP, d_Vdepo, d_particleSizeDistr, EP.EP_endRing, d_ground_deposit,
                d_ND, d_radius, SC.numRad, SC.numTheta);
        cudaDeviceSynchronize();

        update_evac_velocity(EP, current_time);

        blocks = (evacuees.size() + threads_per_block - 1) / threads_per_block;

        evacuation_calculation_1D << <blocks, threads_per_block >> >
            (d_puffs_RCAP, d_dir, d_evacuees, d_radius, SC.numRad, SC.numTheta,
                evacuees.size(), EP.evaEndRing, EP.EP_endRing, d_ground_deposit, dEP, current_time);
        cudaDeviceSynchronize();

        int num_evacuees = evacuees.size();

        // Configure 2D grid for exposure calculation kernel
        // Block dimension: One thread per puff in a simulation
        // Grid dimension: (evacuees per sim) x (number of simulations)
        int block_size = totalpuff_per_Sim;
        dim3 block_dim(block_size);
        dim3 grid_dim(num_evacuees / numSims, numSims);

        // Shared memory for partial exposure sums (2 values per thread)
        size_t shared_mem_size = sizeof(float) * block_size * 2;

        // Calculate radiation exposure for all evacuees
        // Uses Gaussian puff model with shared memory reduction
        ComputeExposureHmix << <grid_dim, block_dim, shared_mem_size >> > (
            d_puffs_RCAP,
            d_evacuees,
            d_exposure,
            dPF
            );

        // Calculate cloudshine dose separately
        // TEMPORARILY DISABLED - Using ComputeExposureHmix for cloudshine instead
        // This kernel computes external gamma dose from airborne radioactive material
        // Each evacuee can receive cloudshine from multiple puffs
        // Weather scenarios (simIdx) are processed independently
        // Need space for both dose reduction and mode tracking
        /*
        size_t cloudshine_shared_mem_size = sizeof(float) * block_size + sizeof(int) * block_size;

        int Nnucl = MAX_NUCLIDES;

        // Pass building height and mixing height from parsed input data
        float build_height = RT[0].build_height;  // From RT215
        float mix_height = WD.mixHeight;          // From RT350

        ComputeCloudshineDose << <grid_dim, block_dim, cloudshine_shared_mem_size >> > (
            d_puffs_RCAP,
            d_evacuees,
            dPF,
            d_tbl,
            d_geom720,
            Nnucl,
            build_height,
            mix_height
            );
        */

        err = cudaGetLastError();
        if (err != cudaSuccess)
            printf("CUDA error: %s, timestep = %d\n", cudaGetErrorString(err), timestep);

        current_time += dt;
        timestep++;

        if (timestep % freq_output == 0) {

            puff_output_binary_RCAP(timestep);
            evac_output_binary_RCAP(timestep);

            cudaDeviceSynchronize();

        }

    }

    // Free allocated device memory for cloudshine structures
    // TEMPORARILY DISABLED - Not using ComputeCloudshineDose kernel
    /*
    free_cloudshine_tables_on_device(d_p_grid, d_dp_point, d_df_sic);
    cudaFree(d_tbl);
    cudaFree(d_geom720);
    */

    std::cout << "Total sum execution time: " << timesum << " ms" << std::endl;

}

/**
 * CPU-only version of RCAP simulation (for testing/validation)
 *
 * Executes same simulation logic as time_update_RCAP2 but entirely on CPU.
 * Used for verification and debugging of GPU kernels.
 *
 * @param SC SimulationControl - Grid configuration
 * @param EP EvacuationData - Evacuation parameters
 * @param RT RadioNuclideTransport - Transport parameters
 * @param ND NuclideData - Nuclide properties
 * @param d_ND Device nuclide data (unused in CPU version)
 * @param dPF Device protection factors (unused)
 * @param input_num Simulation case number
 * @param ED EvacuationDirections - Direction data
 * @param PF ProtectionFactors - Shielding factors
 *
 * Execution: Pure CPU implementation, no GPU memory transfers
 */
void Gpuff::time_update_RCAP_cpu(const SimulationControl& SC, const EvacuationData& EP,
    const std::vector<RadioNuclideTransport>& RT, const std::vector<NuclideData>& ND,
    NuclideData* d_ND, const ProtectionFactors* dPF, int input_num, EvacuationDirections ED, ProtectionFactors PF) {

    cudaError_t err = cudaGetLastError();

    float current_time = 0.0f;
    int threads_per_block = 256;
    int blocks;

    int timestep = 0;

    cudaEvent_t start, stop;
    float timesum = 0;

    while (current_time <= time_end) {

        update_puff_flags2_cpu(current_time, puffs_RCAP.size());
        move_puffs_by_wind_RCAP2_cpu
            (EP.EP_endRing, ND, SC.ir_distances, SC.numRad, SC.numTheta, puffs_RCAP.size());

        ComputeExposureHmix_cpu(evacuees, PF, numSims, 97, totalpuff_per_Sim);

        current_time += dt;
        timestep++;

        if (timestep % freq_output == 0) {

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

        update_puff_flags_kernel<<<blocks, threadsPerBlock>>>
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

// ============================================================================
// GPU Memory Management - RCAP-Specific Data Structures
// ============================================================================

/**
 * Allocates GPU memory and copies RCAP puff data to device
 *
 * Transfers Puffcenter_RCAP structures containing:
 *   - Position (x, y, z) in polar coordinates
 *   - Multi-nuclide concentrations
 *   - Dispersion parameters (sigma_h, sigma_v)
 *   - Deposition tracking arrays
 *
 * Memory Flow:
 *   - Allocates d_puffs_RCAP on device
 *   - Copies from puffs_RCAP (host) to d_puffs_RCAP (device)
 *
 * Memory Ownership:
 *   - Device pointer managed by class
 *   - Must call free_puffs_RCAP_device_memory() before destruction
 */
void Gpuff::allocate_and_copy_puffs_RCAP_to_device() {
    std::cout << "[PUFF_COPY:001] Entering allocate_and_copy_puffs_RCAP_to_device" << std::endl;
    std::cout << "[PUFF_COPY:002] puffs_RCAP.size() = " << puffs_RCAP.size() << std::endl;

    size_t size = puffs_RCAP.size() * sizeof(Puffcenter_RCAP);
    std::cout << "[PUFF_COPY:003] Allocating " << size << " bytes on device" << std::endl;

    cudaMalloc(&d_puffs_RCAP, size);
    std::cout << "[PUFF_COPY:004] Device memory allocated" << std::endl;

    cudaMemcpy(d_puffs_RCAP, puffs_RCAP.data(), size, cudaMemcpyHostToDevice);
    std::cout << "[PUFF_COPY:005] Data copied to device" << std::endl;

    // Print first few puff data to check for unexpected output
    std::cout << "[PUFF_COPY:006] First puff data check:" << std::endl;
    if(puffs_RCAP.size() > 0) {
        for(int i = 0; i < std::min(5, (int)puffs_RCAP.size()); i++) {
            std::cout << "  Puff[" << i << "].x = " << puffs_RCAP[i].x << std::endl;
        }
    }
    std::cout << "[PUFF_COPY:007] Exiting function" << std::endl;
}

/**
 * Allocates GPU memory and copies evacuee data to device
 *
 * Transfers Evacuee structures containing:
 *   - Position (radial index, angular index)
 *   - Speed and direction parameters
 *   - Cumulative dose arrays
 *   - Population count
 *
 * Memory Flow:
 *   - Allocates d_evacuees on device
 *   - Copies from evacuees (host) to d_evacuees (device)
 *   - Device data updated each timestep during simulation
 */
void Gpuff::allocate_and_copy_evacuees_to_device() {
    cudaMalloc(&d_evacuees, evacuees.size() * sizeof(Evacuee));
    cudaMemcpy(d_evacuees, evacuees.data(), evacuees.size() * sizeof(Evacuee), cudaMemcpyHostToDevice);
}

/**
 * Allocates GPU memory and copies radial grid distances to device
 *
 * Transfers radial distance array used for polar coordinate calculations.
 * These distances define the radial rings in the polar grid system.
 *
 * @param SC SimulationControl containing ir_distances array
 *
 * Memory Flow:
 *   - Allocates d_radius on device
 *   - Copies SC.ir_distances (host) to d_radius (device)
 */
void Gpuff::allocate_and_copy_radius_to_device(SimulationControl SC) {
    // Removed debug output that was causing terminal spam
    cudaMalloc(&d_radius, SC.numRad * sizeof(float));
    cudaMemcpy(d_radius, SC.ir_distances, SC.numRad * sizeof(float), cudaMemcpyHostToDevice);
}

/**
 * Frees GPU memory allocated for RCAP puff data
 *
 * Safe deallocation with nullptr check and pointer reset.
 * Called at end of simulation or before reallocating.
 *
 * Memory Safety:
 *   - Checks for nullptr before freeing
 *   - Sets pointer to nullptr after free
 */
void Gpuff::free_puffs_RCAP_device_memory() {
    if (d_puffs_RCAP != nullptr) {
        cudaFree(d_puffs_RCAP);
        d_puffs_RCAP = nullptr;
    }
}

// ============================================================================
// Health Effects Calculation - Post-Processing
// ============================================================================

/**
 * Calculates health effects from evacuee radiation exposure
 *
 * Post-processing function that computes deterministic and stochastic health
 * effects based on cumulative doses received by evacuees during simulation.
 *
 * @param evacuees Vector of evacuees with cumulative dose data
 * @param HE HealthEffect structure with risk coefficients and thresholds
 *
 * Health Effect Categories:
 *
 * Deterministic Effects (threshold-based):
 *   1. HematopoieticSyndrome - Bone marrow damage
 *   2. PulmonarySyndrome - Lung damage
 *   3. Prodromal_Vomit - Early radiation sickness
 *   4. Diarrhea - GI tract damage
 *   5. Pneumonitis - Lung inflammation
 *   6. Thyroiditis - Thyroid inflammation
 *   7. Hypothyroidism - Thyroid dysfunction
 *
 * Stochastic Effects (linear no-threshold):
 *   8. Leukemia (incidence and mortality)
 *   9. Bone cancer (incidence and mortality)
 *   10. Breast cancer (incidence and mortality)
 *   11. Lung cancer (incidence and mortality)
 *   12. Thyroid cancer (incidence and mortality)
 *   13. GI cancer (incidence and mortality)
 *   14. Other cancers (incidence and mortality)
 *
 * Risk Calculation:
 *   - Deterministic: Probit model with threshold
 *   - Stochastic: Linear-quadratic dose response with DDRF
 *
 * Output:
 *   - CSV file: health_effect_output.csv
 *   - Contains risk by location and effect type
 *
 * Execution: CPU-side post-processing after simulation
 */
void Gpuff::health_effect(std::vector<Evacuee>& evacuees, HealthEffect HE) {

    std::vector<ResultLine> results;
    results.reserve(evacuees.size() * 14);

    float risk_deterministic = 0.0f;
    float risk_stochastic_cf = 0.0f;
    float risk_stochastic_ci = 0.0f;
    float dose = 0.0f;

    float H = 0.0;
    float npeople = 0.0;


    for (int eva_idx = 0; eva_idx < (int)evacuees.size(); eva_idx++)
    {
        int met_val = evacuees[eva_idx].met_idx;
        int rad_val = evacuees[eva_idx].prev_rad_idx;
        int theta_val = evacuees[eva_idx].prev_theta_idx;
        float pop = evacuees[eva_idx].population;

        // Deterministic Effects (threshold-based)

        // 1) HematopoieticSyndrome (RED_MARR: 3)
        {
            int organ_idx = 3;
            int nuclide_idx = 0;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.threshold_AF[nuclide_idx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nuclide_idx], HE.beta_f[nuclide_idx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "deterministic",
                        "HematopoieticSyndrome",
                        npeople
                        });
                }
            }
        }

        // 2) PulmonarySyndrome (LUNGS: 2)
        {
            int organ_idx = 2;
            int nuclide_idx = 1;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.threshold_AF[nuclide_idx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nuclide_idx], HE.beta_f[nuclide_idx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "deterministic",
                        "PulmonarySyndrome",
                        npeople
                        });
                }
            }
        }

        // 3) Prodromal_Vomit (STOMACH: 0)
        {
            int organ_idx = 0;
            int nuclide_idx = 0;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.threshold_AF[nuclide_idx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nuclide_idx], HE.beta_f[nuclide_idx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "deterministic",
                        "Prodromal_Vomit",
                        npeople
                        });
                }
            }
        }

        // 4) Diarrhea (STOMACH: 0)
        {
            int organ_idx = 0;
            int nuclide_idx = 1;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.threshold_AF[nuclide_idx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nuclide_idx], HE.beta_f[nuclide_idx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "deterministic",
                        "Diarrhea",
                        npeople
                        });
                }
            }
        }

        // 5) Pneumonitis (LUNGS: 2)
        {
            int organ_idx = 2;
            int nuclide_idx = 2;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.threshold_AF[nuclide_idx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nuclide_idx], HE.beta_f[nuclide_idx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "deterministic",
                        "Pneumonitis",
                        npeople
                        });
                }
            }
        }

        // 6) Thyroiditis (THYROIDH: 11)
        {
            int organ_idx = 11;
            int nuclide_idx = 5;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.threshold_AF[nuclide_idx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nuclide_idx], HE.beta_f[nuclide_idx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "deterministic",
                        "Thyroiditis",
                        npeople
                        });
                }
            }
        }

        // 7) Hypothyrodism (THYROIDH: 11)
        {
            int organ_idx = 11;
            int nuclide_idx = 6;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.threshold_AF[nuclide_idx]) {
                H = std::log(2.0f) * std::pow(dose / HE.alpha_f[nuclide_idx], HE.beta_f[nuclide_idx]);
                risk_deterministic = 1.0f - std::exp(-H);
                npeople = risk_deterministic * pop;

                if (npeople > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "deterministic",
                        "Hypothyrodism",
                        npeople
                        });
                }
            }
        }

        // Stochastic Effects (Linear No-Threshold model)
        // _cf: Cancer fatality (mortality)
        // _ci: Cancer incidence

        // 8) Leukemia (RED_MARR: 3)
        {
            int organ_idx = 3;
            int nuclide_idx = 0;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.LNT_threshold[nuclide_idx]) {
                risk_stochastic_cf = HE.cf_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);
                risk_stochastic_ci = HE.ci_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_cf /= HE.ddrf[nuclide_idx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_ci /= HE.ddrf[nuclide_idx];
                }

                float npeople_cf = HE.sus_frac[nuclide_idx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nuclide_idx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_cf",
                        "Leukemia_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_ci",
                        "Leukemia_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 9) Bone (BONE_SUR: 6)
        {
            int organ_idx = 6;
            int nuclide_idx = 1;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.LNT_threshold[nuclide_idx]) {
                risk_stochastic_cf = HE.cf_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);
                risk_stochastic_ci = HE.ci_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_cf /= HE.ddrf[nuclide_idx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_ci /= HE.ddrf[nuclide_idx];
                }

                float npeople_cf = HE.sus_frac[nuclide_idx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nuclide_idx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_cf",
                        "BoneCancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_ci",
                        "BoneCancer_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 10) Breast (BREAST: 7)
        {
            int organ_idx = 7;
            int nuclide_idx = 2;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.LNT_threshold[nuclide_idx]) {
                risk_stochastic_cf = HE.cf_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);
                risk_stochastic_ci = HE.ci_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_cf /= HE.ddrf[nuclide_idx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_ci /= HE.ddrf[nuclide_idx];
                }

                float npeople_cf = HE.sus_frac[nuclide_idx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nuclide_idx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_cf",
                        "BreastCancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_ci",
                        "BreastCancer_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 11) Lung (LUNGS: 2)
        {
            int organ_idx = 2;
            int nuclide_idx = 3;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.LNT_threshold[nuclide_idx]) {
                risk_stochastic_cf = HE.cf_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);
                risk_stochastic_ci = HE.ci_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_cf /= HE.ddrf[nuclide_idx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_ci /= HE.ddrf[nuclide_idx];
                }

                float npeople_cf = HE.sus_frac[nuclide_idx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nuclide_idx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_cf",
                        "LungCancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_ci",
                        "LungCancer_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 12) Thyroid (THYROIDH: 11)
        {
            int organ_idx = 11;
            int nuclide_idx = 4;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.LNT_threshold[nuclide_idx]) {
                risk_stochastic_cf = HE.cf_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);
                risk_stochastic_ci = HE.ci_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_cf /= HE.ddrf[nuclide_idx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_ci /= HE.ddrf[nuclide_idx];
                }

                float npeople_cf = HE.sus_frac[nuclide_idx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nuclide_idx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_cf",
                        "ThyroidCancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_ci",
                        "ThyroidCancer_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 13) GI (LOWER_LI: 5)
        {
            int organ_idx = 5;
            int nuclide_idx = 5;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.LNT_threshold[nuclide_idx]) {
                risk_stochastic_cf = HE.cf_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);
                risk_stochastic_ci = HE.ci_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_cf /= HE.ddrf[nuclide_idx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_ci /= HE.ddrf[nuclide_idx];
                }

                float npeople_cf = HE.sus_frac[nuclide_idx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nuclide_idx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_cf",
                        "GICancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_ci",
                        "GICancer_mortality",
                        npeople_ci
                        });
                }
            }
        }

        // 14) Other (EDEWBODY: 10)
        {
            int organ_idx = 10;
            int nuclide_idx = 6;
            dose = evacuees[eva_idx].dose_inhalations[organ_idx]
                + evacuees[eva_idx].dose_cloudshines[organ_idx];
            if (dose > HE.LNT_threshold[nuclide_idx]) {
                risk_stochastic_cf = HE.cf_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);
                risk_stochastic_ci = HE.ci_risk[nuclide_idx]
                    * dose * (HE.dos_a[nuclide_idx] + HE.dos_b[nuclide_idx] * dose);

                if (risk_stochastic_cf < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_cf /= HE.ddrf[nuclide_idx];
                }
                if (risk_stochastic_ci < HE.dos_thres[nuclide_idx]) {
                    risk_stochastic_ci /= HE.ddrf[nuclide_idx];
                }

                float npeople_cf = HE.sus_frac[nuclide_idx] * risk_stochastic_cf * pop;
                float npeople_ci = HE.sus_frac[nuclide_idx] * risk_stochastic_ci * pop;

                if (npeople_cf > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
                        "stochastic_cf",
                        "OtherCancer_incidence",
                        npeople_cf
                        });
                }
                if (npeople_ci > 0.0f) {
                    results.push_back({
                        met_val, rad_val, theta_val,
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
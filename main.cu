/**
 * GPUFF-RCAPv3 Main Entry Point
 *
 * This is the main driver for the Gaussian Puff atmospheric dispersion model
 * integrated with RCAP (Radiological Consequence Assessment Package).
 * Simulates radioactive puff transport, deposition, and exposure calculations.
 */

#include "gpuff.cuh"

int main() {
    Gpuff gpuff;

    // Initialize nuclide data structures (host and device)
    std::vector<NuclideData> ND(MAX_NUCLIDES);
    for (auto& nuclide : ND) {
        initializeNuclideData(&nuclide);
    }

    NuclideData* d_ND;
    cudaMalloc(&d_ND, MAX_NUCLIDES * sizeof(NuclideData));

    // Initialize simulation configuration structures
    SimulationControl SC;
    std::vector<RadioNuclideTransport> RT(1);
    RadioNuclideTransport RT_old = RT[0];
    WeatherSamplingData WD;
    EvacuationData EP;
    EvacuationData* dEP;
    EvacuationDirections ED;
    SiteData SD;
    ProtectionFactors PF;
    ProtectionFactors* dPF;
    HealthEffect HE;

    // Read nuclear data libraries
    read_MACCS_DCF_New2(".\\input\\RCAPdata\\MACCS_DCF_New2.LIB", ND);
    read_MACCS60_NDL(".\\input\\RCAPdata\\MACCS60.NDL", ND);

    if (CHECK_NDL) {
        print_MACCS60_NDL(ND);
    }

    std::cout << "Size of NuclideData: " << sizeof(NuclideData) << std::endl;

    // Flatten exposure data for GPU transfer
    for (int i = 0; i < MAX_NUCLIDES; i++) {
        for (int j = 0; j < MAX_ORGANS; j++) {
            for (int k = 0; k < DATA_FIELDS; k++) {
                exposure_data_all[i * MAX_ORGANS * DATA_FIELDS + j * DATA_FIELDS + k] =
                    ND[i].exposure_data[j * DATA_FIELDS + k];
            }
        }
    }

    // Copy exposure data to device
    cudaMalloc(&d_exposure, sizeof(float) * MAX_NUCLIDES * MAX_ORGANS * DATA_FIELDS);
    cudaMemcpy(d_exposure, exposure_data_all, sizeof(float) * MAX_NUCLIDES * MAX_ORGANS * DATA_FIELDS,
               cudaMemcpyHostToDevice);

    // Parse input file list
    std::string filename = ".\\input\\RCAPdata\\Test1.inp";
    std::vector<std::string> multifiles;
    int input_num = check_input_num(filename);

    std::ifstream infile(filename);
    std::string line;
    std::string path = ".\\input\\RCAPdata\\";

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string keyword;

        if (line.empty() || line[0] == '!') {
            continue;
        }

        if (line.find("MU10") != std::string::npos) {
            iss >> keyword;
            for (int i = 0; i < input_num; i++) {
                iss >> keyword;
                multifiles.push_back(path + keyword);
            }
            break;
        }
    }

    infile.close();

    // Initialize radio transport structures for each input
    for (int i = 0; i < input_num; i++) {
        std::cout << "File " << i + 1 << " of " << input_num << " = " << multifiles[i] << std::endl;
        RT.push_back(RT_old);
    }
    std::cout << std::endl;

    // Read input files
    for (int i = 0; i < input_num; ++i) {
        if (i == 0) {
            read_input_RCAP(multifiles[i], SC, ND, RT[i], WD, EP, ED, SD, PF, HE);
        } else {
            read_input_RCAPn(i, multifiles[i], SC, ND, RT[i]);
        }
    }

    // Print configuration if debug flags are set
    if (CHECK_SC) SC.print();
    if (CHECK_RT) {
        for (int i = 0; i < input_num; ++i) {
            RT[i].print(i, input_num);
        }
    }
    if (CHECK_WD) WD.print();
    if (CHECK_EP) EP.print();
    if (CHECK_ED) ED.print();
    if (CHECK_SD) SD.print();
    if (CHECK_HE) HE.print();

    size_t size = EP.EP_endRing * SC.numTheta;

    // Copy nuclide data to device
    cudaMemcpy(d_ND, ND.data(), MAX_NUCLIDES * sizeof(NuclideData), cudaMemcpyHostToDevice);

    // Allocate and copy particle deposition velocity and size distribution
    cudaMalloc((void**)&d_Vdepo, PARTICLE_COUNT * sizeof(float));
    cudaMalloc((void**)&d_particleSizeDistr, MAX_INPUT * ELEMENT_COUNT * PARTICLE_COUNT * sizeof(float));
    cudaMemcpy(d_Vdepo, Vdepo, PARTICLE_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleSizeDistr, particleSizeDistr,
               MAX_INPUT * ELEMENT_COUNT * PARTICLE_COUNT * sizeof(float), cudaMemcpyHostToDevice);

    // Copy evacuation directions to device
    cudaMalloc(&d_dir, ED.rows * ED.cols * sizeof(int));
    cudaMemcpy(d_dir, ED.directions, ED.rows * ED.cols * sizeof(int), cudaMemcpyHostToDevice);

    // Copy protection factors to device
    cudaMalloc((void**)&dPF, sizeof(ProtectionFactors));
    cudaMemcpy(dPF, &PF, sizeof(ProtectionFactors), cudaMemcpyHostToDevice);

    // Copy evacuation data to device
    cudaMalloc((void**)&dEP, sizeof(EvacuationData));
    cudaMemcpy(dEP, &EP, sizeof(EvacuationData), cudaMemcpyHostToDevice);

    // Read meteorological data and initialize simulation
    gpuff.read_meteorological_data_RCAP2(".\\input\\RCAPdata\\METEO.inp");
    gpuff.initializePuffs(input_num, RT, ND);
    gpuff.read_simulation_config();

    std::cout << "nop = " << nop << std::endl;

    gpuff.initializeEvacuees(evacuees, SC, EP, SD);

    std::cout << "totalevacuees_per_Sim = " << totalevacuees_per_Sim << std::endl;

    // Calculate total puffs across all simulations
    for (int j = 0; j < input_num; j++) {
        totalpuff_per_Sim += RT[j].nPuffTotal;
    }
    std::cout << "totalpuff_per_Sim = " << totalpuff_per_Sim << std::endl;

    // Copy simulation parameters to device constant memory
    cudaMemcpyToSymbol(d_totalpuff_per_Sim, &totalpuff_per_Sim, sizeof(int));
    cudaMemcpyToSymbol(d_totalevacuees_per_Sim, &totalevacuees_per_Sim, sizeof(int));
    cudaMemcpyToSymbol(d_numSims, &numSims, sizeof(int));

    // Initialize ground deposition array
    ground_deposit = new float[size * MAX_NUCLIDES * totalpuff_per_Sim];
    for (int i = 0; i < size * MAX_NUCLIDES * totalpuff_per_Sim; ++i) {
        ground_deposit[i] = 0.0f;
    }

    cudaMalloc((void**)&d_ground_deposit, size * MAX_NUCLIDES * totalpuff_per_Sim * sizeof(float));
    cudaMemcpy(d_ground_deposit, ground_deposit,
               size * MAX_NUCLIDES * totalpuff_per_Sim * sizeof(float), cudaMemcpyHostToDevice);

    // Allocate and copy simulation data to device
    gpuff.allocate_and_copy_evacuees_to_device();
    gpuff.allocate_and_copy_radius_to_device(SC);
    gpuff.allocate_and_copy_puffs_RCAP_to_device();

    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);

    // Run main time integration loop on GPU
    gpuff.time_update_RCAP2(SC, EP, RT, ND, d_ND, dPF, dEP, input_num);

    // Calculate execution time
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = getExecutionTime(start, stop);
    std::cout << "Total execution time: " << elapsedTime << " ms" << std::endl;

    cudaEventDestroy(start);
    cudaEventDestroy(stop);

    // Cleanup device memory
    gpuff.free_puffs_RCAP_device_memory();
    cudaFree(d_dir);
    delete[] SC.ir_distances;

    return 0;
}
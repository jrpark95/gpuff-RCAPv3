/**
 * GPUFF-RCAPv3 Main Entry Point
 *
 * This is the main driver for the Gaussian Puff atmospheric dispersion model
 * integrated with RCAP (Radiological Consequence Assessment Package).
 * Simulates radioactive puff transport, deposition, and exposure calculations.
 */

#include "gpuff.cuh"

int main() {
    std::cout << "\n========================================" << std::endl;
    std::cout << "[MAIN] Starting GPUFF-RCAPv3..." << std::endl;
    std::cout << "========================================\n" << std::endl;

    Gpuff gpuff;
    std::cout << "[MAIN:001] Gpuff object created" << std::endl;

    // Initialize nuclide data structures (host and device)
    std::cout << "[MAIN:002] Initializing " << MAX_NUCLIDES << " nuclide data structures..." << std::endl;
    std::vector<NuclideData> ND(MAX_NUCLIDES);
    for (auto& nuclide : ND) {
        initializeNuclideData(&nuclide);
    }
    std::cout << "[MAIN:003] Nuclide initialization complete" << std::endl;

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
    std::cout << "\n[MAIN:004] Reading nuclear data libraries..." << std::endl;
    std::cout << "[MAIN:005] Reading DCF file: .\\input\\RCAPdata\\MACCS_DCF_New2.LIB" << std::endl;
    read_MACCS_DCF_New2(".\\input\\RCAPdata\\MACCS_DCF_New2.LIB", ND);
    std::cout << "[MAIN:006] DCF file read complete" << std::endl;

    std::cout << "[MAIN:007] Reading NDL file: .\\input\\RCAPdata\\MACCS60.NDL" << std::endl;
    read_MACCS60_NDL(".\\input\\RCAPdata\\MACCS60.NDL", ND);
    std::cout << "[MAIN:008] NDL file read complete" << std::endl;

    if (CHECK_NDL) {
        std::cout << "[MAIN:009] CHECK_NDL is TRUE - printing NDL data" << std::endl;
        print_MACCS60_NDL(ND);
    } else {
        std::cout << "[MAIN:009] CHECK_NDL is FALSE - skipping NDL print" << std::endl;
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
    std::cout << "\n[MAIN:010] ===== PARSING INPUT FILES =====" << std::endl;
    std::string filename = ".\\input\\RCAPdata\\Test1.inp";
    std::cout << "[MAIN:011] Opening main input file: " << filename << std::endl;

    std::vector<std::string> multifiles;
    int input_num = check_input_num(filename);
    std::cout << "[MAIN:012] Number of input scenarios detected: " << input_num << std::endl;

    std::ifstream infile(filename);
    std::string line;
    std::string path = ".\\input\\RCAPdata\\";

    std::cout << "[MAIN:013] Reading input file line by line..." << std::endl;
    int lineCount = 0;
    while (std::getline(infile, line)) {
        lineCount++;
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
    std::cout << "\n[MAIN:020] ===== READING DETAILED INPUT FILES =====" << std::endl;
    for (int i = 0; i < input_num; ++i) {
        if (i == 0) {
            std::cout << "[MAIN:021] Reading primary input file: " << multifiles[i] << std::endl;
            std::cout << "[MAIN:022] --> Calling read_input_RCAP() for file #1..." << std::endl;
            read_input_RCAP(multifiles[i], SC, ND, RT[i], WD, EP, ED, SD, PF, HE);
            std::cout << "[MAIN:023] <-- Returned from read_input_RCAP()" << std::endl;
        } else {
            std::cout << "[MAIN:024] Reading secondary input file #" << i+1 << ": " << multifiles[i] << std::endl;
            std::cout << "[MAIN:025] --> Calling read_input_RCAPn()..." << std::endl;
            read_input_RCAPn(i, multifiles[i], SC, ND, RT[i]);
            std::cout << "[MAIN:026] <-- Returned from read_input_RCAPn()" << std::endl;
        }
    }
    std::cout << "[MAIN:027] All input files processed" << std::endl;

    // Print configuration if debug flags are set
    std::cout << "\n[MAIN:028] ===== CONFIGURATION PRINTING SECTION =====" << std::endl;
    // Force disable all printing regardless of CHECK flags
    const bool FORCE_DISABLE_PRINT = true;

    if (!FORCE_DISABLE_PRINT && CHECK_SC) {
        std::cout << "[MAIN:029] CHECK_SC is TRUE, calling SC.print()" << std::endl;
        SC.print();
    }
    if (!FORCE_DISABLE_PRINT && CHECK_RT) {
        std::cout << "[MAIN:030] CHECK_RT is TRUE, calling RT.print()" << std::endl;
        for (int i = 0; i < input_num; ++i) {
            RT[i].print(i, input_num);
        }
    }
    if (!FORCE_DISABLE_PRINT && CHECK_WD) {
        std::cout << "[MAIN:031] CHECK_WD is TRUE, calling WD.print()" << std::endl;
        WD.print();
    }
    if (!FORCE_DISABLE_PRINT && CHECK_EP) {
        std::cout << "[MAIN:032] CHECK_EP is TRUE, calling EP.print()" << std::endl;
        EP.print();
    }
    if (!FORCE_DISABLE_PRINT && CHECK_ED) {
        std::cout << "[MAIN:033] CHECK_ED is TRUE, calling ED.print()" << std::endl;
        ED.print();
    }
    if (!FORCE_DISABLE_PRINT && CHECK_SD) {
        std::cout << "[MAIN:034] CHECK_SD is TRUE, calling SD.print()" << std::endl;
        SD.print();
    }
    if (!FORCE_DISABLE_PRINT && CHECK_HE) {
        std::cout << "[MAIN:035] CHECK_HE is TRUE, calling HE.print()" << std::endl;
        HE.print();
    }
    std::cout << "[MAIN:036] Configuration printing section complete (FORCE_DISABLE=" << FORCE_DISABLE_PRINT << ")" << std::endl;

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
    std::cout << "\n[MAIN:040] ===== METEOROLOGICAL DATA & INITIALIZATION =====" << std::endl;
    std::cout << "[MAIN:041] Reading meteorological data from METEO.inp..." << std::endl;
    gpuff.read_meteorological_data_RCAP2(".\\input\\RCAPdata\\METEO.inp");
    std::cout << "[MAIN:042] Meteorological data loaded" << std::endl;

    std::cout << "[MAIN:043] Initializing puffs..." << std::endl;
    gpuff.initializePuffs(input_num, RT, ND);
    std::cout << "[MAIN:044] Puffs initialized" << std::endl;

    std::cout << "[MAIN:045] Reading simulation config..." << std::endl;
    gpuff.read_simulation_config();
    std::cout << "[MAIN:046] Simulation config loaded" << std::endl;

    std::cout << "[MAIN:047] nop = " << nop << std::endl;

    std::cout << "[MAIN:048] Initializing evacuees..." << std::endl;
    gpuff.initializeEvacuees(evacuees, SC, EP, SD);
    std::cout << "[MAIN:049] Evacuees initialized" << std::endl;

    std::cout << "[MAIN:050] totalevacuees_per_Sim = " << totalevacuees_per_Sim << std::endl;

    // Calculate total puffs across all simulations
    for (int j = 0; j < input_num; j++) {
        totalpuff_per_Sim += RT[j].nPuffTotal;
    }
    std::cout << "totalpuff_per_Sim = " << totalpuff_per_Sim << std::endl;

    std::cout << "[MAIN:051] Copying simulation parameters to device constant memory..." << std::endl;
    // Copy simulation parameters to device constant memory
    cudaMemcpyToSymbol(d_totalpuff_per_Sim, &totalpuff_per_Sim, sizeof(int));
    cudaMemcpyToSymbol(d_totalevacuees_per_Sim, &totalevacuees_per_Sim, sizeof(int));
    cudaMemcpyToSymbol(d_numSims, &numSims, sizeof(int));
    std::cout << "[MAIN:052] Device constant memory updated" << std::endl;

    std::cout << "[MAIN:053] Initializing ground deposition array..." << std::endl;
    // Initialize ground deposition array
    ground_deposit = new float[size * MAX_NUCLIDES * totalpuff_per_Sim];
    for (int i = 0; i < size * MAX_NUCLIDES * totalpuff_per_Sim; ++i) {
        ground_deposit[i] = 0.0f;
    }
    std::cout << "[MAIN:054] Ground deposition array initialized" << std::endl;

    std::cout << "[MAIN:055] Allocating GPU memory for ground deposition..." << std::endl;
    cudaMalloc((void**)&d_ground_deposit, size * MAX_NUCLIDES * totalpuff_per_Sim * sizeof(float));
    cudaMemcpy(d_ground_deposit, ground_deposit,
               size * MAX_NUCLIDES * totalpuff_per_Sim * sizeof(float), cudaMemcpyHostToDevice);
    std::cout << "[MAIN:056] GPU memory for ground deposition allocated" << std::endl;

    std::cout << "[MAIN:057] Allocating and copying evacuees to device..." << std::endl;
    // Allocate and copy simulation data to device
    gpuff.allocate_and_copy_evacuees_to_device();
    std::cout << "[MAIN:058] Evacuees copied to device" << std::endl;

    std::cout << "[MAIN:059] Allocating and copying radius to device..." << std::endl;
    gpuff.allocate_and_copy_radius_to_device(SC);
    std::cout << "[MAIN:060] Radius copied to device" << std::endl;

    std::cout << "[MAIN:061] Allocating and copying puffs to device..." << std::endl;
    gpuff.allocate_and_copy_puffs_RCAP_to_device();
    std::cout << "[MAIN:062] Puffs copied to device" << std::endl;

    std::cout << "[MAIN:063] Creating CUDA events for timing..." << std::endl;
    // Create CUDA events for timing
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    cudaEventRecord(start);
    std::cout << "[MAIN:064] CUDA events created" << std::endl;

    std::cout << "[MAIN:065] >>> Calling time_update_RCAP2()..." << std::endl;
    // Run main time integration loop on GPU
    gpuff.time_update_RCAP2(SC, EP, RT, ND, d_ND, dPF, dEP, input_num);
    std::cout << "[MAIN:066] <<< Returned from time_update_RCAP2()" << std::endl;

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
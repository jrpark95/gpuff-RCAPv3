#include "gpuff.cuh"
 
int main() {

    //double a = 1.0f;
    //double b = 1.0e-15;

    //printf("%1.18f\n", a + b);

    Gpuff gpuff;

    std::vector<NuclideData> ND(MAX_NUCLIDES);
    for (auto& nuclide : ND) initializeNuclideData(&nuclide); 

    NuclideData* d_ND; 
    cudaMalloc(&d_ND, MAX_NUCLIDES * sizeof(NuclideData));

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
    

    read_MACCS_DCF_New2(".\\input\\RCAPdata\\MACCS_DCF_New2.LIB", ND);
    //read_MACCS_DCF_New2("./input/RCAPdata/MACCS_DCF_New2.LIB", ND);

    read_MACCS60_NDL(".\\input\\RCAPdata\\MACCS60.NDL", ND);
    //read_MACCS60_NDL("./input/RCAPdata/MACCS60.NDL", ND); 
      
    if (CHECK_NDL) print_MACCS60_NDL(ND);
    
    std::cout << "Size of NuclideData: " << sizeof(NuclideData) << std::endl; 

    //float exposure_data_all[MAX_NUCLIDES * MAX_ORGANS * DATA_FIELDS]; 
     
    for (int i = 0; i < MAX_NUCLIDES; i++) 
        for (int j = 0; j < MAX_ORGANS; j++)
            for (int k = 0; k < DATA_FIELDS; k++) {
                exposure_data_all[i * MAX_ORGANS * DATA_FIELDS + j * DATA_FIELDS + k] = ND[i].exposure_data[j * DATA_FIELDS + k];
                //std::cout << "exp["<<i<<"]["<<j<<"]["<<k<<"] = " << exposure_data_all[i * MAX_ORGANS * DATA_FIELDS + j * DATA_FIELDS + k] << std::endl;
            }

    cudaMalloc(&d_exposure, sizeof(float) * MAX_NUCLIDES * MAX_ORGANS * DATA_FIELDS);
    cudaMemcpy(d_exposure, exposure_data_all, sizeof(float) * MAX_NUCLIDES * MAX_ORGANS * DATA_FIELDS, cudaMemcpyHostToDevice);

    std::string filename = ".\\input\\RCAPdata\\Test1.inp";
    //std::string filename = "./input/RCAPdata/Test.inp";

    std::vector<std::string> multifiles;

    int input_num = check_input_num(filename); 

    std::ifstream infile(filename); 
    std::string line; 
     
    std::string path = ".\\input\\RCAPdata\\"; 
    //std::string path = "./input/RCAPdata/"; 


    while (std::getline(infile, line)){ 
        std::istringstream iss(line);
        std::string keyword;

        if (line.empty() || line[0] == '!') continue;

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
     
    for (int i = 0; i < input_num; i++) {
        std::cout << "File " << i + 1 << " of " << input_num << " = " << multifiles[i] << std::endl;
        RT.push_back(RT_old);
    }
    std::cout << std::endl; 


    for (int i = 0; i < input_num; ++i) {  
        if (i == 0) read_input_RCAP(multifiles[i], SC, ND, RT[i], WD, EP, ED, SD, PF, HE);
        else read_input_RCAPn(i, multifiles[i], SC, ND, RT[i]);
    } 
    if (CHECK_SC) SC.print();
    if (CHECK_RT) for (int i = 0; i < input_num; ++i) RT[i].print(i, input_num);
    if (CHECK_WD) WD.print();
    if (CHECK_EP) EP.print();
    if (CHECK_ED) ED.print();
    if (CHECK_SD) SD.print();
    if (CHECK_HE) HE.print();

    size_t size = EP.EP_endRing * SC.numTheta;  

    cudaMemcpy(d_ND, ND.data(), MAX_NUCLIDES * sizeof(NuclideData), cudaMemcpyHostToDevice);
    //printNuclideData << <1, 1 >> > (d_ND);

    cudaMalloc((void**)&d_Vdepo, PARTICLE_COUNT * sizeof(float)); 
    cudaMalloc((void**)&d_particleSizeDistr, MAX_INPUT * ELEMENT_COUNT * PARTICLE_COUNT * sizeof(float));

    cudaMemcpy(d_Vdepo, Vdepo, PARTICLE_COUNT * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_particleSizeDistr, particleSizeDistr, MAX_INPUT * ELEMENT_COUNT * PARTICLE_COUNT * sizeof(float), cudaMemcpyHostToDevice);
     

    cudaMalloc(&d_dir, ED.rows * ED.cols * sizeof(int));  
    cudaMemcpy(d_dir, ED.directions, ED.rows * ED.cols * sizeof(int), cudaMemcpyHostToDevice);
     
    cudaMalloc((void**)&dPF, sizeof(ProtectionFactors)); 
    cudaMemcpy(dPF, &PF, sizeof(ProtectionFactors), cudaMemcpyHostToDevice);
      
    cudaMalloc((void**)&dEP, sizeof(EvacuationData)); 
    cudaMemcpy(dEP, &EP, sizeof(EvacuationData), cudaMemcpyHostToDevice);
    
    gpuff.read_meteorological_data_RCAP2(".\\input\\RCAPdata\\METEO.inp");
    //gpuff.read_meteorological_data_RCAP2("./input/RCAPdata/METEO.inp");
     
    gpuff.initializePuffs(input_num, RT, ND);
    gpuff.read_simulation_config(); 

    std::cout << "nop = " << nop << std::endl;    
    
    gpuff.initializeEvacuees(evacuees, SC, EP, SD); 
    //gpuff.initializeEvacuees_xy(evacuees, SC, EP, SD);  
     
    std::cout << "totalevacuees_per_Sim = " << totalevacuees_per_Sim << std::endl;
    // for (const auto& evacuee : evacuees) evacuee.print();  

    for (int j = 0; j < input_num; j++) totalpuff_per_Sim += RT[j].nPuffTotal; 
    std::cout << "totalpuff_per_Sim = " << totalpuff_per_Sim << std::endl; 
      
    cudaMemcpyToSymbol(d_totalpuff_per_Sim, &totalpuff_per_Sim, sizeof(int)); 
    cudaMemcpyToSymbol(d_totalevacuees_per_Sim, &totalevacuees_per_Sim, sizeof(int));
    cudaMemcpyToSymbol(d_numSims, &numSims, sizeof(int)); 
     
    ground_deposit = new float[size * MAX_NUCLIDES * totalpuff_per_Sim];
    for (int i = 0; i < size * MAX_NUCLIDES * totalpuff_per_Sim; ++i) ground_deposit[i] = 0.0f;
    
    cudaMalloc((void**)&d_ground_deposit, size * MAX_NUCLIDES * totalpuff_per_Sim * sizeof(float));
    cudaMemcpy(d_ground_deposit, ground_deposit, size * MAX_NUCLIDES * totalpuff_per_Sim * sizeof(float), cudaMemcpyHostToDevice);
      
    gpuff.allocate_and_copy_evacuees_to_device();  
    gpuff.allocate_and_copy_radius_to_device(SC); 
    gpuff.allocate_and_copy_puffs_RCAP_to_device();
     
    cudaEvent_t start, stop;
    cudaEventCreate(&start); 
    cudaEventCreate(&stop); 
    cudaEventRecord(start);  
     
    gpuff.time_update_RCAP2(SC, EP, RT, ND, d_ND, dPF, dEP, input_num); 
    
    
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    float elapsedTime = getExecutionTime(start, stop);
    std::cout << "Total execution time: " << elapsedTime << " ms" << std::endl;
     
    cudaEventDestroy(start); 
    cudaEventDestroy(stop);  
     
    //gpuff.health_effect(evacuees, HE); 

    //using namespace std::chrono; 
    //auto start = high_resolution_clock::now();

    //gpuff.time_update_RCAP_cpu(SC, EP, RT, ND, d_ND, dPF, input_num, ED, PF);

    //auto stop = high_resolution_clock::now();
    //auto duration = duration_cast<milliseconds>(stop - start); 
    //std::cout << "Total execution time: " << duration.count() << " ms" << std::endl;


    gpuff.free_puffs_RCAP_device_memory();
    cudaFree(d_dir); 
    delete[] SC.ir_distances;

    //cudaFree(d_Vdepo);
    //cudaFree(d_particleSizeDistr);  
      
    return 0; 
      
}

// Set 1 : RCAP + Polar + vdepo 

//gpuff.vdepo_initialization(); 
//gpuff.read_simulation_config();
//gpuff.read_meteorological_data_RCAP();
//gpuff.puff_initialization_RCAP();
//gpuff.allocate_and_copy_to_device();
//gpuff.time_update_RCAP();
//cudaFree(d_vdepo); 
//cudaFree(d_size);


// Set 2 : pres, unis, etas

// gpuff.read_simulation_config();
// gpuff.puff_initialization();
// gpuff.receptor_initialization_ldaps();
// gpuff.read_etas_altitudes();
// gpuff.read_meteorological_data("pres.bin", "unis.bin", "etas.bin");
// gpuff.allocate_and_copy_to_device();
// gpuff.time_update_polar();
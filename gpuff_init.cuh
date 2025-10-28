// ============================================================================
// GPUFF Initialization Module
// ============================================================================
// This file contains initialization and input file parsing functions for the
// GPUFF (Gaussian Puff) atmospheric dispersion model with radiological
// consequence assessment capabilities (RCAP).
//
// Main Responsibilities:
// - Parse MACCS nuclear library data files
// - Read dose conversion factors (DCF)
// - Initialize simulation parameters
// - Configure meteorological data structures
// - Setup radionuclide transport parameters
// ============================================================================

#include "gpuff.cuh"

// ============================================================================
// Chemical Group Classification
// ============================================================================
// Maps chemical element group names to integer codes for radionuclide
// classification. This follows the MACCS (MELCOR Accident Consequence Code
// System) categorization scheme.
//
// Chemical Groups:
//   1 - Xenon (xen)     : Noble gases
//   2 - Iodine (iod)    : Halogens
//   3 - Cesium (ces)    : Alkali metals
//   4 - Tellurium (tel) : Chalcogens
//   5 - Strontium (str) : Alkaline earth metals
//   6 - Ruthenium (rut) : Transition metals
//   7 - Lanthanum (lan) : Lanthanides
//   8 - Cerium (cer)    : Lanthanides
//   9 - Barium (bar)    : Alkaline earth metals
//   0 - Unknown/Other
// ============================================================================
int setCG(const char* group) {
    if (strncmp(group, "xen", 3) == 0) return 1;
    else if (strncmp(group, "iod", 3) == 0) return 2;
    else if (strncmp(group, "ces", 3) == 0) return 3;
    else if (strncmp(group, "tel", 3) == 0) return 4;
    else if (strncmp(group, "str", 3) == 0) return 5;
    else if (strncmp(group, "rut", 3) == 0) return 6;
    else if (strncmp(group, "lan", 3) == 0) return 7;
    else if (strncmp(group, "cer", 3) == 0) return 8;
    else if (strncmp(group, "bar", 3) == 0) return 9;
    else return 0;
}

// ============================================================================
// String Utility Functions
// ============================================================================

// Removes leading and trailing whitespace from a C-string in place
void trim(char* str) {
    char* p = str;
    while (*p == ' ') ++p;
    memmove(str, p, strlen(p) + 1);

    p = str + strlen(str) - 1;
    while (p >= str && *p == ' ') --p;
    *(p + 1) = '\0';
}

// Searches for a nuclide by name in the nuclide data vector
// Returns the index if found, -1 if not found
// Uses case-insensitive comparison
int find_nuclide_index(std::vector<NuclideData>& ND, const std::string& nuclide_name) {
    for (int i = 0; i < ND.size(); i++) {
        if (STRICMP(ND[i].name, nuclide_name.c_str()) == 0) {
            return i;
        }
    }
    return -1;
}

// ============================================================================
// Simulation Configuration Reader
// ============================================================================
// Reads basic simulation parameters from setting.txt and source.txt
//
// INPUT FILES:
// ------------
// 1. setting.txt format:
//    Time_end(s): <float>              - Total simulation duration in seconds
//    dt(s): <float>                    - Time step size in seconds
//    Plot_output_freq: <int>           - Output frequency (every N timesteps)
//    Total_number_of_puff: <int>       - Total number of puffs to simulate
//    Rural/Urban: <0/1>                - 0=Urban, 1=Rural (affects dispersion)
//    Pasquill-Gifford/Briggs-McElroy-Pooler: <0/1>
//                                      - 0=Briggs, 1=Pasquill-Gifford
//
// 2. source.txt format:
//    [SOURCE]
//    <lat> <lon> <height>              - Source location (degrees, meters)
//    ...
//
//    [SOURCE_TERM]
//    <srcnum> <decay> <depvel>         - Source term properties
//    ...                                 decay: decay constant (1/s)
//                                        depvel: dry deposition velocity (m/s)
//
//    [RELEASE_CASES]
//    <location> <sourceterm> <value>   - Release case definition
//    ...                                 location: source index (1-based)
//                                        sourceterm: term index (1-based)
//                                        value: concentration value (Bq)
// ============================================================================
void Gpuff::read_simulation_config(){

    FILE* file;
    FILE* sourceFile;

    #ifdef _WIN32
        file = fopen(".\\input\\setting.txt", "r");
        sourceFile = fopen(".\\input\\source.txt", "r");
    #else
        file = fopen("./input/setting.txt", "r");
        sourceFile = fopen("./input/source.txt", "r");
    #endif

    if (!file){
        std::cerr << "Failed to open setting.txt" << std::endl;
        exit(1);
    }

    char buffer[256];
    int tempValue;

    while (fgets(buffer, sizeof(buffer), file)){

        if (buffer[0] == '#') continue;

        if (strstr(buffer, "Time_end(s):")){
            sscanf(buffer, "Time_end(s): %f", &time_end);
        } else if (strstr(buffer, "dt(s):")){
            sscanf(buffer, "dt(s): %f", &dt);
        } else if (strstr(buffer, "Plot_output_freq:")){
            sscanf(buffer, "Plot_output_freq: %d", &freq_output);
            std::cout << "freq = " << freq_output << std::endl;
        } else if (strstr(buffer, "Total_number_of_puff:")){
            sscanf(buffer, "Total_number_of_puff: %d", &nop);
        } else if (strstr(buffer, "Rural/Urban:")){
            sscanf(buffer, "Rural/Urban: %d", &tempValue);
            isRural = tempValue;
        } else if (strstr(buffer, "Pasquill-Gifford/Briggs-McElroy-Pooler:")){
            sscanf(buffer, "Pasquill-Gifford/Briggs-McElroy-Pooler: %d", &tempValue);
            isPG = tempValue;
        }

    }

    fclose(file);

    if (!sourceFile){
        std::cerr << "Failed to open source.txt" << std::endl;
        exit(1);
    }

    while (fgets(buffer, sizeof(buffer), sourceFile)){
        if (buffer[0] == '#') continue;
    
        // SOURCE coordinates
        if (strstr(buffer, "[SOURCE]")) {
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[SOURCE_TERM]")) {
                if (buffer[0] == '#') continue;
    
                Source src;
                sscanf(buffer, "%f %f %f", &src.lat, &src.lon, &src.height);
                sources.push_back(src);
            }
            sources.pop_back();
        }
    
        // SOURCE_TERM values
        if (strstr(buffer, "[SOURCE_TERM]")){
            while (fgets(buffer, sizeof(buffer), sourceFile) && !strstr(buffer, "[RELEASE_CASES]")) {
                if (buffer[0] == '#') continue;
    
                int srcnum;
                float decay, depvel;
                sscanf(buffer, "%d %f %f", &srcnum, &decay, &depvel);
                decayConstants.push_back(decay);
                drydepositionVelocity.push_back(depvel);
            }
            decayConstants.pop_back();
            drydepositionVelocity.pop_back();
        }
    
        // RELEASE_CASES
        if (strstr(buffer, "[RELEASE_CASES]")){
            while (fgets(buffer, sizeof(buffer), sourceFile)) {
                if (buffer[0] == '#') continue;
    
                Concentration conc;
                sscanf(buffer, "%d %d %lf", &conc.location, &conc.sourceterm, &conc.value);
                concentrations.push_back(conc);
            }
        }
    }
    
    fclose(sourceFile);

    //nop = floor(nop/(sources.size()*decayConstants.size()))*sources.size()*decayConstants.size();
    nop = puffs_RCAP.size();

    cudaError_t err;

    err = cudaMemcpyToSymbol(d_time_end, &time_end, sizeof(float));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_dt, &dt, sizeof(float));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_freq_output, &freq_output, sizeof(int));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_nop, &nop, sizeof(int));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_isRural, &isRural, sizeof(bool));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));
    err = cudaMemcpyToSymbol(d_isPG, &isPG, sizeof(bool));
    if (err != cudaSuccess) printf("Error copying to symbol: %s\n", cudaGetErrorString(err));

    // for(const auto& source : sources){
    //     std::cout << source.lat << ", " << source.lon << ", " << source.height << std::endl;
    // }
    // for(float decay : decayConstants){
    //     std::cout << decay << std::endl; 
    // }
    // for(float depvel : drydepositionVelocity){
    //     std::cout << depvel << std::endl; 
    // }
    // for(const auto& conc : concentrations){
    //     std::cout << conc.location << ", " << conc.sourceterm << ", " << conc.value << std::endl;
    // }

    // std::cout << "isRural = " << isRural << std::endl;
    // std::cout << "isPG = " << isPG << std::endl;

}

// ============================================================================
// Meteorological Vertical Level Reader
// ============================================================================
// Reads vertical coordinate (eta) levels from LDAPS meteorological data files
//
// INPUT FILES:
// ------------
// hgt_uv.txt : Heights for horizontal wind components (U/V)
// hgt_w.txt  : Heights for vertical wind component (W)
//
// File Format:
// Each line contains colon-separated fields with altitude value in 5th field
// Example: field1:field2:field3:field4:<altitude>:...
//
// Units: meters above ground level (AGL)
//
// LDAPS: Local Data Assimilation and Prediction System
// Eta coordinates: Terrain-following vertical coordinate system
// ============================================================================
void Gpuff::read_etas_altitudes(){


    #ifdef _WIN32
        FILE* file = fopen(".\\input\\hgt_uv.txt", "r");
    #else
        FILE* file = fopen("./input/hgt_uv.txt", "r");
    #endif


    if (!file){
        std::cerr << "Failed to open setting.txt" << std::endl;
        exit(1);
    }

    char line[1000];
    int count = 0;
    int idx = 0;

    while(fgets(line, sizeof(line), file)){

        fgets(line, sizeof(line), file);

        char* token = strtok(line, ":");
        count = 0;
        char* val = nullptr;

        while(token){
            if (count == 4){
                val = token;
                break;
            }
            token = strtok(nullptr, ":");
            count++;
        }
        etas_hgt_uv[idx++] = atoi(strtok(val, " "));

    }

    cudaError_t err = cudaMemcpyToSymbol(d_etas_hgt_uv, etas_hgt_uv, sizeof(float) * (dimZ_etas-1));
    if (err != cudaSuccess) std::cerr << "Failed to copy data to constant memory: " << cudaGetErrorString(err) << std::endl;

    #ifdef _WIN32
        file = fopen(".\\input\\hgt_w.txt", "r");
    #else
        file = fopen("./input/hgt_w.txt", "r");
    #endif

    if (!file){
        std::cerr << "Failed to open setting.txt" << std::endl;
        exit(1);
    }

    count = 0;
    idx = 0;

    while(fgets(line, sizeof(line), file)){

        char* token = strtok(line, ":");
        count = 0;
        char* val = nullptr;

        while(token){
            if (count == 4){
                val = token;
                break;
            }
            token = strtok(nullptr, ":");
            count++;
        }
        etas_hgt_w[idx++] = atof(strtok(val, " "));
        //printf("hgt_w[%d] = %f\n", idx-1, etas_hgt_w[idx-1]);

    }

    err = cudaMemcpyToSymbol(d_etas_hgt_w, etas_hgt_w, sizeof(float) * (dimZ_etas-1));
    if (err != cudaSuccess) std::cerr << "Failed to copy data to constant memory: " << cudaGetErrorString(err) << std::endl;

}

void Gpuff::puff_initialization_val(){
    int puffsPerConc = nop / concentrations.size();

    for (const auto& conc : concentrations){
        for (int i = 0; i < puffsPerConc; ++i){
            
            float x = sources[conc.location - 1].lat;
            float y = sources[conc.location - 1].lon;
            float z = sources[conc.location - 1].height;

            puffs.push_back(Puffcenter(x, y, z, decayConstants[conc.sourceterm - 1], conc.value*time_end/nop, 
                drydepositionVelocity[conc.sourceterm - 1], i + 1, 0, 0, 0));
        }
    }

    // Sort the puffs by timeidx
    std::sort(puffs.begin(), puffs.end(), [](const Puffcenter& a, const Puffcenter& b){
        return a.timeidx < b.timeidx;
    });
}

void Gpuff::puff_initialization(){
    int puffsPerConc = nop / concentrations.size();

    for (const auto& conc : concentrations){
        for (int i = 0; i < puffsPerConc; ++i){
            
            float x = Lambert2x(
                sources[conc.location - 1].lat, 
                sources[conc.location - 1].lon);
            float y = Lambert2y(
                sources[conc.location - 1].lat, 
                sources[conc.location - 1].lon);
            float z = sources[conc.location - 1].height;

            puffs.push_back(Puffcenter(x, y, z, decayConstants[conc.sourceterm - 1], conc.value, 
                drydepositionVelocity[conc.sourceterm - 1], i + 1, 0, 0, 0));

        }
    }

    // Sort the puffs by timeidx
    std::sort(puffs.begin(), puffs.end(), [](const Puffcenter& a, const Puffcenter& b){
        return a.timeidx < b.timeidx;
    });
}


void Gpuff::puff_initialization_RCAP(){

    for (int i=0; i<nop; i++){
        int tt = floor((float)i/(float)nop*(float)time_end/3600.0);
        puffs.push_back(Puffcenter(0.0f, 0.0f, 0.0f, 0.0f, 5.0e+13, 
            0.0f, i + 1, RCAP_winvel[tt], RCAP_windir[tt], RCAP_stab[tt]));
    }

    // Sort the puffs by timeidx
    std::sort(puffs.begin(), puffs.end(), [](const Puffcenter& a, const Puffcenter& b){
        return a.timeidx < b.timeidx;
    });

    //float radii[] = {100000.0f/2.0f, 300000.0f/2.0f, 500000.0f/2.0f};
    //float radii[] = {1609.34f, 16093.40f, 80467.20f, 804672.00f};

    float angleIncrement = 22.5;

    for (float radius : radi) {
        for (int i = 0; i < 16; ++i) {
            float angle = angleIncrement * i * PI / 180.0;
            receptors.push_back(receptors_RCAP(radius * cos(angle), radius * sin(angle)));
        }
    }

    cudaMalloc(&d_receptors, receptors.size() * sizeof(receptors_RCAP));
    cudaMemcpy(d_receptors, receptors.data(), receptors.size() * sizeof(receptors_RCAP), cudaMemcpyHostToDevice);

}

void Gpuff::receptor_initialization_ldaps(){

    float radii[] = {400.0f, 1200.0f, 2000.0f};
    float angleIncrement = 22.5;

    for (float radius : radii) {
        for (int i = 0; i < 16; ++i) {
            float angle = angleIncrement * i * PI / 180.0;
            receptors.push_back(receptors_RCAP(751563.8f + radius * cos(angle), 420629.3f + radius * sin(angle)));
        }
    }

    cudaMalloc(&d_receptors, receptors.size() * sizeof(receptors_RCAP));
    cudaMemcpy(d_receptors, receptors.data(), receptors.size() * sizeof(receptors_RCAP), cudaMemcpyHostToDevice);

}

void Gpuff::vdepo_initialization(){
    
    cudaMalloc((void**)&d_vdepo, 10 * sizeof(float));
    //cudaMalloc((void **)&d_size, 9 * 10 * sizeof(float));
    cudaMalloc((void**)&d_radi, (RNUM + 1) * sizeof(float));

    cudaMemcpy(d_vdepo, vdepo, 10 * sizeof(float), cudaMemcpyHostToDevice);
    //cudaMemcpy(d_size, size, 9 * 10 * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_radi, radi, (RNUM + 1) * sizeof(float), cudaMemcpyHostToDevice);


    cudaMalloc(&d_size, 9 * sizeof(float*));
    for (int i = 0; i < 9; i++) {
        float* d_row;
        cudaMalloc(&d_row, 10 * sizeof(float));
        cudaMemcpy(d_row, size[i], 10 * sizeof(float), cudaMemcpyHostToDevice);
        cudaMemcpy(&d_size[i], &d_row, sizeof(float*), cudaMemcpyHostToDevice);
    }

}




// Helper function for RT130 parsing (dry deposition velocity by particle size)
void parse_RT130(const std::string& line) {
    std::istringstream iss(line);
    std::string keyword;
    int particle_size_id;
    float deposition_velocity_m_per_s;

    iss >> keyword >> particle_size_id >> deposition_velocity_m_per_s;

    if (particle_size_id >= 1 && particle_size_id <= PARTICLE_COUNT) {
        Vdepo[particle_size_id - 1] = deposition_velocity_m_per_s;
    }
}

// ============================================================================
// RCAP Main Input File Parser
// ============================================================================
// Reads comprehensive RCAP (Radiological Consequence Assessment Program) input
// This is the main scenario configuration file containing all parameters for
// nuclear accident consequence modeling
//
// INPUT FILE STRUCTURE (in order):
// ---------------------------------
// *Simul_Con (Simulation Control)
//   SC10: Simulation title
//   SC20: Plant info (name, power, type, location)
//   SC40: Spatial grid (coordinate system, radial rings, angular sectors)
//   SC41: Radial distances (km, converted to meters)
//   SC50: Input file names (weather, nuclide library, DCF, FCM)
//   SC90: Dose calculation flags (early/late)
//
// *RN_Trans (Radionuclide Transport)
//   RT110: Core inventory by nuclide (Bq)
//   RT120: Release class properties (wet/dry deposition)
//   RT130: Dry deposition velocity by particle size (m/s)
//   RT150: Particle size distribution by element group
//   RT200: Number of release pathways and puffs
//   RT210: Puff release parameters (time, duration, height, heat)
//   RT215: Building dimensions (height, width in meters)
//   RT220: Release fractions by chemical group
//   RT310: Weather sampling type (constant/stratified)
//   RT320: Samples per day
//   RT340: Random seed
//   RT350: Constant weather (if applicable)
//
// *EP_Sim (Emergency Planning Simulation)
//   EP200: Alarm time (seconds)
//   EP210: Evacuation rings (inner/outer)
//   EP220: Shelter delay by ring (hours)
//   EP230: Shelter duration by ring (hours)
//   EP240: Evacuation speeds and durations (m/s, hours)
//   EP250: Evacuation directions by ring/sector (F/B/L/R)
//
// *Site_Data
//   SD50: Surface roughness by sector (cm)
//   SD150: Population by ring and sector
//
// *Model_Par (Model Parameters)
//   MP130: Washout coefficients (wc1, wc2 for wet deposition)
//   MP210: Protection factors (cloudshine, groundshine, inhalation, ingestion, resuspension)
//   MP220: Resuspension parameters (coefficient, half-life)
//   MP250: Acute fatality model (organ, alpha, beta, threshold)
//   MP260: Acute morbidity model (injury types)
//   MP270: Cancer effect model (risk coefficients)
// ============================================================================
void read_input_RCAP(const std::string& filename, SimulationControl& SC,
    std::vector<NuclideData>& ND, RadioNuclideTransport& RT, WeatherSamplingData& WD,
    EvacuationData& EP, EvacuationDirections& ED, SiteData& SD, ProtectionFactors& PF, HealthEffect& HE) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    int plumeIndex = 0;

    while (std::getline(infile, line)) {

        if (line.empty() || line[0] == '!') {
            continue;
        }

        std::istringstream iss(line);
        std::string keyword;

        if (line.find("*END Simul_Con") != std::string::npos) {
            break;
        }

        if (line.find("SC10") != std::string::npos) {
            iss >> keyword >> SC.sim_title;
        }
        else if (line.find("SC20") != std::string::npos) {
            std::string longitude, latitude;
            iss >> keyword >> SC.plant_name >> SC.plant_power >> SC.plant_type >> longitude >> latitude;

            RT.lon = std::stof(longitude.substr(1)) * (longitude[0] == 'W' ? -1.0f : 1.0f);
            RT.lat = std::stof(latitude.substr(1)) * (latitude[0] == 'S' ? -1.0f : 1.0f);

            //std::cout << RT.lon << std::endl;
            //std::cout << RT.lat << std::endl;
        }
        else if (line.find("SC40") != std::string::npos) {
            std::string coord;
            iss >> keyword >> coord >> SC.numRad >> SC.numTheta;
            SC.ir_distances = new float[SC.numRad];
            ED.resize(SC.numRad, SC.numTheta);
        }
        else if (line.find("SC41") != std::string::npos) {
            iss >> keyword;
            for (int i = 0; i < SC.numRad; ++i) {
                iss >> SC.ir_distances[i];
                SC.ir_distances[i] = SC.ir_distances[i] * 1000.0;
            }
        }
        else if (line.find("SC50") != std::string::npos) {
            iss >> keyword >> SC.weather_file >> SC.nucl_lib_file >> SC.dcf_file >> SC.fcm_file;
            trim(SC.dcf_file);
            trim(SC.fcm_file);
        }
        else if (line.find("SC90") != std::string::npos) {
            std::string early_dose, late_dose;
            iss >> keyword >> early_dose >> late_dose;
            SC.early_dose = (early_dose == "y" || early_dose == "Y");
            SC.late_dose = (late_dose == "y" || late_dose == "Y");
        }
    }

    int puffIndex = 0;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string keyword;

        if (line.empty() || line[0] == '!') {
            continue;
        }

        if (line.find("*END RN_Trans") != std::string::npos) {
            break;
        }

        if (line.find("RT110") != std::string::npos) {
            int id;
            std::string nuclide_name;
            double core_inventory_bq;
            iss >> keyword >> id >> nuclide_name >> core_inventory_bq;

            int nuclide_index = find_nuclide_index(ND, nuclide_name);
            if (nuclide_index != -1) {
                RT.conc[nuclide_index] = core_inventory_bq;
                //std::cout << id << ")     No." << nuclide_index << "\t" << nuclide_name << "\t\t" 
                //    << maccsData.nuclides[nuclide_index].core_inventory_bq << "   Bq" << std::endl;
            }
        }
        else if (line.find("RT120") != std::string::npos) {
            int id;
            std::string rel_class_name, wet_dep_str, dry_dep_str;
            bool wet_dep = false, dry_dep = false;

            iss >> keyword >> id >> rel_class_name >> wet_dep_str >> dry_dep_str;

            if (wet_dep_str == "y" || wet_dep_str == "Y") {
                wet_dep = true;
            }
            if (dry_dep_str == "y" || dry_dep_str == "Y") {
                dry_dep = true;
            }

            for (int i = 0; i < ND.size(); ++i) {
                if (ND[i].chemical_group == setCG(rel_class_name.c_str())) {
                    ND[i].wet_deposition = wet_dep;
                    ND[i].dry_deposition = dry_dep;
                }
            }
        }
        else if (line.find("RT130") != std::string::npos) {
            std::istringstream iss(line);
            std::string keyword;
            int particleSizeID;
            float vDepo;

            iss >> keyword >> particleSizeID >> vDepo;

            if (particleSizeID >= 1 && particleSizeID <= PARTICLE_COUNT) {
                Vdepo[particleSizeID - 1] = vDepo;
            }
        }
        else if (line.find("RT150") != std::string::npos) {
            int iType, iSize;
            float values[ELEMENT_COUNT] = { 0.0f };
            iss >> keyword >> iType >> iSize;

            for (int i = 0; i < ELEMENT_COUNT; ++i) {
                iss >> values[i];
            }

            if (iSize >= 1 && iSize <= PARTICLE_COUNT) {
                for (int i = 0; i < ELEMENT_COUNT; ++i) {
                    particleSizeDistr[0][i][iSize - 1] = values[i];
                }
            }

            for (int count = 0; count < 9; ++count) {
                std::getline(infile, line);
                std::istringstream iss_continued(line);
                iss_continued >> keyword >> iType >> iSize;
                for (int i = 0; i < ELEMENT_COUNT; ++i) {
                    iss_continued >> values[i];
                }

                if (iSize >= 1 && iSize <= PARTICLE_COUNT) {
                    for (int i = 0; i < ELEMENT_COUNT; ++i) {
                        particleSizeDistr[0][i][iSize - 1] = values[i];
                    }
                }
            }
        }
        else if (line.find("RT200") != std::string::npos) {

            int nRelePath, nPuffTotal;
            iss >> keyword >> nRelePath >> nPuffTotal;
            RT.allocatePuffs(nPuffTotal);
        }
        else if (line.find("RT210") != std::string::npos) {

            for (int i = 0; i < RT.nPuffTotal; ++i) {
                std::string relePath;
                RadioNuclideTransport::RT_Puff& puff = RT.RT_puffs[i];
                int iRelePath;
                iss >> keyword >> puff.puffID >> relePath >> puff.rele_time >> puff.duration >> puff.rele_height >> puff.rel_heat >> puff.sizeDistri_iType;
                
                // std::cout << RT.RT_puffs[i].puffID << "\t" << relePath << "\t" << RT.RT_puffs[i].rele_time << "\t" << RT.RT_puffs[i].duration << "\t" << RT.RT_puffs[i].rele_height << "\t" << RT.RT_puffs[i].rel_heat << std::endl;
                
                if (i < RT.nPuffTotal - 1) {
                    std::getline(infile, line);
                    iss.clear();
                    iss.str(line);
                }
            }
        }
        else if (line.find("RT215") != std::string::npos) {
            RadioNuclideTransport& puff = RT;
            iss >> keyword >> keyword >> puff.build_height >> puff.build_width;
        }
        else if (line.find("RT220") != std::string::npos) {
            for (int i = 0; i < RT.nPuffTotal; ++i) {
                RadioNuclideTransport::RT_Puff& puff = RT.RT_puffs[i];
                iss >> keyword >> puff.puffID;
                for (int j = 0; j < 9; ++j) {
                    iss >> puff.release_fractions[j];
                }
                if (i < RT.nPuffTotal - 1) {
                    std::getline(infile, line);
                    iss.clear();
                    iss.str(line);
                }
            }
        }
        else if (line.find("RT310") != std::string::npos) {
            std::string option;
            iss >> keyword >> option;
            WD.isConstant = (option[0] == 'c' || option[0] == 'C');
        }
        else if (line.find("RT320") != std::string::npos) {
            iss >> keyword >> WD.nSamplePerDay;
        }
        else if (line.find("RT340") != std::string::npos) {
            iss >> keyword >> WD.randomSeed;
        }
        else if (line.find("RT350") != std::string::npos) {
            iss >> keyword >> WD.windSpeed >> WD.stability
                >> WD.rainRate >> WD.mixHeight;
        }


    }

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string keyword;

        if (line.empty() || line[0] == '!') {
            continue;
        }

        if (line.find("*END EP_Sim") != std::string::npos) {
            break;
        }

        if (line.find("EP200") != std::string::npos) {
            iss >> keyword >> keyword >> keyword >> EP.alarmTime;
            currentSpeedEndTime = EP.alarmTime;
            //std::cout << EP.alarmTime << std::endl;
        }
        else if (line.find("EP210") != std::string::npos) {
            iss >> keyword >> keyword >> EP.evaEndRing >> EP.EP_endRing;
            //ED.resize(EP.EP_endRing, SC.numTheta);
            SD.resize(EP.EP_endRing, SC.numTheta);
        }
        else if (line.find("EP220") != std::string::npos) {
            iss >> keyword >> keyword;
            //EP.shelterDelay.resize(EP.evaEndRing);
            for (int i = 0; i < EP.evaEndRing; ++i) {
                iss >> EP.shelterDelay[i];
            }
        }
        else if (line.find("EP230") != std::string::npos) {
            iss >> keyword >> keyword;
           // EP.shelterDuration.resize(EP.evaEndRing);
            for (int i = 0; i < EP.evaEndRing; ++i) {
                iss >> EP.shelterDuration[i];
            }
        }
        else if (line.find("EP240") != std::string::npos) {
            iss >> keyword >> keyword >> EP.nSpeedPeriod;
            //EP.speeds.resize(EP.nSpeedPeriod);
            //EP.durations.resize(EP.nSpeedPeriod - 1);

            for (int i = 0; i < EP.nSpeedPeriod; ++i) {
                iss >> EP.speeds[i];
            }
            for (int i = 0; i < EP.nSpeedPeriod - 1; ++i) {
                iss >> EP.durations[i];
            }
        }
        //else if (line.find("EP250") != std::string::npos || line[0] == '+') {
        //    int id, iR;
        //    char dir;
        //    iss >> keyword >> id >> iR;
        //    if (iR > 0 && iR <= EP.EP_endRing) {
        //        if (iR - 1 >= ED.directions.size()) {
        //            ED.directions.resize(iR);
        //        }
        //        for (int j = 0; j < SC.numTheta && j < ED.directions[iR - 1].size(); ++j) {
        //            iss >> dir;
        //            ED.directions[iR - 1][j] = ED.convertDirection(dir);
        //        }
        //    }
        //}
        else if (line.find("EP250") != std::string::npos || line[0] == '+') {
            int id, iR;
            char dir;
            iss >> keyword >> id >> iR;
            if (iR > 0 && iR <= EP.EP_endRing) {
                if (iR > ED.rows) {
                    ED.resize(iR, SC.numTheta);
                }
                for (int j = 0; j < SC.numTheta && j < ED.cols; ++j) {
                    iss >> dir;
                    ED.set(iR - 1, j, ED.convertDirection(dir));
                }
            }
        }

    }

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string keyword;

        if (line.empty() || line[0] == '!') {
            continue;
        }

        if (line.find("*END Site_Data") != std::string::npos) {
            break;
        }

        //else if (line.find("SD50") != std::string::npos) {
        //    iss >> keyword;
        //    for (int i = 0; i < MAX_COLS; ++i) {
        //        iss >> SD.roughness[i];
        //    }
        //}
        //else if (line.find("SD150") != std::string::npos || line[0] == '+') {
        //    int iR;
        //    int val;
        //    iss >> keyword >> iR;
        //    if (iR > 0 && iR <= EP.EP_endRing && iR <= MAX_ROWS) {
        //        for (int j = 0; j < SC.numTheta && j < MAX_COLS; ++j) {
        //            iss >> SD.population[iR - 1][j];
        //        }
        //    }
        //}

        else if (line.find("SD50") != std::string::npos) {
            iss >> keyword;
            SD.roughness.resize(SC.numTheta);
            for (int i = 0; i < SC.numTheta; ++i) {
                iss >> SD.roughness[i];
            }
        }
        else if (line.find("SD150") != std::string::npos || line[0] == '+') {
            int iR;
            iss >> keyword >> iR;
            if (iR > 0 && iR <= EP.EP_endRing && iR <= SD.population.size()) {
                for (int j = 0; j < SC.numTheta; ++j) {
                    iss >> SD.population[iR - 1][j];
                }
            }
        }
    }

    int mp250_idx = 0;  // acute fatality
    int mp260_idx = 0;  // acute morbidity
    int mp270_idx = 0;  // cancer effect

    while (std::getline(infile, line)) {
        std::string keyword1, keyword;

        if (line.empty() || line[0] == '!') {
            continue;
        }

        if (line.find("*END Model_Par") != std::string::npos) {
            break;
        }

        else if (line.find("MP130") != std::string::npos || line[0] == '+') {
            std::istringstream iss(line);
            iss >> keyword >> wc1 >> wc2;
            std::cout << "wc1 = " << wc1 << ", wc2 = " << wc2 << std::endl;
            cudaMemcpyToSymbol(d_wc1, &wc1, sizeof(float));
            cudaMemcpyToSymbol(d_wc2, &wc2, sizeof(float));
        }

        else if (line.find("MP210") != std::string::npos) {
            //for (int i = 0; i < 3; ++i) {
            //    std::istringstream iss(line);
            //    iss >> keyword >> keyword1 >> PF.pfactor[i][0] >> PF.pfactor[i][1] 
            //        >> PF.pfactor[i][2] >> PF.pfactor[i][3] >> PF.pfactor[i][4];
            //    std::getline(infile, line);
            //}
            std::istringstream iss(line);
            iss >> keyword >> keyword1 >> PF.pfactor[1][0] >> PF.pfactor[1][1]
                >> PF.pfactor[1][2] >> PF.pfactor[1][3] >> PF.pfactor[1][4];
            std::getline(infile, line);

            std::istringstream iss1(line);
            iss1 >> keyword >> keyword1 >> PF.pfactor[0][0] >> PF.pfactor[0][1]
                >> PF.pfactor[0][2] >> PF.pfactor[0][3] >> PF.pfactor[0][4];
            std::getline(infile, line);

            std::istringstream iss2(line);
            iss2 >> keyword >> keyword1 >> PF.pfactor[2][0] >> PF.pfactor[2][1]
                >> PF.pfactor[2][2] >> PF.pfactor[2][3] >> PF.pfactor[2][4];
            //std::getline(infile, line);

            //std::cout << PF.pfactor[0][0] << std::endl;
            //std::cout << PF.pfactor[0][1] << std::endl;
            //std::cout << PF.pfactor[0][2] << std::endl;
            //std::cout << PF.pfactor[0][3] << std::endl;
            //std::cout << PF.pfactor[0][4] << std::endl << std::endl;

            //std::cout << PF.pfactor[1][0] << std::endl;
            //std::cout << PF.pfactor[1][1] << std::endl;
            //std::cout << PF.pfactor[1][2] << std::endl;
            //std::cout << PF.pfactor[1][3] << std::endl;
            //std::cout << PF.pfactor[1][4] << std::endl << std::endl;

            //std::cout << PF.pfactor[2][0] << std::endl;
            //std::cout << PF.pfactor[2][1] << std::endl;
            //std::cout << PF.pfactor[2][2] << std::endl;
            //std::cout << PF.pfactor[2][3] << std::endl;
            //std::cout << PF.pfactor[2][4] << std::endl;

        }
        else if (line.find("MP220") != std::string::npos || line[0] == '+') {
            std::istringstream iss(line);
            iss >> keyword >> PF.resus_coef >> PF.resus_half_life;
            //std::cout << "PF.resus_coef = " << PF.resus_coef << ", PF.resus_half_life = " << PF.resus_half_life << std::endl;
        }
        else if (line.find("MP250") != std::string::npos) {

            std::istringstream iss(line);
            int    id;

            iss >> keyword >> id >> HE.FatalityName[0] >> HE.TargetOrgan_AF[0] >> HE.alpha_f[0] >> HE.beta_f[0] >> HE.threshold_AF[0];
            std::getline(infile, line);
            iss.clear();
            iss.str(line);
            iss >> keyword >> id >> HE.FatalityName[1] >> HE.TargetOrgan_AF[1] >> HE.alpha_f[1] >> HE.beta_f[1] >> HE.threshold_AF[1];

        }
        // MP260: Acute Injury
        else if (line.find("MP260") != std::string::npos) {
            std::istringstream iss(line);
            int    id;

            for (int j = 0; j < 7; j++) {
                iss >> keyword >> id >> HE.InjuryName[j] >> HE.TargetOrgan_AM[j] >> HE.alpha_i[j] >> HE.beta_i[j] >> HE.threshold_AM[j];
                if (j < 6) {
                    std::getline(infile, line);
                    iss.clear();
                    iss.str(line);
                }
            }
        }
        // MP270: Cancer Effect
        else if (line.find("MP270") != std::string::npos) {
            std::istringstream iss(line);
            int    id;

            for (int j = 0; j < 7; j++) {
                iss >> keyword >> id >>
                    HE.CancerName[j] >> HE.TargetOrgan[j] >>
                    HE.dos_a[j] >> HE.dos_b[j] >>
                    HE.cf_risk[j] >> HE.ci_risk[j] >>
                    HE.ddrf[j] >> HE.dos_thres[j] >>
                    HE.dosRate_thres[j] >> HE.LNT_threshold[j] >> HE.sus_frac[j];

                if (j < 6) {
                    std::getline(infile, line);
                    iss.clear();
                    iss.str(line);
                }
            }
        }
    }

    infile.close();
}

void read_input_RCAPn(int numi, const std::string& filename, SimulationControl& SC, std::vector<NuclideData>& ND, RadioNuclideTransport& RT) {

    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string keyword;

        if (line.empty() || line[0] == '!') {
            continue;
        }

        if (line.find("*END Simul_Con") != std::string::npos) {
            break;
        }
        else if (line.find("SC20") != std::string::npos) {
            std::string longitude, latitude;
            iss >> keyword >> SC.plant_name >> SC.plant_power >> SC.plant_type >> longitude >> latitude;

            //std::cout << longitude << std::endl;
            //std::cout << latitude << std::endl;

            RT.lon = std::stof(longitude.substr(1)) * (longitude[0] == 'W' ? -1.0f : 1.0f);
            RT.lat = std::stof(latitude.substr(1)) * (latitude[0] == 'S' ? -1.0f : 1.0f);

            //std::cout << std::setprecision(9) << RT.lon << std::endl;
            //std::cout << std::setprecision(9) << RT.lat << std::endl;
        }
    }

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string keyword;

        if (line.empty() || line[0] == '!') {
            continue;
        }

        if (line.find("*END RN_Trans") != std::string::npos) {
            break;
        }

        if (line.find("RT110") != std::string::npos) {
            int id;
            std::string nuclide_name;
            double core_inventory_bq;
            iss >> keyword >> id >> nuclide_name >> core_inventory_bq;

            int nuclide_index = find_nuclide_index(ND, nuclide_name);
            //std::cout << "nuclide_index = " << nuclide_index << std::endl;
            if (nuclide_index != -1) {
                RT.conc[nuclide_index] = core_inventory_bq;
            }
        }
        else if (line.find("RT150") != std::string::npos) {
            int iType, iSize;
            float values[ELEMENT_COUNT] = { 0.0f };
            iss >> keyword >> iType >> iSize;

            for (int i = 0; i < ELEMENT_COUNT; ++i) {
                iss >> values[i];
            }

            if (iSize >= 1 && iSize <= PARTICLE_COUNT) {
                for (int i = 0; i < ELEMENT_COUNT; ++i) {
                    particleSizeDistr[numi][i][iSize - 1] = values[i];
                }
            }

            for (int count = 0; count < 9; ++count) {
                std::getline(infile, line);
                std::istringstream iss_continued(line);
                iss_continued >> keyword >> iType >> iSize;
                for (int i = 0; i < ELEMENT_COUNT; ++i) {
                    iss_continued >> values[i];
                }

                if (iSize >= 1 && iSize <= PARTICLE_COUNT) {
                    for (int i = 0; i < ELEMENT_COUNT; ++i) {
                        particleSizeDistr[numi][i][iSize - 1] = values[i];
                    }
                }
            }
        }
        else if (line.find("RT200") != std::string::npos) {

            int nRelePath, nPuffTotal;
            iss >> keyword >> nRelePath >> nPuffTotal;
            RT.allocatePuffs(nPuffTotal);
        }
        else if (line.find("RT210") != std::string::npos) {

            for (int i = 0; i < RT.nPuffTotal; ++i) {
                std::string relePath;
                RadioNuclideTransport::RT_Puff& puff = RT.RT_puffs[i];
                int iRelePath;
                iss >> keyword >> puff.puffID >> relePath >> puff.rele_time >> puff.duration >> puff.rele_height >> puff.rel_heat >> puff.sizeDistri_iType;

                if (i < RT.nPuffTotal - 1) {
                    std::getline(infile, line);
                    iss.clear();
                    iss.str(line);
                }
            }
        }
        else if (line.find("RT220") != std::string::npos) {
            for (int i = 0; i < RT.nPuffTotal; ++i) {
                RadioNuclideTransport::RT_Puff& puff = RT.RT_puffs[i];
                iss >> keyword >> puff.puffID;
                for (int j = 0; j < 9; ++j) {
                    iss >> puff.release_fractions[j];
                }
                if (i < RT.nPuffTotal - 1) {
                    std::getline(infile, line);
                    iss.clear();
                    iss.str(line);
                }
            }
        }
    }

    infile.close();
}



int check_input_num(const std::string& filename) {

    int n;
    std::ifstream infile(filename);

    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    while (std::getline(infile, line)) {
        std::istringstream iss(line);
        std::string keyword;

        if (line.empty() || line[0] == '!') {
            continue;
        }

        if (line.find("*END Multi_input") != std::string::npos) {
            break;
        }

        if (line.find("MU00") != std::string::npos) {
            iss >> keyword >> n;
        }
    }

    infile.close();

    return n;
}

void copy_simulation_data_to_gpu(SimulationControl* h_simControls, SimulationControl** d_simControls, int input_num) {
    cudaMalloc(d_simControls, sizeof(SimulationControl) * input_num);

    for (int i = 0; i < input_num; ++i) {
        float* d_ir_distances;
        cudaMalloc(&d_ir_distances, sizeof(float) * h_simControls[i].numRad);
        cudaMemcpy(d_ir_distances, h_simControls[i].ir_distances, sizeof(float) * h_simControls[i].numRad, cudaMemcpyHostToDevice);

        SimulationControl temp = h_simControls[i];
        temp.ir_distances = d_ir_distances;

        cudaMemcpy(&(*d_simControls)[i], &temp, sizeof(SimulationControl), cudaMemcpyHostToDevice);
    }
}

// ============================================================================
// MACCS Dose Conversion Factor (DCF) Reader
// ============================================================================
// Reads dose conversion factors from MACCS-format DCF file
//
// INPUT FILE FORMAT:
// ------------------
// DC20 <nuclide_name>
//   Starts a new nuclide entry
//
// DC30 + <organ_name> <cloud_shine> <ground_shine> <inhal_early> <inhal_late> <ingestion>
//   Defines dose conversion factors for one organ
//   Multiple DC30 lines can follow each DC20
//
// DOSE CONVERSION FACTORS:
// ------------------------
// Cloud shine    : External dose from passing radioactive cloud (Sv/Bq-s/m^3)
// Ground shine   : External dose from deposited radionuclides (Sv-m^2/Bq-s)
// Inhal early    : Inhalation dose, early phase (Sv/Bq inhaled)
// Inhal late     : Inhalation dose, late phase (Sv/Bq inhaled)
// Ingestion      : Ingestion dose (Sv/Bq ingested)
//
// ORGAN TYPES:
// ------------
// Common organs include: TESTES, BREAST, LUNGS, R_MARR (red marrow),
// BONE_SU (bone surface), THYROID, REMAINDER, EFFECTIVE, SKIN
//
// MACCS: MELCOR Accident Consequence Code System
// DCF values are based on ICRP (International Commission on Radiological
// Protection) dose coefficients
// ============================================================================
void read_MACCS_DCF_New2(const std::string& filename, std::vector<NuclideData>& ND) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    bool is_reading_nuclides = false;
    int current_nuclide_id = -1;
    std::string current_nuclide_name;

    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '!') {
            continue;
        }

        std::istringstream iss(line);
        std::string code;
        iss >> code;

        if (code == "DC20") {
            current_nuclide_id++;
            iss >> current_nuclide_name;

            if (current_nuclide_id >= ND.size()) {
                std::cerr << "Error: Exceeded the maximum number of nuclides" << std::endl;
                exit(EXIT_FAILURE);
            }

            is_reading_nuclides = true;
            ND[current_nuclide_id].id = current_nuclide_id;
            strncpy(ND[current_nuclide_id].name, current_nuclide_name.c_str(), MAX_STRING_LENGTH);
            ND[current_nuclide_id].organ_count = 0;

            if (CHECK_DCF) {
                std::cout << std::endl << "No." << current_nuclide_id << " [" << ND[current_nuclide_id].name << "]" << std::endl << std::endl;
                std::cout << std::left << std::setw(9) << "\t" << "(Cloud shine)\t(Ground shine)\t(Inhal early)\t(Inhal late)\t(Ingestion)" << std::endl;
            }
            continue;
        }

        if (is_reading_nuclides && (code == "DC30" || code == "+")) {
            std::string organ_name;
            double cloud_shine_dcf, ground_shine_dcf, inhalation_early_dcf, inhalation_late_dcf, ingestion_dcf;

            iss >> code;
            iss >> organ_name >> cloud_shine_dcf >> ground_shine_dcf >> inhalation_early_dcf >> inhalation_late_dcf >> ingestion_dcf;

            int organ_index = ND[current_nuclide_id].organ_count;
            if (organ_index >= MAX_ORGANS) {
                std::cerr << "Error: Exceeded the maximum number of organs for nuclide " << ND[current_nuclide_id].name << std::endl;
                exit(EXIT_FAILURE);
            }

            strncpy(&ND[current_nuclide_id].organ_names[organ_index * MAX_STRING_LENGTH], organ_name.c_str(), MAX_STRING_LENGTH);
            ND[current_nuclide_id].exposure_data[organ_index * DATA_FIELDS + 0] = cloud_shine_dcf;
            ND[current_nuclide_id].exposure_data[organ_index * DATA_FIELDS + 1] = ground_shine_dcf;
            ND[current_nuclide_id].exposure_data[organ_index * DATA_FIELDS + 2] = inhalation_early_dcf;
            ND[current_nuclide_id].exposure_data[organ_index * DATA_FIELDS + 3] = inhalation_late_dcf;
            ND[current_nuclide_id].exposure_data[organ_index * DATA_FIELDS + 4] = ingestion_dcf;

            ND[current_nuclide_id].organ_count++;

            if (CHECK_DCF) {
                std::cout << std::left << std::setw(10) << &ND[current_nuclide_id].organ_names[organ_index * MAX_STRING_LENGTH] << "\t";
                std::cout << std::scientific << std::setprecision(2);
                std::cout << ND[current_nuclide_id].exposure_data[organ_index * DATA_FIELDS + 0] << "\t" <<
                    ND[current_nuclide_id].exposure_data[organ_index * DATA_FIELDS + 1] << "\t" <<
                    ND[current_nuclide_id].exposure_data[organ_index * DATA_FIELDS + 2] << "\t" <<
                    ND[current_nuclide_id].exposure_data[organ_index * DATA_FIELDS + 3] << "\t" <<
                    ND[current_nuclide_id].exposure_data[organ_index * DATA_FIELDS + 4] << std::endl;
            }

        }

        if (code == "*END" || code == "*EOF") {
            break;
        }
    }

    infile.close();
}



// ============================================================================
// MACCS Nuclide Data Library (NDL) Reader
// ============================================================================
// Reads radionuclide physical properties from MACCS60 nuclide library file
//
// INPUT FILE FORMAT:
// ------------------
// NL10 <name> <half_life> <unit> <atomic_weight> <chemical_group>
//   Nuclide identification and basic properties
//   half_life units: s (seconds), m (minutes), h (hours), d (days), y (years)
//   chemical_group: xen, iod, ces, tel, str, rut, lan, cer, bar
//
// NL20 <core_inventory>
//   Core inventory in Ci/MWth (Curies per megawatt thermal)
//
// NL30 <daughter_name> <branching_fraction>
//   Decay chain information (can have multiple NL30 lines)
//   branching_fraction: probability of this decay path (0.0 to 1.0)
//
// NUCLEAR PHYSICS CONCEPTS:
// -------------------------
// Half-life: Time for half the radioactive atoms to decay
// Atomic weight: Mass number of the isotope (amu)
// Core inventory: Amount of nuclide in reactor core per unit thermal power
// Branching fraction: Probability of decay to a specific daughter nuclide
// Decay chain: Series of radioactive decays from parent to daughter nuclides
//
// UNITS:
// ------
// Half-life: Converted to seconds internally
// Core inventory: Ci/MWth (Curies per Megawatt thermal)
// Atomic weight: g/mol (grams per mole)
// ============================================================================
void read_MACCS60_NDL(const std::string& filename, std::vector<NuclideData>& ND) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    int current_nuclide_index = -1;

    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '!') {
            continue;
        }

        std::istringstream iss(line);
        std::string code;
        iss >> code;

        if (code == "NL10") {
            std::string nuclide_name, half_life_str, half_life_unit;
            iss >> nuclide_name >> half_life_str >> half_life_unit;

            current_nuclide_index = find_nuclide_index(ND, nuclide_name);

            if (current_nuclide_index == -1) {
                std::cerr << "Error: Nuclide " << nuclide_name << " not found in existing data." << std::endl;
                continue;
            }

            // Convert half-life to seconds based on input unit
            double half_life_seconds = std::stod(half_life_str);
            if (half_life_unit == "d") {
                half_life_seconds *= 86400.0;          // days to seconds
            }
            else if (half_life_unit == "h") {
                half_life_seconds *= 3600.0;           // hours to seconds
            }
            else if (half_life_unit == "m") {
                half_life_seconds *= 60.0;             // minutes to seconds
            }
            else if (half_life_unit == "y") {
                half_life_seconds *= 31557600.0;       // years to seconds (365.25 days)
            }
            ND[current_nuclide_index].half_life = half_life_seconds;

            iss >> ND[current_nuclide_index].atomic_weight;
            std::string chemical_group_name;
            iss >> chemical_group_name;
            ND[current_nuclide_index].chemical_group = setCG(chemical_group_name.c_str());
        }
        else if (code == "NL20") {
            double core_inventory_ci_per_mwth;
            iss >> core_inventory_ci_per_mwth;
            if (current_nuclide_index != -1) {
                ND[current_nuclide_index].core_inventory = core_inventory_ci_per_mwth;
            }
        }
        else if (code == "NL30") {
            if (current_nuclide_index != -1 && ND[current_nuclide_index].decay_count < MAX_DNUC) {
                std::string daughter_nuclide_name;
                double decay_branching_fraction;
                iss >> daughter_nuclide_name >> decay_branching_fraction;

                strncpy(&ND[current_nuclide_index].daughter[ND[current_nuclide_index].decay_count * MAX_STRING_LENGTH],
                        daughter_nuclide_name.c_str(), MAX_STRING_LENGTH);
                ND[current_nuclide_index].branching_fraction[ND[current_nuclide_index].decay_count] = decay_branching_fraction;
                ND[current_nuclide_index].decay_count++;
            }
        }

        if (code == "*EOF") {
            break;
        }
    }

    infile.close();
}

void print_MACCS60_NDL(const std::vector<NuclideData>& ND) {
    for (int i = 1; i < ND.size(); i++) {
        const NuclideData& nuclide = ND[i];
        std::cout << "No." << i << " " << nuclide.name << "\n"
            << "  Half Life: " << nuclide.half_life << " seconds\n"
            << "  Atomic Weight: " << nuclide.atomic_weight << " g\n"
            << "  Chemical Group: " << nuclide.chemical_group << "\n"
            << "  Core Inventory: " << nuclide.core_inventory << " Ci/MWth\n";

        for (int j = 0; j < nuclide.decay_count; ++j) {
            std::cout << "  Decay Daughter: " << nuclide.daughter[j]
                << " (Branching Fraction: " << nuclide.branching_fraction[j] << ")\n";
        }
        std::cout << std::endl;
    }
}



// ============================================================================
// Coordinate Conversion Utilities
// ============================================================================

// Converts degrees to radians
inline double toRadians(double degrees) {
    return degrees * PI / 180.0;
}

// ============================================================================
// Puff Initialization for RCAP
// ============================================================================
// Creates puff objects for each meteorological scenario and release source
// Converts geographic coordinates to local Cartesian coordinates
//
// COORDINATE SYSTEM:
// ------------------
// Uses spherical Earth approximation for coordinate conversion:
// - Origin: First source location (RT[0].lon, RT[0].lat)
// - X-axis: Eastward (meters)
// - Y-axis: Northward (meters)
// - Earth radius: 6,371,000 meters
//
// PUFF PROPERTIES:
// ----------------
// - Position: (x, y, z) in meters from origin
// - Concentration: Activity per nuclide (Bq)
// - Release time: Seconds from simulation start
// - Meteorology: Wind speed, direction, stability, precipitation
// ============================================================================
void Gpuff::initializePuffs(
    int input_num,
    const std::vector<RadioNuclideTransport>& RT,
    const std::vector<NuclideData>& ND
) {
    int totalPuffs = 0;
    for (int i = 0; i < input_num; i++) {
        totalPuffs += RT[i].nPuffTotal;
    }

    puffs_RCAP.reserve(RCAP_metdata.size() * totalPuffs);

    double base_longitude = RT[0].lon;
    double base_latitude = RT[0].lat;

    printf("RCAP_metdata = %d\n", RCAP_metdata.size());



    for (size_t met_index = 0; met_index < RCAP_metdata.size(); met_index++) {
        for (int source_index = 0; source_index < input_num; source_index++) {
            for (int puff_index = 0; puff_index < RT[source_index].nPuffTotal; puff_index++) {

                // Convert geographic coordinates to local Cartesian (meters)
                double longitude_diff_rad = toRadians(RT[source_index].lon - base_longitude);
                double latitude_diff_rad = toRadians(RT[source_index].lat - base_latitude);
                double base_latitude_rad = toRadians(base_latitude);

                float puff_x_meters = static_cast<float>(EARTH_RADIUS * longitude_diff_rad * std::cos(base_latitude_rad));
                float puff_y_meters = static_cast<float>(EARTH_RADIUS * latitude_diff_rad);
                float puff_z_meters = RT[source_index].RT_puffs[puff_index].rele_height;

                // Calculate concentration for each nuclide
                float nuclide_concentrations[MAX_NUCLIDES];
                for (int nuclide_idx = 0; nuclide_idx < MAX_NUCLIDES; nuclide_idx++) {
                    nuclide_concentrations[nuclide_idx] = RT[source_index].conc[nuclide_idx] *
                        RT[source_index].RT_puffs[puff_index].release_fractions[ND[nuclide_idx].chemical_group];
                    if (nuclide_concentrations[nuclide_idx] > 1.0) {
                        std::cout << "nuc = " << nuclide_idx << ", conc = " << nuclide_concentrations[nuclide_idx] << std::endl;
                    }
                }

                float release_time_seconds = RT[source_index].RT_puffs[puff_index].rele_time;
                int unit_index = source_index;

                // Assign meteorology from current timestep
                float wind_speed_m_per_s = RCAP_metdata[met_index].spd;

                // Calculate wind direction index based on release time
                int time_steps_since_start = static_cast<int>(release_time_seconds / 360);
                int wind_direction_index = (met_index + time_steps_since_start) % RCAP_metdata.size();

                float wind_direction_degrees = RCAP_metdata[wind_direction_index].dir;
                int stability_class = RCAP_metdata[met_index].stab;
                float rain_rate_mm_per_h = RCAP_metdata[met_index].rain;

                puffs_RCAP.emplace_back(puff_x_meters, puff_y_meters, puff_z_meters,
                    nuclide_concentrations, release_time_seconds, unit_index,
                    wind_speed_m_per_s, wind_direction_degrees, stability_class, rain_rate_mm_per_h, met_index);
            }
        }
    }
}

// ============================================================================
// Evacuee Initialization (Polar Coordinates)
// ============================================================================
// Creates evacuee objects for each populated grid cell in polar coordinates
// Each evacuee represents a population group at a specific radial distance
// and angular direction from the source
//
// SPATIAL DISCRETIZATION:
// -----------------------
// - Radial: Defined by SC.ir_distances[] (meters from source)
// - Angular: Divided into numTheta sectors (typically 16, representing compass directions)
// - Population: Distributed according to SD.population[ring][sector]
//
// EVACUEE PROPERTIES:
// -------------------
// - Position: (r, theta) in polar coordinates
// - Population: Number of people in this cell
// - Speed: Initially 0.0 (set by evacuation model during simulation)
// - Flag: true if within evacuation zone, false otherwise
//
// METEOROLOGY SCENARIOS:
// ----------------------
// Creates evacuees for each meteorological scenario to enable
// probabilistic consequence assessment
// ============================================================================
void Gpuff::initializeEvacuees(std::vector<Evacuee>& evacuees, const SimulationControl& SC,
    const EvacuationData& EP, const SiteData& SD) {

    int num_radial_rings = SC.numRad;
    int num_angular_sectors = SC.numTheta;

    for (int met_scenario = 0; met_scenario < RCAP_metdata.size(); met_scenario++) {
        for (int ring_index = 0; ring_index < num_radial_rings; ++ring_index) {
            for (int sector_index = 0; sector_index < num_angular_sectors; ++sector_index) {
                int population_count = SD.population[ring_index][sector_index];
                if (population_count > 0) {
                    Evacuee evacuee;
                    evacuee.population = static_cast<float>(population_count);
                    evacuee.r = SC.ir_distances[ring_index];
                    evacuee.theta = (2 * PI * sector_index / num_angular_sectors);
                    evacuee.speed = 0.0f;
                    evacuee.rad0 = ring_index;

                    // Set evacuation flag based on evacuation zone
                    if (ring_index > EP.evaEndRing - 1) {
                        evacuee.flag = false;  // Outside evacuation zone
                    } else {
                        evacuee.flag = true;   // Inside evacuation zone
                    }

                    evacuee.prev_rad_idx = ring_index;
                    evacuee.prev_theta_idx = sector_index;
                    evacuee.met_idx = met_scenario;

                    evacuees.push_back(evacuee);
                    if (met_scenario == 0) totalevacuees_per_Sim++;
                }
            }
        }
    }

}

// ============================================================================
// Evacuee Initialization (Cartesian Coordinates)
// ============================================================================
// Creates evacuee objects on a uniform Cartesian grid
// Used for detailed spatial analysis or visualization
//
// GRID CONFIGURATION:
// -------------------
// - X range: -6000 to 500 meters (6.5 km west-to-east extent)
// - Y range: -1500 to 1500 meters (3 km south-to-north extent)
// - Resolution: 1000 x 250 grid cells
// - Population per cell: 100 persons (uniform distribution)
//
// DOSE TRACKING:
// --------------
// Each evacuee tracks:
// - Inhalation dose (Sv)
// - Cloud shine dose (Sv)
// - Initialized to 1.0e-40 (near-zero) to avoid numerical issues
//
// NOTE: This function creates a fixed grid for testing/visualization
//       Production runs should use initializeEvacuees() with realistic
//       population distribution
// ============================================================================
void Gpuff::initializeEvacuees_xy(std::vector<Evacuee>& evacuees, const SimulationControl& SC,
    const EvacuationData& EP, const SiteData& SD) {

    const int grid_x_cells = 1000;
    const int grid_y_cells = 250;
    const float x_min_meters = -6000.0f;
    const float x_max_meters = 500.0f;
    const float y_min_meters = -1500.0f;
    const float y_max_meters = 1500.0f;
    const float population_per_cell = 100.0f;

    for (int met_scenario = 0; met_scenario < RCAP_metdata.size(); met_scenario++) {
        for (int i = 0; i < grid_x_cells; ++i) {
            for (int j = 0; j < grid_y_cells; ++j) {

                Evacuee evacuee;
                evacuee.population = population_per_cell;
                evacuee.speed = 0.0f;
                evacuee.flag = true;

                evacuee.prev_rad_idx = i;
                evacuee.prev_theta_idx = j;

                evacuee.x = (x_max_meters - x_min_meters) * i / grid_x_cells + x_min_meters;
                evacuee.y = (y_max_meters - y_min_meters) * j / grid_y_cells + y_min_meters;

                evacuee.dose_inhalation = 1.0e-40f;
                evacuee.dose_cloudshine = 1.0e-40f;

                evacuees.push_back(evacuee);
                if (met_scenario == 0) totalevacuees_per_Sim++;
            }
        }
    }

}
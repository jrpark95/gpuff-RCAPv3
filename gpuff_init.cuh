#include "gpuff.cuh"

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

void trim(char* str) {
    char* p = str;
    while (*p == ' ') ++p;
    memmove(str, p, strlen(p) + 1);

    p = str + strlen(str) - 1;
    while (p >= str && *p == ' ') --p;
    *(p + 1) = '\0';
}

int find_nuclide_index(std::vector<NuclideData>& ND, const std::string& nuclide_name) {
    for (int i = 0; i < ND.size(); i++) {
        if (STRICMP(ND[i].name, nuclide_name.c_str()) == 0) {
            return i;
        }
    }
    return -1;
}

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


//void read_input_RCAP(const std::string& filename, SimulationControl& simControl) {
//    std::ifstream infile(filename);
//    if (!infile.is_open()) {
//        std::cerr << "Error: Could not open file " << filename << std::endl;
//        exit(EXIT_FAILURE);
//    }
//
//    std::string line;
//    while (std::getline(infile, line)) {
//        std::istringstream iss(line);
//        std::string keyword;
//
//        // Skip empty lines and comments
//        if (line.empty() || line[0] == '!') {
//            continue;
//        }
//
//        if (line.find("SC10") != std::string::npos) {
//            // SC10: Read sim_title
//            iss >> keyword >> simControl.sim_title;
//        }
//        else if (line.find("SC20") != std::string::npos) {
//            // SC20: Read plant_name, plant_power, plant_type, loc_longitude, loc_latitude
//            std::string longitude, latitude;
//            iss >> keyword >> simControl.plant_name >> simControl.plant_power >> simControl.plant_type >> longitude >> latitude;
//
//            // Process longitude and latitude (remove E/N and convert to float)
//            if (longitude[0] == 'E' || longitude[0] == 'W') {
//                simControl.loc_longitude = std::stof(longitude.substr(1));
//                if (longitude[0] == 'W') simControl.loc_longitude = -simControl.loc_longitude;
//            }
//            if (latitude[0] == 'N' || latitude[0] == 'S') {
//                simControl.loc_latitude = std::stof(latitude.substr(1));
//                if (latitude[0] == 'S') simControl.loc_latitude = -simControl.loc_latitude;
//            }
//        }
//        else if (line.find("SC40") != std::string::npos) {
//            // SC40: Read coord, numRad, numTheta
//            std::string coord;
//            iss >> keyword >> coord >> simControl.numRad >> simControl.numTheta;
//            simControl.ir_distances = new float[simControl.numRad];
//        }
//        else if (line.find("SC41") != std::string::npos) {
//            // SC41: Read ir distances
//            iss >> keyword;
//            for (int i = 0; i < simControl.numRad; ++i) {
//                iss >> simControl.ir_distances[i];
//            }
//        }
//        else if (line.find("SC50") != std::string::npos) {
//            iss >> keyword >> simControl.weather_file >> simControl.nucl_lib_file;
//            iss >> simControl.dcf_file >> simControl.fcm_file;
//
//            // Remove leading and trailing spaces
//            trim(simControl.dcf_file);
//            trim(simControl.fcm_file);
//        }
//        else if (line.find("SC90") != std::string::npos) {
//            // SC90: Read early_dose and late_dose
//            std::string early_dose, late_dose;
//            iss >> keyword >> early_dose >> late_dose;
//            simControl.early_dose = (early_dose == "y" || early_dose == "Y");
//            simControl.late_dose = (late_dose == "y" || late_dose == "Y");
//        }
//
//
//    }
//
//    infile.close();
//}


void parse_RT130(const std::string& line) {
    std::istringstream iss(line);
    std::string keyword;
    int particleSizeID;
    float vDepo;

    iss >> keyword >> particleSizeID >> vDepo;

    if (particleSizeID >= 1 && particleSizeID <= PARTICLE_COUNT) {
        Vdepo[particleSizeID - 1] = vDepo;
    }
}




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

void read_MACCS_DCF_New2(const std::string& filename, std::vector<NuclideData>& ND) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    bool is_reading_nuclides = false;
    int current_id = -1; 
    std::string current_name;

    while (std::getline(infile, line)) {
        if (line.empty() || line[0] == '!') {
            continue;
        }

        std::istringstream iss(line);
        std::string code;
        iss >> code;

        if (code == "DC20") {
            current_id++; 
            iss >> current_name;

            if (current_id >= ND.size()) {
                std::cerr << "Error: Exceeded the maximum number of nuclides" << std::endl;
                exit(EXIT_FAILURE);
            }

            is_reading_nuclides = true;
            ND[current_id].id = current_id;
            strncpy(ND[current_id].name, current_name.c_str(), MAX_STRING_LENGTH);
            ND[current_id].organ_count = 0;

            if (CHECK_DCF) {
                std::cout << std::endl << "No." << current_id << " [" << ND[current_id].name << "]" << std::endl << std::endl;
                std::cout << std::left << std::setw(9) << "\t" << "(Cloud shine)\t(Ground shine)\t(Inhal early)\t(Inhal late)\t(Ingestion)" << std::endl;
            }
            continue;
        }

        if (is_reading_nuclides && (code == "DC30" || code == "+")) {
            std::string organ_name;
            double cloud_shine, ground_shine, inhal_early, inhal_late, ingestion;

            iss >> code;
            iss >> organ_name >> cloud_shine >> ground_shine >> inhal_early >> inhal_late >> ingestion;

            int organ_idx = ND[current_id].organ_count;
            if (organ_idx >= MAX_ORGANS) {
                std::cerr << "Error: Exceeded the maximum number of organs for nuclide " << ND[current_id].name << std::endl;
                exit(EXIT_FAILURE);
            }

            strncpy(&ND[current_id].organ_names[organ_idx * MAX_STRING_LENGTH], organ_name.c_str(), MAX_STRING_LENGTH);
            ND[current_id].exposure_data[organ_idx * DATA_FIELDS + 0] = cloud_shine;
            ND[current_id].exposure_data[organ_idx * DATA_FIELDS + 1] = ground_shine;
            ND[current_id].exposure_data[organ_idx * DATA_FIELDS + 2] = inhal_early;
            ND[current_id].exposure_data[organ_idx * DATA_FIELDS + 3] = inhal_late;
            ND[current_id].exposure_data[organ_idx * DATA_FIELDS + 4] = ingestion;

            ND[current_id].organ_count++;

            if (CHECK_DCF) {
                std::cout << std::left << std::setw(10) << &ND[current_id].organ_names[organ_idx * MAX_STRING_LENGTH] << "\t";
                std::cout << std::scientific << std::setprecision(2);
                std::cout << ND[current_id].exposure_data[organ_idx * DATA_FIELDS + 0] << "\t" <<
                    ND[current_id].exposure_data[organ_idx * DATA_FIELDS + 1] << "\t" <<
                    ND[current_id].exposure_data[organ_idx * DATA_FIELDS + 2] << "\t" <<
                    ND[current_id].exposure_data[organ_idx * DATA_FIELDS + 3] << "\t" <<
                    ND[current_id].exposure_data[organ_idx * DATA_FIELDS + 4] << std::endl;
            }

        }

        if (code == "*END" || code == "*EOF") {
            break;
        }
    }

    infile.close();
}



void read_MACCS60_NDL(const std::string& filename, std::vector<NuclideData>& ND) {
    std::ifstream infile(filename);
    if (!infile.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;
    int current_index = -1;

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

            // Find the corresponding nuclide index
            current_index = find_nuclide_index(ND, nuclide_name);

            if (current_index == -1) {
                std::cerr << "Error: Nuclide " << nuclide_name << " not found in existing data." << std::endl;
                continue;
            }

            // Process half-life and convert to seconds
            double half_life = std::stod(half_life_str);
            if (half_life_unit == "d") {
                half_life *= 86400; // days to seconds
            }
            else if (half_life_unit == "h") {
                half_life *= 3600; // hours to seconds
            }
            else if (half_life_unit == "m") {
                half_life *= 60; // minutes to seconds
            }
            else if (half_life_unit == "y") {
                half_life *= 31557600; // years to seconds (365.25 days)
            }
            ND[current_index].half_life = half_life;

            iss >> ND[current_index].atomic_weight;
            std::string chemical_group;
            iss >> chemical_group;
            ND[current_index].chemical_group = setCG(chemical_group.c_str());
        }
        else if (code == "NL20") {
            double core_inventory;
            iss >> core_inventory;
            if (current_index != -1) {
                ND[current_index].core_inventory = core_inventory;
            }
        }
        else if (code == "NL30") {
            if (current_index != -1 && ND[current_index].decay_count < MAX_DNUC) {
                std::string daughter;
                double branching_fraction;
                iss >> daughter >> branching_fraction;

                strncpy(&ND[current_index].daughter[ND[current_index].decay_count * MAX_STRING_LENGTH], daughter.c_str(), MAX_STRING_LENGTH);
                ND[current_index].branching_fraction[ND[current_index].decay_count] = branching_fraction;
                ND[current_index].decay_count++;
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


//void copy_MACCS_data_to_device(MACCSData& h_maccsData, MACCSData*& d_maccsData) {
//
//    cudaMalloc((void**)&d_maccsData, sizeof(MACCSData));
//
//    NuclideData* d_nuclides;
//    cudaMalloc((void**)&d_nuclides, sizeof(NuclideData) * h_maccsData.nuclide_count);
//
//    cudaMemcpy(d_nuclides, h_maccsData.nuclides, sizeof(NuclideData) * h_maccsData.nuclide_count, cudaMemcpyHostToDevice);
//
//    cudaMemcpy(&(d_maccsData->nuclides), &d_nuclides, sizeof(NuclideData*), cudaMemcpyHostToDevice);
//
//    cudaMemcpy(d_maccsData, &h_maccsData, sizeof(MACCSData), cudaMemcpyHostToDevice);
//}
//
//void Gpuff::initializePuffs(
//    int input_num,
//    const std::vector<RadioNuclideTransport>& RT,
//    const std::vector<NuclideData>& ND
//) {
//    int totalPuffs = 0;
//
//    for (int i = 0; i < input_num; i++) {
//        totalPuffs += RT[i].nPuffTotal;
//    }
//
//    puffs_RCAP.reserve(RCAP_metdata.size() * totalPuffs);
//
//    double baseLon = RT[0].lon;
//    double baseLat = RT[0].lat;
//
//    const double metersPerLatDegree = 111320.0;
//    const double metersPerLonDegree = 88290.0;
//
//    printf("RCAP_metdata = %d\n", RCAP_metdata.size());
//
//    double xx1, yy1, xx3, yy3, xx5, yy5;
//
//    for (int i = 0; i < 6; i++) {
//        xx1 = (RT[i].lon - baseLon) * metersPerLonDegree;
//        yy1 = (RT[i].lat - baseLat) * metersPerLonDegree;
//        printf("[%d] x = %f, y = %f\n", i, xx1, yy1);
//    }
//
//    xx3 = (RT[3].lon - baseLon) * metersPerLonDegree;
//    yy3 = (RT[3].lat - baseLat) * metersPerLonDegree;
//    xx5 = (RT[5].lon - baseLon) * metersPerLonDegree;
//    yy5 = (RT[5].lat - baseLat) * metersPerLonDegree;
//
//    printf("dist = %f\n", sqrt((xx3 - xx5) * (xx3 - xx5) + (yy3 - yy5) * (yy3 - yy5)));
// 
//
//    for (size_t i = 0; i < RCAP_metdata.size(); i++) {
//        for (int j = 0; j < input_num; j++) {
//            for (int k = 0; k < RT[j].nPuffTotal; k++) {
//
//                //std::cout << RT[j].nPuffTotal << std::endl;
//
//                float _x = static_cast<float>((RT[j].lon - baseLon) * metersPerLonDegree);
//                float _y = static_cast<float>((RT[j].lat - baseLat) * metersPerLatDegree);
//                float _z = RT[j].RT_puffs[k].rele_height;
//
//                float _concentration[MAX_NUCLIDES];
//                    //RT[j].conc;
//                for (int nuc = 0; nuc < MAX_NUCLIDES; nuc++) {
//                    _concentration[nuc] = RT[j].conc[nuc] * RT[j].RT_puffs[k].release_fractions[ND[nuc].chemical_group];
//                }
//
//                float _releasetime = RT[j].RT_puffs[k].rele_time;
//                int _unitidx = j;
//                //std::cout << "input_num = " << input_num << std::endl;
//                //std::cout << "j = " << j << std::endl;
//
//                float _windvel = RCAP_metdata[i].spd;
//
//                int time_steps = static_cast<int>(_releasetime / 360); // Calculate the number of hours passed
//                int new_diridx = (i + time_steps) % RCAP_metdata.size(); // Use modulo to wrap around if we exceed the dataset
//
//                //if(i+k< RCAP_metdata.size()) new_diridx  = i + k;
//                //else new_diridx = i + k - RCAP_metdata.size();
//
//                float _windir = RCAP_metdata[new_diridx].dir;
//                //std::cout << "_windir = " << RCAP_metdata[i].dir << ", new_diridx = "<< RCAP_metdata[new_diridx].dir << std::endl;
//
//                int _stab = RCAP_metdata[i].stab;
//                float _rain = RCAP_metdata[i].rain;
//
//                //std::cout << "_concentration[0] =  " << _concentration[0] << std::endl;
//                //std::cout << "_concentration[1] =  " << _concentration[1] << std::endl;
//                //std::cout << "_concentration[2] =  " << _concentration[2] << std::endl;
//                //std::cout << "_concentration[3] =  " << _concentration[3] << std::endl;
//                //std::cout << "_concentration[4] =  " << _concentration[4] << std::endl;
//
//
//                puffs_RCAP.emplace_back(_x, _y, _z, 
//                    _concentration, _releasetime, _unitidx, _windvel, _windir, _stab, _rain, i);
//
//            }
//        }
//    }
//}

inline double toRadians(double degrees) {
    return degrees * PI / 180.0;
}

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

    double baseLon = RT[0].lon;
    double baseLat = RT[0].lat;

    printf("RCAP_metdata = %d\n", RCAP_metdata.size());

    //double lonDiff3 = toRadians(RT[3].lon - baseLon);
    //double latDiff3 = toRadians(RT[3].lat - baseLat);
    //double baseLatRad3 = toRadians(baseLat);

    //float _x3 = static_cast<float>(EARTH_RADIUS * lonDiff3 * cos(baseLatRad3));
    //float _y3 = static_cast<float>(EARTH_RADIUS * latDiff3);

    //double lonDiff5 = toRadians(RT[5].lon - baseLon);
    //double latDiff5 = toRadians(RT[5].lat - baseLat);
    //double baseLatRad5 = toRadians(baseLat);

    //float _x5 = static_cast<float>(EARTH_RADIUS * lonDiff5 * cos(baseLatRad5));
    //float _y5 = static_cast<float>(EARTH_RADIUS * latDiff5);

    //float dist = sqrt((_x3 - _x5) * (_x3 - _x5) + (_y3 - _y5) * (_y3 - _y5));
    //printf("dist = %f\n", dist);



    for (size_t i = 0; i < RCAP_metdata.size(); i++) {
        for (int j = 0; j < input_num; j++) {
            for (int k = 0; k < RT[j].nPuffTotal; k++) {

                double lonDiff = toRadians(RT[j].lon - baseLon);
                double latDiff = toRadians(RT[j].lat - baseLat);

                double baseLatRad = toRadians(baseLat);
                double targetLatRad = toRadians(RT[j].lat);

                float _y = static_cast<float>(EARTH_RADIUS * latDiff);
                float _x = static_cast<float>(EARTH_RADIUS * lonDiff * std::cos(baseLatRad));

                float _z = RT[j].RT_puffs[k].rele_height;

                float _concentration[MAX_NUCLIDES];
                for (int nuc = 0; nuc < MAX_NUCLIDES; nuc++) {
                    _concentration[nuc] = RT[j].conc[nuc] * RT[j].RT_puffs[k].release_fractions[ND[nuc].chemical_group];
                    if (_concentration[nuc]>1.0) std::cout << "nuc = " << nuc << ", conc = " << _concentration[nuc] << std::endl;
                }

                float _releasetime = RT[j].RT_puffs[k].rele_time;
                int _unitidx = j;

                float _windvel = RCAP_metdata[i].spd;
                int time_steps = static_cast<int>(_releasetime / 360);
                int new_diridx = (i + time_steps) % RCAP_metdata.size();

                float _windir = RCAP_metdata[new_diridx].dir;
                int _stab = RCAP_metdata[i].stab;
                float _rain = RCAP_metdata[i].rain;

                puffs_RCAP.emplace_back(_x, _y, _z,
                    _concentration, _releasetime, _unitidx, _windvel, _windir, _stab, _rain, i);
            }
        }
    }
}

void Gpuff::initializeEvacuees(std::vector<Evacuee>& evacuees, const SimulationControl& SC, 
    const EvacuationData& EP, const SiteData& SD) {

    int numRad = SC.numRad;
    int numTheta = SC.numTheta;

    for (int met = 0; met < RCAP_metdata.size(); met++) {
        for (int i = 0; i < numRad; ++i) {
            for (int j = 0; j < numTheta; ++j) {
                int population = SD.population[i][j];
                if (population > 0) {
                    Evacuee evacuee;
                    evacuee.population = static_cast<float>(population);
                    evacuee.r = SC.ir_distances[i];
                    evacuee.theta = (2 * PI * j / numTheta);
                    evacuee.speed = 0.0f;
                    evacuee.rad0 = i;
                    if(i > EP.evaEndRing-1) evacuee.flag = false;
                    else evacuee.flag = true;

                    evacuee.prev_rad_idx = i;
                    evacuee.prev_theta_idx = j;
                    evacuee.met_idx = met;

                    evacuees.push_back(evacuee);
                    if (met == 0) totalevacuees_per_Sim++;
                }
            }
        }
    }

}

//void Gpuff::initializeEvacuees_xy(std::vector<Evacuee>& evacuees, const SimulationControl& SC,
//    const EvacuationData& EP, const SiteData& SD) {
//
//    int NN = 500;
//
//    //printf("RCAP_metdata = %d\n", RCAP_metdata.size());
//
//
//    for (int met = 0; met < RCAP_metdata.size(); met++) {
//        for (int i = 0; i < NN; ++i) {
//            for (int j = 0; j < NN; ++j) {
//
//                if (1) {
//                    Evacuee evacuee;
//                    evacuee.population = 100;
//                    evacuee.speed = 0.0f;
//                    evacuee.flag = true;
//
//                    evacuee.prev_rad_idx = i;
//                    evacuee.prev_theta_idx = j;
//
//                    evacuee.x = (4000.0 - (-300.0)) * i / NN - 300.0;
//                    evacuee.y = (700.0 - (-3600.0)) * j / NN - 3600.0;
//
//                    evacuee.dose_inhalation = 1.0e-40;
//                    evacuee.dose_cloudshine = 1.0e-40;
//
//
//                    //printf("i = %d, j = %d\n", i, j);
//
//
//                    evacuees.push_back(evacuee);
//                    if (met == 0) totalevacuees_per_Sim++;
//                }
//            }
//        }
//    }
//
//}


//void Gpuff::initializeEvacuees_xy(std::vector<Evacuee>& evacuees, const SimulationControl& SC,
//    const EvacuationData& EP, const SiteData& SD) {
//
//    int NN = 500;
//
//    //printf("RCAP_metdata = %d\n", RCAP_metdata.size());
//
//
//    for (int met = 0; met < RCAP_metdata.size(); met++) {
//        for (int i = 0; i < 250; ++i) {
//            for (int j = 0; j < 1000; ++j) {
//
//                if (1) {
//                    Evacuee evacuee;
//                    evacuee.population = 100;
//                    evacuee.speed = 0.0f;
//                    evacuee.flag = true;
//
//                    evacuee.prev_rad_idx = i;
//                    evacuee.prev_theta_idx = j;
//
//                    evacuee.x = (1500.0 - (-1500.0)) * i / 250 - 1500.0;
//                    evacuee.y = (6000.0 - (-500.0)) * j / 1000 - 500.0;
//
//                    evacuee.dose_inhalation = 1.0e-40;
//                    evacuee.dose_cloudshine = 1.0e-40;
//
//
//                    //printf("i = %d, j = %d\n", i, j);
//
//
//                    evacuees.push_back(evacuee);
//                    if (met == 0) totalevacuees_per_Sim++;
//                }
//            }
//        }
//    }
//
//}

void Gpuff::initializeEvacuees_xy(std::vector<Evacuee>& evacuees, const SimulationControl& SC,
    const EvacuationData& EP, const SiteData& SD) {

    int NN = 500;

    //printf("RCAP_metdata = %d\n", RCAP_metdata.size());


    for (int met = 0; met < RCAP_metdata.size(); met++) {
        for (int i = 0; i < 1000; ++i) {
            for (int j = 0; j < 250; ++j) {

                if (1) {
                    Evacuee evacuee;
                    evacuee.population = 100;
                    evacuee.speed = 0.0f;
                    evacuee.flag = true;

                    evacuee.prev_rad_idx = i;
                    evacuee.prev_theta_idx = j;

                    evacuee.x = (500.0 - (-6000.0)) * i / 1000 - 6000.0;
                    evacuee.y = (1500.0 - (-1500.0)) * j / 250 - 1500.0;

                    evacuee.dose_inhalation = 1.0e-40;
                    evacuee.dose_cloudshine = 1.0e-40;


                    //printf("i = %d, j = %d\n", i, j);


                    evacuees.push_back(evacuee);
                    if (met == 0) totalevacuees_per_Sim++;
                }
            }
        }
    }

}
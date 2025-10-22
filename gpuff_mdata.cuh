#include "gpuff.cuh"

float Gpuff::Lambert2x(float LDAPS_LAT, float LDAPS_LON){

	float lam = LDAPS_LAT*PI/180.0;
	float phi = LDAPS_LON*PI/180.0;

	float lam0 = LDAPS_W*PI/180.0;
	float phi0 = LDAPS_S*PI/180.0;

	float phi1 = 29.0*PI/180.0;
	float phi2 = 45.0*PI/180.0;

	float n = log(cos(phi1)/cos(phi2))/log(tan(PI/4+phi2/2)/tan(PI/4+phi1/2));
	float R = 6371000.0;
	float F = cos(phi1)*pow(tan(PI/4+phi1/2),n)/n;
	float rho = F*R*pow(1/tan((PI/4+phi/2)),n);
	float rho0 = F*R*pow(1/tan((PI/4+phi0/2)),n);

	return rho*sin(n*(lam-lam0));

}

float Gpuff::Lambert2y(float LDAPS_LAT, float LDAPS_LON){

	float lam = LDAPS_LAT*PI/180.0;
	float phi = LDAPS_LON*PI/180.0;

	float lam0 = LDAPS_W*PI/180.0;
	float phi0 = LDAPS_S*PI/180.0;

	float phi1 = 29.0*PI/180.0;
	float phi2 = 45.0*PI/180.0;

	float n = log(cos(phi1)/cos(phi2))/log(tan(PI/4+phi2/2)/tan(PI/4+phi1/2));
	float R = 6371000.0;
	float F = cos(phi1)*pow(tan(PI/4+phi1/2),n)/n;
	float rho = F*R*pow(1/tan((PI/4+phi/2)),n);
	float rho0 = F*R*pow(1/tan((PI/4+phi0/2)),n);

	return rho0-rho*cos(n*(lam-lam0));

}

void Gpuff::read_meteorological_data(
    const char* filename_pres, 
    const char* filename_unis, 
    const char* filename_etas)
{

    #ifdef _WIN32
        const char* path = ".\\input\\ldapsdata\\";
    #else
        const char* path = "./input/ldapsdata/";
    #endif

    char filepath_pres[256], filepath_unis[256], filepath_etas[256];

    sprintf(filepath_pres, "%s%s", path, filename_pres);
    sprintf(filepath_unis, "%s%s", path, filename_unis);
    sprintf(filepath_etas, "%s%s", path, filename_etas);

    FILE* file_pres = fopen(filepath_pres, "rb");
    FILE* file_unis = fopen(filepath_unis, "rb");
    FILE* file_etas = fopen(filepath_etas, "rb");

    if(file_pres == 0) std::cerr << "Failed to open a PRES meteorological data." << std::endl;
    if(file_unis == 0) std::cerr << "Failed to open a UNIS meteorological data." << std::endl;
    if(file_etas == 0) std::cerr << "Failed to open a ETAS meteorological data." << std::endl;
    
    PresData* host_data_pres = new PresData[dimX * dimY * dimZ_pres];
    UnisData* host_data_unis = new UnisData[dimX * dimY];
    EtasData* host_data_etas = new EtasData[dimX * dimY * dimZ_etas];

    float val;
    int valt;

    int idx;
    int debug=0;
    int debug_=0;



 

    ///////////////////////////////////
    // READ PRES METEOROLOGICAL DATA //
    ///////////////////////////////////

    for(int k=0; k<dimZ_pres; k++){

        if(k>0)fread(&val, sizeof(float), 1, file_pres);
        fread(&val, sizeof(float), 1, file_pres);

        for(int j=0; j<dimY; j++){
            for(int i=0; i<dimX; i++){

                fread(&val, sizeof(float), 1, file_pres);
                if (val>1000000.0) 
                    val = 0.000010;

                debug++;
                idx = i*(dimY)*(dimZ_pres) + j*(dimZ_pres) + k;
                host_data_pres[idx].DZDT=val;

                if(CHECK_METDATA) 
                    if(debug<10 || debug > dimX*dimY*dimZ_pres-15) 
                        printf("DZDT[%d] = %f\n", idx, host_data_pres[idx].DZDT);
                
            }
        }
    }
    debug=0;

    for(int k=0; k<dimZ_pres; k++){
        for(int uvidx=0; uvidx<2; uvidx++){
            
            fread(&val, sizeof(float), 1, file_pres);
            fread(&val, sizeof(float), 1, file_pres);

            for(int j=0; j<dimY; j++){
                for(int i=0; i<dimX; i++){

                    fread(&val, sizeof(float), 1, file_pres);
                    //if(uvidx==0) printf("UGRD = %f\n", val);
                    //else printf("VGRD = %f\n", val);
                    if(val>1000000.0) 
                        val = 0.000011;

                    idx = i*(dimY)*(dimZ_pres) + j*(dimZ_pres) + k;

                    if(!uvidx){
                        debug++;
                        host_data_pres[idx].UGRD=val;
                    }
                    else{
                        debug_++;
                        host_data_pres[idx].VGRD=val;
                    }

                    if(CHECK_METDATA && !uvidx) 
                        if(debug<10 || debug > dimX*dimY*dimZ_pres-15) 
                            printf("UGRD[%d] = %f\n", idx, host_data_pres[idx].UGRD);

                    if(CHECK_METDATA && uvidx)
                        if(debug<10 || debug_ > dimX*dimY*dimZ_pres-15) 
                            printf("VGRD[%d] = %f\n", idx, host_data_pres[idx].VGRD);

                }
            }
        }
    }
    debug=0;
    debug_=0;

    for(int k=0; k<dimZ_pres; k++){

        fread(&val, sizeof(float), 1, file_pres);
        fread(&val, sizeof(float), 1, file_pres);

        for(int j=0; j<dimY; j++){
            for(int i=0; i<dimX; i++){

                fread(&val, sizeof(float), 1, file_pres);
                //printf("hgt = %f \n", val);
                if (val>1000000.0) 
                    val = 0.000012;

                debug++;
                idx = i*(dimY)*(dimZ_pres) + j*(dimZ_pres) + k;
                host_data_pres[idx].HGT=val;

                if(CHECK_METDATA) 
                    if(debug<10 || debug > dimX*dimY*dimZ_pres-15)  
                        printf("HGT[%d] = %f\n", idx, host_data_pres[idx].HGT);
            }
        }
    }
    debug=0;

    for(int k=0; k<dimZ_pres; k++){

        fread(&val, sizeof(float), 1, file_pres);
        fread(&val, sizeof(float), 1, file_pres);

        for(int j=0; j<dimY; j++){
            for(int i=0; i<dimX; i++){

                fread(&val, sizeof(float), 1, file_pres);
                //printf("TMP = %f \n", val);
                if (val>1000000.0) 
                    val = 0.000013;

                debug++;
                idx = i*(dimY)*(dimZ_pres) + j*(dimZ_pres) + k;
                host_data_pres[idx].TMP=val;

                if(CHECK_METDATA) 
                    if(debug<10 || debug > dimX*dimY*dimZ_pres-15)
                        printf("TMP[%d] = %f\n", idx, host_data_pres[idx].TMP);
                
            }
        }
    }
    debug=0;

    for(int k=0; k<dimZ_pres; k++){

        fread(&val, sizeof(float), 1, file_pres);
        fread(&val, sizeof(float), 1, file_pres);

        for(int j=0; j<dimY; j++){
            for(int i=0; i<dimX; i++){
                fread(&val, sizeof(float), 1, file_pres);    
            }
        }
    }
    debug=0;

    for(int k=0; k<dimZ_pres; k++){

        fread(&val, sizeof(float), 1, file_pres);
        fread(&val, sizeof(float), 1, file_pres);

        for(int j=0; j<dimY; j++){
            for(int i=0; i<dimX; i++){

                fread(&val, sizeof(float), 1, file_pres);
                //printf("RH = %f \n", val);
                if (val>1000000.0) 
                    val = 0.000013;

                debug++;
                idx = i*(dimY)*(dimZ_pres) + j*(dimZ_pres) + k;
                host_data_pres[idx].RH=val;

                if(CHECK_METDATA) 
                    if(debug<10 || debug > dimX*dimY*dimZ_pres-15)
                        printf("RH[%d] = %f\n", idx, host_data_pres[idx].RH);
                
            }
        }
    }
    debug=0;

    printf("PRES data loaded successfully.\n");





    ///////////////////////////////////
    // READ UNIS METEOROLOGICAL DATA //
    ///////////////////////////////////

    for(int varidx=1; varidx<137; varidx++){
        for(int j=0; j<dimY; j++){
            for(int i=0; i<dimX; i++){

                fread(&val, sizeof(float), 1, file_unis);

                if(varidx==12) // No.12 [HPBLA] Boundary Layer Depth after B. LAYER
                {
                    if (val>1000000.0) 
                        val = 0.000014;

                    debug++;
                    idx = i*(dimY) + j;
                    host_data_unis[idx].HPBLA=val;

                    if(CHECK_METDATA) 
                        if(debug<10 || debug > dimX*dimY-15) 
                            printf("HPBLA[%d] = %f\n", idx, host_data_unis[idx].HPBLA);
                  
                }

                if(varidx==21) // No.21 [TMP] Temperature at 1.5m above ground
                {
                    if (val>1000000.0) 
                        val = 0.000014;

                    debug++;
                    idx = i*(dimY) + j;
                    host_data_unis[idx].T1P5=val;

                    if(CHECK_METDATA) 
                        if(debug<10 || debug > dimX*dimY-15) 
                            printf("T1P5[%d] = %f\n", idx, host_data_unis[idx].T1P5);
                  
                }

                else if(varidx>33 && varidx<43) // No.39 [SHFLT] Surface Sensible Heat Flux on Tiles
                {
                    if (val>1000000.0) 
                        val = 0.000015;

                    debug++;
                    idx = i*(dimY) + j;
                    host_data_unis[idx].SHFLT+=val;

                    if(CHECK_METDATA) 
                        if(debug<10 || debug > dimX*dimY-15)
                            printf("SHFLT[%d] = %f\n", idx, host_data_unis[idx].SHFLT);
                }

                else if(varidx==43) // No.43 [HTBM] Turbulent mixing height after B. Layer
                {
                    if (val>1000000.0) 
                        val = 0.000016;

                    debug++;
                    idx = i*(dimY) + j;
                    host_data_unis[idx].HTBM=val;

                    if(CHECK_METDATA) 
                        if(debug<10 || debug > dimX*dimY-15) 
                            printf("HTBM[%d] = %f\n", idx, host_data_unis[idx].HTBM);
                }

                else if(varidx==131) // No.131 [HPBL] Planetary Boundary Layer Height
                {
                    if (val>1000000.0) 
                        val = 0.000017;

                    debug++;
                    idx = i*(dimY) + j;
                    host_data_unis[idx].HPBL=val;

                    if(CHECK_METDATA) 
                        if(debug<10 || debug > dimX*dimY-15) 
                            printf("HPBL[%d] = %f\n", idx, host_data_unis[idx].HPBL);
                }

                else if(varidx==132) // No.132 [SFCR] Surface Roughness
                {
                    if (val>1000000.0) 
                        val = 0.000018;

                    debug++;
                    idx = i*(dimY) + j;
                    host_data_unis[idx].SFCR=val;

                    if(CHECK_METDATA) 
                        if(debug<10 || debug > dimX*dimY-15) 
                            printf("SFCR[%d] = %f\n", idx, host_data_unis[idx].SFCR);
                }
            }
        }
        debug=0;
    }

    printf("UNIS data loaded successfully.\n");

    ///////////////////////////////////
    // READ ETAS METEOROLOGICAL DATA //
    ///////////////////////////////////

    for(int k=0; k<dimZ_etas-1; k++){ // UGRD, VGRD
        for(int uvidx=0; uvidx<2; uvidx++){
            for(int j=0; j<dimY; j++){
                for(int i=0; i<dimX; i++){

                    fread(&val, sizeof(float), 1, file_etas);
                    if(val>1000000.0) 
                        val = 0.000019;

                    idx = i*(dimY)*(dimZ_etas) + j*(dimZ_etas) + k;

                    if(!uvidx){
                        debug++;
                        host_data_etas[idx].UGRD=val;
                    }
                    else{
                        debug_++;
                        host_data_etas[idx].VGRD=val;
                    }

                    if(CHECK_METDATA && !uvidx) 
                        if(debug<10 || debug > dimX*dimY*(dimZ_etas-1)-15) 
                            printf("UGRD[%d] = %f\n", idx, host_data_etas[idx].UGRD);
                    if(CHECK_METDATA && uvidx) 
                        if(debug<10 || debug_ > dimX*dimY*(dimZ_etas-1)-15) 
                            printf("VGRD[%d] = %f\n", idx, host_data_etas[idx].VGRD);

                }
            }
        }
    }
    debug=0;
    debug_=0;

    for(int k=0; k<dimZ_etas-1; k++){ // POT
        for(int j=0; j<dimY; j++){
            for(int i=0; i<dimX; i++){
                fread(&val, sizeof(float), 1, file_etas);
            }
        }
    }

    for(int k=0; k<dimZ_etas-1; k++){ // SPFH
        for(int j=0; j<dimY; j++){
            for(int i=0; i<dimX; i++){
                fread(&val, sizeof(float), 1, file_etas);
            }
        }
    }

    for(int k=0; k<dimZ_etas-1; k++){ // QCF
        for(int j=0; j<dimY; j++){
            for(int i=0; i<dimX; i++){
                fread(&val, sizeof(float), 1, file_etas);
            }
        }
    }

    for(int k=0; k<dimZ_etas; k++){ // DZDT
        for(int j=0; j<dimY; j++){
            for(int i=0; i<dimX; i++){

                fread(&val, sizeof(float), 1, file_etas);
                if (val>1000000.0) 
                    val = 0.000020;

                debug++;
                idx = i*(dimY)*(dimZ_etas) + j*(dimZ_etas) + k;
                host_data_etas[idx].DZDT=val;

                if(CHECK_METDATA) 
                    if(debug<10 || debug > dimX*dimY*dimZ_etas-15) 
                        printf("DZDT[%d] = %f\n", idx, host_data_etas[idx].DZDT);

            }
        }
    }
    debug=0;

    for(int k=0; k<dimZ_etas; k++){ // DEN
        for(int j=0; j<dimY; j++){
            for(int i=0; i<dimX; i++){

                debug++;
                idx = i*(dimY)*(dimZ_etas) + j*(dimZ_etas) + k;

                if(k==dimZ_etas-1){
                    host_data_etas[idx].DEN=host_data_etas[idx-1].DEN;
                    break;
                }

                fread(&val, sizeof(float), 1, file_etas);
                host_data_etas[idx].DEN=val;

                if(CHECK_METDATA) 
                    if(debug<10 || debug > dimX*dimY*dimZ_etas-15)  
                        printf("DEN[%d] = %f\n", idx, host_data_etas[idx].DEN);
            }
        }
    }
    debug=0;

    printf("ETAS data loaded successfully.\n");


    fclose(file_pres);
    fclose(file_unis);
    fclose(file_etas);

    cudaMalloc((void**)&device_meteorological_data_pres, 
    dimX * dimY * dimZ_pres * sizeof(PresData));
    cudaMalloc((void**)&device_meteorological_data_unis, 
    dimX * dimY * sizeof(UnisData));
    cudaMalloc((void**)&device_meteorological_data_etas, 
    dimX * dimY * dimZ_etas * sizeof(EtasData));
    
    cudaMemcpy(device_meteorological_data_pres, 
        host_data_pres, dimX * dimY * dimZ_pres * sizeof(PresData), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_data_unis, 
        host_data_unis, dimX * dimY * sizeof(UnisData), 
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_data_etas, 
        host_data_etas, dimX * dimY * dimZ_etas * sizeof(EtasData), 
        cudaMemcpyHostToDevice);

        
    delete[] host_data_pres;
    delete[] host_data_unis;
    delete[] host_data_etas;

}


void Gpuff::read_meteorological_data_RCAP(){
    std::ifstream file(".\\input\\RCAPdata\\METEO.inp");
    std::string line;

    std::getline(file, line);
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int num1, num2, num3, num4, num5;
        if (!(iss >> num1 >> num2 >> num3 >> num4 >> num5)) { break; }

        int last = num4 % 10;

        RCAP_windir.push_back(static_cast<float>(num3)*PI/8.0f);
        RCAP_winvel.push_back(static_cast<float>(num4/10)/10.0f); // (m/s)
        RCAP_stab.push_back(static_cast<int>(last));

    }

    // for(int i=0; i<32; i++){
    //     std::cout << RCAP_windir[i] << " ";
    // }
    // std::cout << std::endl;

    cudaMalloc((void**)&d_RCAP_windir, RCAP_windir.size() * sizeof(float));
    cudaMalloc((void**)&d_RCAP_winvel, RCAP_winvel.size() * sizeof(float));
    cudaMalloc((void**)&d_RCAP_stab, RCAP_stab.size() * sizeof(int));

    cudaMemcpy(d_RCAP_windir, RCAP_windir.data(), RCAP_windir.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RCAP_winvel, RCAP_winvel.data(), RCAP_winvel.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RCAP_stab, RCAP_stab.data(), RCAP_stab.size() * sizeof(int), cudaMemcpyHostToDevice);

}

//void Gpuff::read_meteorological_data_RCAP2() {
//    std::ifstream file(".\\input\\RCAPdata\\METEO.inp");
//    std::string line;
//
//    std::getline(file, line);
//    std::getline(file, line);
//
//    while (std::getline(file, line)) {
//        std::istringstream iss(line);
//        int day, hour, dir, spd, rain;
//        if (!(iss >> day >> hour >> dir >> spd)) { break; }
//
//        if (dir >= 100) {
//            rain = spd;
//            spd = dir % 10000;
//            dir = dir / 10000;
//        }
//        else {
//            iss >> rain;
//        }
//
//        int stab = spd % 10;
//        spd = spd / 10;
//
//        float dir_data = static_cast<float>(dir) * PI / 8.0f;
//        float spd_data = static_cast<float>(spd) / 10.0f;
//
//        RCAP_METDATA a = { day, hour, dir_data, spd_data, stab, static_cast<float>(rain) };
//        RCAP_metdata.push_back(a);
//
//    }
//
//    if (CHECK_METDATA) for (const auto& data : RCAP_metdata) {
//        std::cout << "day: " << data.day << " - " << data.hour << "\t\t"
//            << "dir: " << data.dir << "\t\t"
//            << "spd: " << data.spd << "   \t"
//            << "stab: " << data.stab << "\t\t"
//            << "rain: " << data.rain << std::endl;
//    }
//
//}

void Gpuff::read_meteorological_data_RCAP2(const std::string& filename) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    //std::ifstream file(".\\input\\RCAPdata\\METEO.inp");
    std::string line;

    std::getline(file, line);
    std::getline(file, line);

    while (std::getline(file, line)) {
        if (line.length() > 18) {
            //std::cerr << "Terminating data read due to invalid line: " << line << std::endl;
            break;
        }

        int day = std::stoi(line.substr(1, 3));   // Julian day (2-4)
        int hour = std::stoi(line.substr(5, 2));  // Hour of the day (6-7)
        int dir = std::stoi(line.substr(8, 2));   // Wind direction (9-10)
        int spd = std::stoi(line.substr(10, 3));  // Wind speed (11-13)
        int stab = std::stoi(line.substr(13, 1)); // Stability category (14)
        int rain = std::stoi(line.substr(14, 3)); // Accumulated precipitation (15-17)

        if (spd >= 1 && spd <= 4) {
            spd = 5;
        }

        float dir_data = static_cast<float>(dir) * PI / 8.0f;
        float spd_data = static_cast<float>(spd) / 10.0f;
        float rain_data = static_cast<float>(rain) / 10.0f;

        RCAP_METDATA a = { day, hour, dir_data, spd_data, stab, rain_data };
        RCAP_metdata.push_back(a);
    }

    numSims = RCAP_metdata.size();

    if (CHECK_METDATA) {
        for (const auto& data : RCAP_metdata) {
            std::cout << "day: " << data.day << " - " << data.hour << "\t\t"
                << "dir: " << data.dir << "\t\t"
                << "spd: " << data.spd << "   \t"
                << "stab: " << data.stab << "\t\t"
                << "rain: " << data.rain << std::endl;
        }
    }
}

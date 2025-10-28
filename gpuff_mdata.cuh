#include "gpuff.cuh"

// Lambert Conformal Conic Projection: Latitude/Longitude to X coordinate
//
// Converts geographic coordinates (lat/lon in degrees) to Lambert projection X coordinate (meters).
// Uses a Lambert Conformal Conic projection with two standard parallels.
//
// Mathematical formula:
//   n = ln(cos(phi1) / cos(phi2)) / ln(tan(pi/4 + phi2/2) / tan(pi/4 + phi1/2))
//   F = cos(phi1) * tan(pi/4 + phi1/2)^n / n
//   rho = F * R * (1/tan(pi/4 + phi/2))^n
//   x = rho * sin(n * (lambda - lambda0))
//
// Parameters:
//   LDAPS_LAT: Input latitude in degrees
//   LDAPS_LON: Input longitude in degrees
//
// Returns:
//   X coordinate in meters from the projection origin
//
// Projection parameters:
//   Standard parallels: phi1 = 29 degrees N, phi2 = 45 degrees N (Korea region)
//   Origin: (phi0, lambda0) = (LDAPS_S, LDAPS_W) in degrees
//   Earth radius: R = 6371000 meters (mean spherical Earth)
//
float Gpuff::Lambert2x(float LDAPS_LAT, float LDAPS_LON) {

    float lam = LDAPS_LAT * PI / 180.0;
    float phi = LDAPS_LON * PI / 180.0;

    float lam0 = LDAPS_W * PI / 180.0;
    float phi0 = LDAPS_S * PI / 180.0;

    float phi1 = 29.0 * PI / 180.0;
    float phi2 = 45.0 * PI / 180.0;

    float n = log(cos(phi1) / cos(phi2)) / log(tan(PI / 4 + phi2 / 2) / tan(PI / 4 + phi1 / 2));
    float R = 6371000.0;
    float F = cos(phi1) * pow(tan(PI / 4 + phi1 / 2), n) / n;
    float rho = F * R * pow(1 / tan((PI / 4 + phi / 2)), n);
    float rho0 = F * R * pow(1 / tan((PI / 4 + phi0 / 2)), n);

    return rho * sin(n * (lam - lam0));
}

// Lambert Conformal Conic Projection: Latitude/Longitude to Y coordinate
//
// Converts geographic coordinates (lat/lon in degrees) to Lambert projection Y coordinate (meters).
// Uses a Lambert Conformal Conic projection with two standard parallels.
//
// Mathematical formula:
//   n = ln(cos(phi1) / cos(phi2)) / ln(tan(pi/4 + phi2/2) / tan(pi/4 + phi1/2))
//   F = cos(phi1) * tan(pi/4 + phi1/2)^n / n
//   rho = F * R * (1/tan(pi/4 + phi/2))^n
//   rho0 = F * R * (1/tan(pi/4 + phi0/2))^n
//   y = rho0 - rho * cos(n * (lambda - lambda0))
//
// Parameters:
//   LDAPS_LAT: Input latitude in degrees
//   LDAPS_LON: Input longitude in degrees
//
// Returns:
//   Y coordinate in meters from the projection origin
//
// Projection parameters:
//   Standard parallels: phi1 = 29 degrees N, phi2 = 45 degrees N (Korea region)
//   Origin: (phi0, lambda0) = (LDAPS_S, LDAPS_W) in degrees
//   Earth radius: R = 6371000 meters (mean spherical Earth)
//
float Gpuff::Lambert2y(float LDAPS_LAT, float LDAPS_LON) {

    float lam = LDAPS_LAT * PI / 180.0;
    float phi = LDAPS_LON * PI / 180.0;

    float lam0 = LDAPS_W * PI / 180.0;
    float phi0 = LDAPS_S * PI / 180.0;

    float phi1 = 29.0 * PI / 180.0;
    float phi2 = 45.0 * PI / 180.0;

    float n = log(cos(phi1) / cos(phi2)) / log(tan(PI / 4 + phi2 / 2) / tan(PI / 4 + phi1 / 2));
    float R = 6371000.0;
    float F = cos(phi1) * pow(tan(PI / 4 + phi1 / 2), n) / n;
    float rho = F * R * pow(1 / tan((PI / 4 + phi / 2)), n);
    float rho0 = F * R * pow(1 / tan((PI / 4 + phi0 / 2)), n);

    return rho0 - rho * cos(n * (lam - lam0));
}

// Read LDAPS Meteorological Data from Binary Files
//
// Loads three types of meteorological data files (PRES, UNIS, ETAS) used in LDAPS
// (Local Data Assimilation and Prediction System) format and transfers to GPU memory.
//
// File Format:
//   All files are binary format with float32 values
//   PRES: Pressure-level data (3D: dimX x dimY x dimZ_pres)
//   UNIS: Surface/uniform data (2D: dimX x dimY)
//   ETAS: Eta-coordinate data (3D: dimX x dimY x dimZ_etas)
//
// PRES Data Variables (pressure levels):
//   - DZDT: Vertical velocity (m/s)
//   - UGRD: U-component wind (m/s, eastward positive)
//   - VGRD: V-component wind (m/s, northward positive)
//   - HGT: Geopotential height (m)
//   - TMP: Temperature (K)
//   - RH: Relative humidity (%)
//
// UNIS Data Variables (surface):
//   - HPBLA: Boundary layer depth after B. Layer (m)
//   - T1P5: Temperature at 1.5m above ground (K)
//   - SHFLT: Surface sensible heat flux on tiles (W/m^2)
//   - HTBM: Turbulent mixing height after B. Layer (m)
//   - HPBL: Planetary boundary layer height (m)
//   - SFCR: Surface roughness (m)
//
// ETAS Data Variables (eta coordinates):
//   - UGRD: U-component wind (m/s)
//   - VGRD: V-component wind (m/s)
//   - DZDT: Vertical velocity (m/s)
//   - DEN: Air density (kg/m^3)
//
// Data Validation:
//   Invalid values (>1000000.0) are replaced with small non-zero values to prevent
//   numerical errors while maintaining data integrity checks.
//
// Parameters:
//   filename_pres: PRES data filename
//   filename_unis: UNIS data filename
//   filename_etas: ETAS data filename
//
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

    if (file_pres == 0) std::cerr << "Failed to open a PRES meteorological data." << std::endl;
    if (file_unis == 0) std::cerr << "Failed to open a UNIS meteorological data." << std::endl;
    if (file_etas == 0) std::cerr << "Failed to open a ETAS meteorological data." << std::endl;

    PresData* host_data_pres = new PresData[dimX * dimY * dimZ_pres];
    UnisData* host_data_unis = new UnisData[dimX * dimY];
    EtasData* host_data_etas = new EtasData[dimX * dimY * dimZ_etas];

    float val;
    int valt;

    int idx;
    int debug = 0;
    int debug_ = 0;





    // Read PRES Meteorological Data (Pressure Level Data)
    // Data structure: 3D grid [dimX x dimY x dimZ_pres]
    // Index mapping: idx = i * dimY * dimZ_pres + j * dimZ_pres + k

    // Read DZDT (Vertical Velocity in m/s)
    for (int k = 0; k < dimZ_pres; k++) {

        if (k > 0) fread(&val, sizeof(float), 1, file_pres);
        fread(&val, sizeof(float), 1, file_pres);

        for (int j = 0; j < dimY; j++) {
            for (int i = 0; i < dimX; i++) {

                fread(&val, sizeof(float), 1, file_pres);
                if (val > 1000000.0)
                    val = 0.000010;

                debug++;
                idx = i * (dimY) * (dimZ_pres) + j * (dimZ_pres) + k;
                host_data_pres[idx].DZDT = val;

                if (CHECK_METDATA)
                    if (debug < 10 || debug > dimX * dimY * dimZ_pres - 15)
                        printf("DZDT[%d] = %f\n", idx, host_data_pres[idx].DZDT);
            }
        }
    }
    debug = 0;

    // Read UGRD and VGRD (Wind Components in m/s)
    // UGRD: U-component (eastward wind), VGRD: V-component (northward wind)
    for (int k = 0; k < dimZ_pres; k++) {
        for (int uvidx = 0; uvidx < 2; uvidx++) {

            fread(&val, sizeof(float), 1, file_pres);
            fread(&val, sizeof(float), 1, file_pres);

            for (int j = 0; j < dimY; j++) {
                for (int i = 0; i < dimX; i++) {

                    fread(&val, sizeof(float), 1, file_pres);
                    if (val > 1000000.0)
                        val = 0.000011;

                    idx = i * (dimY) * (dimZ_pres) + j * (dimZ_pres) + k;

                    if (!uvidx) {
                        debug++;
                        host_data_pres[idx].UGRD = val;
                    }
                    else {
                        debug_++;
                        host_data_pres[idx].VGRD = val;
                    }

                    if (CHECK_METDATA && !uvidx)
                        if (debug < 10 || debug > dimX * dimY * dimZ_pres - 15)
                            printf("UGRD[%d] = %f\n", idx, host_data_pres[idx].UGRD);

                    if (CHECK_METDATA && uvidx)
                        if (debug < 10 || debug_ > dimX * dimY * dimZ_pres - 15)
                            printf("VGRD[%d] = %f\n", idx, host_data_pres[idx].VGRD);
                }
            }
        }
    }
    debug = 0;
    debug_ = 0;

    // Read HGT (Geopotential Height in meters)
    for (int k = 0; k < dimZ_pres; k++) {

        fread(&val, sizeof(float), 1, file_pres);
        fread(&val, sizeof(float), 1, file_pres);

        for (int j = 0; j < dimY; j++) {
            for (int i = 0; i < dimX; i++) {

                fread(&val, sizeof(float), 1, file_pres);
                if (val > 1000000.0)
                    val = 0.000012;

                debug++;
                idx = i * (dimY) * (dimZ_pres) + j * (dimZ_pres) + k;
                host_data_pres[idx].HGT = val;

                if (CHECK_METDATA)
                    if (debug < 10 || debug > dimX * dimY * dimZ_pres - 15)
                        printf("HGT[%d] = %f\n", idx, host_data_pres[idx].HGT);
            }
        }
    }
    debug = 0;

    // Read TMP (Temperature in Kelvin)
    for (int k = 0; k < dimZ_pres; k++) {

        fread(&val, sizeof(float), 1, file_pres);
        fread(&val, sizeof(float), 1, file_pres);

        for (int j = 0; j < dimY; j++) {
            for (int i = 0; i < dimX; i++) {

                fread(&val, sizeof(float), 1, file_pres);
                if (val > 1000000.0)
                    val = 0.000013;

                debug++;
                idx = i * (dimY) * (dimZ_pres) + j * (dimZ_pres) + k;
                host_data_pres[idx].TMP = val;

                if (CHECK_METDATA)
                    if (debug < 10 || debug > dimX * dimY * dimZ_pres - 15)
                        printf("TMP[%d] = %f\n", idx, host_data_pres[idx].TMP);
            }
        }
    }
    debug = 0;

    // Skip unused variable (read but not stored)
    for (int k = 0; k < dimZ_pres; k++) {

        fread(&val, sizeof(float), 1, file_pres);
        fread(&val, sizeof(float), 1, file_pres);

        for (int j = 0; j < dimY; j++) {
            for (int i = 0; i < dimX; i++) {
                fread(&val, sizeof(float), 1, file_pres);
            }
        }
    }
    debug = 0;

    // Read RH (Relative Humidity in percent)
    for (int k = 0; k < dimZ_pres; k++) {

        fread(&val, sizeof(float), 1, file_pres);
        fread(&val, sizeof(float), 1, file_pres);

        for (int j = 0; j < dimY; j++) {
            for (int i = 0; i < dimX; i++) {

                fread(&val, sizeof(float), 1, file_pres);
                if (val > 1000000.0)
                    val = 0.000013;

                debug++;
                idx = i * (dimY) * (dimZ_pres) + j * (dimZ_pres) + k;
                host_data_pres[idx].RH = val;

                if (CHECK_METDATA)
                    if (debug < 10 || debug > dimX * dimY * dimZ_pres - 15)
                        printf("RH[%d] = %f\n", idx, host_data_pres[idx].RH);
            }
        }
    }
    debug = 0;

    printf("PRES data loaded successfully.\n");





    // Read UNIS Meteorological Data (Surface/Uniform Data)
    // Data structure: 2D grid [dimX x dimY]
    // Index mapping: idx = i * dimY + j
    // UNIS file contains 137 variables, only specific ones are extracted

    for (int varidx = 1; varidx < 137; varidx++) {
        for (int j = 0; j < dimY; j++) {
            for (int i = 0; i < dimX; i++) {

                fread(&val, sizeof(float), 1, file_unis);

                // Variable 12: HPBLA - Boundary Layer Depth after B. Layer (m)
                if (varidx == 12) {
                    if (val > 1000000.0)
                        val = 0.000014;

                    debug++;
                    idx = i * (dimY) + j;
                    host_data_unis[idx].HPBLA = val;

                    if (CHECK_METDATA)
                        if (debug < 10 || debug > dimX * dimY - 15)
                            printf("HPBLA[%d] = %f\n", idx, host_data_unis[idx].HPBLA);
                }

                // Variable 21: T1P5 - Temperature at 1.5m above ground (K)
                if (varidx == 21) {
                    if (val > 1000000.0)
                        val = 0.000014;

                    debug++;
                    idx = i * (dimY) + j;
                    host_data_unis[idx].T1P5 = val;

                    if (CHECK_METDATA)
                        if (debug < 10 || debug > dimX * dimY - 15)
                            printf("T1P5[%d] = %f\n", idx, host_data_unis[idx].T1P5);
                }

                // Variables 34-42: SHFLT - Surface Sensible Heat Flux on Tiles (W/m^2)
                // Accumulated over multiple tile types
                else if (varidx > 33 && varidx < 43) {
                    if (val > 1000000.0)
                        val = 0.000015;

                    debug++;
                    idx = i * (dimY) + j;
                    host_data_unis[idx].SHFLT += val;

                    if (CHECK_METDATA)
                        if (debug < 10 || debug > dimX * dimY - 15)
                            printf("SHFLT[%d] = %f\n", idx, host_data_unis[idx].SHFLT);
                }

                // Variable 43: HTBM - Turbulent Mixing Height after B. Layer (m)
                else if (varidx == 43) {
                    if (val > 1000000.0)
                        val = 0.000016;

                    debug++;
                    idx = i * (dimY) + j;
                    host_data_unis[idx].HTBM = val;

                    if (CHECK_METDATA)
                        if (debug < 10 || debug > dimX * dimY - 15)
                            printf("HTBM[%d] = %f\n", idx, host_data_unis[idx].HTBM);
                }

                // Variable 131: HPBL - Planetary Boundary Layer Height (m)
                else if (varidx == 131) {
                    if (val > 1000000.0)
                        val = 0.000017;

                    debug++;
                    idx = i * (dimY) + j;
                    host_data_unis[idx].HPBL = val;

                    if (CHECK_METDATA)
                        if (debug < 10 || debug > dimX * dimY - 15)
                            printf("HPBL[%d] = %f\n", idx, host_data_unis[idx].HPBL);
                }

                // Variable 132: SFCR - Surface Roughness (m)
                else if (varidx == 132) {
                    if (val > 1000000.0)
                        val = 0.000018;

                    debug++;
                    idx = i * (dimY) + j;
                    host_data_unis[idx].SFCR = val;

                    if (CHECK_METDATA)
                        if (debug < 10 || debug > dimX * dimY - 15)
                            printf("SFCR[%d] = %f\n", idx, host_data_unis[idx].SFCR);
                }
            }
        }
        debug = 0;
    }

    printf("UNIS data loaded successfully.\n");

    // Read ETAS Meteorological Data (Eta-Coordinate Data)
    // Data structure: 3D grid [dimX x dimY x dimZ_etas]
    // Index mapping: idx = i * dimY * dimZ_etas + j * dimZ_etas + k
    // Eta coordinates follow terrain-following vertical coordinates

    // Read UGRD and VGRD (Wind Components in m/s)
    for (int k = 0; k < dimZ_etas - 1; k++) {
        for (int uvidx = 0; uvidx < 2; uvidx++) {
            for (int j = 0; j < dimY; j++) {
                for (int i = 0; i < dimX; i++) {

                    fread(&val, sizeof(float), 1, file_etas);
                    if (val > 1000000.0)
                        val = 0.000019;

                    idx = i * (dimY) * (dimZ_etas) + j * (dimZ_etas) + k;

                    if (!uvidx) {
                        debug++;
                        host_data_etas[idx].UGRD = val;
                    }
                    else {
                        debug_++;
                        host_data_etas[idx].VGRD = val;
                    }

                    if (CHECK_METDATA && !uvidx)
                        if (debug < 10 || debug > dimX * dimY * (dimZ_etas - 1) - 15)
                            printf("UGRD[%d] = %f\n", idx, host_data_etas[idx].UGRD);
                    if (CHECK_METDATA && uvidx)
                        if (debug < 10 || debug_ > dimX * dimY * (dimZ_etas - 1) - 15)
                            printf("VGRD[%d] = %f\n", idx, host_data_etas[idx].VGRD);
                }
            }
        }
    }
    debug = 0;
    debug_ = 0;

    // Skip POT (Potential Temperature) - read but not stored
    for (int k = 0; k < dimZ_etas - 1; k++) {
        for (int j = 0; j < dimY; j++) {
            for (int i = 0; i < dimX; i++) {
                fread(&val, sizeof(float), 1, file_etas);
            }
        }
    }

    // Skip SPFH (Specific Humidity) - read but not stored
    for (int k = 0; k < dimZ_etas - 1; k++) {
        for (int j = 0; j < dimY; j++) {
            for (int i = 0; i < dimX; i++) {
                fread(&val, sizeof(float), 1, file_etas);
            }
        }
    }

    // Skip QCF (Cloud Fraction) - read but not stored
    for (int k = 0; k < dimZ_etas - 1; k++) {
        for (int j = 0; j < dimY; j++) {
            for (int i = 0; i < dimX; i++) {
                fread(&val, sizeof(float), 1, file_etas);
            }
        }
    }

    // Read DZDT (Vertical Velocity in m/s)
    for (int k = 0; k < dimZ_etas; k++) {
        for (int j = 0; j < dimY; j++) {
            for (int i = 0; i < dimX; i++) {

                fread(&val, sizeof(float), 1, file_etas);
                if (val > 1000000.0)
                    val = 0.000020;

                debug++;
                idx = i * (dimY) * (dimZ_etas) + j * (dimZ_etas) + k;
                host_data_etas[idx].DZDT = val;

                if (CHECK_METDATA)
                    if (debug < 10 || debug > dimX * dimY * dimZ_etas - 15)
                        printf("DZDT[%d] = %f\n", idx, host_data_etas[idx].DZDT);
            }
        }
    }
    debug = 0;

    // Read DEN (Air Density in kg/m^3)
    // Top level is extrapolated from level below
    for (int k = 0; k < dimZ_etas; k++) {
        for (int j = 0; j < dimY; j++) {
            for (int i = 0; i < dimX; i++) {

                debug++;
                idx = i * (dimY) * (dimZ_etas) + j * (dimZ_etas) + k;

                if (k == dimZ_etas - 1) {
                    host_data_etas[idx].DEN = host_data_etas[idx - 1].DEN;
                    break;
                }

                fread(&val, sizeof(float), 1, file_etas);
                host_data_etas[idx].DEN = val;

                if (CHECK_METDATA)
                    if (debug < 10 || debug > dimX * dimY * dimZ_etas - 15)
                        printf("DEN[%d] = %f\n", idx, host_data_etas[idx].DEN);
            }
        }
    }
    debug = 0;

    printf("ETAS data loaded successfully.\n");


    fclose(file_pres);
    fclose(file_unis);
    fclose(file_etas);

    // Allocate GPU memory for meteorological data
    cudaMalloc((void**)&device_meteorological_data_pres,
        dimX * dimY * dimZ_pres * sizeof(PresData));
    cudaMalloc((void**)&device_meteorological_data_unis,
        dimX * dimY * sizeof(UnisData));
    cudaMalloc((void**)&device_meteorological_data_etas,
        dimX * dimY * dimZ_etas * sizeof(EtasData));

    // Transfer meteorological data from host to device
    cudaMemcpy(device_meteorological_data_pres,
        host_data_pres, dimX * dimY * dimZ_pres * sizeof(PresData),
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_data_unis,
        host_data_unis, dimX * dimY * sizeof(UnisData),
        cudaMemcpyHostToDevice);
    cudaMemcpy(device_meteorological_data_etas,
        host_data_etas, dimX * dimY * dimZ_etas * sizeof(EtasData),
        cudaMemcpyHostToDevice);

    // Free host memory after GPU transfer
    delete[] host_data_pres;
    delete[] host_data_unis;
    delete[] host_data_etas;
}


// Read RCAP Meteorological Data (Legacy Format)
//
// Reads simplified meteorological data from RCAP METEO.inp file.
// This is a legacy function, superseded by read_meteorological_data_RCAP2().
//
// File Format:
//   Text file with 5 columns per line
//   Column 1: num1 (unused)
//   Column 2: num2 (unused)
//   Column 3: Wind direction (in 16-point compass, 0-15)
//   Column 4: Wind speed encoded with stability (format: speed*10 + stability)
//   Column 5: num5 (unused)
//
// Wind Direction Conversion:
//   Input: 16-point compass direction (0-15)
//   Output: Radians (value * PI / 8)
//
// Wind Speed Extraction:
//   Input format: XXXS where XXX is speed*10, S is stability class
//   Speed extraction: (num4 / 10) / 10 to get m/s
//
// Stability Class:
//   Extracted from last digit of num4
//
void Gpuff::read_meteorological_data_RCAP() {
    std::ifstream file(".\\input\\RCAPdata\\METEO.inp");
    std::string line;

    std::getline(file, line);
    std::getline(file, line);

    while (std::getline(file, line)) {
        std::istringstream iss(line);
        int num1, num2, num3, num4, num5;
        if (!(iss >> num1 >> num2 >> num3 >> num4 >> num5)) { break; }

        int last = num4 % 10;

        RCAP_windir.push_back(static_cast<float>(num3) * PI / 8.0f);
        RCAP_winvel.push_back(static_cast<float>(num4 / 10) / 10.0f); // (m/s)
        RCAP_stab.push_back(static_cast<int>(last));
    }

    // Allocate GPU memory and transfer RCAP meteorological data
    cudaMalloc((void**)&d_RCAP_windir, RCAP_windir.size() * sizeof(float));
    cudaMalloc((void**)&d_RCAP_winvel, RCAP_winvel.size() * sizeof(float));
    cudaMalloc((void**)&d_RCAP_stab, RCAP_stab.size() * sizeof(int));

    cudaMemcpy(d_RCAP_windir, RCAP_windir.data(), RCAP_windir.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RCAP_winvel, RCAP_winvel.data(), RCAP_winvel.size() * sizeof(float), cudaMemcpyHostToDevice);
    cudaMemcpy(d_RCAP_stab, RCAP_stab.data(), RCAP_stab.size() * sizeof(int), cudaMemcpyHostToDevice);
}

// Read RCAP Meteorological Data (Fixed-Format Version)
//
// Reads meteorological data from fixed-format RCAP METEO.inp file.
// Uses fixed-column positions for robust parsing.
//
// File Format (Fixed Columns):
//   Positions 2-4:   Julian day (001-365)
//   Positions 6-7:   Hour of day (00-23)
//   Positions 9-10:  Wind direction (16-point compass, 00-15)
//   Positions 11-13: Wind speed with stability (format: SSS where last digit is stability)
//   Position 14:     Stability class (1-7)
//   Positions 15-17: Accumulated precipitation (tenths of mm)
//
// Wind Direction:
//   16-point compass rose (0=N, 4=E, 8=S, 12=W)
//   Converted to radians: direction * PI / 8
//
// Wind Speed:
//   Encoded in positions 11-13, decoded to m/s by dividing by 10
//   Minimum speed adjustment: if speed 1-4, set to 0.5 m/s
//
// Stability Class:
//   Pasquill-Gifford classes (1=A very unstable to 7=G very stable)
//
// Precipitation:
//   Accumulated precipitation in mm (value / 10)
//
// Data Storage:
//   Each hourly record stored in RCAP_metdata vector
//   Number of simulations set to total records (numSims)
//
void Gpuff::read_meteorological_data_RCAP2(const std::string& filename) {

    std::ifstream file(filename);
    if (!file.is_open()) {
        std::cerr << "Error: Could not open file " << filename << std::endl;
        exit(EXIT_FAILURE);
    }

    std::string line;

    // Skip header lines
    std::getline(file, line);
    std::getline(file, line);

    // Read meteorological data records
    while (std::getline(file, line)) {
        // Stop if line is too long (indicates end of valid data)
        if (line.length() > 18) {
            break;
        }

        // Parse fixed-format fields
        int day = std::stoi(line.substr(1, 3));   // Julian day (columns 2-4)
        int hour = std::stoi(line.substr(5, 2));  // Hour (columns 6-7)
        int dir = std::stoi(line.substr(8, 2));   // Wind direction (columns 9-10)
        int spd = std::stoi(line.substr(10, 3));  // Wind speed (columns 11-13)
        int stab = std::stoi(line.substr(13, 1)); // Stability class (column 14)
        int rain = std::stoi(line.substr(14, 3)); // Precipitation (columns 15-17)

        // Apply minimum wind speed threshold
        if (spd >= 1 && spd <= 4) {
            spd = 5;
        }

        // Convert to physical units
        float dir_data = static_cast<float>(dir) * PI / 8.0f;  // Convert to radians
        float spd_data = static_cast<float>(spd) / 10.0f;      // Convert to m/s
        float rain_data = static_cast<float>(rain) / 10.0f;    // Convert to mm

        RCAP_METDATA a = { day, hour, dir_data, spd_data, stab, rain_data };
        RCAP_metdata.push_back(a);
    }

    numSims = RCAP_metdata.size();

    // Debug output if requested
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

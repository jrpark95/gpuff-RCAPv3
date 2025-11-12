/**
 * GPUFF Structure Definitions
 *
 * Contains all data structures and constants used by the Gaussian Puff
 * atmospheric dispersion model and RCAP consequence assessment system.
 *
 * Key Components:
 * - Meteorological data structures (Pres, Unis, Etas)
 * - Simulation configuration structures
 * - Radionuclide transport and decay data
 * - Evacuation and health effect parameters
 * - Grid and receptor definitions
 */

#pragma once

// Lambert conformal projection bounds for Korean Peninsula
#define LDAPS_E 132.36   // Eastern longitude
#define LDAPS_W 121.06   // Western longitude
#define LDAPS_N 43.13    // Northern latitude
#define LDAPS_S 32.20    // Southern latitude

// Mathematical constants
#define PI 3.141592
#define invPI 0.31831    // 1/PI

// Meteorological grid dimensions
#define dimX 602         // Longitude grid points
#define dimY 781         // Latitude grid points
#define dimZ_pres 24     // Pressure level vertical layers
#define dimZ_etas 71     // ETAS model vertical layers

// Simulation limits
#define INPUT_NUM 1              // Number of input files
#define MAX_STRING_LENGTH 32     // Maximum string length for names
#define MAX_NUCLIDES 80          // Maximum number of radionuclides
#define MAX_ORGANS 20            // Maximum number of body organs for dose calculation
#define DATA_FIELDS 5            // Dose conversion factor data fields
#define MAX_DNUC 2               // Maximum daughter nuclides per decay chain

// Physical constants
#define EARTH_RADIUS 6371000.0   // Earth radius in meters


struct PresData {
    float DZDT;     // No.1 [DZDT] Vertical velocity (m/s)
    float UGRD;     // No.2 [UGRD] U-component of wind (m/s)
    float VGRD;     // No.3 [VGRD] V-component of wind (m/s)
    float HGT;      // No.4 [HGT] Geopotential height (m)
    float TMP;      // No.5 [TMP] Temperature (K)
    float RH;       // No.7 [RH] Relative Humidity (%)
};

struct EtasData {
    float UGRD;     // No.1 [UGRD] U-component of wind (m/s)
    float VGRD;     // No.2 [VGRD] V-component of wind (m/s)
    float DZDT;     // No.6 [DZDT] Vertical velocity (m/s)
    float DEN;      // No.7 [DEN] Density of the air (kg/m)
};

struct UnisData {
    float HPBLA;    // No.12 [HPBLA] Boundary Layer Depth after B. LAYER (m)
    float T1P5;     // No.21 [TMP] Temperature at 1.5m above ground (K)
    float SHFLT;    // No.39 [SHFLT] Surface Sensible Heat Flux on Tiles (W/m^2)
    float HTBM;     // No.43 [HTBM] Turbulent mixing height after B. Layer (m)
    float HPBL;     // No.131 [HPBL] Planetary Boundary Layer Height (m)
    float SFCR;     // No.132 [SFCR] Surface Roughness (m)
};

/**
 * Source and Output Structures
 */

// Release point location
struct Source {
    float lat;      // Latitude (degrees)
    float lon;      // Longitude (degrees)
    float height;   // Release height above ground (m)
};

// Concentration result at a specific location
struct Concentration {
    int location;      // Location/receptor index
    int sourceterm;    // Source term index
    double value;      // Concentration value
};

/**
 * RectangleGrid Class
 *
 * Rectangular grid for concentration calculations and output.
 * Automatically sizes grid based on domain extent.
 */
class RectangleGrid {
private:

public:
    // Domain boundaries
    float minX, minY, maxX, maxY;
    float intervalX, intervalY, intervalZ;
    int rows, cols, zdim;

    // Grid point structure
    struct GridPoint {
        float x;       // X coordinate (m)
        float y;       // Y coordinate (m)
        float z;       // Height (m)
        float conc;    // Concentration
    };

    GridPoint* grid;

    RectangleGrid(float _minX, float _minY, float _maxX, float _maxY) {

        float width = _maxX - _minX;
        float height = _maxY - _minY;

        minX = _minX - width * 0.5;
        maxX = _maxX + width * 0.5;
        minY = _minY - height * 0.5;
        maxY = _maxY + height * 0.5;

        rows = std::sqrt(3000 * (height / width));
        cols = std::sqrt(3000 * (width / height));

        intervalX = (maxX - minX) / (cols - 1);
        intervalY = (maxY - minY) / (rows - 1);
        intervalZ = 10.0f;

        grid = new GridPoint[rows * cols];
        for(int i = 0; i < rows; ++i){
            for(int j = 0; j < cols; ++j){
                int index = i * cols + j;
                grid[index].x = minX + j * intervalX;
                grid[index].y = minY + i * intervalY;
                grid[index].z = 20.0;
            }
        }

    }

    ~RectangleGrid() {
        delete[] grid;
    }
};

/**
 * SimulationControl Structure
 *
 * Contains all top-level simulation parameters and file paths.
 * Read from RCAP input files (typically SC100-SC400 sections).
 */
struct SimulationControl {
    // Simulation identification
    char sim_title[MAX_STRING_LENGTH];
    char plant_name[MAX_STRING_LENGTH];
    float plant_power;                      // Thermal power (MWth)
    char plant_type[MAX_STRING_LENGTH];

    // Spatial discretization
    int numRad;                             // Number of radial rings
    int numTheta;                           // Number of angular sectors
    float* ir_distances;                    // Radial distances (m)

    // Input file names
    char weather_file[MAX_STRING_LENGTH];
    char nucl_lib_file[MAX_STRING_LENGTH];  // Nuclide library file
    char dcf_file[MAX_STRING_LENGTH];       // Dose conversion factors
    char fcm_file[MAX_STRING_LENGTH];       // Food chain model

    // Dose calculation flags
    bool early_dose;                        // Calculate early (acute) dose
    bool late_dose;                         // Calculate late (chronic) dose

    void print() const {

        std::cout << "--------------------------------------------------------" << std::endl << std::endl;
        std::cout << "Simulation\t: " << sim_title << std::endl << std::endl;
        std::cout << "Plant Name\t: " << plant_name << std::endl << std::endl;
        std::cout << "Plant Power\t: " << plant_power << " MWth" << std::endl << std::endl;
        std::cout << "Plant Type\t: " << plant_type << std::endl << std::endl;
        //std::cout << "Location\t: " << loc_longitude << " (Longitude), " << loc_latitude << " (Latitude)" << std::endl << std::endl;
        std::cout << "Num of Radial\t: " << numRad << std::endl << std::endl;
        std::cout << "Num of Theta\t: " << numTheta << std::endl << std::endl;
        std::cout << "Radial Distances (ir)\n";
        for (int i = 0; i < numRad; ++i) {
            std::cout << i + 1 << ")\t" << ir_distances[i] << " km" << std::endl;
        }

        std::cout << std::endl << std::endl;
        std::cout << "Weather File\t: " << weather_file << std::endl << std::endl;
        std::cout << "Nuclide Library\t: " << nucl_lib_file << std::endl << std::endl;
        std::cout << "DCF File\t: " << dcf_file << std::endl << std::endl;
        std::cout << "FCM File\t: " << fcm_file << std::endl << std::endl;
        std::cout << "Early Dose\t: " << (early_dose ? "Yes" : "No") << std::endl << std::endl;
        std::cout << "Late Dose\t: " << (late_dose ? "Yes" : "No") << std::endl << std::endl;
        std::cout << "--------------------------------------------------------" << std::endl << std::endl;
    }
};

/**
 * RadioNuclideTransport Structure
 *
 * Describes radionuclide release scenarios including:
 * - Source location and building dimensions
 * - Temporal release patterns (puffs)
 * - Nuclide-specific release fractions
 * Read from RT100-RT220 sections in RCAP input files.
 */
struct RadioNuclideTransport {
    int nPuffTotal;                 // Total number of puffs to release
    float build_height;             // Building height for wake effects (m)
    float build_width;              // Building width for wake effects (m)

    double lon, lat;                // Source location (degrees)
    float conc[MAX_NUCLIDES];       // Initial concentrations per nuclide

    /**
     * RT_Puff - Individual puff release definition
     */
    struct RT_Puff {
        int puffID;                   // Unique puff identifier
        float rele_time;              // Release start time (s)
        float duration;               // Release duration (s)
        float rele_height;            // Release height above ground (m)
        float rel_heat;               // Heat release rate for buoyancy (W)
        int sizeDistri_iType;         // Particle size distribution type
        float release_fractions[9];   // Release fractions for 9 nuclide groups

        RT_Puff() : puffID(0), rele_time(0.0f), duration(0.0f), rele_height(0.0f),
            rel_heat(0.0f), sizeDistri_iType(0) {
            std::fill(std::begin(release_fractions), std::end(release_fractions), 0.0f);
        }
    };

    std::vector<RT_Puff> RT_puffs;

    RadioNuclideTransport() : nPuffTotal(0), lon(0.0f), lat(0.0f) {
        std::fill(std::begin(conc), std::end(conc), 0.0f);
    }

    void allocatePuffs(int totalPuffs) {
        nPuffTotal = totalPuffs;
        RT_puffs.resize(totalPuffs);
    }
     
    void print(int ii, int input_num) const {
        std::cout << "<< RadioNuclide Transport Data >> " << ii + 1 << " of " << input_num << std::endl << std::endl;
        std::cout << "Number of Puffs: " << nPuffTotal << std::endl;
        std::cout << "Location (Longitude, Latitude): (" << std::setprecision(9) << lon << ", " << lat << ")\n" << std::endl;

        std::cout << "[Concentrations]\n";
        for (int i = 0; i < MAX_NUCLIDES; ++i) {
            std::cout << "No." << i+1 << "\t" << conc[i] << std::endl;
        }
        std::cout << std::endl;

        std::cout << ii + 1 << " of " << input_num << std::endl << std::endl;
        for (int i = 0; i < nPuffTotal; ++i) {
            const RT_Puff& puff = RT_puffs[i];
            std::cout << "\nPuff ID: " << puff.puffID << std::endl;
            std::cout << "  Release Time (s): " << puff.rele_time << std::endl;
            std::cout << "  Duration (s): " << puff.duration << std::endl;
            std::cout << "  Release Height (m): " << puff.rele_height << std::endl;
            std::cout << "  Release Heat (W): " << puff.rel_heat << std::endl;
            std::cout << "  Size Distribution iType: " << puff.sizeDistri_iType << std::endl << std::endl;

            std::cout << "[Release Fractions]" << std::endl << std::endl;
            for (int j = 0; j < 9; ++j) {
                std::cout << puff.release_fractions[j] <<std::endl;
            }
            std::cout << std::endl;
        }
    }

};

/**
 * WeatherSamplingData Structure
 *
 * Controls weather sampling for Monte Carlo simulations.
 * Read from RT310-RT350 sections in RCAP input files.
 */
struct WeatherSamplingData {
    bool isConstant;              // True for constant weather, false for time-varying
    int nSamplePerDay;            // Number of weather samples per day
    int randomSeed;               // Random seed for sampling
    float windSpeed;              // Wind speed (m/s)
    char stability;               // Pasquill-Gifford stability class (A-F)
    float rainRate;               // Rainfall rate (mm/hr)
    float mixHeight;              // Mixing layer height (m)

    WeatherSamplingData() : isConstant(false), nSamplePerDay(0), randomSeed(0),
        windSpeed(0.0f), stability('D'), rainRate(0.0f), mixHeight(0.0f) {}

    void print() const {
        std::cout << "Weather Sampling Data:" << std::endl;
        std::cout << "----------------------" << std::endl;
        std::cout << "Is Constant: " << std::boolalpha << isConstant << std::endl;
        std::cout << "Samples per day: " << nSamplePerDay << std::endl;
        std::cout << "Random Seed: " << randomSeed << std::endl;
        std::cout << "Wind Speed (m/s): " << std::fixed << std::setprecision(2) << windSpeed << std::endl;
        std::cout << "Stability: " << stability << std::endl;
        std::cout << "Rain Rate (mm/hr): " << std::fixed << std::setprecision(2) << rainRate << std::endl;
        std::cout << "Mix Height (m): " << std::fixed << std::setprecision(2) << mixHeight << std::endl;
    }
};

/**
 * EvacuationData Structure
 *
 * Describes evacuation parameters including:
 * - Shelter-in-place timing
 * - Evacuation speeds and durations
 * Read from EP200-EP240 sections in RCAP input files.
 */
struct EvacuationData {
    float alarmTime;                  // Time of evacuation alarm (s)
    int evaEndRing;                   // Last ring to evacuate
    int EP_endRing;                   // Evacuation planning zone end ring

    float shelterDelay[20];           // Delay before sheltering per ring (s)
    float shelterDuration[20];        // Duration of sheltering per ring (s)

    int nSpeedPeriod;                 // Number of speed/duration periods
    float speeds[5];                  // Evacuation speeds per period (m/s)
    float durations[5];               // Duration of each speed period (s)

    EvacuationData() : alarmTime(0.0f), evaEndRing(0), EP_endRing(0), nSpeedPeriod(0) {}

    void print() const {
        std::cout << "Alarm Time: " << alarmTime << " seconds\n";
        std::cout << "Eva End Ring: " << evaEndRing << "\n";
        std::cout << "EP End Ring: " << EP_endRing << "\n";

        // Commented out to prevent terminal spam
        // std::cout << "Shelter Delay: ";
        // for (float delay : shelterDelay) {
        //     std::cout << delay << " ";
        // }
        // std::cout << "\n";

        // std::cout << "Shelter Duration: ";
        // for (float duration : shelterDuration) {
        //     std::cout << duration << " ";
        // }
        // std::cout << "\n";

        // std::cout << "Speeds: ";
        // for (float speed : speeds) {
        //     std::cout << speed << " ";
        // }
        // std::cout << "\n";

        // std::cout << "Durations: ";
        // for (float duration : durations) {
        //     std::cout << duration << " ";
        // }
        // std::cout << "\n";
    }
};

/**
 * NuclideData Structure
 *
 * Contains complete radionuclide information including:
 * - Physical properties (half-life, atomic weight)
 * - Chemical group (for deposition modeling)
 * - Deposition parameters
 * - Decay chain information
 * - Dose conversion factors for multiple organs
 *
 * Read from MACCS60.NDL and MACCS_DCF_New2.LIB files.
 * Packed structure for efficient GPU transfer.
 *
 * Chemical Groups:
 *   1: Xenon (Xe)   2: Iodine (I)   3: Cesium (Cs)
 *   4: Tellurium (Te)   5: Strontium (Sr)   6: Ruthenium (Ru)
 *   7: Lanthanum (La)   8: Cerium (Ce)   9: Barium (Ba)
 */
#pragma pack(push, 1)
struct NuclideData {
    char name[MAX_STRING_LENGTH];                           // Nuclide name (e.g., "Xe-133")
    int id;                                                 // Unique identifier
    float half_life;                                        // Half-life (seconds)
    float atomic_weight;                                    // Atomic weight (amu)
    int chemical_group;                                     // Chemical group (1-9, see above)
    float wet_deposition;                                   // Wet deposition parameter
    float dry_deposition;                                   // Dry deposition velocity (m/s)
    float core_inventory;                                   // Core inventory (Ci/MWth)
    int decay_count;                                        // Number of daughter nuclides
    float exposure_data[MAX_ORGANS * DATA_FIELDS];          // Dose conversion factors (flattened)
    char organ_names[MAX_ORGANS * MAX_STRING_LENGTH];       // Organ names (flattened)
    int organ_count;                                        // Number of organs
    char daughter[MAX_DNUC * MAX_STRING_LENGTH];            // Daughter nuclide names (flattened)
    float branching_fraction[MAX_DNUC];                     // Decay branching fractions
};
#pragma pack(pop)


__device__ __host__ void initializeNuclideData(NuclideData* nuclide) {
    for (int i = 0; i < MAX_STRING_LENGTH; ++i) {
        nuclide->name[i] = 0;
    }
    nuclide->id = -1;
    nuclide->half_life = 0.0f;
    nuclide->atomic_weight = 0.0f; 
    nuclide->chemical_group = 0;

    nuclide->wet_deposition = 0.0f;
    nuclide->dry_deposition = 0.0f;

    nuclide->core_inventory = 0.0f;
    nuclide->decay_count = 0;

    nuclide->organ_count = 0;

    for (int i = 0; i < MAX_ORGANS; ++i) {
        for (int j = 0; j < MAX_STRING_LENGTH; ++j) {
            nuclide->organ_names[i * MAX_STRING_LENGTH + j] = 0;
        }
        for (int j = 0; j < DATA_FIELDS; ++j) {
            nuclide->exposure_data[i * DATA_FIELDS + j] = 0.0f;
        }
    }

    for (int i = 0; i < MAX_DNUC; ++i) {
        for (int j = 0; j < MAX_STRING_LENGTH; ++j) {
            nuclide->daughter[i * MAX_STRING_LENGTH + j] = 0;
        }
        nuclide->branching_fraction[i] = 0.0f;
    }
}

__device__ __host__ void setChemicalGroup(NuclideData* nuclide, const char* group) {
    if (group[0] == 'x' && group[1] == 'e' && group[2] == 'n') {
        nuclide->chemical_group = 1;
    }
    else if (group[0] == 'i' && group[1] == 'o' && group[2] == 'd') {
        nuclide->chemical_group = 2;
    }
    else if (group[0] == 'c' && group[1] == 'e' && group[2] == 's') {
        nuclide->chemical_group = 3;
    }
    else if (group[0] == 't' && group[1] == 'e' && group[2] == 'l') {
        nuclide->chemical_group = 4;
    }
    else if (group[0] == 's' && group[1] == 't' && group[2] == 'r') {
        nuclide->chemical_group = 5;
    }
    else if (group[0] == 'r' && group[1] == 'u' && group[2] == 't') {
        nuclide->chemical_group = 6;
    }
    else if (group[0] == 'l' && group[1] == 'a' && group[2] == 'n') {
        nuclide->chemical_group = 7;
    }
    else if (group[0] == 'c' && group[1] == 'e' && group[2] == 'r') {
        nuclide->chemical_group = 8;
    }
    else if (group[0] == 'b' && group[1] == 'a' && group[2] == 'r') {
        nuclide->chemical_group = 9;
    }
    else {
        nuclide->chemical_group = 0;
    }
}
enum DirectionCode {
    DIR_NONE = 0,  
    DIR_F = 1,     // Forward
    DIR_B = 2,     // Backward
    DIR_L = 3,     // Left
    DIR_R = 4      // Right
};

//struct EvacuationDirections {
//    int* directions;
//    int rows, cols;
//
//    EvacuationDirections() : directions(nullptr), rows(0), cols(0) {}
//
//    EvacuationDirections(int r, int c)
//        : rows(r), cols(c) {
//        directions = new int[rows * cols]();
//    }
//
//    ~EvacuationDirections() {
//        delete[] directions;
//    }
//
//    void resize(int r, int c) {
//        delete[] directions;
//        rows = r;
//        cols = c;
//        directions = new int[rows * cols]();
//    }
//
//    int convertDirection(char dir) {
//        switch (dir) {
//        case 'F': return DIR_F;
//        case 'f': return DIR_F;
//        case 'B': return DIR_B;
//        case 'b': return DIR_B;
//        case 'L': return DIR_L;
//        case 'l': return DIR_L;
//        case 'R': return DIR_R;
//        case 'r': return DIR_R;
//        default: return DIR_NONE;
//        }
//    }
//
//    int get(int row, int col) const {
//        return directions[row * cols + col];
//    }
//
//    void set(int row, int col, int value) {
//        directions[row * cols + col] = value;
//    }
//
//    void print() const {
//        for (int i = 0; i < rows; ++i) {
//            std::cout << "Row " << i + 1 << ": ";
//            for (int j = 0; j < cols; ++j) {
//                std::cout << get(i, j) << " ";
//            }
//            std::cout << std::endl;
//        }
//    }
//};

struct EvacuationDirections {
    int* directions;
    int rows, cols;

    EvacuationDirections() : directions(nullptr), rows(0), cols(0) {}

    EvacuationDirections(int r, int c)
        : rows(r), cols(c) {
        directions = new int[rows * cols]();
    }

    EvacuationDirections(const EvacuationDirections& other)
        : rows(other.rows), cols(other.cols) {
        directions = new int[rows * cols];
        std::copy(other.directions, other.directions + rows * cols, directions);
    }

    EvacuationDirections& operator=(const EvacuationDirections& other) {
        if (this != &other) {
            delete[] directions;
            rows = other.rows;
            cols = other.cols;
            directions = new int[rows * cols];
            std::copy(other.directions, other.directions + rows * cols, directions);
        }
        return *this;
    }

    ~EvacuationDirections() {
        delete[] directions;
    }

    void resize(int r, int c) {
        delete[] directions;
        rows = r;
        cols = c;
        directions = new int[rows * cols]();
    }

    int convertDirection(char dir) {
        switch (dir) {
        case 'F': return DIR_F;
        case 'f': return DIR_F;
        case 'B': return DIR_B;
        case 'b': return DIR_B;
        case 'L': return DIR_L;
        case 'l': return DIR_L;
        case 'R': return DIR_R;
        case 'r': return DIR_R;
        default: return DIR_NONE;
        }
    }

    int get(int row, int col) const {
        return directions[row * cols + col];
    }

    void set(int row, int col, int value) {
        directions[row * cols + col] = value;
    }

    void print() const {
        // Commented out to prevent terminal spam
        // for (int i = 0; i < rows; ++i) {
        //     std::cout << "Row " << i + 1 << ": ";
        //     for (int j = 0; j < cols; ++j) {
        //         std::cout << get(i, j) << " ";
        //     }
        //     std::cout << std::endl;
        // }
    }
};


struct SiteData {
    std::vector<float> roughness;                // SD50
    std::vector<std::vector<int>> population;    // SD150

    SiteData() = default;

    SiteData(int numRows, int numCols)
        : roughness(numCols, 0.0f),
        population(numRows, std::vector<int>(numCols, 0)) {}

    void resize(int numRows, int numCols) {
        roughness.resize(numCols, 0.0f);
        population.resize(numRows);
        for (auto& row : population) {
            row.resize(numCols, 0);
        }
    }

    void print() const {
        // Commented out to prevent terminal spam
        // std::cout << "Surface Roughness (cm):\n\n";
        // for (int i = 0; i < roughness.size(); ++i) {
        //     std::cout << "Dir" << i + 1 << "\t:" << roughness[i] << std::endl;
        // }
        // std::cout << std::endl;

        // std::cout << "\nPopulation Distribution:\n\n";
        // for (int i = 0; i < population.size(); ++i) {
        //     std::cout << "Row " << i + 1 << ": ";
        //     for (int j = 0; j < population[i].size(); ++j) {
        //         std::cout << population[i][j] << "\t";
        //     }
        //     std::cout << std::endl;
        // }
    }
};

struct ProtectionFactors {

    float pfactor[3][5];
    float resus_coef;
    float resus_half_life;

};

struct RCAP_METDATA {
    int day;
    int hour;
    float dir;
    float spd;
    int stab;
    float rain;
};

struct HealthEffect {
    // MP250 - Acute Fatality
    std::string FatalityName[2];
    std::string TargetOrgan_AF[2];
    float       alpha_f[2];
    float       beta_f[2];
    float       threshold_AF[2];

    // MP260 - Acute Morbidity / Injury
    std::string InjuryName[7];
    std::string TargetOrgan_AM[7];
    float       alpha_i[7];
    float       beta_i[7];
    float       threshold_AM[7];

    // MP270 - Cancer Effect
    std::string CancerName[7];
    std::string TargetOrgan[7];
    float       dos_a[7];
    float       dos_b[7];
    float       cf_risk[7];
    float       ci_risk[7];
    float       ddrf[7];
    float       dos_thres[7];
    float       dosRate_thres[7];
    float       LNT_threshold[7];
    float       sus_frac[7];

    void print() const {
        std::cout << "=== HealthEffect Print ===\n\n";

        // MP250
        std::cout << "[MP250: Acute Fatality Model]\n";
        for (int i = 0; i < 2; ++i) {
            std::cout << "  ID: " << i + 1
                << ", FatalityName: " << FatalityName[i]
                << ", TargetOrgan: " << TargetOrgan_AF[i]
                << ", alpha_f: " << alpha_f[i]
                << ", beta_f: " << beta_f[i]
                << ", threshold: " << threshold_AF[i]
                << "\n";
        }
        std::cout << std::endl;

        // MP260
        std::cout << "[MP260: Acute Injury Model]\n";
        for (int i = 0; i < 7; ++i) {
            std::cout << "  ID: " << i + 1
                << ", InjuryName: " << InjuryName[i]
                << ", TargetOrgan: " << TargetOrgan_AM[i]
                << ", alpha_i: " << alpha_i[i]
                << ", beta_i: " << beta_i[i]
                << ", threshold: " << threshold_AM[i]
                << "\n";
        }
        std::cout << std::endl;

        // MP270
        std::cout << "[MP270: Cancer Effect Model]\n";
        for (int i = 0; i < 7; ++i) {
            std::cout << "  ID: " << i + 1
                << ", CancerName: " << CancerName[i]
                << ", TargetOrgan: " << TargetOrgan[i]
                << ", dos_a: " << dos_a[i]
                << ", dos_b: " << dos_b[i]
                << ", cf_risk: " << cf_risk[i]
                << ", ci_risk: " << ci_risk[i]
                << ", ddrf: " << ddrf[i]
                << ", dos_thres: " << dos_thres[i]
                << ", dosRate_thres: " << dosRate_thres[i]
                << ", LNT_threshold: " << LNT_threshold[i]
                << ", sus_frac: " << sus_frac[i]
                << "\n";
        }
        std::cout << std::endl;
    }
};

struct Evacuee {
    float population;
    float r;
    float theta;
    float speed;
    float dose;
    int flag;
    int rad0;

    float x, y;

    float dose_inhalation = 0.0f;
    float dose_cloudshine = 0.0f;

    float dose_inhalations[MAX_ORGANS] = { 0.0f, };
    float dose_cloudshines[MAX_ORGANS] = { 0.0f, };

    // Cloudshine tracking fields
    float dose_cloudshine_cumulative = 0.0f;  // Cumulative cloudshine dose
    float dose_cloudshine_instant = 0.0f;     // Instantaneous cloudshine dose (current timestep)
    int cloudshine_mode = -1;                 // 0: small_puff, 1: plane_source, 2: semi_infinite, -1: none

    int prev_rad_idx;
    int prev_theta_idx;
    int met_idx;

    Evacuee() : population(0.0f), r(0.0f), theta(0.0f), speed(0.0f), dose(0.0f),
                dose_inhalation(0.0f), dose_cloudshine(0.0f),
                dose_cloudshine_cumulative(0.0f), dose_cloudshine_instant(0.0f),
                cloudshine_mode(-1), flag(true) {}

    void print() const {
        std::cout << "Population: " << population
            << ", Radius (r): " << r
            << ", Angle (theta): " << theta
            << ", Speed: " << speed << std::endl;
    }
};

float getExecutionTime(cudaEvent_t start, cudaEvent_t stop) {
    float milliseconds = 0;
    cudaEventElapsedTime(&milliseconds, start, stop);
    return milliseconds;
}

struct ResultLine {
    int metVal;
    int radVal;
    int thetaVal;
    std::string effect_type;   // "deterministic" or "stochastic_cf" or "stochastic_ci"
    std::string effect_name;   // "HematopoieticSyndrome", "Leukemia_incidence" ��
    float npeople;
};
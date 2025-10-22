#pragma once

#define LDAPS_E 132.36
#define LDAPS_W 121.06
#define LDAPS_N 43.13
#define LDAPS_S 32.20
#define PI 3.141592
#define invPI 0.31831

#define dimX 602
#define dimY 781
#define dimZ_pres 24
#define dimZ_etas 71

#define INPUT_NUM 1
#define MAX_STRING_LENGTH 32
#define MAX_NUCLIDES 80
#define MAX_ORGANS 20
#define DATA_FIELDS 5
#define MAX_DNUC 2

#define EARTH_RADIUS 6371000.0


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

struct Source {
    float lat;
    float lon;
    float height;
};

struct Concentration {
    int location;
    int sourceterm;
    double value;
};

class RectangleGrid {
private:

public:

    float minX, minY, maxX, maxY;
    float intervalX, intervalY, intervalZ;
    int rows, cols, zdim;

    struct GridPoint{
        float x;
        float y;
        float z;
        float conc;
    };

    GridPoint* grid;

    RectangleGrid(float _minX, float _minY, float _maxX, float _maxY){

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

    ~RectangleGrid(){
        delete[] grid;
    }
};


struct SimulationControl {

    char sim_title[MAX_STRING_LENGTH];
    char plant_name[MAX_STRING_LENGTH];
    float plant_power;
    char plant_type[MAX_STRING_LENGTH];
    //float loc_longitude;
    //float loc_latitude;
    int numRad;
    int numTheta;
    float* ir_distances;
    char weather_file[MAX_STRING_LENGTH];
    char nucl_lib_file[MAX_STRING_LENGTH];
    char dcf_file[MAX_STRING_LENGTH];
    char fcm_file[MAX_STRING_LENGTH];
    bool early_dose;
    bool late_dose;

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


struct RadioNuclideTransport {

    int nPuffTotal;                 // RT200
    float build_height;             // RT215
    float build_width;              // RT215

    double lon, lat;
    float conc[MAX_NUCLIDES];

    struct RT_Puff {
        int puffID;                 // RT210
        float rele_time;            // RT210
        float duration;             // RT210
        float rele_height;          // RT210
        float rel_heat;             // RT210
        int sizeDistri_iType;       // RT210
        float release_fractions[9]; // RT220

        RT_Puff() : puffID(0), rele_time(0.0f), duration(0.0f), rele_height(0.0f),
            rel_heat(0.0f), sizeDistri_iType(0) {
            std::fill(std::begin(release_fractions), std::end(release_fractions), 0.0f);
        }
    };

    std::vector<RT_Puff> RT_puffs;

    RadioNuclideTransport() : nPuffTotal(0), lon(0.0f), lat(0.0f) {
        std::fill(std::begin(conc), std::end(conc), 0.0f);
    }
    //RadioNuclideTransport() : nPuffTotal(0) {}

    void allocatePuffs(int totalPuffs) {
        nPuffTotal = totalPuffs;
        RT_puffs.resize(totalPuffs);
    }
     
    void print(int ii, int input_num) const {

        std::cout << "<< RadioNuclide Transport Data >> " << ii + 1 << " of " << input_num << std::endl << std::endl;

        std::cout << "Number of Puffs: " << nPuffTotal << std::endl;
        //std::cout << "Building Height (m): " << build_height << std::endl;
        //std::cout << "Building Width (m): " << build_width << std::endl;
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


struct WeatherSamplingData {

    bool isConstant;              // RT310 (true for const, false for stratified)
    int nSamplePerDay;            // RT320
    int randomSeed;               // RT340
    float windSpeed;              // RT350
    char stability;               // RT350
    float rainRate;               // RT350
    float mixHeight;              // RT350

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

struct EvacuationData {
    float alarmTime;                            // EP200
    int evaEndRing;                             // EP210
    int EP_endRing;                             // EP210
    //std::vector<float> shelterDelay;          // EP220
    //std::vector<float> shelterDuration;       // EP230

    float shelterDelay[20];
    float shelterDuration[20];

    int nSpeedPeriod;                           // EP240
    //std::vector<float> speeds;                // EP240
    //std::vector<float> durations;             // EP240

    float speeds[5];
    float durations[5];

    EvacuationData() : alarmTime(0.0f), evaEndRing(0), EP_endRing(0), nSpeedPeriod(0) {}

    void print() const {
        std::cout << "Alarm Time: " << alarmTime << " seconds\n";
        std::cout << "Eva End Ring: " << evaEndRing << "\n";
        std::cout << "EP End Ring: " << EP_endRing << "\n";

        std::cout << "Shelter Delay: ";
        for (float delay : shelterDelay) {
            std::cout << delay << " ";
        }
        std::cout << "\n";

        std::cout << "Shelter Duration: ";
        for (float duration : shelterDuration) {
            std::cout << duration << " ";
        }
        std::cout << "\n";

        std::cout << "Speeds: ";
        for (float speed : speeds) {
            std::cout << speed << " ";
        }
        std::cout << "\n";

        std::cout << "Durations: ";
        for (float duration : durations) {
            std::cout << duration << " ";
        }
        std::cout << "\n";
    }
};


//struct DecayData {
//    char daughter[MAX_STRING_LENGTH];
//    double branching_fraction;
//
//    void initialize() {
//        memset(daughter, 0, sizeof(daughter));
//        branching_fraction = 0.0;
//    }
//};

//__device__ __host__ struct NuclideData {
//    char name[MAX_STRING_LENGTH];
//    int id;
//    double half_life;
//    double atomic_weight;
//    //char chemical_group[MAX_STRING_LENGTH];
//    int chemical_group;
//
//    bool wet_deposition;
//    bool dry_deposition;
//
//    double core_inventory;  // Ci/MWth
//    //double core_inventory_bq;  // Bq
//
//    DecayData decay[2];
//    int decay_count;
//
//    char organ_names[MAX_ORGANS][MAX_STRING_LENGTH];
//    double exposure_data[MAX_ORGANS][DATA_FIELDS];
//    int organ_count;
//
//    void initialize() {
//        memset(name, 0, sizeof(name));
//        id = -1;
//        half_life = 0.0;
//        atomic_weight = 0.0;
//        chemical_group = 0;
//
//        wet_deposition = false;
//        dry_deposition = false;
//
//        core_inventory = 0.0;
//        decay_count = 0;
//
//        for (int i = 0; i < 2; ++i) {
//            decay[i].initialize();
//        }
//
//        organ_count = 0;
//        for (int i = 0; i < MAX_ORGANS; ++i) {
//            memset(organ_names[i], 0, sizeof(organ_names[i]));
//            for (int j = 0; j < DATA_FIELDS; ++j) {
//                exposure_data[i][j] = 0.0;
//            }
//        }
//    }
//
//    void setChemicalGroup(const char* group) {
//        if (strncmp(group, "xen", 3) == 0) {
//            chemical_group = 1;
//        }
//        else if (strncmp(group, "iod", 3) == 0) {
//            chemical_group = 2;
//        }
//        else if (strncmp(group, "ces", 3) == 0) {
//            chemical_group = 3;
//        }
//        else if (strncmp(group, "tel", 3) == 0) {
//            chemical_group = 4;
//        }
//        else if (strncmp(group, "str", 3) == 0) {
//            chemical_group = 5;
//        }
//        else if (strncmp(group, "rut", 3) == 0) {
//            chemical_group = 6;
//        }
//        else if (strncmp(group, "lan", 3) == 0) {
//            chemical_group = 7;
//        }
//        else if (strncmp(group, "cer", 3) == 0) {
//            chemical_group = 8;
//        }
//        else if (strncmp(group, "bar", 3) == 0) {
//            chemical_group = 9;
//        }
//        else {
//            chemical_group = 0;
//        }
//    }
//
//};

//#pragma pack(push, 1)
//
//__device__ __host__ struct NuclideData {
//    float exposure_data[MAX_ORGANS][DATA_FIELDS];
//    char name[MAX_STRING_LENGTH];
//    int id;
//    float half_life;
//    float atomic_weight;
//    int chemical_group;
//
//    float wet_deposition;
//    float dry_deposition;
//
//    float core_inventory;  // Ci/MWth
//
//    //DecayData decay[2];
//    int decay_count;
//
//    //float exposure_data[MAX_ORGANS][DATA_FIELDS];
//    char organ_names[MAX_ORGANS][MAX_STRING_LENGTH];
//    int organ_count;
//
//    char daughter[MAX_DNUC][MAX_STRING_LENGTH];
//    float branching_fraction[MAX_DNUC];
//
//    //__device__ __host__ void initialize() {
//    //    for (int i = 0; i < MAX_STRING_LENGTH; ++i) {
//    //        name[i] = 0;
//    //    }
//    //    id = -1;
//    //    half_life = 0.0;
//    //    atomic_weight = 0.0;
//    //    chemical_group = 0;
//
//    //    wet_deposition = 0;
//    //    dry_deposition = 0;
//
//    //    core_inventory = 0.0;
//    //    decay_count = 0;
//
//    //    //for (int i = 0; i < 2; ++i) {
//    //    //    decay[i].initialize();
//    //    //}
//
//    //    organ_count = 0;
//    //    for (int i = 0; i < MAX_ORGANS; ++i) {
//    //        for (int j = 0; j < MAX_STRING_LENGTH; ++j) {
//    //            organ_names[i][j] = 0;
//    //        }
//    //        for (int j = 0; j < DATA_FIELDS; ++j) {
//    //            exposure_data[i][j] = 0.0;
//    //        }
//    //    }
//    //    for (int i = 0; i < MAX_DNUC; ++i) {
//    //        for (int j = 0; j < MAX_STRING_LENGTH; ++j) {
//    //            daughter[i][j] = 0;
//    //        }
//    //        branching_fraction[i] = 0.0;
//    //    }
//    //}
//
//    //__device__ __host__ void setChemicalGroup(const char* group) {
//    //    if (group[0] == 'x' && group[1] == 'e' && group[2] == 'n') {
//    //        chemical_group = 1;
//    //    }
//    //    else if (group[0] == 'i' && group[1] == 'o' && group[2] == 'd') {
//    //        chemical_group = 2;
//    //    }
//    //    else if (group[0] == 'c' && group[1] == 'e' && group[2] == 's') {
//    //        chemical_group = 3;
//    //    }
//    //    else if (group[0] == 't' && group[1] == 'e' && group[2] == 'l') {
//    //        chemical_group = 4;
//    //    }
//    //    else if (group[0] == 's' && group[1] == 't' && group[2] == 'r') {
//    //        chemical_group = 5;
//    //    }
//    //    else if (group[0] == 'r' && group[1] == 'u' && group[2] == 't') {
//    //        chemical_group = 6;
//    //    }
//    //    else if (group[0] == 'l' && group[1] == 'a' && group[2] == 'n') {
//    //        chemical_group = 7;
//    //    }
//    //    else if (group[0] == 'c' && group[1] == 'e' && group[2] == 'r') {
//    //        chemical_group = 8;
//    //    }
//    //    else if (group[0] == 'b' && group[1] == 'a' && group[2] == 'r') {
//    //        chemical_group = 9;
//    //    }
//    //    else {
//    //        chemical_group = 0;
//    //    }
//    //}
//};
//#pragma pack(pop)

#pragma pack(push, 1)
struct NuclideData {
    char name[MAX_STRING_LENGTH];
    int id;
    float half_life;
    float atomic_weight;
    int chemical_group;
    float wet_deposition;
    float dry_deposition;
    float core_inventory;
    int decay_count;
    float exposure_data[MAX_ORGANS * DATA_FIELDS];
    char organ_names[MAX_ORGANS * MAX_STRING_LENGTH];
    int organ_count;
    char daughter[MAX_DNUC * MAX_STRING_LENGTH];
    float branching_fraction[MAX_DNUC];
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
        for (int i = 0; i < rows; ++i) {
            std::cout << "Row " << i + 1 << ": ";
            for (int j = 0; j < cols; ++j) {
                std::cout << get(i, j) << " ";
            }
            std::cout << std::endl;
        }
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
        std::cout << "Surface Roughness (cm):\n\n";
        for (int i = 0; i < roughness.size(); ++i) {
            std::cout << "Dir" << i + 1 << "\t:" << roughness[i] << std::endl;
        }
        std::cout << std::endl;

        std::cout << "\nPopulation Distribution:\n\n";
        for (int i = 0; i < population.size(); ++i) {
            std::cout << "Row " << i + 1 << ": ";
            for (int j = 0; j < population[i].size(); ++j) {
                std::cout << population[i][j] << "\t";
            }
            std::cout << std::endl;
        }
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

    int prev_rad_idx;
    int prev_theta_idx;
    int met_idx;

    Evacuee() : population(0.0f), r(0.0f), theta(0.0f), speed(0.0f), dose(0.0f), dose_inhalation(0.0f), dose_cloudshine(0.0f), flag(true) {}

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
    std::string effect_name;   // "HematopoieticSyndrome", "Leukemia_incidence" µî
    float npeople;
};
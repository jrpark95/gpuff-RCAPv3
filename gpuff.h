/**
 * GPUFF Header File - Minimal definitions for build
 */

#pragma once

#include <vector>
#include <string>

// Forward declarations
struct NuclideData;
struct SimulationControl;
struct RadioNuclideTransport;
struct WeatherSamplingData;
struct EvacuationData;
struct EvacuationDirections;
struct SiteData;
struct ProtectionFactors;
struct HealthEffect;
struct Evacuee;
struct PuffRCAP;

// Constants
#define MAX_NUCLIDES 80
#define MAX_ORGANS 20
#define DATA_FIELDS 5

// Dummy Gpuff class to satisfy compilation
class Gpuff {
public:
    Gpuff() {}

    // Dummy methods to satisfy compilation
    static float Lambert2x(float lat, float lon) { return 0.0f; }
    static float Lambert2y(float lat, float lon) { return 0.0f; }

    static void read_meteorological_data(
        void* unis, void* pres, void* etas,
        float* unis_raw, float* pres_raw, float* etas_raw,
        int dimX, int dimY, int dimZ_pres, int dimZ_etas,
        int total, int metIndex) {}

    static void read_meteorological_data_RCAP() {}
    static void read_meteorological_data_RCAP2(const std::string& filename) {}

    static void initializePuffs_RCAP(
        std::vector<PuffRCAP>& puffs,
        const std::vector<RadioNuclideTransport>& RT,
        float* radius, int numRad, int numTheta,
        int totalPuffs, int startPuffId,
        int npuc, int freqMet) {}

    static void initializeEvacuees(
        std::vector<Evacuee>& evacuees,
        const SimulationControl& SC,
        const EvacuationData& EP,
        float* radius, int numRad, int numTheta,
        int totalEvacuees, int startEvacueeId) {}

    static void initializeEvacuees_xy(
        std::vector<Evacuee>& evacuees,
        const SimulationControl& SC,
        const EvacuationData& EP,
        float* const* const xy,
        int totalEvacuees, int startEvacueeId) {}

    void update_puff_flags2_cpu(float currentTime, int nop) {}

    void move_puffs_by_wind_RCAP2_cpu(
        int EP_endRing, std::vector<NuclideData> ND, float* radius,
        int numRad, int numTheta, int nop) {}

    void ComputeExposureHmix_cpu(
        std::vector<Evacuee> evacuees,
        ProtectionFactors PF,
        int numSims,
        int totalEvacueesPerSim,
        int totalPuffsPerSim) {}
};

// Global variables declarations
extern std::vector<int> RCAP_metdata;
extern std::vector<PuffRCAP> puffs_RCAP;
extern std::vector<float> RCAP_windir;
extern std::vector<float> RCAP_winvel;
extern std::vector<int> RCAP_stab;

extern float* d_RCAP_windir;
extern float* d_RCAP_winvel;
extern int* d_RCAP_stab;

extern float* device_meteorological_data_pres;
extern float* device_meteorological_data_unis;
extern float* device_meteorological_data_etas;

// Function declarations
void initializeNuclideData(NuclideData* nuclide);
void read_MACCS_DCF_New2(const std::string& filename, std::vector<NuclideData>& ND);
void read_MACCS60_NDL(const std::string& filename, std::vector<NuclideData>& ND);
void print_MACCS60_NDL(const std::vector<NuclideData>& ND);
int check_input_num(const std::string& filename);

// Global arrays
extern float exposure_data_all[MAX_NUCLIDES * MAX_ORGANS * DATA_FIELDS];
extern float* d_exposure;
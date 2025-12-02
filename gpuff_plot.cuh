#include "gpuff.cuh"

// Count Active Puffs
//
// Counts the number of active (flagged) puff particles in the standard puff array.
// Used to determine how many puffs need to be written to output files.
//
// Returns:
//   Number of puffs with flag == 1
//
int Gpuff::countflag() {
    int count = 0;
    for (int i = 0; i < nop; ++i) {
        if (puffs[i].flag == 1) count++;
    }
    return count;
}

// Count Active RCAP Puffs
//
// Counts the number of active (flagged) puff particles in the RCAP puff array.
// RCAP version uses boolean flag instead of integer.
//
// Returns:
//   Number of puffs with flag == true
//
int Gpuff::countflag_RCAP() {
    int count = 0;
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (puffs_RCAP[i].flag == true) count++;
    }
    return count;
}

void Gpuff::puff_output_ASCII(int timestep){

    cudaMemcpy(puffs.data(), d_puffs, nop * sizeof(Puffcenter), cudaMemcpyDeviceToHost);

    int part_num = countflag();

    std::ostringstream filenameStream;

    std::string path;    

    #ifdef _WIN32

        path = ".\\output";
        _mkdir(path.c_str());
        filenameStream << ".\\output\\puff_" << std::setfill('0') 
        << std::setw(5) << timestep << "stp.vtk";

    #else

        path = "./output";
        mkdir(path.c_str(), 0777);
        filenameStream << "./output/puff_" << std::setfill('0') 
        << std::setw(5) << timestep << "stp.vtk";

    #endif


    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename);

    if (!vtkFile.is_open()){
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0" << std::endl;
    vtkFile << "puff data" << std::endl;
    vtkFile << "ASCII" << std::endl;
    vtkFile << "DATASET POLYDATA" << std::endl;

    vtkFile << "POINTS " << part_num << " float" << std::endl;
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        vtkFile << puffs[i].x << " " << puffs[i].y << " " << puffs[i].z << std::endl;
    }

    // vtkFile << "POINTS " << nop << " float" << std::endl;
    // for (int i = 0; i < nop; ++i){
    //     //if(!puffs[i].flag) continue;
    //     vtkFile << puffs[i].x << " " << puffs[i].y << " " << i << " " << puffs[i].flag << std::endl;
    // }

    vtkFile.close();
}


// Byte Swap for Float (Big-Endian Conversion)
//
// Swaps byte order of float value from little-endian to big-endian format.
// Required for VTK binary output format which uses big-endian encoding.
//
// VTK Binary Format Requirement:
//   VTK binary files must use big-endian byte order regardless of system architecture.
//   This ensures portability across different platforms.
//
// Parameters:
//   value: Float value to be byte-swapped (modified in place)
//
void Gpuff::swapBytes(float& value) {
    char* valuePtr = reinterpret_cast<char*>(&value);
    std::swap(valuePtr[0], valuePtr[3]);
    std::swap(valuePtr[1], valuePtr[2]);
}

// Byte Swap for Integer (Big-Endian Conversion)
//
// Swaps byte order of integer value from little-endian to big-endian format.
// Required for VTK binary output format which uses big-endian encoding.
//
// Parameters:
//   value: Integer value to be byte-swapped (modified in place)
//
void Gpuff::swapBytes_int(int& value) {
    char* valuePtr = reinterpret_cast<char*>(&value);
    std::swap(valuePtr[0], valuePtr[3]);
    std::swap(valuePtr[1], valuePtr[2]);
}

void Gpuff::puff_output_binary(int timestep){

    cudaMemcpy(puffs.data(), d_puffs, nop * sizeof(Puffcenter), cudaMemcpyDeviceToHost);

    int part_num = countflag();

    std::ostringstream filenameStream;

    std::string path;

    #ifdef _WIN32
        path = ".\\output";
        _mkdir(path.c_str());
        filenameStream << ".\\output\\puff_" << std::setfill('0') 
        << std::setw(5) << timestep << ".vtk";
    #else
        path = "./output";
        mkdir(path.c_str(), 0777);
        filenameStream << "./output/puff_" << std::setfill('0') 
        << std::setw(5) << timestep << ".vtk";
    #endif


    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()){
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "puff data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    vtkFile << "POINTS " << part_num << " float\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float x = puffs[i].x;
        float y = puffs[i].y;
        float z = puffs[i].z;

        swapBytes(x);
        swapBytes(y);
        swapBytes(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }

    vtkFile << "POINT_DATA " << part_num << "\n";
    vtkFile << "SCALARS sigma_h float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float vval = puffs[i].sigma_h;
        swapBytes(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS sigma_z float 1\n"; 
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float vval = puffs[i].sigma_z;
        swapBytes(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS virtual_dist float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float vval = puffs[i].virtual_distance;
        swapBytes(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS Q float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float vval = puffs[i].conc;
        swapBytes(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS windvel float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float vval = puffs[i].windvel;
        swapBytes(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS windir float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        float vval = puffs[i].windir;
        swapBytes(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }
    
    vtkFile << "SCALARS stab int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        int vval = puffs[i].stab;
        swapBytes_int(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    vtkFile << "SCALARS tidx int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < part_num; ++i){
        if(!puffs[i].flag) continue;
        int vval = puffs[i].timeidx;
        swapBytes_int(vval); 
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }


    vtkFile.close();
}

// Write RCAP Puff Data to VTK Binary File
//
// Exports puff center positions and properties to VTK format for visualization in ParaView.
// Transfers data from GPU to host memory before writing.
//
// VTK File Format:
//   Format: Legacy VTK (Version 4.0)
//   Type: POLYDATA (point cloud dataset)
//   Encoding: BINARY (big-endian)
//
// Output File Structure:
//   - Header: VTK version and description
//   - Points: 3D coordinates (x, y, z) of active puffs
//   - Point Data: Scalar fields attached to each point
//
// Point Data Fields:
//   - sigma_h: Horizontal dispersion parameter (m)
//   - sigma_z: Vertical dispersion parameter (m)
//   - virtual_dist: Virtual source distance (m)
//   - Q: Puff concentration/mass (Bq or kg)
//   - stab: Atmospheric stability class (1-7)
//   - unitidx: Source unit index
//   - rain: Precipitation rate (mm)
//   - simulnum: Simulation number (meteorological condition index)
//
// File Naming Convention:
//   Format: puff_RCAP_XXXXX.vtk
//   XXXXX: 5-digit zero-padded timestep number
//
// ParaView Compatibility:
//   Files can be loaded as time series by using "..." wildcard pattern
//   Example: puff_RCAP_....vtk loads entire sequence
//
// Parameters:
//   timestep: Current simulation timestep (used for file naming)
//
void Gpuff::puff_output_binary_RCAP(int timestep) {

    // Transfer puff data from GPU to host memory
    cudaMemcpy(puffs_RCAP.data(), d_puffs_RCAP, puffs_RCAP.size() * sizeof(Puffcenter_RCAP), cudaMemcpyDeviceToHost);

    int part_num = countflag_RCAP();

    std::ostringstream filenameStream;
    std::string path;

#ifdef _WIN32
    path = ".\\output";
    _mkdir(path.c_str());
    filenameStream << ".\\output\\puff_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#else
    path = "./output";
    mkdir(path.c_str(), 0777);
    filenameStream << "./output/puff_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#endif

    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    // Write VTK header (ASCII format)
    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "puff data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    // Write point coordinates (binary big-endian)
    vtkFile << "POINTS " << part_num << " float\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float x = puffs_RCAP[i].x;
        float y = puffs_RCAP[i].y;
        float z = puffs_RCAP[i].z;

        swapBytes(x);
        swapBytes(y);
        swapBytes(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }

    // Write point data attributes
    vtkFile << "POINT_DATA " << part_num << "\n";

    // Horizontal dispersion parameter
    vtkFile << "SCALARS sigma_h float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].sigma_h;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    // Vertical dispersion parameter
    vtkFile << "SCALARS sigma_z float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].sigma_z;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    // Virtual source distance
    vtkFile << "SCALARS virtual_dist float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].virtual_distance;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    // Puff concentration/mass (first nuclide)
    vtkFile << "SCALARS Q float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].conc[0];
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    // Atmospheric stability class
    vtkFile << "SCALARS stab int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        int vval = puffs_RCAP[i].stab;
        swapBytes_int(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    // Source unit index
    vtkFile << "SCALARS unitidx int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        int vval = puffs_RCAP[i].unitidx;
        swapBytes_int(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    // Precipitation rate
    vtkFile << "SCALARS rain float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].rain;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    // Simulation number (meteorological condition index)
    vtkFile << "SCALARS simulnum int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        int vval = puffs_RCAP[i].simulnum;
        swapBytes_int(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    vtkFile.close();
}

// Write RCAP Puff Data to VTK Binary File (CPU Version)
//
// CPU-only version of puff output (no GPU transfer).
// Used when puff data is already in host memory.
//
// Note: This function assumes puffs_RCAP is already populated in host memory.
// The GPU-to-host transfer (cudaMemcpy) is commented out.
//
// See puff_output_binary_RCAP() documentation for VTK format details.
//
void Gpuff::puff_output_binary_RCAP_cpu(int timestep) {

    // Note: GPU transfer skipped for CPU-only execution
    // cudaMemcpy(puffs_RCAP.data(), d_puffs_RCAP, puffs_RCAP.size() * sizeof(Puffcenter_RCAP), cudaMemcpyDeviceToHost);

    int part_num = countflag_RCAP();

    std::ostringstream filenameStream;
    std::string path;

#ifdef _WIN32
    path = ".\\output";
    _mkdir(path.c_str());
    filenameStream << ".\\output\\puff_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#else
    path = "./output";
    mkdir(path.c_str(), 0777);
    filenameStream << "./output/puff_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#endif

    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "puff data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    vtkFile << "POINTS " << part_num << " float\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float x = puffs_RCAP[i].x;
        float y = puffs_RCAP[i].y;
        float z = puffs_RCAP[i].z;

        swapBytes(x);
        swapBytes(y);
        swapBytes(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }

    vtkFile << "POINT_DATA " << part_num << "\n";
    vtkFile << "SCALARS sigma_h float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].sigma_h;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS sigma_z float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].sigma_z;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS virtual_dist float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].virtual_distance;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS Q float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].conc[0];
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS windvel float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].windvel;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS windir float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].windir;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS stab int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        int vval = puffs_RCAP[i].stab;
        swapBytes_int(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    vtkFile << "SCALARS unitidx int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        int vval = puffs_RCAP[i].unitidx;
        swapBytes_int(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    vtkFile << "SCALARS rain float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        float vval = puffs_RCAP[i].rain;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS simulnum int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (int i = 0; i < puffs_RCAP.size(); ++i) {
        if (!puffs_RCAP[i].flag) continue;
        int vval = puffs_RCAP[i].simulnum;
        swapBytes_int(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    vtkFile.close();
}

// Write Evacuee Data to VTK Binary File
//
// Exports evacuee positions and dose data to VTK format for visualization.
// Tracks population movement and radiation exposure during evacuation scenarios.
//
// VTK File Format:
//   Format: Legacy VTK (Version 4.0)
//   Type: POLYDATA (point cloud dataset)
//   Encoding: BINARY (big-endian)
//
// Coordinate System:
//   Evacuees stored in polar coordinates (r, theta) around source
//   Converted to Cartesian (x, y) for VTK output:
//     x = r * cos(theta)
//     y = r * sin(theta)
//     z = 0 (ground level)
//
// Point Data Fields:
//   - population: Number of people in this evacuee group
//   - speed: Current evacuation speed (m/s)
//   - ridx: Radial index in grid
//   - dose_inhalation_0/1/2: Inhalation dose for 3 nuclides (Sv)
//   - dose_cloudshine_0/1/2: Cloudshine dose for 3 nuclides (Sv)
//
// File Naming Convention:
//   Format: evac_RCAP_XXXXX.vtk
//   XXXXX: 5-digit zero-padded timestep number
//   Directory: ./evac/ or .\evac\
//
// Use Case:
//   Visualize evacuation progress and dose accumulation
//   Identify high-dose evacuation routes
//   Analyze population exposure patterns
//
// Parameters:
//   timestep: Current simulation timestep (used for file naming)
//
void Gpuff::evac_output_binary_RCAP(int timestep) {

    std::ostringstream filenameStream;
    std::string path;

    // Transfer evacuee data from GPU to host memory
    cudaMemcpy(evacuees.data(), d_evacuees, evacuees.size() * sizeof(Evacuee), cudaMemcpyDeviceToHost);

#ifdef _WIN32
    path = ".\\evac";
    _mkdir(path.c_str());
    filenameStream << ".\\evac\\evac_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#else
    path = "./evac";
    mkdir(path.c_str(), 0777);
    filenameStream << "./evac/evac_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#endif

    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    // Write VTK header (ASCII format)
    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "evacuation data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    // Write evacuee positions (convert polar to Cartesian)
    vtkFile << "POINTS " << evacuees.size() << " float\n";
    for (const auto& evacuee : evacuees) {
        float r = evacuee.r;
        float theta = evacuee.theta;
        float x = r * cos(theta);
        float y = r * sin(theta);
        float z = 0.0f;

        swapBytes(x);
        swapBytes(y);
        swapBytes(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }

    // Write point data attributes
    vtkFile << "POINT_DATA " << evacuees.size() << "\n";

    // Population in this evacuee group
    vtkFile << "SCALARS population float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.population;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));

        //if (evacuee.prev_theta_idx == 8) printf("%f  ", evacuee.speed);
    }
    //printf("\n");

    // Evacuation speed
    vtkFile << "SCALARS speed float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.speed;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    // Radial grid index
    vtkFile << "SCALARS ridx int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        int vval = evacuee.prev_rad_idx;
        swapBytes_int(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    // // Inhalation dose for nuclide 0
    // vtkFile << "SCALARS dose_inhalation_0 float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (const auto& evacuee : evacuees) {
    //     float vval = evacuee.dose_inhalations[0];
    //     swapBytes(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // // Inhalation dose for nuclide 1
    // vtkFile << "SCALARS dose_inhalation_1 float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (const auto& evacuee : evacuees) {
    //     float vval = evacuee.dose_inhalations[1];
    //     swapBytes(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // // Inhalation dose for nuclide 2
    // vtkFile << "SCALARS dose_inhalation_2 float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (const auto& evacuee : evacuees) {
    //     float vval = evacuee.dose_inhalations[2];
    //     swapBytes(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // // Cloudshine dose for nuclide 0
    // vtkFile << "SCALARS dose_cloudshine_0 float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (const auto& evacuee : evacuees) {
    //     float vval = evacuee.dose_cloudshines[0];
    //     swapBytes(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // // Cloudshine dose for nuclide 1
    // vtkFile << "SCALARS dose_cloudshine_1 float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (const auto& evacuee : evacuees) {
    //     float vval = evacuee.dose_cloudshines[1];
    //     swapBytes(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // // Cloudshine dose for nuclide 2
    // vtkFile << "SCALARS dose_cloudshine_2 float 1\n";
    // vtkFile << "LOOKUP_TABLE default\n";
    // for (const auto& evacuee : evacuees) {
    //     float vval = evacuee.dose_cloudshines[2];
    //     swapBytes(vval);
    //     vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    // }

    // Cumulative cloudshine dose
    vtkFile << "SCALARS dose_cloudshine_cumulative float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_cloudshine_cumulative;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile.close();
}

void Gpuff::evac_output_binary_RCAP_xy(int timestep) {

    std::ostringstream filenameStream;

    std::string path;

    cudaMemcpy(evacuees.data(), d_evacuees, evacuees.size() * sizeof(Evacuee), cudaMemcpyDeviceToHost);

#ifdef _WIN32
    path = ".\\evac";
    _mkdir(path.c_str());
    filenameStream << ".\\evac\\evac_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#else
    path = "./evac";
    mkdir(path.c_str(), 0777);
    filenameStream << "./evac/evac_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#endif

    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "evacuation data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    vtkFile << "POINTS " << evacuees.size() << " float\n";
    for (const auto& evacuee : evacuees) {
        float r = evacuee.r;
        float theta = evacuee.theta;
        float x = evacuee.x;
        float y = evacuee.y;
        float z = 0.0f;

        swapBytes(x);
        swapBytes(y);
        swapBytes(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }

    vtkFile << "POINT_DATA " << evacuees.size() << "\n";

    vtkFile << "SCALARS population float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.population;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS speed float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.speed;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS dose_inhalation float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_inhalation;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS dose_cloudshine float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_cloudshine;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile.close();
}

void Gpuff::evac_output_binary_RCAP_xy_single(int timestep) {

    std::ostringstream filenameStream;

    std::string path;

    cudaMemcpy(evacuees.data(), d_evacuees, evacuees.size() * sizeof(Evacuee), cudaMemcpyDeviceToHost);

#ifdef _WIN32
    path = ".\\evac";
    _mkdir(path.c_str());
    filenameStream << ".\\evac\\evac_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#else
    path = "./evac";
    mkdir(path.c_str(), 0777);
    filenameStream << "./evac/evac_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#endif

    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "evacuation data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    vtkFile << "POINTS " << evacuees.size() << " float\n";
    for (const auto& evacuee : evacuees) {
        float r = evacuee.r;
        float theta = evacuee.theta;
        float x = evacuee.x;
        float y = evacuee.y;
        float z = 0.0f;

        swapBytes(x);
        swapBytes(y);
        swapBytes(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }

    vtkFile << "POINT_DATA " << evacuees.size() << "\n";

    vtkFile << "SCALARS population float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.population;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS speed float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.speed;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS dose_multi_unit float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_inhalation;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS dose_single_unit float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_cloudshine;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile.close();
}


void Gpuff::evac_output_binary_RCAP_cpu(int timestep) {

    std::ostringstream filenameStream;

    std::string path;

    //cudaMemcpy(evacuees.data(), d_evacuees, evacuees.size() * sizeof(Evacuee), cudaMemcpyDeviceToHost);

#ifdef _WIN32
    path = ".\\evac";
    _mkdir(path.c_str());
    filenameStream << ".\\evac\\evac_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#else
    path = "./evac";
    mkdir(path.c_str(), 0777);
    filenameStream << "./evac/evac_RCAP_" << std::setfill('0')
        << std::setw(5) << timestep << ".vtk";
#endif

    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "evacuation data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    vtkFile << "POINTS " << evacuees.size() << " float\n";
    for (const auto& evacuee : evacuees) {
        float r = evacuee.r;
        float theta = evacuee.theta;
        float x = r * cos(theta);
        float y = r * sin(theta);
        float z = 0.0f;

        swapBytes(x);
        swapBytes(y);
        swapBytes(z);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }

    vtkFile << "POINT_DATA " << evacuees.size() << "\n";

    vtkFile << "SCALARS population float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.population;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS speed float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.speed;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS dose_inhalation float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_inhalation;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS dose_cloudshine float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_cloudshine;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile.close();
}

void Gpuff::grid_output_binary(RectangleGrid& rect, float* h_concs){

    std::string path;
    std::string filename;

    #ifdef _WIN32
        path = ".\\grids";
        _mkdir(path.c_str());
        filename = ".\\grids\\grid.vtk";
    #else
        path = "./grids";
        mkdir(path.c_str(), 0777);
        filename = "./grids/grid.vtk";
    #endif

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()){
    std::cerr << "Cannot open file for writing: " << filename << std::endl;
    return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "Grid data generated by RectangleGrid\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET STRUCTURED_GRID\n";
    vtkFile << "DIMENSIONS " << rect.cols << " " << rect.rows << " " << 1 << "\n";
    vtkFile << "POINTS " << rect.rows * rect.cols << " float\n";

    for (int i = 0; i < rect.rows; ++i) {
        for (int j = 0; j < rect.cols; ++j) {

            int index = i * rect.cols + j;

            float x = rect.grid[index].x;
            float y = rect.grid[index].y;
            float z = 0.0f;

            swapBytes(x);
            swapBytes(y);
            swapBytes(z);

            vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
            vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
            vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));

        }
    }

        vtkFile << "\nPOINT_DATA " << rect.rows * rect.cols << "\n";
        vtkFile << "SCALARS concentration float 1\n";
        vtkFile << "LOOKUP_TABLE default\n";
    
        for (int i = 0; i < rect.rows; ++i) {
            for (int j = 0; j < rect.cols; ++j) {
                float conc = h_concs[i * rect.cols + j];
                swapBytes(conc);
                vtkFile.write(reinterpret_cast<char*>(&conc), sizeof(float));
            }
        }
    

    vtkFile.close();
}

void Gpuff::grid_output_binary_val(RectangleGrid& rect, float* h_concs){

    std::string path;

    #ifdef _WIN32
        path = "./grids";
        _mkdir(path.c_str());
    #else
        path = ".\\grids";
        mkdir(path.c_str(), 0777);
    #endif

    for(int zidx = 0; zidx < 22; ++zidx){

        std::stringstream ss;

        #ifdef _WIN32
            ss << ".\\grids\\grid" << std::setw(3) << std::setfill('0') << zidx << ".vtk";
        #else
            ss << "./grids/grid" << std::setw(3) << std::setfill('0') << zidx << ".vtk";
        #endif

        std::string filename = ss.str();
    
        std::ofstream vtkFile(filename, std::ios::binary);
    
        if (!vtkFile.is_open()) {
            std::cerr << "Cannot open file for writing: " << filename << std::endl;
            continue; 
        }

        vtkFile << "# vtk DataFile Version 4.0\n";
        vtkFile << "Grid data generated by RectangleGrid\n";
        vtkFile << "BINARY\n";
        vtkFile << "DATASET STRUCTURED_GRID\n";
        vtkFile << "DIMENSIONS " << rect.cols << " " << rect.rows << " " << 1 << "\n";
        vtkFile << "POINTS " << rect.rows * rect.cols << " float\n";

        for (int i = 0; i < rect.rows; ++i) {
            for (int j = 0; j < rect.cols; ++j) {

                int index = i * rect.cols * 21 + j * 21 + zidx;

                float x = rect.grid[index].x;
                float y = rect.grid[index].y;
                float z = 0.0f;

                swapBytes(x);
                swapBytes(y);
                swapBytes(z);

                vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
                vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
                vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));

            }
        }

            vtkFile << "\nPOINT_DATA " << rect.rows * rect.cols << "\n";
            vtkFile << "SCALARS concentration float 1\n";
            vtkFile << "LOOKUP_TABLE default\n";
        
            for (int i = 0; i < rect.rows; ++i) {
                for (int j = 0; j < rect.cols; ++j) {
                    float conc = h_concs[i * rect.cols * 21 + j * 21 + zidx];
                    swapBytes(conc);
                    vtkFile.write(reinterpret_cast<char*>(&conc), sizeof(float));
                }
            }
        

        vtkFile.close();
    }
}

void Gpuff::grid_output_csv(RectangleGrid& rect, float* h_concs){

    std::string path;
    std::stringstream ss;

    #ifdef _WIN32
        path = ".\\grids_csv";
        _mkdir(path.c_str());
        ss << ".\\grids_csv\\grid.csv";
    #else
        path = "./grids_csv";
        mkdir(path.c_str(), 0777);
        ss << "./grids_csv/grid.csv";
    #endif

    std::string filename = ss.str();

    std::ofstream csvFile(filename);

    if (!csvFile.is_open()) {
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
    }

    for (int i = 0; i < rect.rows; ++i) {
        for (int j = 0; j < rect.cols; ++j){

            int index = i * rect.cols + j;
            float conc = h_concs[index];

            csvFile << conc;
            if (j < rect.cols - 1) {
                csvFile << ",";
            }
        }
        csvFile << "\n";
    }

    csvFile.close();

}

void Gpuff::receptor_output_binary_RCAP(int timestep){

    cudaMemcpy(receptors.data(), d_receptors, 16*RNUM * sizeof(receptors_RCAP), cudaMemcpyDeviceToHost);

    std::ostringstream filenameStream;

    std::string path;

    #ifdef _WIN32
        path = ".\\receptors";
        _mkdir(path.c_str());
        filenameStream << ".\\receptors\\receptors_" << std::setfill('0') 
        << std::setw(5) << timestep << ".vtk";
    #else
        path = "./receptors";
        mkdir(path.c_str(), 0777);
        filenameStream << "./receptors/receptors_" << std::setfill('0') 
        << std::setw(5) << timestep << ".vtk";
    #endif


    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename, std::ios::binary);

    if (!vtkFile.is_open()){
        std::cerr << "Cannot open file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 3.0\n";
    vtkFile << "receptors_RCAP data\n";
    vtkFile << "BINARY\n";
    vtkFile << "DATASET POLYDATA\n";

    vtkFile << "POINTS " << receptors.size() << " float\n";
    for (const auto& receptor : receptors) {
        float x = receptor.x, y = receptor.y, z = 0.0f;  

        swapBytes(x);
        swapBytes(y);

        vtkFile.write(reinterpret_cast<char*>(&x), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&y), sizeof(float));
        vtkFile.write(reinterpret_cast<char*>(&z), sizeof(float));
    }

    vtkFile << "POINT_DATA " << receptors.size() << "\n";
    vtkFile << "SCALARS concentration float 1\n"; 
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& receptor : receptors) {
        float conc = receptor.conc;

        swapBytes(conc);

        vtkFile.write(reinterpret_cast<char*>(&conc), sizeof(float));
    }

    vtkFile.close();
}


// Write Plant/Source Locations to VTK File
//
// Exports nuclear plant unit locations and release heights to VTK format.
// Used for visualizing source locations in multi-unit accident scenarios.
//
// VTK File Format:
//   Format: Legacy VTK (Version 4.0)
//   Type: POLYDATA (point cloud dataset)
//   Encoding: ASCII (human-readable)
//
// Coordinate Transformation:
//   Input: Geographic coordinates (latitude, longitude in degrees)
//   Output: Local Cartesian coordinates (meters)
//
//   Conversion factors:
//     metersPerLatDegree = 111320.0 m/degree (approximate)
//     metersPerLonDegree = 88290.0 m/degree (for Korea region ~37N)
//
//   Transformation:
//     x = (lon - baseLon) * metersPerLonDegree
//     y = (lat - baseLat) * metersPerLatDegree
//     z = release_height
//
// Base Coordinate:
//   Reference point (0, 0) is set to the first unit's location
//   All other units positioned relative to this reference
//
// Use Case:
//   Display source locations in ParaView
//   Overlay with puff and evacuee data
//   Multi-unit scenario visualization
//
// File Naming Convention:
//   Format: plant_RCAP_00001.vtk (fixed to timestep 1)
//   Directory: ./plants/ or .\plants\
//
// Parameters:
//   input_num: Number of nuclear plant units
//   RT: Vector of RadioNuclideTransport objects (contains position and release data)
//   ND: Vector of NuclideData objects (not currently used)
//
void Gpuff::plant_output_binary_RCAP(
    int input_num,
    const std::vector<RadioNuclideTransport>& RT,
    const std::vector<NuclideData>& ND
) {
    // Calculate total number of puffs across all units
    int totalPuffs = 0;
    for (int i = 0; i < input_num; i++) {
        totalPuffs += RT[i].nPuffTotal;
    }

    puffs_RCAP.reserve(RCAP_metdata.size() * totalPuffs);

    // Set base coordinate to first unit location
    double baseLon = RT[0].lon;
    double baseLat = RT[0].lat;

    // Approximate conversion factors for Korea region
    const double metersPerLatDegree = 111320.0;  // Constant globally
    const double metersPerLonDegree = 88290.0;   // At ~37N latitude

    // Create output directory
    std::string path = ".\\plants";
#ifdef _WIN32
    _mkdir(path.c_str());
#else
    mkdir(path.c_str(), 0777);
#endif

    // Generate filename
    std::ostringstream filenameStream;
    filenameStream << ".\\plants\\plant_RCAP_" << std::setfill('0') << std::setw(5) << 1 << ".vtk";
    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename);
    if (!vtkFile.is_open()) {
        std::cerr << "Cannot open VTK file for writing: " << filename << std::endl;
        return;
    }

    // Write VTK header (ASCII format)
    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "Transformed puff coordinates\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET POLYDATA\n";
    vtkFile << "POINTS " << (input_num * RT[0].nPuffTotal) << " float\n";

    // Write plant unit positions
    for (int j = 0; j < input_num; j++) {
        for (int k = 0; k < RT[j].nPuffTotal; k++) {
            // Transform geographic to local Cartesian coordinates
            float _x = static_cast<float>((RT[j].lon - baseLon) * metersPerLonDegree);
            float _y = static_cast<float>((RT[j].lat - baseLat) * metersPerLatDegree);
            float _z = RT[j].RT_puffs[k].rele_height;

            vtkFile << std::fixed << std::setprecision(6) << _x << " " << _y << " " << _z << "\n";
        }
    }

    vtkFile.close();
}

// ============================================================================
// init_max_tracking
//
// Initialize the maximum value tracking arrays. Call this once before simulation.
// ============================================================================
void Gpuff::init_max_tracking(int numRad) {
    max_center_air_conc.assign(numRad, 0.0f);
    max_ground_air_conc.assign(numRad, 0.0f);
    max_ground_conc.assign(numRad, 0.0f);
    max_xq.assign(numRad, 0.0f);
    max_sigma_y.assign(numRad, 0.0f);
    max_sigma_z.assign(numRad, 0.0f);
    max_dir_center_air.assign(numRad, 1);
    max_dir_ground_air.assign(numRad, 1);
    max_dir_ground.assign(numRad, 1);
    max_dir_xq.assign(numRad, 1);
    max_tracking_nuclide = -1;
}

// ============================================================================
// update_max_values_cpu (Legacy - kept for reference and comparison)
//
// Update maximum values using RCAP Segmented Plume model on CPU.
// For each ring, calculate concentration at the ring's center distance using
// Pasquill-Gifford dispersion coefficients.
//
// NOTE: This function is now replaced by move_puffs_by_wind_RCAP3 GPU kernel.
// The GPU kernel performs identical calculations. This function is kept for
// debugging and validation purposes.
//
// RCAP Center Air Conc formula: Q / (π * σ_y * σ_z * u)
// This is the crosswind-integrated Gaussian plume formula.
// ============================================================================
void Gpuff::update_max_values_cpu(const SimulationControl& SC, const std::vector<NuclideData>& ND) {

    // Copy puff data from device to host
    cudaMemcpy(puffs_RCAP.data(), d_puffs_RCAP, puffs_RCAP.size() * sizeof(Puffcenter_RCAP), cudaMemcpyDeviceToHost);

    int numRad = SC.numRad;
    int numTheta = SC.numTheta;

    // Find nuclide to track (first one with non-zero concentration)
    if (max_tracking_nuclide < 0) {
        for (int n = 0; n < MAX_NUCLIDES; n++) {
            for (size_t i = 0; i < puffs_RCAP.size(); i++) {
                if (puffs_RCAP[i].conc[n] > 0.0f) {
                    max_tracking_nuclide = n;
                    break;
                }
            }
            if (max_tracking_nuclide >= 0) break;
        }
    }

    if (max_tracking_nuclide < 0) return;

    // For each active puff, calculate concentrations at each ring it has passed
    for (size_t i = 0; i < puffs_RCAP.size(); i++) {
        const Puffcenter_RCAP& puff = puffs_RCAP[i];
        if (puff.flag == 0) continue;

        float Q = puff.conc[max_tracking_nuclide];
        if (Q <= 0.0f) continue;

        float ws = puff.windvel > 0.1f ? puff.windvel : 0.1f;
        int stab_class = puff.stab - 1;  // Convert from 1-6 to 0-5 index for sigma arrays
        float H = puff.he > 0.0f ? puff.he : puff.z;  // Effective release height

        // Current puff distance from source
        float puff_distance = sqrtf(puff.x * puff.x + puff.y * puff.y);

        // Get puff direction for theta index
        float theta = atan2f(puff.y, puff.x);
        if (theta < 0) theta += 2.0f * PI;
        int theta_idx = static_cast<int>(theta / (2.0f * PI) * numTheta) % numTheta;

        // Calculate concentration at each ring the puff has passed through
        for (int rad_idx = 0; rad_idx < numRad; rad_idx++) {
            // Get ring boundaries
            float r_inner = (rad_idx == 0) ? 0.0f : SC.ir_distances[rad_idx - 1];
            float r_outer = SC.ir_distances[rad_idx];

            // Use ring center (midpoint) as reference distance
            float ring_distance = (r_inner + r_outer) / 2.0f;

            // [Alternative] Use outer boundary - commented out for comparison
            // float ring_distance = r_outer;  // Use outer boundary like RCAP

            // Only calculate if puff has reached or passed this ring
            if (puff_distance < r_inner) continue;

            // Calculate sigma_y and sigma_z at ring distance using Briggs-McElroy-Pooler model
            float sigma_y = Sigma_h_Briggs_McElroy_Pooler_cpu(stab_class, ring_distance);
            float sigma_z = Sigma_z_Briggs_McElroy_Pooler_cpu(stab_class, ring_distance);

            // σy 보정: Ft (샘플시간 보정인자)만 적용
            const float FT = 1.27f;  // Ft = (td/tref)^0.2 = (600/180)^0.2
            sigma_y = FT * sigma_y;

            // σz 상한값 적용: σz_max = mixing_height / sqrt(2π)
            // mixing height 내에서 균일 혼합 가정
            const float SQRT_2PI = 2.506628f;
            const float MIXING_HEIGHT = 1000.0f;  // mixing height (m)
            float sigma_z_max = MIXING_HEIGHT / SQRT_2PI;
            sigma_z = fminf(sigma_z, sigma_z_max);

            // Ensure minimum values
            sigma_y = fmaxf(sigma_y, 0.1f);
            sigma_z = fmaxf(sigma_z, 0.1f);

            // ============================================================
            // RCAP Ground Reflection Model (RCAP 모델설명서 식 3.2.4, 3.2.5)
            // ============================================================
            // g = g1 + g2 + g3
            // g1: 반사 없는 수직확산항
            // g2: 지면반사에 의한 수직확산항
            // g3: 다중반사(역전층)에 의한 수직확산항
            // ============================================================

            float L = 1000.0f;  // 역전층 높이 (m)
            float z = 0.0f;     // 지표면 높이에서 계산
            float sigma_z2 = sigma_z * sigma_z;
            float two_sigma_z2 = 2.0f * sigma_z2;

            // g1: 반사 없는 수직확산항 - exp[-(z-H)²/(2σz²)]
            float g1 = expf(-((z - H) * (z - H)) / two_sigma_z2);

            // g2: 지면반사에 의한 수직확산항 - exp[-(z+H)²/(2σz²)]
            float g2 = expf(-((z + H) * (z + H)) / two_sigma_z2);

            // g3: 다중반사 (역전층 반사) - n≤5 고려
            float g3 = 0.0f;
            for (int n = 1; n <= 5; n++) {
                float iL = static_cast<float>(n) * L;
                // g_i = exp[-(z-H-2iL)²/(2σz²)] + exp[-(z+H-2iL)²/(2σz²)]
                //     + exp[-(z-H+2iL)²/(2σz²)] + exp[-(z+H+2iL)²/(2σz²)]
                g3 += expf(-((z - H - 2.0f * iL) * (z - H - 2.0f * iL)) / two_sigma_z2);
                g3 += expf(-((z + H - 2.0f * iL) * (z + H - 2.0f * iL)) / two_sigma_z2);
                g3 += expf(-((z - H + 2.0f * iL) * (z - H + 2.0f * iL)) / two_sigma_z2);
                g3 += expf(-((z + H + 2.0f * iL) * (z + H + 2.0f * iL)) / two_sigma_z2);
            }

            // 수직확산 매개함수 g = g1 + g2 + g3
            float g_total = g1 + g2 + g3;

            // RCAP 공식: C = Q/(u) * f/(σy√2π) * g/(σz√2π)
            // 여기서 f = exp[-y²/(2σy²)], 중심선(y=0)에서 f=1
            // 따라서: C = Q / (2π * σy * σz * u) * g
            float base_conc = Q / (2.0f * PI * sigma_y * sigma_z * ws);

            // Center Air Conc: 플룸 중심선(z=H)에서의 농도 (지면반사 포함)
            // z=H일 때: g1=1, g2=exp[-(2H)²/(2σz²)]
            float g1_center = 1.0f;
            float g2_center = expf(-(4.0f * H * H) / two_sigma_z2);
            float g3_center = 0.0f;
            for (int n = 1; n <= 5; n++) {
                float iL = static_cast<float>(n) * L;
                g3_center += expf(-((H - H - 2.0f * iL) * (H - H - 2.0f * iL)) / two_sigma_z2);
                g3_center += expf(-((H + H - 2.0f * iL) * (H + H - 2.0f * iL)) / two_sigma_z2);
                g3_center += expf(-((H - H + 2.0f * iL) * (H - H + 2.0f * iL)) / two_sigma_z2);
                g3_center += expf(-((H + H + 2.0f * iL) * (H + H + 2.0f * iL)) / two_sigma_z2);
            }
            float g_center = g1_center + g2_center + g3_center;
            float center_air = base_conc * g_center;

            // Ground Air Conc: 지표면(z=0)에서의 농도 (지면반사 포함)
            float ground_air = base_conc * g_total;

            // Ground concentration (deposition)
            // Using cesium dry deposition velocity from RCAP
            float vd = puff.drydep_vel > 0.0f ? puff.drydep_vel : 0.001f;
            float ground_conc = ground_air * vd * dt;

            // X/Q (dilution factor) - normalized by source term
            float xq = ground_air / Q;

            // Update maximum values
            if (center_air > max_center_air_conc[rad_idx]) {
                max_center_air_conc[rad_idx] = center_air;
                max_dir_center_air[rad_idx] = theta_idx + 1;
                max_sigma_y[rad_idx] = sigma_y;  // 농도 계산에 사용된 sigma 저장
                max_sigma_z[rad_idx] = sigma_z;
            }
            if (ground_air > max_ground_air_conc[rad_idx]) {
                max_ground_air_conc[rad_idx] = ground_air;
                max_dir_ground_air[rad_idx] = theta_idx + 1;
            }
            if (ground_conc > max_ground_conc[rad_idx]) {
                max_ground_conc[rad_idx] = ground_conc;
                max_dir_ground[rad_idx] = theta_idx + 1;
            }
            if (xq > max_xq[rad_idx]) {
                max_xq[rad_idx] = xq;
                max_dir_xq[rad_idx] = theta_idx + 1;
            }
        }
    }
}

// ============================================================================
// Print Results Summary (GPU version)
// ============================================================================
// Prints a formatted table of maximum radionuclide dispersion values
// for each radial distance ring. Output format matches RCAP legacy output.
//
// This version copies GPU-computed values from move_puffs_by_wind_RCAP3 kernel
// to host memory and outputs them without any CPU recalculation.
//
// Metrics per ring (all computed on GPU):
//   - Center Air Concentration (Bq-s/m3): Time-integrated air concentration
//   - Ground Air Concentration (Bq-s/m3): Ground-level air concentration
//   - Center Ground Concentration (Bq/m2): Ground deposition
//   - Ground Dilution X/Q (s/m3): Dilution factor
//   - Sigma-y, Sigma-z (m): Dispersion parameters
//
// Parameters:
//   SC: SimulationControl - Contains radial distances and grid info
//   ND: NuclideData vector - Nuclide names and properties
// ============================================================================
void Gpuff::print_results_summary(const SimulationControl& SC, const std::vector<NuclideData>& ND) {

    int numRad = SC.numRad;
    int numTheta = SC.numTheta;

    // Copy GPU-computed max values to host
    cudaMemcpy(max_center_air_conc.data(), d_max_center_air_conc, numRad * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_ground_air_conc.data(), d_max_ground_air_conc, numRad * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_ground_conc.data(), d_max_ground_conc, numRad * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_xq.data(), d_max_xq, numRad * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_sigma_y.data(), d_max_sigma_y, numRad * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_sigma_z.data(), d_max_sigma_z, numRad * sizeof(float), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_dir_center_air.data(), d_max_dir_center_air, numRad * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_dir_ground_air.data(), d_max_dir_ground_air, numRad * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_dir_ground.data(), d_max_dir_ground, numRad * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(max_dir_xq.data(), d_max_dir_xq, numRad * sizeof(int), cudaMemcpyDeviceToHost);
    cudaMemcpy(&max_tracking_nuclide, d_max_tracking_nuclide, sizeof(int), cudaMemcpyDeviceToHost);

    // Get nuclide name for display
    std::string nuclide_name = "Unknown";
    if (max_tracking_nuclide >= 0 && max_tracking_nuclide < MAX_NUCLIDES) {
        nuclide_name = ND[max_tracking_nuclide].name;
    }

    if (max_tracking_nuclide < 0) {
        std::cout << "\n[WARNING] No nuclide with non-zero concentration found.\n";
        return;
    }

    // Open output file
    std::ofstream outFile(".\\output\\dispersion_results.txt");
    if (!outFile.is_open()) {
        std::cerr << "[ERROR] Failed to open output file: .\\output\\dispersion_results.txt" << std::endl;
        return;
    }

    // Helper lambda for printing to both console and file
    auto printBoth = [&](const std::string& text) {
        //std::cout << text;
        outFile << text;
    };

    // Print header
    printBoth("\n ---------------------------------------------------------\n");
    printBoth("  Maximum Values for Radionuclide Dispersion (All Plumes) \n");
    printBoth(" ---------------------------------------------------------\n");
    printBoth(std::string("  Dispersion Model: ") + (isPG ? "Pasquill-Gifford" : "Briggs-McElroy-Pooler") + "\n");
    printBoth(std::string("  Terrain Type    : ") + (isRural ? "Rural" : "Urban") + "\n");
    printBoth(" ---------------------------------------------------------\n\n");

    // Print nuclide name and column headers
    std::ostringstream headerLine;
    headerLine << "  " << std::left << std::setw(26) << nuclide_name << " |";
    for (int i = 0; i < numRad; i++) {
        float r_start = (i == 0) ? 0.0f : SC.ir_distances[i-1];
        float r_end = SC.ir_distances[i];

        std::ostringstream header;
        // Format distance in meters with appropriate precision
        if (r_end < 1000.0f) {
            header << std::fixed << std::setprecision(0) << r_start << "~" << r_end << "m";
        } else if (r_end < 10000.0f) {
            header << std::fixed << std::setprecision(1) << r_start/1000.0f << "~" << r_end/1000.0f << "km";
        } else {
            header << std::fixed << std::setprecision(0) << r_start/1000.0f << "~" << r_end/1000.0f << "km";
        }
        headerLine << " " << std::setw(12) << header.str();
    }
    headerLine << "\n";
    printBoth(headerLine.str());

    // Print (iDir/Total) row
    std::ostringstream dirHeaderLine;
    dirHeaderLine << std::setw(28) << " " << " ";
    for (int i = 0; i < numRad; i++) {
        dirHeaderLine << "  (iDir/Total)  ";
    }
    dirHeaderLine << "\n";
    printBoth(dirHeaderLine.str());

    // Print separator
    std::ostringstream sepLine;
    sepLine << "------------------------------+";
    for (int i = 0; i < numRad; i++) {
        sepLine << "---------------" << (i < numRad - 1 ? "+" : "-");
    }
    sepLine << "\n";
    printBoth(sepLine.str());

    // Helper lambda for printing a data row
    auto printRow = [&](const std::string& label, const std::vector<float>& values,
                        const std::vector<int>& dirs, int total) {
        std::ostringstream dataLine;
        dataLine << " " << std::left << std::setw(27) << label << " |";
        for (int i = 0; i < numRad; i++) {
            dataLine << "  " << std::scientific << std::setprecision(5) << values[i] << "   ";
        }
        dataLine << "\n";
        printBoth(dataLine.str());

        // Print direction row
        std::ostringstream dirLine;
        dirLine << std::setw(28) << " " << " ";
        for (int i = 0; i < numRad; i++) {
            dirLine << "  (" << std::setw(2) << dirs[i] << "/" << std::setw(2) << total << ")       ";
        }
        dirLine << "\n";
        printBoth(dirLine.str());
    };

    // Print data rows
    printRow("Center Air Conc. (Bq-s/m3)", max_center_air_conc, max_dir_center_air, numTheta);
    printRow("Ground Air Conc. (Bq-s/m3)", max_ground_air_conc, max_dir_ground_air, numTheta);
    printRow("Center Ground Conc. (Bq/m2)", max_ground_conc, max_dir_ground, numTheta);
    printRow("Ground Dilution, X/Q (s/m3)", max_xq, max_dir_xq, numTheta);

    // ============================================================
    // Sigma-y, Sigma-z 출력 (거리별)
    // ============================================================
    // 첫 번째 puff의 안정도 등급 사용 (모든 puff가 동일하다고 가정)
    int stab_class = 0;
    if (!puffs.empty()) {
        stab_class = puffs[0].stab - 1;  // 1-based to 0-based
        if (stab_class < 0) stab_class = 0;
        if (stab_class > 6) stab_class = 6;
    }

    // 농도 계산에 사용된 실제 sigma 값 출력 (max_sigma_y, max_sigma_z에 저장됨)
    // Helper lambda for printing sigma row (without direction info)
    auto printSigmaRow = [&](const std::string& label, const std::vector<float>& values) {
        std::ostringstream dataLine;
        dataLine << " " << std::left << std::setw(27) << label << " |";
        for (int i = 0; i < numRad; i++) {
            dataLine << "  " << std::scientific << std::setprecision(5) << values[i] << "   ";
        }
        dataLine << "\n";
        printBoth(dataLine.str());
    };

    printSigmaRow("Plume Sigma-y (m)", max_sigma_y);
    printSigmaRow("Plume Sigma-z (m)", max_sigma_z);

    printBoth("\n");

    outFile.close();
    std::cout << "[INFO] Results saved to: .\\output\\dispersion_results.txt" << std::endl;
}

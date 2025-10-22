#include "gpuff.cuh"

int Gpuff::countflag(){
    int count = 0;
    for(int i = 0; i < nop; ++i) if(puffs[i].flag == 1) count++;
    return count;
}

int Gpuff::countflag_RCAP() {
    int count = 0;
    for (int i = 0; i < puffs_RCAP.size(); ++i) if (puffs_RCAP[i].flag == true) count++;
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


void Gpuff::swapBytes(float& value){
    char* valuePtr = reinterpret_cast<char*>(&value);
    std::swap(valuePtr[0], valuePtr[3]);
    std::swap(valuePtr[1], valuePtr[2]);
}

void Gpuff::swapBytes_int(int& value){
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

void Gpuff::puff_output_binary_RCAP(int timestep) {

    cudaMemcpy(puffs_RCAP.data(), d_puffs_RCAP, puffs_RCAP.size() * sizeof(Puffcenter_RCAP), cudaMemcpyDeviceToHost);

    int part_num = countflag_RCAP();
    //std::cout << "part_num = " << part_num << std::endl;
    //std::cout << "puffs_RCAP.size() = " << puffs_RCAP.size() << std::endl;


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

    //vtkFile << "SCALARS windvel float 1\n";
    //vtkFile << "LOOKUP_TABLE default\n";
    //for (int i = 0; i < puffs_RCAP.size(); ++i) {
    //    if (!puffs_RCAP[i].flag) continue;
    //    float vval = puffs_RCAP[i].windvel;
    //    swapBytes(vval);
    //    vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    //}

    //vtkFile << "SCALARS windir float 1\n";
    //vtkFile << "LOOKUP_TABLE default\n";
    //for (int i = 0; i < puffs_RCAP.size(); ++i) {
    //    if (!puffs_RCAP[i].flag) continue;
    //    float vval = puffs_RCAP[i].windir;
    //    swapBytes(vval);
    //    vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    //}

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

void Gpuff::puff_output_binary_RCAP_cpu(int timestep) {

    //cudaMemcpy(puffs_RCAP.data(), d_puffs_RCAP, puffs_RCAP.size() * sizeof(Puffcenter_RCAP), cudaMemcpyDeviceToHost);

    int part_num = countflag_RCAP();
    //std::cout << "part_num = " << part_num << std::endl;
    //std::cout << "puffs_RCAP.size() = " << puffs_RCAP.size() << std::endl;


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

void Gpuff::evac_output_binary_RCAP(int timestep) {

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

        if (evacuee.prev_theta_idx == 8) printf("%f  ", evacuee.speed);
    }
    printf("\n");

    vtkFile << "SCALARS speed float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.speed;
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS ridx int 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        int vval = evacuee.prev_rad_idx;
        swapBytes_int(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(int));
    }

    vtkFile << "SCALARS dose_inhalation_0 float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_inhalations[0];
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }
    vtkFile << "SCALARS dose_inhalation_1 float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_inhalations[1];
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }
    vtkFile << "SCALARS dose_inhalation_2 float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_inhalations[2];
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }

    vtkFile << "SCALARS dose_cloudshine_0 float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_cloudshines[0];
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }
    vtkFile << "SCALARS dose_cloudshine_1 float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_cloudshines[1];
        swapBytes(vval);
        vtkFile.write(reinterpret_cast<char*>(&vval), sizeof(float));
    }
    vtkFile << "SCALARS dose_cloudshine_2 float 1\n";
    vtkFile << "LOOKUP_TABLE default\n";
    for (const auto& evacuee : evacuees) {
        float vval = evacuee.dose_cloudshines[2];
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


void Gpuff::plant_output_binary_RCAP(
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

    const double metersPerLatDegree = 111320.0;
    const double metersPerLonDegree = 88290.0;

    std::string path = ".\\plants";
#ifdef _WIN32
    _mkdir(path.c_str()); // Windows
#else
    mkdir(path.c_str(), 0777); // POSIX
#endif

    std::ostringstream filenameStream;
    filenameStream << ".\\plants\\plant_RCAP_" << std::setfill('0') << std::setw(5) << 1 << ".vtk";
    std::string filename = filenameStream.str();

    std::ofstream vtkFile(filename);
    if (!vtkFile.is_open()) {
        std::cerr << "Cannot open VTK file for writing: " << filename << std::endl;
        return;
    }

    vtkFile << "# vtk DataFile Version 4.0\n";
    vtkFile << "Transformed puff coordinates\n";
    vtkFile << "ASCII\n";
    vtkFile << "DATASET POLYDATA\n";
    vtkFile << "POINTS " << (input_num * RT[0].nPuffTotal) << " float\n";

    for (int j = 0; j < input_num; j++) {
        for (int k = 0; k < RT[j].nPuffTotal; k++) {
            float _x = static_cast<float>((RT[j].lon - baseLon) * metersPerLonDegree);
            float _y = static_cast<float>((RT[j].lat - baseLat) * metersPerLatDegree);
            float _z = RT[j].RT_puffs[k].rele_height;

            vtkFile << std::fixed << std::setprecision(6) << _x << " " << _y << " " << _z << "\n";
        }
    }

    vtkFile.close();
}

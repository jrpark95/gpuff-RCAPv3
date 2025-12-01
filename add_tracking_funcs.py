with open('gpuff_plot.cuh', 'r', encoding='utf-8') as f:
    content = f.read()

old_text = '''    vtkFile.close();
}

// ============================================================================
// Print Results Summary
// ============================================================================
// Prints a formatted table of maximum radionuclide dispersion values'''

new_text = '''    vtkFile.close();
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
    max_dir_center_air.assign(numRad, 1);
    max_dir_ground_air.assign(numRad, 1);
    max_dir_ground.assign(numRad, 1);
    max_dir_xq.assign(numRad, 1);
    max_tracking_nuclide = -1;
}

// ============================================================================
// update_max_values
//
// Update maximum values based on current puff state. Call this every timestep.
// ============================================================================
void Gpuff::update_max_values(const SimulationControl& SC, const std::vector<NuclideData>& ND) {

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

    // Calculate values for each puff and update maximum per ring
    for (size_t i = 0; i < puffs_RCAP.size(); i++) {
        const Puffcenter_RCAP& puff = puffs_RCAP[i];
        if (puff.flag == 0) continue;

        float r = sqrtf(puff.x * puff.x + puff.y * puff.y);
        float theta = atan2f(puff.y, puff.x);
        if (theta < 0) theta += 2.0f * PI;

        // Find radial ring index
        int rad_idx = numRad - 1;
        for (int ri = 0; ri < numRad; ri++) {
            if (r < SC.ir_distances[ri]) {
                rad_idx = ri;
                break;
            }
        }

        // Find angular sector index
        int theta_idx = static_cast<int>(theta / (2.0f * PI) * numTheta) % numTheta;

        // Calculate concentrations using Gaussian plume formula
        float sigma_y = puff.sigma_h > 0.1f ? puff.sigma_h : 0.1f;
        float sigma_z = puff.sigma_z > 0.1f ? puff.sigma_z : 0.1f;
        float Q = puff.conc[max_tracking_nuclide];
        float ws = puff.windvel > 0.1f ? puff.windvel : 0.1f;

        if (Q <= 0.0f) continue;

        // Center air concentration (at plume centerline)
        float center_air = Q / (2.0f * PI * sigma_y * sigma_z * ws);

        // Ground-level air concentration (with ground reflection)
        float H = puff.z;  // Release height
        float ground_factor = 2.0f * expf(-0.5f * (H * H) / (sigma_z * sigma_z));
        float ground_air = center_air * ground_factor;

        // Ground concentration (simplified deposition model)
        float vd = 0.01f;  // Dry deposition velocity (m/s)
        float ground_conc = ground_air * vd * dt;

        // X/Q (dilution factor) - normalized by source term
        float xq = ground_air / Q;

        // Update maximum values
        if (center_air > max_center_air_conc[rad_idx]) {
            max_center_air_conc[rad_idx] = center_air;
            max_dir_center_air[rad_idx] = theta_idx + 1;
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

// ============================================================================
// Print Results Summary
// ============================================================================
// Prints a formatted table of maximum radionuclide dispersion values'''

if old_text in content:
    content = content.replace(old_text, new_text)
    with open('gpuff_plot.cuh', 'w', encoding='utf-8') as f:
        f.write(content)
    print("gpuff_plot.cuh: Added init_max_tracking and update_max_values")
else:
    print("gpuff_plot.cuh: Pattern not found")

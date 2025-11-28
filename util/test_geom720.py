"""
Test for build_geom720_normalized function
Verifies that 720 equal-mass sectors are constructed and validated correctly.

Two paths:
1) exact_cell_probabilities: exact probability mass per cell via CDF differences
2) midpoint_cell_masses: diagnostic-only midpoint rule pdf(center)*dV for inner cells
"""

import numpy as np
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.stats import norm

# ----------------------------
# Geometry construction
# ----------------------------

def inv_norm_cdf(p):
    return norm.ppf(p)

def build_radial_centers_6():
    # 6 equal-mass Rayleigh shells, sigma_y=1 reference
    r_centers = []
    for j in range(1, 7):
        p_c = (j - 0.5) / 6.0
        r_c = np.sqrt(-2.0 * np.log(max(1e-12, 1.0 - p_c)))
        r_centers.append(r_c)
    return r_centers

def build_vertical_centers_10():
    # 10 equal-mass normal layers, sigma_z=1 reference
    z_centers = []
    for i in range(1, 11):
        p_c = (i - 0.5) / 10.0
        z_centers.append(inv_norm_cdf(min(1.0 - 1e-12, max(1e-12, p_c))))
    return z_centers

def build_theta_12():
    return [2.0 * np.pi * j / 12.0 for j in range(12)]

def build_geom720_normalized():
    r6 = build_radial_centers_6()
    z10 = build_vertical_centers_10()
    th12 = build_theta_12()

    pts = []
    for r in r6:
        for z in z10:
            for th in th12:
                x = r * np.cos(th)
                y = r * np.sin(th)
                pts.append([x, y, z])
    return np.array(pts), r6, z10, th12

# ----------------------------
# Cell boundaries and exact masses
# ----------------------------

def radial_bounds_6():
    # true quantile bounds including infinities
    rb = [0.0]
    for i in range(1, 6):
        rb.append(np.sqrt(-2.0 * np.log(1.0 - i/6.0)))
    rb.append(np.inf)
    return rb

def vertical_bounds_10():
    zb = [-np.inf]
    for k in range(1, 10):
        zb.append(norm.ppf(k/10.0))
    zb.append(np.inf)
    return zb

def exact_cell_probabilities():
    """
    Exact probability of each of 6*10*12 cells:
    P = [F_R(r2)-F_R(r1)] * [Phi(z2)-Phi(z1)] * [dtheta/(2*pi)]
    where F_R(r) = 1 - exp(-r^2/2) under sigma-normalized coords.
    """
    rb = radial_bounds_6()
    zb = vertical_bounds_10()
    dtheta = 2*np.pi/12.0

    probs = []
    for i in range(6):
        r1, r2 = rb[i], rb[i+1]
        FR1 = 0.0 if r1 == 0.0 else 1.0 - np.exp(-0.5*r1*r1)
        FR2 = 1.0 if np.isinf(r2) else 1.0 - np.exp(-0.5*r2*r2)
        p_r = FR2 - FR1
        for k in range(10):
            z1, z2 = zb[k], zb[k+1]
            p_z = norm.cdf(z2) - norm.cdf(z1)
            p_theta = dtheta/(2*np.pi)
            for _ in range(12):
                probs.append(p_r * p_z * p_theta)
    return np.array(probs)

# ----------------------------
# Midpoint rule diagnostic
# ----------------------------

def gaussian_pdf_3d(x, y, z, sigma_y, sigma_z):
    return (1.0/((2*np.pi)**1.5 * sigma_y**2 * sigma_z)) * \
           np.exp(-0.5*((x*x + y*y)/sigma_y**2 + z*z/sigma_z**2))

def calculate_sector_pdf_at_centers(points, sigma_y, sigma_z):
    pdf = []
    for xn, yn, zn in points:
        x, y, z = xn*sigma_y, yn*sigma_y, zn*sigma_z
        pdf.append(gaussian_pdf_3d(x, y, z, sigma_y, sigma_z))
    return np.array(pdf)

def cell_volume_sigma_scaled(r1, r2, dtheta, z1, z2, sigma_y, sigma_z):
    # exact dV for finite bounds; caller ensures finiteness
    area = 0.5*(r2*r2 - r1*r1)*dtheta
    height = (z2 - z1)
    return area * height * (sigma_y**2) * sigma_z

def midpoint_cell_masses(points, sigma_y=2.0, sigma_z=1.5):
    """
    Diagnostic-only midpoint masses.
    내부 셀(유한 경계)에는 pdf(center)*dV 적용.
    끝 셀(무한 경계 포함)은 해석적 질량으로 대체하여 편향 제거.
    """
    rb = radial_bounds_6()
    zb = vertical_bounds_10()
    dtheta = 2*np.pi/12.0

    pdf = calculate_sector_pdf_at_centers(points, sigma_y, sigma_z)
    probs_exact = exact_cell_probabilities()

    masses = []
    idx = 0
    for i in range(6):
        r1, r2 = rb[i], rb[i+1]
        r_finite = np.isfinite(r1) and np.isfinite(r2)
        for k in range(10):
            z1, z2 = zb[k], zb[k+1]
            z_finite = np.isfinite(z1) and np.isfinite(z2)
            for _ in range(12):
                if r_finite and z_finite:
                    dV = cell_volume_sigma_scaled(r1, r2, dtheta, z1, z2, sigma_y, sigma_z)
                    masses.append(pdf[idx] * dV)
                else:
                    # tail cells use exact mass to avoid truncation bias
                    masses.append(probs_exact[idx])
                idx += 1
    return np.array(masses)

# ----------------------------
# Testing and reporting
# ----------------------------

def aggregates_from_vector(vec):
    shell  = np.add.reduceat(vec, np.arange(0, 6*10*12, 10*12))      # 6
    layer  = np.array([vec[i*12::10*12].sum() for i in range(10)])    # 10
    sector = np.array([vec[j::12].sum() for j in range(12)])          # 12
    return shell, layer, sector

def test_uniform_distribution():
    print("="*80)
    print("Testing build_geom720_normalized with exact masses for validation")
    print("="*80)

    points, r6, z10, th12 = build_geom720_normalized()
    print(f"Total points: {len(points)}, shape: {points.shape}")

    test_cases = [(1.0,1.0), (2.0,1.5), (5.0,3.0)]
    results = []

    for sy, sz in test_cases:
        print("\n" + "-"*60)
        print(f"sigma_y={sy}, sigma_z={sz}")
        masses_mid = midpoint_cell_masses(points, sy, sz)
        probs_exact = exact_cell_probabilities()

        # sums
        print(f"sum(midpoint) = {masses_mid.sum():.8f}  ~1.0")
        print(f"sum(exact)    = {probs_exact.sum():.8f}  =1.0")

        # CV across 720
        cv_mid  = masses_mid.std()/masses_mid.mean()
        cv_exact= probs_exact.std()/probs_exact.mean()
        print(f"Cell CV midpoint = {cv_mid:.6%}")
        print(f"Cell CV exact    = {cv_exact:.6%}")

        # aggregates
        shell_e, layer_e, sector_e = aggregates_from_vector(probs_exact)
        shell_cv = shell_e.std()/shell_e.mean()
        layer_cv = layer_e.std()/layer_e.mean()
        sector_cv= sector_e.std()/sector_e.mean()
        print(f"Shell CV(exact)  = {shell_cv:.6%}")
        print(f"Layer CV(exact)  = {layer_cv:.6%}")
        print(f"Sector CV(exact) = {sector_cv:.6%}")

        results.append({
            "sigma_y": sy, "sigma_z": sz,
            "cell_cv_midpoint": cv_mid,
            "cell_cv_exact": cv_exact,
            "shell_cv_exact": shell_cv,
            "layer_cv_exact": layer_cv,
            "sector_cv_exact": sector_cv,
            "sum_mid": masses_mid.sum(),
            "sum_exact": probs_exact.sum()
        })
    return points, results

# ----------------------------
# Visualization
# ----------------------------

def visualize_points(points):
    fig = plt.figure(figsize=(15,5))
    ax1 = fig.add_subplot(131, projection='3d')
    r = np.sqrt(points[:,0]**2 + points[:,1]**2)
    c = r + points[:,2]
    sct = ax1.scatter(points[:,0], points[:,1], points[:,2], c=c, cmap='viridis', s=6, alpha=0.7)
    ax1.set_xlabel('X (sigma_y)')
    ax1.set_ylabel('Y (sigma_y)')
    ax1.set_zlabel('Z (sigma_z)')
    ax1.set_title('720 points 6×10×12')
    plt.colorbar(sct, ax=ax1, label='r+z', shrink=0.6, pad=0.1)

    ax2 = fig.add_subplot(132)
    s2 = ax2.scatter(points[:,0], points[:,1], c=points[:,2], cmap='coolwarm', s=10, alpha=0.7)
    ax2.set_aspect('equal')
    ax2.set_title('XY colored by Z')
    plt.colorbar(s2, ax=ax2, label='Z')

    ax3 = fig.add_subplot(133)
    rz = np.sqrt(points[:,0]**2 + points[:,1]**2)
    ang = np.arctan2(points[:,1], points[:,0])
    s3 = ax3.scatter(rz, points[:,2], c=ang, cmap='hsv', s=10, alpha=0.7, vmin=-np.pi, vmax=np.pi)
    ax3.set_xlabel('r')
    ax3.set_ylabel('z')
    ax3.set_title('RZ colored by angle')
    plt.colorbar(s3, ax=ax3, label='angle')
    plt.tight_layout()
    plt.savefig('geom720_points_visualization.png', dpi=150, bbox_inches='tight')
    print("Saved: geom720_points_visualization.png")

def visualize_mass_distribution(points, sigma_y=2.0, sigma_z=1.5):
    """
    막대와 CV는 정확 질량으로 시각화.
    중점적분은 참고용으로 합계만 출력.
    """
    masses_mid  = midpoint_cell_masses(points, sigma_y, sigma_z)
    probs_exact = exact_cell_probabilities()

    shell_e, layer_e, sector_e = aggregates_from_vector(probs_exact)

    fig = plt.figure(figsize=(15,10))

    # 3D PDF at centers
    pdf_vals = calculate_sector_pdf_at_centers(points, sigma_y, sigma_z)
    ax1 = fig.add_subplot(221, projection='3d')
    sct = ax1.scatter(points[:,0]*sigma_y, points[:,1]*sigma_y, points[:,2]*sigma_z,
                      c=pdf_vals, cmap='hot', s=16, alpha=0.8)
    ax1.set_title(f'PDF at 720 centers  sigma_y={sigma_y}, sigma_z={sigma_z}')
    plt.colorbar(sct, ax=ax1, label='PDF', shrink=0.6)

    # shell bars from exact masses
    ax2 = fig.add_subplot(222)
    ax2.bar(range(6), shell_e, color='steelblue', alpha=0.75)
    ax2.set_title('Mass by radial shell  [exact]')
    ax2.set_xlabel('Shell index'); ax2.set_ylabel('Probability mass'); ax2.grid(True, alpha=0.3)

    # layer bars from exact masses
    ax3 = fig.add_subplot(223)
    ax3.bar(range(10), layer_e, color='darkgreen', alpha=0.75)
    ax3.set_title('Mass by vertical layer  [exact]')
    ax3.set_xlabel('Layer index'); ax3.set_ylabel('Probability mass'); ax3.grid(True, alpha=0.3)

    # sector polar from exact masses
    ax4 = fig.add_subplot(224, projection='polar')
    th12 = build_theta_12()
    sec_plot_th  = th12 + [th12[0]]
    sec_plot_val = list(sector_e) + [sector_e[0]]
    ax4.plot(sec_plot_th, sec_plot_val, 'o-', linewidth=2, color='darkred')
    ax4.fill(sec_plot_th, sec_plot_val, alpha=0.3, color='coral')
    cv_sector = sector_e.std()/sector_e.mean()
    ax4.set_title(f'Mass by angular sector [exact]  CV={cv_sector:.2%}', pad=20)
    ax4.grid(True)

    plt.tight_layout()
    fn = f'mass_distribution_exact_sy{sigma_y}_sz{sigma_z}.png'
    plt.savefig(fn, dpi=150, bbox_inches='tight')
    print(f"Saved: {fn}")
    print(f"Midpoint sum ~ {masses_mid.sum():.8f}, Exact sum = {probs_exact.sum():.8f}")

# ----------------------------
# Main
# ----------------------------

if __name__ == "__main__":
    points, results = test_uniform_distribution()

    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print("{:<8} {:<8} {:<16} {:<16} {:<14} {:<14} {:<14}".format(
        "sigma_y","sigma_z","CellCV(mid)","CellCV(exact)","ShellCV","LayerCV","SectorCV"))
    print("-"*96)
    for r in results:
        print("{:<8.1f} {:<8.1f} {:<16.6%} {:<16.6%} {:<14.6%} {:<14.6%} {:<14.6%}".format(
            r["sigma_y"], r["sigma_z"],
            r["cell_cv_midpoint"], r["cell_cv_exact"],
            r["shell_cv_exact"], r["layer_cv_exact"], r["sector_cv_exact"]))

    print("\nCreating visualizations")
    visualize_points(points)
    visualize_mass_distribution(points, sigma_y=2.0, sigma_z=1.5)
    print("Done")

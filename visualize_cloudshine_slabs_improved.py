"""
Improved 3D Slab Structure Visualization for Cloudshine
========================================================
Better visualization with distinct colors, proper transparency, and visible puff center.
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import matplotlib.pyplot as plt
from matplotlib import cm
import warnings
warnings.filterwarnings('ignore')

# Constants
MU_AIR = 0.01
K_BUILD = 1.4
PI = np.pi

class ImprovedSlabVisualizer:
    """Enhanced slab structure visualization."""

    def __init__(self, puff_z, sigma_y, sigma_z, h_mix=1000.0):
        self.puff_z = puff_z
        self.sigma_y = sigma_y
        self.sigma_z = sigma_z
        self.h_mix = h_mix
        self.slab_heights = self._generate_slab_heights()

    def _generate_slab_heights(self):
        """Generate 10 slab heights with ground and mixing layer reflection."""
        c = np.array([0.127, 0.385, 0.674, 1.037, 1.645])
        z_slabs = []

        for ci in c:
            z_slabs.append(self.puff_z + ci * self.sigma_z)
            z_slabs.append(self.puff_z - ci * self.sigma_z)

        z_slabs = np.array(z_slabs)
        z_slabs = np.abs(z_slabs)  # Ground reflection

        if self.h_mix > 0:
            z_slabs = np.where(z_slabs > self.h_mix,
                             2.0 * self.h_mix - z_slabs,
                             z_slabs)

        return np.sort(z_slabs)

    def create_improved_3d_visualization(self):
        """Create improved 3D visualization with better colors and transparency."""

        fig = go.Figure()

        # Grid for slabs
        extent = 3 * self.sigma_y
        grid_points = 30
        x = np.linspace(-extent, extent, grid_points)
        y = np.linspace(-extent, extent, grid_points)
        X, Y = np.meshgrid(x, y)

        # Color scheme - distinct colors for each slab
        # Use a gradient from bottom to top
        colors_rgb = [
            'rgba(25, 25, 112, 0.4)',   # Midnight Blue
            'rgba(65, 105, 225, 0.4)',  # Royal Blue
            'rgba(100, 149, 237, 0.4)',  # Cornflower Blue
            'rgba(135, 206, 250, 0.4)',  # Light Sky Blue
            'rgba(0, 191, 255, 0.4)',    # Deep Sky Blue
            'rgba(127, 255, 212, 0.4)',  # Aquamarine
            'rgba(144, 238, 144, 0.4)',  # Light Green
            'rgba(173, 255, 47, 0.4)',   # Green Yellow
            'rgba(255, 255, 0, 0.4)',    # Yellow
            'rgba(255, 165, 0, 0.4)',    # Orange
        ]

        # Add ground plane first (so it's behind everything)
        Z_ground = np.zeros_like(X)
        fig.add_trace(go.Surface(
            x=X, y=Y, z=Z_ground,
            colorscale=[[0, 'rgba(139, 139, 139, 0.2)'], [1, 'rgba(139, 139, 139, 0.2)']],
            showscale=False,
            name='Ground',
            hoverinfo='skip',
            lighting=dict(ambient=0.8, diffuse=0.2, specular=0.1, roughness=0.9)
        ))

        # Add slabs with improved visualization
        for i, z_s in enumerate(self.slab_heights):
            Z = np.ones_like(X) * z_s

            # Calculate Gaussian density for this slab
            density = np.exp(-0.5 * ((X/self.sigma_y)**2 + (Y/self.sigma_y)**2))

            # Create custom colorscale for this slab
            color = colors_rgb[i]
            colorscale = [[0, 'rgba(255,255,255,0)'], [0.3, color], [1, color]]

            fig.add_trace(go.Surface(
                x=X, y=Y, z=Z,
                surfacecolor=density,
                colorscale=colorscale,
                showscale=False,
                name=f'Slab {i+1}: {z_s:.0f}m',
                opacity=0.6,  # Increased opacity for better visibility
                hovertemplate=f'Slab {i+1}<br>Height: {z_s:.0f}m<br>Density: %{{surfacecolor:.3f}}<extra></extra>',
                lighting=dict(ambient=0.6, diffuse=0.4, specular=0.2, roughness=0.8),
                lightposition=dict(x=100, y=200, z=500)
            ))

        # Add wireframe edges for each slab (optional, for better definition)
        theta = np.linspace(0, 2*np.pi, 50)
        for i, z_s in enumerate(self.slab_heights):
            # Add circular boundary at 2 sigma
            circle_x = 2 * self.sigma_y * np.cos(theta)
            circle_y = 2 * self.sigma_y * np.sin(theta)
            circle_z = np.ones_like(theta) * z_s

            fig.add_trace(go.Scatter3d(
                x=circle_x, y=circle_y, z=circle_z,
                mode='lines',
                line=dict(color=colors_rgb[i].replace('0.4', '0.8'), width=2),
                showlegend=False,
                hoverinfo='skip'
            ))

        # Add puff center - make it more prominent
        fig.add_trace(go.Scatter3d(
            x=[0], y=[0], z=[self.puff_z],
            mode='markers+text',
            marker=dict(
                size=15,
                color='red',
                symbol='diamond',
                line=dict(color='darkred', width=2)
            ),
            text=['Puff Center'],
            textposition='top center',
            textfont=dict(size=14, color='red'),
            name='Puff Center',
            hovertemplate=f'Puff Center<br>Height: {self.puff_z:.0f}m<extra></extra>'
        ))

        # Add vertical line from ground to puff center
        fig.add_trace(go.Scatter3d(
            x=[0, 0], y=[0, 0], z=[0, self.puff_z],
            mode='lines',
            line=dict(color='red', width=3, dash='dash'),
            showlegend=False,
            hoverinfo='skip'
        ))

        # Add sigma boundaries on ground
        for n_sigma, style, width in [(1, 'solid', 3), (2, 'dash', 2)]:
            circle_x = n_sigma * self.sigma_y * np.cos(theta)
            circle_y = n_sigma * self.sigma_y * np.sin(theta)
            circle_z = np.zeros_like(theta)

            fig.add_trace(go.Scatter3d(
                x=circle_x, y=circle_y, z=circle_z,
                mode='lines',
                line=dict(color='red', width=width, dash=style),
                name=f'{n_sigma}σ_y boundary',
                showlegend=True,
                hoverinfo='skip'
            ))

        # Update layout for better visualization
        fig.update_layout(
            title=dict(
                text=f'<b>10-Slab Plane Source Structure</b><br>' +
                     f'<sub>σ_y={self.sigma_y:.0f}m, σ_z={self.sigma_z:.0f}m, Release Height={self.puff_z:.0f}m</sub>',
                x=0.5,
                xanchor='center',
                font=dict(size=18)
            ),
            scene=dict(
                xaxis=dict(
                    title='X Position (m)',
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='rgba(230, 230, 250, 0.1)'
                ),
                yaxis=dict(
                    title='Y Position (m)',
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='rgba(230, 230, 250, 0.1)'
                ),
                zaxis=dict(
                    title='Height (m)',
                    gridcolor='lightgray',
                    showbackground=True,
                    backgroundcolor='rgba(230, 230, 250, 0.1)',
                    range=[0, max(self.slab_heights) * 1.1]
                ),
                camera=dict(
                    eye=dict(x=2.0, y=2.0, z=1.0),
                    center=dict(x=0, y=0, z=0.2)
                ),
                aspectmode='manual',
                aspectratio=dict(x=1, y=1, z=0.6)
            ),
            showlegend=True,
            legend=dict(
                x=1.02,
                y=0.98,
                xanchor='left',
                yanchor='top',
                bgcolor='rgba(255, 255, 255, 0.9)',
                bordercolor='gray',
                borderwidth=1
            ),
            margin=dict(l=0, r=100, t=80, b=0),
            height=800,
            width=1200
        )

        return fig

    def create_side_by_side_visualization(self):
        """Create side-by-side 3D and profile view like the reference image."""

        # Create subplot with 3D and 2D views
        fig = make_subplots(
            rows=1, cols=2,
            column_widths=[0.6, 0.4],
            specs=[[{'type': 'scene'}, {'type': 'xy'}]],
            subplot_titles=('10-Slab Plane Source Structure',
                           'Vertical Distribution and Slab Positions')
        )

        # Left panel: 3D visualization (simplified for subplots)
        extent = 3 * self.sigma_y
        grid_points = 20
        x = np.linspace(-extent, extent, grid_points)
        y = np.linspace(-extent, extent, grid_points)
        X, Y = np.meshgrid(x, y)

        # Add ground
        Z_ground = np.zeros_like(X)
        fig.add_trace(
            go.Surface(
                x=X, y=Y, z=Z_ground,
                colorscale=[[0, 'gray'], [1, 'gray']],
                showscale=False,
                opacity=0.2,
                name='Ground'
            ),
            row=1, col=1
        )

        # Add slabs
        colors = plt.cm.viridis(np.linspace(0.2, 0.9, 10))
        for i, z_s in enumerate(self.slab_heights):
            Z = np.ones_like(X) * z_s
            density = np.exp(-0.5 * ((X/self.sigma_y)**2 + (Y/self.sigma_y)**2))

            color_hex = f'rgba({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)}, 0.4)'

            fig.add_trace(
                go.Surface(
                    x=X, y=Y, z=Z,
                    surfacecolor=density,
                    colorscale=[[0, 'white'], [1, color_hex]],
                    showscale=False,
                    opacity=0.5,
                    name=f'Slab {i+1}'
                ),
                row=1, col=1
            )

        # Add puff center marker
        fig.add_trace(
            go.Scatter3d(
                x=[0], y=[0], z=[self.puff_z],
                mode='markers',
                marker=dict(size=10, color='red', symbol='diamond'),
                name='Puff Center',
                showlegend=False
            ),
            row=1, col=1
        )

        # Right panel: Vertical profile
        z_profile = np.linspace(0, max(self.slab_heights) + 50, 200)
        gaussian = np.exp(-0.5 * ((z_profile - self.puff_z) / self.sigma_z)**2)

        # Add Gaussian profile
        fig.add_trace(
            go.Scatter(
                x=gaussian, y=z_profile,
                mode='lines',
                line=dict(color='blue', width=3),
                name='Gaussian Profile',
                fill='tozerox',
                fillcolor='rgba(0, 100, 255, 0.1)'
            ),
            row=1, col=2
        )

        # Add slab lines
        for i, z_s in enumerate(self.slab_heights):
            color_rgb = f'rgba({int(colors[i][0]*255)}, {int(colors[i][1]*255)}, {int(colors[i][2]*255)}, 0.7)'
            fig.add_trace(
                go.Scatter(
                    x=[0, 1], y=[z_s, z_s],
                    mode='lines+text',
                    line=dict(color=color_rgb, width=2, dash='dash'),
                    text=[None, f'Slab {i+1}'],
                    textposition='middle right',
                    showlegend=False
                ),
                row=1, col=2
            )

        # Add puff center line
        fig.add_trace(
            go.Scatter(
                x=[0, 1], y=[self.puff_z, self.puff_z],
                mode='lines',
                line=dict(color='red', width=3),
                name='Puff Center'
            ),
            row=1, col=2
        )

        # Update layout
        fig.update_layout(
            title=dict(
                text=f'<b>Plane Source Method: 10-Slab Structure</b><br>' +
                     f'<sub>σ_y={self.sigma_y:.0f}m, σ_z={self.sigma_z:.0f}m, Release Height={self.puff_z:.0f}m</sub>',
                x=0.5,
                xanchor='center'
            ),
            scene=dict(
                xaxis_title='X (m)',
                yaxis_title='Y (m)',
                zaxis_title='Height (m)',
                camera=dict(eye=dict(x=1.5, y=1.5, z=0.8))
            ),
            xaxis=dict(title='Normalized Concentration', range=[0, 1.1]),
            yaxis=dict(title='Height (m)', range=[0, max(self.slab_heights) + 50]),
            height=700,
            width=1400,
            showlegend=True
        )

        return fig

def main():
    """Generate improved slab visualizations."""

    print("="*60)
    print("Improved Slab Structure Visualization")
    print("="*60)

    scenarios = [
        {
            'name': 'medium_puff',
            'puff_z': 100,
            'sigma_y': 500,
            'sigma_z': 200,
            'h_mix': 1000
        },
        {
            'name': 'large_puff',
            'puff_z': 150,
            'sigma_y': 800,
            'sigma_z': 350,
            'h_mix': 1500
        },
        {
            'name': 'boundary_case',
            'puff_z': 80,
            'sigma_y': 450,
            'sigma_z': 380,
            'h_mix': 800
        }
    ]

    for scenario in scenarios:
        print(f"\nProcessing: {scenario['name']}")
        print(f"  σ_y={scenario['sigma_y']}m, σ_z={scenario['sigma_z']}m, z={scenario['puff_z']}m")

        visualizer = ImprovedSlabVisualizer(
            scenario['puff_z'],
            scenario['sigma_y'],
            scenario['sigma_z'],
            scenario['h_mix']
        )

        # Create improved 3D visualization
        fig_3d = visualizer.create_improved_3d_visualization()
        filename_3d = f"cloudshine_slabs_{scenario['name']}_improved.html"
        fig_3d.write_html(filename_3d)
        print(f"  [OK] Saved improved 3D visualization: {filename_3d}")

        # Create side-by-side visualization
        fig_side = visualizer.create_side_by_side_visualization()
        filename_side = f"cloudshine_slabs_{scenario['name']}_combined.html"
        fig_side.write_html(filename_side)
        print(f"  [OK] Saved combined view: {filename_side}")

        # Print slab information
        print(f"  Slab heights:")
        for i, z in enumerate(visualizer.slab_heights):
            print(f"    Slab {i+1:2d}: {z:6.1f}m")

    print("\n" + "="*60)
    print("Improved visualizations complete!")
    print("="*60)

if __name__ == "__main__":
    main()
import matplotlib.pyplot as plt
import numpy as np

# GPU models and their performance values (inverse of execution time)
gpu_models = ["GTX 1080", "Tesla P100", "RTX 3090", "A100"]
execution_times = [32.173, 28.571, 9.602, 4.662]
performance = [1 / time for time in execution_times]

# Reversing the order for plotting
gpu_models_reversed = gpu_models[::-1]
performance_reversed = performance[::-1]

# Base color for orange and calculating shades based on performance
base_color = np.array([1.0, 0.6, 0.0])  # RGB for orange
performance_reversed_array = np.array(performance_reversed)
normalized_performance = (performance_reversed_array - min(performance_reversed_array)) / (max(performance_reversed_array) - min(performance_reversed_array))
shades_of_orange = [base_color * (0.6 + 0.4 * norm) for norm in normalized_performance]

# Function to add gradient bars
def draw_gradient_barh(ax, y, width, height, color):
    for alpha in np.linspace(0.3, 1, 100):  # Creating a gradient effect using alpha values
        ax.barh(y, width, height=height, color=color, alpha=alpha)

# Creating the plot
fig, ax = plt.subplots(figsize=(14, 4))
for i, (gpu, perf, color) in enumerate(zip(gpu_models_reversed, performance_reversed, shades_of_orange)):
    draw_gradient_barh(ax, i, perf, 0.5, color)  # Reduced height for narrower spacing
    ax.text(perf * 0.95, i, f"{perf:.3f}", va='center', ha='right', fontsize=14, color='black')  # Adjusted text position

# Customizing the axes and labels for a cleaner look
ax.set_yticks(range(len(gpu_models_reversed)))
ax.set_yticklabels(gpu_models_reversed, fontsize=17)
ax.set_xlabel("Performance (1/s)", fontsize=12, fontweight='bold')
# ax.set_ylabel("GPU Models", fontsize=12, fontweight='bold')
ax.grid(visible=True, axis='x', linestyle='--', alpha=0.4)
ax.set_axisbelow(True)

plt.tight_layout()
plt.show()

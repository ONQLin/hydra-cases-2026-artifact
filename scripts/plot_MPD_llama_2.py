import pandas as pd
import matplotlib.pyplot as plt

# Enlarge font sizes for all plot elements
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

# Define the name of your CSV file.
file_name = "results_summary_LLAMA3-ARXIV.csv"

# Read the data from the CSV file into a pandas DataFrame.
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
    exit()

# --- New Logic for Dual-Variable Mapping ---

# 1. Get unique values for color (batchsize) and marker (NoI_bw).
unique_bs = sorted(df['batchsize'].unique())
unique_bw = sorted(df['NoI_bw(GBps)'].unique())

# 2. Define colors and markers
# Use a colorblind-friendly palette for batchsize
bs_colors = ['red', 'blue', 'green', 'orange', 'purple', 'brown', 'pink', 'gray']
# Use distinct markers for NoI_bw
bw_markers = ['o', 's', 'D', '^', 'v', 'X']

# 3. Create mapping dictionaries
bs_color_map = {bs: bs_colors[i % len(bs_colors)] for i, bs in enumerate(unique_bs)}
bw_marker_map = {bw: bw_markers[i % len(bw_markers)] for i, bw in enumerate(unique_bw)}

# Create a figure and axes for the plot.
plt.figure(figsize=(10, 7))
ax = plt.gca() # Get the current axes object

# 4. Plot all points by iterating through the DataFrame
for _, row in df.iterrows():
    # Get the style properties from the mappings
    color = bs_color_map[row['batchsize']]
    marker = bw_marker_map[row['NoI_bw(GBps)']]

    # Plot the point
    ax.scatter(
        row['TTFT(s)'],
        row['tokens_per_sec'],
        marker=marker,
        color=color,
        alpha=0.8,
        # s=100, # Increased size for better visibility
        # No label here, as we'll create custom legends
    )

# --- Legend Creation for Dual Mapping ---

# 5. Create a dummy legend for Batch Size (Color)
color_handles = []
color_labels = []
for bs in unique_bs:
    # Plot a placeholder for the legend
    handle = ax.scatter([], [], marker='o', color=bs_color_map[bs])
    color_handles.append(handle)
    color_labels.append(f'Batch Size: {bs}')

# Place the first legend
legend1 = ax.legend(color_handles, color_labels, title="Batch Size (Color)", loc='upper right', frameon=True)

# 6. Create a dummy legend for NoI_bw (Marker)
marker_handles = []
marker_labels = []
for bw in unique_bw:
    # Plot a placeholder for the legend (using black for consistent handle color)
    handle = ax.scatter([], [], marker=bw_marker_map[bw], color='black')
    marker_handles.append(handle)
    marker_labels.append(f'NoI BW: {bw}')

# Add the second legend, making sure to re-add the first one as an artist
legend2 = ax.legend(marker_handles, marker_labels, title="NoI BW (Marker)", loc='lower right', frameon=True)
ax.add_artist(legend1) # Re-add the first legend

# Add title and axis labels to the plot for clarity.
plt.title("Throughput vs. Time To First Tokens (TTFT)")
plt.xlabel("Time To First Tokens (s)", fontweight='bold')
plt.ylabel("Throughput (tokens per second)", fontweight='bold')

# Add a grid for better readability.
plt.grid(True)

# Save the plot to an image file.
plt.savefig(f'tokens_per_sec_vs_ttft_scatter_batchsize_bw_{file_name.split(".")[0].split("_")[-1]}.png')
plt.close() # Close the figure to free memory
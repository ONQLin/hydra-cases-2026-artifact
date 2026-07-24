import pandas as pd
import matplotlib.pyplot as plt
import json

# Enlarge font sizes for all plot elements
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 16,
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

# Define the name of your CSV file.
# file_name = "results_summary_LLAMA3-ARXIV.csv"
file_name = "results_summary_LLAMA3-BWB.csv"

# Read the data from the CSV file into a pandas DataFrame.
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
    exit()

# Load filtered JSON
filtered_path = "Fast_Estimate/filtered_configs/throughput_bounds_llama3-8b_bwb.json"
# filtered_path = "Fast_Estimate/filtered_configs/throughput_bounds_llama3-8b_arxiv4k.json"
with open(filtered_path, "r") as f:
    filtered_results = json.load(f)
filtered_df = pd.DataFrame(filtered_results)
# Rename JSON columns to match CSV naming
filtered_df = filtered_df.rename(columns={
    "bw": "NoI_bw(GBps)",
    "M": "Num M",
    "P": "Num Ap",
    "D": "Num Ad"
})
# Merge on (bw, M, P, D)
merged = pd.merge(
    df, filtered_df,
    on=["NoI_bw(GBps)", "Num M", "Num Ap", "Num Ad"],
    how="inner"
)

# Get the unique values from the 'NoI_bw(GBps)' column to use for grouping.
unique_noi_bw = sorted(df['NoI_bw(GBps)'].unique())

# Define a list of markers to cycle through for each group.
colors = ['blue', 'green', 'orange', 'brown', 'purple']
markers = ['o', 's', 'D', '^', 'v']

# Create a figure and axes for the plot.
plt.figure(figsize=(10, 7))

# Loop through each unique 'NoI_bw(GBps)' value and plot the corresponding data points.
for i, bw_value in enumerate(unique_noi_bw):
    # Select the data for the current 'NoI_bw(GBps)' value.
    subset_df = df[df['NoI_bw(GBps)'] == bw_value]

    # Plot all the points in the group with a grey color. before filtering
    plt.scatter(
        subset_df['TTFT(s)'],
        subset_df['tokens_per_sec'],
        marker=markers[i % len(markers)],
        color='grey',
        label=f'NoI_bw(GBps): {bw_value}',
        alpha=0.6,
        # edgecolor=colors[i % len(colors)],
    )
    
    # TODO:Plot the filtered points in the group with the assigned color
    # Filtered configs (real performance)
    filtered_subset = merged[merged['NoI_bw(GBps)'] == bw_value]
    plt.scatter(
        filtered_subset['TTFT(s)'],
        filtered_subset['tokens_per_sec'],
        marker=markers[i % len(markers)],
        color=colors[i % len(colors)],
        #s=90,
    )
    
# Add title and axis labels to the plot for clarity.
plt.title("Tokens per Second vs. TTFT(s) in the DSE experiments")
plt.xlabel("Time To First Tokens(s)", fontweight='bold')
plt.ylabel("Throughput (tokens per second)", fontweight='bold')

# Set the x-axis limits to be (0, 10).
# plt.xlim(3, 15)

# Display a legend.
# Note: The legend will show only one entry per group, as the highlighted points are not added to the legend.
plt.legend()

# Add a grid for better readability.
plt.grid(True)

# Save the plot to an image file.
plt.savefig(f'tokens_per_sec_vs_ttft_scatter_group_highlight2filter_{file_name.split(".")[0].split("_")[-1]}.png')
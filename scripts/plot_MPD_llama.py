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
file_name = "results_summary_LLAMA3-BWB.csv"

# Read the data from the CSV file into a pandas DataFrame.
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found.")
    exit()

# Get the unique values from the 'NoI_bw(GBps)' column to use for grouping.
unique_noi_bw = sorted(df['NoI_bw(GBps)'].unique())

# Define a list of markers to cycle through for each group.
markers = ['o', 's', 'D', '^', 'v']

# Create a figure and axes for the plot.
plt.figure(figsize=(10, 7))

# Loop through each unique 'NoI_bw(GBps)' value and plot the corresponding data points.
for i, bw_value in enumerate(unique_noi_bw):
    # Select the data for the current 'NoI_bw(GBps)' value.
    subset_df = df[df['NoI_bw(GBps)'] == bw_value]

    # Find the row with the highest 'tokens_per_sec' within this specific group.
    highest_tokens_row = subset_df.loc[subset_df['tokens_per_sec'].idxmax()]
    print("BW:", bw_value, "Highest Tokens per sec:", highest_tokens_row['tokens_per_sec'])
    print("  Configuration - Num M:", highest_tokens_row['Num M'], 
          "Num Ap:", highest_tokens_row['Num Ap'], 
          "Num Ad:", highest_tokens_row['Num Ad'])
    # Plot all the points in the group with a grey color.
    plt.scatter(
        subset_df['TTFT(s)'],
        subset_df['tokens_per_sec'],
        marker=markers[i % len(markers)],
        color='grey', # All points are grey by default
        label=f'NoI_bw(GBps): {bw_value}'
    )

    # Highlight the highest point of the group in red.
    plt.scatter(
        highest_tokens_row['TTFT(s)'],
        highest_tokens_row['tokens_per_sec'],
        marker=markers[i % len(markers)], # Use the group's default shape
        color='red',                       # Change color to red
        zorder=5                           # Ensure it's plotted on top of other points
    )

# Add title and axis labels to the plot for clarity.
plt.title("Tokens per Second vs. TTFT(s) in the DSE experiments")
plt.xlabel("Time To First Tokens(s)", fontweight='bold')
plt.ylabel("Throughput (tokens per second)", fontweight='bold')

# Set the x-axis limits to be (0, 10).
plt.xlim(3, 15)

# Display a legend.
# Note: The legend will show only one entry per group, as the highlighted points are not added to the legend.
plt.legend()

# Add a grid for better readability.
plt.grid(True)

# Save the plot to an image file.
plt.savefig('tokens_per_sec_vs_ttft_scatter_group_highlight.png')

print("Revised scatter plot with group highlights saved as 'tokens_per_sec_vs_ttft_scatter_group_highlight.png'")
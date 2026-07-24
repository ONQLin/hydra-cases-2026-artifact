import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import os
import matplotlib.lines as mlines

# --- Pareto Frontier Function ---
def find_pareto_frontier(df: pd.DataFrame, minimize_cols: list, maximize_cols: list, prj_name:str = 'llamaArxiv_RR') -> pd.DataFrame:
    """
    Identifies the Pareto frontier points in a DataFrame and marks them with a boolean column.
    
    Objectives: Minimize 'TTFT(s)' and Maximize 'tokens_per_sec'
    """
    df = df.drop_duplicates().reset_index(drop=True)
    is_pareto = np.ones(len(df), dtype=bool)

    # Iterate through each point (P1)
    for i in range(len(df)):
        P1 = df.iloc[i]
        
        # Check against every other point (P2)
        for j in range(len(df)):
            if i == j:
                continue
            
            P2 = df.iloc[j]
            
            # Check for better or equal in ALL objectives (dominance criteria)
            dominates = True
            
            for col in minimize_cols:
                if P2[col] > P1[col]: dominates = False; break
            if not dominates: continue

            for col in maximize_cols:
                if P2[col] < P1[col]: dominates = False; break
            if not dominates: continue

            # Check for STRICT dominance (strictly better in at least one objective)
            strictly_better = False
            for col in minimize_cols:
                if P2[col] < P1[col]: strictly_better = True; break
            
            if not strictly_better:
                for col in maximize_cols:
                    if P2[col] > P1[col]: strictly_better = True; break
                        
            # If P2 strictly dominates P1, then P1 is NOT on the Pareto frontier
            if strictly_better:
                is_pareto[i] = False
                break 
                
    # Add the is_pareto column to the original DataFrame
    df['is_pareto'] = is_pareto
    # save this configurations into a csv only keep pareto points
    pareto_df = df[df['is_pareto']]
    #drop the pareto_df if there is no value in tokens_per_sec or TTFT(s)
    pareto_df = pareto_df.dropna(subset=['tokens_per_sec', 'TTFT(s)'])
    # get the Num Ap: Num Ad average ratio for all rows
    pareto_df['Ap_Ad_ratio'] = pareto_df['Num Ap'] / pareto_df['Num Ad'].replace(0, np.nan)
    # get average by all rows
    avg_ratio = pareto_df['Ap_Ad_ratio'].mean()
    print(f"Average Ap/Ad ratio for Pareto points: {avg_ratio:.2f}")
    pareto_df.to_csv(f"pareto_frontier_results_{prj_name}.csv", index=False)
    return df

# --- Plotting Script ---

# Enlarge font sizes for all plot elements
plt.rcParams.update({
    'font.size': 18,
    'axes.labelsize': 16,
    'legend.fontsize': 18, # Smaller legend font for multi-column legend
    'xtick.labelsize': 16,
    'ytick.labelsize': 16,
})

# Define the name of your CSV file.
placer = "bw"
file_name = f"results_summary_LLAMA3-ARXIV_{placer}.csv"
prj_name = f"llamaArxiv_{placer.upper()}"

# Read the data from the CSV file into a pandas DataFrame.
try:
    df = pd.read_csv(file_name)
except FileNotFoundError:
    print(f"Error: The file '{file_name}' was not found. Please ensure the data file exists.")
    # Exit or provide placeholder logic if necessary, but here we assume the file will be generated.
    exit()

# 1. Calculate the Pareto Frontier and mark points
# Objectives: Minimize 'TTFT(s)' and Maximize 'tokens_per_sec'
df = find_pareto_frontier(
    df,
    minimize_cols=['TTFT(s)'],
    maximize_cols=['tokens_per_sec'],
    prj_name=prj_name
)
pareto_df = df[df['is_pareto']].copy()
non_pareto_df = df[~df['is_pareto']].copy()

# --- Dual-Variable Mapping Setup for Pareto Points ---

# 2. Get unique values for color (batchsize) and marker (NoI_bw) from Pareto points
unique_bs = sorted(pareto_df['batchsize'].unique())
unique_bw = sorted(pareto_df['NoI_bw(GBps)'].unique())
unique_bw.remove(128)  # remove 128 if exists

# 3. Define colors and markers
bs_colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b']
bw_markers = ['o', 's', 'D', '^', 'v']
bw_colors = ['blue', 'green', 'orange', 'brown', 'purple']

# 4. Create mapping dictionaries
bs_color_map = {bs: bs_colors[i % len(bs_colors)] for i, bs in enumerate(unique_bs)}
bw_marker_map = {bw: bw_markers[i % len(bw_markers)] for i, bw in enumerate(unique_bw)}
bw_color_map = {bw: bw_colors[i % len(bw_colors)] for i, bw in enumerate(unique_bw)}

# Create a figure and axes for the plot.
plt.figure(figsize=(10, 8)) # Increased height for the bottom legend
ax = plt.gca()

# 5. Plot ALL points
# --- A. Plot NON-PARETO points in GREY ---
ax.scatter(
    non_pareto_df['TTFT(s)'],
    non_pareto_df['tokens_per_sec'],
    marker='x',
    color='grey',
    alpha=0.4,
    s=50,
    zorder=1 # Plot below the Pareto points
)

# --- B. Plot the Pareto Curve (Line) ---
# Sort the Pareto points by TTFT(s) to ensure the line is drawn correctly
pareto_df_sorted = pareto_df.sort_values(by='TTFT(s)')
ax.plot(
    pareto_df_sorted['TTFT(s)'],
    pareto_df_sorted['tokens_per_sec'],
    color='red',
    linestyle='-',
    linewidth=2,
    alpha=0.7,
    zorder=2 # Plot below points but above non-pareto points
)

# --- C. Plot PARETO points with custom colors/markers ---
for _, row in pareto_df.iterrows():
    #color = bs_color_map[row['batchsize']]
    if row['NoI_bw(GBps)'] not in bw_color_map:
        continue
    color = bw_color_map[row['NoI_bw(GBps)']]
    marker = bw_marker_map[row['NoI_bw(GBps)']]

    ax.scatter(
        row['TTFT(s)'],
        row['tokens_per_sec'],
        marker=marker,
        color=color,
        alpha=0.7,
        # s=150, # Increased size for better visibility
        zorder=3
    )

# --- Legend Creation ---

all_handles = []
all_labels = []

# 6. Add main plot elements to the legend
# Non-Pareto
all_handles.append(mlines.Line2D([], [], color='grey', marker='x', linestyle='None', markersize=8, label='Non-Pareto Points'))
all_labels.append('Non-Pareto Points')

# Pareto Curve
all_handles.append(mlines.Line2D([], [], color='red', linewidth=2, linestyle='-', label='Pareto Frontier Curve'))
all_labels.append('Pareto Frontier Curve')

# # 7. Add Batch Size (Color) legend entries
# for bs in unique_bs:
#     handle = mlines.Line2D([], [], color=bs_color_map[bs], marker='o', linestyle='None', markersize=8, label=f'Batch Size: {bs}')
#     all_handles.append(handle)
#     all_labels.append(f'Batch Size: {bs}')

# # 8. Add NoI_bw (Marker) legend entries
# for bw in unique_bw:
#     handle = mlines.Line2D([], [], color='black', marker=bw_marker_map[bw], linestyle='None', markersize=8, label=f'NoI BW: {bw} GB/s')
#     all_handles.append(handle)
#     all_labels.append(f'NoI BW: {bw} GB/s')

# Place the comprehensive legend below the plot
#final_legend = ax.legend(all_handles, all_labels, title="Legend", loc='lower center', ncol=3, frameon=True, bbox_to_anchor=(0.5, -0.25))
final_legend = ax.legend(all_handles, all_labels, loc='lower center', ncol=3, frameon=True)

# Adjust plot layout to make room for the large legend
plt.subplots_adjust(bottom=0.3)

# Add title and axis labels to the plot for clarity.
plt.title("Pareto Frontier & All Points: Throughput vs. TTFT")
plt.xlabel("Time To First Tokens (s)", fontweight='bold')
plt.ylabel("Throughput (tokens per second)", fontweight='bold')

# Add a grid for better readability.
plt.grid(True)
plt.tight_layout()
# Save the plot to an image file.
output_filename = f'pareto_all_points_tokens_per_sec_vs_ttft_batchsize_bw_{file_name.split(".")[0].split("_")[-1]}.png'
plt.savefig(output_filename)
plt.close()
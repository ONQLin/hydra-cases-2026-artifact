import sys
import os
from math import sqrt
import math
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker # Imported for Origin-style minor ticks

parent_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(parent_dir)
import json
from Sim.config.sys_config import ChipletsConfig
from Sim.config.utils import used_bws_dict
import Sim.config.utils as utils
from Sim.entities.chips_network import chip_graph
import Sim.common as common
from Sim.entities.mem_sys import mem_sys
from Sim.entities.comp_sys import comp_sys


def generate_area_breakdown():
    
    BWs = [128, 256, 384, 512, 640]
    chiplet_area = 144  # mm^2
    breakdowns = {}
    for bw in BWs:
        # based on D2D Bw, confirm the comp chiplet configs
        chiplet_config = ChipletsConfig(D2D_NoI_bw=bw, area_limit=chiplet_area, Fixed_chiplet_area=True)
        breakdowns[bw] = chiplet_config.chips_lib.area_breakdown
    
    BWs = list(map(int, breakdowns.keys()))
    components = list(next(iter(breakdowns.values()))["marca_p"].keys())
    
    # Prepare data arrays
    data = {comp: [breakdowns[bw]["marca_p"][comp] for bw in BWs] for comp in components}
    
    # --- PLOT CONFIGURATION ---
    
    # 1 & 2. Origin Style & Enlarged Fonts
    plt.rcParams.update({
        'font.size': 24,             # Overall base font size increased
        'axes.labelsize': 30,        # Axis labels
        'xtick.labelsize': 28,       # X-axis ticks
        'ytick.labelsize': 28,       # Y-axis ticks
        'legend.title_fontsize': 24, # Legend title
        'legend.fontsize': 22,       # Legend items
        'axes.linewidth': 2.0,       # Thicker, Origin-like plot outline
        'xtick.major.width': 2.0,    # Thicker ticks
        'ytick.major.width': 2.0,
        'xtick.minor.width': 1.5,
        'ytick.minor.width': 1.5,
        'xtick.major.size': 8,       # Longer ticks
        'ytick.major.size': 8,
        'xtick.minor.size': 4,
        'ytick.minor.size': 4,
        'xtick.direction': 'in',     # Origin-style inward facing ticks
        'ytick.direction': 'in',
        'xtick.top': True,           # Ticks on top border
        'ytick.right': True          # Ticks on right border
    })
    
    # 4. Define Textures (Hatches) and Colors
    # Matplotlib hatches: '/', '\', '|', '-', '+', 'x', 'o', 'O', '.', '*'
    hatches = ['//', 'oo', 'xx', '\\\\', '--', '||', '..', '**']
    colors = plt.cm.Set3.colors  # Using a nice qualitative colormap
    
    fig, ax = plt.subplots(figsize=(12, 8)) # Slightly taller to accommodate bottom legend
    
    x = np.arange(len(BWs)) * 1.2  # Adds horizontal gap between bars
    bottom = np.zeros(len(BWs))
    
    # Reorder components so 'links' is at the top of the stack (or as requested)
    components = [c for c in components if c != "links"] + ["links"]
    
    for i, comp in enumerate(components):
        label_name = comp if comp != 'links' else 'PHY+Bump'
        
        # Plot with colors, hatches, and black edges (edges are required for hatches to show clearly)
        ax.bar(x, data[comp], bottom=bottom, label=label_name, 
               color=colors[i % len(colors)],
               edgecolor='black', 
               linewidth=1.5,
               hatch=hatches[i % len(hatches)],
               width=0.8)
        
        bottom += np.array(data[comp])
    
    # Labeling and aesthetics
    ax.set_xlabel("D2D Bandwidth (GB/s)", fontweight='bold')
    ax.set_ylabel("Chiplet Area (mm²)", fontweight='bold')
    
    ax.set_xticks(x)
    ax.set_xticklabels(BWs)
    
    # Origin-style minor ticks
    ax.yaxis.set_minor_locator(ticker.AutoMinorLocator())
    ax.tick_params(which='both', top=True, right=True)
    
    # 3. Legend at the Bottom
    # bbox_to_anchor pulls it below the x-axis, ncol spreads it out horizontally
    ax.legend(title="Component", 
              loc='upper center', 
              bbox_to_anchor=(0.5, -0.15), 
              ncol=min(len(components), 5), # Automatically adjust columns based on items
              frameon=False)                # Origin legends usually lack heavy borders
              
    # Grid configuration (Optional, left it similar to your original but pushed to background)
    ax.grid(True, axis='y', linestyle='--', alpha=0.5, zorder=0)
    ax.set_axisbelow(True) # Ensure grid lines stay behind the bars
    ax.set_title("Marca_p", fontweight='bold', fontsize=30, pad=20)
    # tight_layout might cut off external legends, bbox_inches='tight' fixes this during save
    plt.savefig("area_breakdown_vs_bw_marca_p.png", bbox_inches='tight', dpi=300)
    plt.close()

if __name__ == "__main__":
    generate_area_breakdown()
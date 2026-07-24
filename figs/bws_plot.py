import json
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.colors import Normalize
from matplotlib.cm import ScalarMappable
import sys

def plot_chiplet_network(json_file, output_file='figs/plcmt/chiplet_network.png', figsize=(14, 8), dpi=150):
    """
    Plot chiplet network with link utilization heatmap.
    
    Args:
        json_file: Path to chipletBW_utilization.json
        output_file: Output PNG filename
        figsize: Figure size tuple (width, height)
        dpi: DPI for output image
    """
    
    # Load JSON data
    with open(json_file, 'r') as f:
        data = json.load(f)
    
    nodes_data = data['nodes']
    edges_data = data['edges']
    
    # Create node position map
    node_positions = {}
    node_labels = {}
    for node in nodes_data:
        node_id = node['id']
        node_positions[tuple(node['coord'])] = node_id
        node_labels[node_id] = node['label']
    
    # Extract edge utilization and create deduplicated edge list
    # Keep only one direction of each edge (u->v where u < v lexicographically)
    edge_dict = {}
    edge_utils = {}
    
    for edge in edges_data:
        u = tuple(edge['u'])
        v = tuple(edge['v'])
        util = edge['avg_utilization']  # Convert to percentage
        
        # Normalize edge (smaller coordinate first)
        edge_key = tuple([u, v])
        
        # Average if edge exists in both directions, otherwise just store
        if edge_key not in edge_dict:
            edge_dict[edge_key] = [u, v]
            edge_utils[edge_key] = util * 150
        else:
            # Average utilization if bidirectional
            raise ValueError("Edge already exists, should not happen in this logic.")
            edge_utils[edge_key] = (edge_utils[edge_key] + util)/2
    
    # Create figure
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    
    # Normalize utilization for colormap
    util_values = list(edge_utils.values())
    # norm = Normalize(vmin=min(util_values), vmax=max(util_values))
    norm = Normalize(vmin=0, vmax=80)
    cmap = plt.cm.RdYlGn_r  # Red-Yellow-Green reversed: green -> red
    sm = ScalarMappable(cmap=cmap, norm=norm)
    sm.set_array([])
    
    # Draw edges (links)
    for (u, v), util in edge_utils.items():
        color = cmap(norm(util))
        # Line width proportional to utilization (wider links)
        # linewidth = 2 + 8 * norm(util)
        linewidth = 26
        ax.plot([u[0], v[0]], [u[1], v[1]], color=color, linewidth=linewidth, zorder=1, solid_capstyle='round')
    
    # Draw nodes
    node_size = 0.7  # Size of node rectangles
    
    for coord, node_id in node_positions.items():
        label = node_labels[node_id]
        x, y = coord
        
        # Draw white rectangle with black border
        rect = patches.Rectangle(
            (x - node_size/2, y - node_size/2),
            node_size, node_size,
            linewidth=2,
            edgecolor='black',
            facecolor='white',
            zorder=2
        )
        ax.add_patch(rect)
        
        # Add label text in the center
        ax.text(x, y, label, ha='center', va='center', fontsize=32, fontweight='bold', zorder=3)
    
    # Set axis properties
    all_coords = [tuple(node['coord']) for node in nodes_data]
    xs = [c[0] for c in all_coords]
    ys = [c[1] for c in all_coords]
    
    margin = 0.5
    ax.set_xlim(min(xs) - margin, max(xs) + margin)
    ax.set_ylim(min(ys) - margin, max(ys) + margin)
    ax.set_aspect('equal')
    ax.invert_yaxis()  # Invert Y for top-left origin like typical grid
    
    # Hide X and Y axes
    ax.set_xticks([])
    ax.set_yticks([])
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    
    # ax.set_title('Chiplet Network Link Utilization Heatmap', fontsize=16, fontweight='bold', pad=20)
    
    # Add grid for clarity
    ax.grid(True, alpha=0.2, linestyle='--')
    
    # Add colorbar
    cbar = plt.colorbar(sm, ax=ax, label='Link Utilization (%)', pad=0.02, orientation='horizontal', fraction=0.03)
    cbar.ax.tick_params(labelsize=24)
    cbar.set_label('Link Utilization (%)', fontsize=24, labelpad=10, loc='left')
    cbar.set_ticks([0, 40, 80])
    cbar.set_ticklabels(['0', '40', '80'])
    
    # Set background
    ax.set_facecolor('white')
    
    plt.tight_layout()
    plt.savefig(output_file, dpi=dpi, bbox_inches='tight', facecolor='white')
    print(f"Figure saved to {output_file}")
    plt.close()

if __name__ == '__main__':
    mode = 'bw'  # 'rr' or 'bw'
    if len(sys.argv) > 1:
        json_file = sys.argv[1]
    else:
        # Default path
        if mode == 'rr':
            # json_file = 'simulator_output/2026-01-26_16-14-39-061592+/chipletBW_utilization.json'
            json_file = 'simulator_output/2026-01-26_13-33-12-rrplcmt/chipletBW_utilization.json'
        else:
            # json_file = 'simulator_output/2026-01-26_16-14-28-980582+/chipletBW_utilization.json'
            json_file = 'simulator_output/2026-01-26_13-29-12-bwplcmt/chipletBW_utilization.json'
    if mode == 'rr':
        plot_chiplet_network(json_file=json_file, output_file='figs/plcmt/chiplet_network_rr.png') 
    else:
        plot_chiplet_network(json_file=json_file, output_file='figs/plcmt/chiplet_network_bw.png')
from dataclasses import dataclass, field
import math
from typing import Dict, List, Tuple
import os
import json
import numpy as np

from Sim.placer.BasePlacer import BasePlacer
from Sim.config.model_config import BaseModelConfig
from Sim.entities.chips_network import chip_graph
import Sim.common as common
import Sim.config.utils as utils
import random

from Sim.logger import init_logger, _setup_logger
_setup_logger()   # root logger handles console + file
logger = init_logger(__name__)  # child logger inherits handlers

Coord = Tuple[int, int]


class RR_Placer(BasePlacer):

    def __init__(self, chiplet_num: int, chiplet_alloc: Dict[str, int]):
        super().__init__(chiplet_num, chiplet_alloc)
    
    def init_mems(self, chip_graph: chip_graph, model_config:BaseModelConfig , mem_size:int = 16): # 16GB
        """
        Initialize the placement of memory chiplets in the system using bandwidth-aware strategy.
        In this method, we place HBM3 chiplets on the outer ring of the interposer to maximize bandwidth.
        """
        if chip_graph.num_nodes != self.chiplet_num:
            raise ValueError("Input graph nodes do not match the chiplet number.")
        for chiplet_type, count in self.chiplet_alloc.items():
            if count < 0:
                raise ValueError(f"Chiplet allocation for {chiplet_type} cannot be negative.")
            if chiplet_type not in utils.avail_chiplets:
                raise ValueError(f"Chiplet type {chiplet_type} not found in the graph.")
        
        height, width = chip_graph.intp_height, chip_graph.intp_width
        if height * width < self.chiplet_num:
            raise ValueError("interposer needs to follow a rectangular shape.")
        
        outer_ring_nodes = []
        
        # round robin distributed in fan outward order for HBMs
        # Step 1 & 2: left and right columns in middle-out order, alternating
        for j in RR_Placer.middle_out_order(height):
            outer_ring_nodes.append(chip_graph.graph.nodes[(0, j)]['id'])           # left
            outer_ring_nodes.append(chip_graph.graph.nodes[(width - 1, j)]['id'])  # right

        # Step 3 & 4: top and bottom rows in middle-out order, alternating
        for i in RR_Placer.middle_out_order(width):
            outer_ring_nodes.append(chip_graph.graph.nodes[(i, 0)]['id'])           # top
            outer_ring_nodes.append(chip_graph.graph.nodes[(i, height - 1)]['id'])  # bottom
            
        # Step 5: add corners in alternating order
        corners = [
            (0, 0),                # top-left
            (width - 1, height - 1), # bottom-right
            (width - 1, 0),        # top-right
            (0, height - 1)        # bottom-left
        ]
        for coord in corners:
            outer_ring_nodes.append(chip_graph.graph.nodes[coord]['id'])
            
        # Ensure enough capacity
        if len(outer_ring_nodes) < self.chiplet_alloc["HBM3"]:
            raise ValueError("Not enough outer ring nodes for HBM3 chiplets.")
        
        for i in range(self.chiplet_alloc["HBM3"]):
            node_id = outer_ring_nodes[i]
            coord = chip_graph.id_to_coord[node_id]
            chip_graph.graph.nodes[coord]['chiplet_type'] = utils.chiplet_types_dict["HBM3"]
            chip_graph.graph.nodes[coord]['dram(g)'] = mem_size # 16G HBM3 chiplet
            chip_graph.graph.nodes[coord]['sram(g)'] = 0 # for future memory management
            chip_graph.graph.nodes[coord]['bw_inuse'] = 0
            chip_graph.graph.nodes[coord]['bandwidth'] = utils.IO_bw # GB/s
            chip_graph.graph.nodes[coord]['power'] = 10 # default staic power 10mW
            
        # Place other chiplets in the remaining nodes
        assigned_nodes = outer_ring_nodes[:self.chiplet_alloc["HBM3"]]
        remaining_nodes = [node for node in range(chip_graph.num_nodes) if node not in assigned_nodes]
        remaining_alloc = {
            k: v for k, v in self.chiplet_alloc.items() if k != "HBM3"
        }
        return assigned_nodes, remaining_nodes, remaining_alloc
    
    
    def init_comps(self, assigned_nodes:list, remaining_nodes:list, remaining_alloc:dict, chip_graph: chip_graph, model_config:BaseModelConfig , mem_size:int = 16):
        # Round-robin allocation
        chiplet_types_left = remaining_alloc.copy()
        idx = 0
        while any(count > 0 for count in chiplet_types_left.values()):
            for chiplet_type, count in chiplet_types_left.items():
                if count > 0:
                    node_id = remaining_nodes[idx]
                    coord = chip_graph.id_to_coord[node_id]

                    chip_graph.graph.nodes[coord]['chiplet_type'] = utils.chiplet_types_dict[chiplet_type]
                    chip_graph.graph.nodes[coord]['dram(g)'] = 0
                    chip_graph.graph.nodes[coord]['sram(g)'] = 0  # placeholder for future hierarchy
                    chip_graph.graph.nodes[coord]['bw_inuse'] = 0
                    chip_graph.graph.nodes[coord]['bandwidth'] = utils.IO_bw
                    chip_graph.graph.nodes[coord]['power'] = 5  # default static power

                    # Update trackers
                    chiplet_types_left[chiplet_type] -= 1
                    idx += 1
        return [],[]
    
    def make_chiplet_placement(self, chip_graph: chip_graph, model_config:BaseModelConfig , mem_size:int = 16, output_folder = "output", logging: bool = True):
        assigned_nodes, remaining_nodes, remaining_alloc = self.init_mems(chip_graph, model_config, mem_size)
        group_M_HBMs, group_A_HBMs = self.init_comps(assigned_nodes, remaining_nodes, remaining_alloc, chip_graph, model_config, mem_size)            
        # Output to json file
        if output_folder != "":
            output_file = os.path.join(output_folder, "placement.json")
            data = {
                "chiplet_alloc": self.chiplet_alloc,
                "nodes": [{"id": n, **attrs} for n, attrs in chip_graph.graph.nodes(data=True)],
                "edges": [{"source": u, "target": v, **attrs} for u, v, attrs in chip_graph.graph.edges(data=True)],
            }

            if logging:
                with open(output_file, "w") as f:
                    json.dump(data, f, indent=4)
                logger.info(f"Placement initialized for {chip_graph.num_nodes} chiplets")
                logger.info(f"Placement information saved to {output_file}")
        return group_M_HBMs, group_A_HBMs
    
    @staticmethod
    def get_name() -> str:
        """
        Get the name of the placer strategy.
        """
        return "rr"
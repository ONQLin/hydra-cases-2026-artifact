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

class Trace_Placer(BasePlacer):
    
    def __init__(self, chiplet_num: int, chiplet_alloc: Dict[str, int]):
        super().__init__(chiplet_num, chiplet_alloc)
    
    def init_mems(self, chip_graph: chip_graph, model_config:BaseModelConfig , mem_size:int = 16): # 16GB
        """
        Initialize the placement of memory chiplets.
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

        assigned_ids = []
        # In chip graph, loop over all nodes and check their types are HBM3 or GDDR7
        for node in chip_graph.graph.nodes(data=True):
            chiplet_id = node[0]

            # if chiplet id is HBM3 or GDDR7, assign 'bandwidth' etc
            if node[1]['chiplet_type'] == utils.chiplet_types_dict["HBM3"] or \
               node[1]['chiplet_type'] == utils.chiplet_types_dict["GDDR7"] or \
               node[1]['chiplet_type'] == utils.chiplet_types_dict["HBM3e"]:
                
                chip_graph.graph.nodes[chiplet_id]['dram(g)'] = mem_size # 16G HBM3 chiplet
                chip_graph.graph.nodes[chiplet_id]['sram(g)'] = 0 # for future memory management
                chip_graph.graph.nodes[chiplet_id]['bw_inuse'] = 0
                chip_graph.graph.nodes[chiplet_id]['bandwidth'] = utils.NoI_bw
                chip_graph.graph.nodes[chiplet_id]['latency(ns)'] = 10

                assigned_ids.append(chiplet_id)

        return assigned_ids

    def init_comps(self, assigned_ids: List[int], chip_graph: chip_graph, model_config:BaseModelConfig , mem_size:int = 16):
        
        # loop over all nodes in chip graph, if not in assigned_ids, assign compute properties
        for node in chip_graph.graph.nodes(data=True):
            chiplet_id = node[0]
            if chiplet_id not in assigned_ids:
                # assign properties
                chip_graph.graph.nodes[chiplet_id]['dram(g)'] = 0
                chip_graph.graph.nodes[chiplet_id]['sram(g)'] = 0  # placeholder for future hierarchy
                chip_graph.graph.nodes[chiplet_id]['bw_inuse'] = 0
                chip_graph.graph.nodes[chiplet_id]['bandwidth'] = utils.IO_bw
                chip_graph.graph.nodes[chiplet_id]['power'] = 5  # default static power

    def make_chiplet_placement(self, chip_graph: chip_graph, model_config:BaseModelConfig , mem_size:int = 16, output_folder = "output"):

        assigned_ids = self.init_mems(chip_graph, model_config, mem_size)
        self.init_comps(assigned_ids, chip_graph, model_config, mem_size)
        # Output to json file
        if output_folder != "":
            output_file = os.path.join(output_folder, "placement.json")
            data = {
                "chiplet_alloc": self.chiplet_alloc,
                "nodes": [{"id": n, **attrs} for n, attrs in chip_graph.graph.nodes(data=True)],
                "edges": [{"source": u, "target": v, **attrs} for u, v, attrs in chip_graph.graph.edges(data=True)],
            }

            with open(output_file, "w") as f:
                json.dump(data, f, indent=4)
            logger.info(f"Placement initialized for {chip_graph.num_nodes} chiplets")
            logger.info(f"Placement information saved to {output_file}")
        return [],[]
    
    @staticmethod
    def get_name() -> str:
        """
        Get the name of the placer strategy.
        """
        return "trace"
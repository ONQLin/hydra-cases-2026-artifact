from dataclasses import dataclass, field
import math
from typing import Dict, List, Optional, Tuple
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


class BW_Placer(BasePlacer):

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
        for j in BW_Placer.middle_out_order(height):
            outer_ring_nodes.append(chip_graph.graph.nodes[(0, j)]['id'])           # left
            outer_ring_nodes.append(chip_graph.graph.nodes[(width - 1, j)]['id'])  # right

        # Step 3 & 4: top and bottom rows in middle-out order, alternating
        for i in BW_Placer.middle_out_order(width):
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
        """
        Initialize the placement of compute chiplets in the system.
        In this method, we place compute chiplets in the remaining nodes after placing memory chiplets.
        """
        # idx = 0
        # chiplet_types_left = remaining_alloc.copy()
        
        if not model_config.hybrid:
            N_A = model_config.num_A_blocks
            N_M = model_config.num_M_blocks
            if model_config.num_M_blocks == 0:
                # Pure Attention model
                Mem_params_mamba = 0
                Mem_states_mamba = 0
                Mem_params_att = common.convert_param_mB(model_config.block_config.parameter_count)
                Mem_kv_att = common.convert_param_mB(model_config.block_config.max_position_embeddings *
                                                    model_config.block_config.states + model_config.block_config.peak_intermed)
            elif model_config.num_A_blocks == 0:
                # Pure Mamba model
                Mem_params_att = 0
                Mem_kv_att = 0
                Mem_params_mamba = common.convert_param_mB(model_config.block_config.parameter_count)
                Mem_states_mamba = common.convert_param_mB(model_config.block_config.states + model_config.block_config.peak_intermed)
            else:
                raise ValueError("Non-hybrid model cannot have both Mamba and Attention blocks.")
        else:
            # Hybrid model
            N_A, N_M = 0, 0
            Mem_params_mamba, Mem_states_mamba = 0, 0
            Mem_params_att, Mem_kv_att = 0, 0
            for block_type in model_config.block_type_sequence:
                if "mamba" in model_config.hybrid_blocks[block_type].type_name.lower():
                    N_M += 1
                    Mem_params_mamba += common.convert_param_mB(model_config.hybrid_blocks[block_type].parameter_count)
                    Mem_states_mamba += common.convert_param_mB(model_config.hybrid_blocks[block_type].states + model_config.hybrid_blocks[block_type].peak_intermed)
                elif "transformer" in model_config.hybrid_blocks[block_type].type_name.lower():
                    N_A += 1
                    Mem_params_att += common.convert_param_mB(model_config.hybrid_blocks[block_type].parameter_count)
                    Mem_kv_att += common.convert_param_mB(model_config.hybrid_blocks[block_type].max_position_embeddings *
                                                        model_config.hybrid_blocks[block_type].states + model_config.hybrid_blocks[block_type].peak_intermed)
                else:
                    raise ValueError(f"Unknown block type {block_type} in hybrid model.")
            if N_M != model_config.num_M_blocks or N_A != model_config.num_A_blocks:
                raise ValueError("Mismatch in number of Mamba/Attention blocks in hybrid model.")
            Mem_params_mamba = Mem_params_mamba/model_config.num_M_blocks
            Mem_states_mamba = Mem_states_mamba/model_config.num_M_blocks
            Mem_params_att = Mem_params_att/model_config.num_A_blocks
            Mem_kv_att = Mem_kv_att/model_config.num_A_blocks
            
        # For pure-Mamba models, BW clustering tends to over-pack compute chiplets
        # near HBM edges. Use RR-style compute spreading while keeping HBM placement.
        if (not model_config.hybrid) and model_config.num_A_blocks == 0 and model_config.num_M_blocks > 0:
            assigned_nodes.sort(reverse=False)
            group_M_nodes = assigned_nodes[:self.chiplet_alloc["HBM3"]]
            group_A_nodes = []
            self._round_robin_assign_comps(remaining_nodes, remaining_alloc, chip_graph)
            return group_M_nodes, group_A_nodes

        best_NM = self.group_hbms(
            L_M=model_config.num_M_blocks,
            L_A=model_config.num_A_blocks,
            Cap_HBM=mem_size * 1024, # in MB
            Mem_params_mamba=Mem_params_mamba,
            Mem_states_mamba=Mem_states_mamba,
            Mem_params_att=Mem_params_att,
            Mem_kv_att=Mem_kv_att,
            num_hbms=self.chiplet_alloc["HBM3"]
        )
        
        # based on the best_NM, we can decide the placement of HBM placements in groups first
        layer_m = N_M
        layer_a = N_A
        N_M = best_NM
        N_A = self.chiplet_alloc["HBM3"] - N_M
        if N_A > layer_a and layer_a > 0:
            N_A = layer_a
            N_M = self.chiplet_alloc["HBM3"] - N_A
        elif N_M > layer_m and layer_m > 0:
            N_M = layer_m
            N_A = self.chiplet_alloc["HBM3"] - N_M
        
        # sort assigned nodes to have a deterministic order
        assigned_nodes.sort(reverse=False)
        group_M_nodes = assigned_nodes[:N_M] if N_M > 0 else []
        group_A_nodes = assigned_nodes[N_M:N_M+N_A] if N_A > 0 else []
        bw_req = {k: 64 if 'p' in k else 256 for k in remaining_alloc.keys()}
        
        placement = self.assign_accs(
            group_M_nodes=group_M_nodes,
            group_A_nodes=group_A_nodes,
            accs_dict=remaining_alloc,
            loc_avail=[chip_graph.id_to_coord[nid] for nid in remaining_nodes],
            bw_reqs=bw_req,
            chip_graph=chip_graph
        )
        
        for coord, acc_type in placement.items():
            chip_graph.graph.nodes[coord]['chiplet_type'] = utils.chiplet_types_dict[acc_type]
            chip_graph.graph.nodes[coord]['dram(g)'] = 0
            chip_graph.graph.nodes[coord]['sram(g)'] = 0 
            chip_graph.graph.nodes[coord]['bw_inuse'] = 0
            chip_graph.graph.nodes[coord]['bandwidth'] = utils.IO_bw
            chip_graph.graph.nodes[coord]['power'] = 5 # default static power 5mW

        return group_M_nodes, group_A_nodes

    def _round_robin_assign_comps(self, remaining_nodes: List[int], remaining_alloc: Dict[str, int], chip_graph: chip_graph) -> None:
        """
        RR-style compute placement used as a stable fallback for pure-Mamba models.
        This mirrors RR placement behavior for compute chiplets while preserving BW HBM placement.
        """
        chiplet_types_left = remaining_alloc.copy()
        idx = 0
        while any(count > 0 for count in chiplet_types_left.values()):
            for chiplet_type, count in chiplet_types_left.items():
                if count > 0:
                    node_id = remaining_nodes[idx]
                    coord = chip_graph.id_to_coord[node_id]

                    chip_graph.graph.nodes[coord]['chiplet_type'] = utils.chiplet_types_dict[chiplet_type]
                    chip_graph.graph.nodes[coord]['dram(g)'] = 0
                    chip_graph.graph.nodes[coord]['sram(g)'] = 0
                    chip_graph.graph.nodes[coord]['bw_inuse'] = 0
                    chip_graph.graph.nodes[coord]['bandwidth'] = utils.IO_bw
                    chip_graph.graph.nodes[coord]['power'] = 5

                    chiplet_types_left[chiplet_type] -= 1
                    idx += 1

    # Step 1: Group HBMs for Mamba and Attention blocks
    def group_hbms(
        self,
        L_M: int,                     # # Mamba blocks
        L_A: int,                     # # Attention blocks
        Cap_HBM: float,               # per-HBM capacity (MB)
        Mem_params_mamba: float,      # Mamba params per block (MB)
        Mem_states_mamba: float,      # Mamba states per request (MB)
        Mem_params_att: float,        # Attention params per block (MB)
        Mem_kv_att: float,            # Attention KV-cache per request (MB)
        num_hbms: int,                # total HBMs
        # --- bandwidth inputs (per-request BW), in GB/s ---
        per_req_bw_mamba: float = 100,      # Mamba per-request BW
        per_req_bw_att: float   = 200,      # Attention per-request BW
        hbm_bw_cap: float       = 600,      # per-HBM BW capacity
        # --- scoring knobs ---
        balance_penalty: float = 0.5,        # penalize |r_M - r_A|
    ) -> int:
        """
        Partition HBMs between Mamba and Attention.
        Returns optimal N_M (HBMs for Mamba) considering BOTH memory and bandwidth.

        r_M_mem = floor((Cap_HBM - Mem_params_mamba * ceil(L_M / N_M)) / Mem_states_mamba)
        r_A_mem = floor((Cap_HBM - Mem_params_att *   ceil(L_A / N_A)) / Mem_kv_att)

        Bandwidth side (per HBM):
        blocks_per_hbm_M = ceil(L_M / N_M)
        blocks_per_hbm_A = ceil(L_A / N_A)
        per-request BW budget on an HBM ~= hbm_bw_cap / blocks_per_hbm_*
        r_M_bw = floor( (hbm_bw_cap / blocks_per_hbm_M) / per_req_bw_mamba )
        r_A_bw = floor( (hbm_bw_cap / blocks_per_hbm_A) / per_req_bw_att )

        Effective concurrency per HBM group is the min of mem- and BW-limits.
        We choose N_M maximizing min(r_M, r_A), with a small penalty on imbalance.
        """

        # Non-hybrid model shortcuts
        if num_hbms <= 0:
            raise ValueError("Number of HBMs must be positive.")
        if L_M == 0:
            return 0
        if L_A == 0:
            return num_hbms

        best_NM: Optional[int] = None
        best_score = -float("inf")
        best_tuple = None  # (r_M, r_A)

        for N_M in range(1, num_hbms):  # ensure both groups non-empty
            N_A = num_hbms - N_M
            if N_M <= 0 or N_A <= 0:
                raise ValueError("Both N_M and N_A must be positive.")
            # How many distinct blocks' params are parked on each HBM (per group)
            blocks_per_hbm_M = max(1, math.ceil(L_M / N_M))
            blocks_per_hbm_A = max(1, math.ceil(L_A / N_A))

            # ---- Memory-limited requests per HBM (clamped at >= 0) ----
            avail_mem_M = Cap_HBM - Mem_params_mamba * blocks_per_hbm_M
            avail_mem_A = Cap_HBM - Mem_params_att   * blocks_per_hbm_A

            r_M_mem = math.floor(avail_mem_M / Mem_states_mamba) if Mem_states_mamba > 0 else 0
            r_A_mem = math.floor(avail_mem_A / Mem_kv_att)       if Mem_kv_att       > 0 else 0
            r_M_mem = max(0, r_M_mem)
            r_A_mem = max(0, r_A_mem)

            # ---- Bandwidth-limited requests per HBM (clamped at >= 0) ----
            # Split HBM BW across the blocks hosted on that HBM
            if hbm_bw_cap > 0:
                per_req_budget_M = hbm_bw_cap / blocks_per_hbm_M
                per_req_budget_A = hbm_bw_cap / blocks_per_hbm_A
                
                r_M_bw = math.floor(per_req_budget_M / per_req_bw_mamba) if per_req_bw_mamba > 0 else 0
                r_A_bw = math.floor(per_req_budget_A / per_req_bw_att)   if per_req_bw_att   > 0 else 0

                r_M_bw = max(0, r_M_bw)
                r_A_bw = max(0, r_A_bw)
            else:
                # No BW constraint provided; fall back to mem-only
                r_M_bw = r_M_mem
                r_A_bw = r_A_mem

            # Effective per-HBM concurrency in each group is min of the two limits
            r_M = r_M_bw
            r_A = r_A_bw

            # Primary objective: maximize the bottleneck across groups
            # Secondary: prefer balanced splits (penalize |r_M - r_A|)
            score = min(r_M, r_A)


            if (score >= best_score):
                best_score = score
                
                best_NM = N_M
                best_tuple = (r_M, r_A)

        # Fallback: if nothing chosen (very tight constraints), bias by memory-only split
        # if best_NM is None:
        #     best_score = -float("inf")
        #     for N_M in range(1, num_hbms):
        #         N_A = num_hbms - N_M
        #         blocks_per_hbm_M = max(1, math.ceil(L_M / N_M))
        #         blocks_per_hbm_A = max(1, math.ceil(L_A / N_A))
        #         avail_mem_M = Cap_HBM - Mem_params_mamba * blocks_per_hbm_M
        #         avail_mem_A = Cap_HBM - Mem_params_att   * blocks_per_hbm_A
        #         r_M_mem = max(0, math.floor(avail_mem_M / Mem_states_mamba)) if Mem_states_mamba > 0 else 0
        #         r_A_mem = max(0, math.floor(avail_mem_A / Mem_kv_att))       if Mem_kv_att       > 0 else 0
        #         score = min(r_M_mem, r_A_mem)
        #         if score > best_score:
        #             best_score = score
        #             best_NM = N_M

        return max(1, int(best_NM))  # type: ignore
    
    
    def assign_accs(self,
        group_M_nodes: List[int],
        group_A_nodes: List[int],
        accs_dict: Dict[str, int],
        loc_avail: List[Coord],
        bw_reqs: Dict[str, float],
        chip_graph: chip_graph,
        n_trials: int = 1000,
    ) -> Dict[Coord, str]:
        """
        Step 2: Assign accelerators to locations minimizing Σ d_ij * V_req.
        """
        num_hbms = len(group_M_nodes) + len(group_A_nodes)
        loc_hbms_m = [chip_graph.id_to_coord[nid] for nid in group_M_nodes]
        loc_hbms_a = [chip_graph.id_to_coord[nid] for nid in group_A_nodes]

        # Divide groups by type
        group_M = [k for k in accs_dict if k in utils.chiplets_division["Mamba"]]
        group_A = [k for k in accs_dict if k in utils.chiplets_division["Attention"]]

        # Sort by number of chiplets (larger group first)
        groups = sorted(
            [(group_M, loc_hbms_m), (group_A, loc_hbms_a)],
            key=lambda g: sum(accs_dict[a] for a in g[0]),
            reverse=True,
        )
        
        # check the groups
        acc_to_merge = []
        for group, hbms in groups:
            if len(hbms) == 0 and sum(accs_dict[a] for a in group) > 0:
                # It is a pure A/M model
                for a in group:
                    acc_to_merge.extend([a] * accs_dict[a])

        placement = {}
        used_coords = set()

        def total_cost(mapping: Dict[Coord, str], hbms: List[Coord]) -> float:
            """Compute Σ d_ij * V_req(acc_i) over all accelerators."""
            total = 0.0
            for loc, acc in mapping.items():
                bw = bw_reqs[acc]
                total += sum(BW_Placer.manhattan(loc, h) * bw for h in hbms)
            return total
        
        for group, hbms in groups:
            # Compute the "region" = nearest |Accs_G| coords to HBMs
            if len(hbms) == 0:
                continue
            all_dists = []
            for loc in loc_avail:
                if loc in used_coords:
                    continue
                d_min = min(BW_Placer.manhattan(loc, h) for h in hbms)
                all_dists.append((d_min, loc))

            all_dists.sort(key=lambda x: x[0])
            total_accs = sum(accs_dict[a] for a in group)
            # assign the region for this group's accelerators
            region = [loc for _, loc in all_dists[:total_accs]]

            # Flatten all accelerator instances
            acc_instances = []
            if len(group) == 0:
                select_group = utils.avail_chiplets
            else:
                select_group = group
            for acc_type in select_group:
                acc_instances.extend([acc_type] * accs_dict[acc_type])

            best_trial_cost = float("inf")
            best_trial_map = {}

            # --- Monte Carlo search loop ---
            for _ in range(n_trials):
                random.shuffle(region)
                mapping = {region[i]: acc_instances[i] for i in range(total_accs)}
                cost = total_cost(mapping, hbms)
                if cost < best_trial_cost:
                    best_trial_cost = cost
                    best_trial_map = mapping

            # Commit the best found mapping
            for loc, acc in best_trial_map.items():
                placement[loc] = acc
                used_coords.add(loc)
            
            if acc_to_merge:
                # assign the remaining accelerators to the remaining locations
                remaining_locs = [loc for loc in loc_avail if loc not in used_coords]
                if len(remaining_locs) < len(acc_to_merge):
                    raise ValueError("Not enough locations to assign remaining accelerators.")
                remove_list = []
                for i, acc_type in enumerate(acc_to_merge):
                    loc = remaining_locs[i]
                    placement[loc] = acc_type
                    used_coords.add(loc)
                    remove_list.append(acc_type)
                for acc_type in remove_list:
                    acc_to_merge.remove(acc_type)

        return placement

    def make_chiplet_placement(self, chip_graph: chip_graph, model_config:BaseModelConfig , mem_size:int = 16, output_folder = "output", logging: bool = True):
        assigned_mem_nodes, remaining_nodes, remaining_alloc = self.init_mems(chip_graph, model_config, mem_size)
        group_M_HBMs, group_A_HBMs = self.init_comps(assigned_mem_nodes, remaining_nodes, remaining_alloc, chip_graph, model_config, mem_size)
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
    def manhattan(a: Coord, b: Coord) -> int:
        return abs(a[0] - b[0]) + abs(a[1] - b[1])
    
    @staticmethod    
    def get_name() -> str:
        return "bw"

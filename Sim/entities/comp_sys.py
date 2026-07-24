from Sim.entities.chips_network import chip_graph
from Sim.config.utils import chiplet_types_list
from Sim.config.model_config import BaseModelConfig, BaseBlockConfig
import Sim.common as common
from Sim.entities.comp_chiplet import comp_chiplet
from Sim.metrics.monitor import comp_monitor

import networkx as nx
import random

class comp_sys:
    def __init__(self, chip_graph: chip_graph):
        self.graph: nx.Graph = chip_graph.graph
        self._total_budget: int = 0
        self._current_run: int = 0
        self._comp_chiplets = []
        self._total_power: float = 0.0
        for coord, attr in self.graph.nodes(data=True):
            if attr['dram(g)'] == 0:
                if '_' not in chiplet_types_list[attr['chiplet_type']]:
                    raise ValueError(f"Non-comp chiplet {attr['chiplet_type']} found with 0 DRAM budget.")
                self._total_budget += 1
                self._comp_chiplets.append(comp_chiplet(
                    chiplet_type=attr['chiplet_type'],
                    chiplet_id=attr['id'],
                    chiplet_loc=coord if isinstance(coord, tuple) else attr['coord'][0],
                    power=attr.get('power', 10)  # Default power if not specified
                ))
                self._total_power += attr.get('power', 10)  # Default power if not specified

        self.total_comp_chiplets = len(self._comp_chiplets)
        comp_monitor.init_monitor(self._comp_chiplets)

    def find_available_chiplets_by_type(self, chiplet_types: list, chip_graph:chip_graph, use_bw) -> list[comp_chiplet]:
        """
        Find all chiplets of the specified types.
        """
        chiplets = []
        for chiplet in self._comp_chiplets:
            assert isinstance(chiplet, comp_chiplet), "Chiplet is not of type comp_chiplet."
            if chiplet.chiplet_type in chiplet_types and chiplet.is_available():
                if chip_graph.check_bw_availability(chiplet.chiplet_id, use_bw):
                    chiplets.append(chiplet)
        return chiplets
    
    def find_chiplet_by_id(self, chiplet_id: int) -> comp_chiplet:
        """
        Find the chiplet with the specified ID.
        """
        for chiplet in self._comp_chiplets:
            assert isinstance(chiplet, comp_chiplet), "Chiplet is not of type comp_chiplet."
            if chiplet.chiplet_id == chiplet_id:
                return chiplet
        raise ValueError(f"Chiplet with ID {chiplet_id} not found.")
    
    # all types of comp chiplets
    def find_all_available_chiplets(self, chip_graph:chip_graph, use_bw) -> list[comp_chiplet]:
        """
        Find all available chiplets of the specified types.
        """
        chiplets = []
        for chiplet in self._comp_chiplets:
            assert isinstance(chiplet, comp_chiplet), "Chiplet is not of type comp_chiplet."
            if chiplet.is_available():
                if chip_graph.check_bw_availability(chiplet.chiplet_id, use_bw):
                    chiplets.append(chiplet)
        return chiplets
    
    def find_chiplets_by_type(self, chiplet_types: list) -> list[comp_chiplet]:
        return [c for c in self._comp_chiplets if c.chiplet_type in chiplet_types]
    
    def set_busy_byid(self, chiplet_id: int):
        """
        Set the chiplet with the given ID as busy. # it is not running, we set busy then it would start runnuing
        """
        for chiplet in self._comp_chiplets:
            assert isinstance(chiplet, comp_chiplet), "Chiplet is not of type comp_chiplet."
            if chiplet.chiplet_id == chiplet_id:
                chiplet.set_busy()
                self._current_run += 1
                # comp_monitor.update_utilization(self._comp_chiplets, self._current_run, 0)
                return
        raise ValueError(f"Chiplet with ID {chiplet_id} not found.")

    def set_used_byid(self, chiplet_id: int, util: float, power: float, time: int):
        """
        Set the chiplet with the given ID as used.
        """
        for chiplet in self._comp_chiplets:
            assert isinstance(chiplet, comp_chiplet), "Chiplet is not of type comp_chiplet."
            if chiplet.chiplet_id == chiplet_id:
                chiplet.set_util(util=util, power=power)  # Set utilization and power
                comp_monitor.update_utilization(self._comp_chiplets, self._current_run, time)
                return
        raise ValueError(f"Chiplet with ID {chiplet_id} not found.")
        
    def set_free_byid(self, chiplet_id: int, time):
        """
        Set the chiplet with the given ID as free.
        """
        for chiplet in self._comp_chiplets:
            assert isinstance(chiplet, comp_chiplet), "Chiplet is not of type comp_chiplet."
            if chiplet.chiplet_id == chiplet_id:
                chiplet.set_free()
                self._current_run -= 1
                comp_monitor.update_utilization(self._comp_chiplets, self._current_run, time)
                return
        raise ValueError(f"Chiplet with ID {chiplet_id} not found.")
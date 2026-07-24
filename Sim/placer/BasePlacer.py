from dataclasses import dataclass, field
from abc import ABC, abstractmethod
from Sim.config.utils import get_all_subclasses
from typing import Any

class BasePlacer(ABC):

    chiplet_num: int = field(
        default=16, metadata={"description": "Number of chiplets in the system"}
    )

    chiplet_alloc: dict = field(
        default_factory=lambda: {
            "SSM_pf": 0,
            "SSM_dc": 0,
            "Att_pf": 4,
            "Att_dc": 8,
            "HBM3": 4, # 16G
        },
        metadata={"description": "Chiplet allocation in the system"},
    )    

    def __init__(self, chiplet_num:int=None, chip_alloc:dict=None):
        """
        Post-initialization method to set up the chiplet placement.
        """
        if chip_alloc is not None:
            self.chiplet_alloc = chip_alloc

        # check the num and allocation of chiplets
        if chiplet_num != sum(chip_alloc.values()):
            raise ValueError("Total chiplet number does not match the allocation.")
        
        self.chiplet_num = chiplet_num
            
    @classmethod
    def create_from_name(cls, name: str) -> Any:
        for subclass in get_all_subclasses(cls):
            if subclass.get_name() == name:
                return subclass
        raise ValueError(f"[{cls.__name__}] Invalid name: {name}")
    
    
    # Helper to generate row order starting from middle (fan outward order)
    @staticmethod
    def middle_out_order(n):
        mid = n // 2
        order = [mid]
        for offset in range(1, mid + 1):
            if mid + offset < n - 1:  # stay inside (exclude corners)
                order.append(mid + offset)
            if mid - offset > 0:      # stay inside (exclude corners)
                order.append(mid - offset)
        return order
    
    @abstractmethod
    def init_mems():
        """
        Initialize the placement of memory chiplets in the system.
        """
        raise NotImplementedError
    
    @abstractmethod
    def init_comps():
        """
        Initialize the placement of compute chiplets in the system.
        """
        raise NotImplementedError
    
    @abstractmethod
    def make_chiplet_placement():
        """
        Generate the chiplet placement mapping.
        """
        raise NotImplementedError
    
    @abstractmethod
    def get_name() -> str:
        """
        Get the name of the placer strategy.
        """
        raise NotImplementedError
    
    def __str__(self) -> str:
        # use to_dict to get a dict representation of the object
        # and convert it to a string
        class_name = self.__class__.__name__
        return f"{class_name}({str(self.to_dict())})"
    
    
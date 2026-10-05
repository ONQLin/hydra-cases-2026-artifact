
from dataclasses import dataclass, field

class mem_chiplet:
    # at least it reqires chiplet_type, dram_budget, chiplet_id and chiplet_loc
    chiplet_type: int
    dram_budget: float # in MB
    chiplet_id: int
    chiplet_loc: tuple
    BW_budget: float
    
    def __init__(self, **attr):
        for key, value in attr.items():
            setattr(self, key, value)
        if not hasattr(self, 'chiplet_type'):
            raise ValueError("Chiplet type must be specified.")
        if not hasattr(self, 'dram_budget'):
            raise ValueError("DRAM budget must be specified.")
        if not hasattr(self, 'chiplet_id'):
            raise ValueError("Chiplet ID must be specified.")
        if not hasattr(self, 'chiplet_loc'):
            raise ValueError("Chiplet location must be specified.")
        self.dram_budget = float(self.dram_budget * 1024) # convert to MB
        self.inuse_budget = 0.0 # in MB, how much is currently used
        self.content_params: dict = {}  # dict of Params objects representing the content of the chiplet
        # self.content_intermediate: dict = {}  # dict of intermediate objects representing the intermediate data in the chiplet
        self.bw_inuse = 0.0 # in GB/s, how much bandwidth is currently used
        self._current_allocated = 0.0 # in MB, how much is currently allocated - allocated resources but not yet used
        self.assigned = "A"
        
    def __repr__(self):
        return f"mem_chiplet(type={self.chiplet_type}, budget={self.dram_budget})"
    
    def is_capacity_available(self, size: float) -> bool:
        """
        Check if the chiplet has enough capacity to accommodate the given size.
        """
        return self.inuse_budget + size <= self.dram_budget
    
    def is_bandwidth_available(self, size: float) -> bool:
        """
        Check if the chiplet has enough bandwidth to accommodate the given size.
        """
        return self.bw_inuse + size <= self.BW_budget

@dataclass
class Params:
    size: int = field(default=0, metadata={"help": "Size of the parameters in MB."})
    chunk_id: int = field(default=0, metadata={"help": "ID of the chunk."})
    block_id: int = field(default=0, metadata={"help": "ID of the block."})
    name: str = field(default="", metadata={"help": "Name of the parameter."})  
    #loc: int = field(default=0, metadata={"help": "Location ID of the data in the chiplet."})
    chip_id: int = field(default=-1, metadata={"help": "ID of the chiplet where the parameters are stored."})
    
@dataclass
class intermediate:
    size: int = field(default=0, metadata={"help": "Size of the intermediate data in MB."})
    block_id: int = field(default=0, metadata={"help": "ID of the block."})
    name: str = field(default="", metadata={"help": "Name of the intermediate data."})
    #loc: int = field(default=0, metadata={"help": "Location ID of the data in the chiplet."})
    chip_id: int = field(default=-1, metadata={"help": "ID of the chiplet where the intermediate data is stored."})
    req_id: int | str = field(default=-1, metadata={"help": "Request ID or retained-cache owner for tracking."})

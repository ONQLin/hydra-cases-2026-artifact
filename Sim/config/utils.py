from dataclasses import fields, is_dataclass
from typing import Union, get_args, get_origin, List, Dict
from typing import List, Tuple, Union, Optional
import yaml, csv
import importlib
import os
from typing import List

from Sim.logger import init_logger, _setup_logger
_setup_logger()   # root logger handles console + file
logger = init_logger(__name__)  # child logger inherits handlers

primitive_types = {int, str, float, bool, type(None)}
NoI_bw = 128  # in GBps Full-duplex D2D NoI bandwidth 8.2Tb w.r.t. UCIE 3.0
IO_bw = 600  # in GBps use HBM3 setup by default
used_bws = None # in GBps would be configured wrt NoIbw and chiplets types
total_area = 0
HW_D2D_BW = 0
set_chiplet_area = 121
Fixed_chiplet_area = False
verbose = False

used_bws_dict = {
    128: [64,64],
    # 256: [128,128], # AK: prefill package
    256: [64,96],
    384: [64,160],
    512: [64,192],
    640: [64,192], # limit by IO
    # 1024: [512,512] # AK: decode package
}

used_bws_dict_tensor = {
    256: [128,128],
    1024: [512,512]
}

def get_all_subclasses(cls):
    subclasses = cls.__subclasses__()
    return subclasses + [g for s in subclasses for g in get_all_subclasses(s)]


def is_primitive_type(field_type: type) -> bool:
    # Check if the type is a primitive type
    return field_type in primitive_types


def is_generic_composed_of_primitives(field_type: type) -> bool:
    origin = get_origin(field_type)
    if origin in {list, dict, tuple, Union}:
        # Check all arguments of the generic type
        args = get_args(field_type)
        return all(is_composed_of_primitives(arg) for arg in args)
    return False


def is_composed_of_primitives(field_type: type) -> bool:
    # Check if the type is a primitive type
    if is_primitive_type(field_type):
        return True

    # Check if the type is a generic type composed of primitives
    if is_generic_composed_of_primitives(field_type):
        return True

    return False


def to_snake_case(name: str) -> str:
    return "".join(["_" + i.lower() if i.isupper() else i for i in name]).lstrip("_")


def is_optional(field_type: type) -> bool:
    return get_origin(field_type) is Union and type(None) in get_args(field_type)


def is_list(field_type: type) -> bool:
    # Check if the field type is a List
    return get_origin(field_type) is list


def is_dict(field_type: type) -> bool:
    # Check if the field type is a Dict
    return get_origin(field_type) is dict


def is_bool(field_type: type) -> bool:
    return field_type is bool


def get_inner_type(field_type: type) -> type:
    return next(t for t in get_args(field_type) if t is not type(None))


def is_subclass(cls, parent: type) -> bool:
    return hasattr(cls, "__bases__") and parent in cls.__bases__


def dataclass_to_dict(obj):
    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    elif is_dataclass(obj):
        data = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            data[field.name] = dataclass_to_dict(value)
        # Include members created in __post_init__
        for key, value in obj.__dict__.items():
            if key not in data:
                data[key] = dataclass_to_dict(value)
        # Include the name of the class
        if hasattr(obj, "get_type") and callable(getattr(obj, "get_type")):
            data["name"] = str(obj.get_type())
        elif hasattr(obj, "get_name") and callable(getattr(obj, "get_name")):
            data["name"] = obj.get_name()
        return data
    else:
        return obj

def dataclass_to_dict(obj):
    if isinstance(obj, list):
        return [dataclass_to_dict(item) for item in obj]
    elif is_dataclass(obj):
        data = {}
        for field in fields(obj):
            value = getattr(obj, field.name)
            if not isinstance(value, list) and not is_dataclass(value):
                value = str(value)
            data[field.name] = dataclass_to_dict(value)
        # Include members created in __post_init__
        for key, value in obj.__dict__.items():
            if key not in data:
                if not isinstance(value, list) and not is_dataclass(value):
                    value = str(value)
                data[key] = dataclass_to_dict(value)
        # Include the name of the class
        if hasattr(obj, "get_type") and callable(getattr(obj, "get_type")):
            data["name"] = str(obj.get_type())
        elif hasattr(obj, "get_name") and callable(getattr(obj, "get_name")):
            data["name"] = obj.get_name()
        return data
    else:
        return obj


def load_placement_trace_from_csv(csv_path: Optional[str] = None) -> Optional[List[Tuple[Union[int,str], List[Tuple[int,int]]]]]:
    """
    Load placement_trace from CSV.
    CSV columns: chiplet_type, tiles
    tiles format: "x:y" or "x1:y1;x2:y2"
    Returns: [(chip_id_or_name, [(x,y), ...]), ...] or None if not found/empty.
    """
    if csv_path is None:
        csv_path = os.path.normpath(os.path.join(os.path.dirname(__file__), "..", "..", "placement_trace.csv"))
    if not os.path.exists(csv_path):
        return None

    placement: List[Tuple[Union[int,str], List[Tuple[int,int]]]] = []
    placement = []
    with open(csv_path, newline='') as f:
        reader = csv.DictReader(f)
        for row in reader:
            chip_type = row["chiplet_type"]
            # tiles column is semicolon separated "x:y;..."
            tiles_raw = row["tiles"].strip()
            tiles = []
            if tiles_raw:
                for t in tiles_raw.split(";"):
                    x_str, y_str = t.split(":")
                    tiles.append((int(x_str), int(y_str)))
            # convert chip_type string to type id if available
            if chip_type in chiplet_types_dict:
                chip_id = chiplet_types_dict[chip_type]
            else:
                try:
                    chip_id = int(chip_type)
                except Exception:
                    chip_id = chip_type  # keep as string fallback
            placement.append((chip_id, tiles))
        
    return placement

# Make them from yaml files later
chiplet_types_dict = {
    "marca_p": 0,
    "marca_d": 1,
    "tscs_p": 2,
    "tscs_d": 3,
    "systolicarray_p": 4,
    "HBM3": 5,
    "GDDR7": 6,
    "HBM3e": 7,
    "B200_p": 8,
    "vuarray_d": 9,
    "unified_p": 10,
}

T_P = [2, 4, 8, 10]
M_P = [0, 4, 8, 10]
T_D = [3, 9]
M_D = [1, 9]

chiplet_types_list = list(chiplet_types_dict.keys())

avail_chiplets = ['marca_p', 'marca_d', 'tscs_p', 'tscs_d', 'systolicarray_p', 'HBM3', 'GDDR7', 'HBM3e', 'B200_p', 'vuarray_d', 'unified_p']

chiplets_division = {
    "Attention": ["tscs_p", "tscs_d", "systolicarray_p", "B200_p", "vuarray_d", "unified_p"],
    "Mamba": ["marca_p", "marca_d", "systolicarray_p", "B200_p", "vuarray_d", "unified_p"],
    "Memory": ["HBM3", "GDDR7", "HBM3e"],
}


class chiplets_lib:
    def __init__(self, node:str="28nm", logic_lib:List[str] = ['marca_d','marca_p','tscs_d','tscs_p'],
                 mem_lib:List[str] = ['hbm3'], area_limit:int = 121, area_utilization:float = 1.0, NoI_bw:int = 128, hw_bw:int = 128):
        self.comp_lib = {}
        self.mem_lib = {}
        self.area_limit = area_limit
        self.area_utilization = area_utilization
        acc_component_file = f"analytic_profile/comp_acc/component_{node}.yml"
        NoI_bw = NoI_bw
        global HW_D2D_BW
        global set_chiplet_area
        HW_D2D_BW = hw_bw
        set_chiplet_area = area_limit
        try:
            with open(acc_component_file, 'r') as file:
                self.acc_components = yaml.safe_load(file)
        except FileNotFoundError:
            raise FileNotFoundError(f"Error: The file {acc_component_file} was not found.")

        for lib in logic_lib:
            lib_file = f"analytic_profile/comp_acc/{lib}_{set_chiplet_area}mm2.yml"
            try:
                with open(lib_file, 'r') as file:
                    self.comp_lib[lib] = yaml.safe_load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"Error: The file {lib_file} was not found.")
            
        for lib in mem_lib:
            lib_file = f"analytic_profile/mem/{lib}.yml"
            try:
                with open(lib_file, 'r') as file:
                    self.mem_lib[lib] = yaml.safe_load(file)
            except FileNotFoundError:
                raise FileNotFoundError(f"Error: The file {lib_file} was not found.")
        
        # store the specified logics configurations
        self.logics = {}
        # global total_area
        total_area = 0
        
        self.area_breakdown = {}
        for lib in self.comp_lib.keys():
            self.area_breakdown[lib] = {}
        # check the area constraints
        for key, logic_configs in self.comp_lib.items():
            try:
                logic_die:Dict = logic_configs[hw_bw]
            except KeyError:
                raise KeyError(f"Error: No D2D BW '{hw_bw}' found in logic_configs.")
            
            area = 0
            config_dict = {}
            for comp, num in logic_die.items():
                if 'config' in comp:
                    if not isinstance(num, dict):
                        raise TypeError(f"Expected dict for '{comp}', got {type(num)}")
                    config_dict = num
                    continue
                if comp == 'core':
                    identies = [key.split('_')[0]]
                elif comp == 'links':
                    identies = ['bump','PHY']
                else:
                    identies = [comp]
                # [key][hw_bw][component]
                self.area_breakdown[key][comp] = 0
                for id in identies: 
                    if id in config_dict.keys():
                        area += num * self.acc_components[id][config_dict[id]]['area']
                        self.area_breakdown[key][comp] += num * self.acc_components[id][config_dict[id]]['area']
                    else:
                        area += num * self.acc_components[id][hw_bw]['area']
                        self.area_breakdown[key][comp] += num * self.acc_components[id][hw_bw]['area']
            try:
                logic_die['core_MAC'] = self.acc_components[key.split('_')[0]][logic_die['config'][key.split('_')[0]]]['MAC']
                logic_die['rows'] = self.acc_components[key.split('_')[0]][logic_die['config'][key.split('_')[0]]]['rows']
                logic_die['cols'] = self.acc_components[key.split('_')[0]][logic_die['config'][key.split('_')[0]]]['cols']
            except KeyError:
                pass
            self.logics[key] = logic_die
            if area < self.area_limit*self.area_utilization:
                self.logics[key]['area'] = area
            else:
                raise ValueError(f"Logic {key} exceeds area limit {self.area_limit} with area {area}.")
            # extra_bw = int((NoI_bw - hw_bw)/128)
            # extra_area = extra_bw * ((self.acc_components['PHY'][128]['area']) * 4 + 
            #                          self.acc_components['bump'][128]['area'] + self.acc_components['dma'][128]['area'])
            # total_area += area + extra_area
            # logger.info(f"Total area is {total_area} = {area} + {extra_area}")

        self.mem_lib = {}
        for key, mem_lib in self.mem_lib.items():
            self.mem_lib[key] = mem_lib 
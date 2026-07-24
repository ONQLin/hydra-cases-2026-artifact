def init_ramulator_csv(filename="ramulator_log.csv"):
    with open(filename, "w") as f:
        f.write("Address,Type,IssueCycle,CompleteCycle,Latency\n")

# Call it before init_ramulator
init_ramulator_csv()
import ctypes
import os
# os.environ["LD_LIBRARY_PATH"] = "/disk/jlin445/hybrid/perf_model/"
# Load the shared library
lib = ctypes.CDLL("./libramulator_wrapper.so")
# Define function signatures
lib.init_ramulator.argtypes = [ctypes.c_char_p]
lib.send_memory_request.argtypes = [ctypes.c_uint64, ctypes.c_bool, ctypes.c_int]
lib.send_memory_request.restype = ctypes.c_bool
lib.tick_ramulator.argtypes = []
lib.finalize_ramulator.argtypes = []

# Initialize Ramulator
lib.init_ramulator(b"test_config.yaml")

# Send a memory request
for i in range(1024):
    addr = int(i * 8)
    print(f"i = {i}, addr = {addr}")
    while(not lib.send_memory_request(addr, True, 0)):
        lib.tick_ramulator()

# Tick the simulator
for _ in range(1000000):
    lib.tick_ramulator()

# Finalize
lib.finalize_ramulator()
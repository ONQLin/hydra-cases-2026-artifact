# Simulation integrations

[Architecture](../README.md#multi-fidelity-validation) →
[Accuracy and simulation cost](../docs/multi_fidelity/README.md) →
Run [Packet validation](packet/README.md) or [CHIPSIM validation](chipsim/README.md).

HYDRA retains the experiment setup and serving runtime. These integrations
implement execution providers; their completion events release HYDRA resources
and enable subsequent work. Installation and ordered experiment commands belong
in each provider's README. Curated measurements belong in
[the validation suites](../tests/validation/packet_validation/README.md).

The high-fidelity provider is called **CHIPSIM/Garnet**: CHIPSIM supplies the
integration foundation, and Garnet executes the network. Compute profiles and
compute/network coordination remain in HYDRA. See the
[precise scope](chipsim/README.md#terminology-and-responsibilities).

## Code ownership

| Layer | Classes / location | Responsibility |
|---|---|---|
| Simulation entry | `BaseSimulationBackend`, `Sim/backends/` | Select the backend using HYDRA's subclass factory; run the experiment |
| Policy support | `StaticServingMixin` | Reject unsupported batching, mapping, topology, or profile settings before a refined run |
| Execution interface | `BaseExecutionBackend`, `Sim/execution/` | Initialize the placed system, return block/transfer completion events, release resources |
| Shared compute | `NativeExecutionBackend` | Profile blocks and export ordered compute/memory phases |
| Hardware export | `NetworkSystemSnapshot` | Export consistent IDs, topology, memory bandwidth, and mapping |
| Network coordination | `NetworkExecutionMixin`, `CoSimulationEnvironment` | Bound advancement by the next application event; deliver completions and sequence phases |
| Medium provider | `PacketNetwork`, `packet/` | Own C++ packet events behind a versioned C ABI and Python `ctypes` wrapper |
| High provider | `PersistentGarnetNetwork`, `chipsim/` | Own the persistent Garnet child and its control protocol |

Both factories use `create_from_name()` and return a class, following the
existing placer/scheduler convention. Implementations register through imports
in the corresponding `__init__.py`; serving code does not switch on backend
names. `chipsim` (standalone snapshot) and `chipsim_runtime` (isolated transfers)
remain diagnostic entry points. The high-fidelity serving reference is
`chipsim_contended`.

## Extending a backend

1. Add a `BaseSimulationBackend` subclass with a unique `get_name()`. A serving
   backend can reuse `NativeSimulationBackend` and the applicable policy mixin.
2. Add an execution subclass with the same name and import both implementations
   into their factories. Keep backend parameters in `Sim/config/sys_config.py`.
3. For a network provider, reuse `NetworkSystemSnapshot` and
   `NetworkExecutionMixin`. Supply `_request(message)`, `memory_bandwidth`,
   `output_dir`, and `trace`. Implement `submit` and `advance` responses with
   integer picosecond ticks and a flat `[flow_id, completion_tick, ...]` list.
4. `advance` must stop at a completion or the requested time boundary. It must
   not run past a future HYDRA submission. Preserve unique completions, monotonic
   time, and flow/byte conservation. `close()` must release resources after
   partial initialization or failure and tolerate repeated calls.
5. Check isolated/disjoint/contending/staggered flows and fixed cutoffs, then run
   paired serving with identical input snapshots. Record process wall time
   separately from simulated time; use the comparison tools to check equivalence.

Shared phase sequencing and compute profiles should stay in the execution
layer. A new provider should implement its network mechanism without copying
the scheduler, workload handling, or metric recorder. Broader policies such as
elastic mapping require their own contract and validation before being enabled.

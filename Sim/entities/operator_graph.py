"""Tensor dependencies and conservative memory lowering for experimental operators."""

from dataclasses import asdict, dataclass
import math
import heapq

from Sim.entities.execution import ExecutionPhase


@dataclass(frozen=True)
class TensorSpec:
    name: str
    shape: tuple
    bytes_per_element: int = 1
    storage: str = 'activation'

    def __post_init__(self):
        if not self.name or not self.shape or any(type(d) is not int or d < 0 for d in self.shape):
            raise ValueError('Tensor shapes require nonnegative integer dimensions and a name.')
        if self.bytes_per_element not in (1, 2, 4, 8):
            raise ValueError('Unsupported tensor element size.')
        if self.storage not in ('input', 'activation', 'output', 'weight', 'state'):
            raise ValueError('Unknown tensor storage class.')

    @property
    def nbytes(self):
        return math.prod(self.shape) * self.bytes_per_element


@dataclass(frozen=True)
class OperatorNode:
    name: str
    inputs: tuple
    outputs: tuple
    macs: int = 0
    element_ops: int = 0
    dependencies: tuple = ()
    workspace_bytes: int = 0
    recurrent_bytes: int = 0
    recurrent_steps: int = 1

    def __post_init__(self):
        counts = (self.macs, self.element_ops, self.workspace_bytes, self.recurrent_bytes)
        if any(type(v) is not int or v < 0 for v in counts):
            raise ValueError('Operator costs require nonnegative integer counts.')
        if type(self.recurrent_steps) is not int or self.recurrent_steps < 1:
            raise ValueError('Recurrence length must be positive.')
        if not self.recurrent_bytes and self.recurrent_steps != 1:
            raise ValueError('A recurrence requires explicit state storage.')


class OperatorGraph:
    """Single-assignment tensors; state input/output versions have distinct names.

    A deterministic topological schedule uses one admitted compute chiplet.
    Independent branches are represented but are not executed concurrently.
    Weights/state reside in the mapped HBM. Activations spill as a whole graph
    when their peak live set plus workspace exceeds SRAM. This intentionally
    conservative policy is explicit; it is not a cache replacement simulator.
    """

    def __init__(self, name):
        self.name = name
        self.tensors = {}
        self.nodes = []
        self.schedules = {}
        self.metadata = {}

    def include(self, prefix, graph, input_tensor):
        """Compose an operator graph while preserving state and schedule bindings."""
        if self.tensors[input_tensor].shape != graph.tensors['input'].shape:
            raise ValueError('Composed operator input shapes must agree.')
        names = {'input': input_tensor}
        for key, tensor in graph.tensors.items():
            if key != 'input':
                names[key] = self.tensor(prefix+key, tensor.shape, tensor.bytes_per_element,
                    'activation' if tensor.storage == 'output' else tensor.storage)
        for node in graph.ordered_nodes():
            values = asdict(node)
            values.update(name=prefix+node.name, inputs=tuple(names[k] for k in node.inputs),
                          outputs=tuple(names[k] for k in node.outputs),
                          dependencies=tuple(prefix+k for k in node.dependencies))
            self.nodes.append(OperatorNode(**values))
        for key, schedule in graph.schedules.items():
            self.bind_schedule(prefix+key,schedule)
        self.metadata[prefix.rstrip('.')] = graph.metadata
        return names['output']

    def bind_schedule(self, node_name, schedule):
        """Attach an explicit tiled implementation to an existing graph node."""
        if node_name not in {node.name for node in self.nodes} or node_name in self.schedules:
            raise ValueError('A schedule requires a unique existing operator node.')
        self.schedules[node_name] = schedule

    def tensor(self, name, shape, bytes_per_element=1, storage='activation'):
        if name in self.tensors:
            raise ValueError(f'Duplicate tensor {name}.')
        self.tensors[name] = TensorSpec(name, tuple(shape), bytes_per_element, storage)
        return name

    def add(self, name, inputs, outputs, **costs):
        self.nodes.append(OperatorNode(name, tuple(inputs), tuple(outputs), **costs))
        return outputs[0] if outputs else None

    def ordered_nodes(self):
        nodes = {node.name: node for node in self.nodes}
        if len(nodes) != len(self.nodes):
            raise ValueError('Duplicate operator name.')
        producers = {}
        for node in self.nodes:
            for key in node.inputs + node.outputs:
                if key not in self.tensors:
                    raise ValueError(f'Unknown tensor {key}.')
            if len(set(node.outputs)) != len(node.outputs):
                raise ValueError('Duplicate node output.')
            for key in node.outputs:
                if key in producers or self.tensors[key].storage in ('input', 'weight'):
                    raise ValueError(f'Multiple or invalid producers for {key}.')
                producers[key] = node.name
        dependencies = {}
        for node in self.nodes:
            deps = set(node.dependencies)
            if not deps <= nodes.keys():
                raise ValueError('Unknown operator dependency.')
            for key in node.inputs:
                if key in producers:
                    deps.add(producers[key])
                elif self.tensors[key].storage in ('activation', 'output'):
                    raise ValueError(f'Missing producer for {key}.')
            dependencies[node.name] = deps
        positions = {node.name: i for i,node in enumerate(self.nodes)}
        consumers = {name: [] for name in nodes}
        remaining = {name: len(deps) for name,deps in dependencies.items()}
        for name,deps in dependencies.items():
            for parent in deps:
                consumers[parent].append(name)
        ready = [positions[name] for name,count in remaining.items() if count == 0]
        heapq.heapify(ready)
        result = []
        while ready:
            node = self.nodes[heapq.heappop(ready)]
            result.append(node)
            for name in consumers[node.name]:
                remaining[name] -= 1
                if remaining[name] == 0:
                    heapq.heappush(ready,positions[name])
        if len(result) != len(nodes):
            raise ValueError('Operator dependency cycle.')
        return result

    def peak_activation_bytes(self, retained_inputs=()):
        # Tile programs may retain inputs across invocations after their final
        # local consumer (for example Q across online-softmax key tiles).
        retained = set(retained_inputs)
        if any(key not in self.tensors or self.tensors[key].storage != 'input' for key in retained):
            raise ValueError('Retained tensors must be graph inputs.')
        schedule = self.ordered_nodes()
        last_use = {key: i for i, node in enumerate(schedule) for key in node.inputs}
        live = {key for key, t in self.tensors.items() if t.storage == 'input'}
        peak = sum(self.tensors[key].nbytes for key in live)
        for i, node in enumerate(schedule):
            live.update(key for key in node.outputs
                        if self.tensors[key].storage in ('activation', 'output'))
            peak = max(peak, sum(self.tensors[key].nbytes for key in live) + node.workspace_bytes)
            live = {key for key in live if key in retained or last_use.get(key, i) > i}
        return peak

    def lower(self, rates, force_spill=False):
        speeds = (rates.macs_per_ns, rates.elements_per_ns, rates.sram_bytes_per_ns)
        if any(not math.isfinite(v) or v <= 0 for v in speeds):
            raise ValueError('Accelerator rates must be positive.')
        if type(rates.sram_capacity_bytes) is not int or rates.sram_capacity_bytes < 0:
            raise ValueError('SRAM capacity must be a nonnegative integer.')
        schedule = self.ordered_nodes()
        peak = self.peak_activation_bytes()
        # Tiled kernels consume slices from HBM staging buffers. Materializing
        # outer intermediates makes that boundary explicit and capacity-safe.
        spill = force_spill or bool(self.schedules) or peak > rates.sram_capacity_bytes
        phases = []
        for node in schedule:
            if node.name in self.schedules:
                phases.extend(self.schedules[node.name].lower(rates))
                continue
            inputs = [self.tensors[key] for key in node.inputs]
            outputs = [self.tensors[key] for key in node.outputs]
            reads = sum(t.nbytes for t in inputs if spill or t.storage in ('input', 'weight', 'state'))
            writes = sum(t.nbytes for t in outputs if spill or t.storage in ('state', 'output'))
            # Node-private scratch must fit locally. A larger materialized
            # temporary needs a tensor and separate producer/consumer nodes,
            # so spilling cannot read it before its producer has run.
            if node.workspace_bytes > rates.sram_capacity_bytes:
                raise ValueError('Oversized workspace requires explicit intermediate tensors.')
            arithmetic = node.macs / rates.macs_per_ns + node.element_ops / rates.elements_per_ns
            local_bytes = sum(t.nbytes for t in inputs + outputs) + 2*node.workspace_bytes
            # A recurrence accesses its matrix for every token, even when it
            # stays in SRAM. External traffic is counted separately below.
            local_bytes += 2 * node.recurrent_bytes * node.recurrent_steps
            compute = max(arithmetic, local_bytes / rates.sram_bytes_per_ns)
            if node.recurrent_bytes:
                state_read = sum(t.nbytes for t in inputs if t.storage == 'state')
                state_write = sum(t.nbytes for t in outputs if t.storage == 'state')
                if state_write != node.recurrent_bytes or state_read not in (0, node.recurrent_bytes):
                    raise ValueError('Recurrence state versions must match the declared matrix size.')
            if node.recurrent_bytes and peak + node.recurrent_bytes > rates.sram_capacity_bytes:
                # Explicit token order: read state -> update/query -> write state.
                # Projection/activation traffic is charged once at the boundary;
                # intermediate state traffic repeats without expanding the graph.
                phases.append(ExecutionPhase(0, reads-state_read, node.name+':inputs'))
                per_step = compute / node.recurrent_steps
                first_read = node.recurrent_bytes if state_read else 0
                phases.append(ExecutionPhase(per_step, first_read+node.recurrent_bytes,
                    node.name+':first', ordered=True, read_bytes=first_read,
                    write_bytes=node.recurrent_bytes))
                if node.recurrent_steps > 1:
                    phases.append(ExecutionPhase(per_step, 2*node.recurrent_bytes,
                        node.name+':next', ordered=True, read_bytes=node.recurrent_bytes,
                        write_bytes=node.recurrent_bytes, iterations=node.recurrent_steps-1))
                phases.append(ExecutionPhase(0, writes-state_write, node.name+':outputs', 'write'))
            else:
                phases.append(ExecutionPhase(compute, reads+writes, node.name,
                    ordered=True, read_bytes=reads, write_bytes=writes))
        return phases

    def export(self):
        return {'name': self.name, 'schedule': [n.name for n in self.ordered_nodes()],
                'tensors': [dict(asdict(t), nbytes=t.nbytes) for t in self.tensors.values()],
                'nodes': [asdict(n) for n in self.nodes],
                'peak_activation_bytes': self.peak_activation_bytes(),
                'schedule_policy': ('serial topological; staged tiled kernels' if self.schedules
                                    else 'serial topological; whole-graph activation spill'),
                'kernel_schedules': {name: schedule.export() for name, schedule in self.schedules.items()},
                'metadata': self.metadata}


class StagedOperatorGraph(OperatorGraph):
    """Materialize operator boundaries in mapped HBM for a serial baseline."""
    def lower(self, rates, force_spill=False):
        return super().lower(rates,force_spill=True)

    def export(self):
        data = super().export()
        data['schedule_policy'] = 'serial topological; explicit HBM staging'
        return data

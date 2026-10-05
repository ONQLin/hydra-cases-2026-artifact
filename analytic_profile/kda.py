"""KDA core algorithms and explicit SRAM/HBM tile schedules.

Chunk work follows a triangular-solve formulation of the gated delta rule.
It is an analytical implementation choice, not a timing model of FLA's Triton
kernel. Outer projections are shared with the original KDA profile.
"""

from abc import ABC, abstractmethod
from dataclasses import asdict

from Sim.config.utils import get_all_subclasses
from Sim.entities.execution import ExecutionPhase
from Sim.entities.operator_graph import OperatorGraph


class BaseKDAProgram(ABC):
    def __init__(self, heads, key_dim, value_dim, tokens, initial_state=True):
        self.heads, self.key_dim, self.value_dim = heads, key_dim, value_dim
        self.tokens = tokens
        self.graph = OperatorGraph(self.get_name())
        g = self.graph
        for name, width in (('q', key_dim), ('k', key_dim), ('v', value_dim), ('g', key_dim)):
            g.tensor(name, (heads, tokens, width), storage='input')
        g.tensor('beta', (heads, tokens), storage='input')
        g.tensor('state_before', (heads if initial_state else 0, key_dim, value_dim), 4, 'state')
        g.tensor('state_after', (heads, key_dim, value_dim), 4, 'state')
        g.tensor('output', (heads, tokens, value_dim), storage='output')
        self.build()

    @staticmethod
    @abstractmethod
    def get_name():
        raise NotImplementedError

    @abstractmethod
    def build(self):
        raise NotImplementedError

    @classmethod
    def create_from_name(cls, name, **kwargs):
        for subtype in get_all_subclasses(cls):
            if subtype.get_name() == name:
                return subtype(**kwargs)
        raise ValueError(f'Unknown KDA core algorithm {name}.')

    @property
    def state_bytes(self):
        return self.heads*self.key_dim*self.value_dim*4

    @property
    def activation_bytes(self):
        return self.graph.peak_activation_bytes()

    @property
    def resident_bytes(self):
        # State is updated in place after all old-state consumers finish.
        return self.activation_bytes + self.state_bytes

    def operation_counts(self):
        return (sum(n.macs for n in self.graph.nodes), sum(n.element_ops for n in self.graph.nodes))

    def resident_compute_ns(self, rates):
        total = 0
        for node in self.graph.ordered_nodes():
            arithmetic = node.macs/rates.macs_per_ns + node.element_ops/rates.elements_per_ns
            traffic = sum(self.graph.tensors[key].nbytes for key in node.inputs+node.outputs)
            total += max(arithmetic, (traffic+2*node.workspace_bytes)/rates.sram_bytes_per_ns)
        return total

    def io_bytes(self):
        reads = sum(t.nbytes for t in self.graph.tensors.values() if t.storage == 'input')
        writes = self.graph.tensors['output'].nbytes
        return reads, writes


class RecurrentKDAProgram(BaseKDAProgram):
    @staticmethod
    def get_name():
        return 'recurrent'

    def build(self):
        if self.tokens != 1:
            raise ValueError('A tiled recurrent step consumes exactly one token.')
        h, k, v = self.heads, self.key_dim, self.value_dim
        self.graph.add('update_query', ['q', 'k', 'v', 'g', 'beta', 'state_before'],
                       ['output', 'state_after'], macs=3*h*k*v,
                       element_ops=h*(k*v+4*k+v))


class ChunkKDAProgram(BaseKDAProgram):
    @staticmethod
    def get_name():
        return 'chunk'

    def build(self):
        h, k, v, t = self.heads, self.key_dim, self.value_dim, self.tokens
        g = self.graph
        strict, causal = t*(t-1)//2, t*(t+1)//2
        g.tensor('gate_prefix', (h,t,k), 4)
        g.add('gate_prefix', ['g'], ['gate_prefix'], element_ops=h*t*k)
        # L_ij = beta_i * <k_i, exp(G_i-G_j)*k_j>, j<i.
        # Triangular arithmetic is explicit; buffers retain square storage.
        g.tensor('lower', (h,t,t), 4)
        g.add('key_products', ['k','gate_prefix','beta'], ['lower'],
              macs=h*strict*k, element_ops=h*strict*(3*k+1))
        g.tensor('rhs', (h,t,k+v), 4)
        g.add('scaled_rhs', ['k','v','gate_prefix','beta'], ['rhs'],
              element_ops=h*t*(3*k+v))
        # Forward substitution: (I+L)[W,U] = beta*[exp(G)*K,V].
        g.tensor('wu', (h,t,k+v), 4)
        g.add('triangular_solve', ['lower','rhs'], ['wu'], macs=h*strict*(k+v))
        g.tensor('residual', (h,t,v), 4)
        g.add('state_residual', ['wu','state_before'], ['residual'],
              macs=h*t*k*v, element_ops=h*t*v)
        g.tensor('causal_qk', (h,t,t), 4)
        g.add('query_products', ['q','k','gate_prefix'], ['causal_qk'],
              macs=h*causal*k, element_ops=h*causal*(3*k+1))
        g.tensor('prior_output', (h,t,v), 4)
        g.add('prior_output', ['q','gate_prefix','state_before'], ['prior_output'],
              macs=h*t*k*v, element_ops=3*h*t*k)
        g.add('output', ['causal_qk','residual','prior_output'], ['output'],
              macs=h*causal*v, element_ops=2*h*t*v)
        g.add('state_update', ['k','gate_prefix','residual','state_before'], ['state_after'],
              macs=h*t*k*v, element_ops=h*(k*v+3*t*k), dependencies=('output',))


class KDACoreSchedule:
    """Serial sequence/head/value tiles with ordered chunks inside each tile.

    K is kept whole; value columns are independent recurrence state columns.
    Q/K/decay are reread and chunk preparation is recomputed for each value tile.
    A resident state tile is read once and written once per call. Otherwise all
    program tensors are materialized in HBM and dependencies order every access.
    """
    def __init__(self, config, batch_size, tokens, stage):
        self.config, self.batch_size, self.tokens, self.stage = config, batch_size, tokens, stage
        # Decode uses the one-token recurrence even when prefill selects chunks.
        self.algorithm = config.algorithm if stage == 'prefill' else 'recurrent'
        self.chunk_size = min(tokens, config.chunk_size) if self.algorithm == 'chunk' else 1
        self._programs = {}

    def tiles(self):
        cfg = self.config
        value_step = cfg.value_tile_size or cfg.head_dim
        for head in range(0, cfg.num_heads, cfg.head_tile_size):
            for value in range(0, cfg.head_dim, value_step):
                yield head, min(cfg.head_tile_size, cfg.num_heads-head), value, min(value_step, cfg.head_dim-value)

    def lengths(self):
        count, tail = divmod(self.tokens, self.chunk_size)
        yield self.chunk_size, count
        if tail:
            yield tail, 1

    def program(self, heads, values, tokens, initial_state=True):
        key = heads, values, tokens, initial_state
        if key not in self._programs:
            self._programs[key] = BaseKDAProgram.create_from_name(self.algorithm, heads=heads,
                key_dim=self.config.head_dim, value_dim=values, tokens=tokens, initial_state=initial_state)
        return self._programs[key]

    def operation_counts(self):
        macs = elements = 0
        for _, heads, _, values in self.tiles():
            for tokens, count in self.lengths():
                m, e = self.program(heads, values, tokens).operation_counts()
                macs += self.batch_size*count*m
                elements += self.batch_size*count*e
        return macs, elements

    def peak_scratch_bytes(self):
        return max(self.program(h,v,t).resident_bytes
                   for _,h,_,v in self.tiles() for t,_ in self.lengths())

    def lower(self, rates):
        phases = []
        for batch in range(self.batch_size):
            for head, heads, value, values in self.tiles():
                label = f'kda_{self.algorithm}:b{batch}:h{head}:v{value}'
                largest = self.program(heads, values, self.chunk_size)
                resident = largest.resident_bytes <= rates.sram_capacity_bytes
                if resident:
                    initial = largest.state_bytes if self.stage == 'decode' else 0
                    if initial:
                        phases.append(ExecutionPhase(0, initial, label+':state_read'))
                    for tokens, count in self.lengths():
                        program = self.program(heads, values, tokens)
                        reads, writes = program.io_bytes()
                        phases.append(ExecutionPhase(program.resident_compute_ns(rates), reads+writes,
                            label+f':chunk{tokens}', ordered=True, read_bytes=reads,
                            write_bytes=writes, iterations=count))
                    phases.append(ExecutionPhase(0, largest.state_bytes, label+':state_write', 'write'))
                else:
                    offset = 0
                    for tokens, count in self.lengths():
                        for _ in range(count):
                            program = self.program(heads, values, tokens,
                                initial_state=self.stage == 'decode' or offset > 0)
                            for phase in program.graph.lower(rates, force_spill=True):
                                data = asdict(phase)
                                data['name'] = label+f':t{offset}:'+phase.name
                                phases.append(ExecutionPhase(**data))
                            offset += tokens
        return phases

    def export(self):
        templates = {}
        for _,h,_,v in self.tiles():
            for t,_ in self.lengths():
                program = self.program(h,v,t)
                templates[f'h{h}_v{v}_t{t}'] = dict(resident_bytes=program.resident_bytes,
                    state_bytes=program.state_bytes, graph=program.graph.export())
        return dict(algorithm=self.algorithm, batch_size=self.batch_size, tokens=self.tokens,
                    chunk_size=self.chunk_size, head_tile_size=self.config.head_tile_size,
                    value_tile_size=self.config.value_tile_size or self.config.head_dim,
                    state_binding='Each chunk consumes the previous chunk state; disjoint sequence/head/value tiles.',
                    state_writeback='Once per resident tile; after every chunk when spilled.',
                    templates=templates)

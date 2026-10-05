"""Serial SRAM-resident online-softmax tiles for MLA.

This is an explicit analytical schedule, not a calibrated FlashAttention kernel.
Q/K/V and output use the A8 surrogate; logits, probabilities and accumulators
remain FP32 inside the core. No inter-head latent-cache reuse is assumed.
"""

from functools import cached_property

from Sim.entities.execution import ExecutionPhase
from Sim.entities.operator_graph import OperatorGraph


class OnlineSoftmaxProgram:
    """One rectangular QK/PV tile, including masked diagonal arithmetic.

    Running max, denominator and numerator have separate old/new buffers.
    The next tile consumes the new buffers only after the current tile finishes.
    """
    def __init__(self, heads, queries, keys, key_dim, value_dim):
        self.heads, self.queries, self.keys = heads, queries, keys
        self.key_dim, self.value_dim = key_dim, value_dim
        h, q, k, d, v = heads, queries, keys, key_dim, value_dim
        g = self.graph = OperatorGraph('online-softmax-tile')
        for name, shape in (('q', (h,q,d)), ('k', (h,k,d)), ('v', (h,k,v))):
            g.tensor(name, shape, storage='input')
        for suffix in ('before', 'after'):
            for name, width in (('max',1), ('sum',1), ('acc',v)):
                g.tensor(name+'_'+suffix, (h,q,width), 4, 'state')
        for name, shape in (('scores',(h,q,k)), ('alpha',(h,q,1)),
                            ('probabilities',(h,q,k)), ('tile_sum',(h,q,1)),
                            ('scaled_acc',(h,q,v)), ('weighted',(h,q,v))):
            g.tensor(name, shape, 4)
        g.add('scores', ['q','k'], ['scores'], macs=h*q*k*d, element_ops=2*h*q*k)
        g.add('running_max', ['scores','max_before'], ['max_after'], element_ops=h*q*(k+1))
        g.add('rescale', ['max_before','max_after'], ['alpha'], element_ops=2*h*q)
        g.add('probabilities', ['scores','max_after'], ['probabilities'], element_ops=2*h*q*k)
        g.add('tile_sum', ['probabilities'], ['tile_sum'], element_ops=h*q*k)
        g.add('running_sum', ['sum_before','alpha','tile_sum'], ['sum_after'], element_ops=2*h*q)
        g.add('scaled_acc', ['acc_before','alpha'], ['scaled_acc'], element_ops=h*q*v)
        g.add('weighted', ['probabilities','v'], ['weighted'], macs=h*q*k*v)
        g.add('running_acc', ['scaled_acc','weighted'], ['acc_after'], element_ops=h*q*v)
        self._operation_counts = (sum(n.macs for n in g.nodes), sum(n.element_ops for n in g.nodes))
        self._compute_cache = {}

    @cached_property
    def resident_bytes(self):
        # Both state versions are private scratch, not persistent KV cache.
        # Q stays live across key tiles even after its local QK consumer.
        return self.graph.peak_activation_bytes(retained_inputs=('q',)) + 8*self.heads*self.queries*(self.value_dim+2)

    def operation_counts(self):
        return self._operation_counts

    def compute_ns(self, rates):
        # Programs are immutable after construction. A schedule reuses the same
        # tile shape many times; only throughput rates affect its compute cost.
        key = (rates.macs_per_ns,rates.elements_per_ns,rates.sram_bytes_per_ns)
        if key in self._compute_cache:
            return self._compute_cache[key]
        total = 0
        for node in self.graph.ordered_nodes():
            traffic = sum(self.graph.tensors[t].nbytes for t in node.inputs+node.outputs)
            total += max(node.macs/rates.macs_per_ns+node.element_ops/rates.elements_per_ns,
                         traffic/rates.sram_bytes_per_ns)
        self._compute_cache[key] = total
        return total


class MLAStreamingSchedule:
    """Query-major/head-major tiles with serial cache reads and online state.

    Q is loaded once per query tile. Cache is reread for each query/head tile;
    absorbed C channels supply both K and V and are fetched once per key tile.
    Fully future key tiles are skipped; the remaining rectangles include masked
    positions. An oversized tile fails explicitly instead of inventing a spill.
    """
    def __init__(self, config, batch_size, tokens, context):
        self.config, self.batch_size = config, batch_size
        self.tokens, self.context = tokens, context
        self.key_dim = (config.kv_lora_rank if config.cache_layout == 'absorbed'
                        else config.qk_nope_head_dim) + config.qk_rope_head_dim
        self.value_dim = config.kv_lora_rank if config.cache_layout == 'absorbed' else config.v_head_dim
        self._programs = {}

    def query_tiles(self):
        cfg = self.config
        for head in range(0, cfg.num_heads, cfg.head_tile_size):
            for query in range(0, self.tokens, cfg.query_tile_size):
                yield head, min(cfg.head_tile_size, cfg.num_heads-head), query, min(cfg.query_tile_size, self.tokens-query)

    def key_tiles(self, query, queries):
        end = self.context-self.tokens+query+queries
        for key in range(0, end, self.config.key_tile_size):
            yield key, min(self.config.key_tile_size, end-key)

    def program(self, heads, queries, keys):
        signature = heads, queries, keys
        if signature not in self._programs:
            self._programs[signature] = OnlineSoftmaxProgram(heads, queries, keys, self.key_dim, self.value_dim)
        return self._programs[signature]

    def operation_counts(self):
        macs = elements = 0
        for _,h,q,n in self.query_tiles():
            for _,k in self.key_tiles(q,n):
                m,e = self.program(h,n,k).operation_counts()
                macs += m
                elements += e
            elements += h*n*self.value_dim  # Final normalization, before A8 output.
        return self.batch_size*macs, self.batch_size*elements

    def peak_scratch_bytes(self):
        # Maximal tile dimensions occur in the final full query tile, or tail.
        return max(self.program(h,n,k).resident_bytes for _,h,q,n in self.query_tiles()
                   for _,k in self.key_tiles(q,n))

    def lower(self, rates):
        required = self.peak_scratch_bytes()
        if required > rates.sram_capacity_bytes:
            raise ValueError(f'MLA streaming tile requires {required} SRAM bytes; '
                             f'available {rates.sram_capacity_bytes}. Reduce tile sizes.')
        phases = []
        for batch in range(self.batch_size):
            for head,h,q,n in self.query_tiles():
                label = f'mla_stream:b{batch}:h{head}:q{q}'
                phases.append(ExecutionPhase(0,h*n*self.key_dim,label+':query'))
                for key,k in self.key_tiles(q,n):
                    program = self.program(h,n,k)
                    reads = (k*self.key_dim if self.config.cache_layout == 'absorbed'
                             else h*k*(self.key_dim+self.value_dim))
                    phases.append(ExecutionPhase(program.compute_ns(rates),reads,label+f':k{key}',
                                                 ordered=True,read_bytes=reads))
                outputs = h*n*self.value_dim
                compute = max(outputs/rates.elements_per_ns,
                              (5*outputs+4*h*n)/rates.sram_bytes_per_ns)
                phases.append(ExecutionPhase(compute,outputs,label+':normalize',ordered=True,write_bytes=outputs))
        return phases

    def export(self):
        return dict(algorithm='streaming',tokens=self.tokens,context=self.context,batch_size=self.batch_size,
                    cache_layout=self.config.cache_layout,query_tile_size=self.config.query_tile_size,
                    key_tile_size=self.config.key_tile_size,head_tile_size=self.config.head_tile_size,
                    peak_scratch_bytes=self.peak_scratch_bytes(),
                    state_binding='Initialize max=-inf, sum=0, acc=0 per query tile; carry serially across key tiles.',
                    precision='A8 Q/K/V/output; FP32 logits, probabilities, running max/sum/acc.',
                    traffic_policy='HBM outer staging; query read once; cache reread per query/head tile; no double buffering.')

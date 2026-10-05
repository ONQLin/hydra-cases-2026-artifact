"""Dependency-aware KDA and MLA operator rooflines; no full Kimi/DeepSeek model.

W8/A8/cache8 storage is a HYDRA performance surrogate, with FP32 recurrent
state/logits. Nonlinear operation rates and accelerator mappings need calibration.
"""

from abc import abstractmethod
from dataclasses import fields
import math

import Sim.common as common
from Sim.config.attention_operator_config import KDAConfig, MLAConfig
from Sim.entities.operator_graph import OperatorGraph
from Sim.metrics.monitor import analytics
from analytic_profile.modern import AcceleratorRates, BaseOperatorProfile


class GraphOperatorProfile(BaseOperatorProfile):
    @classmethod
    def get_execution_family(cls, config):
        return cls.execution_family

    @abstractmethod
    def graph(self, **kwargs):
        raise NotImplementedError

    @staticmethod
    def validate_call(bs, batch_size, L_seq, stage):
        if any(type(v) is not int or v <= 0 for v in (bs, batch_size, L_seq)):
            raise ValueError('Token, batch and context lengths must be positive integers.')
        if stage not in ('prefill', 'decode') or bs != (L_seq if stage == 'prefill' else 1):
            raise ValueError('Only full fresh prefill or one-token decode is supported.')

    def work(self, **kwargs):
        return self.graph(**kwargs).ordered_nodes()

    def profile(self, logic_name, ext_bw, **kwargs):
        if common.ByteperParam != 1 or ext_bw <= 0:
            raise ValueError('Graph profiles require positive bandwidth and the one-byte contract.')
        graph = self.graph(**kwargs)
        phases = graph.lower(AcceleratorRates.from_name(logic_name))
        total = sum(phase.latency_ns(ext_bw) for phase in phases)
        traffic = sum(phase.memory_bytes*phase.iterations for phase in phases)
        arithmetic = sum(phase.compute_ns*phase.iterations for phase in phases)
        return analytics(math.ceil(total), math.ceil(traffic/ext_bw), traffic,
                         [graph.export()], 100*arithmetic/total if total else 0, 0, phases)

    @staticmethod
    def config(config_class, values):
        return config_class(**{f.name: values[f.name] for f in fields(config_class) if f.name in values})


class AttentionGraphBuilder:
    """Small graph construction API; tensor edges determine dependencies."""

    def __init__(self, name, tokens, hidden):
        self.graph = OperatorGraph(name)
        self.tokens = tokens
        self.graph.tensor('input', (tokens, hidden), storage='input')

    def project(self, name, source, in_dim, out_dim):
        g = self.graph
        weight = g.tensor(name+'.weight', (in_dim, out_dim), storage='weight')
        out = g.tensor(name, (self.tokens, out_dim))
        return g.add(name, [source, weight], [out], macs=self.tokens*in_dim*out_dim)

    def transform(self, name, sources, width, element_ops, weight_count=0, dtype=1):
        g = self.graph
        sources = list(sources)
        if weight_count:
            sources.append(g.tensor(name+'.weight', (weight_count,), storage='weight'))
        out = g.tensor(name, (self.tokens, width), dtype)
        return g.add(name, sources, [out], element_ops=element_ops)

    def output(self, source, width):
        out = self.graph.tensor('output', (self.tokens, width), storage='output')
        self.graph.add('output_commit', [source], [out])


class KDAProfile(GraphOperatorProfile):
    config_type = KDAConfig
    execution_family = 'recurrent'

    @staticmethod
    def get_name():
        return 'ModernKDA'

    def graph(self, bs, batch_size, L_seq, stage, **kwargs):
        self.validate_call(bs, batch_size, L_seq, stage)
        cfg = self.config(KDAConfig, kwargs)
        d, h, k, n = cfg.embedding_dim, cfg.num_heads, cfg.head_dim, bs*batch_size
        width, conv = h*k, cfg.conv_kernel_size
        builder = AttentionGraphBuilder('KDA/'+cfg.algorithm+('-tiled-v2' if cfg.tiled else '-v1'), n, d)
        g = builder.graph
        convolved = []
        for component in ('q', 'k', 'v'):
            raw = builder.project(component+'_projection', 'input', d, width)
            past = g.tensor(component+'_conv_before', (batch_size if stage == 'decode' else 0, width, conv), storage='state')
            saved = g.tensor(component+'_conv_after', (batch_size, width, conv), storage='state')
            weight = g.tensor(component+'_conv.weight', (width, conv), storage='weight')
            out = g.tensor(component+'_conv', (n, width))
            g.add(component+'_conv', [raw, past, weight], [out, saved],
                  macs=n*width*conv, element_ops=4*n*width)
            convolved.append(out)
        q = builder.transform('q_normalize', [convolved[0]], width, 5*n*width)
        key = builder.transform('k_normalize', [convolved[1]], width, 5*n*width)
        f_a = builder.project('decay_down', 'input', d, k)
        f_b = builder.project('decay_up', f_a, k, width)
        decay = builder.transform('channel_decay', [f_b], width, 8*n*width, h+width)
        beta_raw = builder.project('beta_projection', 'input', d, h)
        beta = builder.transform('beta_sigmoid', [beta_raw], h, 4*n*h)
        before = g.tensor('matrix_before', (batch_size if stage == 'decode' else 0, h, k, k), 4, 'state')
        after = g.tensor('matrix_after', (batch_size, h, k, k), 4, 'state')
        recurrent = g.tensor('recurrent_output', (n, width))
        # Recurrence is S_t = decay(S_{t-1}) + beta*k*(v-k^T*S), o=q^T*S_t.
        # Three matrix/vector MAC groups; channel decay and vector work explicit.
        if cfg.tiled:
            from analytic_profile.kda import KDACoreSchedule
            schedule = KDACoreSchedule(cfg, batch_size, bs, stage)
            macs, elements = schedule.operation_counts()
            g.add('delta_core', [q, key, convolved[2], decay, beta, before], [recurrent, after],
                  macs=macs, element_ops=elements, workspace_bytes=schedule.peak_scratch_bytes())
            g.bind_schedule('delta_core', schedule)
        else:
            g.add('delta_recurrence', [q, key, convolved[2], decay, beta, before], [recurrent, after],
                  macs=3*n*h*k*k, element_ops=n*(h*k*k+3*width),
                  recurrent_bytes=batch_size*h*k*k*4, recurrent_steps=bs)
        gate_a = builder.project('output_gate_down', 'input', d, k)
        gate = builder.project('output_gate_up', gate_a, k, width)
        gated = builder.transform('gated_rms_norm', [recurrent, gate], width, 9*n*width, k)
        out = builder.project('output_projection', gated, width, d)
        builder.output(out, d)
        return g


class MLAProfile(GraphOperatorProfile):
    config_type = MLAConfig
    execution_family = 'attention'

    @staticmethod
    def get_name():
        return 'ModernMLA'

    def graph(self, bs, batch_size, L_seq, stage, **kwargs):
        self.validate_call(bs, batch_size, L_seq, stage)
        cfg = self.config(MLAConfig, kwargs)
        d, h, c = cfg.embedding_dim, cfg.num_heads, cfg.kv_lora_rank
        p, r, v, n = cfg.qk_nope_head_dim, cfg.qk_rope_head_dim, cfg.v_head_dim, bs*batch_size
        builder = AttentionGraphBuilder('MLA/'+cfg.cache_layout+'/'+cfg.algorithm+'-v2', n, d)
        g = builder.graph
        query_input, query_dim = 'input', d
        if cfg.q_lora_rank:
            query_dim = cfg.q_lora_rank
            compressed_q = builder.project('query_down', 'input', d, query_dim)
            query_input = builder.transform('query_norm', [compressed_q], query_dim, 5*n*query_dim, query_dim)
        query = builder.project('query_projection', query_input, query_dim, h*(p+r))
        kv_raw = builder.project('kv_down', 'input', d, c+r)
        # Normalization affects latent channels only; positional channels pass through.
        kv = builder.transform('kv_norm', [kv_raw], c+r, 5*n*c, c)
        if cfg.rope_enabled and r:
            query = builder.transform('query_rope', [query], h*(p+r), 3*n*h*r)
            kv = builder.transform('key_rope', [kv], c+r, 3*n*r)
        if cfg.cache_layout == 'expanded':
            weight = g.tensor('kv_up.weight', (c, h*(p+v)), storage='weight')
            current = g.tensor('kv_expanded', (n, h*(p+r+v)))
            g.add('kv_up', [kv, weight], [current], macs=n*c*h*(p+v))
            score_dim, value_dim = p+r, v
        else:
            weight = g.tensor('key_absorb.weight', (h, p, c), storage='weight')
            absorbed = g.tensor('query_absorbed', (n, h*(c+r)))
            g.add('key_absorb', [query, weight], [absorbed], macs=n*h*p*c)
            query, current = absorbed, kv
            score_dim, value_dim = c+r, c
        if cfg.cache_layout == 'absorbed':
            key_width, value_width = r, c
        else:
            key_width, value_width = h*(p+r), h*v
        old_key = g.tensor('key_before', (batch_size, L_seq-bs, key_width), storage='state')
        old_value = g.tensor('value_before', (batch_size, L_seq-bs, value_width), storage='state')
        new_key = g.tensor('key_append', (n, key_width), storage='state')
        new_value = g.tensor('value_append', (n, value_width), storage='state')
        g.add('cache_commit', [current], [new_key, new_value])
        score_inputs = [query, old_key, new_key]
        if cfg.cache_layout == 'absorbed':
            score_inputs += [old_value, new_value]  # Latent cache supplies both K and V.
        attended = g.tensor('attention_values', (n, h*value_dim))
        if cfg.algorithm == 'streaming':
            from analytic_profile.mla import MLAStreamingSchedule
            schedule = MLAStreamingSchedule(cfg, batch_size, bs, L_seq)
            macs, elements = schedule.operation_counts()
            g.add('attention_stream', list(dict.fromkeys(score_inputs+[old_value,new_value])),
                  [attended], macs=macs, element_ops=elements,
                  workspace_bytes=schedule.peak_scratch_bytes())
            g.bind_schedule('attention_stream', schedule)
        else:
            # This is a materialized score implementation. The buffer is quadratic
            # in prefill length. Dense QK/AV evaluate the masked triangle as well.
            scores = g.tensor('scores', (batch_size, h, bs, L_seq), 4)
            pairs = bs*L_seq
            g.add('attention_scores', score_inputs, [scores],
                  macs=batch_size*h*pairs*score_dim)
            normalized = g.tensor('softmax_fp32', (batch_size, h, bs, L_seq), 4)
            g.add('softmax', [scores], [normalized], element_ops=5*batch_size*h*pairs)
            probs = g.tensor('probabilities', (batch_size, h, bs, L_seq))
            g.add('probability_cast', [normalized], [probs], element_ops=batch_size*h*pairs)
            g.add('attention_values', [probs, old_value, new_value], [attended],
                  macs=batch_size*h*pairs*value_dim)
        if cfg.cache_layout == 'absorbed':
            weight = g.tensor('value_expand.weight', (h, v, c), storage='weight')
            expanded = g.tensor('expanded_output', (n, h*v))
            g.add('value_expand', [attended, weight], [expanded], macs=n*h*v*c)
            attended = expanded
        out = builder.project('output_projection', attended, h*v, d)
        builder.output(out, d)
        return g

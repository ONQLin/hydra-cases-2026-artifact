"""Colocated MoE and composed decoder graphs with explicit route provenance."""

from Sim.config.moe_config import DecoderLayerConfig, MoEConfig, GroupedMoEConfig
from Sim.config.modern_model_config import SwiGLUConfig
from Sim.entities.expert_routing import BaseExpertRouting
from Sim.entities.operator_graph import OperatorGraph, StagedOperatorGraph
from analytic_profile.attention import GraphOperatorProfile
from analytic_profile.modern import BaseOperatorProfile


class FeedForwardGraphBuilder:
    def __init__(self, graph, hidden):
        self.graph, self.hidden = graph, hidden

    def weights(self, prefix, width):
        g, d = self.graph, self.hidden
        g.tensor(prefix+'.up_gate.weight',(d,2*width),storage='weight')
        g.tensor(prefix+'.down.weight',(width,d),storage='weight')

    def expert(self, prefix, source, tokens, width):
        g, d = self.graph, self.hidden
        up = g.tensor(prefix+'.up_gate',(tokens,2*width))
        g.add(prefix+'.up_gate',[source,prefix+'.up_gate.weight'],[up],macs=2*tokens*d*width)
        activated = g.tensor(prefix+'.silu',(tokens,width))
        g.add(prefix+'.silu',[up],[activated],element_ops=5*tokens*width)
        out = g.tensor(prefix+'.output',(tokens,d))
        g.add(prefix+'.down',[activated,prefix+'.down.weight'],[out],macs=tokens*d*width)
        return out


class DenseFFNProfile(GraphOperatorProfile):
    config_type = SwiGLUConfig
    execution_family = 'attention'

    @staticmethod
    def get_name():
        return 'ModernDenseFFN'

    def graph(self, bs, batch_size, L_seq, stage, **kwargs):
        self.validate_call(bs,batch_size,L_seq,stage)
        cfg = self.config(SwiGLUConfig,kwargs)
        g = OperatorGraph('SwiGLU/graph-v1')
        n, d = bs*batch_size, cfg.embedding_dim
        g.tensor('input',(n,d),storage='input')
        builder = FeedForwardGraphBuilder(g,d)
        builder.weights('ffn',cfg.intermediate_dim)
        out = builder.expert('ffn','input',n,cfg.intermediate_dim)
        g.tensor('output',(n,d),storage='output')
        g.add('commit',[out],['output'])
        return g


class MoEProfile(GraphOperatorProfile):
    config_type = MoEConfig
    execution_family = 'attention'

    @staticmethod
    def get_name():
        return 'ModernMoE'

    def graph(self, bs, batch_size, L_seq, stage, **kwargs):
        self.validate_call(bs,batch_size,L_seq,stage)
        cfg = self.config(self.config_type,kwargs)
        routing = BaseExpertRouting.create_from_name(cfg.routing_policy)
        plan = routing.route(cfg,bs,batch_size,0 if stage=='prefill' else L_seq-1)
        g = StagedOperatorGraph('MoE/colocated-v1')
        n,d,e,k = bs*batch_size,cfg.embedding_dim,cfg.num_experts,cfg.top_k
        g.metadata = dict(routing=plan.export(), expert_placement=cfg.expert_placement,
                          routed_assignments=n*k, dispatch_bytes=n*k*d,
                          return_bytes=n*k*d, expert_execution='serial on the mapped block chiplet')
        g.tensor('input',(n,d),storage='input')
        self.build_router(g,cfg,n)
        builder = FeedForwardGraphBuilder(g,d)
        buffers = []
        for expert,count in enumerate(plan.counts):
            builder.weights(f'expert{expert}',cfg.intermediate_dim)
            if count:
                buffers.append(g.tensor(f'expert{expert}.input',(count,d)))
        g.add('dispatch',['input','route_ids'],buffers,element_ops=n*k)
        outputs = []
        for expert,count in enumerate(plan.counts):
            if count:
                outputs.append(builder.expert(f'expert{expert}',f'expert{expert}.input',count,cfg.intermediate_dim))
        g.tensor('weighted_returns',(n,k,d),4)
        g.add('gather_weight',outputs+['route_ids','route_weights'],['weighted_returns'],element_ops=n*k*d)
        g.tensor('routed_output',(n,d))
        g.add('combine',['weighted_returns'],['routed_output'],element_ops=n*k*d)
        final_inputs = ['routed_output']
        if cfg.shared_experts:
            width = cfg.shared_experts*cfg.intermediate_dim
            builder.weights('shared',width)
            final_inputs.append(builder.expert('shared','input',n,width))
        g.tensor('output',(n,d),storage='output')
        g.add('commit',final_inputs,['output'],element_ops=n*d if cfg.shared_experts else 0)
        return g


    def build_router(self, g, cfg, n):
        d,e,k = cfg.embedding_dim,cfg.num_experts,cfg.top_k
        g.tensor('router.weight',(d,e),storage='weight')
        g.tensor('router.bias',(e,),storage='weight')
        g.tensor('logits',(n,e),4)
        g.add('router',['input','router.weight'],['logits'],macs=n*d*e)
        g.tensor('route_ids',(n,k),4)
        g.tensor('route_weights',(n,k),4)
        # Sigmoid, correction bias, a conservative insertion top-k selection,
        # normalization and scaling. These are uncalibrated element-op costs.
        g.add('route',['logits','router.bias'],['route_ids','route_weights'],
              element_ops=n*e*(5+k)+3*n*k)


class GroupedMoEProfile(MoEProfile):
    config_type = GroupedMoEConfig

    @staticmethod
    def get_name():
        return 'ModernGroupedMoE'

    def build_router(self, g, cfg, n):
        d,e,k,groups = cfg.embedding_dim,cfg.num_experts,cfg.top_k,cfg.num_groups
        g.tensor('router.weight',(d,e),storage='weight')
        g.tensor('router.bias',(e,),storage='weight')
        for name,shape,dtype in (
            ('logits',(n,e),4),('scores',(n,e),4),('selection_scores',(n,e),4),
            ('group_scores',(n,groups),4),('group_ids',(n,cfg.top_k_groups),4),
            ('masked_scores',(n,e),4),('route_ids',(n,k),4),('selected_weights',(n,k),4),
            ('weight_sum',(n,1),4),('route_weights',(n,k),4)):
            g.tensor(name,shape,dtype)
        g.add('router',['input','router.weight'],['logits'],macs=n*d*e)
        g.add('router.sigmoid',['logits'],['scores'],element_ops=4*n*e)
        g.add('router.correction',['scores','router.bias'],['selection_scores'],element_ops=n*e)
        g.add('router.group_scores',['selection_scores'],['group_scores'],
              element_ops=n*(e*cfg.group_score_top_k+groups*(cfg.group_score_top_k-1)))
        g.add('router.group_topk',['group_scores'],['group_ids'],element_ops=n*groups*cfg.top_k_groups)
        g.add('router.group_mask',['selection_scores','group_ids'],['masked_scores'],element_ops=n*e)
        g.add('router.expert_topk',['masked_scores'],['route_ids'],element_ops=n*e*k)
        # The correction bias affects selection only. Combine weights come from
        # original sigmoid scores, then normalize over selected experts and scale.
        g.add('router.gather_weights',['scores','route_ids'],['selected_weights'],element_ops=n*k)
        g.add('router.weight_sum',['selected_weights'],['weight_sum'],element_ops=n*(k-1))
        g.add('router.normalize_scale',['selected_weights','weight_sum'],['route_weights'],element_ops=2*n*k)
        g.metadata['router'] = dict(algorithm='sigmoid-group-limited-topk',num_groups=groups,
            top_k_groups=cfg.top_k_groups,group_score_top_k=cfg.group_score_top_k,
            routed_scaling_factor=cfg.routed_scaling_factor,correction_bias='selection only',
            weights='normalize original selected sigmoid scores, then scale',
            cost_model='conservative insertion top-k comparisons; uncalibrated element rates')


class DecoderBlockProfile(GraphOperatorProfile):
    config_type = DecoderLayerConfig

    @staticmethod
    def get_name():
        return 'ModernDecoderBlock'

    @classmethod
    def get_execution_family(cls, config):
        return BaseOperatorProfile.find(config.attention.operator_name).get_execution_family(config.attention)

    def graph(self, bs, batch_size, L_seq, stage, attention, feed_forward, **kwargs):
        self.validate_call(bs,batch_size,L_seq,stage)
        cfg = DecoderLayerConfig(attention=attention,feed_forward=feed_forward)
        n,d = bs*batch_size,cfg.embedding_dim
        g = StagedOperatorGraph('Decoder/attention-ffn-v1')
        g.tensor('input',(n,d),storage='input')
        source, residual = 'input','input'
        for prefix,operator in (('attention',attention),('ffn',feed_forward)):
            norm = g.tensor(prefix+'.norm',(n,d))
            weight = g.tensor(prefix+'.norm.weight',(d,),storage='weight')
            g.add(prefix+'.norm',[source,weight],[norm],element_ops=5*n*d)
            profile = BaseOperatorProfile.find(operator.operator_name)
            if profile is None:
                raise ValueError(f'Unknown decoder operator {operator.operator_name}.')
            subgraph = profile().graph(**dict(vars(operator),bs=bs,batch_size=batch_size,L_seq=L_seq,stage=stage))
            out = g.include(prefix+'.',subgraph,norm)
            result = 'output' if prefix=='ffn' else 'attention_residual'
            g.tensor(result,(n,d),storage='output' if result=='output' else 'activation')
            g.add(prefix+'.residual',[residual,out],[result],element_ops=n*d)
            source = residual = result
        return g

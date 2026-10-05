#!/usr/bin/env python3
"""Compare KDA schedules and MoE route scenarios under declared synthetic rates."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from Sim.config.attention_operator_config import KDAConfig
from Sim.config.moe_config import MoEConfig
from Sim.config.model_config import BaseModelConfig
from analytic_profile.attention import KDAProfile
from analytic_profile.moe import MoEProfile
from analytic_profile.modern import AcceleratorRates


class KimiOperatorValidation:
    def __init__(self, output_dir):
        self.output_dir = output_dir

    @staticmethod
    def record(label,config,profile,stage,length,rates):
        graph = profile.graph(**dict(vars(config),bs=length if stage=='prefill' else 1,
                                      batch_size=1,L_seq=length,stage=stage))
        phases = graph.lower(rates)
        weights = sum(t.nbytes for t in graph.tensors.values() if t.storage=='weight')
        if weights != config.parameter_count:
            raise RuntimeError('Graph weights disagree with resident parameter count.')
        used = {key for node in graph.nodes for key in node.inputs if graph.tensors[key].storage=='weight'}
        return dict(case=label,config=asdict(config),stage=stage,context_tokens=length,
                    resident_weight_bytes=weights,active_weight_bytes=sum(graph.tensors[k].nbytes for k in used),
                    peak_activation_bytes=graph.peak_activation_bytes(),
                    macs=sum(n.macs for n in graph.nodes),element_ops=sum(n.element_ops for n in graph.nodes),
                    hbm_read_bytes=sum((p.read_bytes if p.ordered else p.memory_bytes if p.memory_direction=='read' else 0)*p.iterations for p in phases),
                    hbm_write_bytes=sum((p.write_bytes if p.ordered else p.memory_bytes if p.memory_direction=='write' else 0)*p.iterations for p in phases),
                    profile_latency_ns=sum(p.latency_ns(128) for p in phases),metadata=graph.metadata,
                    kernel_sram_bytes={name:schedule.peak_scratch_bytes() for name,schedule in graph.schedules.items()})

    def run(self):
        if self.output_dir.exists():
            raise ValueError('Use a fresh profile output directory.')
        self.output_dir.mkdir(parents=True)
        rates = AcceleratorRates(256,16,64,1024*1024)
        results = []
        for name,config in [('recurrent_whole',KDAConfig()),
                            ('recurrent_tiled',KDAConfig(state_mapping='tiled')),
                            ('chunk64_tiled',KDAConfig(algorithm='chunk',chunk_size=64))]:
            for stage,length in (('prefill',32),('prefill',128),('prefill',1024),('decode',4096)):
                results.append(self.record(name,config,KDAProfile(),stage,length,rates))
        for routing in ('cyclic','hotspot'):
            for stage,length in (('prefill',32),('prefill',128),('decode',4096)):
                results.append(self.record('moe_'+routing,MoEConfig(routing_policy=routing),MoEProfile(),stage,length,rates))
        model = BaseModelConfig.create_from_name('kimi-linear-48b-a3b-text')
        parameter_bytes = sum(model.hybrid_blocks[i].parameter_count for i in model.block_type_sequence)
        files = ['analytic_profile/kda.py','analytic_profile/moe.py','analytic_profile/attention.py',
                 'analytic_profile/modern.py','Sim/config/moe_config.py','Sim/config/kimi_model_config.py',
                 'Sim/config/attention_operator_config.py','Sim/config/modern_model_config.py',
                 'Sim/entities/operator_graph.py','Sim/entities/expert_routing.py','Sim/entities/execution.py']
        manifest = dict(scope='Analytical estimates at synthetic rates; not measured kernel latency.',
            rates=asdict(rates),hbm_bandwidth_gbps=128,batch_size=1,decoder_resident_weight_bytes=parameter_bytes,
            source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in files},results=results)
        (self.output_dir/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'Exported {len(results)} cases; decoder resident weights: {parameter_bytes} bytes.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    KimiOperatorValidation(parser.parse_args().output_dir).run()

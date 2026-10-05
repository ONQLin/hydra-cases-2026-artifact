#!/usr/bin/env python3
"""Compare MLA algorithms under explicitly declared synthetic compute rates."""

import argparse
from dataclasses import asdict, replace
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from Sim.config.attention_operator_config import MLAConfig
from analytic_profile.attention import MLAProfile
from analytic_profile.modern import AcceleratorRates


class MLAProfileValidation:
    def run(self, output):
        if output.exists():
            raise ValueError('Use a fresh output file.')
        rates = AcceleratorRates(256,16,64,1024*1024)
        configs = dict(kimi_expanded=MLAConfig(embedding_dim=2304,num_heads=32,q_lora_rank=0,
                       cache_layout='expanded',rope_enabled=False),deepseek_absorbed=MLAConfig())
        records = []
        for name,base in configs.items():
            for stage,length in (('prefill',64),('prefill',512),('prefill',1024),('prefill',4096),('decode',4096)):
                for algorithm in ('materialized','streaming'):
                    cfg = replace(base,algorithm=algorithm)
                    graph = MLAProfile().graph(**dict(vars(cfg),bs=length if stage=='prefill' else 1,
                                                      batch_size=1,L_seq=length,stage=stage))
                    phases = graph.lower(rates)
                    records.append(dict(model_shape=name,algorithm=algorithm,stage=stage,context=length,
                        macs=sum(n.macs for n in graph.nodes),element_ops=sum(n.element_ops for n in graph.nodes),
                        graph_peak_bytes=graph.peak_activation_bytes(),cache_bytes=length*cfg.cache_bytes_per_token,
                        scratch_bytes=max((s.peak_scratch_bytes() for s in graph.schedules.values()),default=0),
                        hbm_bytes=sum(p.memory_bytes*p.iterations for p in phases),
                        latency_ns=sum(p.latency_ns(128) for p in phases)))
        result = dict(rates=asdict(rates),hbm_gbps=128,batch_size=1,
            scope='Synthetic analytical costs; materialized A8 probabilities versus streaming FP32 probabilities; not calibrated kernel timings.',
            source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in
                           ('analytic_profile/attention.py','analytic_profile/mla.py','Sim/config/attention_operator_config.py')},
            records=records)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(f'Wrote {len(records)} operator cases to {output}.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    MLAProfileValidation().run(parser.parse_args().output)

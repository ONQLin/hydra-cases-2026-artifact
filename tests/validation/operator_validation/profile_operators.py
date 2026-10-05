#!/usr/bin/env python3
"""Export Kimi/DeepSeek-shaped operator graphs without claiming full model support."""

import argparse
from dataclasses import asdict
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from Sim.config.attention_operator_config import KDAConfig, MLAConfig
from analytic_profile.attention import KDAProfile, MLAProfile
from analytic_profile.modern import AcceleratorRates


class OperatorValidation:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def run(self):
        # Shapes follow the pinned model inventory; rate values here are a
        # declared synthetic control, not measured Kimi/DeepSeek hardware.
        rates = AcceleratorRates(256, 16, 64, 1024*1024)
        configurations = [
            ('kimi_kda_recurrent', KDAConfig(), KDAProfile()),
            ('kimi_mla_expanded_nope', MLAConfig(embedding_dim=2304, num_heads=32,
                q_lora_rank=0, cache_layout='expanded', rope_enabled=False), MLAProfile()),
            ('deepseek_mla_absorbed', MLAConfig(), MLAProfile()),
        ]
        records = []
        for label, config, profile in configurations:
            for stage, length in (('prefill', 32), ('prefill', 128), ('decode', 128), ('decode', 4096)):
                graph = profile.graph(**dict(vars(config), stage=stage, L_seq=length,
                                      bs=length if stage=='prefill' else 1, batch_size=1))
                phases = graph.lower(rates)
                weights = sum(t.nbytes for t in graph.tensors.values() if t.storage=='weight')
                if weights != config.parameter_count:
                    raise RuntimeError(f'{label}: operator weights disagree with config.')
                state = (config.state_size_bytes if isinstance(config,KDAConfig)
                         else length*config.cache_bytes_per_token)
                records.append(dict(operator=label, stage=stage, context_tokens=length,
                    parameter_bytes=weights, persistent_state_bytes=state,
                    peak_activation_bytes=graph.peak_activation_bytes(),
                    hbm_read_bytes=sum((p.read_bytes if p.ordered else p.memory_bytes
                        if p.memory_direction=='read' else 0)*p.iterations for p in phases),
                    hbm_write_bytes=sum((p.write_bytes if p.ordered else p.memory_bytes
                        if p.memory_direction=='write' else 0)*p.iterations for p in phases),
                    profile_latency_ns=sum(p.latency_ns(128) for p in phases)))
                path = self.output_dir / f'{label}_{stage}_{length}.json'
                path.write_text(json.dumps(dict(config=asdict(config),graph=graph.export(),
                                                phases=[asdict(p) for p in phases]),indent=2)+'\n')
        sources = ['Sim/entities/operator_graph.py','Sim/entities/execution.py',
                   'analytic_profile/attention.py','Sim/config/attention_operator_config.py']
        manifest = dict(scope='Operator-level analytical estimates; no weights or model execution.',
                        rates=asdict(rates), hbm_bandwidth_gbps=128, batch_size=1,
                        source_sha256={name:hashlib.sha256((ROOT/name).read_bytes()).hexdigest() for name in sources},
                        results=records)
        (self.output_dir/'summary.json').write_text(json.dumps(manifest,indent=2)+'\n')
        print(f'Exported {len(records)} operator cases to {self.output_dir}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir',type=Path,required=True)
    OperatorValidation(parser.parse_args().output_dir).run()

#!/usr/bin/env python3
"""Export decoder weight/state requirements without loading checkpoint weights."""

import argparse
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from Sim.config.model_config import BaseModelConfig


class DecoderFootprintReport:
    def run(self, output):
        if output.exists():
            raise ValueError('Use a fresh output file.')
        records=[]
        for name in ('kimi-linear-48b-a3b-text-streaming','deepseek-v3-text'):
            model=BaseModelConfig.create_from_name(name)
            blocks=[model.hybrid_blocks[i] for i in model.block_type_sequence]
            weights=sum(b.parameter_count for b in blocks)
            records.append(dict(model=name,layers=model.num_layers,weight_bytes=weights,
                state_bytes_by_context={str(length):sum(b.states*(length if b.cache_store else 1) for b in blocks)
                                        for length in (128,1024,4096)},
                fits_original_16_hbm_weights=weights <= 16*16*1024**3,
                block_types=[dict(index=i,count=model.block_type_sequence.count(i),weight_bytes=b.parameter_count,
                                  cache_bytes_per_token=b.states if b.cache_store else None,
                                  fixed_state_bytes=None if b.cache_store else b.states)
                             for i,b in enumerate(model.hybrid_blocks)]))
        result=dict(scope='W8 decoder weights and persistent state only; activation/workspace and hardware bandwidth are separate.',
            original_hbm_capacity_bytes=16*16*1024**3,records=records,
            source_sha256={str(path):hashlib.sha256((ROOT/path).read_bytes()).hexdigest() for path in
                (Path('Sim/config/deepseek_model_config.py'),Path('Sim/config/kimi_model_config.py'),
                 Path('Sim/config/moe_config.py'),Path('Sim/config/attention_operator_config.py'))})
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(records,indent=2))


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    DecoderFootprintReport().run(parser.parse_args().output)

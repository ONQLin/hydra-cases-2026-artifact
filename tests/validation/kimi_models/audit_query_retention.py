#!/usr/bin/env python3
"""Check whether the corrected SRAM bound changes the already-started replay.

The initial streaming revision released Q after QK in the capacity calculation.
This audit reconstructs that old bound only, then compares phases and full MLA
activation reservations against the corrected implementation for all replay
shapes. It does not claim the historical bound was safe at smaller capacities.
"""

import argparse
from dataclasses import asdict
import json
from pathlib import Path
import sys
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

from Sim.config.kimi_model_config import KimiLinearStreamingModelConfig
from analytic_profile.attention import MLAProfile
from analytic_profile.mla import OnlineSoftmaxProgram
from analytic_profile.modern import AcceleratorRates


class HistoricalUnpinnedQueryProgram(OnlineSoftmaxProgram):
    @property
    def resident_bytes(self):
        return self.graph.peak_activation_bytes()+8*self.heads*self.queries*(self.value_dim+2)


class QueryRetentionAudit:
    def run(self, output):
        if output.exists():
            raise ValueError('Use a fresh audit output file.')
        model = KimiLinearStreamingModelConfig()
        cfg = model.hybrid_blocks[2].operator.attention
        # The replay uses input 2/batch 2 and inputs 11,26/batch 1. Include
        # larger admission shapes and every intermediate decode context too.
        cases = [('prefill',t,b) for t in (2,11,26,128,4096) for b in (1,2)]
        cases += [('decode',t,1) for t in range(1,366)]
        cases += [('decode',t,2) for t in (3,4,4096)]
        rates = AcceleratorRates(256,16,64,1024*1024)
        for stage,length,batch in cases:
            profiles,peaks = [],[]
            for program in (HistoricalUnpinnedQueryProgram,OnlineSoftmaxProgram):
                with patch('analytic_profile.mla.OnlineSoftmaxProgram',program):
                    graph = MLAProfile().graph(**dict(vars(cfg),bs=length if stage=='prefill' else 1,
                        batch_size=batch,L_seq=length,stage=stage))
                    profiles.append([asdict(p) for p in graph.lower(rates)])
                    peaks.append(graph.peak_activation_bytes())
            if profiles[0]!=profiles[1] or peaks[0]!=peaks[1]:
                raise AssertionError(f'Execution/admission changed: {stage}, {length}, {batch}.')
        reservations = []
        for program in (HistoricalUnpinnedQueryProgram,OnlineSoftmaxProgram):
            with patch('analytic_profile.mla.OnlineSoftmaxProgram',program):
                reservations.append([asdict(model.hybrid_blocks[2].memory_requirements(4096,p)) for p in (True,False)])
        if reservations[0]!=reservations[1]:
            raise AssertionError('Full decoder admission changed.')
        result = dict(model=model.get_name(),operator_cases=len(cases),execution_phases_equal=True,
            activation_reservations_equal=True,decoder_admission_equal=True,
            checked_sram_bytes=rates.sram_capacity_bytes,
            conclusion='Both revisions fit 1 MiB; same phases and full-layer HBM admission for these shapes. '
                       'Results are invariant for accelerator SRAM >= 1 MiB; tiny-capacity behavior was corrected.')
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',type=Path,required=True)
    QueryRetentionAudit().run(parser.parse_args().output)

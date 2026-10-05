#!/usr/bin/env python3
"""Separate phase clock rounding from raw Packet/Garnet network timing."""

import argparse
from itertools import zip_longest
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))
from tests.validation.chipsim_validation.collect_checkpoints import CheckpointRun


class PhaseTimingAudit:
    def __init__(self, directory):
        self.directory = directory

    def phases(self, backend):
        run = CheckpointRun(self.directory/backend)
        runtime = 'packet_runtime' if backend=='hydra_packet' else 'chipsim_runtime'
        with (run.run_dir/runtime/'execution.jsonl').open() as stream:
            for line in stream:
                record = json.loads(line)
                if 'phases' not in record.get('result',{}):
                    continue  # Provider startup/summary records carry no execution phases.
                request = {k:v for k,v in record['request'].items() if k!='start_tick'}
                for phase in record['result']['phases']:
                    yield request,phase

    def run(self, output):
        if output.exists():
            raise ValueError('Use a fresh timing audit output.')
        packet,garnet = (CheckpointRun(self.directory/name) for name in ('hydra_packet','chipsim_contended'))
        packet.verify_inputs(garnet)
        for run in (packet,garnet):
            metrics = run.read('metrics.json')
            if not metrics or metrics['running_requests'] or metrics['pending_requests']:
                raise ValueError('Timing audit requires drained serving runs.')
        totals = {name:dict(phases=0,observed_us=0.0,unrounded_phase_us=0.0,rounding_us=0.0)
                  for name in ('hydra_packet','chipsim_contended')}
        changed = count = 0
        maximum = 0.0
        for left,right in zip_longest(self.phases('hydra_packet'),self.phases('chipsim_contended')):
            if left is None or right is None or left[0]!=right[0]:
                raise ValueError('Phase records differ in order or operation identity; cannot pair raw timings.')
            a,b = left[1],right[1]
            if (a['name'],a['memory_direction']) != (b['name'],b['memory_direction']):
                raise ValueError('Phase identities differ.')
            delta = abs(a['noi_us']-b['noi_us'])
            maximum = max(maximum,delta)
            changed += delta>1e-9
            count += 1
            for name,p in (('hydra_packet',a),('chipsim_contended',b)):
                unrounded = max(p['compute_us'],p['hbm_us'],p['noi_us'])
                excess = p['latency_us']-unrounded
                if not -1e-7 <= excess < 1+1e-7:
                    raise ValueError('Phase timing exceeds the documented one-microsecond rounding bound.')
                t = totals[name]
                t['phases'] += 1
                t['observed_us'] += p['latency_us']
                t['unrounded_phase_us'] += unrounded
                t['rounding_us'] += max(0,excess)
        result = dict(scope='Sum over all executed phases, including concurrent work; not critical-path TTFT attribution.',
                      paired_phases=count,phases_with_different_raw_network_time=changed,
                      max_raw_network_difference_us=maximum,backends=totals)
        output.parent.mkdir(parents=True,exist_ok=True)
        output.write_text(json.dumps(result,indent=2)+'\n')
        print(json.dumps(result,indent=2))
        return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--directory',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    args = parser.parse_args()
    PhaseTimingAudit(args.directory).run(args.output)

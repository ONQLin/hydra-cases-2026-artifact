#!/usr/bin/env python3
"""Replay a completed analytic fixture and verify request-memory reclamation."""

import argparse
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0,str(ROOT))

import tyro
import Sim.common as common
from Sim.config.sys_config import HPSim_Config
from Sim.metrics.monitor import request_counter
from Sim.simulator import Simulator


class MemoryLifecycleValidation:
    def __init__(self, command_path, output_dir):
        self.command = json.loads(command_path.read_text())
        self.output_dir = output_dir.resolve()

    def run(self):
        if self.output_dir.exists():
            raise ValueError('Use a fresh memory audit output directory.')
        if self.command[self.command.index('--simulator-backend')+1] != 'hydra_sim':
            raise ValueError('Memory audit replays the analytic backend only.')
        self.command[self.command.index('--metrics-config.output-dir')+1] = str(self.output_dir)
        config = tyro.cli(HPSim_Config,args=self.command[2:])
        simulator = Simulator(config)
        simulator.run()
        chiplets = []
        for chiplet in simulator.mem_sys._mem_chiplets:
            weights = sum(p.size for p in chiplet.content_params.get('weights',[]))
            retained = sum(item.size for key, contents in chiplet.content_params.items()
                           if isinstance(key, str) and key.startswith('prefix:') for item in contents)
            live = sum(len(contents) for key,contents in chiplet.content_params.items() if isinstance(key, int))
            chiplets.append(dict(chiplet=chiplet.chiplet_id,weights_mib=weights,
                retained_prefix_mib=retained,
                used_mib=chiplet.inuse_budget,reserved_mib=chiplet._current_allocated,live_request_objects=live))
        cache = getattr(simulator._request_generator, 'prefix_cache', None)
        cache_report = cache.snapshot() if cache else None
        result = dict(completed_requests=request_counter.completed_requests,
                      running_requests=request_counter.running_requests,pending_requests=request_counter.pending_requests,
                      outstanding_reservations=len(common.req_allocated_mems),chiplets=chiplets,
                      prefix_cache=cache_report)
        result['passed'] = (result['completed_requests']>0 and result['running_requests']==0
            and result['pending_requests']==0 and result['outstanding_reservations']==0
            and all(c['live_request_objects']==0 and abs(c['used_mib']-c['weights_mib']-c['retained_prefix_mib'])<1e-6
                    and abs(c['reserved_mib']-c['weights_mib']-c['retained_prefix_mib'])<1e-6 for c in chiplets)
            and (cache_report is None or (not cache_report['pinned_references']
                 and cache_report['resident_bytes'] <= cache_report['capacity_bytes']
                 and abs(sum(c['retained_prefix_mib'] for c in chiplets)*1024**2-cache_report['resident_bytes']) < 1)))
        if cache_report and 'host' in cache_report:
            host = cache_report['host']
            result['passed'] = (result['passed'] and not cache_report['pending_transfers']
                and not cache_report['restore_admission_leases'] and not host['pinned_references']
                and host['kv_used_bytes'] + host['tool_workspace_reserved_bytes'] <= host['capacity_bytes'])
        (self.output_dir/'memory_audit.json').write_text(json.dumps(result,indent=2)+'\n')
        if not result['passed']:
            raise RuntimeError('Memory did not drain to weights plus retained prefixes; inspect memory_audit.json.')
        print('Memory lifecycle audit passed.')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--command',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    args = parser.parse_args()
    MemoryLifecycleValidation(args.command,args.output_dir).run()

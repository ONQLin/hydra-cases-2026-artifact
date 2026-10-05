#!/usr/bin/env python3
"""Verify package-local mappings, exact payloads and producer/consumer causality."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT))

from tests.validation.chipsim_validation.collect_checkpoints import CheckpointRun
from Sim.config.model_config import BaseModelConfig


class PackageRunAudit:
    def __init__(self, directory):
        self.run = CheckpointRun(directory.resolve())
        if self.run.run_dir is None:
            raise ValueError(f'No completed setup in {directory}')

    def read_rows(self, name):
        return [json.loads(row) for row in (self.run.run_dir / name).read_text().splitlines()]

    @staticmethod
    def require(condition, message):
        if not condition:
            raise ValueError(message)

    def verify(self):
        setup = self.run.read('system_snapshot.json')
        config = self.run.read('config.json')
        metrics = self.run.read('metrics.json')
        fabric = self.run.read('package_network_summary.json')
        package = {n['id']: n['package_id'] for n in setup['chiplets']}
        owners = {int(k): v for k, v in setup['packages']['layer_packages'].items()}
        weights = {int(k): v for k, v in setup['weight_mapping'].items()}
        for source, destination in setup['links']:
            self.require(package[source] == package[destination], 'NoI edge crosses package boundary.')
        for phase, mapping in setup['task_mapping'].items():
            for layer, chiplet in mapping.items():
                layer = int(layer)
                self.require(package[chiplet] == package[weights[layer]] == owners[layer],
                             f'Nonlocal {phase} compute/weight mapping for layer {layer}.')
        transfers = self.read_rows('package_transfers.jsonl')
        blocks = self.read_rows('package_blocks.jsonl')
        with (self.run.run_dir / 'effective_workload.csv').open() as stream:
            workload = list(csv.DictReader(stream))
        request_ids = sorted({req for row in blocks for req in row['request_ids']})
        self.require(len(request_ids) == len(workload) == metrics['completed_requests'], 'Incomplete request set.')
        self.require(metrics['running_requests'] == metrics['pending_requests'] == metrics['preempted_requests'] == 0,
                     'Validation requires complete requests without preemption.')
        expected_tokens = sum(int(row['num_decode_tokens']) + 1 for row in workload)
        self.require(metrics['output_tokens'] == expected_tokens, 'Output count differs from input lengths.')
        model = BaseModelConfig.create_from_name(config['workload_config']['model'])
        checked_transfers = 0
        for request_id, lengths in zip(request_ids, workload):
            prefill, decode = (int(lengths[key]) for key in ('num_prefill_tokens', 'num_decode_tokens'))
            starts = [b for b in blocks if b['event'] == 'start' and request_id in b['request_ids']]
            ends = [b for b in blocks if b['event'] == 'complete' and request_id in b['request_ids']]
            expected_layers = list(range(model.num_layers)) * (1 + decode)
            self.require([b['block_id'] for b in starts] == expected_layers
                         and [b['block_id'] for b in ends] == expected_layers, 'Incomplete or reordered decoder execution.')
            moves = iter(row for row in transfers if row['request_id'] == request_id)
            for index, (start, end) in enumerate(zip(starts, ends)):
                layer = start['block_id']
                phase = 'prefill' if index < model.num_layers else 'decode'
                self.require(start['stage'] == end['stage'] == phase and start['tick'] <= end['tick'],
                             'Invalid block phase or completion time.')
                self.require(start['package'] == end['package'] == owners[layer], 'Execution outside owning package.')
                if index == 0:
                    continue
                previous = ends[index - 1]
                self.require(previous['tick'] <= start['tick'], 'Consumer started before producer completed.')
                if previous['package'] == start['package']:
                    continue
                row = next(moves, None)
                self.require(row is not None, 'Missing dependency transfer.')
                feedback = layer == 0
                producer = previous['block_id']
                block = model.hybrid_blocks[model.block_type_sequence[producer]]
                expected_bytes = (int(config['package_config']['token_bytes']) if feedback else
                                  block.output_mem * (prefill if previous['stage'] == 'prefill' else 1)
                                  * int(config['workload_config']['bytes_per_param']))
                self.require(row['bytes'] == expected_bytes,
                             f'Wrong payload in {self.run.directory}: request {request_id}, layer {layer}, '
                             f'expected {expected_bytes}, found {row["bytes"]}.')
                self.require(row['kind'] == ('token_feedback' if feedback else 'activation'), 'Wrong transfer kind.')
                self.require(row['producer_block'] == producer
                             and row['source_hbm'] == weights[producer] and row['destination_hbm'] == weights[layer],
                             'Transfer endpoints do not match the dependency.')
                self.require(row['source_package'] == previous['package'] and row['destination_package'] == start['package'],
                             'Transfer names incorrect packages.')
                self.require(previous['tick'] <= row['queued_tick'] <= row['start_tick'] <= row['completed_tick'] <= start['tick'],
                             'Consumer started before package transfer completed.')
                checked_transfers += 1
            self.require(next(moves, None) is None, 'Unexpected extra package transfer.')
        self.require(checked_transfers == len(transfers) == fabric['submitted_transfers'] == fabric['completed_transfers']
                     and fabric['inflight_at_cutoff'] == 0, 'Package transfers did not drain.')
        self.require(sum(row['bytes'] for row in transfers) == fabric['completed_bytes'], 'Fabric byte accounting mismatch.')
        for direction in ('source_package', 'destination_package'):
            for owner in set(owners.values()):
                rows = sorted((r for r in transfers if r[direction] == owner), key=lambda r: r['start_tick'])
                self.require(all(a['completed_tick'] <= b['start_tick'] for a, b in zip(rows, rows[1:])),
                             'Transfers overlap on a serialized package port.')
        return dict(passed=True, run_directory=str(self.run.run_dir),
                    checked_transfers=checked_transfers, checked_blocks=len(blocks) // 2,
                    package_config=config['package_config'], metrics=metrics, fabric=fabric,
                    process_wall_seconds=(json.loads(self.run.directory.joinpath('wall_time.json').read_text())['seconds']
                                          if self.run.directory.joinpath('wall_time.json').exists() else None),
                    evidence_sha256={name: hashlib.sha256((self.run.run_dir / name).read_bytes()).hexdigest()
                        for name in ('system_snapshot.json', 'effective_workload.csv', 'package_transfers.jsonl', 'package_blocks.jsonl')})


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', nargs='+', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    results = [PackageRunAudit(directory).verify() for directory in args.runs]
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(results, indent=2) + '\n')
    print(f'Passed package dependency/payload audits for {len(results)} runs.')

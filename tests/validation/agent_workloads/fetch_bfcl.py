#!/usr/bin/env python3
"""Fetch and hash-check the pinned archive; reproduce the two selected raw rows."""

from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import urllib.request

import tyro


@dataclass
class BFCLArchiveConfig:
    output_dir: Path


class BFCLArchive:
    def __init__(self, config):
        self.config = config
        self.source_dir = Path(__file__).with_name('real_bfcl')
        self.provenance = json.loads((self.source_dir/'provenance.json').read_text())

    def fetch(self):
        if self.config.output_dir.exists():
            raise ValueError('Choose a fresh archive output directory.')
        with urllib.request.urlopen(self.provenance['source_url'], timeout=60) as stream:
            data = stream.read()
        if hashlib.sha256(data).hexdigest() != self.provenance['source_sha256']:
            raise ValueError('Pinned BFCL archive hash mismatch.')
        selected = b''.join(line for line in data.splitlines(keepends=True)
                            if json.loads(line)['id'] in self.provenance['selected_ids'])
        if hashlib.sha256(selected).hexdigest() != self.provenance['selected_sha256']:
            raise ValueError('Selected BFCL rows differ from the curated input.')
        self.config.output_dir.mkdir(parents=True)
        (self.config.output_dir/'qwen_full_base.jsonl').write_bytes(data)
        (self.config.output_dir/'qwen3_8b_fc_base_0_1.jsonl').write_bytes(selected)
        (self.config.output_dir/'archive_audit.json').write_text(json.dumps(self.audit(data), indent=2)+'\n')
        print(f'Verified full archive and selected rows in {self.config.output_dir}')

    def audit(self, data):
        rows = [json.loads(line) for line in data.splitlines() if line.strip()]
        calls = []
        names = set()
        tools = reasoning = 0
        for row in rows:
            for inputs, outputs in zip(row['input_token_count'], row['output_token_count']):
                calls.extend(zip(inputs, outputs))
            for turn in row['inference_log']:
                if not isinstance(turn, dict):
                    continue
                for key, entries in turn.items():
                    if not key.startswith('step_'):
                        continue
                    for entry in entries:
                        tools += entry.get('role') == 'tool'
                        reasoning += entry.get('role') == 'assistant' and bool(entry.get('reasoning_content'))
                        if entry.get('role') == 'handler_log':
                            for call in entry.get('model_response_decoded', []) or []:
                                names.add(call.split('(', 1)[0])
        return dict(source_url=self.provenance['source_url'], source_sha256=hashlib.sha256(data).hexdigest(),
                    sessions=len(rows), llm_calls=len(calls), executed_tools=tools,
                    calls_with_reasoning=reasoning, max_input_plus_output=max(p+o for p, o in calls),
                    total_output_tokens=sum(o for _, o in calls), tool_names=sorted(names),
                    scope='Archive inventory; sampling/tokenizer revisions and per-tool timing are not recorded.')


if __name__ == '__main__':
    BFCLArchive(tyro.cli(BFCLArchiveConfig)).fetch()

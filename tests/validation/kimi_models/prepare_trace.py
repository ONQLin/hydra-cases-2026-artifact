#!/usr/bin/env python3
"""Select complete existing-dataset requests without scaling or truncation."""

import argparse
import csv
import hashlib
import json
from pathlib import Path


class DatasetTraceSelection:
    def __init__(self, source, output, requests, max_context):
        self.source, self.output = source.resolve(), output.resolve()
        self.requests, self.max_context = requests, max_context

    def run(self):
        if self.requests < 1 or self.max_context < 2:
            raise ValueError('Positive request count and context limit >= 2 required.')
        if self.output.exists() or self.output.with_suffix('.json').exists():
            raise ValueError('Use a fresh trace path to preserve provenance.')
        selected, indices = [], []
        total = eligible = 0
        with self.source.open() as stream:
            for index,row in enumerate(csv.DictReader(stream)):
                total += 1
                prefill,decode = (int(row[key]) for key in ('num_prefill_tokens','num_decode_tokens'))
                if prefill < 1 or decode < 1 or prefill+decode > self.max_context:
                    continue
                eligible += 1
                if len(selected) < self.requests:
                    selected.append(dict(num_prefill_tokens=prefill,num_decode_tokens=decode))
                    indices.append(index)
        if len(selected) != self.requests:
            raise ValueError(f'Only {eligible} eligible requests; requested {self.requests}.')
        self.output.parent.mkdir(parents=True,exist_ok=True)
        with self.output.open('w') as stream:
            writer = csv.DictWriter(stream,fieldnames=list(selected[0]))
            writer.writeheader(); writer.writerows(selected)
        metadata = dict(source=str(self.source),source_sha256=hashlib.sha256(self.source.read_bytes()).hexdigest(),
            selection='First eligible rows in source order; no rescaling, padding or truncation.',
            tokenizer='Inherited source-dataset token lengths; no model-specific retokenization.',
            source_rows=total,eligible_rows=eligible,context_limit=self.max_context,
            source_row_indices_zero_based=indices,selected_requests=len(selected),
            total_prefill_tokens=sum(r['num_prefill_tokens'] for r in selected),
            total_decode_tokens=sum(r['num_decode_tokens'] for r in selected),
            trace_sha256=hashlib.sha256(self.output.read_bytes()).hexdigest())
        self.output.with_suffix('.json').write_text(json.dumps(metadata,indent=2)+'\n')
        print(json.dumps(metadata,indent=2))
        return metadata


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('dataset/chat/chat1m_stats_nemo.csv'))
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--requests',type=int,default=2)
    parser.add_argument('--max-context',type=int,default=4096)
    args = parser.parse_args()
    DatasetTraceSelection(args.source,args.output,args.requests,args.max_context).run()

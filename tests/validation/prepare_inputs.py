#!/usr/bin/env python3
"""Generate small demo inputs; keep simulation outputs out of source control."""

import argparse
import csv
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from tests.validation.kimi_models.prepare_trace import DatasetTraceSelection


class ValidationInputs:
    @staticmethod
    def write_trace(path, prefill=32, decode=2, requests=2):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open('x', newline='') as stream:
            writer = csv.writer(stream, lineterminator='\n')
            writer.writerow(['num_prefill_tokens', 'num_decode_tokens'])
            writer.writerows((prefill, decode) for _ in range(requests))
        return path

    @staticmethod
    def bfcl_provenance():
        return dict(benchmark_revision='6ea57973c7a6097fd7c5915698c54c17c5b1b6c8',
                    source_model='synthetic fixture; no LLM executed',
                    tokenizer='synthetic token counts; no tokenizer executed',
                    chat_template='not applicable to synthetic fixture',
                    sampling='not applicable to synthetic fixture', is_synthetic=True)

    @staticmethod
    def bfcl_records():
        records = []
        shapes = [([[32, 64], [96]], [[3, 2], [1]], [[2, 1], [0]]),
                  ([[48, 80]], [[3, 2]], [[1, 0]])]
        for index, (inputs, outputs, tools) in enumerate(shapes):
            turns = [[dict(role='state_info', content={})]] if index == 0 else []
            for counts in tools:
                turn = dict(begin_of_turn_query=[dict(role='user', content='Synthetic format test, not a BFCL task.')])
                for step, count in enumerate(counts):
                    turn[f'step_{step}'] = [dict(role='assistant', content='Synthetic response.')]
                    turn[f'step_{step}'] += [dict(role='tool', content='Synthetic tool return.') for _ in range(count)]
                turns.append(turn)
            records.append(dict(id=f'multi_turn_base_fixture_{index}', input_token_count=inputs,
                                output_token_count=outputs, inference_log=turns))
        return records

    def write(self, output, include_datasets=False):
        output = Path(output)
        output.mkdir(parents=True, exist_ok=False)
        for name, prefill, decode, count in [('decoder_smoke', 32, 2, 2),
                                             ('kimi_short', 2, 2, 2),
                                             ('kimi_concurrent', 128, 4, 8),
                                             ('burst', 128, 8, 4)]:
            self.write_trace(output/f'{name}.csv', prefill, decode, count)
        (output/'bfcl_format.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in self.bfcl_records()))
        (output/'provenance.json').write_text(json.dumps(self.bfcl_provenance(), indent=2)+'\n')
        if include_datasets:
            for name, source, count in [('chat_first1', 'chat/chat1m_stats_nemo.csv', 1),
                                        ('chat_first2', 'chat/chat1m_stats_nemo.csv', 2),
                                        ('chat_first4', 'chat/chat1m_stats_nemo.csv', 4),
                                        ('bwb_first1', 'bwb/bwb_translation_stats_nemo.csv', 1)]:
                DatasetTraceSelection(ROOT/'dataset'/source, output/f'{name}.csv', count, 4096).run()
        print(f'Generated validation inputs in {output}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--include-datasets', action='store_true')
    args = parser.parse_args()
    ValidationInputs().write(args.output_dir, args.include_datasets)

"""Trace scaling must describe the same context for every request mode."""

from pathlib import Path
import tempfile
import unittest

from Sim.config.sys_config import TraceRequestLengthGeneratorConfig
from Sim.request_generator.trace_request_length_generator import TraceRequestLengthGenerator


class TraceRequestLengthTest(unittest.TestCase):
    def test_prefill_scale_is_applied_once(self):
        with tempfile.TemporaryDirectory() as folder:
            trace = Path(folder) / 'lengths.csv'
            trace.write_text('num_prefill_tokens,num_decode_tokens\n100,20\n')
            for scale in (0.5, 1, 2):
                for mode in ('e2e', 'prefill', 'decode'):
                    with self.subTest(scale=scale, mode=mode):
                        config = TraceRequestLengthGeneratorConfig(
                            trace_file=str(trace), prefill_scale_factor=scale,
                            decode_scale_factor=2, request_type=mode, max_tokens=1024)
                        generator = TraceRequestLengthGenerator(config, 1)
                        prefill, decode, context = generator.get_next_num_tokens()
                        self.assertEqual(context, int(100 * scale))
                        self.assertEqual(prefill, 0 if mode == 'decode' else context)
                        self.assertEqual(decode, 0 if mode == 'prefill' else 40)
                        self.assertEqual(generator.get_next_num_tokens(), (None, None, None))


if __name__ == '__main__':
    unittest.main()

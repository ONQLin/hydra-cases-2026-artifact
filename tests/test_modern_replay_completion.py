"""Reject partial or silently transformed traces in strict decoder replay."""

from pathlib import Path
from tempfile import TemporaryDirectory
from types import SimpleNamespace
import unittest

from tests.validation.modern_models.validate import ModernModelValidation


class ModernReplayCompletionTest(unittest.TestCase):
    def test_complete_and_incomplete_token_accounting(self):
        with TemporaryDirectory() as directory:
            root = Path(directory)
            trace = root/'source.csv'
            trace.write_text('num_prefill_tokens,num_decode_tokens\n3,2\n4,3\n')
            effective = root/'effective_workload.csv'
            effective.write_text(trace.read_text())
            validator = ModernModelValidation(SimpleNamespace(output_dir=root,trace=trace,requests=2))
            run = SimpleNamespace(run_dir=root)
            result = dict(completed_requests=2,running_requests=0,pending_requests=0,output_tokens=7)
            validator.verify_complete(run,result)
            for changes in (dict(completed_requests=1),dict(running_requests=1),
                            dict(pending_requests=1),dict(output_tokens=6)):
                with self.subTest(changes=changes), self.assertRaisesRegex(RuntimeError,'did not complete'):
                    validator.verify_complete(run,dict(result,**changes))
            effective.write_text('num_prefill_tokens,num_decode_tokens\n3,2\n4,2\n')
            with self.assertRaisesRegex(RuntimeError,'lengths unchanged'):
                validator.verify_complete(run,result)
            effective.write_text('num_prefill_tokens,num_decode_tokens\n3,2\n')
            with self.assertRaisesRegex(RuntimeError,'lengths unchanged'):
                validator.verify_complete(run,result)


if __name__ == '__main__':
    unittest.main()

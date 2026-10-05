#!/usr/bin/env python3
"""JSON-lines entry point for HYDRA-controlled CHIPSIM execution."""

import argparse
from contextlib import redirect_stdout
import json
from pathlib import Path
import sys


class RuntimeServiceDriver:
    def create_service(self, config, system_path):
        from runtime_execution import RuntimeExecutionService
        return RuntimeExecutionService(config, system_path)

    def run(self, system_path):
        output = sys.stdout
        config = json.loads(system_path.read_text())
        sys.path.insert(0, config['chipsim_root'])
        with redirect_stdout(sys.stderr):
            service = self.create_service(config, system_path)
        try:
            output.write(json.dumps({'ready': True}) + '\n')
            output.flush()
            for line in sys.stdin:
                try:
                    with redirect_stdout(sys.stderr):
                        result = service.execute(json.loads(line))
                except Exception as exc:
                    import traceback
                    traceback.print_exc(file=sys.stderr)
                    result = {'error': str(exc)}
                output.write(json.dumps(result) + '\n')
                output.flush()
                if 'error' in result:
                    break
        finally:
            # The peer can disappear during either the handshake or a reply.
            # Always reap the persistent Garnet process in that case.
            with redirect_stdout(sys.stderr):
                service.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', type=Path, required=True)
    RuntimeServiceDriver().run(parser.parse_args().system.resolve())

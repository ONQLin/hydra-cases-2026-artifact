"""Online CHIPSIM phase execution under HYDRA's resource reservations."""

from contextlib import ExitStack
from dataclasses import asdict
import json
import math
import os
import signal
from pathlib import Path
import selectors
import subprocess

import numpy as np
import Sim.common as common
from Sim.execution.native import NativeExecutionBackend


class ChipsimExecutionBackend(NativeExecutionBackend):
    service_filename = 'runtime_service.py'

    @staticmethod
    def get_name() -> str:
        return "chipsim_runtime"

    def __init__(self, config):
        super().__init__(config)
        self.process = None
        self.log = None
        self.calls = 0
        self.trace = None

    def initialize(self, graph, memory_system, compute_system, mapping):
        from Sim.backends.chipsim import ChipsimBackend
        bridge = ChipsimBackend(self.config)
        bridge.validate_environment()
        self.output_dir = Path(self.config.metrics_config.output_dir).resolve() / 'chipsim_runtime'
        self.output_dir.mkdir(parents=True, exist_ok=True)
        from Sim.execution.network_system import NetworkSystemSnapshot
        snapshot = NetworkSystemSnapshot(self.config).export(graph, memory_system, mapping)
        snapshot.update(chipsim_root=str(bridge.chipsim_root.resolve()),
                        chipsim_revision=bridge._submodule_revision())
        self.memory_bandwidth = snapshot['memory_bandwidth_gbps']
        self.configure_system(snapshot)
        snapshot_path = self.output_dir / 'system.json'
        snapshot_path.write_text(json.dumps(snapshot, indent=2))
        self.log = (self.output_dir / 'service.log').open('w')
        self.trace = (self.output_dir / 'execution.jsonl').open('w')
        command = [str(bridge.python_executable), '-u',
                   str(bridge.repo_root / 'integrations/chipsim' / self.service_filename),
                   '--system', str(snapshot_path)]
        self.process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.PIPE,
                                        stderr=self.log, text=True, bufsize=1,
                                        cwd=str(self.output_dir), start_new_session=True)
        if self._receive() != {"ready": True}:
            raise RuntimeError("CHIPSIM service did not complete initialization.")

    def configure_system(self, snapshot):
        """Allow execution subclasses to add their service contract."""

    def _receive(self):
        timeout = self.config.chipsim_config.timeout_seconds or None
        with selectors.DefaultSelector() as selector:
            selector.register(self.process.stdout, selectors.EVENT_READ)
            if not selector.select(timeout):
                raise TimeoutError(f"CHIPSIM execution timed out; see {self.output_dir / 'service.log'}")
        line = self.process.stdout.readline()
        if not line:
            raise RuntimeError(f"CHIPSIM service exited; see {self.output_dir / 'service.log'}")
        result = json.loads(line)
        if 'error' in result:
            raise RuntimeError(f"CHIPSIM execution failed: {result['error']}")
        return result

    def _request(self, message):
        self.calls += 1
        message = dict(message, request_id=self.calls)
        self.process.stdin.write(json.dumps(message) + '\n')
        self.process.stdin.flush()
        result = self._receive()
        if result.get('request_id') != self.calls:
            raise RuntimeError("CHIPSIM returned a mismatched request ID.")
        self.trace.write(json.dumps({'request': message, 'result': result}) + '\n')
        return result

    def _execute(self, message):
        result = self._request(message)
        return result['latency_us'] * common.time_granularity / 1e6

    def execute_block(self, block, stage, batch_size, chiplet, memory_id, bandwidth):
        profiles = self.profile_block(block, stage, batch_size, chiplet, bandwidth)
        if any(not item.execution_phases for item in profiles):
            raise ValueError(f"Missing execution phase export for {block.block_config.type_name} on chiplet {chiplet.chiplet_type}.")
        phases = [asdict(part) for item in profiles for phase in item.execution_phases
                  for part in phase.expanded()]
        latency = self._execute({
            'kind': 'block', 'source': int(memory_id), 'destination': int(chiplet.chiplet_id),
            'bandwidth_bytes_s': min(bandwidth, self.memory_bandwidth[str(memory_id)]) * 1e9,
            'phases': phases, 'block_id': block.block_num, 'stage': stage,
            'batch_size': batch_size, 'context_length': block.context_length,
        })
        return math.ceil(latency), float(np.mean([item.utilization for item in profiles]))

    def transfer(self, source, destination, size_mb, bandwidth, native_ticks):
        return math.ceil(self._execute({
            'kind': 'transfer', 'source': int(source), 'destination': int(destination),
            'bandwidth_bytes_s': bandwidth * 1024**3,
            'phases': [{'compute_ns': 0, 'memory_bytes': math.ceil(size_mb * 1024**2)}],
        }))

    def close(self):
        with ExitStack() as resources:
            for name in ('log', 'trace'):
                stream = getattr(self, name)
                if stream is not None:
                    resources.callback(stream.close)
                    setattr(self, name, None)
            process, self.process = self.process, None
            if process is not None:
                resources.callback(process.stdout.close)
                try:
                    # Close stdin even if the child has already exited. Its
                    # buffered writer can raise when the peer died mid-request.
                    try:
                        process.stdin.close()
                    except BrokenPipeError:
                        pass
                    if process.poll() is None:
                        process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    try:
                        os.killpg(process.pid, signal.SIGKILL)
                    except ProcessLookupError:
                        pass  # The child exited between wait and kill.
                    process.wait()

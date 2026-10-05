"""Persistent Garnet process using CHIPSIM's topology and transport setup."""

import gzip
import json
import math
import socket
import subprocess

from runtime_execution import RuntimeCommunicationSimulator


class PersistentGarnetNetwork(RuntimeCommunicationSimulator):
    def __init__(self, config, system_path):
        config = dict(config, reuse_mesh_paths=False)
        super().__init__(config, system_path)
        self.current_dir = self.output_dir / 'garnet_online'
        self.current_dir.mkdir(parents=True, exist_ok=True)
        self.process = None
        self.log = None
        self.stream = None
        self.connection, child = socket.socketpair()
        self.connection.settimeout(config['timeout_seconds'] or None)
        try:
            traffic = self.current_dir / 'unused.gz'
            with gzip.open(traffic, 'wt') as stream:
                stream.write('# online traffic arrives over the control socket\n')
            command = self._build_garnet_simulation_command(str(traffic), 1)
            command += ['--hydra-control-fd', str(child.fileno())]
            if config.get('dma_pacing', False):
                bandwidths = [config['memory_bandwidth_gbps'].get(str(node), 0) * 1e9
                              for node in config['chiplet_ids']]
                command += ['--hydra-dma-burst-bytes', str(config['dma_burst_bytes']),
                            '--hydra-hbm-bandwidths', json.dumps(bandwidths),
                            '--hydra-hbm-access-ticks', str(round(config['hbm_access_latency_ns'] * 1000))]
            (self.current_dir / 'command.json').write_text(json.dumps(command, indent=2))
            self.log = (self.current_dir / 'gem5.log').open('w')
            self.process = subprocess.Popen(command, cwd=self.gem5_path, stdout=self.log,
                                            stderr=subprocess.STDOUT, pass_fds=(child.fileno(),))
            child.close()
            self.stream = self.connection.makefile('rw')
            if self._receive() != {'ready': True}:
                raise RuntimeError('Garnet online initialization failed.')
        except BaseException:
            child.close()
            self.close()
            raise

    def _receive(self):
        line = self.stream.readline()
        if not line:
            raise RuntimeError(f'Garnet control connection closed; see {self.current_dir / "gem5.log"}')
        return json.loads(line)

    def execute(self, request):
        command = dict(request)
        if command['kind'] == 'submit':
            command['flits'] = math.ceil(command.pop('bytes') / (self.config['link_width_bits'] / 8))
            if command['flits'] <= 0 or command['flits'] > 2**31 - 1:
                raise ValueError('Transfer exceeds the supported flit count.')
        self.stream.write(json.dumps(command) + '\n')
        self.stream.flush()
        result = self._receive()
        result['request_id'] = request.get('request_id')
        return result

    def close(self):
        if self.process is not None:
            try:
                if self.process.poll() is None:
                    if self.stream is not None:
                        self.stream.write(json.dumps({'kind': 'close'}) + '\n')
                        self.stream.flush()
                    else:
                        self.process.terminate()
                    self.process.wait(timeout=10)
            except (BrokenPipeError, OSError, subprocess.TimeoutExpired):
                self.process.kill()
                self.process.wait()
            self.process = None
        if self.stream is not None:
            self.stream.close()
            self.stream = None
        self.connection.close()
        if self.log is not None:
            self.log.close()
            self.log = None

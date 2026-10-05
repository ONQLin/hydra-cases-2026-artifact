#!/usr/bin/env python3
"""Persistent CHIPSIM/Garnet service for HYDRA's conservative event loop."""

import argparse
from pathlib import Path

from runtime_service import RuntimeServiceDriver


class ContendedServiceDriver(RuntimeServiceDriver):
    def create_service(self, config, system_path):
        from persistent_network import PersistentGarnetNetwork
        return PersistentGarnetNetwork(config, system_path)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--system', type=Path, required=True)
    ContendedServiceDriver().run(parser.parse_args().system.resolve())

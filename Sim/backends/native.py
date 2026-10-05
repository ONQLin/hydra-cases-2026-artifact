"""Native HYDRA simulation entry point."""

from Sim.backends.BaseSimulationBackend import BaseSimulationBackend
import signal
import threading


class NativeSimulationBackend(BaseSimulationBackend):
    display_name = 'HYDRA-Analytic'

    def __init__(self, config):
        self.config = config

    @staticmethod
    def get_name() -> str:
        return "hydra_sim"

    def run(self):
        from Sim.simulator import Simulator
        previous_handler = None
        if threading.current_thread() is threading.main_thread():
            previous_handler = signal.signal(signal.SIGTERM, self._terminate)
        try:
            return Simulator(self.config).run()
        finally:
            if previous_handler is not None:
                signal.signal(signal.SIGTERM, previous_handler)

    @staticmethod
    def _terminate(signum, frame):
        # Unwind Simulator.run's finally block and close any child simulators.
        raise SystemExit(128 + signum)

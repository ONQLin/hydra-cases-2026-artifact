from Sim.request_generator.base_request_interval_generator import BaseRequestIntervalGenerator
from Sim.config.sys_config import StaticRequestIntervalGeneratorConfig

class StaticRequestIntervalGenerator(BaseRequestIntervalGenerator):
    def __init__(self, config: StaticRequestIntervalGeneratorConfig):
        super().__init__(config)
    
    def get_next_inter_request_time(self) -> float:
        assert isinstance(self.config, StaticRequestIntervalGeneratorConfig), "Config must be of type StaticRequestIntervalGeneratorConfig"
        return self.config.interval # in second 

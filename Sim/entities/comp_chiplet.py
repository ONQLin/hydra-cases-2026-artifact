from dataclasses import dataclass, field
from Sim.config.utils import chiplet_types_list

class comp_chiplet:
    # at least it reqires chiplet_type, chiplet_id and chiplet_loc
    chiplet_type: int
    chiplet_id: int
    chiplet_loc: tuple
    idle: bool = True
    power: float = 10.0  # Default power if not specified
    
    def __init__(self, **attr):
        for key, value in attr.items():
            setattr(self, key, value)
        if not hasattr(self, 'chiplet_type'):
            raise ValueError("Chiplet type must be specified.")
        if not hasattr(self, 'chiplet_id'):
            raise ValueError("Chiplet ID must be specified.")
        if not hasattr(self, 'chiplet_loc'):
            raise ValueError("Chiplet location must be specified.")
        if not hasattr(self, 'power'):
            raise ValueError("Chiplet power must be specified.")
        self.idle = True  # Initially, the chiplet is idle
        self.util = 0.0  # Utilization percentage
        self.job_queue = []  # Jobs assigned to this chiplet
        # self.acc_profile = None
        # self.get_module_factory(chiplet_types_list[self.chiplet_type])
        
    # def get_module_factory(self, acc_name: str):
    #     """
    #     creates the accelerator model based on the chiplet type
    #     """
    #     self.acc_profile = BaseAccModel.create_from_name(acc_name)
        
    def is_available(self) -> bool:
        """
        Check if the chiplet is available (idle).
        """
        return self.idle
    
    def set_busy(self):
        """
        Set the chiplet as busy.
        """
        self.idle = False
        
    def set_util(self,util,power):
        self.util = util
        self.power = power

    def set_free(self):
        """
        Set the chiplet as free (idle).
        """
        self.idle = True
        self.util = 0.0
        self.power = 10.0
        if self.job_queue:
            self.dequeue_job()
    
    @property
    def queue_length(self) -> int:
        """
        Get the length of the job queue.
        """
        return len(self.job_queue)
    
    def enqueue_job(self, job):
        """
        Add a job to the chiplet's job queue.
        """
        if job not in self.job_queue:
            self.job_queue.append(job)
        
    def dequeue_job(self):
        """
        Remove and return the next job from the chiplet's job queue.
        """
        if self.job_queue:
            return self.job_queue.pop(0)
        else:
            raise IndexError("Job queue is empty.")

    def remove_job(self, job):
        """
        Remove a specific job from the queue if present.
        """
        if job in self.job_queue:
            self.job_queue.remove(job)

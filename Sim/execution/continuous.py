"""Reuse the physical block executor while yielding at real token boundaries."""

from Sim.processing import processing


class ContinuousProcessing(processing):
    def __init__(self, execution_backend, scheduler):
        super().__init__(execution_backend)
        self.scheduler = scheduler

    def select_inferences(self):
        return [r._infs['prefill'] if r._process_idx == 0 and r.has_prefill()
                else r._infs['decode'] for r in self.requests]

    def restore_iteration(self):
        for request in self.requests:
            saved = self.scheduler.state.pop(request._id, None)
            if saved is not None:
                self.states_cache[request._id], inputs = saved
                self.input_cache.extend(inputs)

    def yield_iteration(self, timestep):
        self.scheduler.finish_iteration(self, timestep)
        self.reset_processing()
        return True

    def drop_request_fromsystem(self, request_id, mem_sys, timestep):
        # Exact lifetime reservations must prevent physical allocation failure.
        # Restarting a partially decoded agent would corrupt its token/event log.
        raise MemoryError(f'Reserved continuous request {request_id} exceeded physical HBM capacity.')

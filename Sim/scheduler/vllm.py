"""Paper-era dynamic request batching."""

import Sim.common as common
import Sim.entities.comp_sys as comp_sys
import Sim.entities.mem_sys as mem_sys
import Sim.metrics.monitor as monitor
from Sim.entities.batch_of_req import BatchOfRequests
from Sim.entities.infer import infer
from Sim.entities.request import Request
from Sim.metrics.monitor import congestion_monitor
from Sim.scheduler.vllm_legacy import VllmLegacyReplicaScheduler


class VllmReplicaScheduler(VllmLegacyReplicaScheduler):
    """Dynamic batching behavior used before continuous-batching emulation."""

    def __init__(
        self,
        chip_graph,
        mem_sys_inst: mem_sys.mem_sys,
        comp_sys_inst: comp_sys.comp_sys,
        bs=1,
    ):
        super().__init__(chip_graph, mem_sys_inst, comp_sys_inst, bs)
        self.throughput_priority_mode = False
        self.enable_token_budget_admission = False
        self.enable_context_bw_guard = True

    def _bs_context_length_threshold(
        self, total_pf: int, max_bs: int, target_tp=100
    ) -> int:
        eff_bw = self.bw_budget * 1e9
        num_job = self._max_concurrent_batch - monitor.request_counter.running_batches
        cur_bs = max_bs * num_job
        if common.total_dc_tokens < 0 or common.total_pf_tokens < 0:
            raise ValueError("Total prefill or decode tokens is negative.")

        while cur_bs > 0:
            post_bytes_per_token = self._get_byte_per_token(
                (common.total_pf_tokens + total_pf * cur_bs, common.total_dc_tokens)
            )
            if target_tp * post_bytes_per_token <= eff_bw:
                break
            cur_bs -= 1

        if 0 < cur_bs < max_bs * num_job:
            cur_bs = max(1, cur_bs // max(1, num_job))
        return min(cur_bs, max_bs)

    def _choose_effective_batchsize(
        self,
        mem_sys_inst: mem_sys.mem_sys,
        comp_sys_inst: comp_sys.comp_sys,
        max_batchsize: int,
        pf_len: int,
    ) -> int:
        if max_batchsize <= 0:
            return 0
        if monitor.request_counter.running_batches >= self._max_concurrent_batch:
            return 0
        if comp_sys_inst._current_run >= comp_sys_inst._total_budget:
            return 0

        outstanding_n = len(common.RequestQueue.outstanding)
        remaining_reqs = (
            self._max_concurrent_requests - monitor.request_counter.running_requests
        )
        if outstanding_n == 0 or remaining_reqs <= 0:
            return 0

        batchsize = min(max_batchsize, outstanding_n, remaining_reqs)
        batchsize = min(
            batchsize,
            self._bs_context_length_threshold(
                total_pf=pf_len,
                max_bs=max_batchsize,
                target_tp=1,
            ),
        )
        batchsize = max(1, batchsize)

        while batchsize > 0:
            candidate_reqs = common.RequestQueue.outstanding[:batchsize]
            ok, allocations = self._can_allocate_batch_memory(
                mem_sys_inst, candidate_reqs
            )
            if ok:
                for req_id, block_name, allocated_memory in allocations:
                    common.req_allocated_mems.setdefault(req_id, []).append(
                        (block_name, allocated_memory)
                    )
                return batchsize
            batchsize -= 1
        return 0

    def schedule_request(
        self,
        mem_sys_inst: mem_sys.mem_sys,
        comp_sys_inst: comp_sys.comp_sys,
        batchsize=1,
    ) -> int:
        if congestion_monitor.total_congested() > batchsize:
            self._current_bs_est = max(1, self._current_bs_est - 2)
        else:
            self._current_bs_est = min(batchsize, self._current_bs_est + 2)
        batchsize = self._current_bs_est

        if len(common.RequestQueue.outstanding) < batchsize:
            return 0
        if monitor.request_counter.running_batches >= self._max_concurrent_batch:
            return -1
        if comp_sys_inst._current_run >= comp_sys_inst._total_budget:
            return -1
        if (
            monitor.request_counter.running_requests + batchsize
            > self._max_concurrent_requests
        ):
            return -1

        effective_batchsize = self._choose_effective_batchsize(
            mem_sys_inst,
            comp_sys_inst,
            batchsize,
            common.RequestQueue.outstanding[0].get_prefill_length(),
        )
        if effective_batchsize <= 0:
            return -1

        incoming_batch = BatchOfRequests(
            common.RequestQueue.outstanding[:effective_batchsize]
        )
        for incoming_req in incoming_batch.ongoing_requests:
            if incoming_req.has_prefill():
                prefill_inf: infer = incoming_req._infs["prefill"]
                if prefill_inf.type != "prefill":
                    raise ValueError(
                        "The first inference in the request must be prefill."
                    )
            if len(incoming_req._infs) == 0:
                raise ValueError("The request has no inferences.")
            if any(len(inf.blocks) == 0 for inf in incoming_req._infs.values()):
                raise ValueError("The inference has no blocks.")

        incoming_batch.step_batch_id()
        common.RequestQueue.ready.append(incoming_batch)
        for _ in range(effective_batchsize):
            request: Request = common.RequestQueue.outstanding[0]
            common.total_pf_tokens += request.get_prefill_length()
            common.RequestQueue.outstanding.pop(0)
            monitor.request_counter.increment_running()
            monitor.request_counter.decrement_pending()
        monitor.request_counter.increment_running_batches()
        return 1

    @staticmethod
    def get_name() -> str:
        return "vllm"

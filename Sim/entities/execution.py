"""Execution profiles shared by analytical and co-simulation backends."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ExecutionPhase:
    """Compute/SRAM work and external-memory traffic with explicit ordering.

    Successive phases are sequential. External bytes retain the accelerator
    model's aggregate traffic convention; they are not an instruction trace.
    Legacy phases overlap compute and traffic. Ordered phases serialize reads,
    compute, then writes; iterations retain recurrent state dependencies.
    """

    compute_ns: float
    memory_bytes: float = 0.0
    name: str = ''
    memory_direction: str = 'read'
    ordered: bool = False
    read_bytes: float = 0.0
    write_bytes: float = 0.0
    iterations: int = 1

    def __post_init__(self):
        import math
        values = (self.compute_ns, self.memory_bytes, self.read_bytes, self.write_bytes)
        if any(not math.isfinite(value) or value < 0 for value in values):
            raise ValueError('Execution costs must be finite and nonnegative.')
        if self.memory_direction not in ('read', 'write'):
            raise ValueError('Unknown memory direction.')
        if type(self.iterations) is not int or self.iterations < 1:
            raise ValueError('Execution iterations must be a positive integer.')
        if self.ordered:
            if self.memory_bytes != self.read_bytes + self.write_bytes:
                raise ValueError('Ordered phase bytes must equal reads plus writes.')
        elif self.read_bytes or self.write_bytes or self.iterations != 1:
            raise ValueError('Separate reads/writes and repetitions require ordered phases.')

    def latency_ns(self, bandwidth_gbps):
        if bandwidth_gbps <= 0:
            raise ValueError('Bandwidth must be positive.')
        if self.ordered:
            return self.iterations * (self.compute_ns + self.memory_bytes / bandwidth_gbps)
        return max(self.compute_ns, self.memory_bytes / bandwidth_gbps)

    def expanded(self):
        """Lower dependent read/compute/write stages without changing legacy overlap."""
        if not self.ordered:
            yield self
            return
        for _ in range(self.iterations):
            if self.read_bytes:
                yield ExecutionPhase(0, self.read_bytes, self.name + ':read')
            if self.compute_ns:
                yield ExecutionPhase(self.compute_ns, name=self.name + ':compute')
            if self.write_bytes:
                yield ExecutionPhase(0, self.write_bytes, self.name + ':write', 'write')

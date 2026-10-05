# DeepSeek-V3 decoder baseline

`deepseek-v3-text` adds the pinned DeepSeek-V3 decoder stack through the existing
model, operator-profile and routing factories. It contains **61 MLA layers**, a
dense SwiGLU FFN in the first three layers and group-limited MoE in the remaining
58. MLA uses absorbed KV cache and the existing streaming schedule. The initial
supported context limit is **4096**, within the source model's base RoPE window.

The [source config and router revision](source_config.json) pin the official
[checkpoint configuration](https://huggingface.co/deepseek-ai/DeepSeek-V3/blob/e815299b0bcbac849fa540c768ef21845365c9eb/config.json)
and [inference implementation](https://github.com/deepseek-ai/DeepSeek-V3/blob/9b4e9788e4a3a731f7567338ed15d3ec549ce03b/inference/model.py).
No checkpoint weights or remote model code are loaded or executed.

## Grouped MoE contract

`GroupedMoEConfig` / `GroupedMoEProfile` extend the existing MoE classes. Router
nodes explicitly represent these dependencies:

```text
logits -> sigmoid scores -> correction bias -> per-group top-2 sum
       -> group top-4 -> masked expert scores -> expert top-8 -> dispatch
original sigmoid scores + selected expert IDs -> normalize -> scale 2.5
expert outputs + normalized routing weights -> gather/combine -> shared-expert join
```

The correction bias affects **selection only**. Combination weights use the
original selected sigmoid scores; the graph keeps those values live through
gather and normalization. Masks exclude every nonselected group, even when
all corrected scores are negative. Group scores, indices, routing weights and
workspace are explicit tensors with producer-before-consumer edges and FP32
intermediates. Top-k costs use conservative insertion comparisons, not measured
kernel timings.

`grouped_cyclic` and `grouped_hotspot` routing policies produce reproducible
**synthetic** assignments that obey the group limit. They retain token-offset
consistency between prefill and decode, record route hashes and expert counts,
and execute through the existing colocated dispatch/expert/join path. The
performance simulator does not compute real checkpoint logits; numerical tests
supply logits independently to validate the router graph's algebra. Actual
expert traffic therefore remains a controlled assignment scenario.

All 256 routed experts and the shared expert contribute to resident weight
capacity. Only experts receiving tokens execute during a call. Experts remain
serial on the mapped block chiplet; this is not expert-parallel all-to-all.
The older ungrouped Kimi routing/profile remains available.

## Capacity and scope

The W8/A8/KV8 performance surrogate requires **669,173,053,952 bytes
(623.22 GiB)** for decoder weights. Persistent absorbed KV storage is 576 bytes
per token per layer, or **137.25 MiB** across 61 layers at context 4096 and batch 1.
Activation/workspace, allocation imbalance and request concurrency are additional.
The model excludes embedding, final normalization, LM head and MTP; it does not
reproduce the source checkpoint's mixed-precision numerical behavior.

The archived 16-HBM point has only 256 GiB capacity and correctly rejects this
model. The reproducible capacity example explicitly changes HBM count to 64.
The existing BW placement policy puts HBMs on the outer ring, so its 72-node
layout is 36-by-2. Both the original hardware row and the effective override are
recorded. This is a capacity example, not a newly optimized Pareto point.

The newer [multi-package path](../../multi_package/README.md) instead partitions
the decoder across four ordinary 256-GiB packages, with local weights/state and
explicit activation/token-feedback transfers. See its
[complete-length Chat validation](../../../tests/validation/multi_package/README.md).

Two generic configuration issues were fixed: heterogeneous dense/MoE block
sequences can contain only one attention/recurrent family, and uniform-grid
initialization now uses the configured dimensions and validates supplied maps.
MLA tile metadata/compute costs are cached per immutable tile shape and throughput
rates, avoiding repeated host calculations without changing simulated costs.

[Validation and workload instructions](../../../tests/validation/deepseek_models/README.md)
cover router algebra, weight/state accounting, capacity rejection, full decoder
runs and a small three-backend smoke test. Multi-fidelity acceptance here is
**execution plus comparable metrics**, with no new fidelity calibration campaign.

## Subsequent work

- Recorded expert routing traces, including layer/request identity, before
  drawing workload-specific expert-utilization conclusions.
- Contexts beyond 4096, with explicit position/scaling and capacity checks.
- Distributed expert placement/dispatch and group-aware runtime policies.
- DeepSeek-V3.2 DSA and newer families as separate operators/model configurations;
  this V3 baseline does not imply their support.

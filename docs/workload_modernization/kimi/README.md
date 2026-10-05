# Kimi Linear decoder baseline

HYDRA registers `kimi-linear-48b-a3b-text` and the opt-in
`kimi-linear-48b-a3b-text-streaming`: the pinned Kimi Linear
48B-A3B **decoder-stack topology**, with shared analytical compute and all three
network backends. It includes 20 KDA layers and 7 expanded-cache NoPE MLA layers,
two norms/residuals per layer, a dense first FFN, and 26 MoE FFNs. Embedding,
final normalization and the LM head remain excluded, as in the earlier decoder
baselines. The precision contract is a W8/A8/KV8 performance surrogate with FP32
matrix state, attention/route intermediates and explicit casts where modeled.
It does not run checkpoint weights or generate text.

This first version uses **synthetic expert routes and colocated serial experts**.
It is suitable for controlled operator/runtime experiments, not a calibrated
prediction of checkpoint inference performance. Reproduction and measured
simulation costs are in the [validation README](../../../tests/validation/kimi_models/README.md).

## Completed steps

### 1. KDA chunk prefill and tile mapping

`KDAConfig(algorithm='chunk', chunk_size=64)` selects chunk prefill and one-token
recurrent decode. `state_mapping='tiled'` also enables tiled execution for the
recurrent algorithm. The default `algorithm='recurrent', state_mapping='auto'`
retains the previous whole-state baseline. `head_tile_size` and `value_tile_size`
select the independent head and state-column tiles; zero value width means the
full width. Partial head/value tiles and partial final chunks are supported.

The chunk program builds cumulative channel gates, lower-triangular key products,
a triangular solve for transformed keys/values, state residuals, causal attention
products, outputs, and the next matrix state. Old state remains live until all
consumers finish. Chunks are sequential within a sequence/head/value tile;
different tiles are also serialized on the block's mapped chiplet.

For `H` tile heads, chunk length `T`, key width `K`, value width `V`, and
`s=T*(T-1)/2`, `c=T*(T+1)/2`, the modeled chunk MAC count is:

```text
3*H*T*K*V + H*s*(2*K+V) + H*c*(K+V)
```

This implements a triangular-solve formulation of the KDA equations; it is not
an instruction-level model of FLA's optimized chunk kernel. It assumes triangular
work can omit masked MACs. Element-operation rates and accelerator utilization
still need calibration.

Outer operator boundaries use explicit HBM staging. Inside a tile, actual graph
liveness plus matrix-state bytes determines whether the complete program fits
SRAM. A resident tile reads prior state once (decode), carries state across its
chunks, then writes final state once. An oversized tile materializes intermediate
tensors and state in HBM in producer/consumer order. Smaller value tiles reduce
capacity requirements but reread Q/K/gates and recompute chunk preparation;
those costs are included. Tiling never reduces the full persistent model state.

The NumPy tests execute the production chunk graph and compare it with an
independent recurrent oracle. They cover FP64/FP32, nonzero and zero initial
state, partial chunks, head/value tails, beta=0, and prefill-to-decode continuation.
They validate algebra before quantization, not the numerical quality of W8/A8.

### 2. Minimal MoE

`MoEConfig`, `BaseExpertRouting` subclasses and `MoEProfile` describe router work,
expert assignments, dispatch buffers, expert SwiGLU, weighted gather/combine,
and shared experts. All configured experts contribute to resident weight bytes;
only experts receiving tokens execute or stream their weights during that call.
The graph joins all active expert outputs before combining them.

The initial policies are deterministic `cyclic` and `hotspot` routes, with
recorded seed, route hash and per-expert token counts. Decode uses its logical
token offset, so its synthetic routes agree with the corresponding prefill
position. Routing is defined by batch slot and token position, not real logits
or globally unique request IDs. These are controlled load scenarios; a measured
route-trace importer remains future work.

Experts share the block's compute chiplet and mapped HBM. Dispatch/return buffers
therefore cause local HBM/NoI staging, not an expert-parallel all-to-all. Load skew
changes active weights and per-expert work, but this serial baseline cannot
predict parallel expert imbalance. `expert_placement='distributed'` fails
explicitly. Kimi retains its ungrouped router. The separate
[DeepSeek-V3 baseline](../deepseek/README.md) now implements group-limited selection
and its distinct routing dependencies.

### 3. Decoder composition and capacity

`DecoderLayerConfig` / `DecoderBlockProfile` compose existing operator graphs,
preserving namespaced edges, state versions, kernel schedules and routing
metadata. The existing model/block/profile factories handle dispatch; runtime
processing contains no model-name branches. The graph sorter now preserves its
deterministic order using a heap, avoiding repeated scans for hundreds of experts.

The model has **48,367,704,704 resident weight bytes** under the declared W8
decoder contract. Loading now checks each HBM's weight capacity, even when total
system capacity would suffice. Request admission uses graph-derived activation
peaks and persistent state; static admission remains conservative about maximum
context. Initial supported context is **4096 tokens**, explicitly narrower than
the source checkpoint's 1,048,576-token limit.

Validation covers per-expert assignment/byte conservation, active versus resident
weights, SRAM capacity transitions, pinned layer/FFN structure and per-HBM
capacity rejection. An eight-request burst returned HBM usage and reservations
to weights-only after completion. A separate cleanup regression checks that
dropping one request frees all its state objects without touching another request.

### 4. Streaming MLA and reliability checks

`MLAConfig(algorithm='streaming')` selects `MLAStreamingSchedule` and its
`OnlineSoftmaxProgram`; the historical `materialized` default remains available.
The streaming model subclasses use the existing model/profile factories. No
model-specific branches were added to the simulator runtime.

The core follows the online-softmax tiling principle described in
[FlashAttention](https://arxiv.org/abs/2205.14135), with an explicit serial
analytical schedule. Query/head tiles traverse visible key tiles in order.
Each tile updates the running maximum, denominator and weighted numerator;
final normalization happens after all key tiles. Fully future tiles are skipped,
while diagonal rectangles still charge masked arithmetic. The scale uses the
original `P+R` query width, including when queries are absorbed into `C+R`.

Q stays resident across key tiles. Liveness includes Q, private intermediates,
and both versions of FP32 running state. Oversized tiles fail with a request to
reduce tile sizes; there is no unmodeled accumulator spill or double buffering.
HBM staging loads Q once per query/head tile and rereads cache per key tile.
Absorbed C channels feed both K and V with one HBM read; no reuse between head
tiles is assumed. Cache append writes only new tokens, before attention reads.

The streaming core retains FP32 probabilities; the historical materialized core
casts probabilities to A8. Numerical tests compare unquantized algebra, including
FP64/FP32, partial tiles, extreme logits, causal decode and absorbed/expanded
layout equivalence. They do not validate quantized model quality or measured
accelerator latency. Both paths still use the shared aggregate compute rates.

The [stage validation](../../../tests/validation/kimi_models/streaming/README.md)
records operator memory/traffic, three-backend serving, raw timing/rounding,
network stress counterexamples, and an existing Chat trace replay. In particular,
equal Packet/Garnet TTFT at a coarse clock is insufficient evidence of network
accuracy under general contention.

## Remaining steps

1. Expand model/workload coverage beyond the current 4096-token contract, with
   explicit position/scaling, cache and admission checks. The
   [next workload stage](../../../tests/validation/deepseek_models/README.md)
   adds complete-length Chat/BWB requests and the DeepSeek-V3 decoder baseline.
2. Add recorded expert routes with layer/request identity, then explicit
   expert-parallel placement/dispatch/join scheduling. Group-limited routing is
   already implemented for DeepSeek-V3; distributed execution remains future work.
3. Extend other model families explicitly. DeepSeek-V3.2 still needs its sparse
   indexer, top-k cache access and gathered attention.

Multi-fidelity acceptance is a small execution/metric-output smoke test. The
previous timing/contention diagnostics remain archived; further network fidelity
calibration is not a prerequisite for this model/workload development stage.
Operation-specific compute calibration remains a separate future refinement.

Sources: [pinned model inventory](../model_inventory.json),
[inspected source hashes](../operators/source_inventory.json), and the earlier
[operator audit](../operators/README.md). No model weights or GPU kernels were
downloaded/executed for this implementation.

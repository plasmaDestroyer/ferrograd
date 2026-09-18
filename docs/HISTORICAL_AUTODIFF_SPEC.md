# Historical autodiff specification

This is the original `SPEC.md`, preserved for project history. It does not describe the current reinforcement-learning analysis package.

# ferrograd — Project Spec (v2)

> v1 (claude.ai draft) is preserved in git history. v2 fixes the thesis, locks the
> architecture, sequences around the August 2026 deadline, and adds the rematerialization
> research direction. See the design rationale in the approved plan file.

## What
A from-scratch tensor-level reverse-mode automatic differentiation engine in Rust,
organized as a **tape** (Wengert list). Trains real neural nets (MLP, then CNN) on MNIST.
Built to support **first-class tensor rematerialization** (gradient checkpointing) as a
pluggable policy — the engine can free intermediate activations and recompute them on
demand to trade memory for compute. Optional PyO3 Python bindings. Zero ML-framework deps.

## Why (honest thesis)
Reverse-mode AD must keep activations alive until backward consumes them — that is the
math, not a language choice, and PyTorch already frees them incrementally during backward.
So the contribution is NOT "Rust frees what PyTorch can't." The contribution is a clean,
tape-based engine that makes **rematerialization** a first-class, swappable scheduling
policy, plus an empirical study of the **memory↔compute tradeoff on CPU**, a regime that
existing remat work (DTR, Checkmate — all GPU-focused) has not characterized.

Positioning vs existing Rust ML (candle, burn, dfdx, tch-rs): those are general frameworks.
ferrograd is a focused research/teaching artifact whose differentiator is the tape + the
pluggable rematerialization policy layer and the CPU memory–compute study.

## Two-phase plan
- **Phase A — Working artifact (resume deliverable, by Aug 2026):** correct tape autograd
  engine, MLP trained on MNIST > 95%, grad-checked, tested, demoable. Remat-ready data
  structures in place but only the trivial `NeverEvict` policy active.
- **Phase B — Research (post-August):** rematerialization policies, the CPU memory–compute
  study, optional CNN/PyO3/type-safe API, paper.

## Core data structures
**Tape** (`Tape { nodes: Vec<Node> }`) owns all tensor storage and IS the analyzable IR.
**Node**: `data: Option<Vec<f32>>` (None = evicted), `shape`, `strides`, `grad:
Option<Vec<f32>>` (lazy), `op: Option<OpKind>` (how to recompute; None = leaf), `inputs:
SmallVec<[NodeId; 2]>`, remat metadata (byte size, last-access tick, measured recompute
cost). **Tensor** is a cheap handle `{ id: NodeId, tape: Rc<RefCell<Tape>> }` carrying
operator overloading. **OpKind**: an enum (or `dyn BackwardOp`) implementing
`backward(grad_output) -> grads-per-input` AND `forward(inputs)` for recompute.

## Supported operations
- **Elementwise:** add, sub, mul, div (tensor–tensor and tensor–scalar), neg, abs, pow;
  NumPy broadcasting.
- **Reductions:** sum (full/axis/keepdim), mean, max (fwd, for ReLU).
- **Linear algebra:** matmul (2D), batched matmul (3D), transpose, reshape, permute.
- **Activations:** ReLU, sigmoid, tanh, softmax(axis), log_softmax.
- **Loss:** MSE; cross-entropy (log_softmax + NLL fused for stability).
- **Convolution (Phase B / stretch):** conv2d via im2col + matmul forward, col2im
  backward; stride + padding.

## Backward pass
`loss.backward()` → process nodes in reverse tape order (an append-only tape is already in
topological order — a node can only reference inputs created before it) → for each node
call `op.backward(grad_output)`, accumulate into each input's `.grad` (handles fan-out).
With `NeverEvict`, saved activations live until the node's backward runs, then may be freed.
With remat policies (Phase B), `materialize(node)` recursively recomputes any evicted input
before running backward.

### Gradient correctness (built first, before ops)
Finite-difference check `(f(x+ε) − f(x−ε)) / 2ε` vs analytic grad, every op, multiple
shapes. The grad-check harness is Phase A, milestone 0 — write the test before the ops.
Note: f32 finite differences are noisy; expect to use ε≈1e-2..1e-3 and a tolerance around
1e-3 (the spec's original 1e-4 is tight for f32 — accumulate sums in f64 where it helps).

## Neural-net API
`Module { forward(&Tensor)->Tensor; parameters()->Vec<Tensor> }`. Built-ins: Linear
(Xavier init), Sequential, ReLU/Sigmoid/Softmax, Flatten, Dropout (fwd-only), Conv2d
(Kaiming init, Phase B). Biases zero-init.

## Optimizers
SGD (lr, optional momentum), `zero_grad()`. Adam = stretch.

## Rematerialization (Phase B — the research core)
Pluggable `EvictionPolicy` trait over the tape:
- `NeverEvict` (Phase A baseline).
- `ChenSqrtN` — segment the graph, keep √n checkpoints (Chen 2016).
- `DTRGreedy` — online; evict the tensor minimizing `cost(t) / (size(t) × staleness(t))`
  (DTR, ICLR 2021).
- `CheckmateILP` (optional) — offline optimal schedule via MILP (Checkmate, MLSys 2020).
Mechanism (free + recursive recompute) is shared; only the policy changes.

## Benchmarks & study
- Models: 2-layer MLP on MNIST (>95%); small CNN (>98%, Phase B).
- Measure vs PyTorch CPU, same arch/hyperparams: training throughput (samples/s over 5
  epochs), peak RSS (`/proc/self/status`), backward wall-clock, and the **memory↔compute
  Pareto frontier** as the memory budget is swept under each eviction policy.
- Honest framing: report what the RSS curves actually show and attribute causes
  empirically (allocator/runtime/framework overhead) rather than asserting them upfront.

## PyO3 bindings (Phase B / stretch)
`Tensor(data, shape, requires_grad)`, Python operators `+ * @`, `.backward()`, `.grad`
(numpy), `Linear`/`Sequential`/`SGD`, `.numpy()` zero-copy. maturin build; crates.io + PyPI.

## Project structure
src/{tensor,ops,backward,loss,init,lib}.rs, src/nn/*, src/optim/*, add
src/remat/{policy.rs, mechanism.rs} for Phase B; tests/, benches/, examples/, python/,
pyproject.toml.

## Roadmap
**Phase A (June–July 2026, the resume artifact):**
| Step | Deliverable | Done when |
|---|---|---|
| A0 | Tape skeleton, NodeId, reverse-order backward, finite-diff grad-check harness | harness runs on a trivial op |
| A1 | elementwise + broadcasting + sum/mean + matmul (fwd+bwd) | all pass grad check |
| A2 | Linear, Sequential, ReLU, log_softmax+NLL, SGD | MLP fwd+bwd on random data |
| A3 | MNIST loader + training loop | MLP > 95% test accuracy |
| A4 | README, demo, one RSS plot, polish | repo presentable on resume |

**Phase B (post-August):** remat mechanism+policies → CPU memory–compute study →
(CNN / PyO3 / type-safe API) → paper. Paper angle chosen from Phase-B data
(lean: CPU Pareto study, with type-safe/RAII-guard remat as a bonus section).

## Scope / non-goals
In: CPU only, f32, fixed shapes, MLP+CNN on MNIST, rematerialization study.
Out: GPU/CUDA, distributed, dynamic shapes, ONNX, RNN/LSTM, batch norm (stretch at best),
main-track-conference-level novelty.

## Related work (cite in paper)
- Chen et al. 2016, "Training Deep Nets with Sublinear Memory Cost" (√n checkpointing).
- Jain et al. 2020 (MLSys), "Checkmate: Breaking the Memory Wall with Optimal Tensor
  Rematerialization" (ILP-optimal).
- Kirisame et al. 2021 (ICLR), "Dynamic Tensor Rematerialization" (online greedy; the
  `cost/(size×staleness)` heuristic).
- Griewank & Walther, "revolve" (optimal AD checkpointing, classical).

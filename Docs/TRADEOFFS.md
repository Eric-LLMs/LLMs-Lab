# LLM Systems Trade-offs — Technical Deep-Dive

> Companion to the [LLM Technology Landscape & Evolution](README.md) knowledge graph. Each section unpacks one architectural trade-off — the engineering rationale, the mechanism, and the open questions driving current research.

## Overview

1. **Memory Wall (Compute-bound → Memory-bound)** — Inference bottlenecks shift from Prefill's compute-bound to Decode's memory-bound. MHA → GQA → MLA is fundamentally a KV-cache compression race: each step trades negligible model fidelity for order-of-magnitude VRAM savings, directly multiplying serving throughput.

2. **Communication vs. Compute Overlap** — TP (Tensor Parallelism) stays within a node thanks to NVLink, but cannot scale cross-node. PP (Pipeline Parallelism) and CP/SP (Context/Sequence Parallelism) use P2P transfer topologies to overlap communication with computation. DualPipe's bidirectional micro-batch injection eliminates pipeline bubbles, pushing MoE training to near-linear scaling.

3. **Precision Co-design (Software vs. Hardware)** — INT4/AWQ quantize weights in pure software, prone to outlier-induced degradation. MXFP4 (Microscaling Formats, 2025–2026) takes a hardware-software co-design approach: shared block-scale exponents across 16/32-element groups preserve FP dynamic range while unlocking native tensor-core throughput.

4. **Training Scaling → Inference Scaling (System 1 → System 2)** — The defining paradigm shift of 2024–2026. Kaplan/Chinchilla (System 1) treat model quality as a function of training tokens and parameter count. o1/R1 (System 2) flip the equation: spend compute at inference time via RL-driven chain-of-thought (GRPO/RLVR), letting the model self-correct through extended reasoning—breaking the pre-training data ceiling with test-time tokens.

---

## 1. The Memory Wall: Compute-bound vs. Memory-bound

### Why It Matters

LLM inference has two phases with fundamentally different bottlenecks:

| Phase | Bottleneck | Why |
|-------|-----------|-----|
| **Prefill** (prompt processing) | Compute-bound | All input tokens processed in parallel; tensor cores saturated |
| **Decode** (token generation) | Memory-bound | One token at a time; each step reads the entire KV-cache from HBM |

For a 70B model serving a 4K context with batch-size 1, the decode step loads ~3 GB of KV-cache from HBM to compute a single token. That's ~3 GB of memory traffic for ~140 GFLOPS of compute—an arithmetic intensity below 0.05, firmly in the memory-bound regime.

### The KV-Cache Compression Race

```
MHA  →  MQA  →  GQA  →  MLA
(1x)    (1/H)    (1/G)    (~1/60x)
```

- **MQA (Multi-Query Attention, 2019)**: All query heads share one KV head. KV-cache shrinks by `1/H`. Works, but hurts model quality noticeably.
- **GQA (Grouped-Query Attention, 2023)**: Group query heads into G groups, each sharing one KV head. The compromise between MHA quality and MQA efficiency. Llama 2/3 use G=8.
- **MLA (Multi-head Latent Attention, DeepSeek-V2, 2024)**: Low-rank joint compression of K and V into a latent vector. Decouples RoPE from the compressed representation. KV-cache drops from ~30.5 GB (MHA-equivalent) to ~0.51 GB on DeepSeek-V3—a ~60× reduction.

**The trade-off**: Each step trades minimal model capacity for massive KV-cache reduction. MLA adds computation (up-projection from latent) but the net throughput gain from serving larger batches dominates.

### Going Deeper

**KV-cache is per-request, so batch size directly multiplies VRAM pressure.** Batch-32 × 4K context × 70B model = ~96 GB of KV-cache alone. Without compression like GQA/MLA, VRAM runs out before compute can be saturated—this is the fundamental reason KV-cache optimization matters more than raw FLOPs for serving throughput.

**FlashAttention and KV-cache are complementary, not overlapping.** FlashAttention reduces *activation memory* during attention computation (the Q·K^T intermediate matrix), not the KV-cache itself. FlashAttention handles the compute path; GQA/MLA handles the storage path. Both are needed for end-to-end serving efficiency.

---

## 2. The Communication Bottleneck in Distributed Scale

### Why It Matters

As models grow to MoE architectures (DeepSeek-V3: 671B total, 37B active), no single GPU can hold the model. The bottleneck shifts from FLOPs to **communication overhead**.

### Parallelism Strategies: Communication Patterns

| Strategy | Communication | Scope | Bottleneck |
|----------|--------------|-------|------------|
| **TP** | All-Reduce per layer | Intra-node (NVLink) | NVLink bandwidth (~900 GB/s on H100) |
| **PP** | P2P send/recv at cut points | Cross-node friendly | Pipeline bubbles |
| **DP/ZeRO** | All-Reduce per step | Cross-node | IB/RoCE bandwidth (~400 GB/s) |
| **CP/SP** | P2P ring transfer | Cross-node | Overlap efficiency |

**The critical insight**: TP is *only viable intra-node*. TP's per-layer All-Reduce requires extremely high bandwidth and low latency. NVLink provides this within an 8-GPU node. But cross-node TP over InfiniBand would introduce ~10× more communication latency per layer, making it impractical beyond a single node.

### Pipeline Bubbles and DualPipe

Standard 1F1B (one-forward-one-backward) pipeline scheduling leaves bubbles—idle time where GPUs wait for the preceding stage. The bubble ratio is `(PP_size - 1) / (num_microbatches)`.

**DualPipe** (DeepSeek-V3, 2024) innovates by:
1. Injecting micro-batches from *both ends* of the pipeline simultaneously
2. Interleaving attention computation with all-to-all dispatch/combine communication
3. Reducing bubble time from `(PP-1)(F+B)` to `(PP/2-1)(F&B+B-3W)`

### Going Deeper

**TP is viable only within a single node, and this is a hard constraint, not a soft preference.** TP's per-layer All-Reduce demands extremely high bandwidth and low latency—NVLink (~900 GB/s on H100) provides this within an 8-GPU node. Attempting TP across nodes over InfiniBand (~400 GB/s, higher latency) would introduce ~10× more communication delay per layer, making cross-node TP infeasible beyond trivial model sizes. In practice: TP within node, PP/DP across nodes, CP/SP for long-context attention.

**RingAttention uses P2P rather than All-Reduce because attention computation is sequential, not aggregative.** All-Reduce collects and broadcasts the full KV; but attention needs each device to see *different* KV blocks one after another. The ring topology maps perfectly: compute attention on the current KV block while asynchronously sending that block to the next device. This communication-computation overlap is what makes million-token contexts feasible.

---

## 3. Precision vs. Generalization — Hardware-Software Co-design

### Why It Matters

Running inference in FP16/BF16 wastes memory bandwidth. Quantization recovers throughput, but the naive approach—uniformly shrinking bit-width—hits an outlier wall.

### Software-only Quantization (INT4, AWQ, GPTQ)

- **GPTQ (2022)**: Layer-wise optimal brain quantization. Reconstructs weights column-by-column, compensating for quantization error in remaining columns.
- **AWQ (2023)**: Observes that <1% of weight channels are "salient" (high magnitude). Scales those channels up before quantization, preserving their precision at minimal cost.
- **The problem**: Activations still have outliers. A single outlier activation value multiplied by quantized weights can cause large errors. SmoothQuant shifts quantization difficulty from activations to weights via per-channel scaling.

### The MXFP4 Shift (2025–2026)

Microscaling Formats (MX) are not a quantization algorithm—they are a *hardware-native data format*:

| Format | Bits | Exponent | Mantissa | Key difference from INT4 |
|--------|------|----------|-----------|--------------------------|
| INT4 | 4 | 0 | 4 (unsigned) | No dynamic range; relies on per-group scale calibrated offline |
| MXFP4 | 4 | 1 (shared) | 3 | Shared block exponent across 16–32 elements provides FP-like dynamic range |

**The co-design**: NVIDIA Blackwell B200 and subsequent chips have native MXFP4 tensor-core support. Unlike INT4 which requires dequantization before compute, MXFP4 tensor cores operate on the compressed format directly, delivering 2× throughput over FP8 with negligible accuracy loss.

### Going Deeper

**FP8 is production-ready since Hopper (H100), so why push to MXFP4?** Because FP8 halves memory vs. BF16, but the decode phase is still memory-bound. MXFP4 halves memory again vs. FP8, directly translating to 2× batch throughput in decode where KV-cache and weights dominate. The incremental accuracy cost (0.1–0.5% vs. FP8's <0.1%) is small enough that the throughput gain wins for most production workloads.

**The accuracy hierarchy**: BF16 (baseline, zero loss) > FP8 (E4M3, <0.1% loss) > MXFP4 (0.1–0.5% loss) > INT4 (0.5–2% loss). Larger models (70B+) are more quantization-resilient because their weight distributions are smoother and individual outlier channels matter less. The choice of precision format depends on model size, the target hardware generation, and the acceptable accuracy budget.

---

## 4. Training Scaling → Inference Scaling (System 1 → System 2)

### The Paradigm Shift

For most of 2017–2023, the dominant narrative was: *bigger models + more data = better performance*.

```
Kaplan Scaling Laws (2020):   Loss ∝ N^(-0.076) × D^(-0.095)
Chinchilla Laws (2022):       N_opt ≈ D_opt (equal scaling of params and tokens)
```

Both treat intelligence as a function of **training compute**. This is System 1 thinking—fast, intuitive, single-pass.

**The 2024 inflection**: OpenAI o1 (Sept 2024) demonstrated that spending more compute *at inference time*—generating and evaluating multiple reasoning chains—produces qualitatively different capabilities. DeepSeek-R1 (Jan 2025) proved this could be done with open weights using GRPO.

### The Mechanism: RL-driven Chain-of-Thought

| Aspect | Classic Alignment (RLHF/PPO) | Reasoning RL (GRPO/RLVR) |
|--------|------------------------------|--------------------------|
| **Goal** | Helpful, harmless, honest | Correct reasoning, self-correction |
| **Reward source** | Human preference labels | Verifiable outcomes (math answer, code execution) |
| **RL algorithm** | PPO (actor + critic) | GRPO (no critic, group-relative baseline) |
| **Key paper** | InstructGPT (2022.1) | DeepSeekMath (2024.2), DeepSeek-R1 (2025.1) |
| **Emergent behavior** | Instruction following | Chain-of-thought exploration, self-verification |

**Why GRPO drops the critic network**: In PPO, the critic model estimates the value function—this doubles memory consumption. GRPO samples a *group* of outputs, scores all of them, and uses the group mean as a baseline. No critic needed, VRAM halved, training stabilizes on verifiable tasks.

### RLVR: The Abstraction Layer

RLVR (Reinforcement Learning with Verifiable Rewards) is the conceptual framework. The reward function is a deterministic verifier:
- Math: `check(answer == ground_truth)`
- Code: `run(test_suite) → pass/fail`
- No learned reward model, no human labels

This is why o1 and R1 are strongest in math and coding—those domains have clean, automatable verifiers. Open-ended generation (creative writing, strategic planning) lacks this property and remains a harder challenge for RLVR-based methods.

### Going Deeper

**The verifier gap: why GRPO excels at math/code but struggles with open-ended generation.** GRPO depends on a reliable, automated reward signal. For math and code, the verifier is deterministic—`check(answer)` or `run(tests)` returns a clean binary. For creative writing or strategic analysis, "quality" is inherently subjective. LLM-as-judge can approximate it, but reward noise is orders of magnitude higher. Without clean rewards, the group-relative baseline in GRPO drifts and training can collapse into reward hacking. Research on noisy-verifier RL (RLVεR, Spurious Rewards) is tackling this gap, but it remains the hardest open problem in RLVR scaling.

**The inference cost of System 2 thinking is not negligible.** o1 and R1 can generate thousands of reasoning tokens before producing a final answer—10× to 100× more than a single-pass LLM call. This cost scales linearly with chain length and is the primary limitation for latency-sensitive applications. The bet (validated by o1/R1 benchmarks) is that certain problem classes—competitive math, formal verification, multi-step code generation—are worth the extra compute. As hardware efficiency improves (MXFP4, persistent kernels, better KV-cache management), test-time scaling will gradually become viable for a wider range of queries.

**Pre-training is not dead—it has been repositioned.** DeepSeek-R1 required a strong pre-trained base (DeepSeek-V3). The relationship is: pre-training sets the capability floor; inference-time RL provides the reasoning multiplier. A weak base model + GRPO produces structured noise; a strong base model + GRPO produces structured chain-of-thought. The scaling paradigm has expanded from one dimension (training tokens × parameters) to two (training compute + inference compute), but both are load-bearing.

---

## Further Reading

The [main README](README.md) knowledge graph gives the *what* and *when* across 13 modules. These notes provide the *why* behind four of the most consequential architectural trade-offs in the LLM stack. Each trade-off is an active research area as of 2026, with open problems and evolving hardware assumptions.

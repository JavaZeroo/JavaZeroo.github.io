# Kimi K3 report (arXiv 2607.24653) — verbatim text of §3.3–3.4, §5, Appendix E

> Extracted from the arXiv HTML on 2026-09-05 for writing the 系统篇 (posts 8–10). Math is inline $...$ alttext.

## 3.3 Training Recipe

Kimi K3 adopts a native multimodal training strategy in which language and vision are jointly optimized from the start of training, rather than grafting a vision encoder onto a pre-trained language model through a post-hoc alignment stage. Under this paradigm, visual and textual tokens are interleaved within a single next-token prediction objective, enabling the shared backbone to learn unified multimodal representations from the outset.

We optimize the model using the Per-Head Muon optimizer (§ 2.5) together with the weight-clipping mechanism introduced in Kimi K2, while adopting QB (§ 2.3.3) for MoE load balancing. We use a cosine learning rate schedule with a  $1\%$  linear warmup. Weight decay is set to  $0.1$  throughout.

Our pre-training begins with a context length of  $8\mathrm{k}$  tokens, which is later extended to  $64\mathrm{k}$  tokens in a subsequent training phase.

## 3.4 Long-Context Extension

## Positional encoding

Kimi K3 uses no explicit positional embedding (NoPE), and instead encodes positional information implicitly through the recurrent gating and decay mechanism of KDA.
As a result, the model extrapolates directly to 1M-token contexts without any positional-encoding modification, such as RoPE rescaling or interpolation [92].

## Long-context data

Long documents and videos from natural sources contain a substantial amount of low-quality content, including near-duplicates, binary blobs, truncated files, video clips, and invalid machine-generated logs.
We therefore process them through a dedicated cleaning pipeline that combines exact and fuzzy deduplication, supplemented by perceptual hashing over frames for video, together with heuristic and classifier-based quality filtering, and structural validation.
Because genuinely long and coherent documents and videos are scarce relative to short text, we upsample them so that the long-context distribution is not overwhelmed by short sequences during cooldown.
Length alone, however, does not confer long-range capability.
To address this, we synthesize additional long-context data by carefully permuting and concatenating multimodal documents and sub-tasks, so that the embedded tasks can be solved only by attending to information scattered across the full 1M-token context.
This trains the attention mechanism at the intended scale and prevents it from degenerating into local patterns.

## Progressive context extension

Kimi K3 supports a context window of up to 1 million tokens.
We achieve this through extending the context window progressively as training proceeds, following a four-stage curriculum.
The window grows from 8K to 64K tokens during pre-training, and from 256K to 1M tokens during the cooldown phase.
Concentrating the costly long-sequence computation within a small fraction of the overall training budget keeps the curriculum economical while still allowing the model to adapt gradually to increasingly long-range dependencies.
The sequence-dimension partitioning that makes million-token training tractable for the KDA layers is described in §5.1.2.

## 4 Post-Training
## 5 Infrastructure

Kimi K3 combines three system challenges rarely encountered in a single model: hybrid KDA attention, 3T-class sparse multimodal training and inference, and million-token agentic workloads.
Our infrastructure is co-designed with these challenges across the model lifecycle.
At the architecture level, high-performance KDA kernels and Context Parallelism make the recurrent formulation efficient within and across devices, in both training and inference.
During pretraining, balanced expert execution, reduced memory footprint, and communication-overlapped scheduling sustain high utilization at scale.
During 1M-token agentic RL, hierarchical state management and resumable sandbox execution preserve long trajectories across iterations.
Finally, state-aware KDA prefix caching, specialized inference kernels, and cache- and budget-aware scheduling translate these efficiencies into predictable production serving.

## 5.1 Algorithm-System Co-Design for KDA

KDA replaces the growing key–value cache of softmax attention with a fixed-size recurrent state  $\mathbf{S}\in\mathbb{R}^{d_{k}\times d_{v}}$  (§2.1.1), whose serial update poses challenges in parallel execution, in exchange for a fixed-size state that is cheap to transfer and reuse.
The designs below address the first property and exploit the second at two levels of execution, with fused kernels within a device and KDA Context Parallelism across devices.

## 5.1.1 KDA Kernels across Regimes

The serial dependence of the KDA state is at odds with the GPU’s preference for wide, uniform parallelism, and it manifests as a different bottleneck in each execution regime.
We design a dedicated kernel for each regime.

## Chunkwise kernel for training and prefill

The chunkwise form of KDA is parallel within each chunk but serial across chunks, since the recurrent state must propagate from chunk to chunk.
Executed naively, these two phases alternate, leaving the SMs idle during the serial propagation.
We therefore develop FlashKDA [13], a CUTLASS-based chunkwise kernel that overlaps intra-chunk computation with cross-chunk state propagation.
The kernel decomposes the work into token-parallel stages and a head-parallel recurrence, each scheduled and tuned independently, and substantially outperforms the Triton reference implementation.
FlashKDA serves both training and inference prefill and is auto-dispatched as a backend of flash-linear-attention [142].

## Intra-device context parallelism for long-context prefill

Tensor parallelism partitions heads across devices but never shortens the recurrence, so under pure TP deployment, prefilling an ultra-long sequence leaves most SMs idle when each rank holds only a few heads.
The key observation is that the state transition of each segment can be evaluated independently of the incoming state and composed exactly afterward.
An automatic SM-level context-parallel (CP) planner [143, 142] therefore partitions the sequence across the SMs of a single rank, evaluates the segment transitions in parallel, and merges them to recover each segment’s exact initial state.
In contrast to the cross-device KCP of §5.1.2, this parallelism is entirely intra-device and incurs no cross-device communication.

KDA decoding presents challenges distinct from those encountered during training and prefill. We discuss these challenges in detail in §5.4.2.

## 5.1.2 KDA Context Parallelism

The communication overhead of context parallelism differs fundamentally between softmax and linear attention.
Softmax attention requires ranks to exchange key–value blocks whose size grows with the sequence length [72].
Linear attention instead carries the preceding context in a fixed-size recurrent state  $\mathbf{S}\in\mathbb{R}^{d_{k}\times d_{v}}$ .
Prior context-parallel methods exploit the additive recurrence of vanilla linear attention by computing, on each rank, the state that the local tokens generate from  $\mathbf{S}=\mathbf{0}$  and summing these local states over the preceding ranks to recover the incoming state [115, 114].

This direct summation, however, is insufficient for KDA.
Recall from Eq. 1 that KDA updates its state as  $\mathbf{S}_{t}=\mathbf{M}_{t}\mathbf{S}_{t-1}+\beta_{t}\bm{k}_{t}\bm{v}_{t}^{\top}$ , where  $\mathbf{M}_{t}:=\left(\mathbf{I}-\beta_{t}\bm{k}_{t}\bm{k}_{t}^{\top}\right)\operatorname{Diag}(\bm{\alpha}_{t})$ .
KDA’s delta rule applies the token-dependent matrix  $\mathbf{M}_{t}$  to the incoming state before adding the current write.
Consequently, the effect of a local sequence segment depends on the state entering that segment and cannot be determined from the state computed with  $\mathbf{S}=\mathbf{0}$  alone.

To preserve this dependence, we introduce KDA Context Parallelism (KCP), which decomposes the effect of each segment into two locally computable quantities, a cumulative transition acting on the incoming state and a state generated locally from zero.
Following the chunkwise notation of §2.1.1, we write  $\mathbf{S}_{[i]}^{t}$  for the recurrent state within the segment of rank  $i$  after  $t$  local tokens, so that  $\mathbf{S}_{[i]}^{T_{i}}$  denotes the state leaving rank  $i$  and entering rank  $i+1$ .
We write  $\widetilde{\mathbf{S}}_{[i]}^{t}$  for the state of the same recurrence started instead from  $\mathbf{S}=\mathbf{0}$ .
For an arbitrary state entering the  $(i+1)$ -th of  $P$  context-parallel ranks, the state after  $t$  local tokens is

 $\displaystyle\mathbf{M}_{[i+1]}^{t\leftarrow 1}:=\prod_{r\leftarrow 1}^{t}\mathbf{M}_{r}\in\mathbb{R}^{d_{k}\times d_{k}},\qquad\mathbf{S}_{[i+1]}^{t}$ 
 $\displaystyle=\widetilde{\mathbf{S}}_{[i+1]}^{t}+\mathbf{M}_{[i+1]}^{t\leftarrow 1}\mathbf{S}_{[i]}^{T_{i}}$ 

(17)

 $\displaystyle=\widetilde{\mathbf{S}}_{[i+1]}^{t}+\mathbf{M}_{[i+1]}^{t\leftarrow 1}\sum_{j=1}^{i}\Big(\prod_{l\leftarrow j+1}^{i}\mathbf{M}_{[l]}^{T_{l}\leftarrow 1}\Big)\widetilde{\mathbf{S}}_{[j]}^{T_{j}}\in\mathbb{R}^{d_{k}\times d_{v}}.$ 

            

 $=$  $\prod_{r}$  $\Biggl($  $\Bigl($  $-$  $\times$  $\Bigr)$  $\Biggr)$  $=$  $+$  $\sum\nolimits_{j}$  $\mathbf{M}_{[i+1]}^{t\leftarrow 1}$  $\mathbf{I}-\beta_{r}\bm{k}_{r}\bm{k}_{r}^{\top}$  $\operatorname{Diag}(\bm{\alpha}_{r})$  $\mathbf{S}_{[i+1]}^{t}$  $\widetilde{\mathbf{S}}_{[i+1]}^{t}$  $\mathbf{M}_{[i+1]}^{t\leftarrow 1}$  $\prod_{l}\mathbf{M}_{[l]}^{T_{l}\leftarrow 1}$  $\widetilde{\mathbf{S}}_{[j]}^{T_{j}}$ 

          

where  $\mathbf{M}_{[i+1]}^{t\leftarrow 1}$  denotes the cumulative transition of the first  $t$  local tokens.
The first term contains the state generated by the local tokens, whereas the second term propagates the context from preceding ranks through the local KDA updates.
At  $t=T_{i+1}$ , both quantities  $\mathbf{M}_{[i+1]}^{T_{i+1}\leftarrow 1}$  and  $\widetilde{\mathbf{S}}_{[i+1]}^{T_{i+1}}$  can be computed using only the local tokens, before  $\mathbf{S}_{[i]}^{T_{i}}$  is available, and are the fragments each rank exchanges with the others.

The summation in Eq. 17 shows that every state is composed purely from locally computed fragments.
These rank-level updates compose associatively, so the incoming state of each rank can be recovered by a prefix scan [77].
Each rank first computes  $\mathbf{M}_{[i]}^{T_{i}\leftarrow 1}$  and  $\widetilde{\mathbf{S}}_{[i]}^{T_{i}}$  locally, then exchanges both tensors with one all-gather [142].22
                2
                
                
                
              The construction builds on DeltaNet context parallelism [143].
The KDA implementation is available in FLA PR #691.

After the all-gather, rank  $i+1$  reconstructs  $\mathbf{S}_{[i]}^{T_{i}}$  by processing preceding fragments of the same document in order, starting from  $\mathbf{S}=\mathbf{0}$  and applying  $\mathbf{S}\leftarrow\mathbf{M}_{[j]}^{T_{j}\leftarrow 1}\mathbf{S}+\widetilde{\mathbf{S}}_{[j]}^{T_{j}}$  at each fragment.
Therefore, KCP requires only a fixed-size all-gather for recurrent-state synchronization and achieves linear compute scaling.

## 5.2 Infra for 3T-class Pre-Training

Kimi K3 pre-training combines Pipeline Parallelism (PP) with virtual stages (VP) [48, 81], Expert Parallelism (EP) [66], ZeRO-1 Data Parallelism [100], Pipeline ZeRO-2 gradient sharding [146], and Context Parallelism (CP, §5.1.2) [50].
The MoE layers employ shared experts replicated across EP ranks, and the all-to-all communication for expert dispatch and combine is overlapped with computation to hide its latency.

Natively multimodal pre-training at the 3T-class poses three critical problems: (i) token loads are imbalanced across EP ranks; (ii) activations, gradients, and optimizer states exceed the memory budget; and (iii) the vision encoder’s highly variable computation is exposed on the critical path.
The following subsections address these problems in turn: perfectly balanced expert-parallel MoE training (§5.2.1), memory-efficient training (§5.2.2), and multimodal encoder optimization (§5.2.3).
Fig. 11 illustrates the resulting execution schedule.

Figure 11: Computation, communication and offloading overlapped in different PP phases.

## 5.2.1 Perfectly Balanced Expert-Parallel MoE Training

In conventional EP schemes, token loads are imbalanced across ranks.
The resulting computational imbalance degrades training throughput, and the dynamically varying shapes of routed-expert activations cause substantial memory fragmentation.
We therefore propose MoonEP33
                3
                
                
                
              https://github.com/MoonshotAI/MoonEP, an EP scheme that achieves perfect load balance with dynamic redundant experts.
MoonEP preserves the overall computation flow of conventional schemes such as DeepEP [148] and additionally introduces online planning and migration of redundant experts.
In the forward pass, we plan the redundant experts from the router outputs of the current micro-batch and layer and prefetch them before the routed-expert computation.
In the backward pass, we stage their gradients in a local reduce buffer and, once the computation completes, reduce them back to the gradient buffers of their home ranks.

## Perfect balance with bounded redundant experts

MoonEP requires every rank to receive exactly  $S\times K$  tokens, where  $S$  is the sequence length and  $K$  is the number of experts selected per token, so that all ranks perform identical amounts of computation.
The key question is how many redundant experts suffice to guarantee such a balance.
Let  $E$  be the number of experts and  $R$  the EP size.
We prove that a balanced plan always exists with at most  $E/R$  redundant experts per rank and that this bound is essentially tight (§ E).
Reserving  $E/R$  redundant-expert slots per rank therefore guarantees that planning always admits a feasible solution, so training is never interrupted.
In contrast, prior work such as ECHO [138] and UltraEP [133] presets the number of redundant experts or imposes a per-rank token cap.
Training is then forced to stop whenever no feasible plan exists within the cap, and the cap itself requires manual tuning while still leaving residual imbalance.

## Online planning

Computing the exact optimum at every training step is prohibitively expensive.
We therefore compute exact solutions offline with integer linear programming (ILP) for representative cases as references and design a GPU planning kernel that is near-optimal, incurs negligible overhead, and always respects the  $E/R$  upper bound.

## Zero-copy communication

Perfect balance also simplifies the communication path.
We implement a fused permute/unpermute operator in which the planning kernel precomputes the destination of every token, so tokens are sent directly to their expert-grouped positions on remote ranks, and views of the communication buffer are returned directly to the computation, eliminating intermediate copies.
Under worst-case imbalance, supporting the same copy-free data path in DeepEP requires a communication buffer of size  $S\times K\times R$ , whereas MoonEP requires only a fixed  $S\times K$  buffer owing to the perfect balance.

## Sync-free execution with static shapes

In conventional MoE implementations, the per-expert token counts vary across steps and layers, and the host must synchronize with the device at every layer to obtain the actual computation shapes before launching the expert computation, stalling the pipeline between layers.
With perfect balance, every rank receives exactly  $S\times K$  tokens and the computation shapes of all layers are statically known. This eliminates the per-layer MoE host synchronization and alleviates the host-side kernel-launch overhead.

## Expert-GEMM scheduling and overlap

Even with the aggregate load perfectly balanced across ranks, the per-expert token counts within each rank remain skewed, and a fixed-order, workload-oblivious schedule turns this skew into an imbalanced makespan across SM workers.
We therefore schedule the routed-expert GEMM with a workload-aware scheduler that adapts its parameters to the current token distribution before launch and keeps them fixed during execution.
A lightweight heuristic selects these parameters using an analytical cost model of hardware metrics, with key coefficients calibrated through offline autotuning.
For the shared experts, we dispatch their GEMMs to a separate stream so that they overlap with other kernels.

## 5.2.2 Memory-Efficient Training

## Unified activation manager

We design a unified storage abstraction for activations, in which every tensor saved for the backward pass is associated with a pluggable storage backend.
Recomputation, quantization, and offload/remote-offload are merely storage policies under this abstraction and can be freely composed at tensor granularity; policies are declared via lightweight annotations on tensors, fully decoupled from the model code.
Recomputation is performed at function granularity, which supports cross-layer recomputation.
In our implementation, all GPU memory is allocated on the main compute stream and managed within a single memory pool, avoiding multi-stream fragmentation and host-bound overhead; activations are prefetched back at layer granularity and overlapped with computation, introducing negligible extra overhead.
In Kimi K3, most activations use block-wise FP8 quantization [58, 27] combined with offload/remote-offload, and element-wise operators are configured with recomputation.

## Memory-efficient MoE

In the native MoE implementation, the gradient computation of permuted probs depends on the forward output output.
Inspired by SonicMoE [41], we rewrite this gradient through a mathematical transformation into a form that depends only on the intermediate activation act_output and the upstream gradient doutput, eliminating the backward dependency on output at the cost of an additional lightweight element-wise computation.
Furthermore, in the forward pass of the group GEMM, we save only the input of the dispatch operation; during the backward pass, the input of the group GEMM is recovered by recomputing dispatch.
As shown in Fig. 11, the communication introduced by this recomputation can be overlapped with part of the group-GEMM backward computation, eliminating this portion of activation storage at a negligible cost.

## Memory-efficient Attention residual

For the attention residual, we design a companion optimization based on Block AttnRes.
The block representation is generated once at the boundary layer and shared by all subsequent layers, residing directly on the GPU.
The AttnRes computation is entirely wrapped with checkpointing, so the activation saved for the backward pass at each layer is identical to that of the standard residual architecture.
For pipeline parallelism, we adopt cache-based pipeline communication [60], in which only newly generated blocks are incrementally transferred between stages and released as soon as the micro-batch finishes, reaching the theoretical lower bound on memory footprint.

## Balancing activations across PP ranks

Under interleaved 1F1B pipeline parallelism, activations are unevenly distributed across PP ranks due to pipeline warmup, and the number of resident activations decreases as the PP rank increases.
To avoid out-of-memory (OOM) errors, we remotely offload activations to the memory of other PP ranks using the Mooncake Transfer Engine [96], achieving balanced activation memory across PP ranks.

## Pipeline ZeRO-2 gradient sharding and offloading

Beyond activations, we use Pipeline ZeRO-2 gradient sharding [146] to shard gradients across data-parallel (DP) ranks.
Furthermore, we store the sharded gradients in CPU memory to reduce peak GPU memory usage, while keeping the double grad buffer on the GPU.
After gradients are reduced across DP ranks into the double grad buffer, they are accumulated into the CPU shards.

## P2P-based Muon orthogonalization

The distributed optimizer shards parameters evenly across DP ranks, whereas the Newton–Schulz orthogonalization in Muon requires the full parameter matrix, necessitating a communication step to gather complete parameters before each update.
The naive approach performs an all-gather over the entire parameter buffer on every rank [73], which incurs a substantial memory footprint on top of making communication the primary bottleneck at scale.
Instead, each rank retrieves only the shards of its locally owned parameters via peer-to-peer (P2P) communication with the corresponding owner ranks, eliminating the full-parameter buffer and reducing both memory usage and communication volume.
Communication and computation are further pipelined at the granularity of model-chunk buffers, hiding the communication overhead.

## 5.2.3 Multimodal Encoder Optimization

## Dynamic CP in multimodal encoder

In long-context multimodal training, large images and long videos substantially increase the computation time of the vision encoder and cause significant load imbalance across devices.
To address this, we extend context parallelism to such large samples.
A single large image is partitioned along the patch dimension across multiple devices, and attention is computed by gathering key–value pairs (gather-KV) across CP ranks.
In addition, we divide each CP group into several sub-CP groups and distribute multiple large images across them in a load-balanced manner, preventing the communication fraction from growing with scale.
This reduces both the encoder latency of large visual samples and the cross-device load imbalance, allowing the remaining encoder computation to be hidden in pipeline bubbles.

## Encoder computation in PP bubbles

In Kimi K2.5, we introduced the Decoupled Encoder Process (DEP) [61], which splits ViT and text training into separate stages and balances vision forward and backward passes across PP stages.
We observe that, under the interleaved 1F1B pipeline schedule, the text forward passes of the first PP micro-batches are all scheduled at the very beginning, while the text backward passes of the last PP micro-batches finish only at the very end.
We therefore further decompose the ViT computation [33].
The ViT forward passes of the first PP micro-batches are executed synchronously upfront, the remaining forward passes are scheduled into pipeline bubbles, and the backward passes are handled analogously.
As a result, most of the ViT computation is hidden within pipeline bubbles, largely eliminating the effective overhead of the vision encoder.

## 5.3 Infra for 1M Agentic RL

Scaling agentic RL for a model as large as Kimi K3 to million-token contexts under a bounded compute budget makes resource efficiency a first-order goal. We therefore develop long-context RL infrastructure for efficient training and rollout, together with high-performance, resumable sandboxes for long-horizon environment interaction.

## 5.3.1 Long-context RL infrastructure

We adopt co-located RL training [58] to keep each 1M-context Kimi K3 RL experiment within a few hundred GPUs, and use partial rollouts [119] to reduce tail latency from ultra-long trajectories. This design improves hardware utilization, but long-context rollouts introduce extra DRAM demand for KV-cache retention, which competes with training-side states. Further, achieving high efficiency for both prefill and decoding requires careful prefix management and request scheduling.

## External KV cache pool

At 1M-context multi-step rollout, a prefix KV-cache miss is extremely expensive. Partial rollout exacerbates this issue at the beginning of each iteration, due to many unfinished long prefill requests from the previous iteration arriving at the same time. Speculative decoding further accelerates request turnover within relatively fixed tool-call intervals, increasing prefix-block churn. These issues can trigger preemption and lower the cache hit rate, which is critical for long-context RL.

We therefore decouple prefix retention from GPU residency with a write-back design.
Active decoding blocks remain in GPU KV cache, while reusable idle prefixes are written back to an external KV cache pool in CPU DRAM only when it is evicted from GPU, and is prefetched back before the next reuse.
KDA states are offloaded and prefetched together with the corresponding MLA KV cache blocks, keeping their lifecycles aligned.
Compared with a write-through strategy, this policy incurs CPU DRAM usage and transfer bandwidth only for prefixes that leave the active decode path, avoiding redundant CPU copies of blocks that are still resident and active on GPU.

To provide sufficient DRAM for the external pool, we offload training states (model weights and optimizer states) to NVMe after a training iteration finishes. After a rollout iteration, the pool is released to avoid contention with training workloads.

## Rollout auto-throttling scheduler

In multi-step rollout, contexts grow progressively as the trajectory advances, making fixed concurrency based on the full-trajectory average length both hard to estimate and overly conservative early on.
Conversely, setting concurrency too high creates KV cache pressure in later stages and can trigger preemption.
We therefore design an auto-throttling mechanism at the LLM request scheduling layer, using runtime signals such as active request count, queued request count, and KV cache utilization to dynamically control how many requests are sent to the inference engine.
This keeps early rollout well utilized while reducing concurrency as KV cache pressure rises, avoiding both under-saturation and overload without manual tuning.

## Gradient-buffer reuse for non-policy model forwarding

RL loss computation often requires forward-only non-policy models, such as reference models, whose weights are too large to keep resident on GPU.
We keep these weights in CPU memory and materialize them only when needed, backing their parameter tensors with the policy model’s FP32 gradient-buffer storage.
This reuses existing GPU memory without extra allocation or fragmentation, and remains safe because the buffers are overwritten when real gradients are later computed.

With ZeRO-2 gradient sharding and offloading (§ 5.2.2), each GPU retains gradient buffers for only two VPP chunks in Kimi K3 RL training.
We stream reference weights into these slots chunk by chunk: one slot is used for the current forward computation while the other prefetches the next chunk, hiding copy overhead without increasing GPU memory.

## 5.3.2 Sandbox Infrastructure

We employ multiple sandbox runtimes to support the diverse requirements of Kimi K3 post-training and evaluation, including a traditional container-based runtime, a GPU sandbox runtime, and, most notably, a new microVM-based sandbox runtime called AgentENV.

AgentENV44
                4
                
                
                
              AgentENV is open-sourced at https://github.com/kvcache-ai/AgentENV, developed in collaboration with our partners, is a sandbox system specifically designed for agentic AI workloads.
It is built around three core design goals:

• 

High-fidelity isolated sandbox runtime As agents become more capable and tasks more difficult, they tend to explore more aggressively and may even attempt reward hacking.
On the one hand, this poses unique security challenges: in our early experiments with traditional container-based sandbox runtimes, we observed several kernel panics and deadlocks caused by unintended agent operations.
On the other hand, we want to permit as much exploration as possible so as not to constrain agent capability, and complex tasks require a sandbox close to a real-world environment — for example, agents should be able to mount disks, run containers, or even launch virtual machines at will.
By running isolated microVMs with Firecracker [2], AgentENV provides a level of isolation and fidelity that container-based runtimes cannot match.

• 

Flexible sandbox life-cycles for agentic RL At the low level, AgentENV supports incremental checkpointing and resuming of sandbox states, where only memory pages dirtied since the last checkpoint are saved during checkpointing, achieving checkpoint and resume latencies as low as 133 ms and 49 ms, respectively.
On top of this, AgentENV provides three high-level operations that help improve agentic RL efficiency.
(a) Pause and Resume: a paused sandbox consumes no memory or CPU resources; a sandbox can therefore be paused while the agent is waiting for the model’s inference result, which can account for as much as 98% of the sandbox lifetime.
(b) Fork: fork creates a new sandbox from the exact state of the original one while keeping the original running, which is useful for reward judging without side effects.
(c) Snapshot: snapshots of a sandbox can be saved at regular intervals for error recovery.

• 

High efficiency and high density In our workloads, tens of thousands of sandboxes, each with a unique set of images, may need to be created within seconds.
We adopt OverlayBD [68] as the image format, together with a custom ublk driver implementation, storage-layer sharing, and P2P transport, achieving sub-second launch latency at large scale.
We further reduce memory usage with copy-on-write memory and page-cache optimizations, achieving a memory overcommit ratio of up to 6.5 $\times$  in real workloads.

Throughout Kimi K3’s training and evaluation, a total of 51,219,741 sandboxes across 1,505,678 images were created.

## 5.4 Inference and Online Serving

Serving Kimi K3 exposes the same challenges from the production side: the hybrid KDA–MLA architecture maintains two fundamentally different caches that must be managed jointly at million-token contexts, its new modules and highly sparse experts demand kernels tailored to each, and production traffic mixes requests whose per-request cost spans three orders of magnitude.
The designs below address these challenges at three levels.
At the engine level, a KDA-aware prefix cache packs the fixed-size recurrent state into the same paged pool as the MLA KV cache and keeps long prefixes reusable across requests.
At the device level, dedicated kernels for KDA decoding, Block AttnRes, and the sparse latent MoE minimize per-token latency and memory traffic.
At the fleet level, cache-aware affinity scheduling and budget-based admission control translate these efficiencies into predictable serving.

## 5.4.1 KDA-Aware Prefix Cache Management

The hybrid architecture in Kimi K3 complicates prefix caching: the KDA recurrent state and the MLA KV cache differ fundamentally in size and lifetime, yet a cached prefix is reusable only when both can be restored together at the same boundary.
We therefore design a KDA-aware prefix cache that manages the two cache types jointly—from a unified paged layout to fine-grained prefix reuse and consistency under concurrent scheduling—keeping million-token prefixes cheap to retain and reusable across requests.

## Unified cache layout for hybrid KDA–MLA attention

Each Kimi K3 block consists of three KDA layers and one Gated MLA layer, whose caches differ fundamentally.
The MLA KV cache grows with sequence length and is paged per token, whereas the KDA recurrent state is fixed in size with a single copy per request.
Maintaining a separate manager for each would duplicate the allocation, eviction, and transfer logic.
We therefore pack KDA states into the same paged block pool as MLA KV, unifying pages to the same byte size so that both page types share one implementation of allocation, reference counting, and eviction.
Within a page, the states of all heads are stored contiguously head by head, so that each head’s byte stream is self-contained and serves as the minimal unit of cross-node transfer.
Under prefill/decode disaggregation, when prefill and decode nodes adopt different TP degrees, re-layout is performed on the transfer path with zero GPU-side reshuffling.
This asymmetry proved useful during development: any type-confused access yields garbage rather than plausible data — a zero-overhead sanity check on the pooled layout.

## KDA prefix cache optimization

Block-hash-based prefix caching reuses the KV cache at the granularity
of one physical block: only complete blocks are hashed, so only
block-aligned prefixes are reusable.

This coupling breaks down in Kimi K3. Block-hash matching requires
one block size shared by all layers, and a prefix hit is reusable only
if the KDA state at the hit boundary has been persisted. A KDA layer
maintains a single large recurrent state per sequence rather than
per-token entries, so state snapshots are affordable only at sparse
boundaries; the shared block size is therefore forced to 1024–6144
tokens—and, since hashing is tied to the storage block, the hash
granularity as well, although MLA’s per-token entries alone would
tolerate much finer blocks. At such a coarse granularity caching is
nearly useless: requests shorter than one block can never be reused,
and chunked prefill exports no cacheable prefix until it crosses a
full block boundary.

physical cache block (6144 tokens) = 12 prefix-hash blockshash block (512 tokens)MLA KVKDA ckpthit boundary  $B=2560$ restore the KDA checkpoint at  $B$ ; copy-on-write the partial MLA block;resume prefill from token  $B$  with zero recompute of  $[0,B)$ 
Figure 12: Fine-grained prefix caching within a physical cache block.
A 6144-token physical block contains twelve 512-token hash blocks, with cached MLA blocks shown in blue and empty blocks in light gray.
The markers below show the KDA checkpoint status at each hash boundary.
An open circle ( $\circ$ ) denotes a boundary without a stored checkpoint, a gray dot ( $\bullet$ ) denotes a persisted KDA checkpoint, and an orange dot ( $\bullet$ ) marks the checkpoint hit at  $B=2560$ .
Persisted checkpoints are sparse and typically coincide with conversation-turn boundaries.
The request reuses the five MLA hash blocks and the KDA checkpoint at  $B$ , then resumes prefill without recomputing  $[0,B)$ .

We therefore decouple the two granularities. Prefix hashing runs on
fine hash blocks (e.g., 512 tokens) inside MLA pages, while the
physical block remains the coarse allocation unit. Alignment runs the
other way for KDA: checkpoints of the recurrent state are saved only
at (a sparse subset of) MLA’s hash endpoints—the only positions a
lookup can ever reference.

During prefill, a partially filled MLA page is registered in the
prefix-cache index under the chained hash of its last complete hash
block, where each hash covers all preceding hash blocks so that
matching an endpoint certifies the whole prefix up to it; the
registered endpoint advances as the page fills. Meanwhile, after each
forward pass, the KDA kernel persists the recurrent state at the last
hash-aligned position processed. Checkpoints are large, so
intermediate checkpoints superseded as the request advances are
recycled, while those at conversation-turn boundaries are retained for
cross-request reuse. Cached checkpoints are read-only snapshots: a hit
restores the state by copying it into the request’s private running
state before the next forward pass, and new checkpoints are written to
fresh slots, so a checkpoint visible to other requests is never
mutated in place.

Lookup proceeds in two stages (Fig. 12). The
MLA stage matches whole physical blocks by chained hash and, at the
first missing block, falls back to the hash endpoints inside it, so
partially filled pages remain hittable. The KDA stage then requires a
checkpoint at the candidate boundary in every KDA cache group, each of
which maintains an independent recurrent state. The hit is the longest
boundary satisfying both stages—always a multiple of the hash block,
and never required to be a multiple of the physical block. In
Fig. 12, a request whose first 2800 tokens
match the cached prefix hits at  $B=2560=5\times 512$ , deep inside a
6144-token physical block, and resumes prefill from token  $B$  instead
of recomputing  $[0,B)$ .

## Consistency under concurrent scheduling

The remaining design points are each dictated by a concrete failure mode of sharing partially filled blocks, in a setting where a hit block is at once a shared cache entry and the growth point of a private request, and where the MLA and KDA cache groups must agree on every hit boundary.
First, all cache groups draw blocks from one shared free list, so allocating a private copy for one group could evict a block that another group has just hit; every hit block is therefore pinned across all groups before anything is allocated.
Second, the copy into the private block executes on the GPU immediately before the forward pass, so a block allocated or registered within the current scheduling step would still hand the previous owner’s bytes to a reader; such blocks are excluded from matching until their copies land.
Third, a checkpoint can restore a request only if it exists in every KDA group, so evicting one group’s checkpoint atomically invalidates its siblings — a checkpoint is either hittable in every group or in none.
With these mechanisms, every registered state always corresponds to exactly its declared token prefix, and prefix caching for hybrid KDA–MLA models reaches the same generality as for full-attention models: any shared prefix is reusable at any 512-token boundary, independently of request length, chunking, or scheduling interleaving.

## 5.4.2 High-Performance Kernels

Kimi K3 introduces several new architectural modules: KDA (§2.1.1), Block AttnRes (§2.2), and Stable LatentMoE (§2.3). We optimize the kernel implementation for each.

## KDA

Compared with KDA prefill (§5.1), KDA decoding presents a distinct set of challenges: the primary bottleneck shifts from exploiting parallelism to efficiently managing the evolving recurrent state, which is updated in place at every decoding step. This in-place update becomes problematic in MTP-based speculative decoding: if verification rejects a subset of the drafted tokens, the state has already advanced beyond the last accepted token and cannot be trivially rolled back. Maintaining a state snapshot for each draft position would enable rollback, but would also multiply state traffic — a cost that dominates at the large batch sizes typical of online serving.

The state after any accepted draft prefix, however, is fully determined by the projected inputs of the draft tokens, which are far smaller than the state itself.
We therefore cache only these projected inputs, rebuild the states of accepted tokens on-chip, and write back the states of the verified and bonus tokens, a design independently proposed in the concurrent work ReplaySSM [23].
The replayed tokens, the bonus token, and the next draft window share one recurrent loop inside a single fused kernel covering short convolution, input normalization, gating, the KDA recurrence, and output normalization.
Verification latency grows sub-linearly with the number of tokens verified and remains below that of state-caching baselines.
Because the projection caches never leave the decode stage, prefix caching and prefill–decode disaggregation operate on the same payload as in non-speculative serving.

## Block AttnRes

Block AttnRes [60] follows a two-phase schedule: a batched inter-block pass reads the cached block representations once per block, after which each layer folds in the intra-block partial sum through an online-softmax merge [79].
Memory access accounts for a substantial fraction of the cost of these kernels in both prefill and decoding, so our optimizations in both stages focus primarily on memory efficiency.

For prefill, materializing the block representations on every tensor-parallel (TP) rank would incur substantial redundant memory consumption. We therefore adopt sequence parallelism (SP) for activations: the TP all-reduce is decomposed into a reduce-scatter and an all-gather, with the intra-block kernel inserted between the two collectives, operating on the sequence-sharded hidden states so that the block representations of each token are materialized on exactly one rank.
This eliminates the additional memory consumption and reduces the I/O overheads of Block AttnRes during prefill.

For decoding, we launch the inter-block kernel on a side stream so that it overlaps with independent computation on the main stream. The intra-block kernel is instead streamlined through fusion: the merging of the AttnRes output with its partial-sum update, together with the subsequent RMSNorm, is fused into the preceding TP all-reduce, eliminating a dedicated kernel for the intra-block phase.
Together, these optimizations hide the latency of the inter-block pass and reduce the memory traffic of the intra-block phase.

## Stable LatentMoE

Stable LatentMoE increases both the total number of experts and the number of activated experts per token. The resulting growth in both the expert space and the per-token expert count raises scheduling and coordination overheads, making it difficult for conventional MoE kernels to sustain high hardware utilization. These challenges motivate dedicated kernel optimizations for this module.

To mitigate the overhead of the latent GEMMs, we adopt three optimizations. First, we fuse the latent down-projection with the MoE router into a single GEMM. Second, we shard latent weight matrices across ranks and fuse the output all-gather into the GEMM epilogue using multimem store instructions. Finally, we overlap the resulting communication with other operators, such as the shared-expert computation. Together, these optimizations eliminate redundant weight traffic and duplicated computation, while hiding the communication latency behind computation.

For routed experts, at small batch sizes, the group GEMMs reduce to memory-bound streaming of weight matrices — a regime for which conventional tile-centric kernels are poorly suited due to their compute-oriented design and preprocessing overheads.
We instead build the MoE decoding kernel upon the token-centric design of WarpDecode [11], in which each warp is responsible for one output neuron and streams the associated weights directly from memory. To further increase parallelism, we subdivide each warp into finer-grained lane teams, each processing a disjoint subset of experts, followed by a warp-wide reduction of the partial results. In addition, the weight layout is permuted offline at a one-time preprocessing cost, substantially reducing the runtime dequantization overhead.

## 5.4.3 Fleet-Level Scheduling

Beyond a single serving instance, the challenge shifts from per-request efficiency to predictability: a prefix-cache miss costs orders of magnitude more than a hit, and a burst of million-token requests can starve short ones.
We propose two fleet-level scheduling policies to address this: cache-aware affinity scheduling routes each session to the cluster holding its prefix cache while bounding the cost of cluster failures, and budget-based admission control grants each request class its own resource budget so that bursty long-context traffic cannot degrade system-wide SLOs.

## Cache-aware affinity scheduling

At 1M context, a typical coding input carries a prefix of 400K tokens but requires a prefill increment of only 4K tokens, so a prefix-cache hit avoids re-prefilling the entire prefix and is orders of magnitude cheaper than a miss.
We therefore route each request to the cluster that holds its prefix cache, as moving the cache to another cluster would require transferring it over inter-cluster links far slower than the intra-cluster fabric.
This cache-aware affinity, however, binds each session to a single cluster, whose failure would interrupt all sessions bound to it.
Consistent hashing therefore pins each session to two clusters, a primary that serves its traffic and a pre-assigned secondary that takes over when the primary fails.
The secondary holds none of the session’s prefix cache and must re-prefill it upon failover.
Since consistent hashing distributes the secondary assignments of different sessions uniformly across the fleet, this re-prefill work is divided among many clusters rather than concentrated on one.
Cache locality is thus preserved in the common case, while the impact of any single cluster failure remains bounded.

## Budget-based admission control

Production traffic mixes short requests under 2K tokens with ultra-long requests up to 1M tokens, so the per-request cost spans roughly three orders of magnitude and the total load imposed by any fixed number of requests is highly unpredictable.
Capacity planning, queueing models, and rate-limiting quotas based on the “average request” all break down under this variance.
In a typical failure mode, a burst of long-context requests saturates the available compute, and short requests arriving afterwards cannot be scheduled promptly, degrading time to first token (TTFT) across all traffic.
We therefore adopt budget-based admission control, allocating separate resource budgets to different request classes so that bursty long-context traffic consumes at most its own share of the capacity and cannot degrade system-wide SLOs experienced by other classes.

## 6 Evaluations
## Appendix E MoonEP General Upper Bound Proof

Let  $m_{r}(P)$  denote the number of redundant experts placed on rank  $r$  under
plan  $P$ . For a router output  $I$ , the planning objective is to minimize the
maximum number of redundant experts on any rank, i.e.,
 $M(I)=\min_{P}\max_{r}\{m_{r}(P)\}$ . We prove that  $M(I)\leq E/R$  always holds
(Theorem 1) and that this bound is essentially tight: there exist router
outputs for which  $M=\lceil E(R-1)/R^{2}\rceil\approx E/R$  (Theorem 2).

## Proof of Theorem 1 (General Upper Bound)

The goal is to prove that  $M(I)\leq E/R$  holds for any router output  $I$ . Key lemma: there exists a plan  $P^{*}$  such that every EP rank receives exactly the same number of tokens ( $S\times K$ ), and the remote tokens of each rank come from only one other EP rank. The construction is as follows: initially, every rank holds only local tokens, and ranks are classified as underloaded or overloaded accordingly. We repeatedly pick an underloaded rank and an overloaded rank, and migrate tokens from the overloaded rank to fill the underloaded rank exactly up to the balanced value  $S\times K$ ; the overloaded rank may remain overloaded, become exactly balanced, or become underloaded, and is put back into the corresponding set. This is repeated until all ranks are perfectly balanced. Each fill makes one underloaded rank balanced and it never changes afterwards, so the process terminates after at most  $R-1$  fills; meanwhile, each rank is filled at most once, so its remote tokens come from a single rank, which proves the lemma. Consequently, supposing all remote tokens of rank  $r$  come from rank  $s$ ; these tokens belong to at most  $E/R$  local experts on rank  $s$ , hence  $m_{r}(P^{*})\leq E/R$ , and therefore

 $M(I)=\min_{P}\ \max_{r}\big\{m_{r}(P)\big\}\leq\max_{r}\big\{m_{r}(P^{*})\big\}\leq\frac{E}{R}$ 

(28)

## Proof of Theorem 2 (Tightness of the Upper Bound)

Construct a router output  $I^{*}$  as follows: the experts on EP rank 0 receive no tokens, while all experts on the other  $R-1$  ranks share all tokens evenly. Then all  $S\times K\times R$  tokens are evenly divided among  $E(R-1)/R$  experts, so each expert receives  $\frac{SKR^{2}}{E(R-1)}$  tokens. Under any plan  $P$ , rank 0 must receive  $S\times K$  tokens, all of which are remote, and these tokens involve at least  $SK\big/\frac{SKR^{2}}{E(R-1)}=\frac{E(R-1)}{R^{2}}$  distinct experts; taking the ceiling, rank 0 requires at least  $\left\lceil\frac{E(R-1)}{R^{2}}\right\rceil$  redundant experts, hence  $M(I^{*})\geq\left\lceil\frac{E(R-1)}{R^{2}}\right\rceil$ . Conversely, by constructing a plan with the filling procedure from the proof of Theorem 1 and migrating tokens expert-wise preferentially, the number of redundant experts on every rank can be kept within this value, so equality holds. Since  $\left\lceil\frac{E(R-1)}{R^{2}}\right\rceil\approx\frac{E}{R}$  when  $R$  is large, the upper bound in Theorem 1 is essentially tight: there is no general upper bound significantly smaller than  $E/R$ .

## Appendix F Chat Template

# Kimi K3 系统篇（8–10）：并行 / 显存 / RL 与服务 —— 写作规划

> 接第 7 篇「结构决定系统」。第 7 篇已经讲完 §5.1（FlashKDA、设备内 CP、KCP）、§5.4.1（前缀缓存）、§5.4.2（内核）和 §5.2.3 的两段话；
> 明确留白的是 §5.2（预训练并行与显存）、§5.3（RL 基础设施）、§5.4.3（集群调度）。原文全文在 `refs/k3-sec5-text.md`。
> 论文**没有给**任何并行度 / GPU 数 / MFU / tokens/s，所有「账」都要用 config.json 的形状自己算、并行度用变量或假设值。

## 0. 定位：挂在 K3 连载「系统」part 下，3 篇

| # | 标题（暂定） | 对应论文 | 统一视角 | 主图 / 交互 | 篇幅 |
|---|---|---|---|---|---|
| 8 | 训练并行：一步训练里谁在等谁 | §5.2 开头 + Fig.11、§5.2.1 MoonEP + App.E、§5.2.3 | **时间账**：利用率 = 没人闲着 | 一步训练时间线图（自己画的 Fig.11）；EP 演化流程图；MoonEP 填充算法交互 demo | 5–6k |
| 9 | 显存：把 2.8T 塞进 GPU | §5.2.2（含 AttnRes 一节回指第 4/7 篇） | **显存账本**：每一行住在哪、搬走要付什么 | 显存账本交互表（拖并行度/序列长/FP8/offload）；1F1B 在途激活阶梯图 | 5k |
| 10 | RL 与服务：状态住在哪里 | §5.3、§5.4.3（前缀缓存回指第 7 篇） | **状态的生命周期**：GPU → DRAM → NVMe → 别的集群 | KV 池 write-back 生命周期图；双槽位流式加载图；主/备集群一致性哈希图 | 4k |

通用的 PP / EP / ZeRO / CP 教程**不**在这三篇里展开成独立文章：第 8 篇开头用「一步训练的时间线」一节（≈1500 字）把四种并行按「切哪个维度、谁等谁」各一句话带过，够读后面的内容即可。
以后「大模型笔记」专栏「并行」章写通用教程时，复用这里的时间线组件，并把 K3 三篇当案例链接过去。

## 1. 三个统一视角（深入浅出的抓手）

论文 §5.2–5.4 是一张技巧清单，读者最容易读成流水账。每篇只用一个视角把清单串起来：

1. **第 8 篇：谁在等谁。** 并行的全部问题都是「某个执行单元在等另一个」。把等待分四类，每个技巧对号入座：
   - rank 等 rank：EP token 负载不均（MoonEP）、PP 气泡（ViT 塞进气泡）
   - host 等 device：动态 shape 每层同步（静态 shape 免同步）
   - SM 等 SM：rank 内 expert 间 token 数偏斜（workload-aware expert GEMM 调度）
   - 计算等通信：all-to-all（重叠）、shared expert（独立 stream）、Muon all-gather（第 9 篇 P2P）
2. **第 9 篇：显存账本。** 先列出 K3 训练时每张卡上住着什么（参数 / 梯度 / 优化器 / 激活 / AttnRes 块代表 / MoE 中间量 / 通信缓冲），每项用 config 形状算出字节数；然后每个技巧 = 把账本某一行搬走，并付一种货币：FLOPs（重算）、精度（FP8）、PCIe 带宽（CPU offload）、网络带宽（跨 rank 远程 offload）、NVMe 带宽。做成一张「省什么 / 付什么 / 藏进哪段空闲」的表。
3. **第 10 篇：状态的生命周期。** RL 和服务的问题不再是算力而是「状态放哪」：KV / KDA 状态、训练态、参考模型权重、沙箱内存、前缀缓存，各自的存储层次和淘汰/预取时机。反复出现的模式是**双槽位轮转**：double grad buffer（第 9 篇）、参考模型两个 VPP-chunk 槽位、主/备两个集群。点出来读者就记住了。

写法沿用前七篇的三条：每节先给动机（论文里那句「问题」）；数字从形状推出来而不是抄；演化用流程图（vanilla EP → DeepEP → ECHO/UltraEP → MoonEP；全量 all-gather Muon → P2P；write-through → write-back；容器沙箱 → microVM）。

## 2. 每篇要点

### 8 训练并行：一步训练里谁在等谁

- **并行布局一句话**：PP + 虚拟流水段（VP，interleaved 1F1B）、EP、ZeRO-1 DP、Pipeline ZeRO-2 梯度分片、CP（KCP，回指第 7 篇）。shared expert 在 EP rank 上复制；dispatch/combine 的 all-to-all 与计算重叠。
- **一步训练的时间线**：自己画 Fig.11：一个 PP stage 上一个 micro-batch 的 attention → router → dispatch(all-to-all) → expert GEMM → combine → 下一层，叠上 1F1B 的 warmup / steady / cooldown 三段。这张图是第 8、9 篇的公共坐标系：第 8 篇标「谁在等」，第 9 篇标「什么时候显存最高」。
- **四种并行速览**（各一段 + 一句「切什么维度、引入什么等待」）：DP 切 batch（等 all-reduce）、PP 切层（等气泡）、EP 切 expert（等最慢 rank + all-to-all）、CP 切序列（等相邻 rank 的状态/KV）。TP 在 K3 预训练里没有列出，说明一下。
- **MoonEP**（本篇核心，从「目标」推）：
  - 目标：每个 rank 恰好收到 $S\times K$ 个 token，计算量完全一致；否则 makespan 由最慢 rank 决定，且动态 shape 造成显存碎片。
  - 手段：冗余 expert 在线迁移。关键问题「要预留多少冗余槽位」→ **定理 1**：任意 router 输出都存在最多 $E/R$ 个冗余 expert/rank 的平衡方案。证明就是一个填充过程：欠载 rank 从一个过载 rank 一次性补满到 $S\times K$，每个 rank 最多被补一次，所以远端 token 只来自一个 rank，最多涉及那个 rank 的 $E/R$ 个本地 expert。用 4 rank × 8 expert 的小例子逐步演示（交互 demo：随机 router 输出 → 逐步填充 → 冗余数从不超过 E/R）。
  - **定理 2**（紧性）：rank 0 一个 token 都没收到、其余 $R-1$ 个 rank 均分，则 rank 0 至少需要 $\lceil E(R-1)/R^2\rceil\approx E/R$ 个冗余 expert。
  - 代入 K3：$E=896$，按 $R=32/64/112$ 分别算本地 expert 数 = 冗余槽位上限 = 28/14/8。
  - 对比 ECHO / UltraEP 的「预设冗余数或每 rank token 上限」：没有可行方案就停训、上限要手调、仍有残余不均衡。演化流程图放这里。
  - 在线规划：ILP 离线求精确最优当参照，GPU 规划内核近似最优、开销可忽略、永远不超 $E/R$。
  - 零拷贝：规划内核预先算好每个 token 的目的地 → 融合 permute/unpermute 直接落到远端 expert 分组位置。缓冲大小推导：最坏情况所有 token 涌向一个 rank，DeepEP 要 $S\times K\times R$，MoonEP 固定 $S\times K$，差 $R$ 倍。
  - 免同步静态 shape：常规 MoE 每层要 host 等 device 拿到实际 token 数才能 launch；完全平衡后所有层 shape 静态已知。
  - expert GEMM 调度：rank 间平衡了，rank 内 expert 间仍偏斜 → 固定顺序调度让 SM makespan 不均 → 按当前 token 分布选参数（解析代价模型 + 离线 autotune 标定），执行中不变。shared expert 单独 stream。
  - 反向：冗余 expert 的梯度先在本地 reduce buffer 暂存，算完 reduce 回 home rank。
  - 和 QB（第 5 篇）的关系：QB 让负载统计上平衡，MoonEP 让每一步执行形状固定。两者缺一不可，一句话讲清。
- **视觉编码器塞进气泡**（从第 7 篇扩写）：interleaved 1F1B 下头几个 micro-batch 的文本前向挤在最前、末几个的反向挤在最后 → ViT 前向：头几个同步先做，其余塞气泡；反向对称。配合动态 CP：大图按 patch 切到多卡、gather-KV 做注意力；CP 组再切子组，多张大图负载均衡分配，通信占比不随规模增长。气泡份额公式 $(P-1)/(mV)$ 顺带推一下，说明气泡够不够放 ViT。

### 9 显存：把 2.8T 塞进 GPU

- **账本**（本篇骨架，做成交互表）：以一张卡为单位列行：
  - 参数：2.78T × 2B / (PP × EP …)，routed expert 占绝大头（896 × 每 expert 3 个矩阵 3584×3072）；attention/AttnRes/embedding 按 config 算。
  - 梯度、优化器：Muon 只有动量 + FP32 主权重；ZeRO-1 把优化器状态按 DP 切；Pipeline ZeRO-2 把梯度也按 DP 切。
  - 激活：每层每 micro-batch，hidden 7168 × S；MoE 中间量 16 × 3072 × S（这一行最大）；AttnRes 块代表每 12 层一份；1F1B warmup 使 rank 0 同时在途 $P$ 份（VP 下 $P\cdot V$ 份的一部分），rank $P-1$ 只有 1 份——画阶梯图。
  - 通信缓冲：all-to-all 缓冲 $S\times K$（回指第 8 篇）。
- 然后逐项讲「搬走」：
  - **统一激活管理器**：每个反向要用的张量绑一个可插拔存储后端；重算 / 量化 / 本地 offload / 远程 offload 只是策略，张量粒度可组合，用注解声明、与模型代码解耦；函数粒度重算支持跨层；单一 stream 单一内存池免碎片；按层粒度预取回来与计算重叠。K3 实际配置：大部分激活 block-wise FP8 + offload，逐元素算子重算。表：技术 | 省的行 | 付的货币。
  - **MoE 反向的两处省**：(a) 推导 router 概率的梯度：$y=\sum_e p_e\,W^{\text{down}}_e a_e$，$\partial L/\partial p_e=\langle dy, W^{\text{down}}_e a_e\rangle=\langle (W^{\text{down}}_e)^{\top}dy,\ a_e\rangle$，右边只需要中间激活 $a_e$（act_output）和上游梯度，不需要保存 expert 输出 —— 多一次逐元素乘加，省一整份 output（SonicMoE 的思路）。(b) group GEMM 前向只存 dispatch 的输入，反向重算 dispatch，其通信藏在 group GEMM 反向的一部分后面（Fig.11）。
  - **AttnRes**：块代表在边界层生成一次全层共享；AttnRes 整体 checkpoint，每层保存的激活与普通残差相同；PP 用 cache-based 通信只增量传新块、micro-batch 结束即释放。第 7 篇已讲，这里只放进账本对应行并回指。
  - **PP rank 间激活均衡**：从阶梯图看 rank 0 最满、rank $P-1$ 最空 → 用 Mooncake Transfer Engine 把激活远程 offload 到别的 PP rank 的显存。付的是节点内/间网络带宽，藏在计算后面。
  - **Pipeline ZeRO-2 + CPU 分片**：梯度按 DP 切；分片存 CPU，GPU 只留 double grad buffer；DP reduce 进 double buffer 后累加到 CPU 分片。双槽位模式第一次出现。
  - **P2P Muon**：分布式优化器把参数按 DP 切，Newton–Schulz 要整矩阵。朴素做法每个 rank all-gather 全部参数：显存多一份完整参数、通信量 ≈ 全量。P2P：每个 rank 只从 owner 拉自己负责正交化的那些矩阵的分片；通信量从「每 rank 收全量」降到「每 rank 收 1/N」，无完整参数缓冲；按 model-chunk 缓冲粒度流水化。推导两者通信量公式。
- 收尾：把账本「优化前 / 优化后」两列并排，读者一眼看到哪几行消失了。

### 10 RL 与服务：状态住在哪里

- **1M 上下文 RL 的资源约束**：co-located RL（训练和推理同一批卡，几百张 GPU 做一个 1M 实验）+ partial rollout（长轨迹分段，压尾延迟）。代价：rollout 阶段 KV 要长期驻留，和训练态抢显存 / DRAM。
- **外部 KV 池（write-back）**：为什么 miss 贵（1M 多步 rollout）、partial rollout 让每轮开头一堆未完成的长 prefill 同时到、投机解码加快请求周转让前缀块换手更频繁 → 抢占、命中率下降。设计：活跃解码块留 GPU，闲置可复用前缀**被驱逐时**才写回 CPU DRAM 池，下次复用前预取；KDA 状态与对应 MLA KV 块一起 offload/prefetch，生命周期对齐（回指第 7 篇统一分页布局）。write-through 对比：只为离开活跃路径的前缀付 DRAM 和带宽。为给池腾 DRAM，训练迭代结束后把权重和优化器状态下到 NVMe；rollout 结束释放池。画四层存储 + 状态流向图。
- **rollout 自动节流**：多步 rollout 上下文逐步变长，按平均长度定固定并发要么估不准要么前期太保守，太高又后期 KV 压力大触发抢占 → 在请求调度层用活跃数、排队数、KV 利用率动态控制发给推理引擎的请求数。
- **梯度缓冲复用给参考模型**：参考模型只前向、权重放不下 GPU → 放 CPU，需要时物化到策略模型的 FP32 梯度缓冲里；安全因为之后真梯度会覆盖。ZeRO-2 分片后每卡只留两个 VPP chunk 的梯度缓冲 → 两个槽位一个算当前 chunk、一个预取下一个。双槽位模式第二次。
- **AgentENV 沙箱**：容器 → microVM（Firecracker）的动机（agent 探索导致 kernel panic / 死锁；又要允许挂盘、跑容器、开 VM）。增量 checkpoint：只存脏页，checkpoint 133 ms / resume 49 ms；Pause/Resume（等模型推理占沙箱生命周期最多 98%）、Fork（判分无副作用）、Snapshot。OverlayBD + 自研 ublk + 存储层共享 + P2P 传输：秒级内起数万沙箱；COW + page cache：内存超售 6.5×。总量 51,219,741 个沙箱 / 1,505,678 个镜像。这一节偏系统工程，控制在 800 字。
- **集群级调度**：
  - cache-aware affinity：典型编码请求前缀 400K、增量 4K，命中比 miss 便宜两个数量级；跨集群链路远慢于集群内，所以请求跟着缓存走。代价是会话绑死一个集群 → 一致性哈希给每个会话钉主 + 备两个集群，备集群没有缓存、故障时重 prefill，但备份均匀散在全集群上，单集群故障影响有界。双槽位模式第三次。
  - budget-based admission：请求代价跨三个数量级（2K 到 1M），按「平均请求」做容量规划 / 排队模型 / 限流全部失效；典型故障是长请求突发吃满算力、后到的短请求 TTFT 恶化 → 按请求类别分预算，长上下文突发最多吃自己那份。
- 收尾：把三篇的「等待 / 账本 / 生命周期」三张图并排，回到第 7 篇的结论「系统在为结构的选择付账」。

## 3. 图与组件

| 组件 | 用在 | 做法 |
|---|---|---|
| `PpTimeline.astro` | 8、9 | 交互：滑块 P / m / V，画 interleaved 1F1B；模式切换「谁在等」（标气泡、all-to-all、host sync）/「在途激活数」（每 rank 阶梯）。第 8 篇同一张图再叠一层 ViT 前向/反向填气泡。 |
| `MoonEpFill.astro` | 8 | 交互：R=4，E=8，随机 router 输出得到每 rank token 数，逐步执行定理 1 的填充过程，实时显示每 rank 冗余 expert 数与上限 E/R=2；提供定理 2 的最坏例子按钮。 |
| EP 演化流程图 | 8 | `FlowDiagram`（不用 mermaid，见 svg-figure-conventions）：vanilla EP → DeepEP → ECHO/UltraEP → MoonEP，每步一句改了什么。 |
| `MemoryLedger.astro` | 9 | 交互表：输入 PP / EP / DP / VP / micro-batch 序列长 / FP8 / offload 开关，按 config.json 形状算每卡各行字节；「优化前 / 后」两列。 |
| MoE 反向依赖图 | 9 | 内联 SVG：output 依赖 → 改写后只依赖 act_output 与 doutput。 |
| 存储层次与状态流向图 | 10 | 内联 SVG：GPU HBM / CPU DRAM / NVMe / 远端集群四层，KV 块、KDA 状态、训练态、参考模型权重各自的箭头与触发时机。 |
| 双槽位示意 | 9、10 | 小 SVG，三处复用（double grad buffer / VPP chunk 槽位 / 主备集群）。 |
| 一致性哈希主备图 | 10 | `FlowDiagram` 或内联 SVG。 |

## 4. 小心的点

- 论文没有并行度、GPU 型号、MFU；账本里所有并行度都是变量或标注「假设」。不要编数字。
- Fig.11 只有标题没有正文描述，自己画的时间线要标明是「示意」。
- §5.2.1「$S$」是 micro-batch 内 token 数（论文写 sequence length），$S\times K$ 是 token-expert 对数，写的时候统一叫「token 份数」。
- MoonEP 的定理证明是「存在性」，实际 GPU 规划器是启发式；别写成「规划器保证最优」。
- 第 7 篇已讲的 KCP、FlashKDA、前缀缓存、内核、AttnRes 内存三段，只回指不重复；第 7 篇里「MoonEP 一句话」「视觉编码器的两件事」两段写完第 8 篇后改成链接。
- SonicMoE 的梯度改写细节论文只给一句话，推导按自己的公式写，标注「按论文描述复原」。
- 第 10 篇里「co-located RL [58]」和 partial rollout [119] 是 K1.5/K2 的做法，简述即可。

## 5. 顺序

8 → 9 → 10。先做 `PpTimeline` 组件（8、9 共用），再写第 8 篇；`MemoryLedger` 的形状计算可以从第 0 篇的参数量计算脚本复用。

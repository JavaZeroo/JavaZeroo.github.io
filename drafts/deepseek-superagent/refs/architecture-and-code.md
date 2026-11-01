# Super-agent 原文与代码精读

## 0. 一句话：这篇要解决什么问题

模型能够执行单个任务后，人类仍需跨会话同步背景、分派任务和维护运行；文章试图将这部分工作交给系统。

## 1. 一手资料清单

- 原文：https://x.com/DeepSeekHarness/status/2116857257245987236 。关联 article id 2116829218935857152；经公开只读转译接口取得正文，阅读 §1–9。X 上的英文版本自述为中文原文的机器翻译。
- 仓库：https://github.com/deepseek-ai/dsh-superagent-prototype ，审阅 commit `6337fc823ce20251c8de8085948165c81aaa701e`（package 0.1.0）。
- 阅读 README、package.json、docs/001-domain、002-authorization、003-execution、006-deployment、007-architecture；查阅 SETUP 的默认值与巡检段落。
- 源码：src/policy.ts、core/intake.ts、core/assignments.ts、core/replies.ts、app/sessions.ts、app/resources.ts、app/store.ts、app/effects.ts、app/outbox.ts，以及 executor/dsh.ts 的工作目录与启动部分。
- 对照 tests/policy.spec.ts、tests/runtime-concurrency.spec.ts；仅阅读，未执行插件测试或部署真实 GitHub/飞书集成。

## 2. 核心机制与推导

- 原文 §2：全局 Router 保持连续性；隔离 Worker 提高并发。Router 不应深度拆解每个任务，单个 Worker 可以再用 Agent Team。
- §3：事件过滤分代码准入和语义研判；订阅关系多对多；Inline/Spill + 引用 ID 实现 Chain of Recall。不能等同完整长期记忆。
- §4：耗时工具异步化，start 返回、完成再注入事件。不是一次推理中途任意打断。
- §5–6：云端计算、持久化状态、终端环境解耦，以及外部 Watchdog 是设计；公开原型另行核验。
- 解读：工单生命周期独立于会话，有利于重启恢复；外部操作不能靠内部事务获得端到端 exactly-once。

## 3. 代码落点

| 文件 / 入口 | 事实 |
| --- | --- |
| package.json | DSH bundle + web client，MIT，版本 0.1.0，private true；不据此编造 npm 安装命令 |
| policy.ts / assessmentOutcome | 检查规则完整覆盖、方向、命中规则、意图依据；inferred 且非 auto 记录 would_accept |
| policy.ts / EvalConfigSchema、ExecutorConfigSchema | 默认 observe，maxWorkers 默认 3 |
| core/intake.ts / Intake.receive | Trigger 去重、记录、订阅扇出、拥有者判断、入 Eval 队列；bot 消息独立走评估规则 |
| app/sessions.ts / noWait、工具 restrict | 主会话仍有 bash/read 等；限制 sleep 和长等待，不是禁止所有任务执行 |
| app/resources.ts / RESOURCES | ticket/thread/trigger 互相引用、分页回读；Executor 仅暴露自身资源范围 |
| app/store.ts / Store | node:sqlite，WAL、synchronous FULL，事务、状态、effect、outbox |
| app/effects.ts / Effects.enqueue | 在调用方事务里写 effect，提交后派发，按类型并发与退避 |
| app/outbox.ts / send、deliver | 回执与 reply effect 同事务；sources.send 后 settle，网络调用与落盘不在一个跨系统事务 |
| core/assignments.ts / RECOVERY_NOTE | 重启后先检查 worktree、git、GitHub，避免重复已完成外部操作 |
| docs/003 | 5 秒转后台；并发槽位另有一个供人类消息或状态请求使用；独立目录非安全边界 |
| docs/006 / Hourly check | 在另一个 DSH profile 安排每小时巡检；尊重手动停止；不是已证明的语义进度 Watchdog |

## 4. 关键数字

| 数字 | 条件 | 出处 |
| --- | --- | --- |
| 每次约 300 条消息、约 20 个工作请求、每方案 5 次 | 编码用耗时工具模拟 | 原文 §7.1 |
| Super-agent p50 12.4 分钟、p95 15.7 分钟 | 同上 | §7.1 |
| 单会话 neutral p95 29.5 分钟 | 5 次都未调用 subagent | §7.1 |
| 约 46.8% | 自算 `(29.5-15.7)/29.5`，仅 p95 完成时间降幅 | 上述数字 |
| 去掉 pass-through 决策约多 30%，cache-read 23.2M→31.8M，约多 37% | 完成时间相近 | §7.1 |
| Trigger 约 1→40/分钟；batch 1→2.8；请求 2.5→3.9 秒；决策摊销 6.4→1.6 秒 | Router 真模型、Worker mock | §7.2 |

## 5. 常见误解与澄清

- Super-agent 不是新基础模型；插件依赖 DSH。
- 原文架构、内部实例、公开原型不是同一个交付物。
- observe 仍执行显式请求；仅推断任务不自动接受。
- 订阅得到上下文不等于得到新权限；范围仍须模型遵守 Mandate。
- 两套 preset 均为 danger-full-access；账户包装器不能隔离宿主机。
- auto-merge 由交付提示词约束，不能替代 GitHub branch protection。
- 本地 SQLite 与重启恢复不等于云端弹性执行、BYOM 全部落地。

## 6. 与本专栏其他文章的接口

现有文章重点是模型架构、训练与服务；本篇为独立 Agent 系统分析，不强塞入 DeepSeek-V4/V4.1 模型结构连载。上下文缓存仅作系统类比，不扩展模型 KV cache 教程。

## 7. 图的建议

两层调度与直达、显式/推断授权分流、SQLite 事务与外部操作边界。

## 8. 待确认 / 没查到的点

未复现实验，未运行原型真实集成。没有据此报告真实修复率、token 账单、跨组织权限安全或全端到端 exactly-once。公开文档没有建立完整 BYOM 和语义进度 Watchdog 已交付的证据。

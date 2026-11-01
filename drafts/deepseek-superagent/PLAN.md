# DeepSeek Super-agent 单篇分析

- 读者：用过 coding agent、想理解团队级长期运行系统的开发者。
- 对应材料：原文 §1–9；原型 commit `6337fc823ce20251c8de8085948165c81aaa701e`。
- 正文：`src/content/posts/deepseek-superagent-architecture-and-prototype.mdx`。
- 范围：架构动机、事件与授权流程、工程可靠性、实验解释、原型边界。
- 不展开：模型训练、安装实操、真实生产性能评测；本次不部署被分析的插件。
- 图：`agents/SuperAgentFlow.astro` 的 architecture、authorization、recovery 三张机制图。
- 依据：`refs/architecture-and-code.md`。
- 状态：正文与三张图完成。Node 24 下类型检查、生产构建、站点浏览器回归均通过；新文章在 1440/390/320px、浅色/深色下检查了布局、图表、锚点与运行时错误。首页与 RSS 已验证。
- 审校：原文实验仅支持调度结论；代码引用固定提交；BYOM 与语义 Watchdog 不写成公开交付能力；exactly-once 分析标明作者解读。用词扫描只剩普通中文“查询”误报，非 attention 的 query。

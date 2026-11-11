---
title: "Kimi K3 模型结构"
description: "从 Kimi K3 技术报告到 K3.1 Pro 推理代码：沿序列、深度、宽度三个维度拆解 KDA、Gated MLA、Attention Residuals、Stable LatentMoE 和视觉通路，再分析 K3.1 新增的 Gated DSA。"
parts:
  - "总览"
  - "序列"
  - "深度"
  - "宽度"
  - "输入端与零件"
  - "系统"
  - "K3.1 更新"
color: "var(--cat-cyan)"
---

读这个专栏的时候，建议手边开着 [Kimi K3 的技术报告](https://arxiv.org/abs/2607.24653)，以及 Hugging Face 上 [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) 的 `config.json`。文章里的每一个维度、每一个层号都能在这两处对上；论文没写、只有代码里才有的细节，会单独标出来。

[第 11 篇：K3.1 Pro 的 Gated DSA](/2026/11/12/kimi-k3-11-k31-dsa/) 改用发布配置、vLLM 和 SGLang 的固定版本代码交叉分析，区分可确认的前向结构与尚未公开的训练细节。

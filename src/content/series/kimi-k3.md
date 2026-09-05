---
title: "Kimi K3 模型结构"
description: "配合 Kimi K3 技术报告 §2 一起读的连载：按论文自己的叙事，沿序列、深度、宽度三个维度，把 KDA、Gated MLA、Attention Residuals、Stable LatentMoE 和视觉通路一个模块一个模块拆开。每篇都有一张「你现在在这里」的总图，和该模块自己的方块图。"
parts:
  - "总览"
  - "序列"
  - "深度"
  - "宽度"
  - "输入端与零件"
  - "系统"
color: "var(--cat-cyan)"
---

读这个专栏的时候，建议手边开着 [Kimi K3 的技术报告](https://arxiv.org/abs/2607.24653)，以及 Hugging Face 上 [moonshotai/Kimi-K3](https://huggingface.co/moonshotai/Kimi-K3) 的 `config.json`。文章里的每一个维度、每一个层号都能在这两处对上；论文没写、只有代码里才有的细节，会单独标出来。

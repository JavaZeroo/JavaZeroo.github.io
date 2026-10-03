---
title: "DeepSeek-V4.1 模型结构"
description: "配合 DeepSeek-V4.1-Flash 技术报告 §2 一起读的连载，是 DeepSeek-V4 连载的续篇，只讲这一代的改动：Causal Encoder-Decoder（CED）、跨层共享 KV 的 CSA2、Hierarchical Sparse Indexer、FP4 KV cache、Single-Pass mHC、Engram、DSpark 和视觉通路。每篇都有一张「你现在在这里」的总图。"
parts:
  - "总览"
  - "序列"
  - "深度"
  - "记忆与解码"
  - "输入端与优化器"
  - "系统"
color: "var(--cat-rose)"
---

读这个专栏的时候，建议手边开着 [DeepSeek-V4.1-Flash 的技术报告](https://arxiv.org/abs/2609.19969)，以及 Hugging Face 上 [deepseek-ai/DeepSeek-V4.1-Flash](https://huggingface.co/deepseek-ai/DeepSeek-V4.1-Flash) 的 `config.json` 和 `inference/model.py`。V4.1 是在 V4 上改出来的，V4 已有的模块在 [DeepSeek-V4 连载](/2026/09/05/deepseek-v4-00-overview/)里讲过，这里只引用。论文没写、只有代码里才有的细节，会单独标出来。

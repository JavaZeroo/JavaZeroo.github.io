---
title: "DeepSeek-V4 模型结构"
description: "配合 DeepSeek-V4 技术报告 §2 一起读的连载：沿序列、深度、宽度三个维度，把压缩稀疏注意力（CSA）、重压缩注意力（HCA）、流形约束的 Hyper-Connections（mHC）、改过打分函数和路由的 DeepSeekMoE，以及 Muon 优化器一个模块一个模块拆开。每篇都有一张「你现在在这里」的总图，和该模块自己的方块图。"
parts:
  - "总览"
  - "序列"
  - "深度"
  - "宽度"
  - "优化器与训练"
  - "系统"
color: "var(--cat-indigo)"
---

读这个专栏的时候，建议手边开着 [DeepSeek-V4 的技术报告](https://arxiv.org/abs/2606.19348)，以及 Hugging Face 上 [deepseek-ai/DeepSeek-V4-Pro](https://huggingface.co/deepseek-ai/DeepSeek-V4-Pro) 的 `config.json` 和 `inference/model.py`。论文里每个模块的公式都能在官方实现里找到对应的几行，文章会把两边对着看；论文没写、只有代码里才有的细节，会单独标出来。Pro 和 Flash 两个模型结构相同，只是尺寸不同，正文以 Pro 的维度为主，Flash 的数字在总览一篇的表里。

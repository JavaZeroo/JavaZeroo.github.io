---
title: NGENet论文理解
date: 2022-08-25 17:09:16
tags:
- 点云配准
- 深度学习
- NGENet
---

由于点云传统配准（ICP：迭代最近点算法）效果不佳，于是考虑使用深度学习的方法。
这里采用(NGENet)[https://paperswithcode.com/paper/neighborhood-aware-geometric-encoding-network]。

Neighborhood-aware Geometric Encoding Network
邻域感知几何编码网络

Task	Dataset	Model	Metric Name	Metric Value	Global Rank
Point Cloud Registration	3DLoMatch (10-30% overlap)	NgeNet	Recall ( correspondence RMSE below 0.2)	71.9	# 2		
Point Cloud Registration	3DMatch (at least 30% overlapped - FCGF setting)	NgeNet	Recall (0.3m, 15 degrees)	95.0	# 1		
RE (all)	4.932	# 1		
TE (all)	0.155	# 1		
Point Cloud Registration	3DMatch (at least 30% overlapped - sample 5k interest points)	NgeNet	Recall ( correspondence RMSE below 0.2)	92.9	# 1		


## Introduction

### 点云配准深度学习方法的分类

作者将其分为了两种

1. end-to-end：将feature learning 和 transformation estimation 融合到了一个模型
	1. 其中有一些方法依赖相关性的建立来进行后续的Procrustes分析
	2. 而其他方法则更关注点云之间的全局特征

2. feature-learning：更加关注learning of discriminative point feature（学习辨别性的点的特征），而transformation则是由pose estimators来估计的

### 提出网络

NgeNet利用多尺度结构明确地生成具有多种邻域大小的点状特征，并利用几何指导的编码模块来最大限度地利用几何信息。具体来说，设计了一个投票机制，为每个点选择一个合适的邻域大小，并拒绝在无法区分的表面上出现虚假的特征。


### 主要贡献

- 多尺度结构与几何引导编码相结合，产生多层次的几何和语义信息编码的特征。
- 多层次的一致性投票，为每个点选择适当的邻域并拒绝虚假的邻域。
- 我们的方法中所提出的技术是与模型无关的，能够很容易地移植到其他骨干结构上并提高性能。


## Preliminaries （引言）

### 点云配准问题的数学表达

> arg    是变元（即自变量argument）的英文缩写。
> arg min 就是使后面这个式子达到最小值时的变量的取值
> arg max 就是使后面这个式子达到最大值时的变量的取值
> 例如 函数F(x,y):
> arg  min F(x,y)就是指当F(x,y)取得最小值时，变量x,y的取值
> arg  max F(x,y)就是指当F(x,y)取得最大值时，变量x,y的取值

$$Some formula$$

> (Cardinal Number（基数）)[https://zh.wikipedia.org/wiki/基数_(数学)]：集合论中刻画集合大小的数

### 邻域分析

在特征学习中常常用到encoder-decoder网络来提取特征；
模型的输入为$X\in\mathbb{R}^3$
维数从$N*3$->$N*C$意思是每个点输出C维的特征

有两种方式去影响领域的范围一个是依靠连续的卷积层，隐形的增加了领域的范围；另一个是在encoder-decoder网络中常常使用两倍大小的分层卷积层来模拟点云的down sample。

## 方法

### 网络

![Architecture-of-NgeNet.png](images/Architecture-of-NgeNet.png)_NgeNet的网络架构

可以很清楚的看到NgeNet是一个encoder-decoder网络
- encoder模块由：**residual-style KPConv**/**strided KPConv**层、**instance norm**层和**Leaky ReLU**层（k=0.1）组成
- decoder模块由：decoder中的上采样块采用最近搜索来进行特征插值。 一元块由一个线性 (MLP) 层、一个实例范数层和一个 Leaky ReLU 层 (k=0.1) 组成，而最后一个一元块仅由一个线性层组成


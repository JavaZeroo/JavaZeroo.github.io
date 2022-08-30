---
title: Pandas的数据操作
date: 2022-08-30 09:09:16
mathjax: false
categories:
- 🖥️Python
tags:
- Pandas
- 数据处理
---
最近在打Kaggle，遇到很夸张的数据预处理。
平时每次要用pandas都是现翻文档，感觉太慢，以后用到什么pandas的函数就在这里记录一下😘

## 提取数据

### loc、iloc、ix

都是对DataFrame类的方法。

loc：通过选取行（列）标签索引数据
iloc：通过选取行（列）位置编号索引数据
ix：既可以通过行（列）标签索引数据，也可以通过行（列）位置编号索引数据
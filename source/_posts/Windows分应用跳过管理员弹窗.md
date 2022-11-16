---
title: Windows分应用跳过管理员弹窗
date: 2022-11-15 20:54:21
categories:
- 💻Windows
tags:
- 分享
---

<iframe style="border-radius:12px" src="https://open.spotify.com/embed/track/5K5Yo96RucQFgZ2FGFdgSQ?utm_source=generator" width="100%" height="152" frameBorder="0" allowfullscreen="" allow="autoplay; clipboard-write; encrypted-media; fullscreen; picture-in-picture" loading="lazy"></iframe>

## 程序设置管理员启动

右键exe文件->属性

![程序设置管理员启动](/images/管理员/软件设置.jpg)

{% note warn %}
**把程序打开一次后关闭**（这个时候还会有弹窗是正常的，别急😠）
{% endnote %}

## 进入注册表
摁住<kbd>Win</kbd>+<kbd>R</kbd>，输入`regedit`打开注册表编辑器

![注册表编辑](/images/管理员/注册表编辑.png)

在`Computer\HKEY_CURRENT_USER\Software\Microsoft\Windows NT\CurrentVersion\AppCompatFlags\Layers`下面找到**对应的程序**，然后修改程序的数值为`RUNASINVOKER`

![编辑值](/images/管理员/编辑值.png)

再次打开软件就不会有弹窗了。
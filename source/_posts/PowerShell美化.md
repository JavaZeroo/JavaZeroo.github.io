---
title: PowerShell美化
date: 2022-09-03 14:40:53
categories:
- 💻Windows
tags:
- 软件
- 分享
---
最近刚刚重装了系统，记录一下美化的过程
## PowerShell 7 安装

在这里面下载最新的PowerShell安装文件，建议使用``.msi``文件
```
https://docs.microsoft.com/en-us/powershell/scripting/install/installing-powershell-on-windows?view=powershell-7.2#installing-the-msi-package
```
下载完之后，一路下一步就行

{% note info %}
⚠️这里建议安装的时候，将PowerShell加入到Windows Terminal中。避免后面手动添加
{% endnote %}

## ohmyposh安装

官方教程: [点这里](https://ohmyposh.dev/docs/installation/windows)

这里我就直接用wget了

```
https://ohmyposh.dev/docs/installation/windows
```

然后要安装Nerd类型的字体(就是一种将一部分字符变成好看的符号的字体🥰)

{% note info %}
官方是用ohmyposh内置的安装方法。其实我更推荐下载字体文件下来双击安装。这样更熟悉一点。(拒绝离开舒适区🙅🏻)
{% endnote %}

当然，安装完字体后需要在Windows Terminal上设置一下
[设置位置](/images/PowerShell美化/terminal.jpg)

## 马上就好了！
现在一切准备就绪，只需要稍微设置一下PowerShell就行了

输入：
```
notepad $PROFILE
```

把下面这行复制进去，**保存!**
最后复制这个！就行了！
``
. $PROFILE
``

> 这里说个题外话😁，其实就是设置Shell的配置
> 用过linux可能比较熟悉
> ```
> vim ~/.bashrc
> source ~/.bashrc
> ```
> 其实是一一对应的，原理都差不多

## 主题！(当然你可以不弄)

这里你可以用下面这个指令看到所有的预设主题
```
Get-PoshThemes
```

选一个你喜欢的，比如说``jandedobbeleer``
那么你就可以输入下面这个指令换成这个主题：
```
oh-my-posh init pwsh --config ~/.jandedobbeleer.omp.json | Invoke-Expression
```
{% note info %}
其实可以看出来``~/.jandedobbeleer.omp.json`` 这个就是``jandedobbeleer``对应的json文件，所有主题都是以json文件储存的。
{% endnote %}

其实你可以自己创建一个主题，这里是[官方教程](https://ohmyposh.dev/docs/configuration/overview)，事实上他包括很多插件，eg. Spotiy；

这里给出我自己的json
你只需要复制以下代码，并且创建一个json文件。然后把上面的``~/.jandedobbeleer.omp.json``换成你的json文件就行;
```
{
  "$schema": "https://raw.githubusercontent.com/JanDeDobbeleer/oh-my-posh/main/themes/schema.json",
  "blocks": [
    {
      "alignment": "left",
      "segments": [
        {
          "background": "#003543",
          "foreground": "#fff",
          "powerline_symbol": "\ue0b0",
          "style": "powerline",
          "template": " {{ if .WSL }}WSL at {{ end }}{{.Icon}}",
          "type": "os"
        },
        {
          "background": "#003543",
          "foreground": "#d2ff5e",
          "powerline_symbol": "\ue0b0",
          "style": "powerline",
          "template": "{{ .HostName }} ",
          "type": "session"
        },
        {
          "background": "#0087D8",
          "foreground": "#003544",
          "powerline_symbol": "\ue0b0",
          "properties": {
            "folder_separator_icon": "/",
            "style": "full"
          },
          "style": "powerline",
          "template": " \ue5ff {{ .Path }} ",
          "type": "path"
        },
        {
          "background": "#d2ff5e",
          "background_templates": [
            "{{ if or (.Working.Changed) (.Staging.Changed) }}#ff9248{{ end }}",
            "{{ if and (gt .Ahead 0) (gt .Behind 0) }}#f26d50{{ end }}",
            "{{ if gt .Ahead 0 }}#89d1dc{{ end }}",
            "{{ if gt .Behind 0 }}#f17c37{{ end }}"
          ],
          "foreground": "#193549",
          "powerline_symbol": "\ue0b0",
          "properties": {
            "fetch_stash_count": true,
            "fetch_status": true
          },
          "style": "powerline",
          "template": " {{ .HEAD }}{{ if .Staging.Changed }} \uf046 {{ .Staging.String }}{{ end }}{{ if and (.Working.Changed) (.Staging.Changed) }} |{{ end }}{{ if .Working.Changed }} \uf044 {{ .Working.String }}{{ end }}{{ if gt .StashCount 0 }} \uf692 {{ .StashCount }}{{ end }} ",
          "type": "git"
        }
      ],
      "type": "prompt"
    },
    {
      "alignment": "right",
      "segments": [
        {
          "background": "#0087D8",
          "foreground": "#003544",
          "invert_powerline": true,
          "powerline_symbol": "\ue0b2",
          "properties": {
            "display_mode": "context",
            "fetch_virtual_env": true
          },
          "style": "powerline",
          "template": " \ue235 {{ .Venv }} ",
          "type": "python"
        },
        {
          "background": "#003543",
          "foreground": "#fff",
          "invert_powerline": true,
          "powerline_symbol": "\ue0b2",
          "style": "powerline",
          "template": "<#fff> \uf64f </>{{ .CurrentDate | date .Format }} ",
          "type": "time"
        }
      ],
      "type": "prompt"
    },
    {
      "alignment": "left",
      "newline": true,
      "segments": [
        {
          "foreground": "#FFD700",
          "style": "plain",
          "template": " \u26a1 ",
          "type": "root"
        },
        {
          "foreground": "#f1184c",
          "style": "plain",
          "template": "💫",
          "type": "text"
        }
      ],
      "type": "prompt"
    }
  ],
  "console_title_template": "PowerShell",
  "final_space": true,
  "transient_prompt": {
    "background": "transparent",
    "foreground": "#FFD700",
    "template": "{{if .Root}}\u26a1 {{end}}💫 "
  },
  "version": 2
}

```
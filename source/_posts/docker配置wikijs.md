---
title: docker配置wikijs 
date: 2022-11-06 15:14:22
mathjax: false
categories:
- 💾服务器
tags:
- 折腾
---

默认你已经安装好了docker

## 安装

安装centos
```
docker run -i -t -d -p 8888:8888 -p 888:888 -p 80:80 -p 3306:3306 --privileged=true -v [随便设置一个文件夹]:/wwwroot --name=baota centos:7 /usr/sbin/init
```

进入docker内的centos
```
docker exec -i -t baota /bin/bash
```


安装宝塔
```
yum install -y wget && wget -O install.sh http://download.bt.cn/install/install_6.0.sh && sh install.sh
```

进入宝塔后，在宝塔内地软件商店安装Postgresql和Node.js

![安装插件](/images/wikijs/安装插件.png)


![Node.js安装版本](/images/wikijs/安装插件_postgresql.png)
![Node.js安装版本](/images/wikijs/安装插件_nodejs.png)

下载[wiki.js](https://github.com/requarks/wiki/releases/)后在```/www/wwwroot```下解压

<!-- ![解压后看起来长这样](/images/wikijs/) -->

![配置数据库](/images/wikijs/配置数据库.png)
![配置数据库](/images/wikijs/配置数据库_2.png)

![配置插件_nginx](/images/wikijs/配置插件_nginx.png)
![配置网站_node](/images/wikijs/配置网站_node.png)
![配置网站文件](/images/wikijs/配置网站文件.png)
![配置网站配置文件](/images/wikijs/配置网站配置文件.png)
![配置网站离线](/images/wikijs/配置网站离线.png)
---
title: wireguard折腾记录
date: 2022-09-21 11:16:22
mathjax: false
mermaid: true
categories:
- 💾服务器
tags:
- 折腾
- 网络
- 分享
---

# wireguard折腾记录

- 条件： 有一个动态公网ip的服务器(openwrt)，NameSilo域名+DDNS

- 需求：有两个局域网需要打通and有一个无公网ip的机器需要内网传统。

## 尝试两个wireguard服务器互联

两个服务器都安装wireguard，然后配置好服务器端和客户端，然后两个服务器互联，可以互相ping通。
用ipv6链接，盲猜校园网可以免流。

等等先

## 解决动态公网ip变更后wireguard无法连接的问题

> 自动重连、自动更新配置文件、自动更新域名解析

> From: [https://v2ex.com/t/863087](https://v2ex.com/t/863087)

官网上说是不会自动解析域名，所以配置里面EndPoint填的是域名也没有用。

> Endpoint with changing IP
> After resolving a server's domain, **WireGuard will not check for changes in DNS again.**
> If the WireGuard server is frequently changing its IP-address due DHCP, Dyndns, IPv6, etc., any WireGuard client is going to lose its connection, until its endpoint is updated via something like wg set "$INTERFACE" peer "$PUBLIC_KEY" endpoint "$ENDPOINT".
> Also be aware, if the endpoint is ever going to change its address (for example when moving to a new provider/datacenter), just updating DNS will not be enough, so periodically running reresolve-dns might make sense on any DNS-based setup.
> Luckily, wireguard-tools provides an example script /usr/share/wireguard-tools/examples/reresolve-dns/reresolve-dns.sh, that parses WG configuration files and automatically resets the endpoint address.
> One needs to run the /usr/share/wireguard-tools/examples/reresolve-dns/reresolve-dns.sh /etc/wireguard/wg.conf periodically to recover from an endpoint that has changed its IP.
> One way of doing so is by updating all WireGuard endpoints once every thirty seconds[6] via a systemd timer:

```bash
git clone https://git.zx2c4.com/wireguard-tools /usr/share/wireguard-tools
```

```bash
# sudo vim /etc/systemd/system/wireguard_reresolve-dns.timer
[Unit]
Description=Periodically reresolve DNS of all WireGuard endpoints

[Timer]
OnCalendar=*:*:0/30

[Install]
WantedBy=timers.target
```

```bash
# sudo vim /etc/systemd/system/wireguard_reresolve-dns.service
[Unit]
Description=Reresolve DNS of all WireGuard endpoints
Wants=network-online.target
After=network-online.target

[Service]
Type=oneshot
ExecStart=/bin/sh -c 'for i in /etc/wireguard/*.conf; do /usr/share/wireguard-tools/contrib/reresolve-dns/reresolve-dns.sh "$i"; done'
```

```bash
sudo systemctl enable wireguard_reresolve-dns.service wireguard_reresolve-dns.timer --now
```
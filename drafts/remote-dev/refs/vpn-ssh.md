# Docker VPN 与 SSH 远程开发实践的参考记录

## 0. 这篇要解决什么问题

在厂商 Linux VPN 客户端可用的前提下，将 VPN 放进容器，通过 SSH 回连与 ProxyJump，让另一台服务器和它的开发容器复用内网 SSH 入口。本文为短篇操作记录，不是连载。

## 1. 一手资料清单

- 本次会话中的实际配置与验证，2026-10-18 至 2026-10-19；公开正文只保留匿名化的连接关系与排障结论。
- OpenSSH ssh_config(5)：https://man.openbsd.org/ssh_config#ProxyJump ，阅读 ProxyJump、HostKeyAlias、IdentityFile。
- OpenSSH ssh(1)：https://man.openbsd.org/ssh#R ，阅读远程转发绑定地址与连接目标。
- Docker host network：https://docs.docker.com/engine/network/drivers/host/ ，阅读与宿主机共用网络命名空间及平台条件。
- Docker bind mounts：https://docs.docker.com/engine/storage/bind-mounts/ ，阅读目录绑定、遮蔽已有内容与创建时配置。

## 2. 核心机制

本机 VPN 容器转发内网跳板的 SSH 到本机回环端口；另一台服务器通过 SSH 远程转发访问该端口。客户端配置 ProxyJump 后，目标仍独立验证发起端密钥。host 网络容器可使用宿主机回连端口，但 SSH 配置和密钥需要单独配置。

## 3. 代码落点

正文给出匿名化的 ssh_config 和 ssh -R 示例。配图为 src/components/network/VpnSshPath.astro，复用 FlowDiagram。没有将原始部署脚本、认证文件或主机清单复制进仓库。

## 4. 数字与条件

正文端口与私网地址为示例，不代表实际部署。未引用小样本测速数值作为性能结论；没有声明 HCCL 多卡测试通过。

## 5. 常见误解

容器 VPN 不会自动接管所有 Windows 应用；网络通不等于 SSH 身份可验证；保活不等于自动重连；设备健康查询不等于多卡通信验证；宿主机目录挂载不等于容器已绑定该目录。

## 6. 与其他文章的关系

归入服务器分类的独立实践记录，不加入模型连载。已有代理相关文章的应用背景不同，本篇聚焦 SSH 访问。

## 7. 配图

一张竖排访问路径图，强调远端到电脑的数据方向与电脑主动建立回连的区别。没有分步动画或数学公式。

## 8. 未确定的事项

目标未收到 VPN 直连的 SYN，只能将范围定位到 VPN 客户端或上游网络路径，不能确定某条网关策略是唯一原因。断网后部分机器认证失败，没有访问权限核实配置变化，也不能断言由谁修改。

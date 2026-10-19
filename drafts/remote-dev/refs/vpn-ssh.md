# Docker UniVPN 多实例部署参考

正文聚焦网络：每套 UniVPN 独立网络命名空间，socat 转发内网 TCP，宿主机以不同回环端口访问。删除 NPU、开发容器、数据盘以及与主线无关的远端回连过程。保留原文章地址。

## 来源与验证边界

- 本次实际部署：Docker Desktop Linux 容器中的 UniVPN 与 Windows VPN 共存；未声称已实测两套不同网关的容器并发。A/B 是扩展配置示例。
- 第三方镜像项目：https://github.com/jesusdf/huawei-vpn 。README 和本机固定摘要镜像中的 /entrypoint.sh、/opt/univpn/connect.exp 已核对。USER、TUN、NET_ADMIN、环境变量、重连行为均以该镜像为准。示例对成功提示做大小写兼容，并关闭 Expect 交互回显。
- Docker 网络：https://docs.docker.com/engine/network/drivers/host/ 。VPN 实例不共享 host 或其他容器网络命名空间。
- Docker 端口映射：https://docs.docker.com/engine/network/port-publishing/ 。回环绑定供本机访问。
- OpenSSH HostKeyAlias：https://man.openbsd.org/ssh_config#HostKeyAlias 。本机端口别名区分目标身份。

所有地址、认证字段、主机名均为示例；不提交实际部署脚本、密码、订阅或认证缓存。Docker Desktop 的开销不能用单个容器的资源上限代表。镜像来源与账号并发限制已在正文交代。

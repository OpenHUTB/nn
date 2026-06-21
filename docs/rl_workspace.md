````
# 强化学习

本文是基于机械臂的强化学习探索内容。
我们正在尝试构建一个模块化环境，通过提供标准的基于 REST 的接口来访问多个基于灵巧机械臂的仿真环境，用于训练基于强化学习的智能体。
另一方面，我们也在设计能够利用该环境架构进行学习并从中受益的智能体。
![RL_Architecture](/doc/images/rl_idea_architecture.png "MarineGEO logo")

## 快速开始

运行 Gazebo 仿真需执行以下命令，要求系统已安装 Docker。

```bash
# make sure your display variable is set
export DISPLAY=:0

# ensure the X server is open to connections from docker
xhost +local:docker

# build the container images for gazebo and ros
make build

# run everything
make up

# You can enter a running container by running the following command
docker exec -it [docker_id] /bin/bash

# You can run a new docker image (e.g. to use for development) for the base image using
docker run -it [image_id] /bin/bash
```

默认构建仅支持 CPU 的环境版本。
如果希望利用 GPU 进行模型推理或仿真，请运行 GPU 版本的 Docker 镜像：

```bash
make up-gpu
```

关于智能体的更多信息，请查看agent目录。
## 安装 pre-commit 工具

pre-commit会在代码提交前自动运行工具，完成代码检查和格式化工作。

```bash
pip install pre-commit
pre-commit install
```

安装完成后，该工具将在每次提交代码时自动运行。

## 安装新依赖

如需为 ROS 包添加新的 Python 依赖，请在对应包的根目录创建requirements.in文件，
然后在项目顶层的requirements.in中添加对该文件的引用。
最后通过pip-tools工具的pip-compile命令编译依赖：
```bash
pip install pip-tools
pip-compile

# or via docker
docker compose run base pip-compile
sudo chown -R $(id -u):$(id -g) requirements.txt
```
重新构建 Docker 镜像以拉取新添加的依赖。

## WSL2 使用注意事项
在 WSL2 中使用 Docker 的主机网络模式时，最终会使用 dockerd 的网络栈，而非 WSL2 环境内部的网络栈。
由于 ROS 需要动态分配端口且各节点使用固定 IP 地址，因此必须使用主机网络模式才能正常工作。
但这与运行 Web 服务的需求存在冲突 ——WSL2 中的 dockerd 环境与外部网络之间存在防火墙隔离。
解决方法是使用 Tailscale 创建连接到主机的 VPN 隧道：
```bash
docker run --name=tailscaled -v /var/lib:/var/lib -v /dev/net/tun:/dev/net/tun --network=host --cap-add=NET_ADMIN --cap-add=NET_RAW tailscale/tailscale
```
完成身份验证，并在需要与该服务交互的主机上也完成 Tailscale 身份验证即可。
## 生成文档

我们使用 Pandoc 工具从 Markdown 文件生成报告：

```bash
make report
```

## 演示

- 2023-10-03：执行随机动作的 FetchReach-v2 环境演示：: https://youtu.be/16np2Y5eIGA

## 笔记与参考资料

- https://gymnasium.farama.org/environments/mujoco/
- https://robotics.farama.org/
  - 注意：该库与撰写本文时最新的 MuJoCo 2.3.7 版本不兼容。
````


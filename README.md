# 神经网络实现代理

利用神经网络/ROS 实现 Carla（车辆、行人的感知、规划、控制）、AirSim、Mujoco 中人和载具的代理。

## 环境配置

* 平台：Windows 10/11，Ubuntu 20.04/22.04
* 软件：Python 3.7-3.12（需支持3.7）、Pytorch（尽量不使用Tensorflow）
* 相关软件下载 [链接](https://pan.baidu.com/s/1IFhCd8X9lI24oeYQm5-Edw?pwd=hutb)


## 贡献指南

准备提交代码之前，请阅读 [贡献指南](https://github.com/OpenHUTB/.github/blob/master/CONTRIBUTING.md) 。
代码的优化包括：注释、[PEP 8 风格调整](https://peps.pythonlang.cn/pep-0008/) 、将神经网络应用到Carla模拟器中、撰写对应 [文档](https://openhutb.github.io/nn/) 、添加 [源代码对应的自动化测试](https://docs.github.com/zh/actions/use-cases-and-examples/building-and-testing/building-and-testing-python) 等（从Carla场景中获取神经网络所需数据或将神经网络的结果输出到场景中）。

### 约定

* 每个模块位于`src/{模块名}`目录下，`模块名`需要用2-3个单词表示，首字母不需要大写，下划线`_`分隔，不能宽泛，越具体越好
* 每个模块的入口须为`main.`开头，比如：main.py、main.cpp、main.bat、main.sh等，提供的ROS功能以`main.launch`文件作为启动配置文件
* 每次pull request都需要保证能够通过main脚本直接运行整个模块，在提交信息中提供运行效果截图，README.md文档中提供运行环境和运行步骤的说明
* 仓库尽量保存文本文件，二进制文件需要慎重，如运行需要示例数据，可以保存少量数据，大量数据可以通过提供网盘链接并说明下载链接和运行说明


### 文档生成

测试生成的文档：
1. 安装python 3.11，并使用以下命令安装`mkdocs`和相关依赖：
```shell
pip install mkdocs -i http://mirrors.aliyun.com/pypi/simple --trusted-host mirrors.aliyun.com
pip install -r requirements.txt
```
（可选）安装完成后使用`mkdocs --version`查看是否安装成功。

2. 在命令行中进入`nn`目录下，运行：
```shell
mkdocs build
mkdocs serve
```
然后使用浏览器打开 [http://127.0.0.1:8000](http://127.0.0.1:8000)，查看文档页面能否正常显示。

## 参考

* [代理模拟器文档](https://openhutb.github.io)
* 已有相关 [无人车](https://openhutb.github.io/doc/used_by/) 、[无人机](https://openhutb.github.io/air_doc/third/used_by/) 、[具身人](https://github.com/google-deepmind/mujoco/network/dependents) （整理之后的 [链接](https://openhutb.github.io/doc/pedestrian/humanoid/) ） 的实现
* [神经网络原理](https://github.com/OpenHUTB/neuro)



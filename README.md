# 神经网络实现代理

实现基于Carla的车辆、行人的感知、规划、控制。

## 环境配置
* 平台：Windows 10/11，Ubuntu 20.04/22.04
* 软件：Python 3.7-3.12（需支持3.7）、Pytorch（不使用Tensorflow）

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

## 贡献指南
准备提交代码之前，请阅读 [贡献指南](https://github.com/OpenHUTB/.github/blob/master/CONTRIBUTING.md) 。
代码的优化包括：注释、[PEP 8 风格调整](https://peps.pythonlang.cn/pep-0008/) 、将神经网络应用到Carla模拟器中、撰写对应 [文档](https://openhutb.github.io/nn/) 、添加 [源代码对应的自动化测试](https://docs.github.com/zh/actions/use-cases-and-examples/building-and-testing/building-and-testing-python) 等（从Carla场景中获取神经网络所需数据或将神经网络的结果输出到场景中）。


## 参考

* [代理模拟器文档](https://openhutb.github.io/carla_doc/)
* [已有相关实现](https://openhutb.github.io/carla_doc/used_by/)
* [神经网络原理](https://github.com/OpenHUTB/neuro)

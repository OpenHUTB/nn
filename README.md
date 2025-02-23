# 神经网络实现代理
实现基于Carla的车辆、行人的感知、规划、控制。

## 环境配置
* 平台：Windows 10/11，Ubuntu 20.04/22.04
* 软件：Python 3.7-3.12（需支持3.7）、Pytorch（不使用Tensorflow）

在Windows 10 和Windows 11上测试了生成文档：
1. 安装python 3.11，并使用以下命令安装`mkdocs`
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

* [代理模拟器文档](https://openhutb.github.io/carla_doc/)
* [已有相关实现](https://openhutb.github.io/carla_doc/used_by/)
* [神经网络原理](https://github.com/OpenHUTB/neuro)

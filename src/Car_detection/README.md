# 自动驾驶车道与路径检测

基于神经网络实现自动驾驶场景中的车道线识别与路径预测，通过处理摄像头视频流实时输出车道和路径信息。

## 环境配置

* 平台：Ubuntu 18.04
* 软件：Python 3.6.9、pip 21.3.1、TensorFlow 1.14.0
* 相关依赖：见requirements.txt

### 安装步骤

1. 安装系统依赖库：
```shell
sudo apt install libssl-dev libcurl4-openssl-dev curl
sudo apt install libarchive-dev
```

2. 配置Python环境：
```shell
# 确保Python版本为3.6.9
python3 --version

# 安装指定版本setuptools
pip3 install setuptools==45.2.0
# 若已安装高版本，先卸载
# pip3 uninstall setuptools
```

3. 安装Python依赖：
```shell
pip3 install -r requirements.txt
```

## 贡献指南

准备提交代码之前，请确保遵循以下规范，以保证项目的可维护性和一致性。

### 约定

* 核心功能模块位于`common/`目录下，工具类脚本位于`common/tools/`目录下
* 主程序入口为`main.py`，最小化示例为`minimal.py`
* 每次提交PR需保证代码可运行，提交信息中需包含运行效果截图
* 新增功能需配套添加测试用例，位于`common/tools/lib/tests/`目录下
* 尽量避免提交大型二进制文件，模型文件和测试数据可提供下载链接

### 代码规范

* 遵循PEP 8代码风格规范
* 关键函数和复杂逻辑需添加详细注释
* 新增模块需在README中补充说明其功能和使用方法

### 测试要求

* 确保新功能通过现有测试用例
* 新增功能需添加对应的单元测试，参考`test_readers.py`的测试风格
* 测试通过后再提交PR

## 参考

* 数据集参考：[comma2k19](https://github.com/commaai/comma2k19)
* 相关库文档：
  * [TensorFlow 1.14](https://www.tensorflow.org/versions/r1.14/api_docs)
  * [OpenCV](https://docs.opencv.org/4.2.0/)
  * [Matplotlib](https://matplotlib.org/3.2.1/contents.html)
* 视频处理：Matroska格式解析参考`common/tools/lib/mkvparse/`目录下的实现
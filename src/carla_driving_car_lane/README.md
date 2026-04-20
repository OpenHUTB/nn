
======================================
   
##   自动驾驶车道与路径检测
   
======================================


本项目基于 Windows11 与 Python 3.7 环境开发，
主要实现自动驾驶场景下的车道线识别与行驶路径实时检
测，可对车载摄像头采集的视频流（如 .hevc 格式行车视频）进行解析，通过视觉算法识别道路车道线、规划安
全行驶轨迹，为自动驾驶决策提供车道感知与路径引导支持。
项目面向自动驾驶辅助系统（ADAS）的基础视觉感知模块，依赖系统底层库与 Python 机器学习 / 视觉处理工具链，可快速部署运行，

完成对行车视频的离线车道与路径分析，适用于自动驾驶算法验证、智能驾驶教学实验等场景。


# 操作系统
	Windows11

# Python
	python Version 3.7
	
	pip Version 21.3.1
	
	Setuptools Version & Install
	
	pip3 install setuptools==45.2.0
	
	(高版本卸载命令pip3 uninstall setuptools）

# 安装依赖
	安装库文件

	sudo apt install libssl-dev libcurl4-openssl-dev curl

	sudo apt install libarchive-dev

# 安装
	pip3 install -r requirements.txt

# 运行程序
  python3 main.py  ../sample.hevc 

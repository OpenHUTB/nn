Active Lane Keeping Assistant (主动式车道保持辅助系统)
这是一个基于 CARLA 模拟器 的自动驾驶车道保持项目。该系统使用计算机视觉技术从摄像头图像中检测车道线，并利用 PID 控制器（或自适应 Twiddle 算法）控制车辆的转向，使其保持在车道中心行驶。


项目简介
该项目主要由三个核心模块组成：
视觉感知 (Lane)：使用 OpenCV 进行图像处理，包括 HLS 阈值过滤、透视变换（鸟瞰图）、滑动窗口车道检测和二阶多项式拟合。
决策控制 (Agent)：实现了 Simple、P、PD 和 PID 四种控制器，并包含 Twiddle 算法用于自动调整 PID 参数。
仿真环境 (World)：封装了 CARLA API，负责车辆生成、传感器管理（RGB 相机、碰撞传感器）以及同步仿真步进。


环境依赖
本项目依赖于 CARLA 模拟器环境。请确保您已安装以下依赖：Python 3.xCARLA Simulator (推荐版本 0.9.13 1)主要的 Python 库依赖如下（详见 requirements.txt 2）：
carla==0.9.13
opencv-python==4.7.0.72
numpy==1.20.0
matplotlib==3.7.1
安装步骤克隆本仓库到本地。启动 CARLA 模拟器（服务器端）。安装 Python 依赖：Bashpip install -r requirements.txt


使用方法
项目的入口文件是 main.py。你可以通过命令行参数来控制运行模式。
基本运行
运行默认的 PID 控制器，行驶 1000 步，并指定本次运行的 ID 为 test_run：Bashpython main.py -id test_run -c pid -s 1000
命令行参数说明
参数,简写,描述,默认值
--identifier,-id,(必填) 本次运行的唯一标识符，用于生成日志和结果文件夹。,无
--controller,-c,"选择控制器类型，可选值：simple, p, pd, pid。",pid
--steps,-s,车辆运行的步数。,100
--adapt,-a,是否开启参数自适应模式（Twiddle 算法调参）。,False
--tolerance,-t,Twiddle 算法的容差值。,0.2
--debug,-d,是否开启调试日志模式。,False
进阶用法：参数自适应 (Twiddle)如果你想让 Agent 自动寻找最优的 PID 参数，可以使用 -a 参数开启 Twiddle 算法：Bashpython main.py -id auto_tune -c pid -s 500 -a -t 0.1
注意：Twiddle 算法仅支持 pid 控制器。


 项目结构
 Plaintextsrc/
├── agent.py           # 智能体类：包含 PID 控制器逻辑和 Twiddle 算法
├── lane.py            # 车道检测类：处理图像，计算车道线偏差 (Error)
├── main.py            # 主程序：处理参数，运行仿真循环
├── recorder.py        # 录像机类：将仿真画面保存为视频文件
├── world.py           # 环境类：管理 CARLA 客户端、地图加载和 Actor 生成
└── requirements.txt   # 依赖列表 
核心算法流程图像获取：
World 类从 CARLA 的 RGB 摄像头获取原始图像。
预处理：Lane.get_lines() 将图像转换为 HLS 空间并提取白色车道线。
透视变换：Lane.extract_roi() 将图像转换为鸟瞰图（BEV），以便更准确地测量曲率。
车道拟合：Lane.get_line_fits() 使用滑动窗口算法定位车道像素，并拟合二阶多项式曲线。
误差计算：计算车辆中心与车道中心的偏差值（Error）。
控制输出：Agent 根据偏差值，通过 PID 控制器计算转向角度（Steer）。
执行控制：将转向和油门指令发送回 CARLA 车辆。


结果输出
运行后，程序会在 assets/<id>/ 目录下生成以下内容：日志文件 (alka_<id>.log)：记录运行过程中的参数和误差。误差曲线图 (<id>_error.jpg)：展示行驶过程中的横向偏差。中间过程图片 (如果在代码中开启 save=True)：包括二值化图、滑动窗口图、原始图像叠加图等。
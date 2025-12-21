无人车安全运维模块 - 环境配置与运行手册
一、环境设置
运行本项目安全运维模块代码前，需确保开发环境满足以下依赖版本要求：
Python 3.8 及以上版本（适配运维数据处理与实时监控需求）
PyTorch 1.8 及以上版本（支持异常检测模型训练推理）
CARLA 模拟器 0.9.13 版本（适配无人车全场景运维仿真）
OpenCV 4.5 及以上版本（用于传感器数据可视化与异常识别）
其余 Python 依赖项详见项目根目录下的 'requirements_ops.txt' 文件
二、依赖安装步骤
1. 克隆项目仓库
bash
运行
git clone https://github.com/yourusername/unmanned_vehicle_safety_and_operation.git
cd av_safety_operation
2. 安装 Python 依赖
bash
运行
pip install -r requirements_ops.txt
三、代码运行方法
运行安全运维仿真
执行以下命令启动 CARLA 模拟器，加载无人车状态监控、故障诊断、应急处置全模块：
bash
运行
python run_safety_operation.py --town Town07 --vehicle_num 3
四、模型训练与测试
1. 训练故障诊断模型
执行以下命令加载无人车传感器故障数据集，训练基于多模态融合的故障诊断模型：
python main_ops.py --mode train --dataset_path ./dataset/fault_data --epochs 50 --batch_size 32
2. 测试故障诊断模型
模型训练完成后，执行以下命令加载测试数据集，评估模型故障识别准确率、召回率等指标：
python main_ops.py --mode test --model_path ./models/trained_fault_model.pth --test_dataset ./dataset/test_fault_data
3. 应急策略验证
测试故障诊断模型后，执行以下命令验证应急处置策略有效性：
python verify_emergency.py --scenario fault_sensor --strategy emergency_stop
五、模型部署
说明：以下为无人车安全运维模块在车载终端 / 云端的部署步骤，适配主流车载计算平台（如 NVIDIA Drive、地平线 Journey）。
1. 车载端模型训练与导出
推理前需将训练好的故障诊断模型导出为车载平台兼容的 ONNX/TensorRT 格式：
python ./deploy/train_export.py --model_path ./models/trained_fault_model.pth --export_format tensorrt --device jetson
2. 车载端加载推理
在车载计算平台部署运维模块，实时监控无人车状态并推理故障：
python ./deploy/vehicle_deploy.py --onnx_model ./models/ops_model_trt.onnx --can_bus_port /dev/ttyUSB0 --monitor_freq 10
3. 云端运维监控部署
在云端部署运维监控后台，实现多车运维数据汇总、故障预警、远程干预：
python ./deploy/cloud_deploy.py --server_ip 192.168.1.100 --port 8080 --max_vehicle 100
总结
环境核心要求：Python 3.8+、PyTorch 1.8+、CARLA 0.9.13，需配置 CARLA 环境变量及车载硬件驱动；
核心操作命令：
仿真运行：run_safety_operation.py（指定地图 / 车辆数）；
模型训练 / 测试：main_ops.py --mode [train/test]；
应急策略验证：verify_emergency.py（指定故障场景 / 处置策略）；
部署关键步骤：
车载端：训练后导出 TensorRT/ONNX 模型，通过vehicle_deploy.py加载实时推理；
#CVIPS: 面向提升行人安全的协同视觉 (Connected Vision for Increased Pedestrian Safety)
CVIPS 项目的实现。
- [X] 数据集
- [X] CARLA 数据生成代码



## 数据集 (Dataset)
CVIPS 数据集可在此处获取:[here](https://drive.google.com/drive/folders/1gCCrIslzVkupyF0lj_1I9qXTB2_a4tjd?usp=drive_link).





## 安装 (Installation)
请参考 [installation](https://carla.readthedocs.io/en/0.9.14/build_windows) 了解如何在 Windows 上设置 CARLA。


## 数据集生成 (Dataset Generation)
我们的数据集是使用 CARLA 模拟器生成的，为协同感知提供了多样化的场景。
###运行数据生成脚本 (Running the Data Generation Script)

1. 克隆此仓库
   ```
   git clone https://github.com/cvips/CVIPS.git
   cd CVIPS
   ```

2. 安装所需的依赖包:
   ```
   pip install -r requirements.txt
   ```

3. 确保 CARLA 已正确安装并运行。在一个单独的终端中启动 CARLA 服务器:
   ```
   /path/to/carla/CarlaUE4.exe
   ```

4. 运行数据生成脚本:
   ```
   python cvips_generation.py
   ```

   该脚本将连接到 CARLA 服务器，并根据指定的参数生成数据集。
注意：请根据你特定的设置需求，调整 cvips_generation.py 中的 CARLA 服务器路径以及任何配置参数。
### Sample Examples

#示例命令 (Sample Examples)

# 1. 默认配置：在 Town04 生成 10 辆车辆，天气为晴天，时间为中午
```shell
python generate_vehicles_only.py --town Town04
```
# 2. 自定义车辆数量：在 Town04 生成 15 辆车辆
```shell
python generate_vehicles_only.py --town Town04 --num_vehicles 15
```
# 3. 自定义天气：在 Town04 生成 10 辆车辆，天气为雨天
```shell
python generate_vehicles_only.py --town Town04 --weather rainy
```
# 4. 自定义时间：在 Town04 生成 10 辆车辆，时间为夜晚
```shell
python generate_vehicles_only.py --town Town04 --time_of_day night
```
# 5. 完整自定义：在 Town07 生成 20 辆车辆，天气为多云，时间为日落
```shell
python generate_vehicles_only.py --town Town07 --num_vehicles 20 --weather cloudy --time_of_day sunset
```
# 6. 设置随机种子（保证场景可复现）：在 Town10HD 生成 12 辆车辆，使用种子 123
```shell
python generate_vehicles_only.py --town Town10HD --num_vehicles 12 --seed 123
```
这些示例展示了城镇、交叉路口类型、天气状况、时间段，车辆和摄像头设置的各种组合。用户可以根据需要修改这些参数以生成不同的场景。

### 样本可视化 (Sample Visualizations)

我们提供了可视化结果来展示我们数据集中的不同视角:

![动图加载失败](images/2025-11-24102317-ezgif.com-speed.gif "车辆环境演示1")


## 致谢 (Acknowledgement)
本项目基于以下开源项目: BEVerse, Fiery, open-mmlab, 以及 DeepAccident。


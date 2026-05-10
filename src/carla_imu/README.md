# Carla_IMU_Classifier [![forthebadge made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/) [![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
适用于 CARLA 模拟器的 IMU（惯性测量单元）传感器数据接收器与分类器。两个脚本均默认选用 `Town03` 地图，已禁用开局随机生成车辆；取消注释以下代码行即可启用随机生成功能：
```spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()```

## 重要注意事项 | IMPORTANT NOTICE
使用本仓库前，**必须先编译构建 CARLA 模拟器**。
构建完成后，将脚本复制到路径：`{PATH}\CARLA\PythonAPI\examples` 文件夹下。
同时，务必通过以下命令启动 CARLA 服务端：执行 `{PATH}\CARLA\CARLAUE4.exe` 并添加 `-carla-server` 参数。

You have to build Carla simulator before use this repo.
After build, just copy the scripts into `{PATH}\CARLA\PythonAPI\examples` folder.
Also, do not forget to execute carla server via `{PATH}\CARLA\CARLAUE4.exe` with `-carla-server` argument.

---

## `data_reciever.py`
通过该脚本可采集 IMU（可根据配置采集其他传感器）数据，新增自定义参数 `name` 用于标注驾驶员名称。

### 中文
使用命令示例：
```python data_reciever.py --filter name --name John29```
脚本会自动保存文件 `out_John29.csv`，文件包含6轴 IMU 数据，且标签列统一填充为 `John29`。

### English
Usage Example:
```python data_reciever.py --filter name --name John29```
The script will save a file named `out_John29.csv`, which contains 6-axis IMU data with a label column filled with `John29`.

### 输出格式示例 | Sample Output
class | accelX | accelY | accelZ | gyroX | gyroY | gyroZ
-- | -- | -- | -- | -- | -- | --
John29 | -0.329013 | 1.111466 | 9.943973 | 0.064446 | -0.0759 | -0.095295
John29 | -0.329013 | 1.111466 | 9.943973 | 0.064446 | -0.0759 | -0.095295

已上传 CSV 示例文件：`examples/out_mehdi_test.csv`
Sample CSV file uploaded: `examples/out_mehdi_test.csv`

---

## `classifier.py`
### 中文
可加载已训练的神经网络模型（TensorFlow `.h5` 格式）并完成预测。
本项目神经网络输入尺寸为 `(1,20,6)`，对应格式：(批次, 时间步长, 特征维度)。
代码第1074行：
```data = np.array(data).reshape(1,20,6)```
已将数据重塑为模型适配尺寸，可根据自身配置修改该行代码。

### English
Load and make predictions with a trained neural network (TensorFlow, `.h5` format).
The input shape of the neural network is `(1,20,6)` = (Batch, timesteps, features).
Line 1074:
```data = np.array(data).reshape(1,20,6)```
Reshapes data to match the model input. Modify this line for your custom configuration.

---

## 环境依赖 | Requirements
- Python (3.6-3.7)
- tensorflow
- pandas
- pygame
- carla
- numpy

---

## 最新更新 | NEWS
✅ 新增支持 AIRSIM 的 IMU 数据接收器！
将 `car_reciever.py` 复制到路径：`path_in_ur_system\AirSim\PythonClient\car`，即可采集 IMU 数据！

IMU Data receiver for AIRSIM added!!!
Just copy `car_reciever.py` to `path_in_ur_system\AirSim\PythonClient\car` and collect IMU data!

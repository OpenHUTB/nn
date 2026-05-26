# CARLA IMU 传感器数据接收与分类

[![made-with-python](http://ForTheBadge.com/images/badges/made-with-python.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

适用于 CARLA 模拟器的 IMU（惯性测量单元）传感器数据接收器与分类器。默认选用 `Town03` 地图，已禁用开局随机生成车辆；取消注释以下代码行即可启用随机生成功能：
```python
spawn_point = random.choice(spawn_points) if spawn_points else carla.Transform()
```

## 重要注意事项

使用本仓库前，**必须先编译构建 CARLA 模拟器**。构建完成后：

1. 将脚本复制到 `{PATH}\CARLA\PythonAPI\examples` 目录
2. 通过 `{PATH}\CARLA\CARLAUE4.exe -carla-server` 启动 CARLA 服务端

> You have to build Carla simulator before use this repo. After build, copy the scripts into `{PATH}\CARLA\PythonAPI\examples` folder, and execute carla server via `{PATH}\CARLA\CARLAUE4.exe` with `-carla-server` argument.

---

## `data_reciever.py`

通过该脚本采集 IMU（可根据配置采集其他传感器）数据，新增自定义参数 `name` 用于标注驾驶员名称。

### 使用示例

```bash
python data_receiver.py --filter name --name John29
```

脚本会自动保存文件 `out_John29.csv`，包含 6 轴 IMU 数据，且标签列统一填充为 `John29`。

### 输出格式

| class  | accelX    | accelY  | accelZ   | gyroX   | gyroY    | gyroZ     |
| ------ | --------- | ------- | -------- | ------- | -------- | --------- |
| John29 | -0.329013 | 1.111466 | 9.943973 | 0.064446 | -0.0759  | -0.095295 |
| John29 | -0.329013 | 1.111466 | 9.943973 | 0.064446 | -0.0759  | -0.095295 |

已上传 CSV 示例文件：`examples/out_mehdi_test.csv`

---

## `classifier.py`

可加载已训练的神经网络模型（TensorFlow `.h5` 格式）并完成预测。

本项目神经网络输入尺寸为 `(1, 20, 6)`，对应格式：(批次, 时间步长, 特征维度)。代码中已通过以下方式将数据重塑为模型适配尺寸，可根据自身配置修改：

```python
data = np.array(data).reshape(1, 20, 6)
```

---

## 环境依赖

- Python 3.6-3.7
- tensorflow
- pandas
- pygame
- carla
- numpy

---

## 更新

已新增支持 AirSim 的 IMU 数据接收器。将 `car_reciever.py` 复制到 `path_in_ur_system\AirSim\PythonClient\car` 目录，即可采集 IMU 数据。

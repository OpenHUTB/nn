# Data Collection

基于 CARLA 模拟器的数据采集功能模块。

## 功能特性

- CARLA 模拟器连接与配置
- 多传感器数据采集
- 多城镇场景支持
- 自定义行驶路线配置

## 环境配置

### CARLA 安装

```bash
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
tar -xvzf CARLA_0.9.10.1.tar.gz -C carla09101
```

### Conda 环境

```bash
cd CBS2
conda env create -f docs/cbs2.yml
conda activate cbs2
```

### 环境变量

```bash
export CARLA_ROOT=<your_path>/carla09101
export CBS2_ROOT=<your_path>/CBS2
export LEADERBOARD_ROOT=${CBS2_ROOT}/leaderboard
export SCENARIO_RUNNER_ROOT=${CBS2_ROOT}/scenario_runner
export PYTHONPATH=${PYTHONPATH}:"${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}"
```

## 配置文件

- `CBS2/autoagents/collector_agents/config_data_collection.yaml`
- `CBS2/autoagents/collector_agents/collector.py`
- `CBS2/rails/data_phase1.py`
- `CBS2/assets/`

## 数据采集步骤

### 启动 CARLA

```bash
source ~/.bashrc
$CBS2_ROOT/scripts/launch_carla.sh 1 2000
```

### 执行采集

```bash
cd CBS2
python rails/data_phase1.py --port 2000 --num-runner=1
```

## 数据格式

- RGB 图像: `camera/rgb/*.png`
- 语义分割标签: `camera/semantic/*.png`
- 深度图像: `camera/depth/*.png`
- 采集日志: `record.txt`

## 注意事项

- 多进程运行时端口号作为进程端口递增偏移量
- 采集完成后检查数据完整性
- 确保有足够磁盘空间
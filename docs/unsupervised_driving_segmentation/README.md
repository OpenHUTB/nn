# USAD

本仓库实现了一种专门面向自动驾驶应用的无监督分割模型。本项目旨在开发一种稳定可靠的分割算法，无需依赖标注数据，可为现实自动驾驶场景提供可落地的通用解决方案。

项目整体分为多个关键环节来完成语义分割任务，整体流程如下：

- data_collection
- data_processing
- image_cropping
- knn_index_precomputation
- model_training
- model_evaluation
- real_time_segmentation

## 代码来源说明

本仓库整合了其他开源项目代码：

### 修改使用

- Cheating by Segmentation 2
- STEGO

### 直接引用未做修改

- DriveAndSegment

## CARLA安装教程

在自定义路径下载安装CARLA：

```bash
wget https://carla-releases.s3.eu-west-3.amazonaws.com/Linux/CARLA_0.9.10.1.tar.gz
tar -xvzf CARLA_0.9.10.1.tar.gz -C carla09101
```

在指定路径克隆USAD项目仓库：

```bash
git clone https://github.com/flixtkc/Unsupervised-Segmentation-for-Autonomous-Driving-Cars.git
cd Unsupervised-Segmentation-for-Autonomous-Driving-Cars
```

本项目需要创建两个独立的Conda虚拟环境，请提前安装Anaconda。

```bash
cd CBS2
conda env create -f docs/cbs2.yml
conda activate cbs2
```

确认第一个环境配置无误后，返回根目录创建第二个环境：

```bash
cd ..
cd STEGO
conda env create -f environment.yml
conda activate stego
```

将以下环境变量添加到 ~/.bashrc 配置文件中：

```bash
export CARLA_ROOT=<your_path>/carla09101
export CBS2_ROOT=<your_path>/CBS2
export LEADERBOARD_ROOT=${CBS2_ROOT}/leaderboard
export SCENARIO_RUNNER_ROOT=${CBS2_ROOT}/scenario_runner
export PYTHONPATH=${PYTHONPATH}:"${CARLA_ROOT}/PythonAPI/carla/":"${SCENARIO_RUNNER_ROOT}":"${LEADERBOARD_ROOT}":"${CARLA_ROOT}/PythonAPI/carla/dist/carla-0.9.10-py3.7-linux-x86_64.egg"
```

激活cbs2环境，执行命令启动CARLA验证环境是否正常：

```bash
source ~/.bashrc
$CBS2_ROOT/scripts/launch_carla.sh 1 2000
```

## 配置说明

进行数据采集前需要配置多个文件，可使用任意文本编辑器查看以下路径文件：

- `CBS2/autoagents/collector_agents/config_data_collection.yaml`
- `CBS2/autoagents/collector_agents/collector.py`
- `CBS2/rails/data_phase1.py`
- `CBS2/assets/`

`config_data_collection.yaml`文件包含数据采集阶段的全部参数设置，其中分辨率配置对分割模型最终性能影响最大。

`collector.py`文件定义了所有数据采集相关功能函数，同时内置了wandb日志工具配置，默认关闭日志记录。

`data_phase1.py`用于配置单次自动驾驶采集任务的专属参数。

`assets`文件夹存放各个仿真城镇的行驶路线与场景文件，是数据采集运行的必要文件，需要在`data_phase1.py`中指定调用对应的路线配置。

## 数据采集步骤

开启两个终端窗口，第一个终端启动CARLA服务，第二个终端进入CBS2目录执行数据采集脚本。

**终端1执行**：

```bash
$CBS2_ROOT/scripts/launch_carla.sh <运行进程数> <端口号>
```

**终端2执行**：

```bash
cd CBS2
python rails/data_phase1.py --port <端口号> --num-runner=<运行进程数>
```

注意：多进程运行时，端口号同时作为各进程的端口递增偏移量。

## 数据格式处理

确认采集的数据合格后，进行数据格式转换。原始数据需要经过处理才能用于STEGO模型训练。回到项目根目录，运行数据集格式转换脚本，运行前需手动设置数据集输入路径和输出路径。

## 环境切换

数据处理完成后，退出cbs2环境并激活stego环境，后续所有操作均需在该环境下进行。

## STEGO模型配置

进入配置文件目录修改参数：

```bash
cd STEGO/src/
vi configs/train_config.yml
```

填写数据集路径，与上一步数据转换的输出路径保持一致；同时调整图像裁剪相关超参数，避免训练过程出现内存溢出错误。

### 裁剪参数配置示例

无标签数据，采用五点裁剪模式，裁剪比例0.5，分辨率200，中心加载裁剪。

裁剪模式支持四角加中心五点裁剪，或随机裁剪；裁剪比例控制裁剪区域相对原图的大小。调整参数至稳定可用后，修改配置文件中的数据集路径，指向裁剪后新数据集文件夹。

## knn_index_precomputation

为大幅加快训练速度，必须提前执行KNN索引预计算：

```bash
python precompute_knns.py
```

## model_training

配置全部完成后，运行训练脚本，在CARLA仿真采集数据集上训练分割模型。需提前在配置文件中指定日志保存文件夹。

```bash
python train_segmentation.py
```

训练过程可通过TensorBoard实时查看训练日志与指标变化，示例命令：

```bash
tensorboard --logdir logs/logs/five_crop_0.5/directory_new_crop_date_Jul25_02-25-32/default/version_0/
```

## model_evaluation

可通过TensorBoard日志查看各项评估指标，也可运行实时分割脚本进行实际效果测试。论文原版暂未实现真正实时部署，后续版本将优化完善。项目还提供通用视频、图片分割脚本，加载训练好的模型即可对普通媒体文件做语义分割并保存结果，可自行注释启用图片或视频分割功能，测试示例可查看testing_videos文件夹。

执行测试脚本命令：

```bash
cd STEGO/src/
python STEGO_create_segmented_video_or_image.py
python STEGO_real_time_segmenter.py
```

## 额外测试工具

项目附带多个辅助测试脚本，可用于测试程序批处理大小、进程线程数、SSH远程X11转发等功能，用于排查各类运行报错问题。

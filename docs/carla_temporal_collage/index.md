# Temporal Collage Prompting：驾驶事故视频识别系统

## 项目简介

本项目实现了基于 GPT-4o 的低成本模拟器驾驶事故视频识别系统，采用时间拼接提示（Temporal Collage Prompting）方法，通过将视频帧拼接成 collage 图片，利用 GPT-4o 的视觉能力进行事故分类。

## 技术架构

### 整体流程

```
视频数据 → 帧提取 → Collage生成 → GPT-4o分析 → 结果输出
```

### 模块说明

| 模块 | 功能 | 对应脚本 |
|------|------|----------|
| 帧提取 | 从视频中提取关键帧 | `main.py extract-frames` |
| Collage生成 | 将帧拼接成网格图片 | `main.py create-collage` |
| 事故分析 | 使用GPT-4o进行分类 | `main.py analyze` |

## 快速开始

### 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 运行命令

#### 1. 提取视频帧
```bash
python src/main.py extract-frames \
    --input data/videos \
    --output data/data-frames/data-frames-3fps \
    --interval 10
```

#### 2. 生成 Collage
```bash
python src/main.py create-collage \
    --input data/data-frames/data-frames-3fps \
    --output data/collages/collages-3fps-2-3 \
    --layout 2-3
```

#### 3. 事故分析
```bash
python src/main.py analyze \
    --input data/collages/collages-3fps-2-3 \
    --model gpt-4o-low
```

## 数据集

### 数据结构

```
data/
├── videos/          # 原始视频数据
│   ├── norm/        # 正常驾驶 (30个视频)
│   ├── ped/         # 行人事故 (15个视频)
│   └── col/         # 车辆碰撞 (15个视频)
├── data-frames/     # 提取的视频帧
└── collages/        # 生成的Collage图片
```

### 数据说明

- **视频分辨率**: 1280x720
- **帧率**: 30fps（提取帧时使用3fps）
- **时长**: 每个视频约10-30秒
- **场景**: CARLA模拟器生成的驾驶场景

## 实验结果

### 分类性能

| 指标 | 值 |
|------|-----|
| 准确率 | 85% |
| 行人事故识别 | Precision: 100%, Recall: 93% |
| 车辆碰撞识别 | Recall: 93% |

### 混淆矩阵

```
              预测
            Normal  Ped  Col
真实  Normal    28    0    2
     Ped         0   14    1
     Col         3    0   12
```

## 引用

如果您觉得我们的工作对您有帮助，请引用：

```bibtex
@inproceedings{suntichaikul2024temporal,
    title        = {{Temporal Collage Prompting: A Cost-Effective Simulator-Based Driving Accident Video Recognition With GPT-4o}},
    author       = {Suntichaikul, Pratch and Taveekitworachai, Pittawat and Nukoolkit, Chakarida and Thawonmas, Ruck},
    year         = 2024,
    booktitle    = {2024 8th International Conference on Information Technology (InCIT)},
    pages        = {708--713},
    doi          = {10.1109/InCIT63192.2024.10810536}
}
```

## 许可证

本项目采用 MIT 许可证，详见 [LICENSE](../LICENSE) 文件。

# Temporal Collage Prompting: 驾驶事故视频识别项目报告

## 项目概述

这是一个基于 GPT-4o 的驾驶事故视频识别系统，使用 CARLA 模拟器生成的视频数据进行测试。

**论文标题**: Temporal Collage Prompting: A Cost-Effective Simulator-Based Driving Accident Video Recognition With GPT-4o

**会议**: 2024 8th International Conference on Information Technology (InCIT 2024)

---

## 目录结构

```
carla-temporal-collage-prompting-main/
├── data/
│   ├── videos/          # 原始视频数据
│   │   ├── norm/        # 正常视频 (30个)
│   │   ├── ped/         # 行人事故视频 (15个)
│   │   └── col/         # 车辆碰撞视频 (15个)
│   ├── data-frames/     # 提取的视频帧
│   └── collages/        # 生成的collage图片
├── experiments/         # 实验日志
│   ├── high/           # 高质量图片实验
│   └── low/            # 低质量图片实验
├── src/                # 源代码
│   ├── collage-4o-high.py
│   ├── collage-4o-low.py
│   ├── dir2Collages-2-2.py
│   ├── dir2Collages-2-3.py
│   ├── dir2Collages-3-2.py
│   └── dirVid2frames.py
└── requirements.txt
```

---

## 任务分类

系统将视频分为三类：

| 类别 | 标签 | 描述 | 数量 |
|------|------|------|------|
| Normal | norm | 正常驾驶，无事故 | 30 |
| Pedestrian Accident | ped | 车辆撞到过马路的行人 (Type A) | 15 |
| Collision | col | 车辆与其他车辆相撞 (Type B) | 15 |
| **总计** | | | **60** |

---

## 实验结果 (Collages-3fps-2-3, High Quality)

### 混淆矩阵

```
[[23  0  7]
 [ 1 14  0]
 [ 1  0 14]]
```

### 分类报告

| 类别 | Precision | Recall | F1-Score | Support |
|------|-----------|--------|----------|---------|
| Normal | 0.92 | 0.77 | 0.84 | 30 |
| Pedestrian Accident | 1.00 | 0.93 | 0.97 | 15 |
| Collision | 0.67 | 0.93 | 0.78 | 15 |
| **Accuracy** | | | **0.85** | **60** |
| **Macro Avg** | 0.86 | 0.88 | 0.86 | 60 |
| **Weighted Avg** | 0.88 | 0.85 | 0.85 | 60 |

### 详细分析

**性能亮点**：
- 🔴 **行人事故检测** - 表现最佳，Precision 100%，Recall 93%
- 🟡 **车辆碰撞检测** - Recall 93%，但 Precision 较低 (67%)
- 🟢 **正常驾驶检测** - 总体良好，Precision 92%，但有 7 个误报

**误分析**：
- 7 个正常视频被误判为车辆碰撞
- 1 个行人事故被误判为正常
- 1 个车辆碰撞被误判为正常

### Token 使用统计

- **Input Tokens**: 505,325
- **Output Tokens**: 9,528

---

## 实验方法

### 1. 视频处理流程

```
原始视频 → 视频帧提取 → Collage 生成 → GPT-4o 分析 → 结果分类
```

### 2. Collage 布局

项目测试了三种不同的 Collage 布局：

| 布局 | 说明 |
|------|------|
| 2×2 | 每幅 Collage 包含 4 帧 |
| 2×3 | 每幅 Collage 包含 6 帧 |
| 3×2 | 每幅 Collage 包含 6 帧 |

### 3. 帧率设置

- **1fps**: 每秒 1 帧
- **3fps**: 每秒 3 帧
- **30fps**: 每秒 30 帧 (原始视频)

### 4. 图片质量

- **High**: 高质量图片分析
- **Low**: 低质量图片分析

---

## 核心代码说明

### 1. 视频转帧 ([`dirVid2frames.py`](file:///c:/11/carla-temporal-collage-prompting-main/carla-temporal-collage-prompting-main/src/dirVid2frames.py))

```python
# 功能：将视频文件转换为帧图片
# 参数：
#   - frame_interval: 帧间隔（默认为 10，即每 10 帧取 1 帧）
# 输出：保存为 JPG 格式的帧图片
```

### 2. Collage 生成 ([`dir2Collages-2-3.py`](file:///c:/11/carla-temporal-collage-prompting-main/carla-temporal-collage-prompting-main/src/dir2Collages-2-3.py))

```python
# 功能：将帧图片组合成 Collage
# 布局：2行3列，共6帧
# 特点：
#   - 白色背景
#   - 帧编号标注
#   - 保持原有目录结构
```

### 3. GPT-4o 分析 ([`collage-4o-high.py`](file:///c:/11/carla-temporal-collage-prompting-main/carla-temporal-collage-prompting-main/src/collage-4o-high.py))

```python
# 功能：调用 GPT-4o 进行视觉问答
# 提示词：
#   - 识别是否有事故
#   - 分类为 Normal 或 Accident
#   - 如果是 Accident，区分 Type A 或 Type B
# 输出：JSON 格式结果
```

---

## 系统提示词

```
Your task is to first identify whether an accident occurs in the video.
You need to classify it as either "Normal" or "Accident".
If it's "Normal", you don't need to take any action.
However, if it's an "Accident", please also specify the type of accident
with the reason in detail. There are only two types of accidents:
Type A: a car crashes into people who are crossing the street.
Type B: a car crashes with another vehicle.
Let's think step-by-step
```

---

## 使用说明

### 1. 环境配置

```bash
# 安装依赖
pip install -r requirements.txt
```

### 2. 配置 API Key

在 `.env` 文件中设置：

```
OPENAI_API_KEY=your_api_key_here
```

### 3. 运行流程

```bash
# 步骤 1: 视频转帧
python src/dirVid2frames.py

# 步骤 2: 生成 Collage
python src/dir2Collages-2-3.py

# 步骤 3: 运行分析
python src/collage-4o-high.py
```

---

## 论文引用

```bibtex
@inproceedings{suntichaikul2024temporal,
  title        = {Temporal Collage Prompting: A Cost-Effective Simulator-Based Driving Accident Video Recognition With GPT-4o},
  author       = {Suntichaikul, Pratch and Taveekitworachai, Pittawat and Nukoolkit, Chakarida and Thawonmas, Ruck},
  year         = 2024,
  booktitle    = {2024 8th International Conference on Information Technology (InCIT)},
  pages        = {708--713},
  doi          = {10.1109/InCIT63192.2024.10810536}
}
```

---

## 总结

本项目成功展示了使用 Temporal Collage Prompting 方法进行驾驶事故视频识别的有效性：

✅ **整体准确率达到 85%**

✅ **行人事故识别率最高 (93%+ Recall)**

✅ **提供了经济高效的视频分析方案**

✅ **通过 Collage 方法减少了 Token 使用量**

---

*报告生成时间: 2026-05-06*

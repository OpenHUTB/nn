# 自动驾驶车辆语义分割


本项目为伍斯特理工学院（WPI）研究生课程 **RBE549 计算机视觉** 课程作业，团队以**语义分割**作为研究课题。项目基于 CARLA 自动驾驶仿真平台，通过自动化程序批量采集数千张带标注图像，构建语义分割专用数据集。

最终采用 **U-Net 网络模型**，结合**稀疏分类交叉熵焦点损失函数**开展训练，以此缓解各类别样本数量严重失衡的问题，优化数据集分布不均衡带来的模型偏差。模型将图像逐像素分为 8 类：未标注、交通标志/灯、道路、车道线、人行道、地面、车辆、行人。

## 运行效果图

输入一张 CARLA 街景图，模型输出语义分割叠加结果：

<p align="center">
<img width="500px" src="examples/result_overlay.png"/>
</p>

纯掩码（蓝=车辆，绿=地面，黄=交通标志，红=行人）：

<p align="center">
<img width="500px" src="examples/result_mask.png"/>
</p>

## 运行环境

- 平台：Windows 10/11（Linux 同理）
- Python 3.10（TensorFlow 2.10 要求 Python 3.7–3.10）
- 纯 CPU 即可运行推理，无需 GPU

推荐用 conda 建独立环境：

```bash
conda create -n py310 python=3.10 -y
conda activate py310
cd src/auto_drive_seg
pip install -r requirements.txt
```

国内网络可加清华镜像：`pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple`

## 获取预训练模型

预训练模型为二进制大文件（每个约 17MB），未随本仓库提交。请从原始项目获取：

```bash
git clone https://github.com/hlfshell/rbe549-project-segmentation
```

将 `rbe549-project-segmentation/models/unet_model_256x256_50` 整个目录复制到本模块下的 `models/` 目录，最终结构为：

```
src/auto_drive_seg/models/unet_model_256x256_50/
    saved_model.pb
    keras_metadata.pb
    variables/
```

原仓 `models/` 下另有 `unet_model_512x512_50`、`unet_model_512x512_focal_loss_with_weights` 等更大输入尺寸的模型，可按需替换。

## 运行步骤

```bash
cd src/auto_drive_seg
# 用自带示例图 + 默认 256x256 模型
python main.py

# 或指定输入图
python main.py examples/sample_input.png

# 或指定输入图和模型目录
python main.py examples/sample_input.png models/unet_model_512x512_50
```

运行后在输入图同目录生成 `*_overlay.png`（分割叠加图）和 `*_mask.png`（纯掩码图）。

## 目录说明

```
src/auto_drive_seg/
    main.py                 推理入口（模块入口，须以 main. 开头）
    requirements.txt        依赖
    semantic/               推理所需代码（U-Net 模型/数据加载/可视化/类别定义）
        unet/               模型结构、数据集、训练、可视化工具
        carla_controller/   CARLA 语义类别与颜色映射
    examples/               示例输入图与运行效果图
    models/                 预训练模型（需自行获取，见上）
```

> 说明：原始项目中 `semantic/carla_controller/` 还含有 CARLA 数据采集脚本（依赖 `carla` 包与运行中的仿真器），与推理无关，本模块未收录。训练数据集需通过 CARLA 仿真器实跑采集，详见原始项目。

---

### 专业术语对照

1. Semantic Segmentation：语义分割
2. CARLA：卡拉自动驾驶仿真平台（通用专有名词不译）
3. U-Net：U 型网络（经典分割模型，保留原名）
4. sparse categorical cross entropy focal loss：稀疏分类交叉熵焦点损失
5. class imbalance：类别不平衡
6. hyperparameter：超参数
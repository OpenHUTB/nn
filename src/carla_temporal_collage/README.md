# 项目简介
`temporal-collage-prompting`：基于 GPT-4o 与仿真场景的低成本交通事故视频识别项目。针对交通事故画面碎片化、动态时序复杂、事故特征难提取、真实事故数据稀缺等痛点，提出时序拼贴提示方案，依托仿真驾驶事故数据集，结合大模型视觉理解能力，完成交通事故事件识别与场景分析，以低成本方案提升驾驶危险视频的识别准确率与泛化能力。

# 项目核心功能
- 基于驾驶仿真环境构建多样化交通事故场景，还原碰撞、急刹、违规行驶等典型危险事件；
- 构建时序拼贴提示策略，对连续驾驶视频帧进行时序重组与特征拼接，强化动态事故表征；
- 接入 GPT-4o 多模态大模型，实现视频级事故语义理解、事件分类与风险识别；
- 完成仿真事故数据集整理、视频时序预处理、提示词工程设计与模型推理调用；
- 提供对比实验方案，支持常规识别方案与时序拼贴提示方案的效果对比、结果分析与可视化。

# 技术栈
- **模型框架**：GPT-4o 多模态大模型
- **编程语言**：Python
- **核心依赖**：OpenAI SDK、OpenCV、NumPy、视频处理工具库
- **核心算法**：时序拼贴提示、多模态大模型推理、视频时序特征建模、事故场景分类

# 项目结构
（预留结构填写位置，可自行补充文件夹划分：数据集、推理脚本、提示词模板、实验结果、工具类等）

# 运行方式
1. 配置本地 Python 运行环境并安装项目所需依赖库；
2. 配置 GPT-4o 调用密钥与接口环境；
3. 在终端执行以下命令运行项目：

```shell
python main.py
程序自动加载驾驶事故视频数据，通过时序拼贴提示生成输入指令，调用大模型完成事故识别，并输出分类结果、置信度与场景分析内容。
项目创新点
面向交通事故识别任务，创新设计时序拼贴提示方法，有效挖掘视频时序动态特征；
依托仿真驾驶数据开展实验，规避真实事故数据采集难、成本高、样本稀缺的问题；
结合 GPT-4o 多模态能力，兼顾视觉画面与时序语义理解，大幅提升复杂事故识别效果；
方案轻量化、成本可控，无需大规模深度学习模型训练，依靠提示工程即可完成任务落地；
适配不同路况、不同类型交通事故场景，可拓展用于驾驶安全检测、车载风险预警等下游任务。
论文原信息
论文标题：时序拼贴提示：基于仿真器的低成本交通事故视频识别（结合 GPT-4o）
作者：Pratch Suntichaikul, Pittawat Taveekitworachai, Chakarida Nukoolkit, Ruck Thawonmas
录用会议：2024 年第八届信息技术国际会议（InCIT 2024）
本仓库为该论文配套开源仓库，包含论文实验完整代码、处理数据与相关实验资源。
引用
若本项目与论文内容对你的研究有所帮助，欢迎引用本文：
@inproceedings{suntichaikul2024temporal,
    title        = {{Temporal Collage Prompting: A Cost-Effective Simulator-Based Driving Accident Video Recognition With GPT-4o}},
    author       = {Suntichaikul, Pratch and Taveekitworachai, Pittawat and Nukoolkit, Chakarida and Thawonmas, Ruck},
    year         = {2024},
    booktitle    = {2024 8th International Conference on Information Technology (InCIT)},
    pages        = {708--713},
    doi          = {10.1109/InCIT63192.2024.10810536}
}
开源协议
本仓库内所有代码、资源文件均遵循 MIT 开源协议 开源共享。
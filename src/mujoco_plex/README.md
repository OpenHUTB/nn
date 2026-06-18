# MuJoCo Menagerie

<p>
  <a href="https://github.com/deepmind/mujoco_menagerie/actions/workflows/build.yml?query=branch%3Amain" alt="GitHub Actions">
    <img src="https://img.shields.io/github/actions/workflow/status/deepmind/mujoco_menagerie/build.yml?branch=main">
  </a>
  <a href="https://mujoco.readthedocs.io/en/latest/models.html" alt="Documentation">
    <img src="https://readthedocs.org/projects/mujoco/badge/?version=latest">
  </a>
  <a href="https://github.com/deepmind/mujoco_menagerie/blob/main/CONTRIBUTING.md">
    <img src="https://img.shields.io/badge/PRs-welcome-green.svg" alt="PRs" height="20">
  </a>
</p>

**Menagerie** 是一个为 [MuJoCo](https://mujoco.org/) 物理引擎收集的高质量模型集合，由 DeepMind 精心策划。

物理模拟器的质量取决于它所模拟的模型，而在像 MuJoCo 这样具有众多建模选项的强大模拟器中，很容易创建出“不良”模型，这些模型的表现往往不符合预期。本集合的目标是为社区提供一个精选的、设计精良的模型库，这些模型开箱即用，表现良好。

- [入门指南](#入门指南)
    - [前置条件](#前置条件)
    - [结构概览](#结构概览)
    - [安装与使用](#安装与使用)
- [模型质量与贡献](#模型质量与贡献)
- [Menagerie 模型列表](#menagerie-模型列表)
- [引用 Menagerie](#引用-menagerie)
- [致谢](#致谢)
- [许可与免责声明](#许可与免责声明)

## 入门指南

### 前置条件

Menagerie 的唯一依赖是 MuJoCo 2.2.2 或更高版本。你可以从 GitHub [发布页面](https://github.com/deepmind/mujoco/releases/) 下载预编译二进制文件，或者如果你使用 Python，可以通过 `pip install mujoco>=2.2.2` 从 [PyPI](https://pypi.org/project/mujoco/) 安装原生绑定。其他安装方式请参见[这里](https://github.com/deepmind/mujoco#installation)。

### 结构概览

Menagerie 的目录结构如下图所示。为简洁起见，我们只展示了一个模型目录，因为其他所有模型目录都遵循完全相同的模式。

```bash
├── agility_cassie
│   ├── assets
│   │   ├── achilles-rod.obj
│   │   ├── ...
│   ├── cassie.png
│   ├── cassie.xml
│   ├── LICENSE
│   ├── README.md
│   └── scene.xml
```

- `assets`：存储模型用于视觉和碰撞的 3D 网格文件（.stl 或 .obj）
- `LICENSE`：描述模型的版权和许可条款
- `README.md`：包含描述 MJCF XML 文件生成过程的详细步骤
- `<model>.xml`：包含模型的 MJCF 定义
- `scene.xml`：包含 `<model>.xml`，并添加了一个平面、一个光源以及可能的其他物体
- `<model>.png`：`scene.xml` 的 PNG 图像

注意，`<model>.xml` 仅描述模型本身，即运动链中不定义其他实体。我们将额外的 body 定义留给了 `scene.xml` 文件，如 Shadow Hand 的 [`scene.xml`](shadow_hand/scene_right.xml) 所示。

### 安装与使用

要安装 Menagerie，只需在你选择的目录中克隆仓库：

```bash
git clone https://github.com/deepmind/mujoco_menagerie.git
```

以交互方式探索模型的最简单方法是将其加载到每个 MuJoCo 发行版附带的 [simulate](https://github.com/deepmind/mujoco/tree/main/simulate) 二进制程序中。这只需将 `scene.xml` 文件拖放到 simulate 窗口中即可。如果你愿意，也可以使用命令行启动 `simulate` 并直接传入 XML 的路径。

在交互式模拟之外，你可以像加载任何其他 XML 文件一样在 MuJoCo 中加载模型，通过 C/C++ API：

```c++
#include <mujoco.h>

mjModel* model = mj_loadXML("unitree_a1/a1.xml", nullptr, nullptr, 0);
mjData* data = mj_makeData(model);
mj_step(model, data);
```

或通过 Python：

```python
import mujoco

model = mujoco.MjModel.from_xml_path("unitree_a1/a1.xml")
data = mujoco.MjData(model)
mujoco.mj_step(model, data)
```

如果你有进一步的问题，请查阅我们的 [FAQ](FAQ.md)。

## 模型质量与贡献

我们的目标是最终使所有 Menagerie 模型尽可能忠实地反映其建模的真实系统。提高模型质量是一项持续的工作，目前许多模型的状态不一定达到理想水平。

然而，通过发布 Menagerie 的当前状态，我们希望整合并提高社区贡献的可见性。为了帮助 Menagerie 用户对每个模型的质量设定合理的预期，我们引入了以下分级系统：

| 等级 | 描述                       |
| ---- | -------------------------- |
| A+   | 数值经过适当的系统辨识     |
| A    | 数值真实，但未经适当的辨识 |
| B    | 稳定，但部分数值不真实     |
| C    | 条件稳定，有显著改进空间   |

有关贡献的更多信息，例如向 Menagerie 添加新模型，请参阅 [CONTRIBUTING](CONTRIBUTING.md)。

## Menagerie 模型列表

| 机器人                                   | 预览                                                         | 等级 |
| ---------------------------------------- | ------------------------------------------------------------ | :--: |
| [Shadow E3M5](shadow_hand/README.md)     | [<img src="shadow_hand/shadow_hand.png" width="400">](shadow_hand/README.md) |  A   |
| [Robotiq 2F-85](robotiq_2f85/README.md)  | [<img src="robotiq_2f85/2f85.png" width="400">](robotiq_2f85/README.md) |  B   |
| [Cassie](agility_cassie/README.md)       | [<img src="agility_cassie/cassie.png" width="400">](agility_cassie/README.md) |  C   |
| [ANYmal B](anybotics_anymal_b/README.md) | [<img src="anybotics_anymal_b/anymal_b.png" width="400">](anybotics_anymal_b/README.md) |  A   |
| [ANYmal C](anybotics_anymal_c/README.md) | [<img src="anybotics_anymal_c/anymal_c.png" width="400">](anybotics_anymal_c/README.md) |  B   |
| [Unitree A1](unitree_a1/README.md)       | [<img src="unitree_a1/a1.png" width="400">](unitree_a1/README.md) |  B   |
| [Panda](franka_emika_panda/README.md)    | [<img src="franka_emika_panda/panda.png" width="400">](franka_emika_panda/README.md) |  B   |
| [UR5e](universal_robots_ur5e/README.md)  | [<img src="universal_robots_ur5e/ur5e.png" width="400">](universal_robots_ur5e/README.md) |  B   |

相应的嵌入式视频请参见 MuJoCo [文档](https://mujoco.readthedocs.io/en/latest/models.html)。

## 引用 Menagerie

如果你在工作中使用了 Menagerie，请使用以下引用格式：

```bibtex
@software{menagerie2022github,
  author = {MuJoCo Menagerie Contributors},
  title = {{MuJoCo Menagerie: A collection of high-quality simulation models for MuJoCo}},
  url = {http://github.com/deepmind/mujoco_menagerie},
  year = {2022},
}
```

## 致谢

本仓库中的模型基于许多才华横溢的人设计的三方模型，如果没有他们慷慨的开源贡献，这一切将不可能实现。我们要感谢所有为 MuJoCo Menagerie 做出贡献的设计师和工程师。

感谢 Pedro Vergani 在视觉和设计方面的帮助。

使本仓库公开的主要工作由 [Kevin Zakka](https://kzakka.com/) 承担，并得到了 DeepMind 机器人仿真团队的支持。

## 许可与免责声明

本仓库每个模型目录下的 XML 和资源文件适用不同的许可条款。请查阅每个具体模型子目录下的 `LICENSE` 文件以了解相关的许可和版权信息。

所有其他内容版权归 DeepMind Technologies Limited 2022 所有，并遵循 Apache License, Version 2.0 许可。本仓库根目录下的 LICENSE 文件提供了该许可的副本。你也可以从 https://www.apache.org/licenses/LICENSE-2.0 获取。

这不是 Google 官方支持的产品。


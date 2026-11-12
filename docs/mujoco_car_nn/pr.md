# 提交与评审

课程要求"提交次数和审核次数分别不少于 10 次"，并且"Pull Request 标题不能随意，需要具体说明修改了什么"。
本章给出可直接照做的提交计划、PR 模板与评审清单。

## 1. 分支与提交计划

从 `main` 切出功能分支 `feature/mujoco-car-nn`，按"一个关注点一次提交"的原则组织 12 次提交：

| # | 提交信息 | 内容 |
|---|---|---|
| 1 | `chore: 初始化 mujoco_car_nn 模块骨架` | 目录、`__init__`、requirements、main.py |
| 2 | `feat(track): 参数化闭环赛道与护栏生成` | `track.py`，227 m 赛道、曲率与墙体 |
| 3 | `feat(sim): MuJoCo 平面刚体车辆与相机/雷达` | `sim.py`，动力学、传感器、离屏渲染 |
| 4 | `feat(expert): pure-pursuit 专家与速度规划` | `expert.py` |
| 5 | `feat(data): 专家数据采集与 Dataset` | `dataset.py`，4137 帧 npz |
| 6 | `feat(model): 端到端 CNN 与 MLP 基线` | `models.py` |
| 7 | `feat(train): 行为克隆与 DAgger 训练流程` | `train.py` |
| 8 | `feat(eval): 闭环评测指标与多场景测试` | `evaluate.py`、`benchmark` 子命令 |
| 9 | `feat(slam): 占据栅格建图与扫描匹配` | `slam.py`，ATE / IoU |
| 10 | `feat(viz): HUD、演示 GIF 与地图渲染` | `viz.py`、`demo`/`slam` 子命令 |
| 11 | `feat(ros): Noetic 与 Humble 节点及 launch` | `ros1/`、`ros2/` |
| 12 | `docs: mkdocs 文档、实验数据与动图` | `docs/`、`mkdocs.yml`、`scripts/` |

提交信息遵循 Conventional Commits，正文用中文说明"改了什么、为什么、怎么验证"：

```text
feat(slam): 占据栅格建图与相关法扫描匹配

- 实现 log-odds 栅格与射线投射，单帧更新 < 1 ms
- 相关法扫描匹配采用粗→精两级搜索（111 候选/帧）
- 仅对横向分量做修正，避免沿赛道方向的累计漂移
- 单圈 ATE 由 2.98 m 降至 2.37 m，IoU 由 0.09 提升至 0.13

验证：python main.py slam --laps 1
```

## 2. PR 模板

```markdown
## 做了什么
新增 `mujoco_car_nn` 模块：MuJoCo 无人车仿真 + 端到端 CNN 驾驶 + 栅格 SLAM，
提供 Noetic / Humble 两套 launch。

## 为什么
对应课程任务（1）～（4）：物理仿真、传感器闭环、建图导航、端到端模型。

## 怎么验证
```bash
pip install -r requirements.txt
python main.py collect --laps 5 --scenes day,dusk
python main.py train --model cnn --epochs 24 --dagger
python main.py eval --model runs/cnn_dagger.pt --laps 2 --record
python -m pytest tests -q
```

## 结果
| 策略 | 单圈(s) | 横向 RMSE(m) | 碰撞 |
|---|---|---|---|
| 专家 | 22.1 | 0.49 | 0 |
| CNN+DAgger | 22.4 | 0.47 | 0 |

## 截图 / 动图
（附 `docs/gifs/nn_drive.gif`）

## 影响面
仅新增文件，不修改既有模块；`main.py` 为唯一入口。
```

PR 标题要具体，例如：

- ✅ `feat(mujoco_car_nn): 新增 MuJoCo 无人车端到端驾驶与栅格 SLAM 模块`
- ❌ `update code`

## 3. 评审清单（Reviewer 用）

把仓库克隆到**另一台机器**（或课程虚拟机）后逐项检查：

- [ ] `pip install -r requirements.txt` 成功，无 GPU 也能跑
- [ ] `python main.py collect --laps 2` 能在 30 s 内产出 `data/dataset.npz`
- [ ] `python main.py train --model cnn --epochs 10` 损失正常下降
- [ ] `python main.py eval --expert --laps 1` 专家能 0 碰撞跑完一圈
- [ ] `python main.py eval --model runs/cnn_dagger.pt --laps 1` 同样 0 碰撞
- [ ] `python -m pytest tests -q` 全绿
- [ ] `roslaunch car_nn_sim car_teleop.launch` 能键盘遥控（Noetic）
- [ ] `roslaunch car_nn_sim car_slam.launch` 的 `/map` 能逐渐展开
- [ ] `docs/` 下 GIF < 5 MB，且能正常播放
- [ ] `mkdocs build` 无报错，导航完整

## 4. 常见 review 意见与处理

| 意见 | 处理 |
|---|---|
| "端到端为什么还要输入车速？" | 单帧图像无法区分静止/巡航，见[算法](algorithm.md)消融；纯图像版本在起跑线趴窝 |
| "扫描匹配为什么只修横向？" | 沿赛道方向相关峰平坦，全量更新会累计漂移，见[SLAM](slam.md) |
| "为什么不用 Gazebo/Carla？" | 课程要求"能在 Noetic/Humble 跑通"，MuJoCo 纯 pip 安装、无 GPU 依赖，复现成本最低；接口层已抽象，替换仿真器只需改 `sim.py` |
| "数据/权重不应该进仓库" | 已 gitignore，`collect`/`train` 两条命令可完全重建 |
| "评测是否过拟合单圈？" | 起点随机 ±1.2 m、三种光照、多随机种子，见[实验](experiments.md) |

## 5. 复现与提交检查

```bash
# 1. 干净环境
python -m venv .venv && . .venv/bin/activate      # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. 全流程
python main.py collect --laps 5 --scenes day,dusk
python main.py train --model cnn --epochs 20 --dagger
python main.py benchmark --epochs 20 --laps 1
python main.py demo  --model runs/cnn_dagger.pt --seconds 12
python main.py slam  --laps 1

# 3. 文档
pip install mkdocs mkdocs-material
mkdocs build          # 输出 site/，可直接发布到 GitHub Pages
```

推送前确认：

```bash
git status            # 不应包含 data/ runs/ site/
git log --oneline     # 至少 10 条有意义的提交
```

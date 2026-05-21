# MuJoCo 小车倒立摆平衡恢复分析模块

本项目使用官方 `mujoco` Python 包加载小车倒立摆 MJCF 模型，执行仿真 step 并导出小车位置、摆杆角度、速度和控制力，用于分析平衡恢复过程。

## 主要内容

- 使用 MuJoCo MJCF 模型 `mujoco_cartpole.xml`。
- 运行 `generate_cartpole_data.py` 调用 MuJoCo 生成仿真 rollout。
- 导出 `mujoco_cartpole_rollout.csv`，包含 qpos、qvel 和 control force。
- 统计初始角度、最终角度、最大控制力、控制努力和稳定时间。
- 生成倒立摆恢复曲线和控制力曲线。

## 运行

```bash
python src/mujoco_cartpole_recovery/generate_cartpole_data.py
python src/mujoco_cartpole_recovery/cartpole_recovery.py --output docs/pr_assets/mujoco_cartpole_recovery
python src/mujoco_cartpole_recovery/tests/test_cartpole_recovery.py
```

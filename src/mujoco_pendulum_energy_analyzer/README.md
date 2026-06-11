# MuJoCo 双连杆摆控制能耗与稳定性分析模块

本项目使用官方 `mujoco` Python 包加载 MJCF 双连杆摆模型，执行仿真 step 并导出关节角、角速度、控制量和瞬时功率，用于分析控制能耗与收敛稳定性。

## 主要内容

- 使用 MuJoCo MJCF 模型 `mujoco_double_pendulum.xml`。
- 运行 `generate_pendulum_data.py` 调用 MuJoCo 生成仿真 rollout。
- 导出 `mujoco_pendulum_rollout.csv`，包含 qpos、qvel、ctrl 和 instant_power。
- 统计总能耗、最大功率、最终角度误差和稳定时间。
- 生成关节角收敛曲线和执行器功率曲线。

## 运行

```bash
python src/mujoco_pendulum_energy_analyzer/generate_pendulum_data.py
python src/mujoco_pendulum_energy_analyzer/pendulum_energy.py --output docs/pr_assets/mujoco_pendulum_energy_analyzer
python src/mujoco_pendulum_energy_analyzer/tests/test_pendulum_energy.py
```

<!-- 感谢提交 pull request! -->
<!-- ⚠️⚠️ 不要删除该文件！这是 Pull Request 的模板 ⚠️⚠️ -->
<!-- 请阅读我们的贡献指南：https://github.com/OpenHUTB/.github/blob/master/CONTRIBUTING.md -->
修改概述: 修复 CarRacing DQN/DoubleDQN 训练中终止语义（terminated/truncated）处理不一致导致的跨 episode bootstrap 问题，并补充可控开关与训练脚本参数化支持。

## 修改的详细描述
1. 回放缓冲区同时记录 `terminated` 与 `truncated`，采样时根据 `treat_truncated_as_terminal` 计算 TD target 的 done mask，避免将不同类型结束混为一谈（见 `dqn_model/base_agent.py`）。
2. DQN/DoubleDQN 训练脚本写入回放时传入 `(terminated, truncated)`，与 Gymnasium 的返回语义一一对应（见 `dqn_model/training_dqn.py`、`doubledqn_model/training_double_dqn.py`）。
3. 新增配置项 `treat_truncated_as_terminal`（默认 true，见 `configs/dqn.yaml`），并在训练脚本增加 CLI 参数 `--treat-truncated-as-terminal` 以便 A/B 对比是否对 time-limit 截断做 bootstrap。
4. 将更新逻辑中的 `terminateds` 更名为 `dones`（表示用于 TD target 的 done mask），提升可读性与减少误用（见 `dqn_model/dqn_agent.py`、`doubledqn_model/doubledqn_agent.py`）。

## 经过了什么样的测试?
1. 操作系统：Linux
2. Python 版本：3.11.14
3. 基础校验：`python -m py_compile` 对以下文件通过
   - `src/car_racing/car_racing_ros/dqn_model/base_agent.py`
   - `src/car_racing/car_racing_ros/dqn_model/dqn_agent.py`
   - `src/car_racing/car_racing_ros/dqn_model/training_dqn.py`
   - `src/car_racing/car_racing_ros/doubledqn_model/doubledqn_agent.py`
   - `src/car_racing/car_racing_ros/doubledqn_model/training_double_dqn.py`

## 运行效果
该改动主要提升训练目标语义正确性与对比可复现性。建议在 GPU 环境固定 `--max-timesteps` 做长跑对比，并附上 reward/loss 曲线图（以及终端输出的 Mean(10)/SPS/UPS 片段截图）。

示例命令（GPU 机器）：
```bash
export MPLCONFIGDIR=/tmp/matplotlib

# truncated 视为 terminal（严格终止，不跨 episode bootstrap）
python /home/zhibing/nn/nn/src/car_racing/car_racing_ros/dqn_model/training_dqn.py \
  --episodes 200 --max-timesteps 200000 --batch-size 64 \
  --report none --log-every 1 --log-filename DQN_trunc_terminal.csv \
  --skip-eval --seed 123 \
  --dueling 1 --normalize-obs 1 --amp 1 --double-q 1 \
  --treat-truncated-as-terminal 1

# truncated 不视为 terminal（允许 bootstrap）
python /home/zhibing/nn/nn/src/car_racing/car_racing_ros/dqn_model/training_dqn.py \
  --episodes 200 --max-timesteps 200000 --batch-size 64 \
  --report none --log-every 1 --log-filename DQN_trunc_bootstrap.csv \
  --skip-eval --seed 123 \
  --dueling 1 --normalize-obs 1 --amp 1 --double-q 1 \
  --treat-truncated-as-terminal 0
```

曲线对比（生成图片后附在 PR）：
```bash
python /home/zhibing/nn/nn/src/car_racing/car_racing_ros/plot_comparison.py \
  --logs /home/zhibing/nn/nn/src/car_racing/car_racing_ros/training/logs/DQN_trunc_terminal.csv /home/zhibing/nn/nn/src/car_racing/car_racing_ros/training/logs/DQN_trunc_bootstrap.csv \
  --labels trunc_terminal trunc_bootstrap \
  --metric reward --smooth 20 \
  --out /home/zhibing/nn/nn/src/car_racing/car_racing_ros/training/dqn_reward_trunc_cmp.png \
  --title "DQN Reward (smooth=20)"

python /home/zhibing/nn/nn/src/car_racing/car_racing_ros/plot_comparison.py \
  --logs /home/zhibing/nn/nn/src/car_racing/car_racing_ros/training/logs/DQN_trunc_terminal.csv /home/zhibing/nn/nn/src/car_racing/car_racing_ros/training/logs/DQN_trunc_bootstrap.csv \
  --labels trunc_terminal trunc_bootstrap \
  --metric loss --smooth 20 \
  --out /home/zhibing/nn/nn/src/car_racing/car_racing_ros/training/dqn_loss_trunc_cmp.png \
  --title "DQN Loss (smooth=20)"
```

生成文件（示例）：
- 日志：`src/car_racing/car_racing_ros/training/logs/DQN_trunc_terminal.csv`、`DQN_trunc_bootstrap.csv`
- 图片：`src/car_racing/car_racing_ros/training/dqn_reward_trunc_cmp.png`、`dqn_loss_trunc_cmp.png`

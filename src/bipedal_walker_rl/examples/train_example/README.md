# Train Example

快速训练示例，演示如何使用项目中的 `env_utils.make_env` 与 `stable_baselines3.PPO` 进行简短训练并保存模型。

用法：

```bash
python train.py
```

说明：
- 该示例用于演示与快速验证，默认训练 `1000` timesteps，适合 CI 或本地快速检查。
- 训练后的模型保存在 `src/bipedal_walker_rl/models/ppo_bipedalwalker_example.zip`。

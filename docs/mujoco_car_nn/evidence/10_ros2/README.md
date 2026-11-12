# ROS 2 证据（BLOCKED）

本机未安装 WSL2 / Docker / ROS 2，无法执行 `colcon build` 与 `ros2 launch`，因此没有截图。

仓库已提供 `ros2/car_nn_sim`（ament_python 包、3 个 launch.py）与 `Dockerfile`，
具备 Linux 环境后可直接：

```bash
colcon build --packages-select car_nn_sim
ros2 launch car_nn_sim car_nn.launch.py policy:=/abs/path/runs/cnn_dagger.pt
```

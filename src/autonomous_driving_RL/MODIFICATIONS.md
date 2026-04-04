# autonomous_driving_RL 包修改说明

本文档记录对 `src/autonomous_driving_RL` 内相关文件的修改：**问题描述**、**造成的影响**、**修改原因**。

---

## 1. 训练/评估脚本错误导入 `carla_env` 子包

| 字段 | 内容 |
|------|------|
| **问题描述** | `train_agent.py`、`eval_agent.py` 使用 `from carla_env.carla_env_multi_obs import CarlaEnvMultiObs`，但仓库内**不存在** `carla_env/` 目录，仅存在同级模块 `carla_env_multi_obs.py`。 |
| **造成的影响** | 在包根目录执行 `python train_agent.py` / `python eval_agent.py` 时触发 `ModuleNotFoundError`，训练与评估无法启动。 |
| **修改原因** | 将导入改为 `from carla_env_multi_obs import CarlaEnvMultiObs`，与当前目录结构一致；运行时应将工作目录设为 `autonomous_driving_RL`（或把该目录加入 `PYTHONPATH`）。 |

**涉及文件**：`train_agent.py`、`eval_agent.py`；另将 `home/wu/catkin_ws/.../scripts/train_agent.py` 中相同错误导入与无效参数一并修正（见第 8 节）。

---

## 8. Catkin `scripts/train_agent.py` 错误导入与不存在的构造参数

| 字段 | 内容 |
|------|------|
| **问题描述** | 该脚本使用 `from carla_env.carla_env_multi_obs import CarlaEnvMultiObs`，并向 `CarlaEnvMultiObs` 传入 `debug=False`，但环境类**无** `debug` 参数。 |
| **造成的影响** | 无法导入模块或运行即报 `TypeError: __init__() got an unexpected keyword argument 'debug'`。 |
| **修改原因** | 改为 `from carla_env_multi_obs import CarlaEnvMultiObs`，并删除 `debug` 参数，与根目录环境类定义一致。 |

**涉及文件**：`home/wu/catkin_ws/src/carla_rl_ros/scripts/train_agent.py`

---

## 2. 使用 `vehicle.is_alive` 判断 Actor 是否存活不可靠

| 字段 | 内容 |
|------|------|
| **问题描述** | CARLA Python API 中 `is_alive` 多为**方法** `is_alive()`；写成 `if not self.vehicle.is_alive` 时，得到的是**可调用对象**，在布尔上下文中几乎恒为真。 |
| **造成的影响** | 车辆已销毁后仍可能被当作存活，继续访问位置/速度导致异常，或无法及时结束 episode、错误计算奖励。 |
| **修改原因** | 新增 `_actor_alive(actor)`：对 `callable` 的 `is_alive` 执行调用，否则按属性布尔值处理，兼容不同 CARLA 版本绑定。 |

**涉及文件**：`carla_env_multi_obs.py`，并与 `home/wu/catkin_ws/.../scripts/carla_env_multi_obs.py` 同步。

---

## 3. 裸 `except:` 捕获所有异常（含系统退出类）

| 字段 | 内容 |
|------|------|
| **问题描述** | `_get_lane_offset`、`get_vehicle_transform` 等处使用 `except:`，等价于捕获 `BaseException`。 |
| **造成的影响** | 可能吞掉 `KeyboardInterrupt` / `SystemExit`，调试与优雅退出困难；不符合常见 Python 规范（如 PEP 8）。 |
| **修改原因** | 改为 `except Exception:`，仅处理一般运行时异常，保留对中断与退出的正常行为。 |

**涉及文件**：`carla_env_multi_obs.py`（及 catkin 下同名脚本副本）。

---

## 4. CARLA 地址与端口写死，不利于多机与容器部署

| 字段 | 内容 |
|------|------|
| **问题描述** | `_connect_carla` 中硬编码 `carla.Client('localhost', 2000)`。 |
| **造成的影响** | CARLA 在远程主机或自定义端口时，必须改源码才能连接，易误提交本地配置。 |
| **修改原因** | 为 `CarlaEnvMultiObs.__init__` 增加可选参数 `carla_host`、`carla_port`；若为 `None`，则从环境变量 `CARLA_HOST`、`CARLA_PORT` 读取（默认 `localhost` / `2000`），与 `autonomous_driving_car` 包习惯一致。 |

**涉及文件**：`carla_env_multi_obs.py`（及 catkin 脚本副本）。

---

## 5. 轨迹与车辆 ID 文件的文本编码未显式指定

| 字段 | 内容 |
|------|------|
| **问题描述** | 读写 `.last_vehicle_id.json`、`trajectory.csv` 时未指定 `encoding`。 |
| **造成的影响** | 在 Windows 默认非 UTF-8 区域设置下，路径或注释含非 ASCII 时可能出现编码错误或乱码。 |
| **修改原因** | 文本读写统一使用 `encoding='utf-8'`；轨迹 CSV 另加 `newline=''` 以符合 CSV 惯例。 |

**涉及文件**：`carla_env_multi_obs.py`（根目录与 catkin 副本；根目录与副本均已对齐）。

---

## 6. `eval_agent.parse_targets` 对用户输入校验不足

| 字段 | 内容 |
|------|------|
| **问题描述** | `--targets` 字符串按 `;` 分割后直接 `split(",")`，缺少去空段与段内格式校验。 |
| **造成的影响** | 多余分号、错误段格式会导致 `ValueError`（解包失败）或坐标静默错误，排障困难。 |
| **修改原因** | 对每段 `strip()`、跳过空段；要求严格为 `x,y` 两段，否则抛出带上下文的 `ValueError`。 |

**涉及文件**：`eval_agent.py`、`home/wu/catkin_ws/.../scripts/eval_agent.py`

---

## 7. ROS 包装节点中 `__init__` 补丁破坏连接状态

| 字段 | 内容 |
|------|------|
| **问题描述** | `ros_train_node.py`、`ros_eval_node.py` 先在补丁里创建 `Client` 并设置 `self.client` / `self.world`，再调用原始 `CarlaEnvMultiObs.__init__`，而原始 `__init__` 会将 `self.client = None` 等**重新初始化**，覆盖已建立的连接。 |
| **造成的影响** | ROS 下训练/评估看似连上 CARLA，随后 `reset()` 仍按默认 localhost 重连，或逻辑与参数不一致，表现为随机连接失败或忽略 `~carla_host`。 |
| **修改原因** | 补丁改为仅通过 `kwargs.setdefault('carla_host', ...)` / `carla_port` 注入 ROS 参数，再调用原始 `__init__(*args, **kwargs)`，由环境类统一在 `reset()` 时 `_connect_carla()`；删除冗余的 `import time` / `import carla`（eval 侧未再使用）。 |

**涉及文件**：`ros_train_node.py`、`ros_eval_node.py`

---

## 修改总结（简要）

- **可运行性**：修正根目录与 catkin `scripts` 下 `train_agent` / `eval_agent` 的模块导入，避免因缺少 `carla_env` 包而无法启动；去掉无效的 `debug` 构造参数。  
- **正确性**：用 `_actor_alive` 正确判断车辆存活；修复 ROS 节点中对 `CarlaEnvMultiObs.__init__` 的补丁逻辑，使 `~carla_host` / `~carla_port` 真正写入环境并在 `reset()` 中生效。  
- **健壮性**：裸 `except` 收窄为 `Exception`；`--targets` 解析更严格；JSON/CSV 读写显式 UTF-8。  
- **可配置性**：CARLA 主机与端口支持构造参数与环境变量 `CARLA_HOST` / `CARLA_PORT`，便于远程仿真与容器部署。  
- **副本同步**：`carla_env_multi_obs.py` 在包根目录与 `home/wu/catkin_ws/.../scripts/` 下内容已对齐，避免 ROS 与本地脚本行为分叉。  

在 `autonomous_driving_RL` 目录下运行示例：

```bash
# 可选
export CARLA_HOST=127.0.0.1
export CARLA_PORT=2000
python train_agent.py --timesteps 10000
python eval_agent.py --model_path ./checkpoints/best_model.zip
```

ROS 侧仍通过私有参数 `~carla_host`、`~carla_port` 配置（由补丁写入环境构造参数）。

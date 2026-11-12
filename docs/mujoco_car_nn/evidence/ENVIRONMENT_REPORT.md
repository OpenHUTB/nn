# 环境检查报告（入口）

完整版见 [`01_environment/ENVIRONMENT_REPORT.md`](01_environment/ENVIRONMENT_REPORT.md)。

## 摘要

| 项 | 状态 |
|---|---|
| Windows 11 家庭版 10.0.26200（64 位） | ✅ |
| AMD Ryzen 9 8945HX / 32 GB / RTX 5060 | ✅ |
| 管理员权限 | ❌（全程未提权，安装时由本人 UAC 授权） |
| WSL2 | ❌ 未安装（2026-11-10 尝试安装，停在下载阶段） |
| Ubuntu 22.04 | ❌ 无 |
| Docker | ❌ 未安装 |
| Python（Harness 运行时） | ✅ 3.12.14 |
| Git（MinGit） | ✅ 2.56.0 |
| MuJoCo / PyTorch / pytest / mkdocs | ✅ 4.0.0 / 2.15.0+cpu / 9.2.0 / 1.5.9 |
| ROS 1 / ROS 2 | ❌ 未安装（BLOCKED） |
| 桌面截图 / 键鼠自动化 | ✅ 已实测 |

## 相关文件

- `01_environment/01_windows_env.png` —— 真实控制台截图
- `01_environment/env_check.log` —— 原始命令输出
- `02_ros_install/STATUS_2026-11-10.md` —— WSL 安装尝试记录

# 环境检查报告

> 生成时间：2026-11-10 21:52（Asia/Shanghai）
> 采集方式：本机 PowerShell 命令 + 真实桌面窗口截图（`01_windows_env.png`）
> 说明：本报告只记录**实际执行**得到的输出，未安装/未验证的项目明确标注。

## 1. 硬件与操作系统

| 项 | 值 |
|---|---|
| 主机 | LENOVO 83LV |
| 操作系统 | Microsoft Windows 11 家庭版 中文版 |
| 版本 | 10.0.26200（Build 26200） |
| 架构 | 64 位 |
| 内存 | 34,142,187,520 B ≈ 31.8 GB |
| CPU | AMD Ryzen 9 8945HX（16 核 / 32 线程） |
| GPU | NVIDIA GeForce RTX 5060 Laptop GPU，驱动 32.0.15.8205 |

## 2. 权限

| 项 | 结果 |
|---|---|
| 当前进程是否管理员 | **False**（未提权，未执行任何需要管理员的安装） |
| 桌面会话 | 有（屏幕截图成功，1707×1067，非黑屏） |
| 窗口/键盘自动化 | 可用（EnumWindows + SendInput + WriteConsoleInput 实测成功） |

## 3. Linux 与容器

| 项 | 结果 | 说明 |
|---|---|---|
| WSL | **未安装** | `wsl --status` 返回"未安装适用于 Linux 的 Windows 子系统" |
| WSL 发行版 | 无 | `wsl --list --verbose` 无输出 |
| Docker | **未安装** | `docker` 命令不存在 |
| Docker Daemon | 不可用 | 无 docker CLI，无法检查守护进程 |

安装 WSL2 需要管理员权限并重启系统，属于任务书第十阶段明确要求"先取得同意"的操作，
因此本轮**没有执行** `wsl --install`。

## 4. 工具链

| 工具 | 版本 / 路径 | 状态 |
|---|---|---|
| Python | 3.12.14（DeepSeek Harness 运行时 `C:\Users\zhoul\.dsh\dsh-runtimes\...\python.exe`） | 可用 |
| 系统 `python` | `C:\Users\zhoul\AppData\Local\Microsoft\WindowsApps\python.exe`（应用商店占位符） | **不可直接使用** |
| Git | 2.56.0.windows.4（`C:\Users\zhoul\.tools\git\cmd\git.exe`，本轮为完成任务下载的 MinGit） | 可用 |
| Node.js / npm | 未安装 | — |
| cmake / gcc | 未检查（Windows 原生不参与构建） | — |
| MuJoCo | 4.0.0（pip） | 可用，离屏渲染正常 |
| PyTorch | 2.15.0+cpu | 可用 |
| matplotlib | 3.11.2 | 可用 |
| mkdocs | 1.5.9（`hutb-doc` 提供的分支） | 可用，站点 strict 构建通过 |
| pytest | 9.2.0 | 可用，6 项测试通过 |
| pandoc | 3.9（pypandoc-binary） | 可用，Word 报告生成成功 |
| ROS 1 / ROS 2 | **未安装** | 无 `roscore`、无 `ros2` |

## 5. Harness 能力检查

| 能力 | 结果 | 证据 |
|---|---|---|
| 终端命令执行 | 可用 | 全部命令输出 |
| 文件读写/修改 | 可用 | 仓库 84 个受控文件 |
| 长时间任务 | 可用 | 后台训练/评测任务 |
| 桌面截图 | 可用 | `01_windows_env.png` 等 |
| 窗口枚举 / 置前 / 移动 | 可用 | `scripts/win_gui.ps1` |
| 键盘事件注入 | 可用 | `WriteConsoleInputW` 实测，msvcrt 能读到 |
| 鼠标点击 | 可用 | `Click-At` 实测 |
| 浏览器截图 | 可用 | headless Edge 截取 mkdocs 站点 |

## 6. 结论

- 本机具备**完整的 Windows 原生开发/训练/仿真/截图**能力，项目在该环境下已真实运行并通过测试。
- **不具备** Linux/ROS 运行环境：WSL2 未安装且需要管理员权限 + 重启，Docker 未安装。
  因此任务书中的"ROS 1/ROS 2 真实运行"在本轮为 **BLOCKED**，而不是代码问题。
- 若需要继续完成 ROS 实机验证，最小代价路径是：安装 WSL2 + Ubuntu 22.04（Humble），
  或 Ubuntu 20.04（Noetic），再按 `docs/ros.md` 执行 `catkin_make` / `colcon build` 与 `roslaunch`。

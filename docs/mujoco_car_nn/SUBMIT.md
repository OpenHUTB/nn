# 提交与评审操作指南

> 目标：把本仓库的模块提交到课程指定的 GitHub 仓库，拿到 PR 链接，并完成 ≥10 次评审。

## 0. 先确认一件事

课程作业要求提交到 **OpenHUTB/nn**（`src/` 下按模块建目录），但请先核对老师通知里的
**仓库地址和目录规范**。下面以 OpenHUTB/nn 为例；如果老师给的是别的仓库，把地址替换掉即可。

## 1. Fork 仓库

1. 浏览器打开目标仓库（例如 `https://github.com/OpenHUTB/nn`）；
2. 右上角 **Fork** → 创建到你自己的账号下；
3. 得到 `https://github.com/<你的用户名>/nn`。

## 2. 克隆到本地

没有 git 命令行也能做（用 GitHub Desktop），有 git 的话：

```powershell
# 本机 git 在 C:\Users\zhoul\.tools\git\cmd\git.exe
cd C:\Users\zhoul\Desktop
git clone https://github.com/<你的用户名>/nn.git
cd nn
git checkout -b feature/mujoco-car-nn
```

## 3. 把模块复制进去

从 `C:\Users\zhoul\OneDrive\Desktop\ROS` 复制这些内容到 `nn/`（保持相对路径）：

```text
src/mujoco_car_nn/          # 核心 Python 包
ros1/car_nn_sim/            # ROS 1 包
ros2/car_nn_sim/            # ROS 2 包
docs/                       # mkdocs 文档 + figures + gifs
tests/                      # 6 项测试
scripts/                    # 图表/消融/演示脚本
main.py  main.bat  main.sh
requirements.txt  mkdocs.yml  Dockerfile  README.md
runs/cnn_dagger.pt          # 演示权重（1.4 MB）
```

PowerShell 一条命令（在 `nn/` 目录执行）：

```powershell
$src = "C:\Users\zhoul\OneDrive\Desktop\ROS"
foreach ($p in @("src","ros1","ros2","docs","tests","scripts","runs")) {
  Copy-Item "$src\$p" -Destination . -Recurse -Force
}
Copy-Item "$src\main.py","$src\main.bat","$src\main.sh","$src\requirements.txt","$src\mkdocs.yml","$src\Dockerfile","$src\README.md" .
```

> `data/` 和 `site/` 不要复制（可由命令重新生成）。

## 4. 提交并推送

```powershell
git add .
git commit -m "feat(mujoco_car_nn): 新增 MuJoCo 无人车端到端驾驶与栅格 SLAM 模块"
git push -u origin feature/mujoco-car-nn
```

想凑够"提交次数 ≥10"，可以按功能拆成多次提交（物理仿真 / 专家 / 数据 / 模型 / 训练 /
评测 / SLAM / ROS / 文档），仓库里已有 25 次提交的历史可以直接参考拆分方式。

## 5. 发起 PR

1. 打开你的 fork，页面会提示 **Compare & pull request**；
2. 标题：

   ```text
   feat(mujoco_car_nn): 新增 MuJoCo 无人车端到端驾驶与栅格 SLAM 模块
   ```

3. 描述框直接粘贴 `docs/PR_BODY.md` 的内容；
4. 创建 PR，记下链接。

## 6. 完成 10 次评审

把 PR 链接发给 10 位同学，请他们：

1. 在自己的机器上 `git clone` 你的分支；
2. 跑两条命令（约 2 分钟）：

   ```bash
   python -m pytest tests -q
   python main.py eval --expert --laps 1
   ```

3. 在 PR 页面点 **Files changed → Review changes → Approve**，留言格式：

   ```text
   已在 <系统/配置> 验证：
   - pytest 6 passed
   - 专家 1 圈 0 碰撞，单圈 22 s
   同意合并。
   ```

10 个人各留一条 Review，就满足"审核次数不少于 10 次"。如果老师要求同一人多次评审，
就按功能分批提 PR（每批一次评审）。

## 7. 可选：键盘演示录屏（把 PARTIAL 变 PASS）

1. 打开 PowerShell，进入项目目录，执行 `python main.py teleop`；
2. 用 **ScreenToGif** 录 MuJoCo 窗口 10 秒，按 W 加速、A/D 转向、空格停车；
3. 导出 `docs/gifs/teleop_manual.gif`（< 5 MB），提交到分支并更新 README。

## 8. 提交前自查

```powershell
git status            # 不应包含 data/、site/
git log --oneline     # 至少 10 条有意义的提交
python -m pytest tests -q
python -m mkdocs build --strict
```

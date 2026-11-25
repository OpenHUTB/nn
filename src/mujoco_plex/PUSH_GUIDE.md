# 推送到 nn 课程仓库指南

本地项目已配置完成。因网络原因无法在此环境自动克隆 GitHub，请按以下步骤手动推送。

## 一、本地预览文档（作业教程第 4 步）

在 PowerShell 中执行：

```powershell
cd C:\Users\18022\Desktop\mujoco_plex\mujoco_plex
pip install -r requirements.txt -i https://pypi.tuna.tsinghua.edu.cn/simple
mkdocs serve
```

浏览器打开 http://127.0.0.1:8000 查看文档。

## 二、推送到 GitHub（作业教程第 5 步）

### 方式 A：用 GitHub Desktop（推荐）

1. 用 GitHub Desktop 克隆 https://github.com/Liu-Yirun/nn 到 `C:\Users\18022\Desktop\nn`
2. 双击运行本目录下的 `integrate_to_nn.ps1`
3. 在 GitHub Desktop 中提交并 Push

### 方式 B：命令行

```powershell
cd C:\Users\18022\Desktop
git clone --depth 1 https://github.com/Liu-Yirun/nn.git nn
cd C:\Users\18022\Desktop\mujoco_plex
powershell -ExecutionPolicy Bypass -File integrate_to_nn.ps1
cd C:\Users\18022\Desktop\nn
git add src/mujoco_plex docs/mujoco_plex mkdocs.yml docs/index.md
git commit -m "添加 MuJoCo ANYmal C 四足机器人仿真项目文档与源码"
git push origin main
```

## 三、已准备的文件

| 路径 | 说明 |
| ---- | ---- |
| `mujoco_plex/index.md` | 项目文档（教程要求的 index.md） |
| `mujoco_plex/mkdocs.yml` | MkDocs 配置 |
| `mujoco_plex/docs/index.md` | MkDocs 文档页 |
| `mujoco_plex/main.py` | 仿真主程序 |
| `nn_bundle/` | 待合并到 nn 仓库的源码与文档 |

## 四、运行仿真

```powershell
cd C:\Users\18022\Desktop\mujoco_plex\mujoco_plex
python main.py
```

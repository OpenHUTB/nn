# openpilot-model 车道线与路径预测工具

## 项目简介
`openpilot-model` 是一个基于 TensorFlow 的轻量化车道线与行驶路径预测工具，支持通过视频文件输入，利用 `supercombo` 预训练模型实时推理左/右车道线及车辆行驶路径，并提供简洁的可视化界面。  
核心设计目标：在保证预测精度的前提下，通过减少帧处理数量、简化可视化渲染，降低 CPU/内存占用，适配资源有限的运行环境（如虚拟机、低配电脑）。


## 核心功能
1. **视频帧预处理**  
   - 自动将视频帧从 BGR 格式转为 YUV_I420 格式（匹配模型输入要求）；  
   - 调整帧尺寸至 512×384（统一输入规格）；  
   - 转换为模型所需的张量格式（6通道特征图，归一化到 [-1, 1]）。

2. **supercombo 模型推理**  
   - 加载预训练的 `supercombo.h5` 模型，预测左车道线（lll）、右车道线（rll）及车辆行驶路径（path）；  
   - 维护模型推理状态（state）和行驶意图（desire），保证帧间预测连续性；  
   - 详细错误捕获（模型加载失败、推理异常均有明确提示）。

3. **轻量化可视化**  
   - 双窗口实时展示：  
     - 原始视频帧窗口（缩小至 480×270，降低渲染压力）；  
     - 预测结果窗口（蓝色=左车道线、红色=右车道线、绿色=行驶路径，固定坐标轴减少重绘计算）；  
   - 支持按 `Q` 键快速退出可视化，避免资源残留。

4. **健壮的错误处理**  
   - 检测视频文件是否存在、是否可正常打开；  
   - 验证模型文件路径有效性，提示缺失解决方案；  
   - 捕获帧读取、预处理、推理过程中的异常，不中断整体程序运行。


## 环境依赖
### 1. 基础环境
- Python 3.7 ~ 3.10（TensorFlow 2.x 对 Python 3.11+ 兼容性较差）

### 2. 依赖库
通过 `pip` 安装以下库（建议指定版本以避免兼容性问题）：
```bash
pip install numpy>=1.21.0          # 数组计算
pip install opencv-python>=4.5.0   # 视频读取与帧处理
pip install tensorflow>=2.6.0      # 模型加载与推理（需包含 keras）
pip install matplotlib>=3.4.0      # 预测结果可视化
pip install tqdm>=4.62.0           # （潜在依赖，部分预处理逻辑可能用到）
```

### 3. 额外依赖
- **FFmpeg**：OpenCV 读取视频需依赖，Ubuntu/Debian 系统安装命令：  
  ```bash
  sudo apt update && sudo apt install ffmpeg -y
  ```
- **common 模块**：项目中 `from common.transformations.xxx` 和 `from common.tools.lib.parser import parser` 依赖 openpilot 开源项目的 `common` 模块，获取方式：  
  1. 克隆 openpilot 仓库：`git clone https://github.com/commaai/openpilot.git`  
  2. 将 `openpilot/common` 文件夹复制到 `openpilot-model` 目录下，或通过 `sys.path.append` 添加模块路径（需修改 `main.py` 开头代码）。


## 文件结构
建议保持以下目录结构，避免路径错误：
```
openpilot-model/                  # 项目根目录
├── main.py                       # 主程序（核心逻辑：视频处理、推理、可视化）
├── common/                       # 从 openpilot 复制的依赖模块
│   ├── transformations/          # 相机/模型坐标转换逻辑
│   │   ├── camera.py
│   │   └── model.py
│   └── tools/
│       └── lib/
│           └── parser.py         # 模型输出解析工具
├── models/                       # 模型存储目录
│   └── supercombo.h5             # supercombo 预训练模型（需自行准备）
├── README.md                     # 项目说明文档（本文档）
└── test_video.mp4                # 测试视频文件（可选，用户可替换为自己的视频）
```


## 使用步骤
### 1. 准备模型文件
1. 获取 `supercombo.h5` 预训练模型（可从 openpilot 相关资源或合法渠道获取）；  
2. 将模型文件放入 `models/` 目录下，确保路径为 `models/supercombo.h5`（若路径不同，需修改 `main.py` 中 `model_path` 变量）。

### 2. 准备测试视频
1. 准备一个视频文件（如 `test_video.mp4`），支持 MP4、AVI 等 OpenCV 兼容格式；  
2. 记住视频文件的完整路径（如 `./test_video.mp4` 或 `/home/user/videos/drive.mp4`）。

### 3. 运行程序
在终端进入 `openpilot-model` 目录，执行以下命令（替换视频路径为实际路径）：
```bash
python main.py ./test_video.mp4
```

### 4. 操作说明
- 程序运行后会弹出两个窗口：  
  1. **原始帧窗口**：显示缩小后的视频原始帧（480×270）；  
  2. **预测结果窗口**：显示车道线（蓝/红）和行驶路径（绿）的预测结果；  
- 按键盘 `Q` 键可退出所有窗口，终止程序运行。


## 关键参数调整
可根据运行环境和需求，修改 `main.py` 中的以下参数：
| 参数名        | 位置                | 默认值 | 说明                                                                 |
|---------------|---------------------|--------|----------------------------------------------------------------------|
| `max_frames`  | `read_video_with_opencv` 函数 | 10     | 最大处理帧数（减少此值可降低 CPU 压力，如改为 5；增大可处理更长视频） |
| `model_path`  | `main` 函数         | "models/supercombo.h5" | 模型文件路径（若模型放在其他位置，需同步修改）                       |
| `cv2.waitKey` | 可视化循环中        | 100    | 帧间等待时间（单位：ms，增大此值可降低窗口刷新频率，减少资源占用）   |
| 预测窗口尺寸  | `plt.subplots`      | (8,6)  | 可修改 `figsize=(6,4)` 缩小窗口，进一步降低渲染压力                  |


## 常见问题解决
### 1. 模型加载失败（`模型加载失败：XXX`）
- 检查 `model_path` 是否正确，确保 `supercombo.h5` 存在；  
- 确认 TensorFlow 版本（建议 2.6~2.10，过高版本可能不兼容旧模型）；  
- 验证模型文件完整性（重新下载模型，避免文件损坏）。

### 2. 视频无法打开（`无法打开视频：XXX，请安装FFmpeg`）
- 执行 `sudo apt install ffmpeg` 安装视频解码依赖；  
- 检查视频文件路径是否正确，确保文件未损坏；  
- 尝试更换视频格式（如将 AVI 转为 MP4）。

### 3. `common` 模块导入错误（`No module named 'common'`）
- 确认 `common` 文件夹已放在 `openpilot-model` 目录下；  
- 若放在其他位置，可在 `main.py` 开头添加路径：  
  ```python
  import sys
  sys.path.append("/path/to/common")  # 替换为 common 文件夹的实际路径
  ```

### 4. 可视化窗口卡住
- 减少 `max_frames` 或增大 `cv2.waitKey` 时间；  
- 关闭其他占用资源的程序（如浏览器、虚拟机快照工具）；  
- 若支持 GPU，可安装 `tensorflow-gpu` 替代 `tensorflow`，利用 GPU 加速推理。


## 注意事项
1. **模型合法性**：`supercombo.h5` 模型需从合法渠道获取，遵守相关开源协议；  
2. **运行性能**：低配电脑/虚拟机可能出现帧间延迟，建议减少 `max_frames` 或关闭其他程序；  
3. **视频分辨率**：建议使用 720p 及以下分辨率的视频，高分辨率视频需先压缩，避免预处理耗时过长；  
4. **资源释放**：程序退出时会自动关闭窗口和释放视频流，但异常终止可能导致资源残留，建议通过 `Q` 键正常退出。
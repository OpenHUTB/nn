# 快速开始指南

## 📋 项目已完成的工作

✅ **环境配置** - Python 依赖已安装
✅ **数据解压** - 视频数据已准备就绪
✅ **视频转帧** - 所有视频已转换为帧图片
✅ **Collage 生成** - 三种布局的 Collage 已生成
✅ **实验完成** - 已有完整的实验结果和日志

## 📁 重要文件位置

- 📄 **项目报告**: [`PROJECT_REPORT.md`](file:///c:/11/carla-temporal-collage-prompting-main/carla-temporal-collage-prompting-main/PROJECT_REPORT.md)
- 📁 **实验日志**: `experiments/high/` 和 `experiments/low/`
- 📁 **视频数据**: `data/videos/`
- 📁 **处理后数据**: `data/data-frames/` 和 `data/collages/`
- 💻 **源代码**: `src/`

## 🎯 核心实验结果

**最佳配置**: Collages-3fps-2-3 (High Quality)

| 指标 | 数值 |
|------|------|
| **准确率** | 85% |
| **正常驾驶 Precision** | 92% |
| **行人事故 Recall** | 93% |
| **车辆碰撞 Recall** | 93% |

详细结果请查看 [`PROJECT_REPORT.md`](file:///c:/11/carla-temporal-collage-prompting-main/carla-temporal-collage-prompting-main/PROJECT_REPORT.md)

## 🚀 如何重新运行实验

### 前置条件

1. **获取 OpenAI API Key**
   - 访问 https://platform.openai.com/account/api-keys
   - 创建新的 API Key
   - 复制并保存

2. **配置 API Key**
   
   编辑 [`.env`](file:///c:/11/carla-temporal-collage-prompting-main/carla-temporal-collage-prompting-main/.env) 文件：
   ```
   OPENAI_API_KEY=你的实际API密钥
   ```

### 运行步骤

```bash
# 1. 视频转帧 (已完成)
python src/dirVid2frames.py

# 2. 生成 Collage (已完成)
python src/dir2Collages-2-3.py

# 3. 运行分析
python src/collage-4o-high.py
```

### 可选的 Collage 布局

```bash
# 2×2 布局
python src/dir2Collages-2-2.py

# 2×3 布局 (推荐)
python src/dir2Collages-2-3.py

# 3×2 布局
python src/dir2Collages-3-2.py
```

### 高质量 vs 低质量分析

```bash
# 高质量分析 (推荐)
python src/collage-4o-high.py

# 低质量分析 (节省 Token)
python src/collage-4o-low.py
```

## 📊 查看已有实验结果

所有实验日志保存在 `experiments/` 目录下，包含：

- 不同帧率 (1fps, 3fps, 30fps)
- 不同布局 (2×2, 2×3, 3×2)
- 不同质量 (high, low)

你可以直接查看这些日志文件来了解不同配置的效果。

## 🆘 常见问题

**Q: OpenAI 网站被屏蔽怎么办？**

A: 可以尝试：
1. 使用支持的网络环境
2. 使用兼容 OpenAI API 的第三方服务
3. 直接使用已有的实验结果

**Q: 如何修改提示词？**

A: 编辑 `src/collage-4o-high.py` 中的 `QUESTION` 变量。

**Q: 如何添加新的视频数据？**

A: 将新视频放入 `data/videos/` 对应的子文件夹，然后重新运行转帧和 Collage 生成脚本。

## 📚 更多信息

详细的项目说明请查看 [`PROJECT_REPORT.md`](file:///c:/11/carla-temporal-collage-prompting-main/carla-temporal-collage-prompting-main/PROJECT_REPORT.md)

---

祝你使用愉快！🎉

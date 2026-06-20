<!-- 感谢提交 pull request! -->
<!-- ⚠️⚠️ 不要删除该文件！这是 Pull Request 的模板 ⚠️⚠️ -->
<!-- 请阅读我们的贡献指南：https://github.com/OpenHUTB/.github/blob/master/CONTRIBUTING.md -->

修改概述: 修复车道检测模块中重复代码和函数签名冗余问题

<!-- Fixes: # -->
<!-- Fixes: # -->

## 修改的详细描述
1. 移除 `lane_advanced.py` 中 `draw_metrics_overlay` 的重复调用，修复曲率/偏移信息被绘制两次的 bug
2. 移除 `lane_advanced.py` 中 `draw_lane_on_original` 和 `draw_metrics_overlay` 函数的重复签名与文档字符串
3. 移除 `lane_video.py` 中重复的 `draw_lane_on_original` 调用和重复的 `save_dir` 初始化代码
4. 移除 `lane_video.py` 中重复的 import 语句
5. 移除 `main.py` 帮助文档中重复的 `--save-docs` 说明和 `choices` 参数

## 经过了什么样的测试?
1. 操作系统：macOS
2. Python 版本：3.14
3. 基础校验：Python 语法检查通过

## 运行效果（动图、视频、图片、链接等）

修复前：曲率半径和偏移信息在图像上被重复绘制两次，导致显示重叠模糊
修复后：信息面板只绘制一次，显示清晰正常

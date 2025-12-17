# main.py

# 设置 matplotlib 使用非交互式后端（'Agg'），避免在无图形界面环境（如服务器）中报错。
# 必须在导入 pyplot 或其他依赖 GUI 的模块之前设置。
import matplotlib
matplotlib.use('Agg')

# 导入程序核心组件：UIHandler 负责用户交互逻辑，Config 提供配置参数
from ui_handler import UIHandler
from config import Config


def main():
    """
    程序主入口函数。
    1. 加载配置（如模型路径、置信度阈值等）
    2. 初始化用户界面处理器
    3. 启动交互流程（支持命令行参数或交互式菜单）
    """
    config = Config()                     # 创建配置实例
    ui_handler = UIHandler(config)        # 传入配置，初始化 UI 处理器
    ui_handler.run()                      # 启动主运行逻辑


# 确保脚本直接运行时才执行 main()，便于模块复用和测试
if __name__ == "__main__":
    main()

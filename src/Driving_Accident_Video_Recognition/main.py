"""
主程序入口：整合所有模块，启动驾驶事故检测工具
"""
import sys
import os

# 关键：将当前脚本所在的目录（code目录）加入Python模块搜索路径
# 确保Python能找到core、utils等子模块
current_dir = os.path.dirname(os.path.abspath(__file__))
sys.path.append(current_dir)

# 现在可以正常导入模块了
from config import REQUIRED_PACKAGES, PYPI_MIRROR
from utils.dependencies import install_dependencies
from core.detector import AccidentDetector

def main():
    """主函数：执行依赖安装 → 初始化检测器 → 启动检测"""
    try:
        print("🚀 启动驾驶事故视频识别工具...")
        # 第一步：自动安装依赖
        install_dependencies(REQUIRED_PACKAGES, PYPI_MIRROR)
        # 第二步：初始化检测器
        detector = AccidentDetector()
        # 第三步：启动检测
        detector.run_detection()
    except KeyboardInterrupt:
        print("\n🛑 用户强制中断程序")
    except Exception as e:
        print(f"\n❌ 程序运行出错：{e}")
    finally:
        print("👋 程序正常退出")

if __name__ == "__main__":
    main()
import sys
import os
# 强制添加源码目录到Python路径（解决ROS编译后导入问题）
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

def main():
    """主函数：初始化+启动仿真"""
    # 1. 检查模型路径
    model_path = os.path.join(SCRIPT_DIR, "../models/humanoid.xml")
    if not os.path.exists(model_path):
        # 备用路径（ROS编译后）
        model_path = os.path.join(SCRIPT_DIR, "models/humanoid.xml")
    if not os.path.exists(model_path):
        print(f"错误：找不到模型文件 {model_path}")
        sys.exit(1)
    print(f"加载模型：{model_path}")

    # 2. 导入核心模块（修复后）
    try:
        from humanoid_stabilizer import HumanoidStabilizer
        from ros_handler import ROSHandler
    except ImportError as e:
        print(f"导入模块失败：{e}")
        sys.exit(1)

    # 3. 初始化机器人控制器
    stabilizer = HumanoidStabilizer(model_path)

    # 4. 启动ROS监听线程
    ros_handler = ROSHandler(stabilizer)
    ros_handler.start()

    # 5. 启动仿真
    try:
        stabilizer.simulate()
    except KeyboardInterrupt:
        print("用户中断仿真")
    finally:
        # 停止ROS线程
        ros_handler.stop()
        ros_handler.join()

if __name__ == "__main__":
    main()

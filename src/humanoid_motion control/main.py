# 标准库
# 导入时间模块，用于控制模拟帧率
import time

# 第三方库
# 导入MuJoCo核心库，提供物理引擎功能
import mujoco
# 从MuJoCo导入可视化工具，用于实时显示模拟过程
from mujoco import viewer

def main():
    """
    主函数：加载人形机器人模型并运行物理模拟
    
    流程：
    1. 加载XML格式的人形机器人模型文件
    2. 初始化模拟数据结构
    3. 设置初始姿势为深蹲姿态
    4. 启动可视化窗口
    5. 无限循环运行模拟（Ctrl+C退出）
    6. 输出关键模拟数据并更新可视化
    """
    # 加载MJCF模型文件
    try:
        model = mujoco.MjModel.from_xml_path("src\humanoid_motion control\humanoid.xml")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建与模型对应的动态数据结构
    data = mujoco.MjData(model)
    
    # 设置初始姿势为深蹲姿态
    mujoco.mj_resetDataKeyframe(model, data, 1)
    
    # 启动被动式可视化窗口
    with viewer.launch_passive(model, data) as v:
        try:
        # 无限循环
            while True:
                # 推进模拟一步
                mujoco.mj_step(model, data)
                
                # 打印机器人关键位置信息
                print(f"时间: {data.time:.2f}, "
                    f"躯干位置: ({data.qpos[0]:.2f}, {data.qpos[1]:.2f}, {data.qpos[2]:.2f})")
                
                # 更新可视化窗口
                v.sync()
                
                # 控制可视化帧率
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("/n正在退出模拟....")
# 程序入口点
if __name__ == "__main__":
    main()
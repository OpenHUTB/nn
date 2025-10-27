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
    5. 运行指定时长的模拟循环
    6. 输出关键模拟数据并更新可视化
    """
    # 加载MJCF模型文件
    # 模型定义了完整的人形机器人结构：包括躯干、四肢、关节和执行器
    try:
        model = mujoco.MjModel.from_xml_path("mujoco01\humanoid.xml")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    # 创建与模型对应的动态数据结构
    # 存储模拟过程中的状态变量（位置、速度、力等）
    data = mujoco.MjData(model)
    
    # 设置初始姿势为深蹲姿态
    # 使用keyframe索引0，对应XML中定义的"squat"姿势
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 启动被动式可视化窗口
    # 被动模式意味着我们手动控制模拟步骤和画面更新
    # 使用with语句确保资源正确释放
    with viewer.launch_passive(model, data) as v:
        # 运行模拟（20秒，按每秒60步计算）
        # 1200步 ≈ 20秒（模型定义的时间步长为0.005秒）
        for _ in range(1200):
            # 推进模拟一步
            # 该函数执行一次完整的前向动力学计算
            mujoco.mj_step(model, data)
            
            # 打印机器人关键位置信息
            # data.qpos存储所有关节的位置，前3个值是躯干位置
            print(f"时间: {data.time:.2f}, "
                f"躯干位置: ({data.qpos[0]:.2f}, {data.qpos[1]:.2f}, {data.qpos[2]:.2f})")
            
            # 更新可视化窗口
            # 将当前模拟状态同步到视图
            v.sync()
            
            # 控制可视化帧率
            # 暂停一小段时间，使可视化更流畅（实际模拟不受影响）
            time.sleep(0.005)

# 程序入口点
if __name__ == "__main__":
    # 调用主函数启动模拟
    main()

# 标准库
import time

# 第三方库
import mujoco
from mujoco import viewer

def main():
    """
    主函数：加载人形机器人模型并运行物理模拟 
    """
    # 1. 加载MJCF模型文件
    model_path = "src/mujoco01/humanoid.xml" 
    try:
        model = mujoco.MjModel.from_xml_path(model_path)
    except Exception as e:
        print(f"模型加载失败: {e}")
        return

    # 2. 初始化模拟数据结构
    data = mujoco.MjData(model)

    # 3. 启动可视化窗口
    print("启动模拟器...")
    with viewer.launch_passive(model, data) as v:
        # 初始化机器人为标准站立姿态（keyframe索引1对应站立姿势）
        mujoco.mj_resetDataKeyframe(model, data, 1)
        last_print_time = 0
        
        while v.is_running():
            mujoco.mj_step(model, data)

            # (修改) 每 0.5 秒打印一次，避免刷屏卡顿
            if data.time - last_print_time > 0.5:
                print(f"时间: {data.time:.2f}, 躯干高度: {data.qpos[2]:.2f}m")
                last_print_time = data.time
        # 4. 运行模拟循环
        while v.is_running():
            # 步进物理引擎
            mujoco.mj_step(model, data)

            # 输出关键模拟数据 (注意：每帧输出会导致刷屏)
            print(f"时间: {data.time:.2f}, "
                  f"躯干位置: ({data.qpos[0]:.2f}, {data.qpos[1]:.2f}, {data.qpos[2]:.2f})")

            # 5. 摔倒检测：躯干高度低于 0.2 米判定摔倒 (假设根节点是自由关节)
            if data.qpos[2] < 0.2:
                print("⚠️ 机器人摔倒了！程序结束。")
                break

            # 6. 更新可视化窗口并控制帧率
            v.sync()
            time.sleep(model.opt.timestep) # 根据物理步长延时，保持真实时间流速

if __name__ == "__main__":
    main()
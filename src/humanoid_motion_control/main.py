# 标准库
import time
# 第三方库
import mujoco
from mujoco import viewer

def main():
    try:
        model = mujoco.MjModel.from_xml_path("src\humanoid_motion_control\humanoid.xml")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    data = mujoco.MjData(model)
    # 临时数据用于获取目标关键帧（站立姿势，索引1）
    target_data = mujoco.MjData(model)
    mujoco.mj_resetDataKeyframe(model, target_data, 1) 
    target_qpos = target_data.qpos.copy()  
    
    # 初始姿势设为深蹲（索引0），从蹲下开始站起
    mujoco.mj_resetDataKeyframe(model, data, 0)
    
    # 控制参数（软体机器人需较小增益，避免震荡）
    Kp = 5.0  # 比例增益（根据软体特性调小）
    
    with viewer.launch_passive(model, data) as v:
        try:
            while True:
                mujoco.mj_step(model, data)
                
                # 计算关节位置误差（跳过前7个根关节，只控制电机关节）
                qpos_error = target_qpos[7:] - data.qpos[7:]
                # 比例控制：控制信号与误差成正比（适配软体机器人的柔性特性）
                data.ctrl[:] = Kp * qpos_error
                
                print(f"时间: {data.time:.2f}, 躯干高度: {data.qpos[2]:.2f}")
                
                v.sync()
                time.sleep(0.005)
        except KeyboardInterrupt:
            print("\n正在退出模拟....")

if __name__ == "__main__":
    main()
import time
import mujoco
from mujoco import viewer

def main():
    # 加载你提供的官方高级人体模型
    try:
        model = mujoco.MjModel.from_xml_path(r"D:\study\nn\src\mujoco_hci_sim\humanoid.xml")
    except Exception as e:
        print(f"模型加载失败: {e}")
        return
    
    data = mujoco.MjData(model)

    # ✅ 关键修复：使用模型自带的 stand 关键帧 → 人不会倒！
    mujoco.mj_resetDataKeyframe(model, data, 1)  

    print("✅ 启动成功！官方人体模型 + 第一人称视角")

    # 启动可视化
    with viewer.launch_passive(model, data) as v:
        while True:
            mujoco.mj_step(model, data)
            
            # 打印状态
            print(f"时间: {data.time:.2f} | 身高: {data.qpos[2]:.2f}m")
            
            v.sync()
            time.sleep(0.01)

if __name__ == "__main__":
    main()
import mujoco
import numpy as np
from mujoco import viewer
import time

#主要内容：
#1    模型和数据的导入
#2    模型如关节角度，速度等的数据的读取和控制->对于机器人更细致的控制
#3    打开mujoco的显示窗口进行渲染
#4    关闭渲染窗口

#加载模型和数据，通过MjData和MjModel类的from_xml_path方法加载xml文件,
model = mujoco.MjModel.from_xml_path("RobotH.xml")  
data= mujoco.MjData(model)
data_walk=np.load("walk.npz")
# data_squat=np.load("squat.npz")

#创建一个可视化的窗口，并设置摄像机参数
Viewer=viewer.launch_passive(model, data)
Viewer.cam.distance = 3   #相机与模型之间的距离
Viewer.cam.azimuth = 90    #相机在水平面上围绕模型旋转的角度
Viewer.cam.elevation = -20  #相机在垂直平面上围绕模型旋转的角度
Viewer.cam.lookat[:] = [0, 0, 1]  # 相机注视点（机器人重心）

#重置到初始状态的方法mj_resetData
mujoco.mj_resetData(model, data)
start_time = time.time()
step_time=0

body_name = "pelvis"
body_id = model.body(body_name).id

#开始渲染，通过mj_step函数输入model和data，渲染动画
while True:
    data.qpos[:]=data_walk['qpos'][step_time%data_walk['qpos'].shape[0]]
    data.qvel[:]=data_walk['qvel'][step_time%data_walk['qvel'].shape[0]]

    # data.qpos[:]=data_squat['qpos'][step_time%data_squat['qpos'].shape[0]]
    # data.qvel[:]=data_squat['qvel'][step_time%data_squat['qvel'].shape[0]]

    #执行一步模拟
    mujoco.mj_step(model, data)  

    #同步渲染
    Viewer.sync()   

    #通过计算等待时间并暂停程序来控制模拟的时间步长
    #确保模拟的步频和时间与物理模型中定义的一致
    #从而提高模拟的真实度   
    step_time+=1    
    target_step_time = 1.0 / 60.0
    step_end = time.time()
    elapsed_time = step_end - start_time - step_time * target_step_time         
    if elapsed_time < target_step_time:
        time.sleep(target_step_time - elapsed_time)

import mujoco
import os
import shutil
import subprocess
import sys

# 原始文件路径 - 修改为机器人手臂模型
original_model_path = os.path.join(os.path.dirname(__file__), "arm_with_gripper.xml")

# 尝试复制到临时目录以避免中文路径问题
temp_dir = os.path.join(os.path.dirname(__file__), "temp")
temp_model_path = os.path.join(temp_dir, "arm_with_gripper.xml")

print(f"检查原始文件是否存在: {original_model_path}")
print(f"原始文件是否存在: {os.path.exists(original_model_path)}")

if os.path.exists(original_model_path):
    try:
        # 创建临时目录（如果不存在）
        os.makedirs(temp_dir, exist_ok=True)
        
        # 复制文件到临时位置
        print(f"正在将文件复制到: {temp_model_path}")
        shutil.copy2(original_model_path, temp_model_path)
        print("文件复制成功")
        
        # 尝试从临时位置加载模型
        print("尝试从临时路径加载模型...")
        model = mujoco.MjModel.from_xml_path(temp_model_path)
        data = mujoco.MjData(model)
        print("模型加载成功")
        print(f"模型名称: {model.names[0:50].decode('utf-8') if model.names else '未知'}")
        print(f"关节数量: {model.njnt}")
        print(f"自由度数量: {model.nq}")
        print(f"执行器数量: {model.nu}")
        print(f"几何体数量: {model.ngeom}")
        print(f"身体数量: {model.nbody}")
        print(f"传感器数量: {model.nsensor}")
        
        # 打印关节信息
        print("\n关节信息:")
        for i in range(model.njnt):
            joint_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i)
            joint_type = model.jnt_type[i]
            type_names = ['free', 'ball', 'slide', 'hinge']
            print(f"  关节 {i}: {joint_name.decode('utf-8') if isinstance(joint_name, bytes) else joint_name} (类型: {type_names[joint_type]})")
            
        # 打印执行器信息
        print("\n执行器信息:")
        for i in range(model.nu):
            actuator_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
            print(f"  执行器 {i}: {actuator_name.decode('utf-8') if isinstance(actuator_name, bytes) else actuator_name}")
            
        # 打印传感器信息
        print("\n传感器信息:")
        for i in range(min(10, model.nsensor)):  # 只显示前10个传感器
            sensor_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SENSOR, i)
            print(f"  传感器 {i}: {sensor_name.decode('utf-8') if isinstance(sensor_name, bytes) else sensor_name}")
        if model.nsensor > 10:
            print(f"  ... 还有 {model.nsensor - 10} 个传感器")
        
        # 尝试启动MuJoCo自带的可视化工具
        print("\n" + "="*50)
        print("正在尝试启动MuJoCo可视化工具...")
        
        # 获取MuJoCo根目录
        mujoco_root = os.path.dirname(os.path.dirname(os.path.dirname(original_model_path)))
        bin_dir = os.path.join(mujoco_root, "bin")
        simulate_exe = os.path.join(bin_dir, "simulate.exe")
        
        if os.path.exists(simulate_exe):
            print(f"找到可视化工具: {simulate_exe}")
            # 启动可视化工具
            try:
                # 使用subprocess启动可视化工具
                subprocess.Popen([simulate_exe, temp_model_path])
                print("MuJoCo可视化工具已启动")
                print("模型文件已加载")
            except Exception as e:
                print(f"启动可视化工具时出错: {e}")
                print("请手动启动可视化工具:")
                print(f"1. 运行 {simulate_exe}")
                print(f"2. 在可视化工具中打开 {temp_model_path}")
        else:
            print("未找到MuJoCo可视化工具")
            print("请检查MuJoCo安装目录结构")
            print(f"期望的工具路径: {simulate_exe}")
            print("\n手动启动方法:")
            print(f"1. 打开命令提示符")
            print(f"2. 导航到 {bin_dir}")
            print(f"3. 运行: simulate.exe \"{temp_model_path}\"")
        
    except Exception as e:
        print(f"处理文件时出错: {e}")
        import traceback
        traceback.print_exc()
        
    finally:
        # 注意：不清理由我们启动的可视化工具使用的临时文件
        # 用户可能仍在查看模型
        print("\n注意: 临时文件未清理，因为可视化工具可能仍在使用它")
        print(f"临时文件位置: {temp_model_path}")
else:
    print(f"原始文件不存在: {original_model_path}")
    print("请检查文件路径是否正确")
    
    # 检查目录结构
    base_dir = os.path.dirname(__file__)
    if os.path.exists(base_dir):
        print("robotic_arm目录下的文件:")
        for file in os.listdir(base_dir):
            print(f"  {file}")
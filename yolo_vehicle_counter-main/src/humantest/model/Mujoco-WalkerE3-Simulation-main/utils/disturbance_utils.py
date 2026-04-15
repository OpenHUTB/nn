"""扰动力相关函数模块"""
import numpy as np
import mujoco
from .math_utils import quat_to_rot_matrix
from .gamepad_utils import get_disturbance_force_base, get_current_mode
from .mode_utils import MODE_DISTURBANCE

# 抗扰动测试相关
disturbance_force_scale = 100.0  # 扰动力缩放系数（N）
disturbance_body_name = "torso_link"  # 默认施加扰动力的body
disturbance_body_id = None  # 将在初始化时设置


def set_disturbance_body_id(m, body_name=None):
    """设置扰动力施加的body ID"""
    global disturbance_body_id, disturbance_body_name
    
    if body_name is not None:
        disturbance_body_name = body_name
    
    try:
        disturbance_body_id = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, disturbance_body_name)
        if disturbance_body_id < 0:
            print(f"警告: 未找到body '{disturbance_body_name}'，将使用body 0")
            disturbance_body_id = 0
        else:
            print(f"扰动力body ID: {disturbance_body_id} ({disturbance_body_name})")
    except Exception:
        print(f"警告: 查找body '{disturbance_body_name}'时出错，将使用body 0")
        disturbance_body_id = 0


def apply_disturbance_force(m, d):
    """在MuJoCo仿真中施加扰动力（从base坐标系变换到世界坐标系），返回base坐标系下的力和torso位置"""
    global disturbance_body_id
    
    current_mode = get_current_mode()
    
    if current_mode != MODE_DISTURBANCE:
        # 非抗扰动模式，清除所有外力
        d.xfrc_applied[:] = 0.0
        return None, None
    
    # 获取base坐标系下的扰动力
    force_base = get_disturbance_force_base()
    
    # 如果扰动力为0，清除外力
    if np.linalg.norm(force_base) < 1e-6:
        d.xfrc_applied[:] = 0.0
        return None, None
    
    # 确保body ID已设置
    if disturbance_body_id is None:
        set_disturbance_body_id(m)
    
    # 获取base（根body，body 0）的旋转四元数
    # 在MuJoCo中，根body的旋转在d.qpos[3:7]
    base_quat = d.qpos[3:7]  # [w, x, y, z]格式
    
    # 将四元数转换为旋转矩阵
    rot_matrix = quat_to_rot_matrix(base_quat)
    
    # 将base坐标系下的力变换到世界坐标系
    force_world = rot_matrix @ force_base
    
    # 清除所有外力
    d.xfrc_applied[:] = 0.0
    
    # 施加扰动力（xfrc_applied是(nbody, 6)数组，前3个是力，后3个是力矩）
    d.xfrc_applied[disturbance_body_id, 0:3] = force_world
    
    # 获取torso_link的位置（用于可视化）
    torso_pos = d.xpos[disturbance_body_id].copy()
    
    return force_base, torso_pos


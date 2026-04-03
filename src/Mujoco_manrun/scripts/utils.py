import numpy as np

def quat_to_euler_xyz(quat):
    """四元数转欧拉角（roll/pitch/yaw）"""
    w, x, y, z = quat
    sinr_cosp = 2 * (w * x + y * z)
    cosr_cosp = 1 - 2 * (x * x + y * y)
    roll = np.arctan2(sinr_cosp, cosr_cosp)
    
    sinp = 2 * (w * y - z * x)
    pitch = np.where(np.abs(sinp) >= 1, np.copysign(np.pi / 2, sinp), np.arcsin(sinp))
    
    siny_cosp = 2 * (w * z + x * y)
    cosy_cosp = 1 - 2 * (y * y + z * z)
    yaw = np.arctan2(siny_cosp, cosy_cosp)
    
    return np.array([roll, pitch, yaw])

def normalize_angle(angle):
    """角度归一化到[-π, π]"""
    angle = np.mod(angle + np.pi, 2 * np.pi) - np.pi
    return angle

def clip_value(value, min_val, max_val, name="value"):
    """带日志的数值限幅"""
    clipped = np.clip(value, min_val, max_val)
    if clipped != value:
        print(f"[警告] {name}超出范围[{min_val}, {max_val}]，已限幅为{clipped}")
    return clipped

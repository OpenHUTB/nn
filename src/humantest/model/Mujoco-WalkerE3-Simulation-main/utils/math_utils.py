"""数学工具函数模块"""
import numpy as np


def get_gravity_orientation(quaternion):
    """计算重力方向（从四元数）"""
    qw = quaternion[0]
    qx = quaternion[1]
    qy = quaternion[2]
    qz = quaternion[3]

    gravity_orientation = np.zeros(3)
    gravity_orientation[0] = 2 * (-qz * qx + qw * qy)
    gravity_orientation[1] = -2 * (qz * qy + qw * qx)
    gravity_orientation[2] = 1 - 2 * (qw * qw + qz * qz)

    return gravity_orientation


def pd_control(target_q, q, kp, target_dq, dq, kd):
    """PD控制器：从位置命令计算力矩"""
    return (target_q - q) * kp + (target_dq - dq) * kd


def quat_to_rot_matrix(quat):
    """将四元数转换为旋转矩阵"""
    w, x, y, z = quat[0], quat[1], quat[2], quat[3]
    rot_mat = np.array([
        [1 - 2*(y*y + z*z), 2*(x*y - w*z), 2*(x*z + w*y)],
        [2*(x*y + w*z), 1 - 2*(x*x + z*z), 2*(y*z - w*x)],
        [2*(x*z - w*y), 2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ])
    return rot_mat


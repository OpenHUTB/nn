"""Viewer控制和相机设置模块"""
import numpy as np
import mujoco

# Viewer控制相关全局变量
track_pelvis = False  # 是否跟踪pelvis
show_contacts = False  # 是否显示接触状态
show_forces = False  # 是否显示足底力

# 使用极坐标系统控制相机（围绕pelvis根节点）
camera_angle = 0.0  # 围绕机器人的水平旋转角度（弧度），0表示在机器人前方
camera_elevation = -0.5  # 相机仰角（弧度），负值表示向下看
camera_distance = 3.0  # 相机距离机器人的距离（米）
camera_angle_speed = 0.05  # 每次旋转的角度增量（弧度）
camera_distance_speed = 0.1  # 每次调整距离的步长（米）

# Reset相关全局变量
reset_requested = False  # 是否请求重置


def update_viewer_settings(viewer, pelvis_body_id, d):
    """更新viewer设置（相机跟踪和显示选项）"""
    global track_pelvis, show_contacts, show_forces
    global camera_angle, camera_elevation, camera_distance
    
    # 设置相机跟踪pelvis并围绕其旋转
    if track_pelvis and pelvis_body_id >= 0:
        # 获取pelvis的位置（d.xpos是(nbody, 3)形状的数组）
        pelvis_pos = d.xpos[pelvis_body_id].copy()
        # 尝试设置相机跟踪（MuJoCo viewer API可能因版本而异）
        try:
            if hasattr(viewer, 'cam'):
                # 设置跟踪目标
                viewer.cam.trackbodyid = pelvis_body_id
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_TRACKING
                # 设置lookat为pelvis位置（相机始终看向pelvis）
                viewer.cam.lookat[:] = pelvis_pos
                # 设置相机距离（围绕pelvis的距离）
                viewer.cam.distance = camera_distance
                # 设置水平旋转角度（azimuth，围绕z轴的旋转，单位：度）
                viewer.cam.azimuth = np.degrees(camera_angle)
                # 设置仰角（elevation，单位：度，负值表示向下看）
                viewer.cam.elevation = np.degrees(camera_elevation)
        except Exception:
            pass  # 如果API不支持，忽略错误
    
    # 设置显示选项（接触和力）
    try:
        if hasattr(viewer, 'opt'):
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = (
                show_contacts
            )
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = (
                show_forces
            )
    except Exception:
        pass  # 如果API不支持，忽略错误


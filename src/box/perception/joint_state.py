import numpy as np
import mujoco
from .base import PerceptionModule

class JointStatePerception(PerceptionModule):
    """关节状态感知模块，采集关节角度、速度、力矩"""
    def __init__(self, model, data, joint_names=None, include_velocity=True, 
                 include_torque=True, normalize=True, **kwargs):
        super().__init__(modality="joint_state", model=model, data=data,** kwargs)
        
        # 关节配置
        self.joint_names = joint_names or self._get_all_joints()  # 默认所有关节
        self.include_velocity = include_velocity
        self.include_torque = include_torque
        self.normalize = normalize
        
        # 获取关节ID和范围（用于归一化）
        self.joint_ids = [mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name) 
                          for name in self.joint_names]
        self.joint_ranges = model.jnt_range[self.joint_ids]  # 关节角度范围(min, max)

    def _get_all_joints(self):
        """获取模型中所有关节名称"""
        joint_names = []
        for i in range(self.model.njnt):
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
            if name:
                joint_names.append(name)
        return joint_names

    def _normalize_qpos(self, qpos):
        """归一化关节角度到[-1, 1]"""
        norm_qpos = (qpos - self.joint_ranges[:, 0]) / (self.joint_ranges[:, 1] - self.joint_ranges[:, 0] + 1e-8)
        return (norm_qpos - 0.5) * 2  # 映射到[-1,1]

    def get_observation(self, model, data, info=None):
        """获取关节状态观测（角度+速度+力矩）"""
        # 关节角度
        qpos = data.qpos[[model.jnt_qposadr[jid] for jid in self.joint_ids]]
        if self.normalize:
            qpos = self._normalize_qpos(qpos)
        
        # 拼接观测
        obs_list = [qpos.astype(np.float32)]
        
        # 关节速度
        if self.include_velocity:
            qvel = data.qvel[[model.jnt_dofadr[jid] for jid in self.joint_ids]]
            # 速度归一化到[-1,1]（基于经验范围）
            if self.normalize:
                qvel = np.clip(qvel / 10.0, -1.0, 1.0)
            obs_list.append(qvel.astype(np.float32))
        
        # 关节力矩
        if self.include_torque:
            torque = data.qfrc_actuator[[model.jnt_dofadr[jid] for jid in self.joint_ids]]
            # 力矩归一化到[-1,1]
            if self.normalize:
                torque = np.clip(torque / 50.0, -1.0, 1.0)
            obs_list.append(torque.astype(np.float32))
        
        return np.concatenate(obs_list)

    def get_observation_space_params(self):
        """定义关节状态观测空间参数"""
        # 计算观测维度
        dim = len(self.joint_ids)
        if self.include_velocity:
            dim += len(self.joint_ids)
        if self.include_torque:
            dim += len(self.joint_ids)
        
        return {
            "low": -1.0 if self.normalize else -np.inf,
            "high": 1.0 if self.normalize else np.inf,
            "shape": (dim,)
        }
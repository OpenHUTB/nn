# 从上级包的base模块导入基础模块类（UITB框架的核心基类）
from ...base import BaseModule
# 导入数值计算库，用于数组操作
import numpy as np


class BasicWithEndEffectorPosition(BaseModule):
    """
    本体感知模块：整合关节状态、肌肉激活度和末端执行器位置的观测生成器
    核心功能：
    1. 标准化关节角度、速度、加速度、肌肉激活值
    2. 获取末端执行器（如手部、机械爪）的全局位置
    3. 拼接所有感知信息，输出一维观测向量（适配强化学习输入）
    继承自：BaseModule（UITB框架的基础模块类，提供模拟器交互接口）
    """

    def __init__(self, model, data, bm_model, end_effector, **kwargs):
        """
        初始化感知模块实例
        Args:
            model: Mujoco模拟器的模型实例（存储物理模型结构，如关节、刚体、执行器）
            data: Mujoco模拟器的数据实例（存储实时物理数据，如关节角度、速度、力）
            bm_model: 生物力学模型实例（继承自uitb.bm_models.base.BaseBMModel，提供关节/执行器映射）
            end_effector (list of lists): 末端执行器定义，格式示例：
                - 单末端执行器：[["site", "right_hand_site"]]
                - 多末端执行器：[["body", "left_hand"], ["geom", "right_foot"]]
                每个子列表第一个元素是Mujoco元素类型（geom/body/site），第二个是元素名称
            **kwargs: 可选参数（如"rng"随机数种子）
        """
        # 调用父类初始化方法，传入模拟器实例和生物力学模型
        super().__init__(model, data, bm_model,** kwargs)

        # 校验末端执行器参数格式：必须是列表类型
        if not isinstance(end_effector, list):
            raise RuntimeError(
                "end_effector必须是长度为2的列表，或嵌套列表（每个子列表长度为2）"
            )

        # 检查是否为单层列表（如["site", "hand"]），若是则转为嵌套列表（统一格式）
        if isinstance(end_effector[0], str):
            end_effector = [end_effector]

        # 校验所有嵌套子列表的长度必须为2（类型+名称）
        if any(len(pair) != 2 for pair in end_effector):
            raise RuntimeError("end_effector的每个子列表必须包含2个元素：类型+名称")

        # 保存标准化后的末端执行器配置（实例变量，供后续方法调用）
        self._end_effector = end_effector

    @staticmethod
    def insert(task, **kwargs):
        """
        静态方法：用于将模块插入到任务配置中（UITB框架预留接口，暂无实现）
        Args:
            task: 任务实例（包含模拟器、评估指标等）
            **kwargs: 扩展参数
        """
        pass

    @property
    def _default_encoder(self):
        """
        属性方法：返回默认的编码器配置（适配强化学习的特征编码）
        此处配置：单层全连接编码器，输出特征维度为128
        Returns:
            dict: 编码器的模块路径、类名、参数
        """
        return {
            "module": "rl.encoders",  # 编码器模块路径
            "cls": "OneLayer",        # 编码器类名（单层全连接）
            "kwargs": {"out_features": 128}  # 编码器参数：输出特征维度
        }

    def get_observation(self, model, data, info=None):
        """
        核心方法：生成标准化的感知观测向量
        Args:
            model: Mujoco模型实例（同__init__）
            data: Mujoco数据实例（同__init__，存储实时数据）
            info: 额外信息字典（可选，如环境状态）
        Returns:
            np.ndarray: 一维数组，包含所有标准化的感知特征
        """
        # -------------------------- 1. 标准化关节角度（qpos） --------------------------
        # 获取独立关节的角度范围（model.jnt_range：[min, max]）
        jnt_range = model.jnt_range[self._bm_model.independent_joints]
        # 复制独立关节的当前角度（避免修改原数据）
        qpos = data.qpos[self._bm_model.independent_qpos].copy()
        # 第一步归一化：将角度映射到[0, 1]区间
        qpos = (qpos - jnt_range[:, 0]) / (jnt_range[:, 1] - jnt_range[:, 0])
        # 第二步归一化：将角度映射到[-1, 1]区间（适配神经网络输入）
        qpos = (qpos - 0.5) * 2

        # -------------------------- 2. 获取关节速度/加速度 --------------------------
        # 复制独立自由度的关节速度（原始值，未标准化）
        qvel = data.qvel[self._bm_model.independent_dofs].copy()
        # 复制独立自由度的关节加速度（原始值，未标准化）
        qacc = data.qacc[self._bm_model.independent_dofs].copy()

        # -------------------------- 3. 获取末端执行器全局位置 --------------------------
        ee_position = []  # 存储所有末端执行器的位置
        for pair in self._end_effector:
            # pair[0]：Mujoco元素类型（geom/body/site），pair[1]：元素名称
            # getattr(data, pair[0])(pair[1])：获取对应元素的实例
            # .xpos：获取元素的全局坐标（3维：x,y,z）
            ee_pos = getattr(data, pair[0])(pair[1]).xpos.copy()
            ee_position.append(ee_pos)
        # 将多个末端执行器的位置拼接为一维数组（如2个末端执行器则为6维）
        ee_position = np.hstack(ee_position)

        # -------------------------- 4. 标准化肌肉/执行器激活值 --------------------------
        # data.act：所有执行器的当前激活值（范围通常[0,1]），映射到[-1,1]
        act = (data.act.copy() - 0.5) * 2
        # 生物力学模型中平滑后的电机激活值，同样映射到[-1,1]
        motor_act = (self._bm_model.motor_act.copy() - 0.5) * 2

        # -------------------------- 5. 拼接所有感知特征 --------------------------
        # 本体感知特征：关节角度+速度+加速度+末端位置+执行器激活+电机激活
        proprioception = np.concatenate([qpos, qvel, qacc, ee_position, act, motor_act])

        # 返回最终的观测向量（一维数组，可直接输入强化学习网络）
        return proprioception

    def _get_state(self, model, data):
        """
        辅助方法：获取末端执行器的详细状态（用于日志/评估，非训练观测）
        Args:
            model: Mujoco模型实例
            data: Mujoco数据实例
        Returns:
            dict: 键为"元素名称_xpos/xmat"，值为对应的位置/旋转矩阵
        """
        state = {}
        for pair in self._end_effector:
            # 存储末端执行器的全局位置（xpos：3维坐标）
            state[f"{pair[1]}_xpos"] = getattr(data, pair[0])(pair[1]).xpos.copy()
            # 存储末端执行器的旋转矩阵（xmat：9维，描述空间姿态）
            state[f"{pair[1]}_xmat"] = getattr(data, pair[0])(pair[1]).xmat.copy()
        return state
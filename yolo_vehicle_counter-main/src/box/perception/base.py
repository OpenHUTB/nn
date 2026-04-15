import numpy as np
import mujoco
from collections import defaultdict

class PerceptionModule:
    """感知子模块基类，所有具体感知模块需继承此类"""
    def __init__(self, modality, model, data, **kwargs):
        self.modality = modality  # 感知模态名称（如vision/joint_state）
        self.model = model        # MuJoCo模型
        self.data = data          # MuJoCo数据
        self.kwargs = kwargs      # 模块参数
        self.cameras = []         # 关联的相机（视觉类模块用）

    def reset(self, model, data):
        """重置感知模块状态"""
        self.model = model
        self.data = data

    def update(self, model, data):
        """每步更新感知数据"""
        pass

    def get_observation(self, model, data, info=None):
        """获取该模块的观测数据（需子类实现）"""
        raise NotImplementedError

    def get_observation_space_params(self):
        """获取Gymnasium观测空间参数（需子类实现）"""
        raise NotImplementedError

    def get_renders(self):
        """获取可视化渲染结果（视觉类模块用）"""
        return []

    def close(self):
        """释放资源"""
        pass


class Perception:
    """感知总控制器，管理所有感知子模块，适配仿真器架构"""
    def __init__(self, model, data, bm_model, perception_modules, common_kwargs):
        self.model = model
        self.data = data
        self.bm_model = bm_model  # 关联生物力学模型
        self.common_kwargs = common_kwargs  # 通用参数（如dt、callbacks）
        
        # 初始化感知子模块
        self.perception_modules = []
        self.cameras_dict = defaultdict(list)  # 相机映射（适配渲染）
        self.nu = 0  # 感知模块动作维度（无动作则为0，保持与仿真器兼容）

        # 遍历配置的感知模块，初始化实例
        for module_cls, module_kwargs in perception_modules.items():
            module = module_cls(
                model=model,
                data=data,
                **module_kwargs
            )
            self.perception_modules.append(module)
            # 关联相机（用于渲染）
            if hasattr(module, 'cameras') and module.cameras:
                self.cameras_dict[module] = module.cameras

    def reset(self, model, data):
        """重置所有感知模块"""
        self.model = model
        self.data = data
        for module in self.perception_modules:
            module.reset(model, data)

    def update(self, model, data):
        """每步更新所有感知模块"""
        self.model = model
        self.data = data
        for module in self.perception_modules:
            module.update(model, data)

    def get_observation(self, model, data, info=None):
        """收集所有感知模块的观测数据，返回字典（适配仿真器观测空间）"""
        observation = {}
        for module in self.perception_modules:
            obs = module.get_observation(model, data, info)
            if obs is not None:
                observation[module.modality] = obs
        return observation

    def get_state(self, model, data):
        """获取感知模块状态（用于仿真器状态记录）"""
        state = {}
        for module in self.perception_modules:
            state[f"perception_{module.modality}"] = module.get_observation(model, data)
        return state

    def get_renders(self):
        """获取所有视觉类模块的渲染结果（适配仿真器可视化）"""
        renders = []
        for module in self.perception_modules:
            renders.extend(module.get_renders())
        return renders

    def set_ctrl(self, model, data, ctrl):
        """感知模块动作控制（无动作则空实现，保持与仿真器接口兼容）"""
        pass

    def close(self, **kwargs):
        """释放所有感知模块资源"""
        for module in self.perception_modules:
            module.close()
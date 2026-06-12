import numpy as np
import mujoco
from mujoco import MjSpec
from loco_mujoco import ImitationFactory
from loco_mujoco import PATH_TO_MODELS
import time

def main():
    # 配置环境参数
    env_name = "UnitreeH1"
    task = "walk"
    n_episodes = 10
    n_steps_per_episode = 500

    # 创建自定义模型规格，添加几何体
    def create_custom_spec():
        # 加载UnitreeH1的默认模型
        h1_model_path = PATH_TO_MODELS / "unitree_h1/h1.xml"
        spec = MjSpec.from_file(str(h1_model_path))
        
        # 打印所有身体名称，确认地板身体的正确名称
        print("模型中所有身体名称：")
        for body in spec.bodies:
            print(body.name)
        
        # 尝试查找地板身体（根据打印结果，模型中没有floor/ground，使用world作为父节点）
        floor_body = spec.find_body("floor")
        if floor_body is None:
            floor_body = spec.find_body("ground")
        if floor_body is None:
            # 若仍找不到，使用根节点"world"作为父节点（模型中存在"world"身体）
            floor_body = spec.find_body("world")
            print("使用根节点'world'作为几何体的父节点")
        
        # 定义圆柱体几何体属性（size改为3个元素，符合Mujoco要求）
        cylinder_attr = dict(
            type=mujoco.mjtGeom.mjGEOM_CYLINDER,
            size=[0.3, 1, 0.0],  # 修正：[半径, 半长度, 0]（必须3个元素）
            pos=[8, 0, 1.0],       # 位置（相对于父节点world）
            rgba=[1.0, 0.0, 0.0, 0.5],  # 红色半透明
            contype=1,
            conaffinity=1
        )
        
        # 向父节点添加几何体
        floor_body.add_geom(name="custom_cylinder", **cylinder_attr)
        
        return spec

    # 创建带自定义几何体的环境
    custom_spec = create_custom_spec()
    env = ImitationFactory.make(
        env_name,
        default_dataset_conf=dict(task=task),
        spec=custom_spec  # 传入自定义规格
    )

    # 播放轨迹并渲染
    env.play_trajectory(
        n_episodes=n_episodes,
        n_steps_per_episode=n_steps_per_episode,
        render=True
    )

if __name__ == "__main__":
    main()
"""
MoblArmsIndex 生物力学模型 - 修改版
这是基于 MoBL ARMS 模型的 MuJoCo 实现，专门用于手指指向任务
"""

import sys
import os

# ================ 动态导入 BaseBMModel ================
try:
    # 尝试1: 从原始相对位置导入
    from ..base import BaseBMModel
    print("✓ 使用相对导入: from ..base import BaseBMModel")
except ImportError:
    try:
        # 尝试2: 从绝对路径导入（假设在 uitb 包中）
        from uitb.base import BaseBMModel
        print("✓ 使用绝对导入: from uitb.base import BaseBMModel")
    except ImportError:
        try:
            # 尝试3: 从常见路径导入
            import sys
            import os
            
            # 将项目根目录添加到路径
            current_dir = os.path.dirname(os.path.abspath(__file__))
            project_root = os.path.dirname(os.path.dirname(current_dir))
            sys.path.insert(0, project_root)
            
            from base import BaseBMModel
            print(f"✓ 使用项目根目录导入: from base import BaseBMModel (路径: {project_root})")
        except ImportError as e:
            # 尝试4: 搜索并动态导入
            print("⚠ 无法自动导入 BaseBMModel，尝试动态查找...")
            
            # 搜索 base.py 文件
            base_path = None
            search_dirs = [
                os.path.dirname(os.path.dirname(os.path.abspath(__file__))),  # 上一级目录
                os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),  # 上上级目录
                os.path.join(os.path.dirname(__file__), '..'),  # 相对路径
            ]
            
            for search_dir in search_dirs:
                possible_path = os.path.join(search_dir, 'base.py')
                if os.path.exists(possible_path):
                    base_path = possible_path
                    break
            
            if base_path:
                # 动态导入
                import importlib.util
                spec = importlib.util.spec_from_file_location("base_module", base_path)
                base_module = importlib.util.module_from_spec(spec)
                sys.modules[spec.name] = base_module
                spec.loader.exec_module(base_module)
                BaseBMModel = base_module.BaseBMModel
                print(f"✓ 动态导入成功: {base_path}")
            else:
                # 如果都失败，创建简化的 BaseBMModel
                print("⚠ 无法找到 BaseBMModel，使用简化版本")
                
                class BaseBMModel:
                    """简化的 BaseBMModel 基类"""
                    def __init__(self, model, data, **kwargs):
                        self.model = model
                        self.data = data
                    
                    def _update(self, model, data):
                        """更新方法 - 子类应重写"""
                        pass
                    
                    @classmethod
                    def _get_floor(cls):
                        """获取地板 - 子类应重写"""
                        return None
                
                BaseBMModel = BaseBMModel
                print("✓ 使用简化版本的 BaseBMModel")

# ================ 继续原来的代码 ================
import numpy as np
import mujoco


class MoblArmsIndex(BaseBMModel):
    """
    基于 MoBL ARMS 模型的生物力学模型
    
    来源:
    - 原始 OpenSim 模型: https://simtk.org/frs/?group_id=657
    - MuJoCo 转换工具: https://github.com/aikkala/O2MConverter
    
    说明:
    此模型与 uitb/bm_models/mobl_arms 中的模型相同，
    但食指处于弯曲状态并包含力传感器。
    """
    
    def __init__(self, model, data, **kwargs):
        """
        初始化 MoblArmsIndex 模型
        
        参数:
            model: MuJoCo 模型实例
            data: MuJoCo 数据实例
            **kwargs: 额外参数
                - shoulder_variant: 肩部变体，可选 "none" (默认) 或 "patch-v1"
        """
        super().__init__(model, data, **kwargs)
        
        # 设置肩部变体
        # 使用 "none" 作为默认值
        # 使用 "patch-v1" 可以获得更合理的运动外观（未经全面测试）
        self.shoulder_variant = kwargs.get("shoulder_variant", "none")
        
        print(f"✓ MoblArmsIndex 初始化完成，肩部变体: {self.shoulder_variant}")

    def _update(self, model, data):
        """
        更新模型状态
        
        此方法在每个时间步被调用，用于更新肩部约束。
        
        参数:
            model: MuJoCo 模型实例
            data: MuJoCo 数据实例
        """
        # 更新肩部等式约束
        if self.shoulder_variant.startswith("patch"):
            # 约束1: shoulder1_r2_con 约束的数据更新
            model.equality("shoulder1_r2_con").data[1] = \
                -((np.pi - 2 * data.joint('shoulder_elv').qpos) / np.pi)

            # patch-v2 变体有额外的约束
            if self.shoulder_variant == "patch-v2":
                # 动态调整肩部旋转范围
                data.joint('shoulder_rot').range[:] = \
                    np.array([-np.pi / 2, np.pi / 9]) - \
                    2 * np.min((data.joint('shoulder_elv').qpos,
                              np.pi - data.joint('shoulder_elv').qpos)) / np.pi \
                    * data.joint('elv_angle').qpos

            # 执行前向计算，更新物理状态
            mujoco.mj_forward(model, data)
            
            # 打印调试信息（可选）
            if hasattr(self, 'debug') and self.debug:
                print(f"更新肩部约束: shoulder_elv={data.joint('shoulder_elv').qpos:.3f}, "
                      f"约束值={model.equality('shoulder1_r2_con').data[1]:.3f}")

    @classmethod
    def _get_floor(cls):
        """
        获取地板配置
        
        此模型不包含地板，返回 None。
        
        返回:
            None: 表示此模型没有地板
        """
        return None
    
    def get_force_sensor_data(self, data, sensor_name="index_force_sensor"):
        """
        获取力传感器数据
        
        参数:
            data: MuJoCo 数据实例
            sensor_name: 传感器名称，默认为 "index_force_sensor"
            
        返回:
            numpy.ndarray: 传感器数据，如果没有找到传感器则返回 None
        """
        try:
            sensor_id = self.model.sensor(sensor_name).id
            return data.sensordata[sensor_id]
        except Exception as e:
            if hasattr(self, 'debug') and self.debug:
                print(f"⚠ 无法获取力传感器数据: {e}")
            return None
    
    def set_debug_mode(self, debug=True):
        """
        设置调试模式
        
        参数:
            debug: 是否启用调试模式
        """
        self.debug = debug
        
    def __str__(self):
        """
        返回模型的字符串表示
        
        返回:
            str: 模型描述
        """
        return (f"MoblArmsIndex 模型 (肩部变体: {self.shoulder_variant})\n"
                f"描述: 基于 MoBL ARMS 的上肢模型，食指弯曲并包含力传感器")


# ================ 测试代码 ================
if __name__ == "__main__":
    """
    直接运行此文件的测试代码
    """
    print("=" * 60)
    print("MoblArmsIndex 模型测试")
    print("=" * 60)
    
    # 测试导入
    print("1. 测试导入和类定义:")
    print(f"   类名: {MoblArmsIndex.__name__}")
    print(f"   基类: {MoblArmsIndex.__bases__[0].__name__}")
    print(f"   文档: {MoblArmsIndex.__doc__.strip().split('\n')[0]}")
    
    # 测试创建实例（需要 MuJoCo 模型）
    try:
        import mujoco
        import numpy as np
        
        print("\n2. 测试创建实例:")
        
        # 创建一个简单的 MuJoCo 模型用于测试
        xml_string = """
        <mujoco>
            <worldbody>
                <body name="base" pos="0 0 0">
                    <joint type="free"/>
                    <geom type="box" size="0.1 0.1 0.1" rgba="0.5 0.5 0.5 1"/>
                </body>
            </worldbody>
        </mujoco>
        """
        
        model = mujoco.MjModel.from_xml_string(xml_string)
        data = mujoco.MjData(model)
        
        # 创建模型实例
        bm_model = MoblArmsIndex(model, data, shoulder_variant="none")
        
        print(f"   ✓ 成功创建 MoblArmsIndex 实例")
        print(f"   肩部变体: {bm_model.shoulder_variant}")
        
        # 测试更新方法
        print("\n3. 测试更新方法:")
        bm_model._update(model, data)
        print("   ✓ _update 方法执行成功")
        
        # 测试获取地板
        print("\n4. 测试获取地板:")
        floor = bm_model._get_floor()
        print(f"   地板配置: {floor}")
        
        # 测试字符串表示
        print("\n5. 测试字符串表示:")
        print(f"   {bm_model}")
        
        print("\n" + "=" * 60)
        print("所有测试通过！")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n✗ 测试过程中出现错误: {type(e).__name__}: {e}")
        import traceback
        traceback.print_exc()
        
        print("\n" + "=" * 60)
        print("注意: 缺少 MuJoCo 模型文件，但导入测试成功")
        print("=" * 60)


# ================ 使用示例 ================
"""
在您的项目中这样使用:

# 方法1: 直接导入使用
from moblARMslndex import MoblArmsIndex

# 创建 MuJoCo 模型和数据
model = mujoco.MjModel.from_xml_path("path/to/model.xml")
data = mujoco.MjData(model)

# 创建生物力学模型实例
bm_model = MoblArmsIndex(model, data, shoulder_variant="patch-v1")

# 在每个时间步更新
for step in range(100):
    # ... 设置控制信号 ...
    mujoco.mj_step(model, data)
    bm_model._update(model, data)

# 方法2: 在模拟器环境中使用
# 在配置文件中指定:
# bm_model:
#   cls: "MoblArmsIndex"
#   kwargs:
#     shoulder_variant: "patch-v1"
"""
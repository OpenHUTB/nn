import gymnasium as gym
from gymnasium import spaces
import pygame
import mujoco
import os
import numpy as np
import scipy
import matplotlib
import sys
import importlib
import shutil
import inspect
import pathlib
from datetime import datetime
import copy
from collections import defaultdict
import xml.etree.ElementTree as ET

# 感知模块基类：统一管理视觉/触觉等感知组件
from .perception.base import Perception
# 渲染工具：Camera负责图像采集，Context提供渲染环境（显存/分辨率等）
from .utils.rendering import Camera, Context
# 通用工具函数：路径处理、配置解析/写入、包名合法性校验
from .utils.functions import output_path, parent_path, is_suitable_package_name, parse_yaml, write_yaml


class Simulator(gym.Env):
    """
    核心仿真器类，继承自gym.Env以兼容强化学习生态
    核心功能：
    1. 从YAML配置文件构建独立的Python仿真包
    2. 集成生物力学模型(BM)、任务模型(Task)、感知模型(Perception)
    3. 实现标准的gym接口（step/reset/render），支持RL训练与可视化
    """

    # 版本号：遵循X.Y.Z格式（主版本.次版本.修订版），用于版本兼容性校验
    version = "1.1.0"

    @classmethod
    def get_class(cls, *args):
        """
        动态导入指定类（反射机制）
        用途：根据配置文件中的字符串路径加载模块类（如BM模型/任务模型/感知模型）
        Args:
            *args: 类的路径片段，最后一个元素为类名，其余为模块路径
                   示例：args=("tasks", "reach_task.ReachTask") → 加载tasks模块下的ReachTask类
        Returns:
            class: 导入的目标类
        """
        # 拼接模块路径（排除最后一个元素<类名>）
        modules = ".".join(args[:-1])
        # 处理类名带模块路径的情况（如 "rl.encoders.SmallCNN"）
        if "." in args[-1]:
            splitted = args[-1].split(".")
            if modules == "":
                modules = ".".join(splitted[:-1])
            else:
                modules += "." + ".".join(splitted[:-1])
            cls_name = splitted[-1]  # 提取最终类名
        else:
            cls_name = args[-1]
        
        # 导入模块并获取类
        module = cls.get_module(modules)
        return getattr(module, cls_name)

    @classmethod
    def get_module(cls, *args):
        """
        动态导入指定模块（辅助get_class）
        Args:
            *args: 模块路径片段，示例：args=("bm_models", "human_arm") → 导入bm_models.human_arm模块
        Returns:
            module: 导入的模块对象
        """
        # src为根模块名（如uitb），拼接完整模块路径
        src = __name__.split(".")[0]
        modules = ".".join(args)
        return importlib.import_module(src + "." + modules)

    @classmethod
    def build(cls, config):
        """
        核心构建方法：从配置文件生成独立的仿真包
        Args:
            config: 配置信息，支持两种格式：
                    1. str: YAML配置文件路径
                    2. dict: 解析后的配置字典（含仿真/模型/感知等配置）
        Returns:
            str: 生成的仿真包文件夹路径
        Raises:
            FileNotFoundError: 配置文件不存在
            AssertionError: 配置缺少必填项（simulation/bm_model/task/run_parameters等）
            NameError: 包名不合法（含大写/空格/数字开头等）
        """
        # 第一步：解析配置文件（若输入为路径）
        if isinstance(config, str):
            if not os.path.isfile(config):
                raise FileNotFoundError(f"配置文件不存在: {config}")
            config = parse_yaml(config)  # 解析YAML为字典

        # 第二步：校验配置必填项
        assert "simulation" in config, "配置必须包含simulation字段"
        assert "bm_model" in config["simulation"], "配置必须指定生物力学模型(bm_model)"
        assert "task" in config["simulation"], "配置必须指定任务模型(task)"
        assert "run_parameters" in config["simulation"], "配置必须指定运行参数(run_parameters)"
        
        # 提取运行参数并校验动作采样频率
        run_parameters = config["simulation"]["run_parameters"].copy()
        assert "action_sample_freq" in run_parameters, "运行参数必须指定动作采样频率(action_sample_freq)"

        # 第三步：设置仿真包基础信息
        config["version"] = cls.version  # 记录仿真器版本
        # 确定仿真包保存路径（优先用配置中的simulator_folder，否则默认到uitb/simulators）
        if "simulator_folder" in config:
            simulator_folder = os.path.normpath(config["simulator_folder"])
        else:
            simulator_folder = os.path.join(output_path(), config["simulator_name"])
        
        # 处理包名（默认用simulator_name，校验合法性）
        if "package_name" not in config:
            config["package_name"] = config["simulator_name"]
        if not is_suitable_package_name(config["package_name"]):
            raise NameError(
                "包名不合法！仅允许小写字母、下划线，且不能以数字开头\n"
                "请检查配置中的package_name/simulator_name字段"
            )
        
        # 生成gym环境名（格式：uitb:<包名>-v0）
        config["gym_name"] = "uitb:" + config["package_name"] + "-v0"

        # 第四步：克隆核心文件到仿真包（创建独立包结构）
        cls._clone(simulator_folder, config["package_name"])

        # 第五步：加载并初始化任务模型
        task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
        # 克隆任务模型文件到仿真包（支持Unity可执行文件路径传递）
        task_cls.clone(
            simulator_folder, 
            config["package_name"], 
            app_executable=config["simulation"]["task"].get("kwargs", {}).get("unity_executable", None)
        )
        # 初始化任务模型，返回MuJoCo的XML根节点（simulation）
        simulation = task_cls.initialise(config["simulation"]["task"].get("kwargs", {}))

        # 第六步：设置MuJoCo编译器默认参数（物理属性相关）
        compiler_defaults = {
            "inertiafromgeom": "auto",    # 从几何形状自动计算惯性
            "balanceinertia": "true",     # 平衡惯性张量
            "boundmass": "0.001",         # 质量下界
            "boundinertia": "0.001",      # 惯性下界
            "inertiagrouprange": "0 1"    # 惯性组范围
        }
        compiler = simulation.find("compiler")
        if compiler is None:
            # 无compiler节点则新建
            ET.SubElement(simulation, "compiler", compiler_defaults)
        else:
            # 已有则更新属性
            compiler.attrib.update(compiler_defaults)

        # 第七步：加载并插入生物力学模型到XML
        bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
        bm_cls.clone(simulator_folder, config["package_name"])  # 克隆BM模型文件
        bm_cls.insert(simulation)  # 将BM模型（如人体上肢）插入到MuJoCo XML

        # 第八步：加载并插入感知模块（如FixedEye/UnityHeadset）到XML
        for module_cfg in config["simulation"].get("perception_modules", []):
            module_cls = cls.get_class("perception", module_cfg["cls"])
            module_kwargs = module_cfg.get("kwargs", {})
            module_cls.clone(simulator_folder, config["package_name"])  # 克隆感知模块文件
            module_cls.insert(simulation, **module_kwargs)  # 插入到XML（如创建相机/传感器）

        # 第九步：克隆RL相关文件（编码器/策略等），使仿真包完全独立
        rl_cls = cls.get_class("rl", config["rl"]["algorithm"])
        rl_cls.clone(simulator_folder, config["package_name"])

        # 第十步：保存MuJoCo XML文件（物理仿真核心配置）
        simulation_file = os.path.join(simulator_folder, config["package_name"], "simulation")
        with open(simulation_file + ".xml", 'w') as file:
            simulation.write(file, encoding='unicode')  # 写入XML内容

        # 第十一步：初始化仿真器，生成二进制模型（加快后续加载）
        model, _, _, _, _, _ = cls._initialise(config, simulator_folder, {**run_parameters, "build": True})
        # 重新加载XML并保存修改后的版本（MuJoCo要求先加载才能保存）
        mujoco.MjModel.from_xml_path(simulation_file + ".xml")
        mujoco.mj_saveLastXML(simulation_file + ".xml", model)
        # 保存二进制模型（.mjcf），加载速度比XML快
        mujoco.mj_saveModel(model, simulation_file + ".mjcf", None)

        # 第十二步：记录构建时间并保存最终配置
        config["built"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        write_yaml(config, os.path.join(simulator_folder, "config.yaml"))

        return simulator_folder

    @classmethod
    def _clone(cls, simulator_folder, package_name):
        """
        内部方法：克隆核心文件到新仿真包，创建独立的Python包结构
        Args:
            simulator_folder: 仿真包根路径
            package_name: 包名（作为子文件夹名）
        """
        # 创建包文件夹（如simulators/reach_task/）
        dst = os.path.join(simulator_folder, package_name)
        os.makedirs(dst, exist_ok=True)

        # 1. 复制当前Simulator类文件到包中
        src = pathlib.Path(inspect.getfile(cls))  # 获取当前文件路径
        shutil.copyfile(src, os.path.join(dst, src.name))

        # 2. 创建__init__.py（使文件夹成为Python包，注册gym环境）
        with open(os.path.join(dst, "__init__.py"), "w") as file:
            file.write("from .simulator import Simulator\n\n")
            file.write("from gymnasium.envs.registration import register\n")
            file.write("import pathlib\n\n")
            file.write("module_folder = pathlib.Path(__file__).parent\n")
            file.write("simulator_folder = module_folder.parent\n")
            file.write("kwargs = {'simulator_folder': simulator_folder}\n")
            # 注册gym环境，使gym.make("uitb:<包名>-v0")可调用
            file.write("register(id=f'{module_folder.stem}-v0', entry_point=f'{module_folder.stem}.simulator:Simulator', kwargs=kwargs)\n")

        # 3. 复制工具文件夹（utils/train/test），保证包独立性
        # 复制utils（渲染/路径等工具）
        shutil.copytree(
            os.path.join(parent_path(src), "utils"),
            os.path.join(simulator_folder, package_name, "utils"),
            dirs_exist_ok=True  # 已存在则覆盖
        )
        # 复制train（训练相关代码）
        shutil.copytree(
            os.path.join(parent_path(src), "train"),
            os.path.join(simulator_folder, package_name, "train"),
            dirs_exist_ok=True
        )
        # 复制test（测试相关代码）
        shutil.copytree(
            os.path.join(parent_path(src), "test"),
            os.path.join(simulator_folder, package_name, "test"),
            dirs_exist_ok=True
        )

    @classmethod
    def _initialise(cls, config, simulator_folder, run_parameters):
        """
        内部方法：初始化仿真核心组件（MuJoCo模型/数据、各模块实例）
        Args:
            config: 配置字典
            simulator_folder: 仿真包路径
            run_parameters: 运行参数（含动作采样频率、渲染分辨率等）
        Returns:
            tuple: (model, data, task, bm_model, perception, callbacks)
                   - model: MuJoCo MjModel对象（物理模型）
                   - data: MuJoCo MjData对象（仿真数据/状态）
                   - task: 任务模型实例（如ReachTask）
                   - bm_model: 生物力学模型实例（如HumanArm）
                   - perception: 感知模块管理器实例
                   - callbacks: 回调函数字典（如课程学习）
        """
        # 1. 加载任务模型类并初始化参数
        task_cls = cls.get_class("tasks", config["simulation"]["task"]["cls"])
        task_kwargs = config["simulation"]["task"].get("kwargs", {})

        # 2. 加载生物力学模型类并初始化参数
        bm_cls = cls.get_class("bm_models", config["simulation"]["bm_model"]["cls"])
        bm_kwargs = config["simulation"]["bm_model"].get("kwargs", {})

        # 3. 初始化感知模块配置（存储{模块类: 模块参数}）
        perception_modules = {}
        for module_cfg in config["simulation"].get("perception_modules", []):
            module_cls = cls.get_class("perception", module_cfg["cls"])
            module_kwargs = module_cfg.get("kwargs", {})
            perception_modules[module_cls] = module_kwargs

        # 4. 加载MuJoCo模型（优先XML，二进制模型存在兼容性问题）
        simulation_file = os.path.join(simulator_folder, config["package_name"], "simulation")
        # 注释：二进制模型加载更快，但跨机器易出错，暂时禁用
        # try:
        #     model = mujoco.MjModel.from_binary_path(simulation_file + ".mjcf")
        # except:
        model = mujoco.MjModel.from_xml_path(simulation_file + ".xml")

        # 5. 初始化MuJoCo数据（存储仿真状态：位置/速度/力等）
        data = mujoco.MjData(model)

        # 6. 计算帧跳过数（frame_skip）和仿真步长（dt）
        # frame_skip：每次step调用多少次mj_step（平衡仿真精度与速度）
        run_parameters["frame_skip"] = int(1 / (model.opt.timestep * run_parameters["action_sample_freq"]))
        # dt：RL step的实际时间长度（= frame_skip * MuJoCo步长）
        run_parameters["dt"] = model.opt.timestep * run_parameters["frame_skip"]

        # 7. 初始化渲染上下文（为视觉模块提供渲染环境）
        run_parameters["rendering_context"] = Context(
            model,
            max_resolution=run_parameters.get("max_resolution", [1280, 960])  # 最大渲染分辨率
        )

        # 8. 初始化回调函数（如课程学习：逐步增加任务难度）
        callbacks = {}
        for cb in run_parameters.get("callbacks", []):
            callbacks[cb["name"]] = cls.get_class(cb["cls"])(cb["name"], **cb["kwargs"])

        # 9. 实例化核心模块（合并参数：任务参数+回调+运行参数）
        task = task_cls(model, data, **{**task_kwargs, **callbacks, **run_parameters})
        bm_model = bm_cls(model, data, **{**bm_kwargs, **callbacks, **run_parameters})
        # 初始化感知模块管理器（统一管理所有感知组件）
        perception = Perception(model, data, bm_model, perception_modules, {**callbacks, **run_parameters})

        return model, data, task, bm_model, perception, callbacks

    @classmethod
    def get(cls, simulator_folder, render_mode="rgb_array", render_mode_perception="embed", render_show_depths=False, run_parameters=None, use_cloned=True):
        """
        获取已构建的仿真器实例（核心入口方法）
        Args:
            simulator_folder: 仿真包路径
            render_mode: 渲染模式：
                         - "rgb_array": 返回RGB数组
                         - "rgb_array_list": 返回RGB数组列表
                         - "human": 弹出Pygame窗口实时显示
            render_mode_perception: 感知模块画面展示方式：
                                    - "embed": 嵌入主画面（画中画）
                                    - "separate": 单独存储
                                    - None: 不展示（用于Unity编辑器调试）
            render_show_depths: 是否显示深度图（转为热力图）
            run_parameters: 运行时覆盖的参数（优先级高于配置文件）
            use_cloned: 是否使用克隆后的包文件（True=用仿真包内的文件，False=用原文件，调试用）
        Returns:
            Simulator: 仿真器实例
        Raises:
            FileNotFoundError: 配置文件不存在
            RuntimeError: 仿真包未构建（无built字段）
            RuntimeError: 版本不兼容（主版本不一致）
        """
        # 1. 加载仿真包配置
        config_file = os.path.join(simulator_folder, "config.yaml")
        try:
            config = parse_yaml(config_file)
        except:
            raise FileNotFoundError(f"无法打开配置文件: {config_file}")

        # 2. 校验仿真包是否已构建
        if "built" not in config:
            raise RuntimeError("仿真包未构建！请先调用Simulator.build()")

        # 3. 将仿真包路径加入Python路径（确保能导入包内模块）
        if simulator_folder not in sys.path:
            sys.path.insert(0, simulator_folder)

        # 4. 导入仿真包内的Simulator类（支持版本兼容）
        gen_cls_cloned = getattr(importlib.import_module(config["package_name"]), "Simulator")
        if hasattr(gen_cls_cloned, "version"):
            _legacy_mode = False  # 新版（带version属性）
            gen_cls_cloned_version = gen_cls_cloned.version.split("-v")[-1]
        else:
            _legacy_mode = True   # 旧版（兼容逻辑）
            gen_cls_cloned_version = gen_cls_cloned.id.split("-v")[-1]

        # 5. 选择使用克隆后的类还是原类（调试用）
        if use_cloned:
            gen_cls = gen_cls_cloned
        else:
            gen_cls = cls
            gen_cls_version = gen_cls.version.split("-v")[-1]
            # 版本兼容性校验
            # 主版本不一致：强制报错（不兼容）
            if gen_cls_version.split(".")[0] > gen_cls_cloned_version.split(".")[0]:
                raise RuntimeError(
                    f"严重版本不兼容！\n"
                    f"仿真包版本: {gen_cls_cloned_version}, 当前uitb版本: {gen_cls_version}\n"
                    f"请设置use_cloned=True使用仿真包内的版本"
                )
            # 次版本不一致：仅警告（兼容）
            elif gen_cls_version.split(".")[1] > gen_cls_cloned_version.split(".")[1]:
                print(
                    f"警告：版本不匹配！\n"
                    f"仿真包版本: {gen_cls_cloned_version}, 当前uitb版本: {gen_cls_version}\n"
                    f"请设置use_cloned=True使用仿真包内的版本"
                )

        # 6. 实例化仿真器（兼容新旧版参数）
        if _legacy_mode:
            _simulator = gen_cls(simulator_folder, run_parameters=run_parameters)
        else:
            try:
                # 新版：支持完整渲染参数
                _simulator = gen_cls(
                    simulator_folder,
                    render_mode=render_mode,
                    render_mode_perception=render_mode_perception,
                    render_show_depths=render_show_depths,
                    run_parameters=run_parameters
                )
            except TypeError:
                # 兼容无render_mode_perception参数的版本
                _simulator = gen_cls(
                    simulator_folder,
                    render_mode=render_mode,
                    render_show_depths=render_show_depths,
                    run_parameters=run_parameters
                )

        return _simulator

    def __init__(self, simulator_folder, render_mode="rgb_array", render_mode_perception="embed", render_show_depths=False, run_parameters=None):
        """
        仿真器实例初始化（内部调用，用户应使用Simulator.get()）
        Args: 同Simulator.get()，略
        """
        # 1. 校验仿真包路径存在
        if not os.path.exists(simulator_folder):
            raise FileNotFoundError(f"仿真包路径不存在: {simulator_folder}")
        self._simulator_folder = simulator_folder

        # 2. 加载配置文件
        self._config = parse_yaml(os.path.join(self._simulator_folder, "config.yaml"))

        # 3. 合并运行参数（配置文件参数 + 运行时覆盖参数）
        self._run_parameters = self._config["simulation"]["run_parameters"].copy()
        self._run_parameters.update(run_parameters or {})

        # 4. 初始化核心组件（MuJoCo模型/数据、各模块）
        self._model, self._data, self.task, self.bm_model, self.perception, self.callbacks = \
            self._initialise(self._config, self._simulator_folder, self._run_parameters)

        # 5. 初始化动作空间（所有执行器：BM模型+感知模块）
        self.action_space = self._initialise_action_space()

        # 6. 初始化观测空间（感知模块输出 + 任务状态信息）
        self.observation_space = self._initialise_observation_space()

        # 7. 初始化回合统计（记录当前回合的时间/步数/奖励）
        self._episode_statistics = {
            "length (seconds)": 0,
            "length (steps)": 0,
            "reward": 0
        }

        # 8. 初始化GUI相机（用于主画面渲染）
        self._GUI_camera = Camera(
            self._run_parameters["rendering_context"],
            self._model,
            self._data,
            camera_id='for_testing',  # 相机ID（对应MuJoCo XML中的camera节点）
            dt=self._run_parameters["dt"]
        )

        # 9. 渲染相关参数初始化
        self._render_mode = render_mode  # 渲染模式
        self._render_mode_perception = render_mode_perception  # 感知画面展示方式
        self._render_show_depths = render_show_depths  # 是否显示深度图
        self._render_stack = []  # 渲染帧栈（rgb_array_list模式用）
        self._render_stack_perception = defaultdict(list)  # 感知模块帧栈（separate模式用）
        self._render_stack_pop = True  # 调用render()后清空帧栈
        self._render_stack_clean_at_reset = True  # reset时清空帧栈
        self._render_screen_size = None  # Pygame窗口尺寸（human模式用）
        self._render_window = None  # Pygame窗口对象（human模式用）
        self._render_clock = None  # Pygame时钟（控制帧率）

    def _initialise_action_space(self):
        """
        初始化动作空间（gym.spaces.Box）
        动作维度 = 生物力学模型执行器数 + 感知模块执行器数（如眼球转动）
        动作范围：所有执行器默认[-1, 1]（标准化，便于RL训练）
        Returns:
            spaces.Box: 动作空间对象
        """
        # 总执行器数
        num_actuators = self.bm_model.nu + self.perception.nu
        # 构建动作上下界（num_actuators × 2的数组，每行为[-1, 1]）
        actuator_limits = np.ones((num_actuators, 2)) * np.array([-1.0, 1.0])
        # 返回Box空间（float32类型，兼容大多数RL框架）
        return spaces.Box(
            low=np.float32(actuator_limits[:, 0]),
            high=np.float32(actuator_limits[:, 1])
        )

    def _initialise_observation_space(self):
        """
        初始化观测空间（gym.spaces.Dict）
        观测空间结构：
        {
            "视觉模块名": Box(视觉观测维度),
            "触觉模块名": Box(触觉观测维度),
            "stateful_information": Box(任务状态维度)  # 如目标位置/自身姿态
        }
        Returns:
            spaces.Dict: 观测空间对象
        """
        # 先获取一次观测样例，确定各维度
        observation = self.get_observation()
        obs_dict = dict()
        # 为每个感知模块初始化观测空间
        for module in self.perception.perception_modules:
            obs_dict[module.modality] = spaces.Box(
                dtype=np.float32,
                **module.get_observation_space_params()  # 模块返回自身的观测维度/上下界
            )
        # 添加任务状态信息（如目标位置、自身关节角度等）
        if "stateful_information" in observation:
            obs_dict["stateful_information"] = spaces.Box(
                dtype=np.float32,
                **self.task.get_stateful_information_space_params()
            )
        return spaces.Dict(obs_dict)

    def step(self, action):
        """
        核心step方法（RL训练的核心循环单元）
        执行逻辑：动作→物理仿真→模块更新→奖励计算→观测生成
        Args:
            action: 动作数组（来自RL策略，范围[-1, 1]）
        Returns:
            tuple: (obs, reward, terminated, truncated, info)
                   - obs: 观测字典（感知模块输出+任务状态）
                   - reward: 即时奖励
                   - terminated: 回合是否完成（如到达目标）
                   - truncated: 回合是否截断（如超时/超出范围）
                   - info: 附加信息（如努力成本、Unity图像等）
        """
        # 1. 设置生物力学模型的控制信号（如关节力矩）
        self.bm_model.set_ctrl(self._model, self._data, action[:self.bm_model.nu])

        # 2. 设置感知模块的控制信号（如眼球转动）
        self.perception.set_ctrl(self._model, self._data, action[self.bm_model.nu:])

        # 3. 执行MuJoCo仿真步（frame_skip次）
        mujoco.mj_step(self._model, self._data, nstep=self._run_parameters["frame_skip"])

        # 4. 更新生物力学模型（如约束检查、肌肉激活计算）
        self.bm_model.update(self._model, self._data)

        # 5. 更新感知模块（如视觉数据采集、预处理）
        self.perception.update(self._model, self._data)

        # 6. 更新任务状态，获取奖励和终止信号
        reward, terminated, truncated, info = self.task.update(self._model, self._data)

        # 7. 增加努力成本（惩罚过大的动作，鼓励节能）
        effort_cost = self.bm_model.get_effort_cost(self._model, self._data)
        info["EffortCost"] = effort_cost  # 记录到info中
        reward -= effort_cost  # 奖励 = 任务奖励 - 努力成本

        # 8. 生成观测（感知模块输出 + 任务状态）
        obs = self.get_observation(info)

        # 9. 渲染处理（根据渲染模式存储/显示帧）
        if self._render_mode == "rgb_array_list":
            # 存储主画面帧到栈
            self._render_stack.append(self._GUI_rendering())
        elif self._render_mode == "human":
            # 实时显示到Pygame窗口
            self._GUI_rendering_pygame()

        # 更新回合统计
        self._episode_statistics["length (seconds)"] += self._run_parameters["dt"]
        self._episode_statistics["length (steps)"] += 1
        self._episode_statistics["reward"] += reward

        return obs, reward, terminated, truncated, info

    def get_observation(self, info=None):
        """
        生成观测字典（整合感知模块和任务状态）
        Args:
            info: 附加信息（如Unity图像、努力成本等）
        Returns:
            dict: 观测字典
        """
        # 1. 获取感知模块的观测（如视觉/触觉数据）
        observation = self.perception.get_observation(self._model, self._data, info)

        # 2. 添加任务状态信息（如目标位置、自身关节角度）
        stateful_information = self.task.get_stateful_information(self._model, self._data)
        # 注：空数组会导致SB3报错，因此仅当有数据时添加
        if stateful_information.size > 0:
            observation["stateful_information"] = stateful_information

        return observation

    def reset(self, seed=None):
        """
        重置仿真环境（新回合开始）
        Args:
            seed: 随机种子（保证复现性）
        Returns:
            tuple: (obs, info) → 重置后的初始观测和附加信息
        """
        # 调用父类reset（设置随机种子）
        super().reset(seed=seed)
        
        # 1. 重置MuJoCo数据（位置/速度等恢复初始状态）
        mujoco.mj_resetData(self._model, self._data)

        # 2. 重置所有模块
        self.bm_model.reset(self._model, self._data)  # 重置生物力学模型
        self.perception.reset(self._model, self._data)  # 重置感知模块
        info = self.task.reset(self._model, self._data)  # 重置任务模型

        # 3. 执行一次mj_forward（更新物理状态，确保初始状态正确）
        mujoco.mj_forward(self._model, self._data)

        # 4. 重置渲染帧栈
        if self._render_mode == "rgb_array_list":
            if self._render_stack_clean_at_reset:
                self._render_stack = []  # 清空主画面栈
                self._render_stack_perception = defaultdict(list)  # 清空感知帧栈
            # 存储初始帧
            self._render_stack.append(self._GUI_rendering())
        elif self._render_mode == "human":
            # 显示初始帧到Pygame窗口
            self._GUI_rendering_pygame()

        # 重置回合统计
        self._episode_statistics = {
            "length (seconds)": 0,
            "length (steps)": 0,
            "reward": 0
        }

        # 返回初始观测和信息
        return self.get_observation(), info

    def render(self):
        """
        渲染方法（返回渲染结果或显示窗口）
        Returns:
            None/list/np.ndarray: 依渲染模式返回对应结果
        """
        if self._render_mode == "rgb_array_list":
            # 返回帧栈并清空（若开启pop）
            render_stack = self._render_stack
            if self._render_stack_pop:
                self._render_stack = []
            return render_stack
        elif self._render_mode == "rgb_array":
            # 返回当前主画面帧
            return self._GUI_rendering()
        else:
            # human模式：已在step/reset中实时显示，返回None
            return None
    
    def get_render_stack_perception(self):
        """
        获取感知模块的渲染帧栈（separate模式用）
        Returns:
            defaultdict(list): 键为"模块名/相机类型"，值为帧列表
        """
        render_stack_perception = self._render_stack_perception
        # 注释：可选清空逻辑，根据需求开启
        # if self._render_stack_pop:
        #     self._render_stack_perception = defaultdict(list)
        return render_stack_perception

    def _GUI_rendering(self):
        """
        内部渲染方法：生成主画面（含感知模块画中画）
        Returns:
            np.ndarray: 主画面RGB数组（H×W×3）
        """
        # 1. 获取主相机画面
        img, _ = self._GUI_camera.render()

        # 2. 处理感知模块画面嵌入（画中画）
        if self._render_mode_perception == "embed":
            # 收集所有感知模块的有效画面（RGB/深度图）
            perception_camera_images = [
                rgb_or_depth_array 
                for camera in self.perception.cameras
                for rgb_or_depth_array in camera.render() 
                if rgb_or_depth_array is not None
            ]
        
            # 有感知画面时才处理嵌入
            if len(perception_camera_images) > 0:
                _img_size = img.shape[:2]  # 主画面尺寸 (H, W)

                # 计算感知画面的目标尺寸（垂直均分主画面高度，宽度为20%主画面宽度）
                _desired_subwindow_height = np.round(_img_size[0] / len(perception_camera_images)).astype(int)
                _maximum_subwindow_width = np.round(0.2 * _img_size[1]).astype(int)

                perception_camera_images_resampled = []
                for ocular_img in perception_camera_images:
                    # 处理深度图：转为热力图（2D→3D RGB）
                    if ocular_img.ndim == 2:
                        if self._render_show_depths:
                            # 深度图转Jet热力图
                            ocular_img = matplotlib.pyplot.imshow(
                                ocular_img, 
                                cmap=matplotlib.pyplot.cm.jet, 
                                interpolation='bicubic'
                            ).make_image('TkAgg', unsampled=True)[0][..., :3]
                            matplotlib.pyplot.close()  # 关闭临时绘图窗口，避免内存泄漏
                        else:
                            # 不显示深度图则跳过
                            continue

                    # 计算缩放因子（保持宽高比，不超出目标尺寸）
                    resample_factor = min(
                        _desired_subwindow_height / ocular_img.shape[0],
                        _maximum_subwindow_width / ocular_img.shape[1]
                    )

                    # 计算缩放后的尺寸
                    resample_height = np.round(ocular_img.shape[0] * resample_factor).astype(int)
                    resample_width = np.round(ocular_img.shape[1] * resample_factor).astype(int)
                    # 初始化缩放后的图像数组
                    resampled_img = np.zeros((resample_height, resample_width, ocular_img.shape[2]), dtype=np.uint8)
                    # 逐通道缩放（保持色彩正确）
                    for channel in range(ocular_img.shape[2]):
                        resampled_img[:, :, channel] = scipy.ndimage.zoom(
                            ocular_img[:, :, channel], 
                            resample_factor, 
                            order=0  # 0阶插值（最近邻），速度快
                        )

                    perception_camera_images_resampled.append(resampled_img)

                # 将感知画面嵌入主画面右下角（垂直排列）
                ocular_img_bottom = _img_size[0]  # 起始Y坐标（主画面底部）
                for ocular_img_idx, ocular_img in enumerate(perception_camera_images_resampled):
                    # 计算嵌入位置（右下角）
                    y_start = ocular_img_bottom - ocular_img.shape[0]
                    y_end = ocular_img_bottom
                    x_start = _img_size[1] - ocular_img.shape[1]
                    x_end = _img_size[1]
                    # 嵌入画面
                    img[y_start:y_end, x_start:x_end] = ocular_img
                    # 更新下一个画面的Y坐标
                    ocular_img_bottom -= ocular_img.shape[0]
        
        # 3. 处理感知模块画面单独存储（separate模式）
        elif self._render_mode_perception == "separate":
            for module, camera_list in self.perception.cameras_dict.items():
                for camera in camera_list:
                    for rgb_or_depth_array in camera.render():
                        if rgb_or_depth_array is not None:
                            # 存储格式："模块名/相机类型" → 帧列表
                            self._render_stack_perception[f"{module.modality}/{type(camera).__name__}"].append(rgb_or_depth_array)

        return img

    def _GUI_rendering_pygame(self):
        """
        内部方法：Pygame窗口渲染（human模式）
        处理图像格式转换、窗口创建、帧率控制
        """
        # 1. 获取主画面并转换格式（MuJoCo: H×W×3 → Pygame: W×H×3）
        rgb_array = np.transpose(self._GUI_rendering(), axes=(1, 0, 2))

        # 2. 初始化窗口尺寸（首次调用时）
        if self._render_screen_size is None:
            self._render_screen_size = rgb_array.shape[:2]

        # 校验画面尺寸是否匹配（避免Pygame报错）
        assert self._render_screen_size == rgb_array.shape[:2], \
            f"期望画面尺寸: {self._render_screen_size}, 实际: {rgb_array.shape[:2]}"

        # 3. 初始化Pygame窗口（首次调用时）
        if self._render_window is None:
            pygame.init()
            pygame.display.init()
            self._render_window = pygame.display.set_mode(self._render_screen_size)

        # 4. 初始化Pygame时钟（控制帧率）
        if self._render_clock is None:
            self._render_clock = pygame.time.Clock()

        # 5. 将numpy数组转换为Pygame表面并显示
        surf = pygame.surfarray.make_surface(rgb_array)
        self._render_window.blit(surf, (0, 0))  # 绘制到窗口
        pygame.event.pump()  # 处理窗口事件（如关闭）
        self._render_clock.tick(self.fps)  # 控制帧率（与相机帧率一致）
        pygame.display.flip()  # 更新窗口显示

    def close(self):
        """
        关闭仿真环境（清理资源）
        关闭Pygame窗口、释放各模块资源
        """
        super().close()
        # 关闭Pygame窗口
        if self._render_window is not None:
            import pygame
            pygame.display.quit()
            pygame.quit()

    @property
    def fps(self):
        """
        获取渲染帧率（与GUI相机帧率一致）
        Returns:
            float: 帧率（FPS）
        """
        return self._GUI_camera._fps

    def callback(self, callback_name, num_timesteps):
        """
        调用指定回调函数（如课程学习）
        Args:
            callback_name: 回调函数名
            num_timesteps: 当前训练步数
        """
        self.callbacks[callback_name].update(num_timesteps)

    def update_callbacks(self, num_timesteps):
        """
        调用所有回调函数（批量更新）
        Args:
            num_timesteps: 当前训练步数
        """
        for callback_name in self.callbacks:
            self.callback(callback_name, num_timesteps)

    @property
    def config(self):
        """
        获取配置字典（深拷贝，避免外部修改）
        Returns:
            dict: 完整配置字典
        """
        return copy.deepcopy(self._config)

    @property
    def run_parameters(self):
        """
        获取运行参数（深拷贝，Context对象除外）
        Returns:
            dict: 运行参数字典
        """
        # Context对象无法深拷贝，单独处理
        exclude = {"rendering_context"}
        run_params = {
            k: copy.deepcopy(self._run_parameters[k]) 
            for k in self._run_parameters.keys() - exclude
        }
        run_params["rendering_context"] = self._run_parameters["rendering_context"]
        return run_params

    @property
    def simulator_folder(self):
        """
        获取仿真包路径
        Returns:
            str: 路径字符串
        """
        return self._simulator_folder

    @property
    def render_mode(self):
        """
        获取当前渲染模式
        Returns:
            str: 渲染模式（rgb_array/rgb_array_list/human）
        """
        return self._render_mode

    def get_state(self):
        """
        获取完整的仿真状态（用于日志/评估，非RL观测）
        包含MuJoCo核心状态 + 各模块状态
        Returns:
            dict: 状态字典
        """
        # 1. MuJoCo核心状态
        state = {
            "timestep": self._data.time,          # 当前仿真时间
            "qpos": self._data.qpos.copy(),       # 关节位置
            "qvel": self._data.qvel.copy(),       # 关节速度
            "qacc": self._data.qacc.copy(),       # 关节加速度
            "act_force": self._data.actuator_force.copy(),  # 执行器力
            "act": self._data.act.copy(),         # 执行器激活值
            "ctrl": self._data.ctrl.copy()        # 执行器控制信号
        }

        # 2. 任务模型状态
        state.update(self.task.get_state(self._model, self._data))

        # 3. 生物力学模型状态
        state.update(self.bm_model.get_state(self._model, self._data))

        # 4. 感知模块状态
        state.update(self.perception.get_state(self._model, self._data))

        return state

    def close(self, **kwargs):
        """
        重载close方法：清理所有模块资源
        Args:
            **kwargs: 模块特定的清理参数
        """
        # 调用各模块的close方法
        self.task.close(**kwargs)
        self.perception.close(**kwargs)
        self.bm_model.close(**kwargs)

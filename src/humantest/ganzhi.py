import numpy as np
import cv2
from typing import Dict, Tuple, Any, Optional, List
import torch
from torchvision import models, transforms
from torch.backends import cudnn

class SensorInterface:
    """模拟传感器采集接口"""
    def __init__(self, sim_env):
        self.sim_env = sim_env
        self.joint_names = ["left_hip", "right_hip", "left_knee", "right_knee"]
    
    def get_rgb_image(self) -> np.ndarray:
        """获取RGB图像（640x480x3）"""
        try:
            img = self.sim_env.render(camera_name="head_cam", width=640, height=480)
            if img is None or img.shape != (480, 640, 3):
                raise ValueError(f"获取的图像格式不正确，期望(480, 640, 3)，实际{img.shape if img is not None else None}")
            return img
        except Exception as e:
            print(f"获取RGB图像失败: {str(e)}")
            return np.zeros((480, 640, 3), dtype=np.uint8)
    
    def get_joint_states(self) -> Dict[str, Tuple[float, float]]:
        """获取关节状态：键为关节名，值为（角度，角速度）"""
        states = {}
        try:
            data = self.sim_env.data()
            if data is None:
                raise ValueError("无法获取仿真环境数据")
                
            for name in self.joint_names:
                try:
                    joint = data.joint(name)
                    angle = float(joint.qpos[0])
                    vel = float(joint.qvel[0])
                    states[name] = (angle, vel)
                except Exception as e:
                    print(f"获取关节 {name} 状态失败: {str(e)}")
                    states[name] = (0.0, 0.0)
        except Exception as e:
            print(f"获取关节状态失败: {str(e)}")
            for name in self.joint_names:
                states[name] = (0.0, 0.0)
                
        return states


class Preprocessor:
    """预处理模块：处理原始数据噪声与格式"""
    @staticmethod
    def process_image(img: np.ndarray, dtype=np.float32) -> np.ndarray:
        """图像预处理：去噪+标准化，可指定输出数据类型"""
        if img.dtype != np.uint8:
            img = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            
        # 高斯模糊去噪
        img_denoised = cv2.GaussianBlur(img, (5, 5), 0)
        # 归一化到[0,1]并转换为指定类型
        processed_img = img_denoised.astype(dtype) / 255.0
        
        # 调试：打印处理后的图像数据类型
        print(f"预处理后图像数据类型: {processed_img.dtype}")
        return processed_img
    
    @staticmethod
    def process_joint_states(
        joint_states: Dict[str, Tuple[float, float]], 
        max_angle: float = np.pi
    ) -> Dict[str, Tuple[float, float]]:
        """关节状态预处理：角度归一化到[-1,1]"""
        processed = {}
        for name, (angle, vel) in joint_states.items():
            angle_norm = np.clip(angle / max_angle, -1.0, 1.0)
            vel_norm = np.clip(vel / 10.0, -1.0, 1.0)
            processed[name] = (float(angle_norm), float(vel_norm))
        return processed


class FeatureExtractor:
    """特征提取模块：从预处理数据中提取决策所需信息"""
    def __init__(self, device: Optional[str] = None, use_double: bool = True):
        # 自动选择设备（GPU如果可用）
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        # 强制使用double类型来匹配错误提示
        self.use_double = use_double  # 现在默认设为True，因为错误提示需要Double
        self.dtype = torch.float64 if self.use_double else torch.float32
        
        print(f"使用设备: {self.device}, 数据类型: {self.dtype}")
        
        # 加载预训练目标检测模型
        self.object_detector = models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
        
        # 关键修复：强制将所有模型参数转换为指定类型
        self.object_detector = self.object_detector.to(self.device)
        self.object_detector = self.object_detector.type(self.dtype)
        
        # 验证模型参数类型
        param = next(self.object_detector.parameters())
        print(f"模型参数类型: {param.dtype}")
        
        self.object_detector.eval()
        
        # 启用cudnn加速（如果使用GPU）
        if self.device == "cuda":
            cudnn.benchmark = True
        
        # 图像预处理转换
        self.img_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # COCO数据集标签映射
        self.label_map = {
            1: "person", 2: "bicycle", 3: "car", 4: "motorcycle", 5: "bus",
            6: "train", 7: "truck", 64: "book", 77: "cell phone"
        }
    
    def detect_objects(self, img: np.ndarray) -> Dict[str, Tuple[float, float, float]]:
        """检测图像中的物体，返回{物体名: (中心x, 中心y, 置信度)}"""
        try:
            # 显式转换图像数据类型
            img = img.astype(np.float64) if self.use_double else img.astype(np.float32)
            print(f"输入图像转换后的数据类型: {img.dtype}")
            
            # 转换为张量并确保类型匹配
            img_tensor = self.img_transform(img)
            print(f"转换为张量后的初始类型: {img_tensor.dtype}")
            
            # 强制转换为所需类型
            img_tensor = img_tensor.type(self.dtype)
            img_tensor = img_tensor.unsqueeze(0).to(self.device)
            print(f"输入模型的张量类型: {img_tensor.dtype}")
            
            with torch.no_grad():
                predictions = self.object_detector(img_tensor)[0]
            
            objects = {}
            # 筛选置信度>0.5的目标
            for label, score, box in zip(
                predictions["labels"], 
                predictions["scores"], 
                predictions["boxes"]
            ):
                if score > 0.5:
                    obj_name = self._get_label_name(int(label))
                    x_center = float((box[0] + box[2]) / 2 / img.shape[1])
                    y_center = float((box[1] + box[3]) / 2 / img.shape[0])
                    objects[obj_name] = (x_center, y_center, float(score))
            
            return objects
        except Exception as e:
            print(f"物体检测失败: {str(e)}")
            return {}
    
    def estimate_self_pose(self, joint_states: Dict[str, Tuple[float, float]]) -> float:
        """从关节状态估计自身姿态"""
        try:
            knee_angles = [
                joint_states["left_knee"][0], 
                joint_states["right_knee"][0]
            ]
            return float(np.mean(knee_angles))
        except Exception as e:
            print(f"自身姿态估计失败: {str(e)}")
            return 0.0
    
    def _get_label_name(self, label: int) -> str:
        """将COCO标签映射到物体名"""
        return self.label_map.get(label, f"object_{label}")


class PerceptionFusion:
    """状态融合模块：整合多模态特征，输出环境状态"""
    def __init__(self):
        self.front_region_x = (0.4, 0.6)
        self.near_region_y = (0.6, 1.0)
    
    def fuse(
        self, 
        objects: Dict[str, Tuple[float, float, float]], 
        self_pose: float
    ) -> Dict[str, Any]:
        """融合物体检测与自身姿态，输出最终感知结果"""
        front_objects = {
            name: (x, y, s) for name, (x, y, s) in objects.items() 
            if self.front_region_x[0] <= x <= self.front_region_x[1]
        }
        
        near_objects = {
            name: (x, y, s) for name, (x, y, s) in objects.items() 
            if self.near_region_y[0] <= y <= self.near_region_y[1]
        }
        
        critical_objects = {
            name: (x, y, s) for name, (x, y, s) in objects.items()
            if name in ["person", "car", "bus", "truck"]
        }
        
        return {
            "self_pose": "standing" if self_pose > 0 else "bending",
            "self_pose_value": self_pose,
            "front_objects": front_objects,
            "near_objects": near_objects,
            "critical_objects": critical_objects,
            "object_count": len(objects),
            "has_obstacle": len(front_objects) > 0 or len(near_objects) > 0
        }


class EmbodiedPerception:
    """具身人感知模块主类"""
    def __init__(self, sim_env, device: Optional[str] = None, use_double: bool = True):
        # 默认为True，因为错误提示需要Double类型
        self.use_double = use_double
        self.sensor = SensorInterface(sim_env)
        self.preprocessor = Preprocessor()
        self.feature_extractor = FeatureExtractor(device, use_double)
        self.fusion = PerceptionFusion()
    
    def perceive(self) -> Dict[str, Any]:
        """完整感知流程：采集→预处理→特征提取→融合"""
        try:
            # 1. 数据采集
            img = self.sensor.get_rgb_image()
            joint_states = self.sensor.get_joint_states()
            
            # 2. 预处理 - 确保图像数据类型与模型一致
            img_dtype = np.float64 if self.use_double else np.float32
            img_processed = self.preprocessor.process_image(img, dtype=img_dtype)
            joints_processed = self.preprocessor.process_joint_states(joint_states)
            
            # 3. 特征提取
            objects = self.feature_extractor.detect_objects(img_processed)
            self_pose = self.feature_extractor.estimate_self_pose(joints_processed)
            
            # 4. 融合输出
            return self.fusion.fuse(objects, self_pose)
        except Exception as e:
            print(f"感知流程失败: {str(e)}")
            return {
                "self_pose": "unknown",
                "self_pose_value": 0.0,
                "front_objects": {},
                "near_objects": {},
                "critical_objects": {},
                "object_count": 0,
                "has_obstacle": False,
                "error": str(e)
            }


# 示例使用
if __name__ == "__main__":
    class MockSimEnv:
        def render(self, *args, **kwargs):
            return np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
        
        def data(self):
            class MockData:
                def joint(self, name):
                    class MockJoint:
                        qpos = [np.random.uniform(-np.pi, np.pi)]
                        qvel = [np.random.uniform(-5, 5)]
                    return MockJoint()
            return MockData()
    
    # 明确设置use_double=True以匹配错误提示中的预期类型
    sim_env = MockSimEnv()
    perception = EmbodiedPerception(sim_env, use_double=True)
    
    print("开始感知流程...")
    for i in range(3):
        env_state = perception.perceive()
        print(f"\n第{i+1}次感知结果：")
        print(f"自身姿态: {env_state['self_pose']}")
        print(f"检测到物体数量: {env_state['object_count']}")
        print(f"前方物体: {list(env_state['front_objects'].keys())}")
        print(f"是否有障碍物: {env_state['has_obstacle']}")
    
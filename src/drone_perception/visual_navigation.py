<<<<<<< HEAD
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import random
import time
import os

class DroneBattery:
    def __init__(self, max_capacity=100, current_charge=100):
        self.max_capacity = max_capacity
        self.current_charge = current_charge
        
    def display_battery_status(self):
        print(f"Battery Status: {self.current_charge}%")
        
    def is_battery_low(self):
        return self.current_charge < 20

# 与训练代码完全相同的模型结构
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        
        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except TypeError:
            self.backbone = models.resnet18(pretrained=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def detect_class_info(train_dir):
    """检测训练数据中的类别信息"""
    if not os.path.exists(train_dir):
        return 6, ['Animal', 'City', 'Fire', 'Forest', 'Vehicle', 'Water']
    
    classes = sorted([d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))])
    
    if not classes:
        return 6, ['Animal', 'City', 'Fire', 'Forest', 'Vehicle', 'Water']
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return len(classes), classes

def load_pytorch_model(model_path, train_dir):
    """加载PyTorch模型"""
    num_classes, class_names = detect_class_info(train_dir)
    
    try:
        model = ImageClassifier(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model, num_classes, class_names
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, 0, []

def preprocess_frame(frame):
    """图像预处理"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(pil_image).unsqueeze(0)

def predict_frame(model, frame, device, class_names):
    """预测函数"""
    try:
        input_tensor = preprocess_frame(frame)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
        
        if predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
        
        return predicted_class, confidence * 100
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "Unknown", 0.0

def decide_navigation(predicted_class):
    """导航决策"""
    navigation_rules = {
        'Fire': "🔥 Fire detected! Navigate away.",
        'Animal': "🦌 Animal ahead. Hovering.",
        'Forest': "🌲 Forest zone detected. Reduce speed.",
        'Water': "🌊 Water body detected. Maintain altitude and avoid descent.",
        'Vehicle': "🚗 Vehicle detected. Hover and wait.",
        'City': "🏙️ Urban area detected. Enable obstacle avoidance and slow navigation."
    }
    
    message = navigation_rules.get(predicted_class, f"✅ {predicted_class} detected. Continue normal navigation.")
    print(message)

def handle_low_battery(drone_battery):
    """处理低电量"""
    print("🔋 Low battery! Returning to base.")
    exit()

def run_visual_navigation():
    """运行视觉导航系统"""
    print("🚁 Starting the drone vision process with PyTorch model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./data/best_model.pth"
    TRAIN_DIR = "./data/train"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return
    
    model, num_classes, class_names = load_pytorch_model(MODEL_PATH, TRAIN_DIR)
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    model = model.to(device)
    
    # 视频源设置
    VIDEO_SOURCE = 0
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("❌ Failed to open video source.")
        return
    
    drone_battery = DroneBattery()
    fps_counter = 0
    fps_time = time.time()
    frame_count = 0
    predicted_class = "Unknown"
    confidence = 0.0
    
    print("\n🎮 控制说明:")
    print("- 按 'q' 键退出程序")
    print("- 按 'b' 键模拟电池放电")
    print("- 按 'c' 键显示电池状态")
    print("- 开始实时视觉导航...\n")
    
    while True:
        if drone_battery.is_battery_low():
            handle_low_battery(drone_battery)
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 5 == 0:
            predicted_class, confidence = predict_frame(model, frame, device, class_names)
            
            cv2.putText(frame, f"{predicted_class} ({confidence:.2f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            decide_navigation(predicted_class)
        
        cv2.imshow("Drone Vision Feed - PyTorch", frame)
        
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter / (time.time() - fps_time)
            print(f"📊 FPS: {fps:.1f} | 预测: {predicted_class} ({confidence:.1f}%)")
            fps_counter = 0
            fps_time = time.time()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            drone_battery.current_charge = 15
        elif key == ord('c'):
            drone_battery.display_battery_status()
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎯 无人机视觉导航系统已安全关闭")

=======
import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import random
import time
import os

class DroneBattery:
    def __init__(self, max_capacity=100, current_charge=100):
        self.max_capacity = max_capacity
        self.current_charge = current_charge
        
    def display_battery_status(self):
        print(f"Battery Status: {self.current_charge}%")
        
    def is_battery_low(self):
        return self.current_charge < 20

# 与训练代码完全相同的模型结构
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        
        try:
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except TypeError:
            self.backbone = models.resnet18(pretrained=True)
        
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

def detect_class_info(train_dir):
    """检测训练数据中的类别信息"""
    if not os.path.exists(train_dir):
        return 6, ['Animal', 'City', 'Fire', 'Forest', 'Vehicle', 'Water']
    
    classes = sorted([d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))])
    
    if not classes:
        return 6, ['Animal', 'City', 'Fire', 'Forest', 'Vehicle', 'Water']
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    return len(classes), classes

def load_pytorch_model(model_path, train_dir):
    """加载PyTorch模型"""
    num_classes, class_names = detect_class_info(train_dir)
    
    try:
        model = ImageClassifier(num_classes=num_classes)
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        model.eval()
        return model, num_classes, class_names
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, 0, []

def preprocess_frame(frame):
    """图像预处理"""
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    transform = transforms.Compose([
        transforms.Resize((128, 128)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(pil_image).unsqueeze(0)

def predict_frame(model, frame, device, class_names):
    """预测函数"""
    try:
        input_tensor = preprocess_frame(frame)
        input_tensor = input_tensor.to(device)
        
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
        
        if predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
        
        return predicted_class, confidence * 100
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "Unknown", 0.0

def decide_navigation(predicted_class):
    """导航决策"""
    navigation_rules = {
        'Fire': "🔥 Fire detected! Navigate away.",
        'Animal': "🦌 Animal ahead. Hovering.",
        'Forest': "🌲 Forest zone detected. Reduce speed.",
        'Water': "🌊 Water body detected. Maintain altitude and avoid descent.",
        'Vehicle': "🚗 Vehicle detected. Hover and wait.",
        'City': "🏙️ Urban area detected. Enable obstacle avoidance and slow navigation."
    }
    
    message = navigation_rules.get(predicted_class, f"✅ {predicted_class} detected. Continue normal navigation.")
    print(message)

def handle_low_battery(drone_battery):
    """处理低电量"""
    print("🔋 Low battery! Returning to base.")
    exit()

def run_visual_navigation():
    """运行视觉导航系统"""
    print("🚁 Starting the drone vision process with PyTorch model...")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    MODEL_PATH = "./data/best_model.pth"
    TRAIN_DIR = "./data/train"
    
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        return
    
    model, num_classes, class_names = load_pytorch_model(MODEL_PATH, TRAIN_DIR)
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    model = model.to(device)
    
    # 视频源设置
    VIDEO_SOURCE = 0
    cap = cv2.VideoCapture(VIDEO_SOURCE)
    
    if not cap.isOpened():
        print("❌ Failed to open video source.")
        return
    
    drone_battery = DroneBattery()
    fps_counter = 0
    fps_time = time.time()
    frame_count = 0
    predicted_class = "Unknown"
    confidence = 0.0
    
    print("\n🎮 控制说明:")
    print("- 按 'q' 键退出程序")
    print("- 按 'b' 键模拟电池放电")
    print("- 按 'c' 键显示电池状态")
    print("- 开始实时视觉导航...\n")
    
    while True:
        if drone_battery.is_battery_low():
            handle_low_battery(drone_battery)
        
        ret, frame = cap.read()
        if not ret:
            break
        
        frame_count += 1
        if frame_count % 5 == 0:
            predicted_class, confidence = predict_frame(model, frame, device, class_names)
            
            cv2.putText(frame, f"{predicted_class} ({confidence:.2f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            decide_navigation(predicted_class)
        
        cv2.imshow("Drone Vision Feed - PyTorch", frame)
        
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter / (time.time() - fps_time)
            print(f"📊 FPS: {fps:.1f} | 预测: {predicted_class} ({confidence:.1f}%)")
            fps_counter = 0
            fps_time = time.time()
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('b'):
            drone_battery.current_charge = 15
        elif key == ord('c'):
            drone_battery.display_battery_status()
    
    cap.release()
    cv2.destroyAllWindows()
    print("🎯 无人机视觉导航系统已安全关闭")

>>>>>>> 377e0f2ca50fbcc5b8db8bc7644a76799e99dd1b

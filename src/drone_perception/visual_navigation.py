import cv2
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
import random
import time
import os

# DroneBattery Class to manage battery
class DroneBattery:
    def __init__(self, max_capacity=100, current_charge=100):
        self.max_capacity = max_capacity
        self.current_charge = current_charge
        
    def display_battery_status(self):
        print(f"Battery Status: {self.current_charge}%")
        
    def charge_battery(self, charge_rate=10):
        while self.current_charge < self.max_capacity:
            self.current_charge += charge_rate
            if self.current_charge > self.max_capacity:
                self.current_charge = self.max_capacity
            print(f"Charging... {self.current_charge}%")
            time.sleep(1)
        print("Battery fully charged!")
        
    def discharge_battery(self, discharge_rate=10):
        while self.current_charge > 0:
            self.current_charge -= discharge_rate
            if self.current_charge < 0:
                self.current_charge = 0
            print(f"Discharging... {self.current_charge}%")
            time.sleep(1)
        print("Battery completely drained!")
    
    def is_battery_low(self):
        return self.current_charge < 20

# 与训练代码完全相同的模型结构
class ImageClassifier(nn.Module):
    def __init__(self, num_classes):
        super(ImageClassifier, self).__init__()
        
        # 使用与训练代码相同的模型结构
        try:
            # 新版本用法（torchvision >= 0.13）
            self.backbone = models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1)
        except TypeError:
            # 旧版本兼容（torchvision < 0.13）
            self.backbone = models.resnet18(pretrained=True)
        
        # 冻结预训练层的参数
        for param in self.backbone.parameters():
            param.requires_grad = False
        
        # 替换最后的全连接层（与训练代码完全一致）
        in_features = self.backbone.fc.in_features
        self.backbone.fc = nn.Sequential(
            nn.Linear(in_features, 128),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.backbone(x)

# 检测类别数量和类别映射
def detect_class_info():
    """检测训练数据中的类别信息"""
    train_dir = "./data/train"
    if not os.path.exists(train_dir):
        print(f"❌ 训练目录不存在: {train_dir}")
        return 6, ['Animal', 'City', 'Fire', 'Forest', 'Vehicle', 'Water']  # 默认值
    
    classes = sorted([d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))])
    
    if not classes:
        print(f"❌ 在 {train_dir} 中没有找到任何类别文件夹!")
        return 6, ['Animal', 'City', 'Fire', 'Forest', 'Vehicle', 'Water']  # 默认值
    
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print(f"✅ 检测到类别: {class_to_idx}")
    return len(classes), classes

# ✅ 加载PyTorch模型
def load_pytorch_model(model_path):
    print("📦 Loading PyTorch model...")
    
    # 检测类别信息
    num_classes, class_names = detect_class_info()
    print(f"🎯 模型配置: {num_classes} 个类别 - {class_names}")
    
    try:
        # 初始化模型结构（与训练时完全相同）
        model = ImageClassifier(num_classes=num_classes)
        
        # 加载训练好的权重
        model.load_state_dict(torch.load(model_path, map_location='cpu'))
        print("✅ Model loaded successfully!")
        
        # 设置为评估模式
        model.eval()
        return model, num_classes, class_names
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        return None, 0, []

def check_emergency():
    emergency_conditions = ['low_battery', 'emergency']
    return random.choice(emergency_conditions)

def handle_low_battery(drone_battery):
    print("🔋 Low battery! Returning to base.")
    drone_battery.charge_battery(charge_rate=15)
    exit()

# 图像预处理（与训练时的test_transform完全一致）
def preprocess_frame(frame):
    # 将OpenCV BGR图像转换为PIL RGB图像
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    pil_image = Image.fromarray(rgb_frame)
    
    # 使用与训练代码中test_transform完全相同的预处理
    transform = transforms.Compose([
        transforms.Resize((128, 128)),  # 与img_size一致
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    return transform(pil_image).unsqueeze(0)  # 添加batch维度

# 预测函数
def predict_frame(model, frame, device, class_names):
    try:
        # 预处理
        input_tensor = preprocess_frame(frame)
        input_tensor = input_tensor.to(device)
        
        # 预测
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs[0], dim=0)
            predicted_class_idx = torch.argmax(probabilities).item()
            confidence = probabilities[predicted_class_idx].item()
        
        # 获取类别名称
        if predicted_class_idx < len(class_names):
            predicted_class = class_names[predicted_class_idx]
        else:
            predicted_class = f"Class_{predicted_class_idx}"
        
        return predicted_class, confidence * 100
        
    except Exception as e:
        print(f"❌ Prediction error: {e}")
        return "Unknown", 0.0

def decide_navigation(predicted_class):
    if predicted_class == 'Fire':
        print("🔥 Fire detected! Navigate away.")
    elif predicted_class == 'Animal':
        print("🦌 Animal ahead. Hovering.")
    elif predicted_class == 'Forest':
        print("🌲 Forest zone detected. Reduce speed.")
    elif predicted_class == 'Water':
        print("🌊 Water body detected. Maintain altitude and avoid descent.")
    elif predicted_class == 'Vehicle':
        print("🚗 Vehicle detected. Hover and wait.")
    elif predicted_class == 'City':
        print("🏙️ Urban area detected. Enable obstacle avoidance and slow navigation.")
    else:
        print(f"✅ {predicted_class} detected. Continue normal navigation.")

def main():
    print("🚁 Starting the drone vision process with PyTorch model...")
    start_time = time.time()
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    
    # 加载模型（使用shijuedaohan.py训练好的模型）
    MODEL_PATH = "./data/best_model.pth"
    
    # 检查模型文件是否存在
    if not os.path.exists(MODEL_PATH):
        print(f"❌ 模型文件不存在: {MODEL_PATH}")
        print("请先运行 shijuedaohan.py 训练模型")
        return
    
    model, num_classes, class_names = load_pytorch_model(MODEL_PATH)
    
    if model is None:
        print("❌ Failed to load model. Exiting.")
        return
    
    model = model.to(device)
    print(f"🎯 模型加载完成: {num_classes} 个类别")
    print(f"📋 类别列表: {class_names}")
    
    # 视频源设置
    VIDEO_SOURCE = 0  # 使用本地摄像头，或者改为您的IP摄像头地址
    # VIDEO_SOURCE = "http://192.168.1.3:4747/video"

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 10)
    cap.set(cv2.CAP_PROP_FPS, 30)

    # 检查视频流
    if not cap.isOpened():
        print("❌ Failed to open video source.")
        return
    else:
        print("✅ Video source opened successfully.")

    drone_battery = DroneBattery()

    # FPS计算
    fps_counter = 0
    fps_time = time.time()
    frame_count = 0

    print("\n🎮 控制说明:")
    print("- 按 'q' 键退出程序")
    print("- 按 'b' 键模拟电池放电")
    print("- 按 'c' 键显示电池状态")
    print("- 开始实时视觉导航...\n")

    while True:
        # 检查电池状态
        if drone_battery.is_battery_low():
            handle_low_battery(drone_battery)

        # 超时检查
        elapsed_time = time.time() - start_time
        if elapsed_time > 300:  # 5分钟
            print("⏰ Timeout reached! Stopping the drone.")
            break

        # 读取帧
        ret, frame = cap.read()
        if not ret:
            print("❌ Failed to capture frame.")
            break

        # 每隔5帧进行一次预测以提升性能
        frame_count += 1
        if frame_count % 5 == 0:
            predicted_class, confidence = predict_frame(model, frame, device, class_names)
            
            # 显示结果
            cv2.putText(frame, f"{predicted_class} ({confidence:.2f}%)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # 导航决策
            decide_navigation(predicted_class)

        cv2.imshow("Drone Vision Feed - PyTorch", frame)

        # FPS计算和显示
        fps_counter += 1
        if time.time() - fps_time >= 1.0:
            fps = fps_counter / (time.time() - fps_time)
            print(f"📊 FPS: {fps:.1f} | 预测: {predicted_class} ({confidence:.1f}%)" if frame_count % 5 == 0 else f"📊 FPS: {fps:.1f}")
            fps_counter = 0
            fps_time = time.time()

        # 键盘控制
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            print("🛑 Manual stop initiated by the user.")
            break
        elif key == ord('b'):
            print("🔋 Simulating battery discharge...")
            drone_battery.current_charge = 15  # 设置为低电量状态
        elif key == ord('c'):
            drone_battery.display_battery_status()

    cap.release()
    cv2.destroyAllWindows()
    print("🎯 无人机视觉导航系统已安全关闭")

if __name__ == "__main__":

    main()

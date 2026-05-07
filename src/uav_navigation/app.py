#!/usr/bin/env python3
"""
🚁 无人机导航系统 - 完整演示版
无需真实数据，立即展示效果
"""

from flask import Flask, render_template, Response, jsonify, request, send_file
import cv2
import numpy as np
import threading
import time
import json
import os
import io
from datetime import datetime
from PIL import Image, ImageDraw, ImageFont
import random

app = Flask(__name__)


# ==================== 配置参数 ====================
class Config:
    """演示配置"""
    # 演示模式配置
    DEMO_MODE = True  # 演示模式，使用虚拟数据
    USE_VIRTUAL_CAMERA = True  # 使用虚拟摄像头

    # 类别配置
    CLASS_NAMES = ['森林 Forest', '火灾 Fire', '城市 City', '动物 Animal', '车辆 Vehicle', '水域 Water']
    CLASS_COLORS = {
        '森林 Forest': (0, 128, 0),
        '火灾 Fire': (255, 0, 0),
        '城市 City': (128, 128, 128),
        '动物 Animal': (255, 165, 0),
        '车辆 Vehicle': (255, 0, 255),
        '水域 Water': (0, 191, 255)
    }

    # 无人机状态
    DRONE_STATUS = {
        'battery': 100,
        'altitude': 0,
        'speed': 0,
        'location': {'x': 0, 'y': 0, 'z': 0},
        'mode': 'LANDED',  # LANDED, TAKEOFF, FLYING, LANDING
        'detected_class': '正在检测...',
        'confidence': 0,
        'timestamp': None,
        'temperature': 25,
        'wind_speed': 5,
        'gps_signal': '强'
    }


config = Config()


# ==================== 虚拟摄像头和检测系统 ====================
class VirtualCamera:
    """虚拟摄像头系统 - 生成模拟的无人机画面"""

    def __init__(self):
        self.frame_width = 640
        self.frame_height = 480
        self.frame_count = 0
        self.current_scene = '城市 City'
        self.scene_transition = 0
        self.scene_history = []

        # 场景切换概率
        self.scene_change_prob = 0.05

        # 创建虚拟场景图像
        self.scene_images = self.create_scene_images()

        print("🎥 虚拟摄像头初始化完成")
        print("📊 可检测场景:", ", ".join(config.CLASS_NAMES))

    def create_scene_images(self):
        """创建虚拟场景图像"""
        images = {}

        for scene in config.CLASS_NAMES:
            # 创建基础图像
            img = np.zeros((self.frame_height, self.frame_width, 3), dtype=np.uint8)

            # 根据场景类型设置不同颜色和模式
            if '森林' in scene:
                # 森林 - 绿色系
                img[:, :, 1] = random.randint(100, 200)  # 绿色
                # 添加树木纹理
                for i in range(20):
                    x = random.randint(50, self.frame_width - 50)
                    y = random.randint(50, self.frame_height - 50)
                    cv2.circle(img, (x, y), 15, (0, random.randint(150, 255), 0), -1)

            elif '火灾' in scene:
                # 火灾 - 红色系
                img[:, :, 2] = random.randint(150, 255)  # 红色
                img[:, :, 1] = random.randint(50, 150)  # 黄色
                # 添加火焰效果
                for i in range(30):
                    x = random.randint(50, self.frame_width - 50)
                    y = random.randint(50, self.frame_height - 50)
                    size = random.randint(5, 20)
                    cv2.circle(img, (x, y), size, (0, random.randint(100, 200), random.randint(200, 255)), -1)

            elif '城市' in scene:
                # 城市 - 灰色系
                gray = random.randint(100, 200)
                img[:, :, 0] = gray
                img[:, :, 1] = gray
                img[:, :, 2] = gray
                # 添加建筑
                for i in range(10):
                    x = random.randint(50, self.frame_width - 50)
                    width = random.randint(20, 60)
                    height = random.randint(40, 150)
                    cv2.rectangle(img, (x, self.frame_height - height),
                                  (x + width, self.frame_height),
                                  (gray + 20, gray + 20, gray + 20), -1)

            elif '动物' in scene:
                # 动物 - 棕色系
                img[:, :, 0] = random.randint(30, 60)  # 蓝色通道（棕色偏黄）
                img[:, :, 1] = random.randint(80, 120)  # 绿色通道
                img[:, :, 2] = random.randint(140, 180)  # 红色通道
                # 添加动物轮廓
                for i in range(5):
                    x = random.randint(50, self.frame_width - 50)
                    y = random.randint(50, self.frame_height - 50)
                    cv2.ellipse(img, (x, y), (30, 20), 0, 0, 360, (100, 70, 40), -1)

            elif '车辆' in scene:
                # 车辆 - 各种颜色
                img[:, :, 0] = random.randint(50, 100)  # 蓝色
                img[:, :, 1] = random.randint(50, 100)  # 绿色
                img[:, :, 2] = random.randint(50, 100)  # 红色
                # 添加车辆
                for i in range(8):
                    x = random.randint(50, self.frame_width - 50)
                    y = random.randint(100, self.frame_height - 50)
                    cv2.rectangle(img, (x - 25, y - 15), (x + 25, y + 15),
                                  (random.randint(100, 255), random.randint(100, 255), random.randint(100, 255)), -1)

            elif '水域' in scene:
                # 水域 - 蓝色系
                img[:, :, 0] = random.randint(150, 255)  # 蓝色
                # 添加波纹效果
                for i in range(15):
                    x = random.randint(50, self.frame_width - 50)
                    y = random.randint(50, self.frame_height - 50)
                    cv2.circle(img, (x, y), random.randint(10, 40),
                               (random.randint(100, 200), random.randint(100, 200), 255), 2)

            images[scene] = img

        return images

    def get_frame(self):
        """获取虚拟摄像头帧"""
        self.frame_count += 1

        # 随机切换场景（模拟摄像头移动）
        if random.random() < self.scene_change_prob and self.scene_transition == 0:
            self.scene_history.append(self.current_scene)
            if len(self.scene_history) > 5:
                self.scene_history.pop(0)

            # 随机选择新场景（排除当前场景）
            available_scenes = [s for s in config.CLASS_NAMES if s != self.current_scene]
            self.current_scene = random.choice(available_scenes)
            self.scene_transition = 30  # 30帧的过渡效果

        # 获取当前场景图像
        frame = self.scene_images[self.current_scene].copy()

        # 添加过渡效果
        if self.scene_transition > 0:
            old_scene = self.scene_history[-1] if self.scene_history else self.current_scene
            old_frame = self.scene_images[old_scene].copy()

            alpha = self.scene_transition / 30.0
            frame = cv2.addWeighted(frame, 1 - alpha, old_frame, alpha, 0)
            self.scene_transition -= 1

        # 模拟摄像头噪声
        noise = np.random.normal(0, 3, frame.shape).astype(np.uint8)
        frame = cv2.add(frame, noise)

        # 添加时间戳
        cv2.putText(frame, f"帧: {self.frame_count}", (10, 20),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 1)

        return frame, self.current_scene

    def simulate_detection(self, scene):
        """模拟深度学习检测结果"""
        # 基础置信度
        if scene in config.CLASS_NAMES:
            base_confidence = 0.85 + random.random() * 0.12  # 85-97%
        else:
            base_confidence = 0.3 + random.random() * 0.4  # 30-70%

        # 添加一些随机波动
        confidence = max(0.1, min(0.99, base_confidence + (random.random() - 0.5) * 0.1))

        return scene, confidence


# 创建虚拟摄像头实例
virtual_camera = VirtualCamera()


# ==================== 无人机智能系统 ====================
class IntelligentDroneSystem:
    """无人机智能控制系统"""

    def __init__(self):
        self.emergency_level = 0  # 0-10，10为最高紧急级别
        self.flight_log = []
        self.last_detection_time = time.time()
        self.detection_interval = 2.0  # 每2秒检测一次
        self.response_actions = {
            '森林 Forest': self.normal_flight,
            '火灾 Fire': self.emergency_response,
            '城市 City': self.urban_flight,
            '动物 Animal': self.avoid_obstacle,
            '车辆 Vehicle': self.traffic_awareness,
            '水域 Water': self.water_precaution
        }

        print("🤖 无人机智能系统初始化完成")

    def normal_flight(self):
        """正常飞行模式"""
        return {
            'action': '正常飞行',
            'speed': 5,
            'altitude': '维持',
            'message': '森林环境，保持正常飞行模式'
        }

    def emergency_response(self):
        """火灾应急响应"""
        self.emergency_level = min(10, self.emergency_level + 2)
        return {
            'action': '紧急响应',
            'speed': 8,
            'altitude': '升高',
            'message': '检测到火灾！正在升高避让并发送警报'
        }

    def urban_flight(self):
        """城市飞行模式"""
        return {
            'action': '谨慎飞行',
            'speed': 3,
            'altitude': '维持',
            'message': '城市环境，注意建筑物和人群'
        }

    def avoid_obstacle(self):
        """避障模式"""
        return {
            'action': '避障飞行',
            'speed': 4,
            'altitude': '微调',
            'message': '检测到动物，保持安全距离'
        }

    def traffic_awareness(self):
        """交通感知模式"""
        return {
            'action': '交通感知',
            'speed': 2,
            'altitude': '维持',
            'message': '检测到车辆，注意交通状况'
        }

    def water_precaution(self):
        """水域预防模式"""
        return {
            'action': '水域预防',
            'speed': 3,
            'altitude': '升高',
            'message': '检测到水域，升高飞行高度避免接触'
        }

    def analyze_scene(self, scene, confidence):
        """分析场景并制定飞行策略"""
        current_time = time.time()

        # 检查是否需要更新检测
        if current_time - self.last_detection_time < self.detection_interval:
            return None

        self.last_detection_time = current_time

        # 根据置信度调整响应
        if confidence < 0.6:
            response = {
                'action': '待确认',
                'speed': 1,
                'altitude': '悬停',
                'message': '检测置信度较低，悬停待确认'
            }
        else:
            # 获取对应场景的响应
            if scene in self.response_actions:
                response = self.response_actions[scene]()
            else:
                response = self.normal_flight()

        # 记录飞行日志
        log_entry = {
            'timestamp': datetime.now().strftime("%H:%M:%S"),
            'scene': scene,
            'confidence': confidence,
            'action': response['action'],
            'message': response['message']
        }
        self.flight_log.append(log_entry)

        # 保持日志大小
        if len(self.flight_log) > 20:
            self.flight_log = self.flight_log[-20:]

        return response


# 创建无人机智能系统
drone_system = IntelligentDroneSystem()


# ==================== 无人机模拟器 ====================
class DroneSimulator(threading.Thread):
    """无人机状态模拟器线程"""

    def __init__(self):
        super().__init__()
        self.running = True
        self.daemon = True
        self.simulation_speed = 1.0
        self.last_update = time.time()

        print("🚁 无人机模拟器启动")

    def run(self):
        """运行模拟器"""
        while self.running:
            try:
                current_time = time.time()
                delta_time = min(1.0, current_time - self.last_update) * self.simulation_speed
                self.last_update = current_time

                # 更新无人机状态
                self.update_drone_status(delta_time)

                # 根据当前模式更新状态
                self.update_by_mode(delta_time)

                # 更新环境参数
                self.update_environment()

                time.sleep(0.1)

            except Exception as e:
                print(f"模拟器错误: {e}")
                time.sleep(1)

    def update_drone_status(self, delta_time):
        """更新无人机状态"""
        # 根据飞行模式更新电池
        if config.DRONE_STATUS['mode'] == 'FLYING':
            config.DRONE_STATUS['battery'] = max(0, config.DRONE_STATUS['battery'] - 0.1 * delta_time)

        # 自动充电（如果着陆且电量低于20%）
        elif config.DRONE_STATUS['mode'] == 'LANDED' and config.DRONE_STATUS['battery'] < 20:
            config.DRONE_STATUS['battery'] = min(100, config.DRONE_STATUS['battery'] + 0.5 * delta_time)

    def update_by_mode(self, delta_time):
        """根据飞行模式更新"""
        mode = config.DRONE_STATUS['mode']

        if mode == 'TAKEOFF':
            config.DRONE_STATUS['altitude'] = min(50, config.DRONE_STATUS['altitude'] + 10 * delta_time)
            config.DRONE_STATUS['speed'] = 2

            if config.DRONE_STATUS['altitude'] >= 50:
                config.DRONE_STATUS['mode'] = 'FLYING'
                print("🛫 起飞完成，进入飞行模式")

        elif mode == 'FLYING':
            # 随机飞行路径
            config.DRONE_STATUS['altitude'] = 50 + random.uniform(-3, 3)
            config.DRONE_STATUS['speed'] = 3 + random.uniform(-1, 1)

            # 随机位置变化
            config.DRONE_STATUS['location']['x'] += random.uniform(-2, 2) * delta_time
            config.DRONE_STATUS['location']['y'] += random.uniform(-2, 2) * delta_time

        elif mode == 'LANDING':
            config.DRONE_STATUS['altitude'] = max(0, config.DRONE_STATUS['altitude'] - 8 * delta_time)
            config.DRONE_STATUS['speed'] = 1

            if config.DRONE_STATUS['altitude'] <= 0:
                config.DRONE_STATUS['mode'] = 'LANDED'
                config.DRONE_STATUS['speed'] = 0
                print("🛬 降落完成")

    def update_environment(self):
        """更新环境参数"""
        # 模拟温度变化
        config.DRONE_STATUS['temperature'] = 20 + random.uniform(-2, 2)

        # 模拟风速变化
        config.DRONE_STATUS['wind_speed'] = max(0, 3 + random.uniform(-1, 1))

        # 更新GPS信号（受环境影响）
        if config.DRONE_STATUS['detected_class'] == '城市 City':
            config.DRONE_STATUS['gps_signal'] = random.choice(['中', '强'])
        else:
            config.DRONE_STATUS['gps_signal'] = '强'

    def stop(self):
        """停止模拟器"""
        self.running = False


# 启动无人机模拟器
drone_simulator = DroneSimulator()
drone_simulator.start()


# ==================== Flask路由和功能 ====================
def generate_video_feed():
    """生成视频流"""
    while True:
        # 获取虚拟摄像头帧
        frame, current_scene = virtual_camera.get_frame()

        # 模拟AI检测
        detected_scene, confidence = virtual_camera.simulate_detection(current_scene)

        # 更新无人机状态
        config.DRONE_STATUS['detected_class'] = detected_scene
        config.DRONE_STATUS['confidence'] = confidence
        config.DRONE_STATUS['timestamp'] = datetime.now().strftime("%H:%M:%S")

        # 获取智能响应
        response = drone_system.analyze_scene(detected_scene, confidence)
        if response:
            # 更新飞行模式（如果响应建议）
            if response['action'] == '紧急响应' and config.DRONE_STATUS['mode'] == 'FLYING':
                config.DRONE_STATUS['mode'] = 'LANDING'

        # 添加检测信息到帧
        self.add_detection_overlay(frame, detected_scene, confidence)

        # 编码为JPEG
        ret, buffer = cv2.imencode('.jpg', frame)
        frame_bytes = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame_bytes + b'\r\n')


def add_detection_overlay(frame, scene, confidence):
    """添加检测信息到视频帧"""
    height, width = frame.shape[:2]

    # 添加半透明状态栏
    overlay = frame.copy()
    cv2.rectangle(overlay, (0, 0), (width, 100), (0, 0, 0), -1)
    cv2.addWeighted(overlay, 0.6, frame, 0.4, 0, frame)

    # 添加标题
    cv2.putText(frame, "🚁 无人机视觉导航演示系统", (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)

    # 添加检测结果
    color = config.CLASS_COLORS.get(scene, (255, 255, 255))
    detection_text = f"检测: {scene}"
    confidence_text = f"置信度: {confidence:.1%}"

    cv2.putText(frame, detection_text, (10, 60),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
    cv2.putText(frame, confidence_text, (10, 85),
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    # 添加无人机状态
    status_text = f"状态: {config.DRONE_STATUS['mode']} | 电量: {config.DRONE_STATUS['battery']:.1f}%"
    cv2.putText(frame, status_text, (width - 300, 30),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)

    # 添加场景指示器
    for i, scene_name in enumerate(config.CLASS_NAMES):
        x = 10 + (i * 100)
        if x + 90 < width:
            color = config.CLASS_COLORS.get(scene_name, (128, 128, 128))
            thickness = 3 if scene_name == scene else 1
            cv2.rectangle(frame, (x, height - 30), (x + 90, height - 10), color, thickness)

            # 简化显示
            scene_short = scene_name.split()[0]
            cv2.putText(frame, scene_short, (x + 5, height - 15),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.4, color, 1)


@app.route('/')
def index():
    """主页面"""
    return render_template('index.html',
                           class_names=config.CLASS_NAMES,
                           drone_status=config.DRONE_STATUS,
                           class_colors=config.CLASS_COLORS)


@app.route('/video_feed')
def video_feed():
    """视频流端点"""
    return Response(generate_video_feed(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/drone_status')
def get_drone_status():
    """获取无人机状态"""
    return jsonify(config.DRONE_STATUS)


@app.route('/flight_log')
def get_flight_log():
    """获取飞行日志"""
    return jsonify(drone_system.flight_log)


@app.route('/system_info')
def get_system_info():
    """获取系统信息"""
    info = {
        'demo_mode': config.DEMO_MODE,
        'virtual_camera': config.USE_VIRTUAL_CAMERA,
        'fps': 30,
        'detection_accuracy': '演示模式',
        'system_time': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        'uptime': time.time() - drone_simulator.last_update,
        'emergency_level': drone_system.emergency_level
    }
    return jsonify(info)


@app.route('/control', methods=['POST'])
def control_drone():
    """控制无人机"""
    data = request.json
    command = data.get('command', '')

    response = {
        'success': True,
        'message': '',
        'command': command,
        'timestamp': datetime.now().strftime("%H:%M:%S")
    }

    try:
        current_mode = config.DRONE_STATUS['mode']

        if command == 'takeoff':
            if current_mode == 'LANDED':
                config.DRONE_STATUS['mode'] = 'TAKEOFF'
                response['message'] = '无人机正在起飞...'
            else:
                response['success'] = False
                response['message'] = f'无法起飞，当前状态: {current_mode}'

        elif command == 'land':
            if current_mode in ['FLYING', 'TAKEOFF']:
                config.DRONE_STATUS['mode'] = 'LANDING'
                response['message'] = '无人机正在降落...'
            else:
                response['success'] = False
                response['message'] = f'无法降落，当前状态: {current_mode}'

        elif command == 'emergency_land':
            config.DRONE_STATUS['mode'] = 'LANDING'
            response['message'] = '紧急降落已启动'

        elif command == 'hover':
            response['message'] = '悬停模式已激活'

        elif command == 'charge':
            config.DRONE_STATUS['battery'] = 100
            response['message'] = '电池已充满'

        elif command == 'auto_pilot':
            response['message'] = '自动驾驶模式已激活'

        elif command == 'return_home':
            config.DRONE_STATUS['location'] = {'x': 0, 'y': 0, 'z': config.DRONE_STATUS['altitude']}
            response['message'] = '正在返回起始点'

        else:
            response['success'] = False
            response['message'] = f'未知命令: {command}'

    except Exception as e:
        response['success'] = False
        response['message'] = f'控制错误: {str(e)}'

    return jsonify(response)


@app.route('/simulate_scene', methods=['POST'])
def simulate_scene():
    """手动模拟特定场景"""
    data = request.json
    scene = data.get('scene', '')

    if scene in config.CLASS_NAMES:
        virtual_camera.current_scene = scene
        virtual_camera.scene_transition = 15

        response = {
            'success': True,
            'message': f'已切换到 {scene} 场景',
            'scene': scene
        }
    else:
        response = {
            'success': False,
            'message': f'未知场景: {scene}',
            'available_scenes': config.CLASS_NAMES
        }

    return jsonify(response)


@app.route('/capture_image')
def capture_image():
    """捕获当前帧图像"""
    frame, _ = virtual_camera.get_frame()

    # 编码为PNG
    ret, buffer = cv2.imencode('.png', frame)

    # 创建内存文件
    img_io = io.BytesIO(buffer.tobytes())
    img_io.seek(0)

    return send_file(img_io, mimetype='image/png',
                     as_attachment=True,
                     download_name=f'drone_capture_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png')


# ==================== 主函数 ====================
if __name__ == '__main__':
    print("=" * 70)
    print("🚁 无人机导航系统 - 完整演示版")
    print("=" * 70)
    print("🎯 无需真实数据，立即展示效果")
    print("📊 检测场景:", ", ".join(config.CLASS_NAMES))
    print("🌐 访问地址: http://localhost:5000")
    print("=" * 70)
    print("🎮 控制功能:")
    print("  - 起飞/降落/紧急降落")
    print("  - 自动驾驶/返航")
    print("  - 手动切换场景")
    print("  - 实时视频流")
    print("=" * 70)

    # 检查模板是否存在
    if not os.path.exists("templates/index.html"):
        print("⚠️  未找到模板文件，正在创建...")
        create_default_template()

    # 检查静态目录
    os.makedirs("static/css", exist_ok=True)
    os.makedirs("static/js", exist_ok=True)

    # 创建静态文件
    create_static_files()

    # 运行Flask应用
    app.run(debug=True, host='0.0.0.0', port=5000, threaded=True)


# ==================== 辅助函数 ====================
def create_default_template():
    """创建默认HTML模板"""
    template_dir = "templates"
    os.makedirs(template_dir, exist_ok=True)

    html_content = """<!DOCTYPE html>
<html lang="zh-CN">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>🚁 无人机视觉导航演示系统</title>
    <link rel="stylesheet" href="{{ url_for('static', filename='css/style.css') }}">
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.4.0/css/all.min.css">
</head>
<body>
    <div class="container">
        <!-- 头部 -->
        <header class="header">
            <div class="header-content">
                <h1><i class="fas fa-drone"></i> 无人机视觉导航演示系统</h1>
                <p class="subtitle">基于深度学习的实时环境识别与智能飞行控制</p>
                <div class="demo-badge">
                    <i class="fas fa-rocket"></i> 演示模式 | 实时效果展示
                </div>
            </div>
        </header>

        <!-- 主内容区 -->
        <div class="main-content">
            <!-- 左侧：视频和检测 -->
            <div class="video-section">
                <div class="section-header">
                    <h2><i class="fas fa-video"></i> 实时视觉识别</h2>
                    <div class="fps-indicator">30 FPS</div>
                </div>

                <div class="video-container">
                    <img id="video-feed" src="{{ url_for('video_feed') }}" alt="实时视频流">
                    <div class="video-overlay">
                        <div class="detection-info">
                            <div class="detection-title">实时检测结果</div>
                            <div class="detection-result">
                                <span id="live-class">{{ drone_status.detected_class }}</span>
                                <span id="live-confidence">{{ "%.1f"|format(drone_status.confidence * 100) }}%</span>
                            </div>
                        </div>
                        <button id="capture-btn" class="btn-capture">
                            <i class="fas fa-camera"></i> 截图
                        </button>
                    </div>
                </div>

                <div class="detection-panel">
                    <h3><i class="fas fa-search"></i> 场景识别面板</h3>
                    <div class="confidence-meter">
                        <div class="meter-label">检测置信度</div>
                        <div class="meter-bar">
                            <div class="meter-fill" id="confidence-fill" 
                                 style="width: {{ drone_status.confidence * 100 }}%"></div>
                        </div>
                        <div class="meter-value" id="confidence-value">
                            {{ "%.1f"|format(drone_status.confidence * 100) }}%
                        </div>
                    </div>

                    <div class="scene-controls">
                        <h4>手动场景切换</h4>
                        <div class="scene-buttons">
                            {% for scene in class_names %}
                            <button class="scene-btn" data-scene="{{ scene }}"
                                    style="border-color: rgb{{ class_colors[scene] }}">
                                {{ scene }}
                            </button>
                            {% endfor %}
                        </div>
                    </div>
                </div>
            </div>

            <!-- 右侧：控制和状态 -->
            <div class="control-section">
                <div class="section-header">
                    <h2><i class="fas fa-gamepad"></i> 飞行控制</h2>
                    <div class="mode-indicator" id="mode-indicator">{{ drone_status.mode }}</div>
                </div>

                <!-- 无人机状态 -->
                <div class="status-grid">
                    <div class="status-card">
                        <div class="status-icon"><i class="fas fa-battery-full"></i></div>
                        <div class="status-content">
                            <div class="status-label">电池电量</div>
                            <div class="status-value" id="battery-value">{{ drone_status.battery|round(1) }}%</div>
                            <div class="status-bar">
                                <div class="bar-fill" id="battery-fill" 
                                     style="width: {{ drone_status.battery }}%"></div>
                            </div>
                        </div>
                    </div>

                    <div class="status-card">
                        <div class="status-icon"><i class="fas fa-mountain"></i></div>
                        <div class="status-content">
                            <div class="status-label">飞行高度</div>
                            <div class="status-value" id="altitude-value">{{ drone_status.altitude|round(1) }} m</div>
                        </div>
                    </div>

                    <div class="status-card">
                        <div class="status-icon"><i class="fas fa-tachometer-alt"></i></div>
                        <div class="status-content">
                            <div class="status-label">飞行速度</div>
                            <div class="status-value" id="speed-value">{{ drone_status.speed|round(1) }} m/s</div>
                        </div>
                    </div>

                    <div class="status-card">
                        <div class="status-icon"><i class="fas fa-map-marker-alt"></i></div>
                        <div class="status-content">
                            <div class="status-label">位置坐标</div>
                            <div class="status-value" id="position-value">
                                ({{ drone_status.location.x|round(1) }}, {{ drone_status.location.y|round(1) }})
                            </div>
                        </div>
                    </div>
                </div>

                <!-- 飞行控制按钮 -->
                <div class="control-panel">
                    <div class="control-row">
                        <button class="control-btn btn-takeoff" id="takeoff-btn">
                            <i class="fas fa-rocket"></i>
                            <span>起飞</span>
                        </button>
                        <button class="control-btn btn-land" id="land-btn">
                            <i class="fas fa-plane-arrival"></i>
                            <span>降落</span>
                        </button>
                        <button class="control-btn btn-emergency" id="emergency-btn">
                            <i class="fas fa-exclamation-triangle"></i>
                            <span>紧急降落</span>
                        </button>
                    </div>

                    <div class="control-row">
                        <button class="control-btn btn-direction" id="forward-btn">
                            <i class="fas fa-arrow-up"></i>
                            <span>前进</span>
                        </button>
                        <button class="control-btn btn-direction" id="backward-btn">
                            <i class="fas fa-arrow-down"></i>
                            <span>后退</span>
                        </button>
                        <button class="control-btn btn-direction" id="left-btn">
                            <i class="fas fa-arrow-left"></i>
                            <span>左转</span>
                        </button>
                        <button class="control-btn btn-direction" id="right-btn">
                            <i class="fas fa-arrow-right"></i>
                            <span>右转</span>
                        </button>
                    </div>

                    <div class="control-row">
                        <button class="control-btn btn-action" id="hover-btn">
                            <i class="fas fa-pause-circle"></i>
                            <span>悬停</span>
                        </button>
                        <button class="control-btn btn-action" id="charge-btn">
                            <i class="fas fa-charging-station"></i>
                            <span>充电</span>
                        </button>
                        <button class="control-btn btn-action" id="auto-btn">
                            <i class="fas fa-robot"></i>
                            <span>自动驾驶</span>
                        </button>
                        <button class="control-btn btn-action" id="home-btn">
                            <i class="fas fa-home"></i>
                            <span>返航</span>
                        </button>
                    </div>
                </div>

                <!-- 环境信息 -->
                <div class="environment-panel">
                    <h3><i class="fas fa-cloud-sun"></i> 环境监测</h3>
                    <div class="env-grid">
                        <div class="env-item">
                            <i class="fas fa-thermometer-half"></i>
                            <span>温度: <span id="temp-value">{{ drone_status.temperature|round(1) }}°C</span></span>
                        </div>
                        <div class="env-item">
                            <i class="fas fa-wind"></i>
                            <span>风速: <span id="wind-value">{{ drone_status.wind_speed|round(1) }} m/s</span></span>
                        </div>
                        <div class="env-item">
                            <i class="fas fa-satellite"></i>
                            <span>GPS: <span id="gps-value">{{ drone_status.gps_signal }}</span></span>
                        </div>
                        <div class="env-item">
                            <i class="fas fa-shield-alt"></i>
                            <span>紧急等级: <span id="emergency-value">0</span>/10</span>
                        </div>
                    </div>
                </div>
            </div>
        </div>

        <!-- 底部：日志和系统信息 -->
        <div class="footer-section">
            <div class="logs-panel">
                <h3><i class="fas fa-clipboard-list"></i> 飞行日志</h3>
                <div class="logs-container" id="logs-container">
                    <!-- 日志将通过JavaScript动态加载 -->
                    <div class="log-entry">
                        <span class="log-time">--:--:--</span>
                        <span class="log-message">系统启动中...</span>
                    </div>
                </div>
            </div>

            <div class="system-info">
                <h3><i class="fas fa-info-circle"></i> 系统信息</h3>
                <div class="info-grid">
                    <div class="info-item">
                        <i class="fas fa-microchip"></i>
                        <span>运行模式: <span id="system-mode">演示模式</span></span>
                    </div>
                    <div class="info-item">
                        <i class="fas fa-clock"></i>
                        <span>系统时间: <span id="system-time">--:--:--</span></span>
                    </div>
                    <div class="info-item">
                        <i class="fas fa-eye"></i>
                        <span>检测精度: <span id="detection-accuracy">演示模式</span></span>
                    </div>
                    <div class="info-item">
                        <i class="fas fa-wifi"></i>
                        <span>连接状态: <span id="connection-status">已连接</span></span>
                    </div>
                </div>

                <div class="system-controls">
                    <button class="sys-btn" id="refresh-btn">
                        <i class="fas fa-sync-alt"></i> 刷新状态
                    </button>
                    <button class="sys-btn" id="help-btn">
                        <i class="fas fa-question-circle"></i> 使用帮助
                    </button>
                    <button class="sys-btn" id="fullscreen-btn">
                        <i class="fas fa-expand"></i> 全屏显示
                    </button>
                </div>
            </div>
        </div>

        <!-- 页脚 -->
        <footer class="page-footer">
            <p>🚁 无人机视觉导航演示系统 | 基于深度学习的实时环境识别 | © 2025</p>
            <p>演示版本 v2.0 | 最后更新: <span id="last-update">--:--:--</span></p>
        </footer>
    </div>

    <!-- 通知容器 -->
    <div id="notification-container"></div>

    <!-- JavaScript文件 -->
    <script src="{{ url_for('static', filename='js/main.js') }}"></script>
</body>
</html>
"""

    template_path = os.path.join(template_dir, "index.html")
    with open(template_path, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"✅ 已创建模板文件: {template_path}")


def create_static_files():
    """创建静态CSS和JS文件"""
    # 创建CSS文件
    css_content = """/* 无人机导航系统样式 */
:root {
    --primary-color: #00b4d8;
    --secondary-color: #0077b6;
    --success-color: #00b894;
    --warning-color: #fdcb6e;
    --danger-color: #e17055;
    --dark-color: #2d3436;
    --light-color: #f5f5f5;
    --gray-color: #636e72;
}

* {
    margin: 0;
    padding: 0;
    box-sizing: border-box;
}

body {
    font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
    background: linear-gradient(135deg, #0a1929 0%, #1a365d 50%, #2d3748 100%);
    color: white;
    min-height: 100vh;
    overflow-x: hidden;
}

.container {
    max-width: 1800px;
    margin: 0 auto;
    padding: 20px;
}

/* 头部样式 */
.header {
    background: rgba(10, 25, 47, 0.9);
    border-radius: 20px;
    padding: 25px;
    margin-bottom: 25px;
    border: 2px solid rgba(0, 180, 216, 0.3);
    box-shadow: 0 10px 30px rgba(0, 0, 0, 0.5);
    backdrop-filter: blur(10px);
}

.header-content {
    text-align: center;
}

.header h1 {
    font-size: 2.8rem;
    margin-bottom: 10px;
    background: linear-gradient(90deg, #00b4d8, #0077b6);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    text-shadow: 0 2px 10px rgba(0, 180, 216, 0.3);
}

.subtitle {
    color: #88ffdd;
    font-size: 1.2rem;
    margin-bottom: 15px;
}

.demo-badge {
    display: inline-block;
    background: linear-gradient(90deg, #00b894, #00cec9);
    color: white;
    padding: 8px 20px;
    border-radius: 25px;
    font-weight: bold;
    box-shadow: 0 4px 15px rgba(0, 184, 148, 0.4);
}

/* 主内容区 */
.main-content {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 25px;
    margin-bottom: 25px;
}

@media (max-width: 1200px) {
    .main-content {
        grid-template-columns: 1fr;
    }
}

.section-header {
    display: flex;
    justify-content: space-between;
    align-items: center;
    margin-bottom: 20px;
    padding-bottom: 15px;
    border-bottom: 2px solid var(--primary-color);
}

.section-header h2 {
    font-size: 1.8rem;
    color: var(--primary-color);
}

.fps-indicator, .mode-indicator {
    background: rgba(0, 0, 0, 0.3);
    padding: 8px 15px;
    border-radius: 15px;
    font-weight: bold;
    border: 1px solid var(--primary-color);
}

/* 视频区域 */
.video-section {
    background: rgba(10, 25, 47, 0.8);
    border-radius: 20px;
    padding: 25px;
    border: 2px solid rgba(0, 180, 216, 0.2);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
}

.video-container {
    position: relative;
    width: 100%;
    border-radius: 15px;
    overflow: hidden;
    background: black;
    margin-bottom: 20px;
    border: 3px solid rgba(255, 255, 255, 0.1);
}

#video-feed {
    width: 100%;
    display: block;
    transition: transform 0.3s;
}

#video-feed:hover {
    transform: scale(1.01);
}

.video-overlay {
    position: absolute;
    top: 20px;
    left: 20px;
    right: 20px;
    background: rgba(0, 0, 0, 0.7);
    padding: 15px;
    border-radius: 10px;
    display: flex;
    justify-content: space-between;
    align-items: center;
    backdrop-filter: blur(5px);
    border: 1px solid rgba(255, 255, 255, 0.1);
}

.detection-info {
    display: flex;
    flex-direction: column;
    gap: 5px;
}

.detection-title {
    font-size: 0.9rem;
    color: #aaa;
}

.detection-result {
    display: flex;
    gap: 20px;
    font-size: 1.2rem;
    font-weight: bold;
}

#live-class {
    color: #00ff88;
}

#live-confidence {
    color: #ffcc00;
}

.btn-capture {
    background: linear-gradient(90deg, #6c5ce7, #a29bfe);
    color: white;
    border: none;
    padding: 10px 20px;
    border-radius: 8px;
    cursor: pointer;
    font-weight: bold;
    transition: all 0.3s;
}

.btn-capture:hover {
    transform: translateY(-2px);
    box-shadow: 0 5px 15px rgba(108, 92, 231, 0.4);
}

/* 检测面板 */
.detection-panel {
    background: rgba(20, 40, 80, 0.6);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(0, 150, 255, 0.3);
}

.confidence-meter {
    margin: 20px 0;
}

.meter-label {
    margin-bottom: 8px;
    color: #88ffcc;
}

.meter-bar {
    height: 20px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 10px;
    overflow: hidden;
    position: relative;
}

.meter-fill {
    height: 100%;
    background: linear-gradient(90deg, #ff0000, #ff9900, #00ff00);
    border-radius: 10px;
    transition: width 0.5s;
}

.meter-value {
    text-align: center;
    margin-top: 5px;
    font-weight: bold;
    font-size: 1.2rem;
}

.scene-controls {
    margin-top: 25px;
}

.scene-controls h4 {
    margin-bottom: 15px;
    color: #00ccff;
}

.scene-buttons {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 10px;
}

.scene-btn {
    padding: 12px;
    background: rgba(255, 255, 255, 0.1);
    border: 2px solid;
    border-radius: 8px;
    color: white;
    cursor: pointer;
    transition: all 0.3s;
    font-size: 0.9rem;
}

.scene-btn:hover {
    background: rgba(255, 255, 255, 0.2);
    transform: translateY(-3px);
}

/* 控制区域 */
.control-section {
    background: rgba(10, 25, 47, 0.8);
    border-radius: 20px;
    padding: 25px;
    border: 2px solid rgba(0, 180, 216, 0.2);
    box-shadow: 0 10px 25px rgba(0, 0, 0, 0.4);
}

.status-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
    margin-bottom: 25px;
}

.status-card {
    background: rgba(20, 40, 80, 0.6);
    padding: 15px;
    border-radius: 12px;
    display: flex;
    align-items: center;
    gap: 15px;
    border-left: 4px solid var(--primary-color);
    transition: transform 0.3s;
}

.status-card:hover {
    transform: translateY(-5px);
    background: rgba(20, 40, 80, 0.8);
}

.status-icon {
    font-size: 2rem;
    color: var(--primary-color);
}

.status-content {
    flex: 1;
}

.status-label {
    font-size: 0.9rem;
    color: #88ffcc;
    margin-bottom: 5px;
}

.status-value {
    font-size: 1.4rem;
    font-weight: bold;
    margin-bottom: 8px;
}

.status-bar {
    height: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
    overflow: hidden;
}

.bar-fill {
    height: 100%;
    background: linear-gradient(90deg, #ff0000, #ff9900, #00ff00);
    border-radius: 4px;
    transition: width 0.5s;
}

/* 控制面板 */
.control-panel {
    margin: 25px 0;
}

.control-row {
    display: flex;
    gap: 15px;
    margin-bottom: 15px;
}

.control-btn {
    flex: 1;
    padding: 20px 10px;
    border: none;
    border-radius: 12px;
    color: white;
    cursor: pointer;
    transition: all 0.3s;
    display: flex;
    flex-direction: column;
    align-items: center;
    gap: 8px;
    font-size: 1rem;
    font-weight: bold;
}

.control-btn i {
    font-size: 1.8rem;
}

.btn-takeoff {
    background: linear-gradient(145deg, #00b09b, #96c93d);
}

.btn-land {
    background: linear-gradient(145deg, #2193b0, #6dd5ed);
}

.btn-emergency {
    background: linear-gradient(145deg, #ff416c, #ff4b2b);
}

.btn-direction {
    background: linear-gradient(145deg, #2a5298, #1e3c72);
}

.btn-action {
    background: linear-gradient(145deg, #8a2387, #f27121);
}

.control-btn:hover {
    transform: translateY(-5px) scale(1.05);
    box-shadow: 0 10px 20px rgba(0, 0, 0, 0.3);
}

.control-btn:active {
    transform: translateY(-2px);
}

/* 环境面板 */
.environment-panel {
    background: rgba(20, 40, 80, 0.6);
    padding: 20px;
    border-radius: 15px;
    border: 1px solid rgba(0, 150, 255, 0.3);
}

.env-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
    margin-top: 15px;
}

.env-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}

.env-item i {
    color: var(--primary-color);
    font-size: 1.2rem;
}

/* 底部区域 */
.footer-section {
    display: grid;
    grid-template-columns: 2fr 1fr;
    gap: 25px;
    margin-bottom: 25px;
}

@media (max-width: 1200px) {
    .footer-section {
        grid-template-columns: 1fr;
    }
}

.logs-panel, .system-info {
    background: rgba(10, 25, 47, 0.8);
    border-radius: 20px;
    padding: 25px;
    border: 2px solid rgba(0, 180, 216, 0.2);
}

.logs-container {
    height: 200px;
    overflow-y: auto;
    margin-top: 15px;
    background: rgba(0, 0, 0, 0.3);
    border-radius: 10px;
    padding: 15px;
}

.log-entry {
    padding: 8px 12px;
    margin-bottom: 8px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 6px;
    border-left: 3px solid var(--primary-color);
}

.log-time {
    color: #ffcc00;
    font-weight: bold;
    margin-right: 15px;
}

.log-message {
    color: white;
}

.info-grid {
    display: grid;
    grid-template-columns: repeat(2, 1fr);
    gap: 15px;
    margin: 20px 0;
}

.info-item {
    display: flex;
    align-items: center;
    gap: 10px;
    padding: 12px;
    background: rgba(255, 255, 255, 0.1);
    border-radius: 8px;
}

.info-item i {
    color: var(--primary-color);
}

.system-controls {
    display: flex;
    gap: 10px;
    margin-top: 20px;
}

.sys-btn {
    flex: 1;
    padding: 12px;
    background: rgba(0, 150, 255, 0.3);
    border: 1px solid rgba(0, 150, 255, 0.5);
    border-radius: 8px;
    color: white;
    cursor: pointer;
    transition: all 0.3s;
}

.sys-btn:hover {
    background: rgba(0, 150, 255, 0.5);
}

/* 页脚 */
.page-footer {
    text-align: center;
    padding: 25px;
    border-top: 1px solid rgba(255, 255, 255, 0.1);
    color: #88aaff;
    background: rgba(0, 20, 40, 0.5);
    border-radius: 15px;
}

/* 通知样式 */
#notification-container {
    position: fixed;
    top: 20px;
    right: 20px;
    z-index: 1000;
}

.notification {
    padding: 15px 25px;
    margin-bottom: 10px;
    border-radius: 10px;
    color: white;
    font-weight: bold;
    box-shadow: 0 5px 15px rgba(0, 0, 0, 0.3);
    animation: slideIn 0.3s ease-out;
    max-width: 300px;
}

.notification.success {
    background: linear-gradient(90deg, #00b09b, #96c93d);
}

.notification.error {
    background: linear-gradient(90deg, #ff416c, #ff4b2b);
}

.notification.info {
    background: linear-gradient(90deg, #2193b0, #6dd5ed);
}

@keyframes slideIn {
    from { transform: translateX(100%); opacity: 0; }
    to { transform: translateX(0); opacity: 1; }
}

@keyframes slideOut {
    from { transform: translateX(0); opacity: 1; }
    to { transform: translateX(100%); opacity: 0; }
}

/* 滚动条样式 */
::-webkit-scrollbar {
    width: 8px;
}

::-webkit-scrollbar-track {
    background: rgba(255, 255, 255, 0.1);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb {
    background: var(--primary-color);
    border-radius: 4px;
}

::-webkit-scrollbar-thumb:hover {
    background: var(--secondary-color);
}
"""

    css_path = "static/css/style.css"
    with open(css_path, "w", encoding="utf-8") as f:
        f.write(css_content)

    # 创建JS文件
    js_content = """// 无人机导航系统交互脚本
document.addEventListener('DOMContentLoaded', function() {
    // 全局变量
    let updateInterval;
    let logsInterval;

    // 元素引用
    const videoFeed = document.getElementById('video-feed');
    const liveClass = document.getElementById('live-class');
    const liveConfidence = document.getElementById('live-confidence');
    const confidenceFill = document.getElementById('confidence-fill');
    const confidenceValue = document.getElementById('confidence-value');
    const modeIndicator = document.getElementById('mode-indicator');
    const batteryValue = document.getElementById('battery-value');
    const batteryFill = document.getElementById('battery-fill');
    const altitudeValue = document.getElementById('altitude-value');
    const speedValue = document.getElementById('speed-value');
    const positionValue = document.getElementById('position-value');
    const tempValue = document.getElementById('temp-value');
    const windValue = document.getElementById('wind-value');
    const gpsValue = document.getElementById('gps-value');
    const emergencyValue = document.getElementById('emergency-value');
    const systemTime = document.getElementById('system-time');
    const logsContainer = document.getElementById('logs-container');
    const lastUpdate = document.getElementById('last-update');

    // 初始化函数
    function init() {
        console.log('🚀 无人机导航系统初始化...');

        // 开始更新循环
        startUpdateLoop();

        // 绑定事件
        bindEvents();

        // 初始加载
        updateDroneStatus();
        updateFlightLog();
        updateSystemInfo();

        // 显示欢迎通知
        showNotification('欢迎使用无人机导航演示系统！', 'info');
    }

    // 开始更新循环
    function startUpdateLoop() {
        // 更新无人机状态（每秒）
        updateInterval = setInterval(updateDroneStatus, 1000);

        // 更新飞行日志（每2秒）
        logsInterval = setInterval(updateFlightLog, 2000);

        // 更新系统时间（每秒）
        setInterval(updateSystemTime, 1000);
    }

    // 绑定所有事件
    function bindEvents() {
        // 控制按钮
        document.getElementById('takeoff-btn').addEventListener('click', () => sendCommand('takeoff'));
        document.getElementById('land-btn').addEventListener('click', () => sendCommand('land'));
        document.getElementById('emergency-btn').addEventListener('click', () => sendCommand('emergency_land'));
        document.getElementById('hover-btn').addEventListener('click', () => sendCommand('hover'));
        document.getElementById('charge-btn').addEventListener('click', () => sendCommand('charge'));
        document.getElementById('auto-btn').addEventListener('click', () => sendCommand('auto_pilot'));
        document.getElementById('home-btn').addEventListener('click', () => sendCommand('return_home'));

        // 方向控制按钮
        document.getElementById('forward-btn').addEventListener('click', () => showNotification('向前飞行', 'info'));
        document.getElementById('backward-btn').addEventListener('click', () => showNotification('向后飞行', 'info'));
        document.getElementById('left-btn').addEventListener('click', () => showNotification('向左转', 'info'));
        document.getElementById('right-btn').addEventListener('click', () => showNotification('向右转', 'info'));

        // 场景切换按钮
        document.querySelectorAll('.scene-btn').forEach(btn => {
            btn.addEventListener('click', function() {
                const scene = this.getAttribute('data-scene');
                simulateScene(scene);
            });
        });

        // 截图按钮
        document.getElementById('capture-btn').addEventListener('click', captureImage);

        // 系统按钮
        document.getElementById('refresh-btn').addEventListener('click', refreshAll);
        document.getElementById('help-btn').addEventListener('click', showHelp);
        document.getElementById('fullscreen-btn').addEventListener('click', toggleFullscreen);

        // 视频点击全屏
        videoFeed.addEventListener('click', function() {
            if (this.requestFullscreen) {
                this.requestFullscreen();
            }
        });
    }

    // 更新无人机状态
    async function updateDroneStatus() {
        try {
            const response = await fetch('/drone_status');
            const data = await response.json();

            // 更新状态显示
            liveClass.textContent = data.detected_class;
            liveConfidence.textContent = (data.confidence * 100).toFixed(1) + '%';

            confidenceFill.style.width = (data.confidence * 100) + '%';
            confidenceValue.textContent = (data.confidence * 100).toFixed(1) + '%';

            modeIndicator.textContent = data.mode;
            batteryValue.textContent = data.battery.toFixed(1) + '%';
            batteryFill.style.width = data.battery + '%';

            altitudeValue.textContent = data.altitude.toFixed(1) + ' m';
            speedValue.textContent = data.speed.toFixed(1) + ' m/s';
            positionValue.textContent = `(${data.location.x.toFixed(1)}, ${data.location.y.toFixed(1)})`;

            tempValue.textContent = data.temperature.toFixed(1) + '°C';
            windValue.textContent = data.wind_speed.toFixed(1) + ' m/s';
            gpsValue.textContent = data.gps_signal;

            // 更新最后更新时间
            if (data.timestamp) {
                lastUpdate.textContent = data.timestamp;
            }

        } catch (error) {
            console.error('更新状态失败:', error);
        }
    }

    // 更新飞行日志
    async function updateFlightLog() {
        try {
            const response = await fetch('/flight_log');
            const logs = await response.json();

            // 清空当前日志
            logsContainer.innerHTML = '';

            // 添加日志条目（最多显示10条）
            const displayLogs = logs.slice(-10);

            displayLogs.forEach(log => {
                const logEntry = document.createElement('div');
                logEntry.className = 'log-entry';

                const timeSpan = document.createElement('span');
                timeSpan.className = 'log-time';
                timeSpan.textContent = log.timestamp;

                const messageSpan = document.createElement('span');
                messageSpan.className = 'log-message';
                messageSpan.textContent = `${log.scene} → ${log.action}: ${log.message}`;

                // 根据动作类型添加颜色
                if (log.action.includes('紧急')) {
                    messageSpan.style.color = '#ff5555';
                } else if (log.action.includes('正常')) {
                    messageSpan.style.color = '#55ff55';
                }

                logEntry.appendChild(timeSpan);
                logEntry.appendChild(messageSpan);
                logsContainer.appendChild(logEntry);
            });

            // 滚动到底部
            logsContainer.scrollTop = logsContainer.scrollHeight;

        } catch (error) {
            console.error('更新日志失败:', error);
        }
    }

    // 更新系统信息
    async function updateSystemInfo() {
        try {
            const response = await fetch('/system_info');
            const info = await response.json();

            document.getElementById('system-mode').textContent = info.demo_mode ? '演示模式' : '实战模式';
            document.getElementById('detection-accuracy').textContent = info.detection_accuracy;
            document.getElementById('connection-status').textContent = '已连接';

            // 更新紧急等级
            emergencyValue.textContent = info.emergency_level;
            if (info.emergency_level > 5) {
                emergencyValue.style.color = '#ff5555';
            } else {
                emergencyValue.style.color = '#55ff55';
            }

        } catch (error) {
            console.error('更新系统信息失败:', error);
        }
    }

    // 更新系统时间
    function updateSystemTime() {
        const now = new Date();
        const timeStr = now.toLocaleTimeString('zh-CN');
        systemTime.textContent = timeStr;
    }

    // 发送控制命令
    async function sendCommand(command) {
        try {
            const response = await fetch('/control', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ command: command })
            });

            const data = await response.json();

            if (data.success) {
                showNotification(data.message, 'success');
                console.log('命令成功:', data.message);
            } else {
                showNotification(data.message, 'error');
                console.error('命令失败:', data.message);
            }

        } catch (error) {
            showNotification('网络连接错误', 'error');
            console.error('请求失败:', error);
        }
    }

    // 模拟场景切换
    async function simulateScene(scene) {
        try {
            const response = await fetch('/simulate_scene', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json',
                },
                body: JSON.stringify({ scene: scene })
            });

            const data = await response.json();

            if (data.success) {
                showNotification(`已切换到 ${scene} 场景`, 'info');
            } else {
                showNotification(data.message, 'error');
            }

        } catch (error) {
            console.error('场景切换失败:', error);
        }
    }

    // 捕获图像
    async function captureImage() {
        try {
            const response = await fetch('/capture_image');
            const blob = await response.blob();

            // 创建下载链接
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `drone_capture_${new Date().getTime()}.png`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);

            showNotification('图像已保存', 'success');

        } catch (error) {
            showNotification('截图失败', 'error');
            console.error('截图失败:', error);
        }
    }

    // 刷新所有数据
    function refreshAll() {
        updateDroneStatus();
        updateFlightLog();
        updateSystemInfo();
        showNotification('系统状态已刷新', 'info');
    }

    // 显示帮助信息
    function showHelp() {
        const helpMessage = `
无人机导航系统使用说明：

🎮 控制功能：
• 起飞/降落 - 控制无人机起降
• 紧急降落 - 立即安全降落
• 自动驾驶 - 启用自动飞行模式
• 返航 - 返回起始位置

🌍 场景模拟：
• 点击场景按钮可手动切换环境
• 系统会自动模拟环境变化
• 每个环境都有独特的视觉特征

📊 状态监控：
• 实时显示无人机状态
• 环境检测结果
• 飞行日志记录

💡 提示：
• 点击视频可全屏显示
• 使用截图功能保存当前画面
• 系统使用虚拟数据演示
        `;

        alert(helpMessage);
    }

    // 切换全屏
    function toggleFullscreen() {
        const elem = document.documentElement;

        if (!document.fullscreenElement) {
            if (elem.requestFullscreen) {
                elem.requestFullscreen();
            }
        } else {
            if (document.exitFullscreen) {
                document.exitFullscreen();
            }
        }
    }

    // 显示通知
    function showNotification(message, type = 'info') {
        const container = document.getElementById('notification-container');

        const notification = document.createElement('div');
        notification.className = `notification ${type}`;
        notification.textContent = message;

        container.appendChild(notification);

        // 3秒后移除
        setTimeout(() => {
            notification.style.animation = 'slideOut 0.3s ease-out';
            setTimeout(() => {
                if (notification.parentNode) {
                    notification.parentNode.removeChild(notification);
                }
            }, 300);
        }, 3000);
    }

    // 键盘快捷键
    document.addEventListener('keydown', function(event) {
        switch(event.key) {
            case ' ':
                // 空格键 - 起飞/降落切换
                const mode = modeIndicator.textContent;
                if (mode === 'LANDED') {
                    sendCommand('takeoff');
                } else if (mode === 'FLYING') {
                    sendCommand('land');
                }
                event.preventDefault();
                break;

            case 'Escape':
                // ESC键 - 紧急降落
                sendCommand('emergency_land');
                break;

            case 'h':
                // H键 - 返航
                sendCommand('return_home');
                break;

            case 'c':
                // C键 - 截图
                captureImage();
                break;
        }
    });

    // 页面卸载时清理
    window.addEventListener('beforeunload', function() {
        if (updateInterval) clearInterval(updateInterval);
        if (logsInterval) clearInterval(logsInterval);
    });

    // 初始化应用
    init();
});
"""

    js_path = "static/js/main.js"
    with open(js_path, "w", encoding="utf-8") as f:
        f.write(js_content)

    print(f"✅ 已创建静态文件: {css_path}, {js_path}")
import pygame
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import layers, models
import os
from datetime import datetime

# ========================= 1. 配置参数 =========================
# 窗口设置
WINDOW_WIDTH = 800
WINDOW_HEIGHT = 600
# 无人车参数
CAR_SIZE = 40
CAR_SPEED = 5
# 环境参数
OBSTACLE_NUM = 5
OBSTACLE_SIZE = 30
# 模型参数
MODEL_PATH = "car_control_model.h5"
IMAGE_SIZE = (64, 64)  # 输入模型的图像尺寸
ACTIONS = ["stop", "forward", "backward", "left", "right"]  # 动作空间
ACTION_TO_IDX = {a: i for i, a in enumerate(ACTIONS)}
IDX_TO_ACTION = {i: a for i, a in enumerate(ACTIONS)}
# 数据记录（用于训练）
DATA_DIR = "training_data"
os.makedirs(DATA_DIR, exist_ok=True)

# ========================= 2. 初始化 Pygame 环境 =========================
pygame.init()
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("深度学习无人车 - 方向键控制")
clock = pygame.time.Clock()

# 颜色定义
WHITE = (255, 255, 255)
BLUE = (0, 0, 255)    # 无人车
BLACK = (0, 0, 0)     # 文字/障碍物
GREEN = (0, 255, 0)   # 路径
RED = (255, 0, 0)     # 提示

# 字体设置
font = pygame.font.Font(None, 32)

# ========================= 3. 无人车与环境类 =========================
class Car:
    def __init__(self):
        # 初始位置（屏幕中心）
        self.x = WINDOW_WIDTH // 2 - CAR_SIZE // 2
        self.y = WINDOW_HEIGHT // 2 - CAR_SIZE // 2
        self.angle = 0  # 朝向（0=向上，90=向右，180=向下，270=向左）

    def draw(self, surface):
        # 绘制车身（旋转的矩形）
        rotated_car = pygame.transform.rotate(
            pygame.Surface((CAR_SIZE, CAR_SIZE), pygame.SRCALPHA),
            -self.angle  # pygame 旋转方向与常规相反
        )
        rotated_car.fill(BLUE)
        # 绘制朝向箭头
        arrow_points = [
            (CAR_SIZE//2, 5),
            (CAR_SIZE//2 - 8, 20),
            (CAR_SIZE//2 + 8, 20)
        ]
        pygame.draw.polygon(rotated_car, BLACK, arrow_points)
        # 贴图到屏幕（修正旋转后的坐标偏移）
        surface.blit(
            rotated_car,
            (self.x - rotated_car.get_width()//2, self.y - rotated_car.get_height()//2)
        )

    def move(self, action):
        # 根据动作更新位置和朝向
        dx, dy = 0, 0
        if action == "forward":
            dx = CAR_SPEED * np.sin(np.radians(self.angle))
            dy = -CAR_SPEED * np.cos(np.radians(self.angle))
        elif action == "backward":
            dx = -CAR_SPEED * np.sin(np.radians(self.angle))
            dy = CAR_SPEED * np.cos(np.radians(self.angle))
        elif action == "left":
            self.angle = (self.angle - 5) % 360  # 左转5度
        elif action == "right":
            self.angle = (self.angle + 5) % 360  # 右转5度

        # 边界碰撞检测
        new_x = self.x + dx
        new_y = self.y + dy
        if 0 <= new_x <= WINDOW_WIDTH - CAR_SIZE and 0 <= new_y <= WINDOW_HEIGHT - CAR_SIZE:
            self.x = new_x
            self.y = new_y

class Obstacle:
    def __init__(self):
        # 随机生成障碍物（不与初始车位置重叠）
        while True:
            self.x = np.random.randint(0, WINDOW_WIDTH - OBSTACLE_SIZE)
            self.y = np.random.randint(0, WINDOW_HEIGHT - OBSTACLE_SIZE)
            if not (
                abs(self.x - (WINDOW_WIDTH//2)) < CAR_SIZE + OBSTACLE_SIZE and
                abs(self.y - (WINDOW_HEIGHT//2)) < CAR_SIZE + OBSTACLE_SIZE
            ):
                break

    def draw(self, surface):
        pygame.draw.rect(surface, BLACK, (self.x, self.y, OBSTACLE_SIZE, OBSTACLE_SIZE))

# ========================= 4. 深度学习模型（CNN） =========================
def build_model():
    """构建用于无人车控制的CNN模型（输入：环境图像，输出：动作概率）"""
    model = models.Sequential([
        # 输入层：64x64x3（RGB图像）
        layers.Conv2D(32, (3, 3), activation='relu', input_shape=(IMAGE_SIZE[0], IMAGE_SIZE[1], 3)),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(64, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(128, activation='relu'),
        layers.Dropout(0.5),  # 防止过拟合
        layers.Dense(len(ACTIONS), activation='softmax')  # 输出动作概率
    ])
    model.compile(
        optimizer='adam',
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

# 加载或创建模型
if os.path.exists(MODEL_PATH):
    model = tf.keras.models.load_model(MODEL_PATH)
    print(f"已加载现有模型：{MODEL_PATH}")
else:
    model = build_model()
    print("已创建新模型")

# ========================= 5. 工具函数 =========================
def get_environment_image(screen):
    """获取当前环境图像（用于模型输入）"""
    # 将Pygame屏幕转换为OpenCV图像（RGB→BGR，尺寸缩放）
    screen_surf = pygame.surfarray.array3d(screen)
    screen_surf = cv2.cvtColor(screen_surf, cv2.COLOR_RGB2BGR)
    screen_surf = cv2.resize(screen_surf, IMAGE_SIZE)
    # 归一化（0-255→0-1）
    return screen_surf / 255.0

def record_training_data(image, action):
    """记录训练数据（环境图像+对应的键盘动作）"""
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S%f")
    action_idx = ACTION_TO_IDX[action]
    # 保存图像
    img_path = os.path.join(DATA_DIR, f"{timestamp}_{action_idx}.png")
    cv2.imwrite(img_path, image * 255)  # 还原为0-255保存
    # 保存动作标签（单独文件，方便后续读取）
    with open(os.path.join(DATA_DIR, "labels.txt"), "a") as f:
        f.write(f"{img_path},{action_idx}\n")

def train_model_from_data():
    """从记录的训练数据中训练模型"""
    # 读取标签文件
    labels_path = os.path.join(DATA_DIR, "labels.txt")
    if not os.path.exists(labels_path):
        print("无训练数据，跳过训练")
        return

    # 加载数据和标签
    X, y = [], []
    with open(labels_path, "r") as f:
        for line in f.readlines():
            img_path, action_idx = line.strip().split(",")
            if os.path.exists(img_path):
                # 读取图像并预处理
                img = cv2.imread(img_path)
                img = cv2.resize(img, IMAGE_SIZE) / 255.0
                X.append(img)
                y.append(int(action_idx))

    if len(X) < 10:
        print("训练数据不足（至少需要10条）")
        return

    # 转换为numpy数组
    X = np.array(X)
    y = np.array(y)

    # 训练模型
    print(f"开始训练，样本数：{len(X)}")
    model.fit(
        X, y,
        epochs=10,
        batch_size=32,
        validation_split=0.2  # 20%数据用于验证
    )

    # 保存模型
    model.save(MODEL_PATH)
    print(f"模型已保存到：{MODEL_PATH}")

# ========================= 6. 主控制逻辑 =========================
def main():
    car = Car()
    obstacles = [Obstacle() for _ in range(OBSTACLE_NUM)]
    mode = "manual"  # 初始模式：手动控制（manual）/自动控制（auto）
    current_action = "stop"

    running = True
    while running:
        # 1. 填充背景
        screen.fill(WHITE)

        # 2. 事件处理
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            # 切换模式（M键）
            if event.type == pygame.KEYDOWN:
                if event.key == pygame.K_m:
                    mode = "auto" if mode == "manual" else "manual"
                    print(f"模式切换为：{mode}")
                # 训练模型（T键）
                if event.key == pygame.K_t:
                    print("开始训练模型...")
                    train_model_from_data()
                # 退出（ESC键）
                if event.key == pygame.K_ESCAPE:
                    running = False

        # 3. 控制逻辑（手动/自动）
        if mode == "manual":
            # 手动控制：方向键映射动作
            keys = pygame.key.get_pressed()
            if keys[pygame.K_UP]:
                current_action = "forward"
            elif keys[pygame.K_DOWN]:
                current_action = "backward"
            elif keys[pygame.K_LEFT]:
                current_action = "left"
            elif keys[pygame.K_RIGHT]:
                current_action = "right"
            else:
                current_action = "stop"

            # 记录训练数据（手动模式下自动记录）
            env_img = get_environment_image(screen)
            record_training_data(env_img, current_action)

        else:
            # 自动控制：模型预测动作
            env_img = get_environment_image(screen)
            # 模型输入：扩展为批量维度（(64,64,3)→(1,64,64,3)）
            env_img_batch = np.expand_dims(env_img, axis=0)
            action_probs = model.predict(env_img_batch, verbose=0)[0]
            current_action = IDX_TO_ACTION[np.argmax(action_probs)]  # 选择概率最大的动作

        # 4. 无人车移动
        car.move(current_action)

        # 5. 绘制元素（障碍物→无人车→提示文字）
        for obs in obstacles:
            obs.draw(screen)
        car.draw(screen)

        # 6. 显示提示信息
        hint_texts = [
            f"模式：{mode}（按M切换）",
            f"当前动作：{current_action}",
            "手动控制：↑↓←→ 移动",
            "按T训练模型 | 按ESC退出"
        ]
        for i, text in enumerate(hint_texts):
            surf = font.render(text, True, BLACK if mode == "manual" else GREEN)
            screen.blit(surf, (20, 20 + i*35))

        # 7. 更新屏幕
        pygame.display.flip()
        clock.tick(30)  # 30帧/秒

    # 退出程序
    pygame.quit()
    print("程序退出")

if __name__ == "__main__":
    main()
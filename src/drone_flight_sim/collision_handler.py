# collision_handler.py
"""碰撞检测和处理模块（深度学习增强版）

本模块负责检测无人机飞行过程中的碰撞事件，
使用神经网络决策避障方向和距离。
"""

import time
import random
import torch
import torch.nn as nn
import numpy as np
from config import FlightConfig, GROUND_OBJECTS


# ========== 新增：神经网络决策模型 ==========
class CollisionAvoidanceNN(nn.Module):
    """碰撞避障神经网络
    
    输入特征：
    - 撞击方向 x (normal.x)
    - 撞击方向 y (normal.y)
    - 撞击速度
    - 当前高度
    - 碰撞物体类型编码（0=墙,1=树,2=建筑,3=其他）
    
    输出：
    - 避障方向 (-1=左后, 0=正后, 1=右后)
    - 避障距离 (2~5米)
    - 上升高度 (1~3米)
    """
    def __init__(self):
        super().__init__()
        # 输入层：5个特征
        self.fc1 = nn.Linear(5, 16)
        self.fc2 = nn.Linear(16, 32)
        self.fc3 = nn.Linear(32, 16)
        # 输出层：3个值（方向、距离、上升高度）
        self.fc_out = nn.Linear(16, 3)
        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()  # 方向用tanh（-1到1）
        self.sigmoid = nn.Sigmoid()  # 距离和高度用sigmoid再映射
    
    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))
        x = self.relu(self.fc3(x))
        out = self.fc_out(x)
        
        # 解析输出
        direction = self.tanh(out[:, 0])  # -1 到 1
        distance = self.sigmoid(out[:, 1]) * 3 + 2  # 2 到 5 米
        height_up = self.sigmoid(out[:, 2]) * 2 + 1  # 1 到 3 米
        
        return direction, distance, height_up


# 全局模型
_avoidance_model = None
_model_trained = False


def train_avoidance_model():
    """训练碰撞避障神经网络"""
    print("\n🧠 训练碰撞避障神经网络...")
    
    # 训练数据：[撞击方向x, 撞击方向y, 撞击速度, 高度, 物体类型]
    # 物体类型: 0=墙, 1=树, 2=建筑, 3=其他
    X_train = [
        # 撞到墙的情况
        [1.0, 0.0, 3.5, 5.0, 0],   # 右边撞墙，高速
        [-1.0, 0.0, 3.0, 4.0, 0],  # 左边撞墙
        [0.8, 0.3, 2.5, 6.0, 0],   # 右前方撞墙
        
        # 撞到树的情况
        [0.6, 0.6, 2.0, 3.0, 1],   # 右前撞树
        [-0.5, -0.5, 1.8, 2.5, 1], # 左后撞树
        [0.0, 1.0, 2.2, 4.0, 1],   # 正前方撞树
        
        # 撞到建筑
        [0.9, 0.1, 4.0, 8.0, 2],   # 高速撞建筑
        [-0.9, -0.1, 3.8, 7.0, 2], # 左撞建筑
        
        # 其他物体
        [0.4, 0.4, 1.5, 2.0, 3],   # 轻微碰撞
        [-0.3, 0.5, 1.2, 3.0, 3],  # 侧面碰撞
    ]
    
    # 期望输出：[避障方向(-1~1), 避障距离(2~5), 上升高度(1~3)]
    # 方向: -1=左后, 0=正后, 1=右后
    y_train = [
        [-0.8, 4.0, 2.5],  # 右撞墙 → 向左后大距离后退
        [0.8, 3.5, 2.0],   # 左撞墙 → 向右后退
        [0.5, 3.0, 2.0],   # 右前撞墙 → 向右后
        
        [0.3, 2.5, 1.5],   # 撞树 → 轻微偏右
        [-0.5, 2.0, 1.0],  # 撞树 → 向左后
        [-0.2, 2.5, 1.5],  # 正前撞树 → 正后方
        
        [-0.9, 4.5, 3.0],  # 撞建筑 → 大距离左后
        [0.7, 4.0, 2.5],   # 左撞建筑 → 右后
        
        [0.0, 2.0, 1.0],   # 轻微碰撞 → 正后短距离
        [-0.3, 2.2, 1.2],  # 侧面碰撞 → 稍左
    ]
    
    # 转换为 tensor
    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)
    
    model = CollisionAvoidanceNN()
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    print("   训练迭代中...")
    for epoch in range(300):
        optimizer.zero_grad()
        direction, distance, height_up = model(X)
        
        # 组合输出用于计算损失
        pred = torch.stack([direction, distance, height_up], dim=1)
        loss = criterion(pred, y)
        
        loss.backward()
        optimizer.step()
        
        if epoch % 100 == 0:
            print(f"   Epoch {epoch}, Loss: {loss.item():.6f}")
    
    # 保存模型
    torch.save(model.state_dict(), "collision_avoidance.pth")
    print(f"✅ 避障神经网络训练完成！最终损失: {loss.item():.6f}")
    print("   模型已保存为 collision_avoidance.pth\n")
    
    return model


def load_or_train_model():
    """加载已有模型或训练新模型"""
    global _avoidance_model, _model_trained
    try:
        _avoidance_model = CollisionAvoidanceNN()
        _avoidance_model.load_state_dict(torch.load("collision_avoidance.pth"))
        _avoidance_model.eval()
        print("📦 加载已有避障神经网络模型")
    except:
        print("⚠️  未找到已有模型，开始训练...")
        _avoidance_model = train_avoidance_model()
        _avoidance_model.eval()
    _model_trained = True
    return _avoidance_model


def get_object_type(object_name: str) -> int:
    """根据物体名称返回类型编码"""
    obj_lower = object_name.lower()
    if any(kw in obj_lower for kw in ['wall', 'fence']):
        return 0
    elif any(kw in obj_lower for kw in ['tree', 'bush', 'plant']):
        return 1
    elif any(kw in obj_lower for kw in ['building', 'house', 'tower']):
        return 2
    else:
        return 3


# ========== 主碰撞处理类 ==========
class CollisionHandler:
    """碰撞检测处理器类（神经网络增强版）"""

    def __init__(self, client):
        self.client = client
        self.collision_count = 0
        self.last_collision_time = 0
        self.is_collided = False
        self.auto_recovery_attempts = 0
        self.max_auto_recovery_attempts = FlightConfig.MAX_AUTO_RECOVERY_ATTEMPTS
        
        # 【新增】加载神经网络模型
        load_or_train_model()

    def check_collision(self):
        """检测碰撞事件（保持原有逻辑）"""
        collision_info = self.client.simGetCollisionInfo()

        if not collision_info.has_collided:
            return False, None

        current_time = time.time()
        if current_time - self.last_collision_time < FlightConfig.COLLISION_COOLDOWN:
            return False, None

        self.last_collision_time = current_time
        self.collision_count += 1

        drone_pos = self.client.getMultirotorState().kinematics_estimated.position
        current_height = -drone_pos.z_val

        # 判断是否为地面接触
        is_ground = (
            current_height < FlightConfig.GROUND_HEIGHT_THRESHOLD or
            any(keyword in collision_info.object_name for keyword in GROUND_OBJECTS)
        )

        if is_ground:
            print(f"⚠️  检测到与 {collision_info.object_name} 接触（高度: {current_height:.2f}m），忽略")
            return False, None

        print(f"\n💥 严重碰撞发生！")
        print(f"   碰撞位置: ({collision_info.position.x_val:.2f}, "
              f"{collision_info.position.y_val:.2f}, {collision_info.position.z_val:.2f})")
        print(f"   碰撞物体: {collision_info.object_name}")
        print(f"   当前高度: {current_height:.2f}m")
        print(f"   碰撞次数: {self.collision_count}")

        return True, collision_info

    def auto_recover(self):
        """自动恢复碰撞（神经网络决策版）"""
        self.auto_recovery_attempts += 1

        if self.auto_recovery_attempts > self.max_auto_recovery_attempts:
            return False

        print(f"\n🔧 尝试自动恢复 ({self.auto_recovery_attempts}/{self.max_auto_recovery_attempts})...")

        try:
            # 1. 取消当前任务
            self.client.cancelLastTask()
            time.sleep(0.5)

            # 2. 获取当前位置和碰撞信息
            pos = self.client.getMultirotorState().kinematics_estimated.position
            collision_info = self.client.simGetCollisionInfo()
            
            # 3. 【神经网络决策】获取避障参数
            # 提取特征
            normal_x = collision_info.normal.x_val if hasattr(collision_info, 'normal') else 0
            normal_y = collision_info.normal.y_val if hasattr(collision_info, 'normal') else 0
            impact_speed = abs(collision_info.impact_speed) if hasattr(collision_info, 'impact_speed') else 2.0
            current_height = -pos.z_val
            obj_type = get_object_type(collision_info.object_name)
            
            # 特征向量
            features = torch.tensor([[
                normal_x, normal_y, impact_speed, current_height, obj_type
            ]], dtype=torch.float32)
            
            # 神经网络推理
            with torch.no_grad():
                direction, distance, height_up = _avoidance_model(features)
                direction_val = direction.item()
                distance_val = distance.item()
                height_up_val = height_up.item()
            
            # 打印决策信息
            dir_desc = "左后" if direction_val < -0.3 else ("右后" if direction_val > 0.3 else "正后")
            print(f"🧠 神经网络避障决策:")
            print(f"   方向: {dir_desc} (系数: {direction_val:.2f})")
            print(f"   后退距离: {distance_val:.1f}米")
            print(f"   上升高度: {height_up_val:.1f}米")
            
            # 4. 执行避障移动
            # 根据神经网络输出计算移动方向
            new_x = pos.x_val - distance_val * 0.7  # 后退
            new_y = pos.y_val + direction_val * distance_val * 0.7  # 横向偏移
            new_z = pos.z_val - height_up_val  # 上升（Z轴负方向）
            
            print(f"   执行避障移动...")
            self.client.moveToPositionAsync(new_x, new_y, new_z, FlightConfig.FLIGHT_VELOCITY)
            time.sleep(3)
            
            # 5. 悬停等待稳定
            self.client.hoverAsync().join()
            time.sleep(1)

            # 6. 检查是否脱离碰撞
            collision_info = self.client.simGetCollisionInfo()
            if not collision_info.has_collided:
                print(f"✅ 神经网络引导的自动恢复成功！")
                self.auto_recovery_attempts = 0
                return True
            else:
                print(f"⚠️  自动恢复后仍处于碰撞状态")
                return False

        except Exception as e:
            print(f"❌ 自动恢复失败: {e}")
            return False

    def request_manual_control(self):
        """请求手动接管控制（保持不变）"""
        print(f"\n{'=' * 50}")
        print(f"🚨 自动恢复失败，需要手动接管！")
        print(f"{'=' * 50}")
        print(f"""
📋 手动接管说明:
   碰撞后无人机可能处于卡住状态，请使用键盘控制：
   
   键盘控制说明:
   - W/↑     : 前进
   - S/↓     : 后退
   - A       : 向左移动
   - D       : 向右移动
   - Q       : 上升
   - E       : 下降
   - 空格    : 悬停
   - L       : 执行降落
   - ESC     : 紧急停止并退出

💡 提示: 先按 Q 上升脱离碰撞，然后按 L 降落
""")
        print(f"{'=' * 50}\n")
        return True

    def reset_collision_state(self):
        """重置碰撞状态"""
        self.is_collided = False
        self.auto_recovery_attempts = 0


# ========== 单独测试入口 ==========
if __name__ == "__main__":
    print("🚁 碰撞避障神经网络测试")
    print("="*40)
    
    # 训练模型
    train_avoidance_model()
    
    # 测试决策
    print("\n🧪 测试神经网络决策效果:")
    model = load_or_train_model()
    
    test_cases = [
        ([1.0, 0.0, 3.5, 5.0, 0], "撞墙"),
        ([0.6, 0.6, 2.0, 3.0, 1], "撞树"),
        ([0.9, 0.1, 4.0, 8.0, 2], "撞建筑"),
        ([0.4, 0.4, 1.5, 2.0, 3], "轻微碰撞"),
    ]
    
    with torch.no_grad():
        for features, desc in test_cases:
            x = torch.tensor([features], dtype=torch.float32)
            direction, distance, height_up = model(x)
            dir_str = "左后" if direction.item() < -0.3 else ("右后" if direction.item() > 0.3 else "正后")
            print(f"  {desc}: {dir_str} 退{distance.item():.1f}m 升{height_up.item():.1f}m")
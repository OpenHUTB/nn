import os
import sys
import io

# 设置UTF-8编码
sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8')

# 将当前目录添加到系统路径
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import time
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import transforms
import matplotlib.pyplot as plt
from datetime import datetime

# 导入自定义模块
try:
    from Data_classfication import split_dataset
    from image_classification import ImageDataset, ImageClassifier
    from visual_navigation import main as run_visual_navigation
    from forecast import predict_image, batch_predict
    from path_planning import DynamicPathPlanner, PathFollower, Node, Obstacle
    from environment_3d import (TerrainGenerator, TerrainConfig, 
                               Environment3DVisualizer, Drone3DModel)
except ImportError as e:
    print(f"导入模块时出错: {e}")
    print("请确保所有模块文件都在同一目录下")
    sys.exit(1)

# 路径设置
base_dir = "data"
train_dir = os.path.join(base_dir, "train")
test_dir = os.path.join(base_dir, "test")
dataset_dir = os.path.join(base_dir, "dataset")

# 获取当前目录
current_dir = os.getcwd()
# 日志系统
class Logger:
    """日志记录器"""
    
    def __init__(self, log_file="drone_system.log"):
        self.log_file = log_file
        self.logs = []
        
    def log(self, message, level="INFO"):
        """记录日志"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        log_entry = f"[{timestamp}] [{level}] {message}"
        
        print(log_entry)
        self.logs.append(log_entry)
        
        # 写入文件 - 修复编码问题
        try:
            with open(self.log_file, 'a', encoding='utf-8', errors='ignore') as f:
                f.write(log_entry + '\n')
        except Exception as e:
            print(f"写入日志文件失败: {e}")
    
    def info(self, message):
        """信息级别日志"""
        self.log(message, "INFO")
    
    def warning(self, message):
        """警告级别日志"""
        self.log(message, "WARNING")
    
    def error(self, message):
        """错误级别日志"""
        self.log(message, "ERROR")
    
    def success(self, message):
        """成功级别日志"""
        self.log(message, "SUCCESS")

# ==================== 初始化日志 ====================
logger = Logger()

def setup_directories():
    """设置数据目录"""
    logger.info("=" * 50)
    logger.info("设置数据目录...")
    
    # 检查并创建目录
    os.makedirs(base_dir, exist_ok=True)
    os.makedirs(train_dir, exist_ok=True)
    os.makedirs(test_dir, exist_ok=True)
    
    # 创建新增功能目录
    os.makedirs(os.path.join(base_dir, "paths"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "environments"), exist_ok=True)
    os.makedirs(os.path.join(base_dir, "logs"), exist_ok=True)
    
    # 检查是否需要分割数据集
    if not os.path.exists(train_dir) or not os.listdir(train_dir):
        logger.info("训练集不存在或为空，开始自动分割数据集...")
        if os.path.exists(dataset_dir):
            success = split_dataset(dataset_dir, train_dir, test_dir, split_ratio=0.8)
            if not success:
                logger.error("数据集分割失败，请检查原始数据集路径")
                return False
        else:
            logger.error(f"原始数据集路径不存在: {dataset_dir}")
            logger.info("请将数据集放入 ./data/dataset/ 目录")
            logger.info("数据集结构应为:")
            logger.info("data/dataset/")
            logger.info("├── 类别1/")
            logger.info("│   ├── image1.jpg")
            logger.info("│   └── image2.jpg")
            logger.info("├── 类别2/")
            logger.info("│   ├── image1.jpg")
            logger.info("│   └── image2.jpg")
            logger.info("└── ...")
            return False
    else:
        logger.success("训练集已存在，跳过数据集分割步骤")
    
    return True

def train_pytorch_model():
    """使用PyTorch训练模型"""
    logger.info("\n" + "=" * 50)
    logger.info("开始PyTorch模型训练...")
    
    # 参数配置
    img_size = (128, 128)
    batch_size = 32
    epochs = 70
    
    # 设置设备
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    logger.info(f"使用设备: {device}")
    
    # 数据预处理
    train_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.RandomRotation(30),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), shear=0.2),
        transforms.ColorJitter(brightness=0.2, contrast=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    test_transform = transforms.Compose([
        transforms.Resize(img_size),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # 创建数据集
    train_dataset = ImageDataset(train_dir, transform=train_transform)
    test_dataset = ImageDataset(test_dir, transform=test_transform)
    
    if len(train_dataset) == 0:
        logger.error("训练集为空，无法训练模型")
        return None, [], []
    
    num_classes = len(train_dataset.class_to_idx)
    logger.info(f"检测到 {num_classes} 个类别: {train_dataset.class_to_idx}")
    
    # 创建数据加载器
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    
    # 初始化模型
    model = ImageClassifier(num_classes=num_classes).to(device)
    
    # 定义损失函数和优化器
    criterion = nn.CrossEntropyLoss()
    import torch.optim as optim
    optimizer = optim.Adam(model.parameters(), lr=0.001)
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
    
    # 训练模型
    best_accuracy = 0.0
    train_losses = []
    val_accuracies = []
    
    for epoch in range(epochs):
        model.train()
        running_loss = 0.0
        
        for images, labels in train_loader:
            images, labels = images.to(device), labels.to(device)
            
            outputs = model(images)
            loss = criterion(outputs, labels)
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item() * images.size(0)
        
        epoch_loss = running_loss / len(train_loader.dataset)
        train_losses.append(epoch_loss)
        
        # 验证
        model.eval()
        all_preds = []
        all_labels = []
        
        with torch.no_grad():
            for images, labels in test_loader:
                images, labels = images.to(device), labels.to(device)
                outputs = model(images)
                _, preds = torch.max(outputs, 1)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
        
        accuracy = np.mean(np.array(all_labels) == np.array(all_preds))
        val_accuracies.append(accuracy)
        
        logger.info(f'Epoch [{epoch+1}/{epochs}], Loss: {epoch_loss:.4f}, Accuracy: {accuracy:.4f}')
        
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(model.state_dict(), os.path.join(base_dir, "best_model.pth"))
            logger.success(f"保存最佳模型，准确率: {accuracy:.4f}")
        
        scheduler.step()
    
    # 保存最终模型
    torch.save(model.state_dict(), os.path.join(base_dir, "final_model.pth"))
    logger.success("最终模型已保存")
    
    # 绘制训练曲线
    plt.figure(figsize=(12, 4))
    plt.subplot(1, 2, 1)
    plt.plot(train_losses)
    plt.title('Training Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    
    plt.subplot(1, 2, 2)
    plt.plot(val_accuracies)
    plt.title('Validation Accuracy')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    
    plt.tight_layout()
    plt.savefig(os.path.join(base_dir, "training_plot.png"))
    logger.success("训练曲线图已保存")
    plt.show()
    
    return model, train_losses, val_accuracies

def run_path_planning_demo():
    """运行路径规划演示"""
    logger.info("\n" + "=" * 50)
    logger.info("开始路径规划演示...")
    
    try:
        # 创建路径规划器
        planner = DynamicPathPlanner(grid_size=0.5, safety_margin=0.5)
        
        # 设置环境边界 (x, y, z)
        planner.set_environment_bounds([0, 0, 0], [50, 50, 20])
        
        # 添加障碍物
        obstacles = [
            Obstacle([10, 10, 5], radius=2.0, height=10),
            Obstacle([25, 20, 3], radius=3.0, height=6),
            Obstacle([40, 35, 4], radius=2.5, height=8),
            Obstacle([15, 40, 2], radius=1.5, height=4)
        ]
        
        for obs in obstacles:
            planner.add_obstacle(obs)
            logger.info(f"添加障碍物: 位置({obs.position[0]}, {obs.position[1]}, {obs.position[2]}), "
                       f"半径{obs.radius}米, 高度{obs.height}米")
        
        # 定义起点和终点
        start = Node(5, 5, 3)
        goal = Node(45, 45, 5)
        
        logger.info(f"起点: ({start.x}, {start.y}, {start.z})")
        logger.info(f"终点: ({goal.x}, {goal.y}, {goal.z})")
        
        # 执行路径规划
        logger.info("正在规划路径...")
        path = planner.hybrid_plan(start, goal)
        
        if path:
            logger.success("路径规划成功！")
            
            # 计算路径长度
            path_length = 0
            for i in range(len(path) - 1):
                path_length += path[i].distance_to(path[i+1])
            logger.info(f"路径节点数: {len(path)}")
            logger.info(f"路径长度: {path_length:.2f} 米")
            
            # 可视化结果
            planner.visualize_path(path, "无人机路径规划结果")
            
            # 导出路径
            path_file = os.path.join(base_dir, "paths", f"path_{int(time.time())}.json")
            planner.export_path_to_json(path, path_file)
            
            # 创建路径跟随器并模拟
            follower = PathFollower(lookahead_distance=1.5, max_velocity=3.0)
            follower.set_path(path)
            
            logger.info("模拟路径跟随...")
            current_pos = np.array([start.x, start.y, start.z])
            current_vel = np.array([0, 0, 0])
            
            for step in range(50):
                target_pos, completed = follower.get_next_target(current_pos)
                
                if completed:
                    logger.success("路径跟随完成！")
                    break
                
                desired_vel, desired_acc = follower.compute_control_command(
                    current_pos, current_vel, target_pos
                )
                
                # 更新位置（简单积分）
                current_vel = desired_vel
                current_pos += current_vel * 0.1
                
                if step % 10 == 0:
                    logger.info(f"步骤 {step+1}: 位置 {current_pos.round(2)}, 目标 {target_pos.round(2)}")
            
            return path, planner
        
        else:
            logger.error("路径规划失败！")
            return None, None
            
    except Exception as e:
        logger.error(f"路径规划演示出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None

def run_3d_environment_demo():
    """运行3D环境创建演示"""
    logger.info("\n" + "=" * 50)
    logger.info("开始3D环境创建演示...")
    
    try:
        # 创建地形配置
        config = TerrainConfig(
            width=80.0,
            length=80.0,
            height_range=15.0,
            resolution=2.0,
            seed=int(time.time()),
            has_hills=True,
            has_valleys=True,
            has_river=True,
            vegetation_density=0.15,
            tree_density=0.03,
            water_level=3.0
        )
        
        # 创建地形生成器
        terrain_gen = TerrainGenerator(config)
        
        # 生成地形
        logger.info("正在生成地形...")
        heightmap = terrain_gen.generate_terrain()
        
        # 添加建筑物障碍物
        logger.info("添加障碍物...")
        buildings = [
            ([20, 20, 0], [8, 6, 12]),
            ([50, 30, 0], [10, 8, 15]),
            ([35, 60, 0], [6, 6, 10]),
        ]
        
        for pos, dim in buildings:
            terrain_gen.add_building(pos, dim)
            logger.info(f"  添加建筑物: 位置{pos}, 尺寸{dim}")
        
        # 添加树木障碍物
        for i in range(15):
            x = np.random.uniform(10, 70)
            y = np.random.uniform(10, 70)
            terrain_gen.add_tree_obstacle((x, y))
        
        logger.info(f"总障碍物数量: {len(terrain_gen.obstacles_mesh)}")
        
        # 创建无人机模型
        logger.info("创建无人机模型...")
        drone1 = Drone3DModel(model_type="quadcopter", scale=1.5)
        drone1.update_pose(np.array([10, 10, 5]), np.array([0, 0, 0]))
        
        drone2 = Drone3DModel(model_type="quadcopter", scale=1.5)
        drone2.update_pose(np.array([70, 70, 5]), np.array([0, 0, np.pi]))
        
        drones = [drone1, drone2]
        logger.info(f"创建了 {len(drones)} 架无人机")
        
        # 创建示例路径
        logger.info("创建示例路径...")
        path1 = np.array([
            [10, 10, 8],
            [25, 20, 12],
            [40, 35, 15],
            [55, 50, 12],
            [70, 70, 8]
        ])
        
        path2 = np.array([
            [70, 70, 10],
            [55, 55, 14],
            [40, 40, 16],
            [25, 25, 14],
            [10, 10, 10]
        ])
        
        paths = [path1, path2]
        logger.info(f"创建了 {len(paths)} 条路径")
        
        # 可视化环境
        logger.info("开始3D可视化...")
        visualizer = Environment3DVisualizer(render_engine="plotly")
        visualizer.visualize_terrain(terrain_gen, drones, paths)
        
        # 保存可视化结果
        env_file = os.path.join(base_dir, "environments", f"environment_{int(time.time())}.html")
        visualizer.save_visualization(env_file)
        
        # 导出环境数据
        env_data_file = os.path.join(base_dir, "environments", f"environment_data_{int(time.time())}.json")
        terrain_gen.export_environment(env_data_file)
        
        # 显示地形统计信息
        if terrain_gen.terrain_mesh:
            vertices = terrain_gen.terrain_mesh.vertices
            min_height = vertices[:, 2].min()
            max_height = vertices[:, 2].max()
            avg_height = vertices[:, 2].mean()
            
            logger.info("\n地形统计信息:")
            logger.info(f"  最低点: {min_height:.2f} 米")
            logger.info(f"  最高点: {max_height:.2f} 米")
            logger.info(f"  平均高度: {avg_height:.2f} 米")
            logger.info(f"  地形范围: {config.width} × {config.length} 米")
            logger.info(f"  顶点数量: {len(vertices)}")
        
        logger.success("3D环境创建完成！")
        return terrain_gen, drones, paths, visualizer
        
    except Exception as e:
        logger.error(f"3D环境创建演示出错: {e}")
        import traceback
        logger.error(traceback.format_exc())
        return None, None, None, None

def integrated_navigation_demo():
    """集成导航演示：路径规划 + 3D环境 + 图像分类"""
    logger.info("\n" + "=" * 50)
    logger.info("开始集成导航演示...")
    
    try:
        # 1. 创建3D环境
        logger.info("阶段1: 创建3D导航环境")
        terrain_gen, drones, paths, visualizer = run_3d_environment_demo()
        
        if terrain_gen is None:
            logger.error("3D环境创建失败，中止集成演示")
            return
        
        # 2. 在环境中进行路径规划
        logger.info("\n阶段2: 在3D环境中进行路径规划")
        
        # 从3D环境中提取边界
        config = terrain_gen.config
        min_bound = [0, 0, 0]
        max_bound = [config.width, config.length, config.height_range]
        
        # 创建路径规划器
        planner = DynamicPathPlanner(grid_size=2.0, safety_margin=1.0)
        planner.set_environment_bounds(min_bound, max_bound)
        
        # 将3D环境中的障碍物添加到路径规划器
        for obstacle in terrain_gen.obstacles_mesh:
            center = obstacle.centroid
            bounds = obstacle.bounds
            radius = max(bounds[1][0] - bounds[0][0], 
                        bounds[1][1] - bounds[0][1]) / 2
            height = bounds[1][2] - bounds[0][2]
            
            planner_obstacle = Obstacle(center, radius=radius, height=height)
            planner.add_obstacle(planner_obstacle)
        
        # 规划路径
        start = Node(10, 10, 5)
        goal = Node(70, 70, 10)
        
        logger.info(f"规划从 {start} 到 {goal} 的路径")
        path = planner.hybrid_plan(start, goal)
        
        if path:
            logger.success(f"路径规划成功，找到 {len(path)} 个路径点")
            
            # 将路径转换为numpy数组用于3D可视化
            path_array = np.array([[node.x, node.y, node.z] for node in path])
            
            # 更新3D可视化
            logger.info("更新3D可视化显示规划路径...")
            visualizer.visualize_terrain(terrain_gen, drones, [path_array])
            
            # 3. 模拟无人机沿路径飞行并进行图像分类
            logger.info("\n阶段3: 模拟无人机飞行与实时图像分类")
            
            # 检查是否有训练好的模型
            model_path = os.path.join(base_dir, "best_model.pth")
            if os.path.exists(model_path):
                logger.info("检测到训练好的模型，加载模型...")
                
                # 这里可以集成图像分类功能
                # 在实际系统中，这里会从无人机摄像头获取图像并进行分类
                logger.info("模拟图像分类过程...")
                logger.info("飞行过程中检测到: 森林、建筑物、道路等场景")
                logger.info("根据场景调整飞行策略...")
                
                # 模拟根据图像分类结果调整路径
                logger.info("检测到前方有建筑物，调整飞行高度...")
                logger.info("检测到森林区域，启用避障模式...")
                
            else:
                logger.warning("未找到训练好的模型，跳过图像分类模拟")
            
            # 4. 路径跟随模拟
            logger.info("\n阶段4: 路径跟随控制模拟")
            follower = PathFollower(lookahead_distance=2.0, max_velocity=2.5)
            follower.set_path(path)
            
            # 模拟飞行过程
            current_pos = np.array([start.x, start.y, start.z])
            logger.info(f"开始飞行，当前位置: {current_pos}")
            
            for step in range(30):
                target_pos, completed = follower.get_next_target(current_pos)
                
                if completed:
                    logger.success("到达目的地！")
                    break
                
                # 模拟控制指令
                desired_vel = np.array([1.0, 1.0, 0.2])  # 简化控制
                current_pos += desired_vel * 0.1
                
                if step % 5 == 0:
                    distance_to_target = np.linalg.norm(target_pos - current_pos)
                    logger.info(f"步骤 {step+1}: 位置 {current_pos.round(2)}, "
                              f"距离目标: {distance_to_target:.2f}米")
            
            logger.success("集成导航演示完成！")
            
        else:
            logger.error("路径规划失败")
            
    except Exception as e:
        logger.error(f"集成导航演示出错: {e}")
        import traceback
        logger.error(traceback.format_exc())

def display_menu():
    """显示主菜单"""
    print("\n" + "=" * 60)
    print("🚀 无人机智能导航系统 v2.0")
    print("=" * 60)
    print("1. 训练图像分类模型")
    print("2. 单张图像预测")
    print("3. 批量图像预测")
    print("4. 启动视觉导航（摄像头实时分类）")
    print("5. 路径规划演示")
    print("6. 3D环境创建演示")
    print("7. 集成导航演示（路径规划 + 3D环境 + 图像分类）")
    print("8. 查看系统日志")
    print("9. 退出系统")
    print("=" * 60)

def main():
    """主函数"""
    logger.info(" 启动无人机智能导航系统...")
    logger.info(f"工作目录: {current_dir}")
    logger.info(f"数据目录: {base_dir}")
    
    # 检查CUDA可用性
    if torch.cuda.is_available():
        logger.success(f"检测到CUDA设备: {torch.cuda.get_device_name(0)}")
    else:
        logger.info("使用CPU进行计算")
    
    # 1. 设置数据目录
    if not setup_directories():
        logger.error("目录设置失败，程序退出")
        return
    
    # 主循环
    while True:
        display_menu()
        
        try:
            choice = input("\n请输入选项编号 (1-9): ").strip()
            
            if choice == '1':
                # 训练图像分类模型
                logger.info("开始训练图像分类模型...")
                model, train_losses, val_accuracies = train_pytorch_model()
                
                if model is None:
                    logger.error("模型训练失败")
                else:
                    logger.success("模型训练完成！")
            
            elif choice == '2':
                # 单张图像预测
                logger.info("单张图像预测模式")
                test_image_path = input("请输入测试图像路径 (或按回车使用默认路径): ").strip()
                
                if not test_image_path:
                    # 使用默认测试图像
                    default_test_dir = os.path.join(test_dir)
                    if os.path.exists(default_test_dir):
                        # 查找第一个图像文件
                        for root, dirs, files in os.walk(default_test_dir):
                            for file in files:
                                if file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                    test_image_path = os.path.join(root, file)
                                    break
                            if test_image_path:
                                break
                
                if test_image_path and os.path.exists(test_image_path):
                    model_path = os.path.join(base_dir, "best_model.pth")
                    
                    if os.path.exists(model_path):
                        result = predict_image(model_path, test_image_path, train_dir)
                        if result:
                            predicted_class, confidence = result
                            logger.success(f"预测结果: {predicted_class} (置信度: {confidence*100:.2f}%)")
                    else:
                        logger.warning("未找到训练好的模型，请先训练模型 (选项1)")
                else:
                    logger.error(f"图像路径不存在: {test_image_path}")
            
            elif choice == '3':
                # 批量图像预测
                logger.info("批量图像预测模式")
                model_path = os.path.join(base_dir, "best_model.pth")
                
                if os.path.exists(model_path):
                    results = batch_predict(model_path, test_dir, train_dir)
                    if results:
                        logger.success(f"批量预测完成，处理了 {len(results)} 张图像")
                else:
                    logger.warning("未找到训练好的模型，请先训练模型 (选项1)")
            
            elif choice == '4':
                # 启动视觉导航
                logger.info("启动视觉导航系统...")
                model_path = os.path.join(base_dir, "best_model.pth")
                
                if os.path.exists(model_path):
                    try:
                        run_visual_navigation()
                        logger.success("视觉导航完成")
                    except Exception as e:
                        logger.error(f"视觉导航出错: {e}")
                else:
                    logger.warning("未找到训练好的模型，请先训练模型 (选项1)")
            
            elif choice == '5':
                # 路径规划演示
                path, planner = run_path_planning_demo()
                if path:
                    logger.success("路径规划演示完成")
            
            elif choice == '6':
                # 3D环境创建演示
                terrain_gen, drones, paths, visualizer = run_3d_environment_demo()
                if terrain_gen:
                    logger.success("3D环境创建演示完成")
            
            elif choice == '7':
                # 集成导航演示
                integrated_navigation_demo()
            
            elif choice == '8':
                # 查看系统日志
                logger.info("系统日志:")
                print("\n" + "=" * 60)
                print("最近日志记录:")
                print("=" * 60)
                
                # 显示最近10条日志
                recent_logs = logger.logs[-10:] if len(logger.logs) > 10 else logger.logs
                for log in recent_logs:
                    print(log)
                
                print("=" * 60)
                input("\n按回车键继续...")
            
            elif choice == '9':
                # 退出系统
                logger.info("感谢使用无人机智能导航系统！")
                print("\n🎉 程序正常退出")
                break
            
            else:
                logger.warning(f"无效选项: {choice}")
                
            # 每次操作后暂停一下
            input("\n按回车键返回主菜单...")
            
        except KeyboardInterrupt:
            logger.warning("用户中断操作")
            continue
        except Exception as e:
            logger.error(f"操作出错: {e}")
            import traceback
            logger.error(traceback.format_exc())
            input("\n按回车键继续...")

if __name__ == "__main__":
    # 添加必要的导入
    import torch.optim as optim
    
    try:
        main()
    except Exception as e:
        print(f"程序发生错误: {e}")
        import traceback
        print(traceback.format_exc())
        input("\n按回车键退出...")
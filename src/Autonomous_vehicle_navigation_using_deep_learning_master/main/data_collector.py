"""
数据收集和可视化模块 - 收集运行数据并绘制图表
使用中文标注，确保图表正确显示中文
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
import matplotlib
matplotlib.use('Agg')  # 使用非GUI后端

# 设置matplotlib使用中文字体
import matplotlib.font_manager as fm
import os

# 创建一个专门的中文字体设置函数
def setup_chinese_font():
    """设置中文字体，兼容Windows、Linux、Mac系统"""
    
    # 常见中文字体文件路径
    font_paths = [
        # Windows
        "C:/Windows/Fonts/simhei.ttf",      # 黑体
        "C:/Windows/Fonts/simsun.ttc",      # 宋体
        "C:/Windows/Fonts/simkai.ttf",      # 楷体
        "C:/Windows/Fonts/simfang.ttf",     # 仿宋
        "C:/Windows/Fonts/msyh.ttc",        # 微软雅黑
        "C:/Windows/Fonts/msyhbd.ttc",      # 微软雅黑粗体
        
        # Linux
        "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",  # 文泉驿微米黑
        "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",    # 文泉驿正黑
        "/usr/share/fonts/truetype/arphic/uming.ttc",      # 文鼎明体
        
        # Mac
        "/System/Library/Fonts/STHeiti Medium.ttc",        # 黑体
        "/System/Library/Fonts/STSong.ttf",                # 宋体
        "/System/Library/Fonts/AppleGothic.ttf",           # 苹果字体
        
        # 通用
        "simhei.ttf",
        "msyh.ttc",
    ]
    
    # 先尝试直接设置字体名称（适用于已安装字体的系统）
    font_names = [
        'SimHei',           # Windows黑体
        'Microsoft YaHei',  # Windows微软雅黑
        'STHeiti',          # Mac黑体
        'STSong',           # Mac宋体
        'WenQuanYi Micro Hei',  # Linux文泉驿微米黑
        'DejaVu Sans',      # 回退字体
        'Arial',            # 基本字体
    ]
    
    # 尝试使用字体名称
    for font_name in font_names:
        try:
            # 检查字体是否可用
            if any(font_name.lower() in f.name.lower() for f in fm.fontManager.ttflist):
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 使用系统字体: {font_name}")
                return True
        except:
            continue
    
    # 如果系统字体不可用，尝试从文件加载
    for font_path in font_paths:
        if os.path.exists(font_path):
            try:
                # 添加字体
                fm.fontManager.addfont(font_path)
                font_prop = fm.FontProperties(fname=font_path)
                font_name = font_prop.get_name()
                
                plt.rcParams['font.sans-serif'] = [font_name]
                plt.rcParams['axes.unicode_minus'] = False
                print(f"✅ 加载字体文件: {font_name} ({font_path})")
                return True
            except Exception as e:
                print(f"⚠️ 加载字体失败 {font_path}: {e}")
                continue
    
    # 如果所有方法都失败，尝试生成一个临时的中文字体解决方案
    print("⚠️ 无法找到合适的中文字体，将尝试使用回退方案")
    
    # 设置默认字体
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial']
    plt.rcParams['axes.unicode_minus'] = False
    
    return False

# 设置中文字体
setup_chinese_font()

import pandas as pd
from datetime import datetime
import config as cfg

class DataCollector:
    """数据收集和可视化类"""
    
    def __init__(self, save_dir="data_logs"):
        self.save_dir = save_dir
        self.start_time = None
        self.end_time = None
        self.episode_start_time = None
        
        # 检查字体是否设置成功
        self.chinese_font_available = self._check_chinese_font()
        
        # 创建保存目录
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        
        # 创建绘图目录
        self.plot_dir = os.path.join(save_dir, "plots")
        if not os.path.exists(self.plot_dir):
            os.makedirs(self.plot_dir)
        
        # 数据存储列表
        self.timestamps = []            # 时间戳（相对运行时间）
        self.real_timestamps = []       # 实际时间戳
        self.obstacle_distances = []    # 障碍物距离
        self.actions = []               # 执行的动作
        self.lateral_errors = []        # 横向偏差
        self.angle_errors = []          # 角度差
        self.speeds = []                # 速度
        self.fps_values = []            # 帧率
        self.rewards = []               # 奖励值
        
        # 用于计算帧率的变量
        self.last_time = None
        self.frame_count = 0
        self.frame_times = []
        
        # 动作名称映射（中文）
        self.action_names_cn = ["刹车", "直行", "左转", "右转", "微左", "微右"]
        
        print(f"📊 数据收集器初始化完成，数据将保存到: {save_dir}")
        print(f"📝 中文字体可用: {'是' if self.chinese_font_available else '否'}")
    
    def _check_chinese_font(self):
        """检查中文字体是否可用"""
        try:
            # 尝试绘制一个包含中文的简单图形来测试字体
            fig, ax = plt.subplots(figsize=(1, 1))
            ax.text(0.5, 0.5, "测试", fontsize=12)
            plt.close(fig)
            return True
        except:
            return False
    
    def start_episode(self):
        """开始一个episode的数据收集"""
        self.episode_start_time = time.time()
        self.timestamps = []
        self.real_timestamps = []
        self.obstacle_distances = []
        self.actions = []
        self.lateral_errors = []
        self.angle_errors = []
        self.speeds = []
        self.fps_values = []
        self.rewards = []
        self.last_time = time.time()
        self.frame_count = 0
        self.frame_times = []
        
        print("🔄 开始记录episode数据")
    
    def record_step(self, env, action, current_state, reward, vehicle_state=None):
        """记录每一步的数据"""
        current_time = time.time()
        
        # 计算相对时间
        if self.episode_start_time:
            elapsed = current_time - self.episode_start_time
        else:
            elapsed = 0
        
        # 记录时间戳
        self.timestamps.append(elapsed)
        self.real_timestamps.append(datetime.now())
        
        # 记录障碍物距离（从环境状态获取）
        if len(current_state) > 0:
            # 当前状态中的第一个元素是归一化的障碍物距离，需要反归一化
            norm_distance = current_state[0]
            actual_distance = norm_distance * 300 + 300  # 反归一化
            self.obstacle_distances.append(actual_distance)
        else:
            self.obstacle_distances.append(0)
        
        # 记录动作
        self.actions.append(action)
        
        # 记录横向偏差和角度差
        if len(current_state) >= 4:
            # 角度差（已归一化）
            angle_error = current_state[2]  # 这是phi
            # 横向偏差（已归一化）
            lateral_error = current_state[3] / 15  # 除以15反归一化
            
            self.angle_errors.append(angle_error)
            self.lateral_errors.append(lateral_error)
        else:
            self.angle_errors.append(0)
            self.lateral_errors.append(0)
        
        # 记录速度（从车辆状态或环境状态获取）
        if vehicle_state and 'speed_2d' in vehicle_state:
            speed = vehicle_state['speed_2d']
        elif len(current_state) >= 2:
            # 从状态计算速度（假设第二个状态是归一化速度）
            norm_speed = (current_state[1] + current_state[0]) * 30 + 30
            speed = norm_speed / 3.6  # km/h转m/s
        else:
            speed = 0
        self.speeds.append(speed)
        
        # 计算并记录帧率
        if self.last_time:
            frame_time = current_time - self.last_time
            self.frame_times.append(frame_time)
            if frame_time > 0:
                fps = 1.0 / frame_time
                self.fps_values.append(fps)
            else:
                self.fps_values.append(0)
        
        self.last_time = current_time
        self.frame_count += 1
        self.rewards.append(reward)
    
    def end_episode(self):
        """结束一个episode的数据收集"""
        self.end_time = time.time()
        episode_duration = self.end_time - self.episode_start_time if self.episode_start_time else 0
        
        print(f"📈 Episode数据收集完成，总时长: {episode_duration:.2f}秒，记录步数: {len(self.timestamps)}")
        
        # 保存数据到CSV
        self.save_to_csv()
        
        # 生成所有图表
        self.generate_all_plots()
        
        return episode_duration
    
    def save_to_csv(self):
        """保存数据到CSV文件"""
        if len(self.timestamps) == 0:
            print("⚠️ 没有数据可保存")
            return
        
        # 创建DataFrame
        data = {
            '时间戳': self.timestamps,
            '实际时间': self.real_timestamps,
            '障碍物距离': self.obstacle_distances,
            '动作编号': self.actions,
            '动作名称': [self.action_names_cn[a] if a < len(self.action_names_cn) else '未知' for a in self.actions],
            '横向偏差': self.lateral_errors,
            '角度差': self.angle_errors,
            '速度_mps': self.speeds,
            '速度_kmh': [s * 3.6 for s in self.speeds],
            '帧率': self.fps_values,
            '奖励值': self.rewards
        }
        
        df = pd.DataFrame(data)
        
        # 生成文件名
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"episode_data_{timestamp_str}.csv"
        filepath = os.path.join(self.save_dir, filename)
        
        # 保存CSV
        df.to_csv(filepath, index=False, encoding='utf-8-sig')
        print(f"💾 数据已保存到: {filepath}")
        
        # 计算并打印统计信息
        self.print_statistics(df)
    
    def print_statistics(self, df):
        """打印统计信息"""
        print("\n" + "="*60)
        print("数据统计")
        print("="*60)
        
        if len(df) > 0:
            print(f"总步数: {len(df)}")
            print(f"平均障碍物距离: {df['障碍物距离'].mean():.2f}米")
            print(f"最小障碍物距离: {df['障碍物距离'].min():.2f}米")
            print(f"最大障碍物距离: {df['障碍物距离'].max():.2f}米")
            print(f"平均横向偏差: {df['横向偏差'].abs().mean():.2f}米")
            print(f"平均角度差: {df['角度差'].abs().mean():.2f}度")
            print(f"平均速度: {df['速度_mps'].mean():.2f}米/秒 ({df['速度_kmh'].mean():.1f}公里/小时)")
            print(f"平均帧率: {df['帧率'].mean():.1f} FPS")
            
            # 动作分布
            action_counts = df['动作名称'].value_counts()
            print("\n动作分布:")
            for action, count in action_counts.items():
                percentage = (count / len(df)) * 100
                print(f"  {action}: {count}次 ({percentage:.1f}%)")
    
    def generate_all_plots(self):
        """生成所有图表"""
        if len(self.timestamps) == 0:
            print("⚠️ 没有数据可绘制图表")
            return
        
        print("📊 生成图表...")
        
        # 1. 障碍物距离与动作图
        self.plot_obstacle_distance_and_actions()
        
        # 2. 横向偏差和角度差折线图
        self.plot_lateral_and_angle_errors()
        
        # 3. 速度和帧率折线图
        self.plot_speed_and_fps()
        
        # 4. 综合图表
        self.plot_comprehensive_chart()
        
        print("✅ 所有图表已生成")
    
    def plot_obstacle_distance_and_actions(self):
        """绘制障碍物距离与动作图"""
        if len(self.timestamps) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 如果中文字体不可用，使用简单的ASCII文本
        if not self.chinese_font_available:
            print("⚠️ 中文字体不可用，图表可能无法正确显示中文")
        
        # 第一个子图：障碍物距离
        ax1.plot(self.timestamps, self.obstacle_distances, 'b-', linewidth=2, label='障碍物距离')
        ax1.fill_between(self.timestamps, 0, self.obstacle_distances, alpha=0.2)
        ax1.set_xlabel('运行时间 (秒)', fontsize=12)
        ax1.set_ylabel('距离 (米)', fontsize=12)
        ax1.set_title('障碍物距离随时间变化', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # 添加安全距离线
        ax1.axhline(y=10, color='r', linestyle='--', alpha=0.5, label='安全距离(10米)')
        ax1.axhline(y=5, color='orange', linestyle='--', alpha=0.5, label='警告距离(5米)')
        
        # 第二个子图：动作
        action_names = [self.action_names_cn[a] if a < len(self.action_names_cn) else '未知' for a in self.actions]
        
        # 使用散点图显示动作
        unique_actions = sorted(set(self.actions))
        for action in unique_actions:
            indices = [i for i, a in enumerate(self.actions) if a == action]
            times = [self.timestamps[i] for i in indices]
            ax2.scatter(times, [action] * len(times), s=50, 
                       label=self.action_names_cn[action] if action < len(self.action_names_cn) else '未知',
                       alpha=0.7)
        
        ax2.set_xlabel('运行时间 (秒)', fontsize=12)
        ax2.set_ylabel('动作', fontsize=12)
        ax2.set_title('执行动作分布', fontsize=14, fontweight='bold')
        ax2.set_yticks(range(len(self.action_names_cn)))
        ax2.set_yticklabels(self.action_names_cn)
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc='upper right', fontsize=10)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"障碍物距离与动作_{timestamp_str}.png"
        filepath = os.path.join(self.plot_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 障碍物距离与动作图已保存: {filename}")
    
    def plot_lateral_and_angle_errors(self):
        """绘制横向偏差和角度差折线图"""
        if len(self.timestamps) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 第一个子图：横向偏差
        ax1.plot(self.timestamps, self.lateral_errors, 'g-', linewidth=2, label='横向偏差')
        ax1.fill_between(self.timestamps, 0, self.lateral_errors, alpha=0.2, color='green')
        ax1.set_xlabel('运行时间 (秒)', fontsize=12)
        ax1.set_ylabel('横向偏差 (米)', fontsize=12)
        ax1.set_title('横向偏差随时间变化', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # 添加参考线
        ax1.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax1.axhline(y=1, color='orange', linestyle='--', alpha=0.5, label='允许偏差(1米)')
        ax1.axhline(y=-1, color='orange', linestyle='--', alpha=0.5)
        ax1.axhline(y=2, color='r', linestyle='--', alpha=0.5, label='最大偏差(2米)')
        ax1.axhline(y=-2, color='r', linestyle='--', alpha=0.5)
        
        # 第二个子图：角度差
        ax2.plot(self.timestamps, self.angle_errors, 'r-', linewidth=2, label='角度差')
        ax2.fill_between(self.timestamps, 0, self.angle_errors, alpha=0.2, color='red')
        ax2.set_xlabel('运行时间 (秒)', fontsize=12)
        ax2.set_ylabel('角度差 (度)', fontsize=12)
        ax2.set_title('角度差随时间变化', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # 添加参考线
        ax2.axhline(y=0, color='k', linestyle='-', alpha=0.3)
        ax2.axhline(y=30, color='orange', linestyle='--', alpha=0.5, label='允许偏差(30°)')
        ax2.axhline(y=-30, color='orange', linestyle='--', alpha=0.5)
        ax2.axhline(y=100, color='r', linestyle='--', alpha=0.5, label='最大偏差(100°)')
        ax2.axhline(y=-100, color='r', linestyle='--', alpha=0.5)
        
        plt.tight_layout()
        
        # 保存图表
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"横向偏差与角度差_{timestamp_str}.png"
        filepath = os.path.join(self.plot_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 横向偏差和角度差图已保存: {filename}")
    
    def plot_speed_and_fps(self):
        """绘制速度和帧率折线图"""
        if len(self.timestamps) == 0:
            return
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
        
        # 第一个子图：速度
        speeds_kmh = [s * 3.6 for s in self.speeds]  # 转换为km/h
        
        ax1.plot(self.timestamps, speeds_kmh, 'purple', linewidth=2, label='速度')
        ax1.fill_between(self.timestamps, 0, speeds_kmh, alpha=0.2, color='purple')
        ax1.set_xlabel('运行时间 (秒)', fontsize=12)
        ax1.set_ylabel('速度 (公里/小时)', fontsize=12)
        ax1.set_title('速度随时间变化', fontsize=14, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        ax1.legend(fontsize=10)
        
        # 第二个子图：帧率
        ax2.plot(self.timestamps, self.fps_values, 'orange', linewidth=2, label='帧率')
        ax2.fill_between(self.timestamps, 0, self.fps_values, alpha=0.2, color='orange')
        ax2.set_xlabel('运行时间 (秒)', fontsize=12)
        ax2.set_ylabel('帧率 (FPS)', fontsize=12)
        ax2.set_title('帧率随时间变化', fontsize=14, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        ax2.legend(fontsize=10)
        
        # 添加平均帧率线
        if len(self.fps_values) > 0:
            avg_fps = np.mean(self.fps_values)
            ax2.axhline(y=avg_fps, color='r', linestyle='--', alpha=0.7, 
                       label=f'平均帧率: {avg_fps:.1f} FPS')
        
        plt.tight_layout()
        
        # 保存图表
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"速度与帧率_{timestamp_str}.png"
        filepath = os.path.join(self.plot_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 速度和帧率图已保存: {filename}")
    
    def plot_comprehensive_chart(self):
        """绘制综合图表"""
        if len(self.timestamps) == 0:
            return
        
        fig = plt.figure(figsize=(16, 12))
        
        # 1. 障碍物距离
        ax1 = plt.subplot(3, 2, 1)
        ax1.plot(self.timestamps, self.obstacle_distances, 'b-', linewidth=1.5)
        ax1.set_xlabel('时间 (秒)', fontsize=10)
        ax1.set_ylabel('距离 (米)', fontsize=10)
        ax1.set_title('障碍物距离', fontsize=12, fontweight='bold')
        ax1.grid(True, alpha=0.3)
        
        # 2. 横向偏差
        ax2 = plt.subplot(3, 2, 2)
        ax2.plot(self.timestamps, self.lateral_errors, 'g-', linewidth=1.5)
        ax2.set_xlabel('时间 (秒)', fontsize=10)
        ax2.set_ylabel('横向偏差 (米)', fontsize=10)
        ax2.set_title('横向偏差', fontsize=12, fontweight='bold')
        ax2.grid(True, alpha=0.3)
        
        # 3. 角度差
        ax3 = plt.subplot(3, 2, 3)
        ax3.plot(self.timestamps, self.angle_errors, 'r-', linewidth=1.5)
        ax3.set_xlabel('时间 (秒)', fontsize=10)
        ax3.set_ylabel('角度差 (度)', fontsize=10)
        ax3.set_title('角度差', fontsize=12, fontweight='bold')
        ax3.grid(True, alpha=0.3)
        
        # 4. 速度
        ax4 = plt.subplot(3, 2, 4)
        speeds_kmh = [s * 3.6 for s in self.speeds]
        ax4.plot(self.timestamps, speeds_kmh, 'purple', linewidth=1.5)
        ax4.set_xlabel('时间 (秒)', fontsize=10)
        ax4.set_ylabel('速度 (公里/小时)', fontsize=10)
        ax4.set_title('速度', fontsize=12, fontweight='bold')
        ax4.grid(True, alpha=0.3)
        
        # 5. 帧率
        ax5 = plt.subplot(3, 2, 5)
        ax5.plot(self.timestamps, self.fps_values, 'orange', linewidth=1.5)
        ax5.set_xlabel('时间 (秒)', fontsize=10)
        ax5.set_ylabel('帧率 (FPS)', fontsize=10)
        ax5.set_title('帧率', fontsize=12, fontweight='bold')
        ax5.grid(True, alpha=0.3)
        
        # 6. 奖励值
        ax6 = plt.subplot(3, 2, 6)
        ax6.plot(self.timestamps, self.rewards, 'brown', linewidth=1.5)
        ax6.set_xlabel('时间 (秒)', fontsize=10)
        ax6.set_ylabel('奖励值', fontsize=10)
        ax6.set_title('奖励值', fontsize=12, fontweight='bold')
        ax6.grid(True, alpha=0.3)
        
        plt.suptitle('自动驾驶性能综合图表', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        # 保存图表
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"综合图表_{timestamp_str}.png"
        filepath = os.path.join(self.plot_dir, filename)
        plt.savefig(filepath, dpi=150, bbox_inches='tight')
        plt.close()
        
        print(f"  ✅ 综合图表已保存: {filename}")
    
    def get_summary(self):
        """获取数据摘要"""
        summary = {
            '总步数': len(self.timestamps),
            '总时长': self.timestamps[-1] if self.timestamps else 0,
            '平均障碍物距离': np.mean(self.obstacle_distances) if self.obstacle_distances else 0,
            '最小障碍物距离': np.min(self.obstacle_distances) if self.obstacle_distances else 0,
            '平均横向偏差': np.mean(np.abs(self.lateral_errors)) if self.lateral_errors else 0,
            '平均角度差': np.mean(np.abs(self.angle_errors)) if self.angle_errors else 0,
            '平均速度': np.mean(self.speeds) if self.speeds else 0,
            '平均帧率': np.mean(self.fps_values) if self.fps_values else 0,
            '总奖励值': np.sum(self.rewards) if self.rewards else 0
        }
        
        return summary
    
    def generate_performance_report(self, episode_num, episode_duration):
        """生成性能报告"""
        if len(self.timestamps) == 0:
            return
        
        summary = self.get_summary()
        
        report = f"""
        ================================================================
        Episode {episode_num} 性能报告
        ================================================================
        
        总体指标:
        - 总步数: {summary['总步数']}
        - 总时长: {summary['总时长']:.2f} 秒
        - 步速: {summary['总步数']/summary['总时长']:.2f} 步/秒
        
        安全指标:
        - 平均障碍物距离: {summary['平均障碍物距离']:.2f} 米
        - 最小障碍物距离: {summary['最小障碍物距离']:.2f} 米
        - 平均横向偏差: {summary['平均横向偏差']:.2f} 米
        - 平均角度差: {summary['平均角度差']:.2f} 度
        
        性能指标:
        - 平均速度: {summary['平均速度']*3.6:.1f} 公里/小时
        - 平均帧率: {summary['平均帧率']:.1f} FPS
        - 总奖励值: {summary['总奖励值']:.1f}
        
        数据保存:
        - CSV文件: data_logs/episode_data_*.csv
        - 图表: data_logs/plots/*.png
        ================================================================
        """
        
        print(report)
        
        # 保存报告到文件
        timestamp_str = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = os.path.join(self.save_dir, f"episode_{episode_num}_报告_{timestamp_str}.txt")
        
        with open(report_file, 'w', encoding='utf-8') as f:
            f.write(report)
        
        print(f"📋 性能报告已保存到: {report_file}")
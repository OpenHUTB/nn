import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
import time
import tempfile
import warnings
warnings.filterwarnings('ignore')

# 设置Matplotlib支持中文显示（解决乱码问题）
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']  # 设置中文字体
plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 尝试导入MuJoCo，如果失败则使用模拟模式
try:
    import mujoco
    import mujoco.viewer
    MUJOCO_AVAILABLE = True
    print("✅ MuJoCo 已安装，将启动可视化界面")
except ImportError:
    MUJOCO_AVAILABLE = False
    print("⚠️  MuJoCo 未安装，将仅生成图表（无可视化界面）")
    print("💡 安装命令: pip install mujoco mujoco-python-viewer")

def generate_collision_analysis():
    """生成碰撞分析图表"""
    print("="*60)
    print("机械臂碰撞风险分析系统")
    print("="*60)

    # 1. 生成模拟工作空间数据
    print("正在生成工作空间数据...")
    np.random.seed(42)
    n_points = 300

    # 模拟机械臂工作空间
    theta = np.random.uniform(0, 2*np.pi, n_points)
    phi = np.random.uniform(0, np.pi, n_points)
    r = 0.5 + 0.2 * np.random.randn(n_points)

    x = 0.6 * np.cos(theta) * np.sin(phi)
    y = 0.6 * np.sin(theta) * np.sin(phi)
    z = 0.5 + 0.3 * np.cos(phi)

    points = np.vstack([x, y, z]).T

    # 2. 计算碰撞风险
    print("正在计算碰撞风险...")
    risks = []
    for point in points:
        risk = 0

        # 墙壁风险 (x=0.7)
        wall_dist = abs(point[0] - 0.7)
        if wall_dist < 0.15:
            risk += 0.8 * (0.15 - wall_dist) / 0.15

        # 中心障碍物
        center_dist = np.sqrt(point[0]**2 + point[1]**2)
        if center_dist < 0.2:
            risk += 0.6 * (0.2 - center_dist) / 0.2

        # 天花板风险
        if point[2] > 0.9:
            risk += 0.4

        # 地面风险
        if point[2] < 0.1:
            risk += 0.3

        risk = min(1.0, risk)
        risks.append(risk)

    risks = np.array(risks)

    # 3. 绘制图表
    print("正在生成可视化图表...")
    fig = plt.figure(figsize=(15, 6))

    # 左侧：3D风险图
    ax1 = fig.add_subplot(121, projection='3d')
    scatter = ax1.scatter(points[:, 0], points[:, 1], points[:, 2],
                         c=risks, cmap='RdYlGn_r',
                         alpha=0.7, s=20, edgecolors='none')

    # 添加障碍物标记
    ax1.plot([0.7, 0.7], [-0.8, 0.8], [0, 1], 'k-', linewidth=3, alpha=0.5, label='墙壁')

    ax1.set_xlabel('X (米)', fontsize=12, labelpad=10)
    ax1.set_ylabel('Y (米)', fontsize=12, labelpad=10)
    ax1.set_zlabel('Z (米)', fontsize=12, labelpad=10)
    ax1.set_title('3D碰撞风险热力图', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.view_init(elev=25, azim=45)
    ax1.grid(True, alpha=0.3)

    plt.colorbar(scatter, ax=ax1, shrink=0.7, pad=0.1, label='碰撞风险')

    # 右侧：统计图
    ax2 = fig.add_subplot(122)

    # 风险等级统计
    low_risk = np.sum(risks < 0.3)
    medium_risk = np.sum((risks >= 0.3) & (risks < 0.7))
    high_risk = np.sum(risks >= 0.7)

    categories = ['低风险', '中风险', '高风险']
    counts = [low_risk, medium_risk, high_risk]
    percentages = [c/n_points*100 for c in counts]
    colors = ['#2E8B57', '#FFA500', '#DC143C']

    bars = ax2.bar(categories, percentages, color=colors, edgecolor='black', alpha=0.8)

    # 添加标签
    for bar, percent in zip(bars, percentages):
        height = bar.get_height()
        ax2.text(bar.get_x() + bar.get_width()/2, height + 1,
                f'{percent:.1f}%', ha='center', fontsize=11, fontweight='bold')

    ax2.set_ylabel('占比 (%)', fontsize=12)
    ax2.set_title('风险区域分布', fontsize=14, fontweight='bold')
    ax2.set_ylim(0, 100)
    ax2.grid(True, alpha=0.3, axis='y')

    # 总结文本
    summary = f'分析结果:\n'
    summary += f'• 安全区域: {percentages[0]:.1f}%\n'
    summary += f'• 危险区域: {percentages[2]:.1f}%\n'
    summary += f'• 总采样点: {n_points}'

    ax2.text(0.05, 0.95, summary, transform=ax2.transAxes, fontsize=11,
            bbox=dict(boxstyle="round,pad=0.3", facecolor="white", alpha=0.8),
            verticalalignment='top')

    plt.suptitle('机械臂工作空间碰撞风险分析', fontsize=16, fontweight='bold')
    plt.tight_layout()

    # 保存图表到当前目录
    output_file = 'collision_analysis_result.png'
    plt.savefig(output_file, dpi=150, bbox_inches='tight')
    plt.show()

    print(f"\n✅ 碰撞分析完成！")
    print(f"📊 图表已保存: {output_file}")
    print(f"📋 安全区域占比: {percentages[0]:.1f}%")
    print(f"📋 危险区域占比: {percentages[2]:.1f}%")

    return True

def run_mujoco_simulation():
    """运行MuJoCo可视化仿真"""
    if not MUJOCO_AVAILABLE:
        print("\n❌ MuJoCo未安装，无法启动可视化界面")
        print("💡 请安装: pip install mujoco mujoco-python-viewer")
        return False

    print("\n" + "="*60)
    print("启动MuJoCo机械臂仿真")
    print("="*60)
    print("控制说明:")
    print("- 窗口中将显示机械臂模型")
    print("- 机械臂会自动进行随机运动")
    print("- 按ESC键退出仿真")
    print("="*60)

    try:
        # 使用相对路径查找模型文件
        current_dir = os.path.dirname(os.path.abspath(__file__))
        model_file_path = os.path.join(current_dir, 'arm_with_gripper.xml')

        # 检查文件是否存在
        if not os.path.exists(model_file_path):
            print(f"❌ 模型文件不存在: {model_file_path}")
            print(f"当前目录: {current_dir}")
            print(f"目录内容: {os.listdir(current_dir)}")
            return False

        print(f"正在加载模型文件: {model_file_path}")

        # 读取模型文件内容
        with open(model_file_path, 'r', encoding='utf-8') as f:
            model_content = f.read()

        # 移除对不存在的资源目录的引用
        model_content = model_content.replace('meshdir="assets/"', '')
        model_content = model_content.replace('texturedir="textures/"', '')

        # 使用临时文件，避免中文字符路径问题
        with tempfile.NamedTemporaryFile(mode='w', suffix='.xml', delete=False, encoding='utf-8') as temp_file:
            temp_model_path = temp_file.name
            temp_file.write(model_content)
            print(f"✅ 临时模型文件已创建: {temp_model_path}")

        # 检查文件是否已创建
        if not os.path.exists(temp_model_path):
            print(f"❌ 临时文件创建失败: {temp_model_path}")
            return False

        print(f"✅ 临时文件存在: {os.path.exists(temp_model_path)}")
        print(f"✅ 临时文件大小: {os.path.getsize(temp_model_path)} 字节")

        # 从临时路径加载模型
        try:
            print("正在加载MuJoCo模型...")
            model = mujoco.MjModel.from_xml_path(temp_model_path)
            data = mujoco.MjData(model)
            print("✅ 模型加载成功")
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            # 尝试直接从XML字符串加载
            print("尝试从XML字符串加载模型...")
            model = mujoco.MjModel.from_xml_string(model_content)
            data = mujoco.MjData(model)
            print("✅ 从字符串加载模型成功")

        print(f"关节数量: {model.njnt}")
        print(f"执行器数量: {model.nu}")

        # 启动可视化界面
        print("正在启动可视化窗口...")

        try:
            with mujoco.viewer.launch_passive(model, data) as viewer:
                # 设置视角
                viewer.cam.azimuth = 45
                viewer.cam.elevation = -20
                viewer.cam.distance = 2.5
                viewer.cam.lookat[:] = [0.2, 0.0, 0.5]

                print("✅ 可视化窗口已启动")
                print("机械臂开始随机运动...")

                # 仿真参数
                simulation_time = 30.0  # 仿真30秒
                start_time = time.time()
                step_count = 0

                # 随机目标角度
                target_angles = np.random.uniform(-0.5, 0.5, model.nu)

                while viewer.is_running() and (time.time() - start_time) < simulation_time:
                    step_start = time.time()

                    # 简单的PD控制，让机械臂随机运动
                    for i in range(min(model.nu, len(target_angles))):
                        # 计算控制信号（简单的PD控制器）
                        error = target_angles[i] - data.qpos[i]
                        data.ctrl[i] = 100 * error - 10 * data.qvel[i]  # PD控制

                    # 每100步重新生成随机目标
                    if step_count % 100 == 0:
                        target_angles = np.random.uniform(-0.5, 0.5, model.nu)

                    # 碰撞检测（简单版本）
                    try:
                        ee_site_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, 'ee_site')
                        if ee_site_id >= 0:
                            ee_pos = data.site_xpos[ee_site_id]

                            # 检查与墙壁的碰撞
                            if abs(ee_pos[0] - 0.7) < 0.1:
                                print("⚠️  警告: 末端接近墙壁!")
                            # 检查与柱子的碰撞
                            if np.sqrt(ee_pos[0]**2 + ee_pos[1]**2) < 0.2:
                                print("⚠️  警告: 末端接近中心柱子!")
                    except:
                        pass  # 如果站点不存在，跳过碰撞检测

                    # 执行模拟步骤
                    mujoco.mj_step(model, data)

                    # 同步viewer
                    viewer.sync()

                    # 控制仿真速度
                    elapsed = time.time() - step_start
                    sleep_time = model.opt.timestep - elapsed
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                    step_count += 1

                print(f"\n仿真结束，共运行 {step_count} 步")

        except KeyboardInterrupt:
            print("\n用户中断仿真")
        except Exception as e:
            print(f"仿真错误: {e}")

        finally:
            # 清理临时文件
            try:
                if os.path.exists(temp_model_path):
                    os.remove(temp_model_path)
                    print(f"临时文件已删除: {temp_model_path}")
            except Exception as e:
                print(f"清理临时文件失败: {e}")

        return True

    except Exception as e:
        print(f"❌ MuJoCo仿真失败: {e}")
        import traceback
        traceback.print_exc()
        return False

def run_collision_detection_system():
    """运行完整的碰撞检测系统"""
    print("🚀 机械臂碰撞检测与可视化系统")
    print("="*60)

    # 步骤1: 生成碰撞分析图表
    print("\n[步骤1] 生成碰撞风险分析图表...")
    success1 = generate_collision_analysis()

    if not success1:
        print("❌ 碰撞分析失败")
        return False

    # 步骤2: 询问是否启动MuJoCo仿真
    print("\n" + "="*60)
    if MUJOCO_AVAILABLE:
        response = input("是否启动MuJoCo机械臂仿真？(y/n): ").strip().lower()
        if response in ['y', 'yes', '是']:
            print("\n[步骤2] 启动MuJoCo可视化仿真...")
            success2 = run_mujoco_simulation()
            if success2:
                print("✅ MuJoCo仿真完成")
            else:
                print("❌ MuJoCo仿真失败")
        else:
            print("跳过MuJoCo仿真")
    else:
        print("⚠️  MuJoCo未安装，跳过仿真步骤")
        print("💡 要启用仿真功能，请安装:")
        print("   pip install mujoco mujoco-python-viewer")

    # 总结
    print("\n" + "="*60)
    print("系统运行完成！")
    print("="*60)
    print("📊 生成的图表:")
    print("  • collision_analysis_result.png - 碰撞风险分析图")
    print("\n🎯 后续步骤:")
    print("  1. 查看生成的图表了解碰撞风险分布")
    print("  2. 根据分析结果优化机械臂工作空间")
    print("  3. 安装MuJoCo以启用仿真功能")
    print("="*60)

    return True

if __name__ == "__main__":
    try:
        run_collision_detection_system()
    except KeyboardInterrupt:
        print("\n程序被用户中断")
    except Exception as e:
        print(f"\n❌ 程序运行出错: {e}")
        import traceback
        traceback.print_exc()
        print("\n💡 常见问题解决方法:")
        print("1. 确保已安装必要依赖: pip install numpy matplotlib")
        print("2. 如需MuJoCo仿真: pip install mujoco mujoco-python-viewer")
        print("3. 检查Python版本兼容性")
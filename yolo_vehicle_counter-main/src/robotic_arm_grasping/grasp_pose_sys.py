import json
import pickle
import time
import warnings

import matplotlib.pyplot as plt
import mujoco
import mujoco.viewer
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from matplotlib import cm
from matplotlib.collections import PatchCollection
from matplotlib.patches import Circle, Wedge, Polygon
from mpl_toolkits.mplot3d import Axes3D
from plotly.subplots import make_subplots
from scipy.spatial.transform import Rotation as R
from scipy.stats import gaussian_kde

warnings.filterwarnings('ignore')


class GraspPoseGenerator:
    """抓取姿态生成器"""

    def __init__(self, model_path):
        """初始化抓取姿态生成器"""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 关节限制
        self.joint_limits = self._get_joint_limits()

        # 物体列表
        self.objects = ['test_sphere', 'test_box', 'test_cylinder',
                        'test_capsule', 'test_ellipsoid']

        # 抓取位置缓存
        self.grasp_positions = {}

    def _get_joint_limits(self):
        """获取关节运动范围"""
        limits = {}
        for i in range(self.model.njnt):
            joint_id = i
            joint_type = self.model.jnt_type[i]
            if joint_type == mujoco.mjtJoint.mjJNT_HINGE:
                limits[f'joint{i + 1}'] = {
                    'min': self.model.jnt_range[i, 0],
                    'max': self.model.jnt_range[i, 1]
                }
        return limits

    def generate_random_grasp_pose(self, object_name):
        """为指定物体生成随机抓取姿态"""
        # 获取物体位置和大小
        obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, object_name)
        obj_pos = self.data.xpos[obj_body_id]

        # 物体包围盒
        obj_size = self._get_object_size(object_name)

        # 生成抓取位置（围绕物体）
        grasp_positions = []
        grasp_orientations = []

        # 生成多个抓取点
        n_points = 8
        for i in range(n_points):
            # 球坐标
            theta = 2 * np.pi * i / n_points
            phi = np.pi / 4  # 45度仰角

            # 计算抓取点位置
            radius = np.linalg.norm(obj_size) + 0.1  # 距离物体表面0.1米
            x = obj_pos[0] + radius * np.sin(phi) * np.cos(theta)
            y = obj_pos[1] + radius * np.sin(phi) * np.sin(theta)
            z = obj_pos[2] + radius * np.cos(phi)

            grasp_positions.append([x, y, z])

            # 计算朝向（指向物体中心）
            direction = obj_pos - np.array([x, y, z])
            direction = direction / np.linalg.norm(direction)

            # 计算四元数
            up_vector = np.array([0, 0, 1])
            right_vector = np.cross(direction, up_vector)
            right_vector = right_vector / np.linalg.norm(right_vector)
            new_up = np.cross(right_vector, direction)

            # 创建旋转矩阵
            rot_matrix = np.column_stack([right_vector, new_up, -direction])
            rotation = R.from_matrix(rot_matrix)
            grasp_orientations.append(rotation.as_quat())

        return grasp_positions, grasp_orientations

    def _get_object_size(self, object_name):
        """获取物体尺寸"""
        geom_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM,
                                    f"{object_name}_geom")

        geom_type = self.model.geom_type[geom_id]
        geom_size = self.model.geom_size[geom_id]

        if geom_type == mujoco.mjtGeom.mjGEOM_SPHERE:
            return np.array([geom_size[0], geom_size[0], geom_size[0]])
        elif geom_type == mujoco.mjtGeom.mjGEOM_BOX:
            return geom_size
        elif geom_type == mujoco.mjtGeom.mjGEOM_CYLINDER:
            return np.array([geom_size[0], geom_size[0], geom_size[1]])
        elif geom_type == mujoco.mjtGeom.mjGEOM_CAPSULE:
            return np.array([geom_size[0], geom_size[0], geom_size[1]])
        elif geom_type == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
            return geom_size

        return np.array([0.1, 0.1, 0.1])

    def compute_ik(self, target_pos, target_quat=None):
        """计算逆运动学（简化版本）"""
        # 这里使用简化方法，实际应用中应使用完整的IK求解器
        # 返回6个关节角度
        joint_angles = np.zeros(6)

        # 计算基座旋转角度
        dx = target_pos[0]
        dy = target_pos[1]
        joint_angles[0] = np.arctan2(dy, dx)

        # 简化计算其他关节角度
        distance = np.sqrt(dx ** 2 + dy ** 2)
        height = target_pos[2]

        # 使用三角形近似计算关节角度
        # 这是一个简化的2连杆逆运动学
        L1 = 0.35  # 上臂长度
        L2 = 0.3  # 前臂长度

        # 计算肘部角度
        D = np.sqrt(distance ** 2 + (height - 0.1) ** 2)
        if D <= L1 + L2 and D >= abs(L1 - L2):
            # 可到达的位置
            cos_q2 = (D ** 2 - L1 ** 2 - L2 ** 2) / (2 * L1 * L2)
            joint_angles[2] = np.arccos(np.clip(cos_q2, -1, 1))

            # 计算肩部角度
            alpha = np.arctan2(height - 0.1, distance)
            beta = np.arcsin(L2 * np.sin(joint_angles[2]) / D)
            joint_angles[1] = alpha - beta

            # 腕部角度（简化）
            joint_angles[3] = 0.0  # 腕部旋转
            joint_angles[4] = np.pi / 2  # 腕部俯仰
            joint_angles[5] = 0.0  # 腕部横滚

        return joint_angles


class GraspEvaluator:
    """抓取稳定性评估器"""

    def __init__(self, model_path):
        """初始化评估器"""
        self.model = mujoco.MjModel.from_xml_path(model_path)
        self.data = mujoco.MjData(self.model)

        # 评估指标权重
        self.weights = {
            'force_closure': 0.4,
            'contact_points': 0.3,
            'wrench_resistance': 0.2,
            'approach_angle': 0.1
        }

    def evaluate_grasp(self, object_name, joint_angles, finger_positions,
                       finger_angles, simulate=True):
        """评估抓取稳定性"""
        if simulate:
            # 设置关节位置
            for i in range(6):
                self.data.qpos[i] = joint_angles[i]

            # 设置手指位置
            finger_joints = ['finger1_slide', 'finger2_slide', 'finger3_slide',
                             'finger1_hinge1', 'finger2_hinge1', 'finger3_hinge1',
                             'finger1_hinge2', 'finger2_hinge2', 'finger3_hinge2']

            for i, joint_name in enumerate(finger_joints):
                joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_JOINT,
                                             joint_name)
                self.data.qpos[6 + i] = finger_positions[i // 3] if i < 3 else finger_angles[i - 3]

            # 向前模拟
            mujoco.mj_forward(self.model, self.data)

            # 运行抓取模拟
            for _ in range(100):
                # 施加抓取力
                for i in range(3):
                    slide_joint_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR,
                                                       f'motor_finger{i + 1}_slide')
                    self.data.ctrl[slide_joint_id] = 0.3  # 抓取力

                mujoco.mj_step(self.model, self.data)

            # 计算评估指标
            metrics = self._compute_grasp_metrics(object_name)
        else:
            # 不模拟，只计算理论指标
            metrics = self._compute_theoretical_metrics(object_name, joint_angles,
                                                        finger_positions, finger_angles)

        # 计算总分
        total_score = self._compute_total_score(metrics)

        return total_score, metrics

    def _compute_grasp_metrics(self, object_name):
        """基于模拟计算抓取指标"""
        metrics = {}

        # 1. 力闭合分析
        metrics['force_closure'] = self._compute_force_closure(object_name)

        # 2. 接触点分析
        metrics['contact_points'] = self._count_contact_points(object_name)

        # 3. 抗扰动力矩
        metrics['wrench_resistance'] = self._compute_wrench_resistance(object_name)

        # 4. 接近角度
        metrics['approach_angle'] = self._compute_approach_angle(object_name)

        return metrics

    def _compute_theoretical_metrics(self, object_name, joint_angles,
                                     finger_positions, finger_angles):
        """计算理论抓取指标"""
        metrics = {}

        # 简化的理论计算
        metrics['force_closure'] = 0.7  # 假设值
        metrics['contact_points'] = 3.0  # 三个手指
        metrics['wrench_resistance'] = 0.6  # 假设值
        metrics['approach_angle'] = 0.8  # 假设值

        return metrics

    def _compute_force_closure(self, object_name):
        """计算力闭合指标"""
        # 简化的力闭合计算
        obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                        object_name)

        # 检查接触力
        contact_force = 0.0
        for i in range(3):
            force_sensor = self.data.sensor(f'sensor_finger{i + 1}_force').data
            contact_force += np.linalg.norm(force_sensor)

        # 归一化到0-1范围
        force_closure_score = min(contact_force / 50.0, 1.0)

        return force_closure_score

    def _count_contact_points(self, object_name):
        """计算接触点数量"""
        contact_count = 0

        for i in range(3):
            touch_sensor = self.data.sensor(f'sensor_finger{i + 1}_touch').data
            if touch_sensor[0] > 0.1:  # 有接触
                contact_count += 1

        return contact_count / 3.0  # 归一化

    def _compute_wrench_resistance(self, object_name):
        """计算抗扰动力矩"""
        # 简化的抗扰动计算
        resistance_score = 0.5

        # 检查物体的运动
        obj_body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY,
                                        object_name)
        obj_vel = np.linalg.norm(self.data.qvel[6:9])  # 物体的线速度

        # 速度越小，抗扰动能力越强
        if obj_vel < 0.01:
            resistance_score = 0.9
        elif obj_vel < 0.05:
            resistance_score = 0.7
        elif obj_vel < 0.1:
            resistance_score = 0.5
        else:
            resistance_score = 0.3

        return resistance_score

    def _compute_approach_angle(self, object_name):
        """计算接近角度"""
        # 计算末端执行器指向物体的角度
        ee_pos = self.data.site('grip_center').xpos
        obj_pos = self.data.body(object_name).xpos

        direction = obj_pos - ee_pos
        direction = direction / np.linalg.norm(direction)

        # 理想方向是垂直向下
        ideal_direction = np.array([0, 0, -1])

        # 计算角度差
        angle = np.arccos(np.clip(np.dot(direction, ideal_direction), -1, 1))

        # 角度越小越好，归一化到0-1
        angle_score = 1.0 - min(angle / (np.pi / 2), 1.0)

        return angle_score

    def _compute_total_score(self, metrics):
        """计算总分"""
        total = 0.0
        for key, weight in self.weights.items():
            total += metrics[key] * weight

        return total


class GraspVisualizer:
    """抓取可视化器"""

    def __init__(self):
        """初始化可视化器"""
        plt.style.use('seaborn-v0_8-darkgrid')
        self.colors = plt.cm.Set3(np.linspace(0, 1, 12))

        # Plotly颜色
        self.plotly_colors = px.colors.qualitative.Set3

    def create_grasp_heatmap(self, grasp_data, object_name):
        """创建抓取热力图"""
        fig = make_subplots(
            rows=2, cols=3,
            subplot_titles=('抓取点分布', '抓取分数热力图', '力闭合分数',
                            '接触点分布', '抗扰动能力', '接近角度'),
            specs=[[{'type': 'scatter3d'}, {'type': 'heatmap'}, {'type': 'bar'}],
                   [{'type': 'scatter'}, {'type': 'radar'}, {'type': 'polar'}]]
        )

        # 提取数据
        positions = np.array([d['position'] for d in grasp_data])
        scores = np.array([d['score'] for d in grasp_data])
        metrics = [d['metrics'] for d in grasp_data]

        # 1. 抓取点分布 (3D散点图)
        fig.add_trace(
            go.Scatter3d(
                x=positions[:, 0],
                y=positions[:, 1],
                z=positions[:, 2],
                mode='markers',
                marker=dict(
                    size=8,
                    color=scores,
                    colorscale='Viridis',
                    showscale=True,
                    colorbar=dict(title='抓取分数')
                ),
                text=[f'分数: {s:.3f}' for s in scores],
                hoverinfo='text'
            ),
            row=1, col=1
        )

        # 添加物体位置
        # (这里需要实际物体位置，暂时使用平均值)
        obj_pos = np.mean(positions, axis=0)
        fig.add_trace(
            go.Scatter3d(
                x=[obj_pos[0]],
                y=[obj_pos[1]],
                z=[obj_pos[2]],
                mode='markers',
                marker=dict(size=15, color='red', symbol='diamond'),
                name='物体位置'
            ),
            row=1, col=1
        )

        # 2. 抓取分数热力图
        # 将3D位置投影到2D
        from sklearn.decomposition import PCA
        pca = PCA(n_components=2)
        positions_2d = pca.fit_transform(positions)

        # 创建网格
        x = positions_2d[:, 0]
        y = positions_2d[:, 1]
        z = scores

        # 插值创建热力图
        from scipy.interpolate import griddata

        xi = np.linspace(x.min(), x.max(), 50)
        yi = np.linspace(y.min(), y.max(), 50)
        xi, yi = np.meshgrid(xi, yi)

        zi = griddata((x, y), z, (xi, yi), method='cubic')

        fig.add_trace(
            go.Heatmap(
                x=xi[0, :],
                y=yi[:, 0],
                z=zi,
                colorscale='Viridis',
                showscale=True,
                colorbar=dict(title='抓取分数')
            ),
            row=1, col=2
        )

        # 3. 力闭合分数柱状图
        force_scores = [m['force_closure'] for m in metrics]
        fig.add_trace(
            go.Bar(
                x=list(range(len(force_scores))),
                y=force_scores,
                marker_color=self.plotly_colors[0],
                name='力闭合分数'
            ),
            row=1, col=3
        )

        # 4. 接触点分布散点图
        contact_scores = [m['contact_points'] for m in metrics]
        fig.add_trace(
            go.Scatter(
                x=list(range(len(contact_scores))),
                y=contact_scores,
                mode='markers+lines',
                marker=dict(size=10, color=self.plotly_colors[1]),
                line=dict(color=self.plotly_colors[1], dash='dash'),
                name='接触点分数'
            ),
            row=2, col=1
        )

        # 5. 抗扰动能力雷达图
        resistance_scores = [m['wrench_resistance'] for m in metrics]
        angles = np.linspace(0, 2 * np.pi, len(resistance_scores), endpoint=False)

        fig.add_trace(
            go.Scatterpolar(
                r=resistance_scores,
                theta=angles * 180 / np.pi,
                fill='toself',
                name='抗扰动能力',
                line_color=self.plotly_colors[2]
            ),
            row=2, col=2
        )

        # 6. 接近角度极坐标图
        approach_scores = [m['approach_angle'] for m in metrics]

        fig.add_trace(
            go.Scatterpolar(
                r=approach_scores,
                theta=angles * 180 / np.pi,
                mode='markers',
                marker=dict(size=10, color=approach_scores,
                            colorscale='Plasma', showscale=True),
                name='接近角度分数'
            ),
            row=2, col=3
        )

        # 更新布局
        fig.update_layout(
            height=800,
            title_text=f'物体 "{object_name}" 的抓取姿态分析',
            showlegend=True
        )

        return fig

    def create_grasp_comparison_radar(self, grasp_results):
        """创建抓取比较雷达图"""
        categories = ['总分', '力闭合', '接触点', '抗扰动', '接近角']

        fig = go.Figure()

        for obj_name, scores in grasp_results.items():
            fig.add_trace(go.Scatterpolar(
                r=[scores['total'], scores['force_closure'],
                   scores['contact_points'], scores['wrench_resistance'],
                   scores['approach_angle'], scores['total']],
                theta=categories + [categories[0]],
                fill='toself',
                name=obj_name
            ))

        fig.update_layout(
            polar=dict(
                radialaxis=dict(
                    visible=True,
                    range=[0, 1]
                )),
            showlegend=True,
            title='不同物体抓取能力比较'
        )

        return fig

    def create_matplotlib_visualization(self, grasp_data, object_name):
        """创建matplotlib可视化"""
        fig = plt.figure(figsize=(20, 12))

        # 提取数据
        positions = np.array([d['position'] for d in grasp_data])
        scores = np.array([d['score'] for d in grasp_data])
        metrics = np.array([[d['metrics']['force_closure'], d['metrics']['contact_points'],
                             d['metrics']['wrench_resistance'], d['metrics']['approach_angle']]
                            for d in grasp_data])

        # 1. 3D抓取点分布
        ax1 = fig.add_subplot(2, 3, 1, projection='3d')
        scatter = ax1.scatter(positions[:, 0], positions[:, 1], positions[:, 2],
                              c=scores, cmap='viridis', s=50, alpha=0.8)
        ax1.set_xlabel('X (m)')
        ax1.set_ylabel('Y (m)')
        ax1.set_zlabel('Z (m)')
        ax1.set_title(f'抓取点分布 - {object_name}')
        plt.colorbar(scatter, ax=ax1, label='抓取分数')

        # 2. 抓取分数分布
        ax2 = fig.add_subplot(2, 3, 2)
        ax2.hist(scores, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
        ax2.axvline(scores.mean(), color='red', linestyle='--',
                    label=f'平均分: {scores.mean():.3f}')
        ax2.axvline(scores.max(), color='green', linestyle=':',
                    label=f'最高分: {scores.max():.3f}')
        ax2.set_xlabel('抓取分数')
        ax2.set_ylabel('频次')
        ax2.set_title('抓取分数分布')
        ax2.legend()
        ax2.grid(True, alpha=0.3)

        # 3. 力闭合分析
        ax3 = fig.add_subplot(2, 3, 3)
        force_scores = metrics[:, 0]
        contact_bins = np.linspace(0, 1, 11)
        force_by_contact = []

        for i in range(len(contact_bins) - 1):
            mask = (metrics[:, 1] >= contact_bins[i]) & (metrics[:, 1] < contact_bins[i + 1])
            if np.any(mask):
                force_by_contact.append(force_scores[mask].mean())
            else:
                force_by_contact.append(0)

        bars = ax3.bar(range(len(force_by_contact)), force_by_contact,
                       color=plt.cm.RdYlBu(np.linspace(0, 1, len(force_by_contact))))
        ax3.set_xlabel('接触点分数区间')
        ax3.set_ylabel('平均力闭合分数')
        ax3.set_title('力闭合 vs 接触点')
        ax3.set_xticks(range(len(force_by_contact)))
        ax3.set_xticklabels([f'{contact_bins[i]:.1f}-{contact_bins[i + 1]:.1f}'
                             for i in range(len(contact_bins) - 1)], rotation=45)

        # 4. 抗扰动能力与接近角度关系
        ax4 = fig.add_subplot(2, 3, 4)
        scatter = ax4.scatter(metrics[:, 2], metrics[:, 3],
                              c=scores, cmap='plasma', s=60, alpha=0.7)
        ax4.set_xlabel('抗扰动能力分数')
        ax4.set_ylabel('接近角度分数')
        ax4.set_title('抗扰动 vs 接近角度')
        plt.colorbar(scatter, ax=ax4, label='总分')

        # 添加趋势线
        z = np.polyfit(metrics[:, 2], metrics[:, 3], 1)
        p = np.poly1d(z)
        x_line = np.linspace(metrics[:, 2].min(), metrics[:, 2].max(), 100)
        ax4.plot(x_line, p(x_line), "r--", alpha=0.5, label=f'趋势线: y={z[0]:.2f}x+{z[1]:.2f}')
        ax4.legend()

        # 5. 各指标相关性热力图
        ax5 = fig.add_subplot(2, 3, 5)
        # 修复维度不匹配问题
        # 将scores转换为列向量，然后与metrics水平拼接
        scores_col = scores.reshape(-1, 1)  # 转换为列向量 (8, 1)
        all_data = np.hstack([scores_col, metrics])  # 水平拼接 (8, 5)
        correlation_matrix = np.corrcoef(all_data.T)

        im = ax5.imshow(correlation_matrix, cmap='coolwarm', vmin=-1, vmax=1)
        ax5.set_xticks(range(len(['总分', '力闭合', '接触点', '抗扰动', '接近角'])))
        ax5.set_yticks(range(len(['总分', '力闭合', '接触点', '抗扰动', '接近角'])))
        ax5.set_xticklabels(['总分', '力闭合', '接触点', '抗扰动', '接近角'])
        ax5.set_yticklabels(['总分', '力闭合', '接触点', '抗扰动', '接近角'])
        ax5.set_title('指标相关性热力图')

        # 添加数值标签
        for i in range(correlation_matrix.shape[0]):
            for j in range(correlation_matrix.shape[1]):
                text = ax5.text(j, i, f'{correlation_matrix[i, j]:.2f}',
                                ha="center", va="center",
                                color="white" if abs(correlation_matrix[i, j]) > 0.5 else "black")

        plt.colorbar(im, ax=ax5)

        # 6. 最佳抓取姿态展示
        ax6 = fig.add_subplot(2, 3, 6, projection='3d')
        best_idx = np.argmax(scores)
        best_pos = positions[best_idx]
        best_score = scores[best_idx]

        # 绘制最佳抓取点
        ax6.scatter(best_pos[0], best_pos[1], best_pos[2],
                    c='gold', s=200, marker='*', label=f'最佳抓取点\n分数: {best_score:.3f}')

        # 绘制其他抓取点
        other_positions = np.delete(positions, best_idx, axis=0)
        ax6.scatter(other_positions[:, 0], other_positions[:, 1], other_positions[:, 2],
                    c='gray', s=20, alpha=0.3, label='其他抓取点')

        # 绘制力闭合锥（简化表示）
        u = np.linspace(0, 2 * np.pi, 30)
        v = np.linspace(0, np.pi, 30)

        x_cone = 0.1 * np.outer(np.cos(u), np.sin(v)) + best_pos[0]
        y_cone = 0.1 * np.outer(np.sin(u), np.sin(v)) + best_pos[1]
        z_cone = 0.1 * np.outer(np.ones(np.size(u)), np.cos(v)) + best_pos[2]

        ax6.plot_surface(x_cone, y_cone, z_cone, alpha=0.2, color='red')

        ax6.set_xlabel('X (m)')
        ax6.set_ylabel('Y (m)')
        ax6.set_zlabel('Z (m)')
        ax6.set_title(f'最佳抓取姿态展示')
        ax6.legend()

        plt.suptitle(f'物体 "{object_name}" 的抓取姿态分析与可视化', fontsize=16, fontweight='bold')
        plt.tight_layout()

        return fig


def main():
    """主函数：运行抓取姿态生成与评估系统"""
    print("=" * 60)
    print("抓取姿态生成与评估可视化系统")
    print("=" * 60)

    # 模型文件路径
    model_path = "arm_with_gripper.xml"  # 您的模型文件

    # 初始化组件
    print("初始化系统组件...")
    generator = GraspPoseGenerator(model_path)
    evaluator = GraspEvaluator(model_path)
    visualizer = GraspVisualizer()

    # 选择要分析的物体
    objects_to_analyze = ['test_sphere', 'test_box', 'test_cylinder']

    all_results = {}

    for obj_name in objects_to_analyze:
        print(f"\n分析物体: {obj_name}")
        print("-" * 40)

        # 1. 生成抓取姿态
        print("生成抓取姿态...")
        grasp_positions, grasp_orientations = generator.generate_random_grasp_pose(obj_name)

        # 2. 评估每个抓取姿态
        print("评估抓取稳定性...")
        grasp_data = []

        for i, (pos, quat) in enumerate(zip(grasp_positions, grasp_orientations)):
            print(f"  评估抓取点 {i + 1}/{len(grasp_positions)}...", end='\r')

            # 计算逆运动学
            joint_angles = generator.compute_ik(pos, quat)

            # 设置手指位置（简化：所有手指相同）
            finger_positions = [0.03, 0.03, 0.03]  # 滑动位置
            finger_angles = [0.5, 0.3, 0.5, 0.3, 0.5, 0.3]  # 铰链角度

            # 评估抓取
            score, metrics = evaluator.evaluate_grasp(
                obj_name, joint_angles, finger_positions, finger_angles,
                simulate=False  # 设置为True进行真实模拟
            )

            grasp_data.append({
                'position': pos,
                'orientation': quat,
                'joint_angles': joint_angles,
                'score': score,
                'metrics': metrics
            })

        print(f"\n完成评估! 生成 {len(grasp_data)} 个抓取姿态")

        # 3. 保存结果
        all_results[obj_name] = {
            'grasp_data': grasp_data,
            'average_score': np.mean([d['score'] for d in grasp_data]),
            'best_score': np.max([d['score'] for d in grasp_data])
        }

        # 4. 创建可视化
        print("生成可视化图表...")

        # 创建matplotlib图表
        mpl_fig = visualizer.create_matplotlib_visualization(grasp_data, obj_name)
        mpl_fig.savefig(f'grasp_analysis_{obj_name}.png', dpi=150, bbox_inches='tight')
        print(f"  ✓ 保存 matplotlib 图表: grasp_analysis_{obj_name}.png")

        # 创建plotly交互式图表
        plotly_fig = visualizer.create_grasp_heatmap(grasp_data, obj_name)
        plotly_fig.write_html(f'grasp_analysis_{obj_name}_interactive.html')
        print(f"  ✓ 保存交互式图表: grasp_analysis_{obj_name}_interactive.html")

        # 显示最佳抓取结果
        best_grasp = max(grasp_data, key=lambda x: x['score'])
        print(f"\n  最佳抓取点:")
        print(f"    位置: {best_grasp['position']}")
        print(f"    分数: {best_grasp['score']:.3f}")
        print(f"    力闭合: {best_grasp['metrics']['force_closure']:.3f}")
        print(f"    接触点: {best_grasp['metrics']['contact_points']:.3f}")

    # 5. 创建比较图表
    print("\n" + "=" * 60)
    print("创建物体间比较图表...")

    # 准备比较数据
    comparison_data = {}
    for obj_name, results in all_results.items():
        # 计算平均指标
        grasp_data = results['grasp_data']
        metrics_list = [d['metrics'] for d in grasp_data]

        comparison_data[obj_name] = {
            'total': results['average_score'],
            'force_closure': np.mean([m['force_closure'] for m in metrics_list]),
            'contact_points': np.mean([m['contact_points'] for m in metrics_list]),
            'wrench_resistance': np.mean([m['wrench_resistance'] for m in metrics_list]),
            'approach_angle': np.mean([m['approach_angle'] for m in metrics_list])
        }

    # 创建比较雷达图
    radar_fig = visualizer.create_grasp_comparison_radar(comparison_data)
    radar_fig.write_html('grasp_comparison_radar.html')
    print("  ✓ 保存比较雷达图: grasp_comparison_radar.html")

    # 创建比较柱状图
    fig, ax = plt.subplots(figsize=(12, 6))
    objects = list(comparison_data.keys())
    metrics_names = ['total', 'force_closure', 'contact_points',
                     'wrench_resistance', 'approach_angle']
    metric_labels = ['总分', '力闭合', '接触点', '抗扰动', '接近角']

    x = np.arange(len(objects))
    width = 0.15

    for i, (metric, label) in enumerate(zip(metrics_names, metric_labels)):
        values = [comparison_data[obj][metric] for obj in objects]
        ax.bar(x + i * width - 2 * width, values, width, label=label)

    ax.set_xlabel('物体')
    ax.set_ylabel('分数')
    ax.set_title('不同物体抓取能力比较')
    ax.set_xticks(x)
    ax.set_xticklabels(objects)
    ax.legend()
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig('grasp_comparison_bar.png', dpi=150)
    print("  ✓ 保存比较柱状图: grasp_comparison_bar.png")

    # 6. 保存数据到文件
    print("\n保存数据到文件...")
    with open('grasp_analysis_results.pkl', 'wb') as f:
        pickle.dump(all_results, f)
    print("  ✓ 保存原始数据: grasp_analysis_results.pkl")

    # 保存摘要报告
    summary = {
        'analysis_date': time.strftime("%Y-%m-%d %H:%M:%S"),
        'objects_analyzed': objects_to_analyze,
        'summary': {}
    }

    for obj_name, results in all_results.items():
        summary['summary'][obj_name] = {
            'num_grasp_poses': len(results['grasp_data']),
            'average_score': float(results['average_score']),
            'best_score': float(results['best_score'])
        }

    with open('grasp_analysis_summary.json', 'w', encoding='utf-8') as f:
        json.dump(summary, f, indent=2, ensure_ascii=False)
    print("  ✓ 保存摘要报告: grasp_analysis_summary.json")

    print("\n" + "=" * 60)
    print("分析完成!")
    print(f"分析了 {len(objects_to_analyze)} 个物体")
    print(f"生成了 {len(objects_to_analyze) * 2 + 2} 个可视化文件")
    print("=" * 60)

    # 显示一个图表
    plt.figure(figsize=(10, 6))
    objects = list(all_results.keys())
    avg_scores = [all_results[obj]['average_score'] for obj in objects]
    best_scores = [all_results[obj]['best_score'] for obj in objects]

    x = np.arange(len(objects))
    width = 0.35

    plt.bar(x - width / 2, avg_scores, width, label='平均分数', alpha=0.8)
    plt.bar(x + width / 2, best_scores, width, label='最佳分数', alpha=0.8)

    plt.xlabel('物体')
    plt.ylabel('抓取分数')
    plt.title('抓取性能总结')
    plt.xticks(x, objects)
    plt.legend()
    plt.grid(True, alpha=0.3)
    plt.tight_layout()

    # 在柱状图上添加数值
    for i, (avg, best) in enumerate(zip(avg_scores, best_scores)):
        plt.text(i - width / 2, avg + 0.01, f'{avg:.3f}',
                 ha='center', va='bottom', fontsize=9)
        plt.text(i + width / 2, best + 0.01, f'{best:.3f}',
                 ha='center', va='bottom', fontsize=9)

    plt.show()


if __name__ == "__main__":
    main()
import pyvista as pv
import numpy as np
import vtk



# 生成更像人形的仿真机器人模型（优化比例+结构）
def create_robot(return_parts=False):
    # 1. 躯干（更修长，接近人形）
    torso = pv.Cube(center=(0, 0, 0.8), x_length=0.4, y_length=0.3, z_length=1.2)
    torso["part_id"] = np.zeros(torso.n_points, dtype=int)  # 躯干ID=0

    # 2. 头部（新增，更像人形）
    head = pv.Cube(center=(0, 0, 1.6), x_length=0.3, y_length=0.3, z_length=0.3)
    head["part_id"] = np.full(head.n_points, 5, dtype=int)  # 头部ID=5

    # 3. 右胳膊（上臂+前臂，比例优化）
    right_upper_arm = pv.Cube(center=(0.5, 0, 1.0), x_length=0.2, y_length=0.2, z_length=0.5)
    right_upper_arm["part_id"] = np.ones(right_upper_arm.n_points, dtype=int)  # 右上臂ID=1
    right_forearm = pv.Cube(center=(0.9, 0, 1.0), x_length=0.2, y_length=0.2, z_length=0.5)
    right_forearm["part_id"] = np.full(right_forearm.n_points, 2, dtype=int)  # 右前臂ID=2

    # 4. 左胳膊（新增，对称结构）
    left_upper_arm = pv.Cube(center=(-0.5, 0, 1.0), x_length=0.2, y_length=0.2, z_length=0.5)
    left_upper_arm["part_id"] = np.full(left_upper_arm.n_points, 6, dtype=int)  # 左上臂ID=6
    left_forearm = pv.Cube(center=(-0.9, 0, 1.0), x_length=0.2, y_length=0.2, z_length=0.5)
    left_forearm["part_id"] = np.full(left_forearm.n_points, 7, dtype=int)  # 左前臂ID=7

    # 5. 右大腿+小腿（比例优化）
    right_thigh = pv.Cube(center=(0.2, 0, -0.2), x_length=0.2, y_length=0.2, z_length=0.6)
    right_thigh["part_id"] = np.full(right_thigh.n_points, 8, dtype=int)  # 右大腿ID=8
    right_calf = pv.Cube(center=(0.2, 0, -0.8), x_length=0.2, y_length=0.2, z_length=0.6)
    right_calf["part_id"] = np.full(right_calf.n_points, 9, dtype=int)  # 右小腿ID=9

    # 6. 左大腿+小腿（比例优化）
    left_thigh = pv.Cube(center=(-0.2, 0, -0.2), x_length=0.2, y_length=0.2, z_length=0.6)
    left_thigh["part_id"] = np.full(left_thigh.n_points, 3, dtype=int)  # 左大腿ID=3
    left_calf = pv.Cube(center=(-0.2, 0, -0.8), x_length=0.2, y_length=0.2, z_length=0.6)
    left_calf["part_id"] = np.full(left_calf.n_points, 4, dtype=int)  # 左小腿ID=4

    # 用MultiBlock合并所有部件
    block = pv.MultiBlock()
    block.append(torso)
    block.append(head)
    block.append(right_upper_arm)
    block.append(right_forearm)
    block.append(left_upper_arm)
    block.append(left_forearm)
    block.append(right_thigh)
    block.append(right_calf)
    block.append(left_thigh)
    block.append(left_calf)

    # 转换为单个UnstructuredGrid
    robot = block.combine()

    # 添加关节标记（右肘、左膝）
    robot.point_data["joints"] = np.zeros(robot.n_points, dtype=int)
    # 右肘关节（右上臂+右前臂连接点）
    elbow_idx = robot.find_closest_point((0.7, 0, 1.0))
    robot.point_data["joints"][elbow_idx] = 1
    # 左膝关节（左大腿+左小腿连接点）
    knee_idx = robot.find_closest_point((-0.2, 0, -0.5))
    robot.point_data["joints"][knee_idx] = 1

    if return_parts:
        return robot, [torso, head, right_upper_arm, right_forearm, left_upper_arm, left_forearm, right_thigh,
                       right_calf, left_thigh, left_calf]
    return robot


# 关节旋转函数（保持逻辑，适配新部件ID）
def rotate_joint(robot, joint_name, angle_deg):
    joint_config = {
        "right_elbow": {
            "center": (0.7, 0, 1.0),  # 右肘关节中心点
            "target_part": 2  # 右前臂ID=2
        },
        "left_knee": {
            "center": (-0.2, 0, -0.5),  # 左膝关节中心点
            "target_part": 4  # 左小腿ID=4
        }
    }

    if joint_name not in joint_config:
        print(f"错误：关节名称{joint_name}不存在！")
        return robot

    config = joint_config[joint_name]
    part_mask = robot["part_id"] == config["target_part"]
    if not np.any(part_mask):
        print(f"错误：未找到part_id={config['target_part']}的部件！")
        return robot

    # 旋转逻辑（与之前一致）
    target_points = robot.points[part_mask].copy()
    vtk_points = vtk.vtkPoints()
    for point in target_points:
        vtk_points.InsertNextPoint(point)

    transform = vtk.vtkTransform()
    transform.Translate(config["center"][0], config["center"][1], config["center"][2])
    transform.RotateY(angle_deg)
    transform.Translate(-config["center"][0], -config["center"][1], -config["center"][2])

    transform_filter = vtk.vtkTransformFilter()
    transform_filter.SetTransform(transform)
    poly_data = vtk.vtkPolyData()
    poly_data.SetPoints(vtk_points)
    transform_filter.SetInputData(poly_data)
    transform_filter.Update()

    rotated_vtk_points = transform_filter.GetOutput().GetPoints()
    rotated_points = np.array([rotated_vtk_points.GetPoint(i) for i in range(rotated_vtk_points.GetNumberOfPoints())])
    robot.points[part_mask] = rotated_points
    return robot


# 本地测试
if __name__ == "__main__":
    robot_initial = create_robot()
    print(f"初始模型：{robot_initial.n_points}个点，{robot_initial.n_cells}个面")

    # 测试右肘+左膝旋转
    robot_rotated = rotate_joint(robot_initial.copy(), "right_elbow", 90)
    robot_rotated = rotate_joint(robot_rotated, "left_knee", 60)

    # 渲染对比
    plotter = pv.Plotter(shape=(1, 2), window_size=(1200, 600))
    # 左窗口：初始姿态
    plotter.subplot(0, 0)
    plotter.add_mesh(
        robot_initial,
        scalars="part_id",
        cmap="tab10",  # 更多颜色区分部件
        edge_color="black",
        show_edges=True,
        scalar_bar_args={"title": "部件ID"}
    )
    plotter.add_axes()
    grid = pv.Plane(center=(0, 0, -1.2), i_size=3, j_size=3, i_resolution=10, j_resolution=10)
    plotter.add_mesh(grid, color="lightgray", opacity=0.5)
    plotter.add_text("初始人形机器人", font_size=14, position="upper_left")

    # 右窗口：旋转后姿态
    plotter.subplot(0, 1)
    plotter.add_mesh(
        robot_rotated,
        scalars="part_id",
        cmap="tab10",
        edge_color="black",
        show_edges=True,
        scalar_bar_args={"title": "部件ID"}
    )
    plotter.add_axes()
    plotter.add_mesh(grid, color="lightgray", opacity=0.5)
    plotter.add_text("抬臂+屈膝动作", font_size=14, position="upper_left")

    plotter.show()
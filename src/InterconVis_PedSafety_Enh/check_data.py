import cv2
import json
import numpy as np
import os
import glob
import cvips_utils as utils

# ================= 配置 =================
# 这里的路径要和你 collector 生成的保持一致
# 比如你刚跑了 Town05_rainy_night，这里就改对应的文件夹名
DATASET_ROOT = "_out_dataset_final" 
SCENE_NAME = "Town05_rainy_night"  # 如果你跑的是默认参数，可能是 Town01_clear_day
# =======================================

def draw_3d_box(img, target, w2c, K):
    """
    核心绘画函数：将 3D 目标画在 2D 图上
    """
    # 1. 解析目标参数
    loc = target['location']       # [x, y, z]
    rot = target['rotation']       # [pitch, yaw, roll]
    extent = target['extent']      # [ex, ey, ez]
    offset = target['center_offset'] # [ox, oy, oz]

    # 2. 恢复目标的位姿矩阵 (Local -> World)
    # 注意：这里我们临时借用 carla 的 transform 对象来算矩阵，
    # 如果不想依赖 carla 库，也可以手动写旋转矩阵公式，但用 carla 最方便
    import carla 
    t_loc = carla.Location(x=loc[0], y=loc[1], z=loc[2])
    t_rot = carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
    obj_transform = carla.Transform(t_loc, t_rot)
    obj_matrix = utils.get_matrix(obj_transform)

    # 3. 定义包围盒的 8 个顶点 (相对于物体中心)
    # CARLA extent 是半长，所以坐标是 +/- extent
    dx, dy, dz = extent[0], extent[1], extent[2]
    
    # 8个顶点的局部坐标 (x, y, z, 1)
    corners_local = np.array([
        [dx, dy, dz, 1],  [dx, -dy, dz, 1],  [dx, -dy, -dz, 1],  [dx, dy, -dz, 1], # 前面4个点
        [-dx, dy, dz, 1], [-dx, -dy, dz, 1], [-dx, -dy, -dz, 1], [-dx, dy, -dz, 1] # 后面4个点
    ]).T # 转置成 4x8

    # 加上中心偏移量 (如果有的话)
    corners_local[0, :] += offset[0]
    corners_local[1, :] += offset[1]
    corners_local[2, :] += offset[2]

    # 4. 坐标变换流水线
    # 4.1 Local -> World
    corners_world = np.dot(obj_matrix, corners_local) # 4x8

    # 4.2 World -> Pixel
    img_points = []
    for i in range(8):
        # 取出单个点的世界坐标
        p_world_vec = corners_world[:, i]
        # 使用 utils 投影到像素
        # 注意 utils.get_image_point 需要 carla.Location 类型，我们手动构造一下
        p_loc = carla.Location(x=p_world_vec[0], y=p_world_vec[1], z=p_world_vec[2])
        
        pixel = utils.get_image_point(p_loc, K, w2c)
        img_points.append(pixel)

    # 5. 连线绘画 (12条棱边)
    # 定义连接关系
    edges = [
        (0, 1), (1, 2), (2, 3), (3, 0), # 前面
        (4, 5), (5, 6), (6, 7), (7, 4), # 后面
        (0, 4), (1, 5), (2, 6), (3, 7)  # 前后连接
    ]

    color = (0, 255, 0) # 绿色代表车辆
    if target['type'] == 'walker':
        color = (0, 0, 255) # 红色代表行人

    for p1_idx, p2_idx in edges:
        p1 = tuple(img_points[p1_idx])
        p2 = tuple(img_points[p2_idx])
        # 画线
        cv2.line(img, p1, p2, color, 2)

    return img

def main():
    base_path = os.path.join(DATASET_ROOT, SCENE_NAME)
    label_dir = os.path.join(base_path, "label")
    
    if not os.path.exists(label_dir):
        print(f"❌ 错误: 找不到路径 {label_dir}")
        print("请先运行 collector 脚本生成数据，或者检查 check_data.py 里的 SCENE_NAME 配置。")
        return

    # 获取所有 json 文件并排序
    json_files = sorted(glob.glob(os.path.join(label_dir, "*.json")))
    print(f"📂 发现 {len(json_files)} 帧数据，开始回放验证...")
    print("⌨️  按任意键下一帧，按 'q' 退出")

    for j_path in json_files:
        with open(j_path, 'r') as f:
            label_data = json.load(f)

        fid = label_data['frame_id']
        
        # 读取图片
        ego_path = os.path.join(base_path, "ego_rgb", f"{fid:08d}.jpg")
        rsu_path = os.path.join(base_path, "rsu_rgb", f"{fid:08d}.jpg")

        if not os.path.exists(ego_path) or not os.path.exists(rsu_path):
            print(f"跳过缺失图片的帧: {fid}")
            continue

        img_ego = cv2.imread(ego_path)
        img_rsu = cv2.imread(rsu_path)

        # 获取矩阵参数
        # 注意: JSON 里存的是 list，转回 numpy array
        ego_w2c = np.array(label_data['matrices']['ego_w2c'])
        rsu_w2c = np.array(label_data['matrices']['rsu_w2c'])
        
        # 内参 (从 JSON 读取或者用 utils 重新生成都可以，这里用 utils 生成)
        h, w = img_ego.shape[:2]
        K = utils.build_projection_matrix(w, h, label_data['camera_params']['fov'])

        # 遍历所有目标并画框
        targets = label_data['targets']
        for tgt in targets:
            # 在主车视角画
            img_ego = draw_3d_box(img_ego, tgt, ego_w2c, K)
            # 在路侧视角画
            img_rsu = draw_3d_box(img_rsu, tgt, rsu_w2c, K)

        # 拼接显示
        # 缩小一点方便看
        img_ego_s = cv2.resize(img_ego, (960, 540))
        img_rsu_s = cv2.resize(img_rsu, (960, 540))
        
        # 上下拼接
        combined = np.vstack((img_ego_s, img_rsu_s))
        
        cv2.imshow(f"Validation - {SCENE_NAME}", combined)
        
        # 按键控制
        key = cv2.waitKey(0) # 0 表示无限等待，按键才继续；改成 30 可以自动播放
        if key == ord('q'):
            break

    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
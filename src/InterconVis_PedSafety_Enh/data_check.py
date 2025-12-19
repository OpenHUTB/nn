import cv2
import json
import numpy as np
import os
import glob
import cvips_utils as utils
import carla 

def draw_box(img, target, w2c, K):
    # 获取真值数据
    loc = target['location']
    rot = target['rotation']
    extent = target['extent']
    offset = target.get('center_offset', [0, 0, 0])
    
    # 1. 构建物体的本地到世界的变换矩阵
    # 利用 carla 现成的 Transform 类来算旋转，最不容易出错
    obj_transform = carla.Transform(
        carla.Location(x=loc[0], y=loc[1], z=loc[2]),
        carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
    )
    obj_to_world = utils.get_matrix(obj_transform)

    # 2. 定义 3D 框的 8 个顶点 (加上 offset 修正中心点位置)
    dx, dy, dz = extent[0], extent[1], extent[2]
    ox, oy, oz = offset[0], offset[1], offset[2]
    corners_local = np.array([
        [ox+dx, oy+dy, oz+dz, 1], [ox+dx, oy-dy, oz+dz, 1],
        [ox+dx, oy-dy, oz-dz, 1], [ox+dx, oy+dy, oz-dz, 1],
        [ox-dx, oy+dy, oz+dz, 1], [ox-dx, oy-dy, oz+dz, 1],
        [ox-dx, oy-dy, oz-dz, 1], [ox-dx, oy+dy, oz-dz, 1]
    ])

    # 3. 投影到像素平面
    img_pts = []
    for pt in corners_local:
        # 变换到世界坐标: World = Matrix * Local
        w_pos = np.dot(obj_to_world, pt)
        # 转换为 Location 传给 utils
        world_loc = carla.Location(x=w_pos[0], y=w_pos[1], z=w_pos[2])
        pixel = utils.get_image_point(world_loc, K, w2c)
        img_pts.append(pixel)
        
    if any(p is None for p in img_pts):
        return img
        
    # 4. 连线绘制
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    color = (0, 255, 0) if target['type'] == 'vehicle' else (0, 0, 255)
    for s, e in edges:
        cv2.line(img, tuple(img_pts[s]), tuple(img_pts[e]), color, 2)
    return img

def main():
    # 自动查找最新的文件夹
    all_dirs = sorted(glob.glob("_out_dataset_final/*"), key=os.path.getmtime)
    if not all_dirs:
        print("没有找到数据目录")
        return
    
    target_dir = all_dirs[-1] # 默认看最新的
    print(f"正在检查目录: {target_dir}")
    
    json_files = sorted(glob.glob(os.path.join(target_dir, "label", "*.json")))
    
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            
        fid = data['frame_id']
        ego_path = os.path.join(target_dir, "ego_rgb", f"{fid:08d}.jpg")
        rsu_path = os.path.join(target_dir, "rsu_rgb", f"{fid:08d}.jpg")
        
        if not os.path.exists(ego_path): continue
        
        img_ego = cv2.imread(ego_path)
        img_rsu = cv2.imread(rsu_path)
        
        # 恢复矩阵
        ego_w2c = np.array(data['matrices']['ego_w2c'])
        rsu_w2c = np.array(data['matrices']['rsu_w2c'])
        
        fov = data['camera_params']['fov']
        h, w = img_ego.shape[:2]
        K = utils.build_projection_matrix(w, h, fov)
        
        for tgt in data['targets']:
            img_ego = draw_box(img_ego, tgt, ego_w2c, K)
            img_rsu = draw_box(img_rsu, tgt, rsu_w2c, K)
            
        # 拼接显示
        vis = np.vstack([img_ego, img_rsu])
        vis = cv2.resize(vis, (0, 0), fx=0.6, fy=0.6) # 缩小一点
        
        cv2.imshow("Check Data", vis)
        if cv2.waitKey(0) == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
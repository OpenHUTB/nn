import cv2
import json
import numpy as np
import os
import glob
import carla
import cvips_utils as utils

def draw_box(img, target, w2c, K):
    loc, rot, extent = target['location'], target['rotation'], target['extent']
    offset = target.get('center_offset', [0, 0, 0])
    
    # 1. 还原物体的世界变换矩阵
    obj_transform = carla.Transform(
        carla.Location(x=loc[0], y=loc[1], z=loc[2]),
        carla.Rotation(pitch=rot[0], yaw=rot[1], roll=rot[2])
    )
    obj_to_world = utils.get_matrix(obj_transform)

    # 2. 定义 3D 框的 8 个顶点 (局部坐标)
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
        # 局部 -> 世界
        w_pos = np.dot(obj_to_world, pt)
        world_loc = carla.Location(x=w_pos[0], y=w_pos[1], z=w_pos[2])
        # 世界 -> 像素 (使用 utils)
        pixel = utils.get_image_point(world_loc, K, w2c)
        img_pts.append(pixel)
        
    if any(p is None for p in img_pts):
        return img
        
    # 4. 连线绘制
    edges = [(0,1), (1,2), (2,3), (3,0), (4,5), (5,6), (6,7), (7,4), (0,4), (1,5), (2,6), (3,7)]
    if target['type'] == 'walker': color = (0, 0, 255)
    elif target['type'] == 'bike': color = (255, 0, 255)
    else: color = (0, 255, 0)
    
    for s, e in edges:
        cv2.line(img, tuple(img_pts[s]), tuple(img_pts[e]), color, 2)
    return img

def main():
    # 查找最新的数据文件夹
    dirs = sorted(glob.glob("_out_dataset_final/*"), key=os.path.getmtime)
    if not dirs:
        print("Error: No data found.")
        return
    
    target_dir = dirs[-1]
    print(f"Checking: {target_dir}")
    
    json_files = sorted(glob.glob(os.path.join(target_dir, "label", "*.json")))
    
    for jf in json_files:
        with open(jf, 'r') as f:
            data = json.load(f)
            
        fid = data['frame_id']
        img_ego = cv2.imread(os.path.join(target_dir, "ego_rgb", f"{fid:08d}.jpg"))
        img_rsu = cv2.imread(os.path.join(target_dir, "rsu_rgb", f"{fid:08d}.jpg"))
        
        if img_ego is None or img_rsu is None: continue
        
        # 恢复矩阵和内参
        ego_w2c = np.array(data['matrices']['ego_w2c'])
        rsu_w2c = np.array(data['matrices']['rsu_w2c'])
        
        cam_p = data['camera_params']
        K = utils.build_projection_matrix(cam_p['w'], cam_p['h'], cam_p['fov'])
        
        for tgt in data['targets']:
            img_ego = draw_box(img_ego, tgt, ego_w2c, K)
            img_rsu = draw_box(img_rsu, tgt, rsu_w2c, K)
            
        # 拼接展示
        vis = np.vstack([img_ego, img_rsu])
        vis = cv2.resize(vis, (0, 0), fx=0.6, fy=0.6)
        
        cv2.imshow("Data Validation (Q to quit, Any key for next)", vis)
        if cv2.waitKey(0) == ord('q'):
            break
            
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
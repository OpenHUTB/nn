import numpy as np
import carla

import utils.cva_utils as cva_utils
import utils.img_utils as img_utils

def actor_bbox_lidar(actor_list, camera, image, lidar_image, max_dist, min_detect, class_id):
    bbox_draw = []
    bbox_save = []
    filtered_out, _ = cva_utils.auto_annotate_lidar(
        actor_list, 
        camera, 
        lidar_image,
        max_dist=max_dist,
        min_detect=min_detect, 
        show_img=None, 
        json_path=None
    )

    for npc, bbox in zip(filtered_out['vehicles'], filtered_out['bbox']):
        annotation_str = ""
        u1 = int(bbox[0,0])
        v1 = int(bbox[0,1])
        u2 = int(bbox[1,0])
        v2 = int(bbox[1,1])

        x_center = ((bbox[0,0] + bbox[1,0])/2) / image.width
        y_center = ((bbox[0,1] + bbox[1,1])/2) / image.height
        width = (bbox[1,0] - bbox[0,0]) / image.width
        height = (bbox[1,1] - bbox[0,1]) / image.height

        if np.max([x_center, y_center, width, height]) <= 1.0 and np.min([x_center, y_center, width, height]) >= 0.0:
            bbox_draw.append((u1,v1,u2,v2))
            bbox_save.append((class_id, x_center, y_center, width, height))
    return bbox_draw, bbox_save

def actor_bbox_depth(actor_list, camera, image, depth_image, max_dist, depth_margin, patch_ratio, resize_ratio, class_id):
    bbox_draw = []
    bbox_save = []
    filtered_out, _ = cva_utils.auto_annotate(
        actor_list, 
        camera, 
        depth_image,
        max_dist=max_dist,
        depth_margin=depth_margin,
    )  

    for npc, bbox in zip(filtered_out['vehicles'], filtered_out['bbox']):
        u1 = int(bbox[0,0])
        v1 = int(bbox[0,1])
        u2 = int(bbox[1,0])
        v2 = int(bbox[1,1])                           

        x_center = ((bbox[0,0] + bbox[1,0])/2) / image.width
        y_center = ((bbox[0,1] + bbox[1,1])/2) / image.height
        width = (bbox[1,0] - bbox[0,0]) / image.width
        height = (bbox[1,1] - bbox[0,1]) / image.height

        if np.max([x_center, y_center, width, height]) <= 1.0 and np.min([x_center, y_center, width, height]) >= 0.0:
            bbox_draw.append((u1,v1,u2,v2))
            bbox_save.append((class_id, x_center, y_center, width, height))

    return bbox_draw, bbox_save

def actor_bbox_depth_semantic(
    actor_list, 
    camera, 
    image, 
    depth_image, 
    semantic_image, 
    max_dist, 
    depth_margin, 
    patch_ratio, 
    resize_ratio, 
    semantic_threshold, 
    semantic_label, 
    class_id
):
    bbox_draw = []
    bbox_save = []
    filtered_out, _ = cva_utils.auto_annotate(
        actor_list, 
        camera, 
        depth_image,
        max_dist=max_dist,
        depth_margin=depth_margin,
    )

    semantic_img = np.reshape(np.copy(semantic_image.raw_data), (semantic_image.height, semantic_image.width, 4))   

    for npc, bbox in zip(filtered_out['vehicles'], filtered_out['bbox']):
        u1 = int(bbox[0,0])
        v1 = int(bbox[0,1])
        u2 = int(bbox[1,0])
        v2 = int(bbox[1,1])

        # Get the per-pixel semantic labels inside the bounding box
        semantic_bb = np.array(semantic_img[v1:v2+1,u1:u2+1,2])
        sem_patch = np.array(semantic_bb == semantic_label)
        s2 = np.sum(sem_patch) >= semantic_bb.shape[0] * semantic_bb.shape[1] * semantic_threshold

        # Passes semantic test
        if s2:
            x_center = ((bbox[0,0] + bbox[1,0])/2) / image.width
            y_center = ((bbox[0,1] + bbox[1,1])/2) / image.height
            width = (bbox[1,0] - bbox[0,0]) / image.width
            height = (bbox[1,1] - bbox[0,1]) / image.height

            if np.max([x_center, y_center, width, height]) <= 1.0 and np.min([x_center, y_center, width, height]) >= 0.0:
                bbox_draw.append((u1,v1,u2,v2))
                bbox_save.append((class_id, x_center, y_center, width, height))

    return bbox_draw, bbox_save

def object_bbox_depth(bbox_list, camera, image, depth_image, vehicle, max_dist, class_id):
    bbox_draw = []
    bbox_save = []
    K = img_utils.build_projection_matrix(image.width, image.height, image.fov) 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    for bb in bbox_list:

        # Filter for distance from ego vehicle
        dist = bb.location.distance(vehicle.get_transform().location)
        if dist < max_dist:

            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the bounding box. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            forward_vec = vehicle.get_transform().get_forward_vector()
            ray = bb.location - vehicle.get_transform().location

            if forward_vec.dot(ray) > 1:
                # Cycle through the vertices
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                x_max = float('-inf')
                x_min = float('inf')
                y_max = float('-inf')
                y_min = float('inf')                        
                for vert in verts:
                    # Join the vertices into edges
                    p = img_utils.get_image_point(vert, K, world_2_camera)
                    if p[0] > x_max:
                        x_max = p[0]
                    if p[0] < x_min:
                        x_min = p[0]
                    if p[1] > y_max:
                        y_max = p[1]
                    if p[1] < y_min:
                        y_min = p[1]

                dist_margin = 10000
                for vert in verts:
                    dist_vert = vehicle.get_transform().location.distance(vert)
                    dist_margin = min(dist_margin, dist_vert)

                xc = int((x_max+x_min)/2)
                yc = int((y_max+y_min)/2)
                wp = int((x_max-x_min) * 0.8/2)
                hp = int((y_max-y_min) * 0.8/2)
                u1 = xc-wp
                u2 = xc+wp
                v1 = yc-hp
                v2 = yc+hp

                depth_bb = np.array(depth_image[v1:v2+1,u1:u2+1])                    
                    
                dist_delta_new = np.full(depth_bb.shape, dist - 10)
                s_patch = np.array(depth_bb > dist_delta_new)
                s = np.sum(s_patch) > s_patch.shape[0]*0.1

                if s:
                    x_center = ((x_min + x_max) / 2) / image.width
                    y_center = ((y_min + y_max) / 2) / image.height
                    width = (x_max - x_min) / image.width
                    height = (y_max - y_min) / image.height
                    if np.max([x_center, y_center, width, height]) <= 1.0 and np.min([x_center, y_center, width, height]) >= 0.0:
                        bbox_draw.append((x_min,y_min,x_max,y_max))
                        bbox_save.append((class_id, x_center, y_center, width, height))

    return bbox_draw, bbox_save

def object_bbox_depth_semantic(bbox_list, camera, image, depth_image, semantic_image, vehicle, max_dist, semantic_threshold, semantic_label, class_id):
    bbox_draw = []
    bbox_save = []
    K = img_utils.build_projection_matrix(image.width, image.height, image.fov) 
    world_2_camera = np.array(camera.get_transform().get_inverse_matrix())
    semantic_img = np.reshape(np.copy(semantic_image.raw_data), (semantic_image.height, semantic_image.width, 4))
    for bb in bbox_list:

        # Filter for distance from ego vehicle
        dist = bb.location.distance(vehicle.get_transform().location)
        if dist < max_dist:

            # Calculate the dot product between the forward vector
            # of the vehicle and the vector between the vehicle
            # and the bounding box. We threshold this dot product
            # to limit to drawing bounding boxes IN FRONT OF THE CAMERA
            forward_vec = vehicle.get_transform().get_forward_vector()
            ray = bb.location - vehicle.get_transform().location

            if forward_vec.dot(ray) > 1:
                # Cycle through the vertices
                verts = [v for v in bb.get_world_vertices(carla.Transform())]
                x_max = -10000
                x_min = 10000
                y_max = -10000
                y_min = 10000                        
                for vert in verts:
                    # Join the vertices into edges
                    p = img_utils.get_image_point(vert, K, world_2_camera)
                    if p[0] > x_max:
                        x_max = p[0]
                    if p[0] < x_min:
                        x_min = p[0]
                    if p[1] > y_max:
                        y_max = p[1]
                    if p[1] < y_min:
                        y_min = p[1]

                dist_margin = 10000
                for vert in verts:
                    dist_vert = vehicle.get_transform().location.distance(vert)
                    dist_margin = min(dist_margin, dist_vert)

                xc = int((x_max+x_min)/2)
                yc = int((y_max+y_min)/2)
                wp = int((x_max-x_min) * 0.8/2)
                hp = int((y_max-y_min) * 0.8/2)
                u1 = xc-wp
                u2 = xc+wp
                v1 = yc-hp
                v2 = yc+hp

                depth_bb = np.array(depth_image[v1:v2+1,u1:u2+1])                    
                    
                dist_delta_new = np.full(depth_bb.shape, dist - 10)
                s_patch = np.array(depth_bb > dist_delta_new)
                s = np.sum(s_patch) > s_patch.shape[0]*0.1

                # Passes depth test
                if s:
                    # Get the per-pixel semantic labels inside the bounding box
                    semantic_bb = np.array(semantic_img[v1:v2+1,u1:u2+1,2])
                    sem_patch = np.array(semantic_bb == semantic_label)
                    s2 = np.sum(sem_patch) >= semantic_bb.shape[0] * semantic_bb.shape[1] * semantic_threshold

                    # Passes semantic test
                    if s2:
                        x_center = ((x_min + x_max) / 2) / image.width
                        y_center = ((y_min + y_max) / 2) / image.height
                        width = (x_max - x_min) / image.width
                        height = (y_max - y_min) / image.height
                        if np.max([x_center, y_center, width, height]) <= 1.0 and np.min([x_center, y_center, width, height]) >= 0.0:
                            bbox_draw.append((x_min,y_min,x_max,y_max))
                            bbox_save.append((class_id, x_center, y_center, width, height))

    return bbox_draw, bbox_save

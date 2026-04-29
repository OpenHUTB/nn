from __future__ import annotations

import random
from pathlib import Path
from typing import Optional

import carla
import cv2
import numpy as np
import torch
from deep_sort.deep_sort import DeepSort
from ultralytics import YOLO



class_id = [2, 3, 5, 7]
class_name = {2: "car", 3: "motobike", 5: "bus", 7: "truck"}

img_w_default = 256 * 4
img_h_default = 256 * 3
palette = (2**11 - 1, 2**15 - 1, 2**20 - 1)


class main:
    def __init__(
        self,
        yolo_weights: str = "weights/yolov8n.pt",
        deepsort_weights: str = "deep_sort/deep/checkpoint/ckpt.t7",
        output_path: str = "output.mp4",
        save_vid: bool = True,
        img_w: int = img_w_default,
        img_h: int = img_h_default,
        max_age: int = 70,
        class_ids: Optional[list[int]] = None,
    ):
        base_dir = Path(__file__).resolve().parent

        self.img_w = img_w
        self.img_h = img_h
        self.output_path = output_path
        self.save_vid = save_vid

        self.class_ids = class_ids if class_ids is not None else class_id

        self.model = self.load_model(base_dir / yolo_weights)
        self.deepsort_weights = deepsort_weights
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        self.deepsort = DeepSort(
            str(base_dir / self.deepsort_weights),
            max_age=max_age,
        )
        
    def load_model(self, weights_path: Path) -> YOLO:
        return YOLO(str(weights_path))

    def yolo_details(self, frame):
        results = self.model(frame, verbose=False)
        if not results:
            return frame, [], [], []

        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return frame, [], [], []

        # ultralytics: xyxy (N,4), conf (N,), cls (N,)
        xyxy = boxes.xyxy.cpu().numpy()
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        mask = np.isin(clss, np.array(self.class_ids, dtype=int))
        if not np.any(mask):
            return frame, [], [], []

        xyxy_f = xyxy[mask].astype(int)
        confs_f = confs[mask].astype(float).tolist()
        clss_f = clss[mask].astype(int).tolist()

        bbox_xyxy = xyxy_f.tolist()
        return frame, bbox_xyxy, confs_f, clss_f

    @staticmethod
    def _iou_xyxy(box_a: list[int] | np.ndarray, box_b: list[int] | np.ndarray) -> float:
        ax1, ay1, ax2, ay2 = map(int, box_a)
        bx1, by1, bx2, by2 = map(int, box_b)

        inter_x1 = max(ax1, bx1)
        inter_y1 = max(ay1, by1)
        inter_x2 = min(ax2, bx2)
        inter_y2 = min(ay2, by2)

        inter_w = max(0, inter_x2 - inter_x1)
        inter_h = max(0, inter_y2 - inter_y1)
        inter_area = inter_w * inter_h

        area_a = max(0, ax2 - ax1) * max(0, ay2 - ay1)
        area_b = max(0, bx2 - bx1) * max(0, by2 - by1)
        union = area_a + area_b - inter_area
        if union <= 0:
            return 0.0
        return inter_area / union

    def _match_cls_by_iou(
        self,
        track_xyxy: list[int] | np.ndarray,
        det_bboxes: list[list[int]],
        det_cls_ids: list[int],
    ) -> int:
        if not det_bboxes or not det_cls_ids:
            return -1

        best_iou = 0.0
        best_idx = -1
        for i, det in enumerate(det_bboxes):
            iou = self._iou_xyxy(track_xyxy, det)
            if iou > best_iou:
                best_iou = iou
                best_idx = i

        return det_cls_ids[best_idx] if best_idx >= 0 else -1
          
    
    def __call__(self):
        # The local Host for carla simulator is 2000
        client = carla.Client('localhost', 2000)
        client.set_timeout(20.0)
        world = client.get_world() 

        '''
       climate = carla.WeatherParameters(
                    cloudiness=50.0,
                    precipitation=90.0,
                    sun_altitude_angle=70.0,
                    wetness = 50.0,
                    fog_density = 50.0)
        
        world.set_weather(climate)
        '''

        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = world.get_blueprint_library().find('vehicle.lincoln.mkz_2020')
        vehicle_bp.set_attribute('role_name', 'ego')
        ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle in CARLA.")

        spectator = world.get_spectator()
        transform = carla.Transform(ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)), carla.Rotation(yaw=-180, pitch=-90))
        spectator.set_transform(transform)

        for i in range(60):
            vehicle_bp = random.choice(world.get_blueprint_library().filter('vehicle'))
            npc = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
            
        if npc:
            for v in world.get_actors().filter('*vehicle*'):
                v.set_autopilot(True)
                   
        camera_bp = world.get_blueprint_library().find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', f"{self.img_w}")
        camera_bp.set_attribute('image_size_y', f"{self.img_h}")
        camera_bp.set_attribute('fov', '110')
        
        camera_location = carla.Location(2,0,1)
        camera_rotation = carla.Rotation(0,180,0)

        camera_init_trans = carla.Transform(camera_location,camera_rotation)
        camera = world.spawn_actor(camera_bp, camera_init_trans, attach_to=ego_vehicle , attachment_type=carla.AttachmentType.SpringArmGhost)
        
        def capture_image(image):
            image_data = np.array(image.raw_data)
            image_rgb = image_data.reshape((image.height, image.width, 4))[:, :, :3]
            return image_rgb

        image_w = camera_bp.get_attribute('image_size_x').as_int()
        image_h = camera_bp.get_attribute('image_size_y').as_int()

        camera_data = {"image": np.zeros((image_h, image_w, 3), dtype=np.uint8)}
        camera.listen(lambda image: camera_data.update({'image': capture_image(image)}))
        
        ego_vehicle.set_autopilot(True)

        video_writer = None
        if self.save_vid:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(
                self.output_path,
                fourcc,
                14.0,
                (self.img_w, self.img_h),
            )

        try:
            while True:
                frame = camera_data['image']
                frame , bbox_xyxy , conf_score , cls_id = self.yolo_details(frame)              
                outputs = self.deepsort.update(bbox_xyxy, conf_score, frame)

                if len(outputs) > 0:
                    for output in outputs:
                        x1, y1, x2, y2, track_id = output.tolist()
                        track_xyxy = [x1, y1, x2, y2]
                        matched_cls = self._match_cls_by_iou(track_xyxy, bbox_xyxy, cls_id)
                        frame = self.draw_bbox(frame, output, matched_cls)

                cv2.imshow("deepSORT", frame)
                if video_writer is not None:
                    video_writer.write(frame)
                   
                if cv2.waitKey(1) == ord('q'):
                    break 
        finally:
            if video_writer is not None:
                video_writer.release()
            self.destroy_world(camera, ego_vehicle, world)
            print('all actors destroyed')
     
    def destroy_world(self, camera, vehicle, world):
        cv2.destroyAllWindows()
        if camera is not None:
            camera.stop()
            camera.destroy()
        if vehicle is not None:
            vehicle_id = vehicle.id
            vehicle.destroy()
        else:
            vehicle_id = None
        for npc in world.get_actors().filter("vehicle*"):
            if npc is not None and npc.id != vehicle_id:
                npc.destroy()

    def colour_label(self , label):

        label_colour = [int((p * (label ** 2 - label + 1)) % 255) for p in palette]
        return tuple(label_colour)    
    
    def draw_bbox(self, frame, output, cls_id: int):
        x1, y1, x2, y2 = map(int,output[0:4])
        id = int(output[4])
        label = class_name.get(cls_id, '')

        frame = np.array(frame) if not isinstance(frame, np.ndarray) else frame

        colour = self.colour_label(id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1 , 1)[0]
        c_id = f'{label} {id}'
        cv2.rectangle(frame, (x1, y1),(x2,y2), colour, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), colour, -1)
        cv2.putText(frame, c_id, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)

        return frame
    
if __name__ == '__main__':
    run = main()
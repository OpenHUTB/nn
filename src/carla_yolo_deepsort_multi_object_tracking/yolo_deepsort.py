import random
from pathlib import Path

import carla
import cv2
import numpy as np
import torch
from deep_sort.deep_sort import DeepSort
from deep_sort.utils.parser import get_config
from ultralytics import YOLO

CLASS_IDS = [2, 3, 5, 7]
CLASS_NAMES = {2: "car", 3: "motorbike", 5: "bus", 7: "truck"}
PALETTE = (2**11 - 1, 2**15 - 1, 2**20 - 1)

IMG_W = 256 * 4
IMG_H = 256 * 3


class main:
    def __init__(self):
        base_dir = Path(__file__).resolve().parent
        self.save_vid = True
        self.output_path = str(base_dir / "output.mp4")

        self.model = YOLO(str(base_dir / "weights" / "yolov8n.pt"))
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        cfg_path = base_dir / "deep_sort" / "configs" / "deep_sort.yaml"
        self.cfg = get_config(str(cfg_path))
        ds_cfg = self.cfg.DEEPSORT

        reid_ckpt = (base_dir / ds_cfg.REID_CKPT).resolve()
        self.deepsort = DeepSort(
            model_path=str(reid_ckpt),
            max_dist=float(ds_cfg.MAX_DIST),
            min_confidence=float(ds_cfg.MIN_CONFIDENCE),
            nms_max_overlap=float(ds_cfg.NMS_MAX_OVERLAP),
            max_iou_distance=float(ds_cfg.MAX_IOU_DISTANCE),
            max_age=int(ds_cfg.MAX_AGE),
            n_init=int(ds_cfg.N_INIT),
            nn_budget=int(ds_cfg.NN_BUDGET),
            use_cuda=torch.cuda.is_available(),
        )

    def yolo_details(self, frame):
        results = self.model(frame, verbose=False)
        boxes = results[0].boxes
        if boxes is None or len(boxes) == 0:
            return frame, [], [], []

        xyxy = boxes.xyxy.cpu().numpy().astype(int)
        confs = boxes.conf.cpu().numpy()
        clss = boxes.cls.cpu().numpy().astype(int)

        mask = np.isin(clss, np.array(CLASS_IDS, dtype=int))
        if not np.any(mask):
            return frame, [], [], []

        return frame, xyxy[mask].tolist(), confs[mask].tolist(), clss[mask].tolist()

    @staticmethod
    def _iou_xyxy(box_a, box_b):
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
        return inter_area / union if union > 0 else 0.0

    def _match_cls_by_iou(self, track_xyxy, det_boxes, det_cls):
        if not det_boxes:
            return -1
        best_iou = 0.0
        best_idx = -1
        for i, box in enumerate(det_boxes):
            iou = self._iou_xyxy(track_xyxy, box)
            if iou > best_iou:
                best_iou = iou
                best_idx = i
        return det_cls[best_idx] if best_idx >= 0 else -1

    def __call__(self):
        client = carla.Client("localhost", 2000)
        client.set_timeout(20.0)
        world = client.get_world()

        spawn_points = world.get_map().get_spawn_points()
        vehicle_bp = world.get_blueprint_library().find("vehicle.lincoln.mkz_2020")
        vehicle_bp.set_attribute("role_name", "ego")
        ego_vehicle = world.try_spawn_actor(vehicle_bp, random.choice(spawn_points))
        if ego_vehicle is None:
            raise RuntimeError("Failed to spawn ego vehicle")

        spectator = world.get_spectator()
        transform = carla.Transform(
            ego_vehicle.get_transform().transform(carla.Location(x=-4, z=2.5)),
            carla.Rotation(yaw=-180, pitch=-90),
        )
        spectator.set_transform(transform)

        spawned = []
        for _ in range(60):
            bp = random.choice(world.get_blueprint_library().filter("vehicle"))
            npc = world.try_spawn_actor(bp, random.choice(spawn_points))
            if npc is not None:
                spawned.append(npc)

        for v in world.get_actors().filter("*vehicle*"):
            v.set_autopilot(True)

        camera_bp = world.get_blueprint_library().find("sensor.camera.rgb")
        camera_bp.set_attribute("image_size_x", f"{IMG_W}")
        camera_bp.set_attribute("image_size_y", f"{IMG_H}")
        camera_bp.set_attribute("fov", "110")

        camera_init_trans = carla.Transform(carla.Location(2, 0, 1), carla.Rotation(0, 180, 0))
        camera = world.spawn_actor(
            camera_bp,
            camera_init_trans,
            attach_to=ego_vehicle,
            attachment_type=carla.AttachmentType.SpringArmGhost,
        )

        def capture_image(image):
            data = np.array(image.raw_data)
            return data.reshape((image.height, image.width, 4))[:, :, :3]

        image_w = camera_bp.get_attribute("image_size_x").as_int()
        image_h = camera_bp.get_attribute("image_size_y").as_int()
        camera_data = {"image": np.zeros((image_h, image_w, 3), dtype=np.uint8)}
        camera.listen(lambda image: camera_data.update({"image": capture_image(image)}))

        video_writer = None
        if self.save_vid:
            fourcc = cv2.VideoWriter_fourcc(*"mp4v")
            video_writer = cv2.VideoWriter(self.output_path, fourcc, 14.0, (IMG_W, IMG_H))

        try:
            while True:
                frame = camera_data["image"]
                frame, bbox_xyxy, conf_score, cls_id = self.yolo_details(frame)
                outputs = self.deepsort.update(bbox_xyxy, conf_score, frame)

                if len(outputs) > 0:
                    for output in outputs:
                        track_box = output[0:4]
                        matched_cls = self._match_cls_by_iou(track_box, bbox_xyxy, cls_id)
                        frame = self.draw_bbox(frame, output, matched_cls)

                cv2.imshow("deepSORT", frame)
                if video_writer is not None:
                    video_writer.write(frame)

                if cv2.waitKey(1) == ord("q"):
                    break
        finally:
            if video_writer is not None:
                video_writer.release()
            self.destroy_world(camera, ego_vehicle, world)
            print("all actors destroyed")

    def destroy_world(self, camera, vehicle, world):
        cv2.destroyAllWindows()
        if camera is not None:
            camera.stop()
            camera.destroy()
        vehicle_id = vehicle.id if vehicle is not None else None
        if vehicle is not None:
            vehicle.destroy()
        for npc in world.get_actors().filter("vehicle*"):
            if npc is not None and npc.id != vehicle_id:
                npc.destroy()

    def colour_label(self, label):
        label_colour = [int((p * (label**2 - label + 1)) % 255) for p in PALETTE]
        return tuple(label_colour)

    def draw_bbox(self, frame, output, cls_id):
        x1, y1, x2, y2 = map(int, output[0:4])
        track_id = int(output[4])
        label = CLASS_NAMES.get(cls_id, "")
        colour = self.colour_label(track_id)
        t_size = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 1, 1)[0]
        c_id = f"{label} {track_id}"
        cv2.rectangle(frame, (x1, y1), (x2, y2), colour, 1)
        cv2.rectangle(frame, (x1, y1), (x1 + t_size[0] + 3, y1 + t_size[1] + 4), colour, -1)
        cv2.putText(frame, c_id, (x1, y1 + t_size[1] + 4), cv2.FONT_HERSHEY_PLAIN, 1, [255, 255, 255], 2)
        return frame


if __name__ == "__main__":
    run = main()
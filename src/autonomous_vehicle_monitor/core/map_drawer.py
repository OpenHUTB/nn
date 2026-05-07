import carla
import numpy as np
import cv2

class MapDrawer:
    def __init__(self, world, vehicle, bev_shape=(960, 640), scale=10):
        self.world = world
        self.vehicle = vehicle
        self.map = world.get_map()
        self.bev_h, self.bev_w = bev_shape
        self.scale = scale
        self.cx = self.bev_w // 2
        self.cy = self.bev_h // 2

    def draw_lanes_and_drivable_area(self, bev_img):
        ego_tf = self.vehicle.get_transform()
        ego_loc = ego_tf.location
        ego_yaw = ego_tf.rotation.yaw

        waypoint = self.map.get_waypoint(ego_loc, project_to_road=True, lane_type=carla.LaneType.Driving)
        if not waypoint:
            return

        steps = 20
        step_len = 2.0
        left_pts = []
        right_pts = []
        center_pts = []

        for i in range(-steps, steps):
            try:
                if i > 0:
                    wps = waypoint.next(i * step_len)
                else:
                    wps = waypoint.previous(-i * step_len)

                if not wps:
                    continue
                wp = wps[0]

                w = wp.lane_width / 2.0
                c = wp.transform.location
                yaw_rad = np.radians(wp.transform.rotation.yaw)

                dx_l = -w * np.sin(yaw_rad)
                dy_l = w * np.cos(yaw_rad)
                dx_r = w * np.sin(yaw_rad)
                dy_r = -w * np.cos(yaw_rad)

                def to_bev(x, y):
                    dx = x - ego_loc.x
                    dy = y - ego_loc.y
                    c = np.cos(np.radians(-ego_yaw))
                    s = np.sin(np.radians(-ego_yaw))
                    rx = dx * c - dy * s
                    ry = dx * s + dy * c
                    px = self.cx + int(ry * self.scale)
                    py = self.cy - int(rx * self.scale)
                    return px, py

                clx, cly = to_bev(c.x, c.y)
                lx, ly = to_bev(c.x + dx_l, c.y + dy_l)
                rx, ry = to_bev(c.x + dx_r, c.y + dy_r)

                center_pts.append((clx, cly))
                left_pts.append((lx, ly))
                right_pts.append((rx, ry))
            except:
                continue

        if center_pts:
            cv2.polylines(bev_img, [np.array(center_pts)], False, (0, 255, 255), 2)
        if left_pts:
            cv2.polylines(bev_img, [np.array(left_pts)], False, (255, 255, 255), 1)
        if right_pts:
            cv2.polylines(bev_img, [np.array(right_pts)], False, (255, 255, 255), 1)

        try:
            if len(left_pts) > 0 and len(right_pts) > 0:
                area_pts = left_pts + right_pts[::-1]
                cv2.fillPoly(bev_img, [np.array(area_pts)], (30, 60, 30))
        except:
            pass
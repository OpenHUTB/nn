import cv2
import math
import carla
import numpy as np
from threading import Lock
from concurrent.futures import ThreadPoolExecutor
import numpy as np

class EagleEyeMap:
    def __init__(self, world, ego_vehicle, window_size=(720, 720), zoom=1.0):
        self.world = world
        self.ego_vehicle = ego_vehicle
        self.window_size = window_size
        self.zoom = zoom
        self.window_name = "EagleEyeMap"
        self.map_window = np.zeros((window_size[1], window_size[0], 3), dtype=np.uint8)
        self.xy_paddings = (-300, 300), (-300, 300)
        
        self.lock = Lock()
        self.running = True
        self.click_callback = None

        # Objects
        self.static_objects = self.get_static_objects(world)
        self.dynamic_objects = self.get_dynamic_objects(world)
        
        # Colors
        self.background_color = (30, 30, 30)
        self.ego_color = (255, 255, 255)
        self.vehicle_color = (255, 10, 10)
        self.roads_color = (250, 80, 80)
        self.roadlines_color = (300, 100, 100)
        self.sidewalks_color = (100, 80, 100)
        self.signs_color = (200, 100, 50)
        self.poles_color = (150, 150, 150)
        self.buildings_color = (0, 0, 255)
        self.border_color = (120, 120, 0)
        self.lane_markings_color = (255, 255, 255)
        self.dot_radius = 3
    
    def pixels_to_world(self, pixel):
        px, py = pixel
        w, h = self.window_size
        (x_min, x_max), (y_min, y_max) = self.xy_paddings
        world_x = int(x_min + (px / w) * (x_max - x_min))
        world_y = int(y_max - (py / h) * (y_max - y_min))
        return (world_x, world_y)
    
    def world_to_pixels(self, pos):
        world_x, world_y = pos
        (x_min, x_max), (y_min, y_max) = self.xy_paddings
        w, h = self.window_size
        px = int((world_x - x_min) / (x_max - x_min) * w)
        py = int((y_max - world_y) / (y_max - y_min) * h)
        return (px, py)
    
    def _on_click_callback(self, event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            world_pos = self.pixels_to_world((x, y))
            print(f"Clicked pixels ({x}, {y}) --> world position: {world_pos}")    
    
    def run(self):
        cv2.namedWindow(self.window_name)
        cv2.setMouseCallback(self.window_name, self._on_click_callback)
        while self.running:
            self.map_window[:] = self.background_color
            with self.lock:
                self.draw_objects()
                cv2.line(self.map_window, (0, self.window_size[1] // 2), (self.window_size[0], self.window_size[1] // 2), (100, 100, 100), 1)
                cv2.line(self.map_window, (self.window_size[0] // 2, 0), (self.window_size[0] // 2, self.window_size[1]), (100, 100, 100), 1)
            cv2.imshow(self.window_name, self.map_window)
            if cv2.waitKey(30) & 0xFF == 27:
                self.stop()
        cv2.destroyAllWindows()

    def stop(self):
        self.running = False
    
    def get_static_objects(self, world):
        buildings = world.get_environment_objects(carla.CityObjectLabel.Buildings)
        traffic_signs = world.get_environment_objects(carla.CityObjectLabel.TrafficSigns)
        poles = world.get_environment_objects(carla.CityObjectLabel.Poles)
        sidewalks = world.get_environment_objects(carla.CityObjectLabel.Sidewalks)
        roads = world.get_environment_objects(carla.CityObjectLabel.Roads)
        roadlines = world.get_environment_objects(carla.CityObjectLabel.RoadLines)
        
        static_objects = {
            'buildings': [bb for bb in buildings],
            'traffic_signs': [sign for sign in traffic_signs],
            'poles': [p for p in poles],
            'sidewalks': [sw for sw in sidewalks],
            'roads': [r for r in roads],
            'roadlines': [rl for rl in roadlines],
        }
        return static_objects
    
    def get_dynamic_objects(self, world):
        actor_list = world.get_actors()
        vehicles = actor_list.filter('vehicle.*')
        pedestrians = actor_list.filter('walker.*')
        traffic_lights = actor_list.filter('traffic.traffic_light*')
        
        dynamic_objects = {
            'vehicles': [(v.id, v.get_transform()) for v in vehicles],
            'pedestrians': [(p.id, p.get_transform()) for p in pedestrians],
            'traffic_lights': [tl.get_state() for tl in traffic_lights]
        }
        return dynamic_objects
    
    def draw_objects(self):
        # static objects
        for building in self.static_objects['buildings']:
            self.draw_bboxs(building.bounding_box, self.buildings_color)
        
        for road in self.static_objects['roads']:
            self.draw_bboxs(road.bounding_box, self.roads_color)
            
        for roadline in self.static_objects['roadlines']:
            self.draw_bboxs(roadline.bounding_box, self.roadlines_color)
            
        for sidewalk in self.static_objects['sidewalks']:
            self.draw_bboxs(sidewalk.bounding_box, self.sidewalks_color)
        
        for sign in self.static_objects['traffic_signs']:
            self.draw_bboxs(sign.bounding_box, self.signs_color)
            
        for pole in self.static_objects['poles']:
            self.draw_bboxs(pole.bounding_box, self.poles_color)
        
        # dynamic objects
        self.dynamic_objects = self.get_dynamic_objects(self.world)

        for _, transform in self.dynamic_objects['vehicles']:
            loc = transform.location
            self.draw_vehicle((loc.x, loc.y), self.vehicle_color)
        
        for _, transform in self.dynamic_objects['pedestrians']:
            loc = transform.location
            self.draw_vehicle((loc.x, loc.y), (0, 200, 200))
            
        ego_location = self.ego_vehicle.get_location()
        self.draw_vehicle((ego_location.x, ego_location.y), self.ego_color)
        
    def draw_vehicle(self, world_pos, color=(255, 255, 50)):
        px, py = self.world_to_pixels(world_pos)
        cv2.circle(self.map_window, (px, py), 5, color, -1)   
    
    def draw_bboxs(self, bboxs, color):
        loc = bboxs.location
        extent = bboxs.extent
        rotation = bboxs.rotation.yaw
        corners = [
            (loc.x - extent.x, loc.y - extent.y),
            (loc.x + extent.x, loc.y - extent.y),
            (loc.x + extent.x, loc.y + extent.y),
            (loc.x - extent.x, loc.y + extent.y),
        ]
        pixels = [self.world_to_pixels((pt[0], pt[1])) for pt in corners]
        angle = -rotation
        # for i in range(4):
        #     cv2.line(self.map_window, pixels[i], pixels[(i+1)%4], color, 1)
        rect = (loc, (extent.x, extent.y), angle)
        box = cv2.boxPoints(rect).astype(int)
        cv2.drawContours(self.map_window, [box], 0, color, -1)
    
    
    
    
    
    
    
    
    # Colors
    # self.background = (30, 30, 30)
    # self.ego_color = (10, 255, 10)
    # self.vehicle_color = (255, 10, 10)
    # self.roads_color = (250, 80, 80)
    # self.buildings_color = (0, 0, 255)
    # self.border_color = (120, 120, 0)
    # self.lane_markings_color = (255, 255, 255)
    # self.dot_radius = 3
    # self.map = self.world.get_map()
    
    # # Calculate map bounds using waypoints
    # waypoints = self.map.generate_waypoints(2.0)
    # x_coords = [wp.transform.location.x for wp in waypoints]
    # y_coords = [wp.transform.location.y for wp in waypoints]
    
    # # Add padding to map bounds
    # padding = 50  # meters of padding around the map
    # self.min_x, self.max_x = min(x_coords) - padding, max(x_coords) + padding
    # self.min_y, self.max_y = min(y_coords) - padding, max(y_coords) + padding
    
    # # Calculate scale factors for better canvas utilization
    # self.x_scale = self.width / (self.max_x - self.min_x)
    # self.y_scale = self.height / (self.max_y - self.min_y)
    # self.scale = min(self.x_scale, self.y_scale) * 0.9  # Use 90% of available space
    
    # # Get all buildings
    # self.buildings = [b for b in world.get_actors().filter('*static.building*') if b.bounding_box]
    
    # # Cache for waypoints
    # self._waypoints_cache = None
    # self._last_update_time = 0
    # self.update_interval = 0.1

    # # Store the initial ego vehicle position for static elements
    # self._initial_ego_position = self.ego_vehicle.get_location()
    
    # # Initialize static elements
    # self._draw_static_elements()
    
    # def world_to_map(self, location):
    #     """Convert world coordinates to map coordinates using absolute positioning"""
    #     # Convert world coordinates to map coordinates using absolute positioning
    #     x = (location.x - self.min_x) / (self.max_x - self.min_x) * self.width
    #     y = (location.y - self.min_y) / (self.max_y - self.min_y) * self.height
    #     y = self.height - y  # Flip Y axis for pygame

    #     # Clamp coordinates to canvas bounds
    #     x = max(0, min(x, self.width - 1))
    #     y = max(0, min(y, self.height - 1))

    #     return int(x), int(y)    
    
    # def _draw_static_elements(self):
    #     """Draw static elements using absolute world coordinates"""
    #     self.static_surface.fill(self.background)
        
    #     # Draw roads
    #     waypoints = self.get_waypoints()
    #     for waypoint in waypoints:
    #         road = waypoint.road_id
    #         if road is None:
    #             continue
                
    #         points = []
    #         current_wp = waypoint
    #         while current_wp is not None:
    #             loc = current_wp.transform.location
    #             points.append(self.world_to_map(loc))
                
    #             next_wps = current_wp.next(2.0)
    #             if not next_wps or next_wps[0].road_id != road:
    #                 break
    #             current_wp = next_wps[0]
            
    #         if len(points) >= 2:
    #             # Draw road with adjusted width based on scale
    #             road_width = max(2, int(3 * self.scale / 10))
    #             pygame.draw.lines(self.static_surface, self.roads_color, False, points, road_width)
                
    #             # Draw lane markings
    #             left_wp = waypoint.get_left_lane()
    #             right_wp = waypoint.get_right_lane()
                
    #             if left_wp:
    #                 left_loc = left_wp.transform.location
    #                 left_point = self.world_to_map(left_loc)
    #                 pygame.draw.circle(self.static_surface, self.lane_markings_color, left_point, max(1, int(self.scale / 20)))
                
    #             if right_wp:
    #                 right_loc = right_wp.transform.location
    #                 right_point = self.world_to_map(right_loc)
    #                 pygame.draw.circle(self.static_surface, self.lane_markings_color, right_point, max(1, int(self.scale / 20)))

    #     # Draw buildings with improved scaling
    #     for building in self.buildings:
    #         if not building.is_alive:
    #             continue
                
    #         bbox = building.bounding_box
    #         transform = building.get_transform()
    #         center = transform.transform(bbox.location)
    #         extent = bbox.extent

    #         corners = [
    #             carla.Location(x=+extent.x, y=+extent.y, z=0),
    #             carla.Location(x=-extent.x, y=+extent.y, z=0),
    #             carla.Location(x=-extent.x, y=-extent.y, z=0),
    #             carla.Location(x=+extent.x, y=-extent.y, z=0),
    #         ]
    #         world_corners = [transform.transform(c + bbox.location) for c in corners]
    #         points = [self.world_to_map(loc) for loc in world_corners]
    #         pygame.draw.polygon(self.static_surface, self.buildings_color, points)

    # def get_waypoints(self):
    #     current_time = self.world.get_snapshot().timestamp.elapsed_seconds
    #     if (self._waypoints_cache is None or 
    #         current_time - self._last_update_time > self.update_interval):
    #         self._waypoints_cache = self.map.generate_waypoints(2.0)
    #         self._last_update_time = current_time
    #     return self._waypoints_cache

    # def draw_dynamic_elements(self):
    #     """Draw dynamic elements using absolute world coordinates"""
    #     self.dynamic_surface.fill((0, 0, 0, 0))
        
    #     for vehicle in self.world.get_actors().filter('*vehicle*'):
    #         if not vehicle.is_alive:
    #             continue
                
    #         transform = vehicle.get_transform()
    #         bbox = vehicle.bounding_box
            
    #         # Calculate vehicle corners with proper scaling
    #         corners = [
    #             carla.Location(x=+bbox.extent.x, y=+bbox.extent.y, z=0),
    #             carla.Location(x=-bbox.extent.x, y=+bbox.extent.y, z=0),
    #             carla.Location(x=-bbox.extent.x, y=-bbox.extent.y, z=0),
    #             carla.Location(x=+bbox.extent.x, y=-bbox.extent.y, z=0),
    #         ]
            
    #         world_corners = [transform.transform(c + bbox.location) for c in corners]
    #         points = [self.world_to_map(loc) for loc in world_corners]
            
    #         # Draw vehicle with color based on type
    #         color = self.ego_color if vehicle.id == self.ego_vehicle.id else self.vehicle_color
    #         pygame.draw.polygon(self.dynamic_surface, color, points)
            
    #         # Draw heading indicator for ego vehicle
    #         if vehicle.id == self.ego_vehicle.id:
    #             heading = math.radians(transform.rotation.yaw)
    #             start = self.world_to_map(transform.location)
    #             # Scale the heading line based on map scale
    #             heading_length = max(10, int(15 * self.scale / 10))
    #             end = (
    #                 start[0] + int(heading_length * math.cos(heading)),
    #                 start[1] - int(heading_length * math.sin(heading))
    #             )
    #             pygame.draw.line(self.dynamic_surface, (0, 255, 255), start, end, max(2, int(self.scale / 10)))

    # def draw_map(self):
    #     """Draw static and dynamic elements"""
    #     self.map_surface.fill(self.background)
    #     self.map_surface.blit(self.static_surface, (0, 0))
    #     self.map_surface.blit(self.dynamic_surface, (0, 0))
    #     return self.map_surface
    
    # def update(self):
    #     """Update the map visualization"""
    #     self.draw_dynamic_elements()  # Only update dynamic elements
    #     return self.draw_map()
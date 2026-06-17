import glob
import os
import sys

try:
    import carla
except ImportError:
    print("Error: carla module not found")
    sys.exit(1)

import time
import numpy as np
import cv2
import pygame
import logging

logging.basicConfig(level=logging.INFO)

WINDOW_SIZE = (1280, 720)  # 减小窗口尺寸

def main():
    pygame.init()
    pygame.font.init()
    
    screen = pygame.display.set_mode(WINDOW_SIZE, pygame.RESIZABLE)
    pygame.display.set_caption("CARLA Simple Demo")
    
    try:
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0)
        world = client.get_world()
        print(f"Connected to CARLA! Map: {world.get_map().name}")
        
        blueprint_library = world.get_blueprint_library()
        
        vehicle_bp = blueprint_library.find('vehicle.tesla.cybertruck')
        spawn_points = world.get_map().get_spawn_points()
        
        ego_vehicle = world.spawn_actor(vehicle_bp, spawn_points[0])
        print(f"Vehicle spawned: {ego_vehicle.type_id}")
        
        camera_bp = blueprint_library.find('sensor.camera.rgb')
        camera_bp.set_attribute('image_size_x', str(WINDOW_SIZE[0]))
        camera_bp.set_attribute('image_size_y', str(WINDOW_SIZE[1]))
        camera_bp.set_attribute('fov', '90')
        
        camera_transform = carla.Transform(carla.Location(x=2.0, z=1.5))
        camera = world.spawn_actor(camera_bp, camera_transform, attach_to=ego_vehicle)
        
        image_queue = []
        def camera_callback(image):
            array = np.frombuffer(image.raw_data, dtype=np.dtype("uint8"))
            array = np.reshape(array, (image.height, image.width, 4))
            array = array[:, :, :3]
            array = array[:, :, ::-1]
            image_queue.append(array)
        
        camera.listen(camera_callback)
        
        control = carla.VehicleControl()
        control.throttle = 0.3
        control.steer = 0.0
        
        clock = pygame.time.Clock()
        running = True
        
        while running:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        running = False
                    elif event.key == pygame.K_w:
                        control.throttle = min(1.0, control.throttle + 0.1)
                    elif event.key == pygame.K_s:
                        control.throttle = max(0.0, control.throttle - 0.1)
                    elif event.key == pygame.K_a:
                        control.steer = max(-1.0, control.steer - 0.1)
                    elif event.key == pygame.K_d:
                        control.steer = min(1.0, control.steer + 0.1)
                    elif event.key == pygame.K_SPACE:
                        control.brake = 1.0
                    else:
                        control.brake = 0.0
            
            ego_vehicle.apply_control(control)
            
            if image_queue:
                img = image_queue[-1]
                img_surface = pygame.surfarray.make_surface(img.swapaxes(0, 1))
                screen.blit(img_surface, (0, 0))
                image_queue.clear()
            
            pygame.display.flip()
            clock.tick(30)
            
            world.tick()
    
    except Exception as e:
        logging.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
    
    finally:
        print("Cleaning up...")
        if 'ego_vehicle' in locals():
            ego_vehicle.destroy()
        if 'camera' in locals():
            camera.destroy()
        pygame.quit()

if __name__ == '__main__':
    main()

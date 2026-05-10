import math
import pygame
import queue
import cv2
import carla
import logging

class EgoVehicleController:
    def __init__(self) -> None:
        self.controller = None
    
    def setup_ego_vehicle(self, ego_vehicle):
        """Configure ego vehicle for automatic movement"""
        # Set up basic control parameters
        self.controller = carla.VehicleControl()
        self.controller.throttle = 0.5  # 50% throttle
        self.controller.steer = 0.0     # No steering initially
        ego_vehicle.apply_control(self.controller)
        return self.controller

    def update_ego_vehicle(self, ego_vehicle, control):
        """Update ego vehicle movement with collision avoidance"""
        # Get current transform and velocity
        transform = ego_vehicle.get_transform()
        velocity = ego_vehicle.get_velocity()
        speed = math.sqrt(velocity.x**2 + velocity.y**2 + velocity.z**2) * 3.6  # Convert to km/h

        # Basic speed control
        if speed < 30.0:  # Target speed of 30 km/h
            control.throttle = 0.5
            control.brake = 0.0
        else:
            control.throttle = 0.0
            control.brake = 0.1

        # Simple lane following
        waypoint = ego_vehicle.get_world().get_map().get_waypoint(transform.location)
        if waypoint:
            # Get the next waypoint
            next_waypoint = waypoint.next(5.0)[0]
            if next_waypoint:
                # Calculate angle to next waypoint
                next_location = next_waypoint.transform.location
                angle = math.atan2(next_location.y - transform.location.y,
                                next_location.x - transform.location.x)
                angle = math.degrees(angle) - transform.rotation.yaw
                angle = (angle + 180) % 360 - 180  # Normalize angle to [-180, 180]

                # Apply steering based on angle
                control.steer = max(-0.5, min(0.5, angle / 90.0))

        # Apply the control
        ego_vehicle.apply_control(control)

class KeyboardController:
    def __init__(self):
        self.controller = carla.VehicleControl()
        self.is_reverse = False
        
        pygame.init()
        pygame.font.init()
        pygame.display.set_caption("CARLA Keyboard Controller Mode")
        self.screen = pygame.display.set_mode((320, 240))
        self.clock = pygame.time.Clock()
        
    def update(self, keys):
        self.controller.throttle = 0.0
        self.controller.brake = 0.0
        self.controller.steer = 0.0

        if keys[pygame.K_UP] or keys[pygame.K_w]:
            self.controller.throttle = 1.0
        if keys[pygame.K_DOWN] or keys[pygame.K_s]:
            self.controller.brake = 1.0
        if keys[pygame.K_LEFT] or keys[pygame.K_a]:
            self.controller.steer = -0.5
        if keys[pygame.K_RIGHT] or keys[pygame.K_d]:
            self.controller.steer = 0.5
        if keys[pygame.K_SPACE]:
            self.controller.hand_brake = True
        else:
            self.controller.hand_brake = False
        if keys[pygame.K_r]:
            self.is_reverse = not self.is_reverse

        self.controller.reverse = self.is_reverse

    def run(self, ego_vehicle, image_queue):
        cv2.namedWindow('Camera', cv2.WINDOW_AUTOSIZE)
        self.ego_vehicle = ego_vehicle
        try:
            while True:
                self.ego_vehicle.get_world().tick()
                pygame.event.pump()
                keys = pygame.key.get_pressed()
                self.update(keys)
                self.ego_vehicle.apply_control(self.controller)

                # Draw key status visually on pygame display
                self.draw_keyboard_state(self.screen, keys)
                pygame.display.flip()

                try:
                    image_frame = image_queue.get(timeout=1.0)
                    cv2.imshow('Camera', image_frame[1])
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        return
                except queue.Empty:
                    print('No camera image received.')

                self.clock.tick(60)
        except Exception as e:
            logging.error(e)

        finally:
            cv2.destroyAllWindows()
            pygame.quit()
            print('Keyboard controller mode exited.')
            
    def draw_keyboard_state(self, screen, keys):
        # Colors
        WHITE = (255, 255, 255)
        GREEN = (0, 255, 0)
        GRAY = (180, 180, 180)
        BLACK = (0, 0, 0)

        font = pygame.font.SysFont(None, 30)

        screen.fill(BLACK)

        # Key positions
        key_map = {
            "UP": (160, 50),
            "LEFT": (100, 100),
            "DOWN": (160, 100),
            "RIGHT": (220, 100),
            "SPACE": (100, 170),
            "REVERSE": (220, 170)
        }

        for key, pos in key_map.items():
            if key == "UP":
                pressed = keys[pygame.K_UP] or keys[pygame.K_w]
            elif key == "DOWN":
                pressed = keys[pygame.K_DOWN] or keys[pygame.K_s]
            elif key == "LEFT":
                pressed = keys[pygame.K_LEFT] or keys[pygame.K_a]
            elif key == "RIGHT":
                pressed = keys[pygame.K_RIGHT] or keys[pygame.K_d]
            elif key == "SPACE":
                pressed = keys[pygame.K_SPACE]
            elif key == "REVERSE":
                pressed = keys[pygame.K_r]
            else:
                pressed = False

            color = GREEN if pressed else GRAY
            rect = pygame.Rect(pos[0], pos[1], 50, 40)
            pygame.draw.rect(screen, color, rect)
            pygame.draw.rect(screen, WHITE, rect, 2)
            label = font.render(key, True, WHITE)
            screen.blit(label, (pos[0] + 5, pos[1] + 10))
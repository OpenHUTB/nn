import glob
import os
import sys
# try:
#     carla_egg = sys.path.append(glob.glob('D:\CARLA\WindowsNoEditor\PythonAPI\carla\dist/carla-*%d.%d-%s.egg' % (
#         sys.version_info.major,
#         sys.version_info.minor,
#         'win-amd64' if os.name == 'nt' else 'linux-x86_64'
#     )))
    
#     if carla_egg:
#         sys.path.append(carla_egg[0])

# except IndexError:
#     pass

import carla
import time
import numpy as np
import cv2
import pygame
import logging
import colorama
from colorama import Fore, Back, Style, Cursor
from concurrent.futures import ThreadPoolExecutor

from utils import test_connection
from utils import EgoVehicleController
from Sensors import SensorManager, WINDOW_SIZE
from utils import weather
from utils import eagle_eye_map
from utils import CarlaEnvironment, DisplayManager, EgoVehicleController, EagleEyeMap
from utils import Logger, DataRecorder
from utils.utils import exit_game

def cleanup(world, sensermanagers):
    """Clean up all actors and reset world settings"""
    for sm in sensermanagers:
        try:
            sm.destroy()
        except RuntimeError:
            pass
    
    for controller in world.get_actors().filter('*controller*'):
        controller.stop()
    for walker in world.get_actors().filter('*pedestrian*'):
        walker.destroy()
    for vehicle in world.get_actors().filter('*vehicle*'):
        vehicle.destroy()

    setting = world.get_settings()
    setting.synchronous_mode = False
    setting.fixed_delta_seconds = None
    world.apply_settings(setting)
    print('Destroy all actors!')
    exit_game()
    
    
def update_display(world, displaymanager, weather_manager, ego_vehicle):
    """Update all display elements"""
    # Update vehicle lights
    # vehicle_light_state(world)
    
    # Update spectator view - use ego vehicle transform if no camera is available
    cameras = world.get_actors().filter('*camera*')
    if cameras:
        spectator_transform = cameras[0].get_transform()
    else:
        spectator_transform = ego_vehicle.get_transform()
    world.get_spectator().set_transform(spectator_transform)
    
    # Update displays
    displaymanager.render()

def main():
    """Main simulation loop"""
    test_connection.run()
    colorama.init()
    InfoLogger = Logger()
    
    # Initialize environment
    CarlaEnv = CarlaEnvironment(world_map='Town10HD', timeout=10.0)
    world, traffic_manager = CarlaEnv.setup_carla_environment()
    CarlaEnv.setup_spectator(world)
    ego_vehicle = CarlaEnv.spawn_actors(world, 70, 10, 'vehicle.tesla.cybertruck', 'Lebron James')
    
    # Setup ego vehicle control
    ego_control = EgoVehicleController()
    ego_control.setup_ego_vehicle(ego_vehicle)
    
    # Setup data recorder for autonomous driving dataset
    data_recorder = DataRecorder(
        output_dir="dataset",
        sampling_rate=10.0,  # 10 Hz sampling rate
        image_size=(400, 224),  # Target image size
        enable_recording=False  # Start disabled, can be toggled
    )
    
    # Setup display and sensors
    displaymanager = DisplayManager(
        grid_size=[2, 4], 
        window_size=[WINDOW_SIZE[0], WINDOW_SIZE[1]]
    )
    
    # sensor = [[x, y, z], [display grid], enabled]
    sensors_dict = {
        'DepthCamera': [[0, 0, 2.4], [0, 0], False],
        'RGBCamera': [[0, 0, 2.4], [0, 0], True],
        'RGBCamera': [[0, 0, 2.4], [0, 0], True],
        'RGBCamera_BEV': [[0, 0, 20.0], [0, 1], False],
        'RGBCamera_Lane': [[2.0, 0, 2.4], [1, 3], False],
        'RGBCamera_Lane_Edges': [[2.0, 0, 2.4], [0, 3], False],
        'SemanticSegmentationCamera': [[0, 0, 2.4], [0, 2], False],
        'InstanceSegmentationCamera': [[0, 0, 2.4], [1, 1], False],
        'LiDAR': [[0, 0, 2.4], [1, 0], False],
        'SemanticLiDAR': [[0, 0, 2.4], [1, 2], False]
    }
    
    sensermanagers = SensorManager.setup_sensors(world, ego_vehicle, displaymanager, sensors_dict)
    
    # Set data recorder for all sensor managers
    for sm in sensermanagers:
        sm.set_data_recorder(data_recorder)
    # Setup eagle eye map
    EagleEye = EagleEyeMap(world, ego_vehicle, zoom=1.0)
    executor = ThreadPoolExecutor(max_workers=2)
    # future = executor.submit(EagleEye.run)
    # displaymanager.add_surface('EagleEyeMap', EagleEye.map_surface, [0, 3])
    
    # Set fixed daytime weather
    daytime_weather = carla.WeatherParameters(
        sun_altitude_angle=70.0,
        sun_azimuth_angle=0.0,
        cloudiness=0.0,
        precipitation=0.0,
        precipitation_deposits=0.0,
        wind_intensity=0.0,
        fog_density=0.0
    )
    world.set_weather(daytime_weather)
    weather_manager = weather.Weather(daytime_weather)
    
    # Print data recording instructions
    print("\n" + "="*60)
    print("AUTONOMOUS DRIVING DATA RECORDING SYSTEM")
    print("="*60)
    print("Controls:")
    print("  R - Toggle data recording ON/OFF")
    print("  S - Show recording status")
    print("  ESC - Exit simulation")
    print("\nData being recorded:")
    print("  - RGB camera images (400x224)")
    print("  - Control signals (steer, throttle, brake)")
    print("  - Vehicle speed and transform")
    print("  - Timestamps and frame IDs")
    print(f"  - Sampling rate: {data_recorder.sampling_rate} Hz")
    print(f"  - Output directory: {data_recorder.output_dir}")
    print("="*60)
    
    # Main simulation loop
    loop_counter = 0
    running = True
    try:
        while running:
            # Handle pygame events
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    running = False
                elif event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_ESCAPE:
                        print('\nQuit Simulation!')
                        running = False
                    elif event.key == pygame.K_r:  # Toggle recording with 'R' key
                        data_recorder.toggle_recording()
                        status = "STARTED" if data_recorder.is_recording else "STOPPED"
                        print(f'\nData recording {status}!')
                    elif event.key == pygame.K_s:  # Show recording status with 'S' key
                        status = data_recorder.get_recording_status()
                        print(f'\nRecording Status: {status}')

            # Update ego vehicle control
            ego_control.update_ego_vehicle(ego_vehicle, ego_control.controller)
            
            # Update data recorder with current vehicle state
            data_recorder.update_vehicle_state(ego_vehicle)
            data_recorder.update_control_signals(
                ego_control.controller.steer,
                ego_control.controller.throttle,
                ego_control.controller.brake
            )
            
            # Record frame if conditions are met
            data_recorder.record_frame()

            # Update simulation state
            update_display(world, displaymanager, weather_manager, ego_vehicle)
            # InfoLogger.push_info(weather_manager, ego_vehicle, world, loop_counter)
            
            # Update world
            if world.get_settings().synchronous_mode and traffic_manager.synchronous_mode:
                world.tick()
                
            vehicle_location = ego_vehicle.get_transform().location
            EagleEyeMap.draw_vehicle(EagleEye, (vehicle_location.x, vehicle_location.y))
            
            loop_counter += 1
    except Exception as e:
        import traceback
        logging.error("An error occurred in the main loop:")
        logging.error(e)
        traceback.print_exc()
       
    # except Exception as e:
    #     logging.error(e)
        

    finally:
        EagleEye.stop()
        data_recorder.cleanup()  # Clean up data recorder
        executor.shutdown()
        cleanup(world, sensermanagers)

if __name__ == '__main__':
    main()
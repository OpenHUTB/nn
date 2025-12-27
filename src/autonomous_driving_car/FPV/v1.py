#!/usr/bin/env python
"""
CARLA Vehicle Spawn with Pygame Display & Plotting (sd_3/__main__.py)

This script connects to CARLA, spawns a vehicle, applies constant
forward throttle, and visualizes the simulation using:
1. PygameDisplay: Shows a rigidly attached camera view in a Pygame window.
2. Plotter: Plots the vehicle's X-coordinate and speed vs. time using Matplotlib.

It also sets the initial spectator camera position to a fixed viewpoint.

Dependencies:
- tools.pygame_display (Specifically the PygameDisplay class)
- tools.plotter_x (Specifically the Plotter class for X-coord/Speed plots)
"""

import carla
import numpy as np 
import time
import pygame # Pygame is implicitly needed by PygameDisplay
import math 

# Import helper classes from 'tools' directory using the specified format
from tools.pygame_display import PygameDisplay 
from tools.plotter_x import Plotter 

# Global variable to store the simulation start time for plotting
simulation_start_time = 0.0 

# --- Vehicle Data Functions ---
def get_location(vehicle):
   """Returns the current location of the vehicle."""
   return vehicle.get_location()

def get_speed_kmh(vehicle):
   """Returns the current speed of the vehicle in km/h."""
   velocity_vector = vehicle.get_velocity()
   # Calculate the magnitude of the velocity vector (speed in m/s)
   speed_meters_per_second = np.linalg.norm([velocity_vector.x, velocity_vector.y, velocity_vector.z]) 
   return 3.6 * speed_meters_per_second 

# --- Simulation Management Functions ---
def remove_previous_vehicle(world): 
    """Finds and removes all vehicles with the role 'my_car'."""
    print("Searching for previous 'my_car' vehicles...")
    actors = world.get_actors().filter('vehicle.*')
    count = 0
    for actor in actors:
        if actor.attributes.get('role_name') == 'my_car':
            print(f"  - Removing previous vehicle: {actor.type_id} (ID {actor.id})")
            if actor.destroy():
                 count += 1
            else:
                 print(f"  - Failed to remove vehicle {actor.id}")
    print(f"Removed {count} previous vehicles.")


# --- Spectator Setup ---
def set_initial_spectator_view(world):
    """Sets the spectator camera to a predefined fixed position and rotation."""
    spectator = world.get_spectator()
    # Define spectator transform using the desired fixed values (adjust as needed)
    spectator_location = carla.Location(x=63.12, y=29.88, z=5.61)
    spectator_rotation = carla.Rotation(pitch=-4.27, yaw=-170.21, roll=0.00)
    spectator_transform = carla.Transform(spectator_location, spectator_rotation)
    try:
        spectator.set_transform(spectator_transform)
        print(f"Initial spectator position set to: Loc=[{spectator_location.x:.2f}, {spectator_location.y:.2f}, {spectator_location.z:.2f}], Rot=[P:{spectator_rotation.pitch:.2f}, Y:{spectator_rotation.yaw:.2f}, R:{spectator_rotation.roll:.2f}]")
    except Exception as e:
        print(f"Error setting spectator transform: {e}")


def main():
    """Main execution function."""
    global simulation_start_time # Allow modification of the global variable
    
    client = None
    world = None
    vehicle = None 
    pygame_display = None 
    plotter = None 

    try:
        # Connect to CARLA
        client = carla.Client('localhost', 2000)
        client.set_timeout(10.0) 
        print("Connecting to CARLA server...")
        world = client.get_world()
        print(f"Connected to world: {world.get_map().name}")
        
        # Cleanup previous actors
        remove_previous_vehicle(world)
        
        # Get spawn point
        spawn_points = world.get_map().get_spawn_points()
        if not spawn_points:
            print("Error: No spawn points found on the map!")
            return
        spawn_point = spawn_points[0] # Use the first spawn point
        print(f"Selected spawn point: {spawn_point.location}")

        # Get vehicle blueprint
        vehicle_bp_library = world.get_blueprint_library()
        vehicle_bp = vehicle_bp_library.filter('vehicle.tesla.model3')[0]
        vehicle_bp.set_attribute('role_name', 'my_car') 
        
        # Spawn vehicle
        print("Attempting to spawn vehicle...")
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)
        if vehicle is None:
             print(f"Error: Failed to spawn vehicle at {spawn_point.location}")
             return 
        print(f"Vehicle {vehicle.type_id} (ID {vehicle.id}) spawned at {vehicle.get_location()}.")

        # Set spectator view
        set_initial_spectator_view(world) 

        # --- Initialize Plotter & Pygame ---
        print("Initializing Plotter...")
        plotter = Plotter() # Uses tools.plotter_x
        plotter.init_plot()
        print("Plotter initialized.")

        print("Initializing Pygame display...")
        pygame_display = PygameDisplay(world, vehicle) # Uses tools.pygame_display
        print("Pygame display initialized.")
        
        # Record the start time for the plot's time axis
        simulation_start_time = time.time() 

        # --- Main Simulation Loop ---
        print("Simulation running. Applying constant throttle. Press ESC or close Pygame window to stop.")
        while True:
            current_loop_time = time.time() # Get time at the start of the loop iteration

            # Handle Pygame events (QUIT, ESC)
            if pygame_display.parse_events():
                print("Quit requested via Pygame window.")
                break # Exit the loop if quit is requested

            # Wait for the next simulation tick
            world.wait_for_tick() 

            # --- Data Collection for Plotter ---
            current_location = get_location(vehicle) 
            current_speed_kmh = get_speed_kmh(vehicle)
            # Calculate time elapsed since simulation start
            current_sim_time_sec = current_loop_time - simulation_start_time 
            
            # --- Render the Pygame display ---
            pygame_display.render()
            
            # --- Update Plot ---
            if plotter is not None and plotter.is_initialized:
                 try:
                     # plotter_x expects time, x, current_speed, desired_speed
                     # We don't have a desired speed here, so pass 0 or current speed
                     plotter.update_plot(current_sim_time_sec, current_location.x, current_speed_kmh, 0.0) 
                 except Exception as plot_update_e:
                      # Handle potential errors if the plot window was closed
                      print(f"Error updating plot (likely closed): {plot_update_e}")
                      plotter.cleanup_plot() # Attempt cleanup
                      plotter = None # Stop trying to update
            
            # --- Vehicle Control ---
            # Apply constant forward throttle
            control = carla.VehicleControl(throttle=0.8, steer=0.0, brake=0.0) 
            vehicle.apply_control(control)

    except KeyboardInterrupt:
        print("\nScript interrupted by user (Ctrl+C).")
    except Exception as e:
        print(f"\nAn unexpected error occurred: {e}") 
        import traceback
        traceback.print_exc()
    
    finally:
        print("Starting resource cleanup...")
        
        # Destroy Pygame display first (handles its own camera cleanup)
        if pygame_display is not None:
            print("Destroying Pygame display...")
            pygame_display.destroy()
            print("Pygame display destroyed.")

        # Cleanup plotter
        if plotter is not None:
             # Check if it needs cleanup (might already be None if closed/error)
             if plotter.is_initialized:
                 print("Cleaning up plotter...")
                 plotter.cleanup_plot()
                 print("Plotter cleaned up.")
            
        # Destroy the main vehicle
        if vehicle is not None and vehicle.is_alive:
             print(f"Destroying vehicle: {vehicle.type_id} (ID {vehicle.id})")
             # vehicle.set_simulate_physics(False) # Optional: might help ensure clean removal
             if vehicle.destroy():
                 print("Vehicle destroyed successfully.")
             else:
                 print("Vehicle destroy() returned False.")
        else:
             print("Vehicle was None or not alive, no destruction needed.")
        
        print("Simulation finished.")

if __name__ == '__main__':
    main()

#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
CARLA Autonomous Vehicle Simulation - Unified Entry Point

Usage:
    python main.py                          # Quick test mode (30s, minimal setup)
    python main.py --config standard         # Full simulation with standard sensors
    python main.py --duration 60 --vehicles 50  # Custom parameters
    python main.py --weather fog             # Specific weather condition
    python main.py --help                    # Show all available options
"""

import argparse
import sys
import io
import logging
import time
import random
from datetime import datetime


# Fix Windows console encoding for Unicode/emoji support
if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')


def setup_logging(verbose=False):
    """Configure logging based on verbosity level"""
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def parse_arguments():
    """Parse and validate command line arguments"""
    parser = argparse.ArgumentParser(
        description='CARLA Autonomous Vehicle Simulation Framework',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  %(prog)s                              Quick test (30s, 20 vehicles)
  %(prog)s --config standard            Standard sensor configuration
  %(prog)s --duration 120 --vehicles 100  Custom duration and traffic
  %(prog)s --weather fog                Foggy weather conditions
  %(prog)s --verbose                    Enable debug logging
        """
    )

    # Mode selection
    mode_group = parser.add_argument_group('Running Mode')
    mode_group.add_argument(
        '--mode', '-m',
        choices=['quick', 'full'],
        default='quick',
        help='Running mode: quick (default) for fast testing, full for complete simulation'
    )
    mode_group.add_argument(
        '--config', '-c',
        choices=['minimal', 'standard', 'advanced'],
        default='minimal',
        help='Sensor configuration: minimal, standard (default), or advanced'
    )

    # Simulation parameters
    sim_group = parser.add_argument_group('Simulation Parameters')
    sim_group.add_argument(
        '--duration', '-d',
        type=int,
        default=30,
        help='Simulation duration in seconds (default: 30)'
    )
    sim_group.add_argument(
        '--vehicles', '-v',
        type=int,
        default=20,
        help='Number of traffic vehicles (default: 20)'
    )
    sim_group.add_argument(
        '--pedestrians', '-p',
        type=int,
        default=10,
        help='Number of pedestrians (default: 10)'
    )
    sim_group.add_argument(
        '--town', '-t',
        type=str,
        default=None,
        help='Town map to use (e.g., Town01, Town05). Default: random'
    )

    # Weather settings
    weather_group = parser.add_argument_group('Weather Settings')
    weather_group.add_argument(
        '--weather', '-w',
        choices=['clear', 'rain', 'heavy_rain', 'fog', 'storm'],
        default='rain',
        help='Weather condition: clear, rain (default), heavy_rain, fog, storm'
    )
    weather_group.add_argument(
        '--rain-intensity',
        type=float,
        default=None,
        help='Custom rain intensity 0-100 (overrides --weather)'
    )
    weather_group.add_argument(
        '--fog-density',
        type=float,
        default=None,
        help='Custom fog density 0-100 (overrides --weather)'
    )

    # Connection settings
    conn_group = parser.add_argument_group('Connection Settings')
    conn_group.add_argument(
        '--host',
        type=str,
        default='localhost',
        help='CARLA server host (default: localhost)'
    )
    conn_group.add_argument(
        '--port',
        type=int,
        default=2000,
        help='CARLA server port (default: 2000)'
    )
    conn_group.add_argument(
        '--timeout',
        type=float,
        default=10.0,
        help='Connection timeout in seconds (default: 10.0)'
    )

    # Output options
    output_group = parser.add_argument_group('Output Options')
    output_group.add_argument(
        '--output-dir', '-o',
        type=str,
        default='./output',
        help='Output directory for data export (default: ./output)'
    )
    output_group.add_argument(
        '--no-export',
        action='store_true',
        help='Disable data export (for testing only)'
    )
    output_group.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose/debug logging'
    )

    return parser.parse_args()


def get_weather_parameters(weather_type):
    """
    Get weather parameters based on preset name.

    Args:
        weather_type (str): Weather preset name

    Returns:
        dict: Weather parameters dictionary
    """
    weather_presets = {
        'clear': {
            'cloudiness': 10.0,
            'precipitation': 0.0,
            'precipitation_deposits': 0.0,
            'wind_intensity': 0.1,
            'fog_density': 0.0,
            'wetness': 0.0,
            'sun_altitude_angle': 75.0
        },
        'rain': {
            'cloudiness': 60.0,
            'precipitation': 40.0,
            'precipitation_deposits': 20.0,
            'wind_intensity': 0.3,
            'fog_density': 10.0,
            'wetness': 40.0,
            'sun_altitude_angle': 60.0
        },
        'heavy_rain': {
            'cloudiness': 100.0,
            'precipitation': 100.0,
            'precipitation_deposits': 100.0,
            'wind_intensity': 50.0,
            'fog_density': 20.0,
            'wetness': 100.0,
            'sun_altitude_angle': 45.0
        },
        'fog': {
            'cloudiness': 90.0,
            'precipitation': 10.0,
            'precipitation_deposits': 0.0,
            'wind_intensity': 0.1,
            'fog_density': 80.0,
            'wetness': 10.0,
            'sun_altitude_angle': 30.0
        },
        'storm': {
            'cloudiness': 100.0,
            'precipitation': 100.0,
            'precipitation_deposits': 100.0,
            'wind_intensity': 100.0,
            'fog_density': 50.0,
            'wetness': 100.0,
            'sun_altitude_angle': 20.0
        }
    }

    return weather_presets.get(weather_type, weather_presets['rain'])


def run_quick_test(args):
    """
    Run quick test mode - lightweight CARLA connection test.

    This mode is perfect for:
    - Verifying CARLA installation and connection
    - Fast prototyping and debugging
    - Testing basic functionality without full sensor setup

    Args:
        args: Parsed command line arguments
    """
    import carla

    print("\n" + "="*70)
    print("🚀 CARLA AV Simulation - Quick Test Mode")
    print("="*70)
    print(f"⏱  Duration: {args.duration}s | 🚗 Vehicles: {args.vehicles} | 🌧 Weather: {args.weather}")
    print("="*70 + "\n")

    try:
        # Connect to CARLA server
        logging.info(f"Connecting to CARLA server at {args.host}:{args.port}...")
        client = carla.Client(args.host, args.port)
        client.set_timeout(args.timeout)

        world = client.get_world()
        logging.info("✅ Successfully connected to CARLA server!")

        # Get map info
        map_name = world.get_map().name
        logging.info(f"📍 Current map: {map_name}")

        # Set up weather
        weather_params = get_weather_parameters(args.weather)

        if args.rain_intensity is not None:
            weather_params['precipitation'] = args.rain_intensity
            logging.info(f"Using custom rain intensity: {args.rain_intensity}")

        if args.fog_density is not None:
            weather_params['fog_density'] = args.fog_density
            logging.info(f"Using custom fog density: {args.fog_density}")

        weather = carla.WeatherParameters(**weather_params)
        world.set_weather(weather)
        logging.info(f"🌧️  Weather set to: {args.weather}")

        # Spawn ego vehicle
        blueprint_library = world.get_blueprint_library()

        vehicle_bp = blueprint_library.find('vehicle.yamaha.yzf')
        spawn_points = world.get_map().get_spawn_points()

        if not spawn_points:
            logging.error("❌ No spawn points available")
            return False

        spawn_point = random.choice(spawn_points)
        vehicle = world.try_spawn_actor(vehicle_bp, spawn_point)

        if not vehicle:
            logging.error("❌ Failed to spawn ego vehicle")
            return False

        logging.info(f"🏍️  Ego vehicle spawned: {vehicle_bp.id}")
        vehicle.set_autopilot(True)
        logging.info("✅ Autopilot enabled!")

        # Setup traffic manager
        tm = client.get_trafficmanager(8000)
        tm.set_synchronous_mode(False)

        # Spawn traffic vehicles
        vehicle_bps = blueprint_library.filter('vehicle.*')
        random.seed(42)

        spawned_vehicles = []
        for i in range(args.vehicles):
            try:
                bp = random.choice(vehicle_bps)
                if bp.has_attribute('color'):
                    color = random.choice(bp.get_attribute('color').recommended_values)
                    bp.set_attribute('color', color)

                spawn_pt = random.choice(spawn_points)
                v = world.try_spawn_actor(bp, spawn_pt)
                if v:
                    v.set_autopilot(True, tm.get_port())
                    spawned_vehicles.append(v)
            except Exception as e:
                logging.warning(f"Failed to spawn vehicle {i+1}: {str(e)}")
                continue

        logging.info(f"🚗 Spawned {len(spawned_vehicles)}/{args.vehicles} traffic vehicles")

        # Run simulation loop
        print("\n" + "-"*70)
        print("🎬 Starting simulation...")
        print(f"   Duration: {args.duration} seconds")
        print(f"   Watch the CARLA window for the simulation!")
        print("-"*70 + "\n")

        start_time = time.time()
        frame_count = 0
        last_log_time = start_time

        try:
            while time.time() - start_time < args.duration:
                world.tick()
                frame_count += 1

                current_time = time.time()
                elapsed = current_time - start_time

                # Log progress every second
                if current_time - last_log_time >= 1.0:
                    progress = (elapsed / args.duration) * 100
                    logging.info(f"⏱️  Progress: {elapsed:.1f}s / {args.duration}s ({progress:.1f}%) | Frames: {frame_count}")
                    last_log_time = current_time

        except KeyboardInterrupt:
            logging.info("\n⚠️  Simulation interrupted by user")

        # Final statistics
        total_time = time.time() - start_time
        avg_fps = frame_count / total_time if total_time > 0 else 0

        print("\n" + "="*70)
        print("✅ Simulation completed successfully!")
        print("="*70)
        print(f"📊 Statistics:")
        print(f"   ⏱  Total time: {total_time:.2f} seconds")
        print(f"   📈 Total frames: {frame_count}")
        print(f"   🎯 Average FPS: {avg_fps:.2f}")
        print(f"   🚗 Vehicles spawned: {len(spawned_vehicles)}")
        print("="*70 + "\n")

        # Cleanup
        logging.info("Cleaning up actors...")
        for v in spawned_vehicles:
            if v.is_alive:
                v.destroy()
        if vehicle.is_alive:
            vehicle.destroy()
        logging.info("✅ Cleanup completed!")

        return True

    except Exception as e:
        logging.error(f"❌ Error during quick test: {str(e)}")
        import traceback
        traceback.print_exc()
        return False


def run_full_simulation(args):
    """
    Run full simulation mode with complete sensor suite.

    This mode provides:
    - Multi-sensor data collection (Camera, LiDAR, Radar, etc.)
    - Real-time Pygame visualization
    - Automatic data export
    - Advanced traffic and pedestrian simulation

    Args:
        args: Parsed command line arguments
    """
    try:
        from carla_av_simulation import AVSimulation
    except ImportError as e:
        logging.error(f"❌ Failed to import AVSimulation module: {str(e)}")
        logging.error("   Make sure carla_av_simulation.py is in the same directory")
        return False

    print("\n" + "="*70)
    print("🎯 CARLA AV Simulation - Full Mode")
    print("="*70)
    print(f"⚙️  Configuration: {args.config}")
    print(f"⏱  Duration: {args.duration}s | 🚗 Vehicles: {args.vehicles} | 🚶 Pedestrians: {args.pedestrians}")
    print(f"🌧 Weather: {args.weather} | 📁 Output: {args.output_dir}")
    print("="*70 + "\n")

    try:
        # Initialize simulation
        simulation = AVSimulation()

        # Run simulation with specified parameters
        # Note: Current AVSimulation.run_simulation() accepts config_name and duration
        # We can extend it later to accept more parameters
        logging.info(f"Starting full simulation with {args.config} configuration...")

        simulation.run_simulation(
            config_name=args.config,
            duration_seconds=args.duration
        )

        logging.info("✅ Full simulation completed successfully!")
        return True

    except KeyboardInterrupt:
        logging.info("\n⚠️  Full simulation interrupted by user")
        return True
    except Exception as e:
        logging.error(f"❌ Error during full simulation: {str(e)}")
        import traceback
        traceback.print_exc()
        return False
    finally:
        # Cleanup will be handled by AVSimulation's finally block
        pass


def main():
    """Main entry point for CARLA AV Simulation"""
    start_time = time.time()

    # Parse command line arguments
    args = parse_arguments()

    # Setup logging
    setup_logging(args.verbose)

    # Print banner
    print("\n" + "█"*70)
    print("█" + " "*68 + "█")
    print("█" + "  🚗 CARLA Autonomous Vehicle Simulation Framework  v2.0  ".center(68) + "█")
    print("█" + " "*68 + "█")
    print("█"*70 + "\n")

    logging.info(f"Mode: {args.mode.upper()} | Config: {args.config} | Duration: {args.duration}s")

    # Run appropriate mode
    success = False
    if args.mode == 'quick':
        success = run_quick_test(args)
    elif args.mode == 'full':
        success = run_full_simulation(args)

    # Print execution summary
    total_execution_time = time.time() - start_time

    print("\n" + "─"*70)
    if success:
        print("🎉 Execution completed successfully!")
    else:
        print("⚠️  Execution finished with errors")
    print(f"⏱  Total execution time: {total_execution_time:.2f} seconds")
    print("─"*70 + "\n")

    # Return exit code
    sys.exit(0 if success else 1)


if __name__ == '__main__':
    main()

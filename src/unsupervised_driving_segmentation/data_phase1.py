from runners import ScenarioRunner
import argparse
import ray
import logging
import carla
import os

def clean_up_actors(host, port):
    client = carla.Client(host, port)
    client.set_timeout(10.0)
    world = client.get_world()
    actors = world.get_actors()
    for actor in actors:
        if actor.is_alive:
            try:
                actor.destroy()
                logging.info(f"Destroyed actor {actor.id}")
            except:
                logging.warning(f"Failed to destroy actor {actor.id}")

def set_weather_conditions(world):
    weather = carla.WeatherParameters(
        cloudiness=20.0,
        precipitation=0.0,
        sun_altitude_angle=45.0,  # Adjust sun altitude to control lighting
        fog_density=0.0,
        fog_distance=100.0,
        fog_falloff=0.0,
        wetness=0.0
    )
    world.set_weather(weather)
    # world.set_time_of_day(hour=10, minute=0) 


def main(args):
    towns = {i: f'Town{i+1:02d}' for i in range(7)}
    towns.update({7: 'Town10HD'})
    town = towns[7]

    scenario = 'assets/all_towns_traffic_scenarios.json'
    route = 'assets/testing_10HD/routes_10hd.xml'

    if not os.path.exists(scenario):
        logging.error(f"Scenario file/path {scenario_file} does not exist.")
        return

    # Check if route file exists
    if not os.path.exists(route):
        logging.error(f"Route file/path {route_file} does not exist.")
        return

    args.agent = 'autoagents/collector_agents/collector'
    args.agent_config = 'autoagents/collector_agents/config_data_collection.yaml'

    client = carla.Client(args.host, args.port)
    client.set_timeout(10.0)
    world = client.get_world()
    set_weather_conditions(world)

    jobs = []
    for i in range(args.num_runners):
        scenario_class = args.scenario
        print(f"Running scenario {scenario_class} on town {town}")
        port = (i+1) * args.port 
        tm_port = port + 2

        checkpoint = f'results/{i:02d}_{args.checkpoint}'
        runner = ScenarioRunner.remote(args, scenario_class, scenario, route, checkpoint=checkpoint, town=town, port=port, tm_port=tm_port)
        jobs.append(runner.run.remote())

    ray.wait(jobs, num_returns=args.num_runners)
    # clean_up_actors(args.host, args.port)


if __name__ == '__main__':
    ray.init(logging_level=40, local_mode=False, log_to_driver=True)

    parser = argparse.ArgumentParser()

    parser.add_argument('--num-runners', type=int, default=1)
    parser.add_argument('--scenario', choices=['train_scenario', 'nocrash_train_scenario'], default='train_scenario')
    parser.add_argument('--host', default='localhost', help='IP of the host server (default: localhost)')
    parser.add_argument('--port', type=int, default=2000)
    parser.add_argument('--trafficManagerSeed', type=str, default='0', help='Seed used by the TrafficManager (default: 0)')
    parser.add_argument('--timeout', type=int, default=600, help='Set the CARLA client timeout value in seconds')
    parser.add_argument('--start_port', type=int, default=2000, help='Starting port number for CARLA servers.')
    parser.add_argument('--repetitions', type=int, default=1, help='Number of repetitions per route.')
    parser.add_argument("--track", type=str, default='MAP', help="Participation track: SENSORS, MAP")
    parser.add_argument('--resume', type=bool, default=False, help='Resume execution from last checkpoint?')
    parser.add_argument("--checkpoint", type=str, default='simulation_results.json', help="Path to checkpoint used for saving statistics and resuming")

    args = parser.parse_args()
    # clean_up_actors(args.host, args.port) 

    try:
        main(args)
    except Exception as e:
        print(f"Error running the main function: {str(e)}")
        ray.shutdown()
    finally:
        ray.shutdown() 
        clean_up_actors(args.host, args.port)

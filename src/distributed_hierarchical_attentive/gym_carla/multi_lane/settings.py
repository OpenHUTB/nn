"""This file defines all high level parameters of carla gym environment"""
import argparse
#
# the following road id sets define the chosen route
ROADS = set()
DISTURB_ROADS = set()
FORWARD_ROADS = set()
BACKWARD_ROADS = set()
STRAIGHT = {12, 35, 36}
CURVE = {37, 38, 34}
JUNCTION = {2344, 2035}
DOUBLE_DIRECTION = {2358, 2363, 2039, 2052}
FORWARD_DOUBLE_DIRECTION = {2358, 2039}
BACKWARD_DOUBLE_DIRECTION = {2363, 2052}
# JUNCTION_LANE={33,85,141}
# JUNCTION_LANE_MINUS={102,109,150,163,46,67,128}
ROADS.update(STRAIGHT)
ROADS.update(CURVE)
ROADS.update(JUNCTION)
DISTURB_ROADS.update(DOUBLE_DIRECTION)
FORWARD_ROADS.update(FORWARD_DOUBLE_DIRECTION)
BACKWARD_ROADS.update(BACKWARD_DOUBLE_DIRECTION)
# the flowing arguments set the simulation parameters
ARGS = argparse.ArgumentParser(
    description='CARLA_gym Client')
ARGS.add_argument(
    '-v', '--verbose',
    action='store_true',
    dest='debug',
    default=False,
    help='Print debug information')
ARGS.add_argument(
    '-t',
    action='store_true',
    dest='train',
    default=True,
    help='Training Reinforcement agent')
ARGS.add_argument(
    '--host',
    metavar='H',
    default='127.0.0.1',
    help='IP of the host server (default: 127.0.0.1)')
ARGS.add_argument(
    '-p', '--port',
    metavar='P',
    default=2000,
    type=int,
    help='TCP port to listen to (default: 2000)')
ARGS.add_argument(
    '--res',
    metavar='WIDTHxHEIGHT',
    default='1280x720',
    help='Window resolution (default: 1280x720)')
ARGS.add_argument(
    '--sync',
    action='store_true',
    default=True,
    help='Synchronous mode execution')
ARGS.add_argument(
    '--fps', metavar='FPS',
    default=10, type=int,
    help="The fps of server running speed")
ARGS.add_argument(
    '--filter',
    metavar='PATTERN',
    default='vehicle.tesla.model3',
    help='Actor filter (default: "vehicle.*")')
ARGS.add_argument(
    '-l', '--loop',
    action='store_true',
    dest='loop',
    default='True',
    help='Sets a new random destination upon reaching the previous one (default: False)')
ARGS.add_argument(
    "-a", "--agent", type=str,
    choices=["Behavior", "Basic"],
    help="select which agent to run",
    default="Behavior")
ARGS.add_argument(
    '-b', '--behavior', type=str,
    choices=["cautious", "normal", "aggressive"],
    help='Choose one of the possible agent behaviors (default: normal) ',
    default='normal')
ARGS.add_argument(
    '-s', '--seed',
    help='Set seed for repeating executions (default: None)',
    default=None,
    type=int)
ARGS.add_argument(
    '-m', '--map', type=str,
    choices=['Town05', 'Town05_Opt'],
    help='Choose one of the possible world maps',
    default='Town05_Opt')
ARGS.add_argument(
    '-n', '--num_of_vehicles', type=list,
    help='Total vehicles number which run in simulation',
    default=[15*3])
ARGS.add_argument(
    '-sa', '--sampling_resolution', type=float,
    help='Distance between generated two waypoints',
    default=1.0)
# ARGS.add_argument(
#     '-we', '--weather', type=float,
#     help='weather setting',
#     default=sunlight)
ARGS.add_argument(
    '--tm-port',
    metavar='P',
    default=8000,
    type=int,
    help='Port to communicate with traffic manager (default: 8000)')
ARGS.add_argument(
    '--hybrid',
    action='store_true',
    default=True,
    help='Activate hybrid mode for Traffic Manager')
ARGS.add_argument(
    '--auto_lane_change',
    action='store_true',
    default=True,
    help='set lane change behaviors for Traffic Manager')
ARGS.add_argument(
    '--no_rendering',
    action='store_true',
    default=False,
    help='Activate no rendering mode')
ARGS.add_argument(
    '--stride', type=int,
    default=10,
    help='The number of waypoints observed by the autonomous vehicle')
ARGS.add_argument(
    '--traffic_light_th', type=float,
    default=50,
    help='the detection range of traffic light')
ARGS.add_argument(
    '--vehicle_th', type=float,
    default=70,
    help='the detection range of conventional vehicles')
ARGS.add_argument(
    '--TTC_th', type=float,
    default=4,
    help='TTC threshold')
ARGS.add_argument(
    '--acceleration threshold', type=float,
    default=3,
    help='acceleration threshold for ego vehicle')
ARGS.add_argument(
    '--speed threshold', type=float,
    default=0.1,
    help='speed threshold for ego vehicle')
# ARGS.add_argument(
#     '--traffic_light_th', type=float,
#     default=50,
#     help='the detection range of traffic light')
# ARGS.add_argument(
#     '--vehicle_th', type=float,
#     default=70,
#     help='the detection range of conventional vehicles')
ARGS.add_argument(
    '--penalty', type=float,
    default=40,
    help='reward penalty for simulation terminated early on account of collision and road invasion')
ARGS.add_argument(
    '--lane_penalty', type=float,
    default=20,
    help='reward penalty for simulation terminated early on account of lane invasion')
ARGS.add_argument(
    '--lane_change_reward', type=float,
    default=25,
    help='reward for lane change according to the distance to the preceding vehicle')
# ARGS.add_argument(
#     '--acceleration threshold', type=float,
#     default=3,
#     help='acceleration threshold for ego vehicle')
# ARGS.add_argument(
#     '--speed threshold', type=float,
#     default=0.1,
#     help='speed threshold for ego vehicle')
ARGS.add_argument(
    '--speed_limit', type=float,
    default=90.0,
    help='Speed limit for ego vehicle, km/h')
ARGS.add_argument(
    '--speed_threshold', type=float,
    default=20.0,
    help='Speed threshold for ego vehicle, start phase for ego vehicle, km/h')
ARGS.add_argument(
    '--speed_min', type=float,
    default=3.6,
    help='When ego vehicle speed reaches down to this threshold, we should let basic agent take control \
        and the action of basic need to add into the replay buffer, km/h')
ARGS.add_argument(
    '--steer_bound', type=float,
    default=1.0,
    help='Steer bound for ego vehicle controller')
ARGS.add_argument(
    '--throttle_bound', type=float,
    default=1.0,
    help='Throttle bound for ego vehicle controller')
ARGS.add_argument(
    '--brake_bound', type=float,
    default=1.0,
    help='Brake bound for ego vehicle controller')
ARGS.add_argument(
    '--pre_train_steps', type=int,
    default=320000,
    help='During pre-train steps, agent is only under PID control.')
ARGS.add_argument(
    '--switch_threshold', type=int,
    default=30,
    help='Let the RL controller and PID controller alternatively take control every 20 episodes')
ARGS.add_argument(
    '--vehicle_proximity', type=float,
    default=50.0,
    help='Distance for searching vehicles in front of ego vehicle, unit -- meters')
ARGS.add_argument(
    '--traffic_light_proximity', type=float,
    default=50.0,
    help='Distance for searching traffic light in front of ego vehicle, unit -- meters,'
    'attention: this value is tricky')
ARGS.add_argument(
    '--min_distance',type=float,
    default=5.0,
    help='Min distance between two vehicles, unit -- meters')
ARGS.add_argument(
    '--guide_change', type=bool,
    default=False,
    help='guide the vehicle to change via lane_center')
ARGS.add_argument(
    '--ignore_traffic_light', type=bool,
    default=False,
    help='Set the vehicles in simulation to ignore traffic lights or not')
ARGS.add_argument(
    '--ego_num', type=int,
    default=1,
    help='The number of RL controlled ego vehicle')
ARGS.add_argument(
    '--modify_change_steer', type=bool,
    default=True,
    help='Useful for pdqn training, make sure the steer >0 when current action is ACTION.CHANGE_LANE_LEFT, \
         and the steer < 0 when current action is Action.CHANGE_LANE_LEFT')
ARGS.add_argument(
    '--pygame', type=bool,
    default=False,
    help='Render another pygame window for ego vehicle and the window style looks like automatic_control.py')
ARGS.add_argument(
    '--alg', type=str,
    default='PDQN',
    help='The RL algorithm currently in use')
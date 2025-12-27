from __future__ import annotations

import argparse
import logging
import numpy as np
import os
import warnings

from itertools import count
from os.path import join
from typing import Iterable

from agent import Agent
from recorder import Recorder
from world import World

def log_parameters(id:str, logger:logging.Logger, parameters: Iterable[float],
    delta_parameters: Iterable[float], error:float, steps:int) -> None:
    """Log one set of Parameters

    Args:
        id (str): An unique identifier used to identify the run.
        logger (logging.Logger): Logger that writes logs.
        parameters (Iterable[float]): Parameters of the controller.
        delta_parameters (Iterable[float]): Delta parameters for adaption of the
        parameters.
        error (float): Summed error received by the environment.
        steps (int): Number of steps taken within the run.
    """
    logger.info((f'id: {id} | parameters: {parameters} | delta_parameters: '
        f'{delta_parameters} | error: {error:.5f} | steps: {steps:5d}'))

def run(id:str, path:str, world:World, agent:Agent, steps:int = 100,
    save:bool=False) -> tuple[float, int]:
    """One Epsisode inside the World

    Performs one run inside the world until the maximum steps is reached or a
    collision was detected.

    Args:
        id (str): An unique identifier used to identify the run.
        path (str): Path where all files for a run will be stored.
        world (World): Environment for the simulation.
        agent (Agent): Agent to decide on actions.
        steps (int, optional): Number of maximum amount of steps to take until
            termination. Defaults to 100.
        save (bool, optional): Whether to create a video of the run. Defaults to
            False.

    Returns:
        tuple[float, int]:
            [0]: Sum of errors as returned by the world.
            [1]: Number of steps taken.
    """
    error, detection_surface_area, _, _ = world.reset()

    if save:
        recorder = Recorder(folder=path)
        recorder.init_new_video(id=id)

    for i in range(steps):
        steer, throttle = agent.get_actions(detection_surface_area, error)
        error, detection_surface_area, img, collision_detected = world.step(
            steer=steer, throttle=throttle)
        #agent.show_error()

        if save:
            recorder.add_image(img=img)

        if collision_detected:
            agent.errors = agent.errors + [max(agent.errors)] * (steps-i)
            agent.show_error()
            break

    agent.save_error_fig(path, id)
    if save:
        recorder.close_recording()

    return sum(agent.errors), i

def check_supported_controller(name:str) -> str:
    """Check supported controller

    Checks if the provided controller string is allowed for the adaption mode.

    Args:
        name (str): Identifies the controller to be used for the run.

    Returns:
        str: Identifies the controller to be used for the run. This may be
            different from the input name.
    """
    if name.lower() != 'pid':
        warnings.warn(('Twiddle does not support other controllers than '
            '\'pid\'. Setting controller to \'pid\''))
        return 'pid'
    return name

def get_logger(id:str, path:str, debug:bool=False) -> logging.Logger:
    """Retrieve logger

    The returned logger 'pacman_rl' writes to a file inside the 'logs' folder
    and streams the logs to the console.

    Args:
        id (str): An unique identifier used to identify the log file.
        path (str): Path where all files for a run will be stored.
        debug (bool, optional): Whether to use the debug mode for logging. If
            False, only infos will be logged. Defaults to False.

    Returns:
        logging.Logger: Logger that streams and writes logs.
    """
    if debug:
        level = logging.DEBUG
    else:
        level = logging.INFO

    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(message)s')
    logger = logging.getLogger('active_lane_keeping_assistant')
    logger.setLevel(level=level)
    fh = logging.FileHandler(join(path, f'alka_{id}.log'), mode='a')
    sh = logging.StreamHandler()
    sh.setFormatter(formatter)
    fh.setFormatter(formatter)
    logger.addHandler(fh)
    logger.addHandler(sh)

    return logger

def twiddle(id:str, path:str, tolerance:float=0.2, controller:str='pid',
    steps:int=100, debug:bool=False) -> None:
    """Twiddle-Algorithm

    Perform the twiddle algorithm to receive near optimal parameters for the
    PID-controller.

    Args:
        id (str): An unique identifier used to identify the run.
        path (str): Path where all files for a run will be stored.
        tolerance (float, optional): Value that must be undershot for the
            algorithm to be terminated. Defaults to 0.2.
        controller (str, optional): Identifies the controller to be used for the
            run. Defaults to 'pid'.
        steps (int, optional): Number of maximum amount of steps to take until
            termination. Defaults to 100.
        debug (bool, optional): Whether to use the debug mode for logging.
            Defaults to False.
    """
    logger = get_logger(id=id, path=path, debug=debug)
    controller = check_supported_controller(name=controller)

    logger.info(f'twiddle: {id}')
    logger.info((f'tolerance: {tolerance} | controller: {controller} | steps: '
        f'{steps}'))

    parameters = np.zeros(3)
    delta_parameters = np.ones(3)
    
    world = World()
    agent = Agent(controller=controller, tau_p=parameters[0],
        tau_i=parameters[1], tau_d=parameters[2])

    best_err, steps_taken = run(id, path, world, agent, steps, save=False)
    best_parameters = parameters
    best_id = id

    log_parameters(id, logger, parameters, delta_parameters, best_err,
        steps_taken)

    for t in count():
        if (sum(delta_parameters) < tolerance):
            break
        for i in range(len(parameters)):
            new_id = f'{id}_{t}_{i}_a'
            world.id = new_id
            parameters[i] += delta_parameters[i]

            agent = Agent(controller=controller, tau_p=parameters[0],
                tau_i=parameters[1], tau_d=parameters[2])
            error, steps_taken = run(new_id, path, world, agent, steps)
            log_parameters(new_id, logger, parameters, delta_parameters, error,
                steps_taken)

            if abs(error) < abs(best_err):
                logger.debug(f'Follow same direction for parameter {i}.')
                best_err = error
                best_parameters = parameters
                best_id = new_id
                delta_parameters[i] *= 1.1
            else:
                logger.debug(f'Trying opposite direction for parameter {i}.')
                new_id = f'{id}_{t}_{i}_a'
                world.id = new_id
                parameters[i] -= 2*delta_parameters[i]

                agent = Agent(controller=controller,
                    tau_p=parameters[0], tau_i=parameters[1],
                    tau_d=parameters[2])
                error, steps_taken = run(new_id, path, world, agent, steps)
                log_parameters(new_id, logger, parameters, delta_parameters,
                    error, steps_taken)

                if abs(error) < abs(best_err):
                    logger.debug((f'Follow opposite direction for parameter '
                        f'{i}.'))
                    best_err = error
                    best_parameters = parameters
                    best_id = new_id
                    delta_parameters[i] *= 1.1
                else:
                    logger.debug((f'Neither direction worked for {i}. '
                        f'Decreasing step size.'))
                    parameters[i] += delta_parameters[i]
                    delta_parameters[i] *= 0.9
    
    logger.info((f'best run - id: {best_id} | parameters: {best_parameters} | '
        f'error: {best_err:.5f}'))

def no_adapt(id:str, path:str, steps:int=100, controller:str='simple') -> None:
    """Run without adaption

    Run the environment without the adaption of the parameters from the
    controller.

    Args:
        id (str): Unique identifier for the run.
        path (str): Path where all files for a run will be stored.
        steps (int, optional): Number of steps to take within the environment.
            Defaults to 100.
        controller (str, optional): Identifies the controller to be used for the
            run. Defaults to 'simple'.
    """
    world = None
    try:
        world = World(time_difference=0.05)
        agent = Agent(controller=controller)
        run(id=id, path=path, world=world, agent=agent, steps=steps, save=False)
    finally:
        if world is not None:
            world.close()


def make_path(folder:str, id:str) -> str:
    """Create folder

    Checks if a folder for the id already exists and creates the folder if
    necessary.

    Args:
        folder (str): Name of the folder where the folder for the run will be
            located.
        id (str): Unique identifier for the run. This will be used to identify
            the specific folder for this run.

    Returns:
        str: Path including the id of the run.
    """
    path = join(os.getcwd(), folder, id)

    if not os.path.exists(path):
        os.makedirs(path)
    else:
        warnings.warn('Path already exists. Files may be overwritten.')
    
    return path


def parse_args() -> argparse.Namespace:
    """Parses command line arguments"""
    parser = argparse.ArgumentParser(prog='Active Lane Keeping System',
                                     description='')
    parser.add_argument('-id', '--identifier', type=str, required=True,
                        help='Unique identifier used to identify the run.', dest='id')
    parser.add_argument('-c', '--controller', default='pid', type=str,
                        choices=['simple', 'p', 'pd', 'pid'], dest='controller',
                        help='The method used to control the car.')
    parser.add_argument('-s', '--steps', default=100, type=int, dest='steps',
                        help='The number of steps that the agent controls the car.')
    parser.add_argument('-a', '--adapt', action='store_true', dest='adapt',
                        help='Whether to adapt the parameters.')
    parser.add_argument('-t', '--tolerance', default=0.2, type=float,
                        help='Tolerance range for the adaption.', dest='tolerance')
    parser.add_argument('-d', '--debug', action='store_true', dest='debug',
                        help='Whether to use debug mode for logging.')

    return parser.parse_args()
if __name__ == '__main__':

    args = parse_args()

    path = make_path(folder='assets', id=args.id)

    if args.adapt:
        twiddle(id=args.id, path=path, tolerance=args.tolerance,
            controller=args.controller, steps=args.steps, debug=args.debug)        
    else:
        no_adapt(id=args.id, path=path, steps=args.steps,
            controller=args.controller)
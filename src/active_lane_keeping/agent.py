from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np
import os

class Agent():
    """Agent that outputs the desired behaviour given 
    """

    def __init__(self, tau_p:float = 0, tau_d:float = 0, tau_i:float = 0,
        surface_lower_threshold:float = 20e6, throttle:float = 0.3,
        suface_upper_threshold=30e6, controller:str = 'simple') -> None:
        """Constructor

        Args:
            tau_p (float, optional): Parameter for the proportional part of the
                PID-Controller. Defaults to 0.
            tau_d (float, optional): Parameter for the derivative part of the
                PID-Controller. Defaults to 0.
            tau_i (float, optional): Parameter for the integral part of the
                PID-Controller. Defaults to 0.
            surface_lower_threshold (float, optional): Minimum amount of surface
                that has to be detected in order to calculate new output. This
                helps to find suitable lanes. Defaults to 20e6.
            throttle (float, optional): Throttle to return at each time step.
                Defaults to 0.3.
            suface_upper_threshold (_type_, optional): Maximum amount of surface
                that has to be detected in order to calculate new output. This
                helps to find suitable lanes. Defaults to 30e6.
            controller (str, optional): Identifies the controller to be used.
                This can be one of the following:
                    - 'simple': hard coded controller that does not use any of
                        the tau parameters
                    - 'p': controller that only uses the proportional part
                    - 'pd': controller that only uses the proportional and
                        derivative part
                    - 'pid': pid-controller
                Defaults to 'simple'.
        """
        self.tau_p = tau_p
        self.tau_d = tau_d
        self.tau_i = tau_i
        self.surface_lower_threshold = surface_lower_threshold
        self.surface_upper_threshold = suface_upper_threshold
        self.prev_error = None
        self.throttle = throttle
        self.func = None
        self.errors = []
        self.controller_name = controller
        self._select_controller_method(name=self.controller_name)

    def _select_controller_method(self, name:str) -> None:
        """Helper method to select the controller

        Args:
            name (str): Name of the controller.

        Raises:
            Exception: If the given name does not map to a controller method.
        """
        name = name.lower()
        if name == 'simple':
            self.func = Agent._simple_controller
        elif name == 'p':
            self.func = self._p_controller
        elif name == 'pd':
            self.func = self._pd_controller
        elif name == 'pid':
            self.func = self._pid_controller
        else:
            raise Exception(f'Controller name \'{name}\' is not applicable.')

    def check_surface_area(self, detection_surface_area:float) -> bool:
        """Checks Surface Area

        Args:
            detection_surface_area (float): _description_

        Returns:
            bool: Whether the given surface area is not inbetween the selected
                interval.
        """
        lower = (detection_surface_area < self.surface_lower_threshold)
        upper = (detection_surface_area > self.surface_upper_threshold)
        return lower or upper

    def show_error(self) -> None:
        """Displays Error

        Displays the difference to the center of all past steps.
        """
        plt.figure(1)
        plt.clf()
        x = range(len(self.errors))
        y = self.errors
        plt.plot(x, y, 'g', label=self.controller_name)
        plt.plot(x, np.zeros(len(self.errors)), 'r', label='baseline')
        plt.xlabel('Step')
        plt.ylabel('Difference to Center')
        plt.legend()
        plt.pause(1e-10)

    def save_error_fig(self, path:str, id:str) -> None:
        """Save the Figure of the Error Plot

        Args:
            path (str): Folder to place the file in.
            id (str): Unique identifier to prevent overwriting files.
        """
        plt.figure(1)
        file_name = os.path.join(path, f'{id}_error.jpg')
        plt.savefig(file_name)

    def get_actions(self, detection_surface_area:float,
        error:float) -> tuple[float, float]:
        """Retrieve action based on Observation

        Args:
            error (float): Difference to the center of the detected lane.
            detection_surface_area (float): Detected surface area.

        Returns:
            tuple[float, float]:
                [0]: Steering angle to use.
                [1]: Throttle to apply.
        """
        if (self.check_surface_area(detection_surface_area)):
            steer = 0
            if len(self.errors) > 0:
                self.errors.append(self.errors[-1])
            else:
                self.errors.append(0)
        else:
            self.errors.append(error)
            steer = self.func(error=error)
        return steer, self.throttle

    @staticmethod
    def _simple_controller(error:float) -> float:
        """Hard Coded Controller

        Args:
            error (float): Difference to the center of the detected lane.

        Returns:
            float: Steering angle to use.
        """
        if (abs(error) < 0.1):
            steer = 0
        elif error > 0:
            steer = -0.75
        else:
            steer = 0.75
        return steer

    def _p_controller(self, error:float) -> float:
        """Proportional Controller

        Args:
            error (float): Difference to the center of the detected lane.

        Returns:
            float: Steering angle to use.
        """
        steer = - self.tau_p * error
        return steer

    def _pd_controller(self, error:float) -> float:
        """Proportional and Derivative Controller

        Args:
            error (float): Difference to the center of the detected lane.

        Returns:
            float: Steering angle to use.
        """
        if len(self.errors) > 2:
            self.prev_error = self.errors[-2]
        else:
            self.prev_error = self.errors[-1]
        
        deviation = error - self.prev_error
        steer = - self.tau_d * deviation/0.1 + self._p_controller(error)
        return steer

    def _pid_controller(self, error:float) -> float:
        """PID-Controller

        Args:
            error (float): Difference to the center of the detected lane.

        Returns:
            float: Steering angle to use.
        """
        sum_error = sum(self.errors)
        steer = - self.tau_i * sum_error + self._pd_controller(error)
        return steer
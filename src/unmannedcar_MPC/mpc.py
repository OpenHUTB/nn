import math
import numpy as np
import utils
from scipy.optimize import minimize


class MPC:
    HORIZON = 10
    L = 2.5

    def __init__(self, drawer, ego):
        self.drawer = drawer
        self.ego = ego

    def run_step(self, path, speed_m_s, dt):
        # convert path to body frame
        path = utils.to_body_frame(self.ego, path)

        # fit the path with a polynomial
        poly = np.polyfit(path[:, 0], path[:, 1], 3)

        # generate lane center locations
        self.speed_dt = speed_m_s * dt

        x_arr = np.arange(1, self.HORIZON + 1) * self.speed_dt
        self.locs = np.vstack((x_arr, np.polyval(poly, x_arr))).T

        # mpc
        bounds = np.full((self.HORIZON, 2), (-0.3, 0.3))
        init_steer_arr = np.full(self.HORIZON, 0)
        solution = minimize(self.objective, init_steer_arr, (), method='SLSQP', bounds=bounds, tol=1e-4)
        eval_states = self.evaluate_states(solution.x)

        # draw lines
        self.drawer.draw_camera_lines((255, 255, 255), utils.to_global_frame(self.ego, self.locs), 1)
        self.drawer.draw_camera_lines((255, 0, 0), utils.to_global_frame(self.ego, np.array(eval_states)[:, 0:2]), 1)

        return solution.x[0]

    def mpc_model(self, state, steering):
        x_t = state[0]
        y_t = state[1]
        psi_t = state[2]

        x_t_1 = x_t + self.speed_dt * np.cos(psi_t)
        y_t_1 = y_t + self.speed_dt * np.sin(psi_t)
        psi_t_1 = psi_t + self.speed_dt * np.tan(steering) / self.L

        return [x_t_1, y_t_1, psi_t_1]

    def evaluate_states(self, steer_arr):
        states = []
        state = [0, 0, 0]
        for steer in steer_arr:
            state = self.mpc_model(state, steer)
            states.append(state)
        return states

    def objective(self, steer_arr, *args):
        final_state = self.evaluate_states(steer_arr)[-1]
        final_loc = self.locs[-1]
        return math.sqrt((final_state[0] - final_loc[0]) ** 2 + (final_state[1] - final_loc[1]) ** 2)
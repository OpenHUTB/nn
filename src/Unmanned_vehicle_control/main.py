import time
import threading
from src.carla_simulator import CarlaSimulator
from src.config import X_INIT_M, Y_INIT_M, N, dt, V_REF, LAPS
from src.help_functions import get_eight_trajectory, get_ref_trajectory, update_reference_point
from src.logger import Logger
from src.mpc_controller import MpcController

def draw_trajectory_in_thread(carla, x_traj, y_traj, dt):
    while True:
        carla.draw_trajectory(x_traj, y_traj, height=0.2, green=255, life_time=dt * 2)
        time.sleep(dt)

carla = CarlaSimulator()
carla.load_world('Town02_Opt')
carla.spawn_ego_vehicle('vehicle.tesla.model3', x=X_INIT_M, y=Y_INIT_M, z=0.1)
carla.print_ego_vehicle_characteristics()
carla.set_spectator(X_INIT_M, Y_INIT_M, z=50, pitch=-90)

logger = Logger()

x_traj, y_traj, v_ref, theta_traj = get_eight_trajectory(X_INIT_M, Y_INIT_M)
current_idx = 0
laps = 0

trajectory_thread = threading.Thread(target=draw_trajectory_in_thread, args=(carla, x_traj, y_traj, dt))
trajectory_thread.daemon = True
trajectory_thread.start()

mpc_controller = MpcController(horizon=N, dt=dt)
try:
    while True:
        start_time = time.time()

        mpc_controller.reset_solver()

        x0, y0, theta0, v0 = carla.get_main_ego_vehicle_state()
        mpc_controller.set_init_vehicle_state(x0, y0, theta0, v0)

        x_ref, y_ref, theta_ref = get_ref_trajectory(x_traj, y_traj, theta_traj, current_idx)
        carla.draw_trajectory(x_ref, y_ref, height=1.0, red=255, life_time=dt * 2)

        logger.log_controller_input(x0, y0, v0, theta0, x_ref[0], y_ref[0], V_REF, theta_ref[0])
        mpc_controller.update_cost_function(x_ref, y_ref, theta_ref, v_ref)

        mpc_controller.solve()

        wheel_angle_rad, acceleration_m_s_2 = mpc_controller.get_controls_value()

        # 增加：限制控制量范围，避免极端值导致错误
        wheel_angle_rad = max(min(wheel_angle_rad, 0.5), -0.5)
        acceleration_m_s_2 = max(min(acceleration_m_s_2, 3.0), -3.0)

        throttle, brake, steer = CarlaSimulator.process_control_inputs(wheel_angle_rad, acceleration_m_s_2)
        logger.log_controller_output(steer, throttle, brake)
        carla.apply_control(steer, throttle, brake)

        prev_current_idx = current_idx
        current_idx = update_reference_point(x0, y0, current_idx, x_traj, y_traj)
        if prev_current_idx == len(x_traj) - 1 and current_idx == 0:
            laps += 1

        end_time = time.time()
        mpc_calculation_time = end_time - start_time
        print(f"Calculation time of MPC controller: {mpc_calculation_time:.6f} seconds")

        time.sleep(max(dt - mpc_calculation_time, 0))

        if laps == LAPS:
            break
finally:
    carla.clean()
    logger.show_plots()
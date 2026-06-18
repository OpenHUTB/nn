from source.utility import sample_n_trajectories
from source.replaybuffer import ReplayBuffer
from source.carlaenv import CarlaEnv
import carla


env = CarlaEnv()
env.client.reload_world(False)
env.client.set_timeout(15)
rb = ReplayBuffer(10000)
paths = sample_n_trajectories(3, env, carla.VehicleControl(1, 0, 0), 1000)
rb.add_rollouts(paths)
print(rb.get_paths_num())
env._exit()

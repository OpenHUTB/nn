import numpy as np
import yaml
import torch
import torch.nn as nn
import carla
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from torchvision import models
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir="./log/", flush_secs=20)


def log_path(path, num_trajs):
    """log the path while sampling."""
    writer.add_scalar("Path/rewards", np.sum(path['rewards']), num_trajs)
    writer.add_scalar("Path/frames", path['frames'], num_trajs)


def log_training(loss, epoch_i):
    """"log training process."""
    writer.add_scalar("Train/loss", loss, epoch_i)


def get_env_settings(filename):
    """get all the settings of the Carla Simulator.

    The settings can be configured in config.yaml

    Returns:
        a dict of the initial settings
    """
    with open(filename, 'r') as f:
        env_settings = yaml.safe_load(f.read())

    # settings should follow the instructions
    assert env_settings['syn']['fixed_delta_seconds'] <= env_settings['substepping']['max_substep_delta_time'] * \
        env_settings['substepping']['max_substeps'], "substepping settings wrong!"
    return env_settings


def Path(obs, acs, rws, next_obs, terminals):
    """wrap a episode to a Path.
    Returns:
        Path(dict):
    """
    if obs != []:
        obs = torch.stack(obs)
        acs = torch.stack(acs).squeeze()
        next_obs = torch.stack(next_obs)
    return {
        "observations": obs,
        "actions": acs,
        "rewards": np.array(rws),
        "next_obs": next_obs,
        "terminals": np.array(terminals),
        "frames": len(obs)
    }


def sample_trajectory(env, action_policy, max_episode_length):
    """Sample one trajectory."""
    ob, _ = env.reset()
    print("reset the environment done.")
    # env.set_timeout(5)
    steps = 0
    obs, acs, rws, next_obs, terminals = [], [], [], [], []
    print("start sample new trajectory")
    while True:
        obs.append(ob)
        ac = action_policy.get_action(ob)
        acs.append(ac)
        # ac = map2action(ac)
        # print("action is: ", ac)
        # ac = action_policy  # test env
        next_ob, reward, done = env.step(ac)
        rws.append(reward)
        next_obs.append(next_ob)
        terminals.append(done)
        ob = next_ob
        steps += 1
        if done or steps >= max_episode_length:
            break

    return Path(obs, acs, rws, next_obs, terminals)


def sample_n_trajectories(n, env, action_policy, max_episode_length, epoch_i):
    paths = []
    for i in tqdm(range(n), desc="sample"):
        path = sample_trajectory(env, action_policy, max_episode_length)
        # log path
        log_path(path, epoch_i * n + i + 1)
        paths.append(path)
    return paths


def convert_path2list(paths):
    """convert the path to five list."""
    observations = [path["observations"] for path in paths]
    actions = [path["actions"] for path in paths]
    rewards = [path["rewards"] for path in paths]
    next_obs = [path["next_obs"] for path in paths]
    terminals = [path["terminals"] for path in paths]
    frames = [path["frames"] for path in paths]
    return observations, actions, rewards, next_obs, terminals, frames


def convert_control2numpy(action: carla.VehicleControl) -> np.ndarray:
    """Convert the control to numpy array."""
    return np.array([action.throttle, action.steer, action.brake])


def convert_tensor2control(pred_action: torch.Tensor) -> carla.VehicleControl:
    ac = tonumpy(pred_action)
    return carla.VehicleControl(ac[0], ac[1], ac[2])


def set_device(gpu_id):
    torch.cuda.set_device(gpu_id)


def totensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float().to(device)


def tonumpy(x: torch.Tensor) -> np.ndarray:
    return x.to('cpu').detach().numpy()


def map2action(index):
    """map action index to action.

    Returns:
        carla.VehicleControl()
    """
    if index == 0:
        return carla.VehicleControl(1, 0, 0)
    elif index == 1:
        return carla.VehicleControl(1, -1, 0)
    elif index == 2:
        return carla.VehicleControl(1, 1, 0)
    elif index == 3:
        return carla.VehicleControl(0, 0, 1)


def check_average_frames(paths):
    r = [np.sum(path['frames']) for path in paths]
    return np.mean(r)


def build_resnet():
    """build resnet return a pretrained resnet50 with no grad."""
    resnet = models.resnet50(pretrained=True)
    for para in resnet.parameters():
        para.requires_grad = False
    return resnet


def setup_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True

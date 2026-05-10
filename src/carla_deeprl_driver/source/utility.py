import numpy as np
import yaml
import torch
import carla
from torch.utils.tensorboard import SummaryWriter
from torchvision import models
from typing import Dict, List, Any, Tuple, Optional

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
writer = SummaryWriter(log_dir="./log/", flush_secs=20)


def log_path(path: Dict[str, Any], num_trajs: int):
    writer.add_scalar("Path/rewards", np.sum(path['rewards']), num_trajs)
    writer.add_scalar("Path/frames", path['frames'], num_trajs)


def log_training(loss: float, epoch_i: int):
    writer.add_scalar("Train/loss", loss, epoch_i)


def get_env_settings(filename: str) -> Dict[str, Any]:
    with open(filename, 'r') as f:
        env_settings = yaml.safe_load(f.read())
    
    assert env_settings['syn']['fixed_delta_seconds'] <= env_settings['substepping']['max_substep_delta_time'] * \
        env_settings['substepping']['max_substeps'], "Substepping settings are invalid!"
    
    return env_settings


def Path(obs: List[torch.Tensor], acs: List[torch.Tensor], rws: List[float], 
         next_obs: List[torch.Tensor], terminals: List[int]) -> Dict[str, Any]:
    if obs:
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


def sample_trajectory(env, action_policy, max_episode_length: int) -> Dict[str, Any]:
    ob, _ = env.reset()
    steps = 0
    obs, acs, rws, next_obs, terminals = [], [], [], [], []
    
    while steps < max_episode_length:
        obs.append(ob)
        ac = action_policy.get_action(ob)
        acs.append(ac)
        
        next_ob, reward, done = env.step(ac)
        rws.append(reward)
        next_obs.append(next_ob)
        terminals.append(done)
        
        ob = next_ob
        steps += 1
        
        if done:
            break
    
    return Path(obs, acs, rws, next_obs, terminals)


def sample_n_trajectories(n: int, env, action_policy, 
                          max_episode_length: int, epoch_i: int) -> List[Dict[str, Any]]:
    from tqdm import tqdm
    
    paths = []
    for i in tqdm(range(n), desc="Sampling trajectories"):
        path = sample_trajectory(env, action_policy, max_episode_length)
        log_path(path, epoch_i * n + i + 1)
        paths.append(path)
    
    return paths


def convert_path2list(paths: List[Dict[str, Any]]) -> Tuple[List, List, List, List, List, List]:
    observations = [path["observations"] for path in paths]
    actions = [path["actions"] for path in paths]
    rewards = [path["rewards"] for path in paths]
    next_obs = [path["next_obs"] for path in paths]
    terminals = [path["terminals"] for path in paths]
    frames = [path["frames"] for path in paths]
    
    return observations, actions, rewards, next_obs, terminals, frames


def convert_control2numpy(action: carla.VehicleControl) -> np.ndarray:
    return np.array([action.throttle, action.steer, action.brake])


def convert_tensor2control(pred_action: torch.Tensor) -> carla.VehicleControl:
    ac = tonumpy(pred_action)
    return carla.VehicleControl(ac[0], ac[1], ac[2])


def set_device(gpu_id: int):
    torch.cuda.set_device(gpu_id)


def totensor(x: np.ndarray) -> torch.Tensor:
    return torch.from_numpy(x).float().to(device)


def tonumpy(x: torch.Tensor) -> np.ndarray:
    return x.to('cpu').detach().numpy()


def map2action(index: int) -> carla.VehicleControl:
    action_map = {
        0: carla.VehicleControl(1, 0, 0),
        1: carla.VehicleControl(1, -1, 0),
        2: carla.VehicleControl(1, 1, 0),
        3: carla.VehicleControl(0, 0, 1)
    }
    return action_map.get(index, carla.VehicleControl(1, 0, 0))


def check_average_frames(paths: List[Dict[str, Any]]) -> float:
    frame_counts = [path['frames'] for path in paths]
    return np.mean(frame_counts)


def build_resnet() -> models.ResNet:
    resnet = models.resnet50(pretrained=True)
    for param in resnet.parameters():
        param.requires_grad = False
    return resnet


def setup_seed(seed: int):
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    torch.backends.cudnn.deterministic = True
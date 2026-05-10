from source.replaybuffer import ReplayBuffer
from source.sac import SAC
from source import utility as util
from torch.utils.tensorboard import SummaryWriter
import torch
config = util.get_env_settings("./config.yaml")


class Trainer(object):
    def __init__(self, env) -> None:
        self.env = env
        self.rb = ReplayBuffer(config['buffer_size'])
        self.sac = SAC(1, config['log_min'], config['log_max'],
                       self.rb, config['gamma'], config['soft_tau'])
        self.summarywriter = SummaryWriter("./log/", flush_secs=20)
        self.max_average_frames = 0

    def train(self, epoch_i):
        # sample n trajectories then add to replaybuffer
        paths = util.sample_n_trajectories(
            config['sample_n'], self.env, self.sac.policy_net, config['max_episode_length'], epoch_i)
        self.rb.add_rollouts(paths)
        # checkpoint
        self.save_model(paths, epoch_i)
        # using previous experience to update
        paths = self.rb.sample_random_rollouts(config['sample_n'])
        soft_q_loss, value_loss, policy_loss = self.sac.update(paths)
        self.log_info(soft_q_loss, value_loss, policy_loss, epoch_i)

    def log_info(self, soft_q_loss, value_loss, policy_loss, epoch_i):
        self.summarywriter.add_scalar(
            "Train/soft_q_loss", soft_q_loss, epoch_i)
        self.summarywriter.add_scalar("Train/value_loss", value_loss, epoch_i)
        self.summarywriter.add_scalar(
            "Train/policy_loss", policy_loss, epoch_i)

    def save_model(self, paths, epoch_i):
        average_frames = util.check_average_frames(paths)
        if average_frames > self.max_average_frames:
            self.max_average_frames = average_frames
            print(f"save model. Average frames is {average_frames}")
            torch.save(self.sac.soft_q_net.state_dict(),
                       f"./checkpoints/sac/softqnet{epoch_i}.pt")
            torch.save(self.sac.target_value_net.state_dict(),
                       f"./checkpoints/sac/target_vnet{epoch_i}.pt")
            torch.save(self.sac.policy_net.state_dict(),
                       f"./checkpoints/sac/policynet{epoch_i}.pt")

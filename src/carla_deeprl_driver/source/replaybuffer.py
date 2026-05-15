from source.utility import convert_path2list
import numpy as np


class ReplayBuffer(object):
    def __init__(self, buffer_size) -> None:
        self.paths = []
        self.observations = None
        self.actions = None
        self.rewards = None
        self.next_observations = None
        self.terminals = None
        self.frames = None
        self.buffer_size = buffer_size

    def add_rollouts(self, paths):
        """add serveral paths to rollouts"""
        for path in paths:
            self.paths.append(path)

        observations, actions, rewards, next_obs, terminals, frames = convert_path2list(
            paths)

        if self.observations is None:
            self.observations = observations[-self.buffer_size:]
            self.actions = actions[-self.buffer_size:]
            self.rewards = rewards[-self.buffer_size:]
            self.next_observations = next_obs[-self.buffer_size:]
            self.terminals = terminals[-self.buffer_size:]
            self.frames = frames[-self.buffer_size:]

        else:
            self.observations.append(
                observations)
            self.observations = self.observations[-self.buffer_size:]
            self.actions.append(actions)
            self.actions = self.actions[-self.buffer_size:]
            self.rewards.append(rewards)
            self.rewards = self.rewards[-self.buffer_size:]
            self.next_observations.append(
                next_obs)
            self.next_observations = self.next_observations[-self.buffer_size:]
            self.terminals.append(terminals)
            self.terminals = self.terminals[-self.buffer_size:]
            self.frames.append(frames)
            self.frames = self.frames[-self.buffer_size:]

    def sample_random_rollouts(self, num_rollouts):
        indices = np.random.permutation(len(self.paths))[:num_rollouts]
        return self.paths[indices]

    def sample_recent_rollouts(self, num_rollouts):
        """Sample recent rollouts.
        Returns:
            paths: A list of path
        """
        return self.paths[-num_rollouts:]

    def get_paths_num(self):
        return len(self.paths)

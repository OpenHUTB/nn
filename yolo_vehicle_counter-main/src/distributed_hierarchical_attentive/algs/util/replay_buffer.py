import logging
import random, collections
import numpy as np


class ReplayBuffer:
    """经验回放池"""

    def __init__(self, capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出

    def add(self, transition):
        self.buffer.append(transition)

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        transition = random.sample(self.buffer, batch_size)
        state, action, reward, next_state, truncated, done, info = zip(*transition)
        return state, action, reward, next_state, truncated, done, info

    def size(self):
        return len(self.buffer)


class SplitReplayBuffer:

    def __init__(self, capacity) -> None:
        self.buffer = collections.deque(maxlen=capacity)  # 队列，先进先出
        self.change_buffer = collections.deque(maxlen=capacity // 10)

    def add(self, transition, buffer=True):
        """Transiton: the rl transition need to sava
        buffer: True, add transition to self.buffer; False, add transition to self.change_buffer"""
        if buffer:
            self.buffer.append(transition)
        else:
            self.change_buffer.append(transition)

    def sample(self, batch_size):  # 从buffer中采样数据,数量为batch_size
        pri_size = min(batch_size // 2, len(self.change_buffer))
        normal_size = batch_size - pri_size
        transition = random.sample(self.buffer, normal_size)
        state, action, action_param, reward, next_state, truncated, done = zip(*transition)
        pri_transition = random.sample(self.change_buffer, pri_size)
        pri_state, pri_action, pri_action_param, pri_reward, pri_next_state, pri_truncated, pri_done = zip(
            *pri_transition)
        state = np.concatenate((state, pri_state), axis=0)
        action = np.concatenate((action, pri_action), axis=0)
        action_param = np.concatenate((action_param, pri_action_param), axis=0)
        reward = np.concatenate((reward, pri_reward), axis=0)
        next_state = np.concatenate((next_state, pri_next_state), axis=0)
        truncated = np.concatenate((truncated, pri_truncated), axis=0)
        done = np.concatenate((done, pri_done), axis=0)
        return state, action, action_param, reward, next_state, truncated, done

    def size(self):
        return len(self.buffer)


class OfflineReplayBuffer:
    """
    manually adjust the sampling of replay buffer and the replay buffer remains unchanged (1,000,000 buffers)
    """

    def __init__(self):
        file_path = "../out/all_replay_buffer.npy"
        # state: [1, 28], action: [1, 2], next_state: [1, 28], reward_list = [1, 6], truncated = [1, 1], done = [1, 1]
        # reward_list = np.array([[reward, reward_ttc, reward_com, reward_eff, reward_lan, reward_yaw]])
        self.replay_buffer = np.load(file_path, allow_pickle=True)
        self.buffer_num = self.replay_buffer.shape[0]
        # print(replay_buffer.shape)
        # split five buffer of different states: dangerous, large off-center, low-efficiency, on-curve, normal
        self.dangerous_buffer = collections.deque(maxlen=250000)
        self.off_center_buffer = collections.deque(maxlen=250000)
        self.low_efficiency_buffer = collections.deque(maxlen=250000)
        self.on_curve_buffer = collections.deque(maxlen=250000)
        self.normal_buffer = collections.deque(maxlen=1000000)

        self.ttc_thr = -0.00001
        self.lane_thr = 0.5
        self.eff_thr = 5
        self.curve_thr = 0.1
        self.split_replay_buffer()

    def size(self):
        return self.buffer_num

    def split_replay_buffer(self):
        for i in range(self.buffer_num):
            current_buffer = self.replay_buffer[i]
            fTTC = current_buffer[59]
            fLane = current_buffer[62]
            feff = current_buffer[61]
            fcurve = abs(current_buffer[1] - current_buffer[19])
            print("fttc: ", fTTC, "flane: ", fLane, "feff: ", feff, "fcurve: ", fcurve)
            if fTTC < self.ttc_thr:
                # print(fTTC)
                self.dangerous_buffer.append(current_buffer)
            elif fLane > self.lane_thr:
                self.off_center_buffer.append(current_buffer)
            elif feff < self.eff_thr:
                self.low_efficiency_buffer.append(current_buffer)
            elif fcurve > self.curve_thr:
                self.on_curve_buffer.append(current_buffer)
            else:
                self.normal_buffer.append(current_buffer)
        print("dangerous_buffer: ", len(self.dangerous_buffer), "off_center_buffer: ", len(self.off_center_buffer),
              "low_efficiency_buffer: ", len(self.low_efficiency_buffer), "on_curve_buffer: ",
              len(self.on_curve_buffer), "normal_buffer: ", len(self.normal_buffer))

    def sample(self, batch_size):
        specific_size = batch_size / 6
        normal_size = batch_size - 4 * specific_size
        dangerous_transition = random.sample(self.dangerous_buffer, specific_size)
        dangerous_transitions = zip(*dangerous_transition)
        off_center_transition = random.sample(self.off_center_buffer, specific_size)
        off_center_transitions = zip(*off_center_transition)
        low_efficiency_transition = random.sample(self.low_efficiency_buffer, specific_size)
        low_efficiency_transitions = zip(*low_efficiency_transition)
        on_curve_transition = random.sample(self.on_curve_buffer, specific_size)
        on_curve_transitions = zip(*on_curve_transition)
        normal_transition = random.sample(self.normal_buffer, normal_size)
        normal_transitions = zip(*normal_transition)
        all_transitions = np.concatenate((dangerous_transitions, off_center_transitions, on_curve_transitions,
                                          low_efficiency_transitions, normal_transitions), axis=0)
        state, action, next_state, reward, truncated, done = all_transitions[:, :28], all_transitions[:, 28:30],
        all_transitions[:, 30:56], all_transitions[:, 56:57], all_transitions[:, -2:-1], all_transitions[:, -1:]
        return state, action, reward, next_state, truncated, done


class SumTree(object):
    """
    This SumTree code is a modified version and the original code is from:
    https://github.com/jaara/AI-blog/blob/master/SumTree.py

    Story data with its priority in the tree.
    """
    data_pointer = 0

    def __init__(self, capacity):
        self.capacity = capacity  # for all priority values
        self.tree = np.zeros(2 * capacity - 1)
        # [--------------Parent nodes-------------][-------leaves to recode priority-------]
        #             size: capacity - 1                       size: capacity
        # self.data = np.zeros(capacity, dtype=object)  # for all transitions
        self.data = collections.deque(maxlen=capacity)
        # [--------------data frame-------------]
        #             size: capacity

    def add(self, p, transition):
        tree_idx = self.data_pointer + self.capacity - 1
        # self.data[self.data_pointer] = data  # update data_frame
        if self.size < self.capacity:
            self.data.append(transition)
        else:
            self.data[self.data_pointer] = transition
        self.update(tree_idx, p)  # update tree_frame

        self.data_pointer += 1
        if self.data_pointer >= self.capacity:  # replace when exceed the capacity
            self.data_pointer = 0

    def update(self, tree_idx, p):
        change = p - self.tree[tree_idx]
        self.tree[tree_idx] = p
        # then propagate the change through tree
        while tree_idx != 0:  # this method is faster than the recursive loop in the reference code
            tree_idx = (tree_idx - 1) // 2
            self.tree[tree_idx] += change

    def get_leaf(self, v):
        """
        Tree structure and array storage:

        Tree index:
             0         -> storing priority sum
            / \
          1     2
         / \   / \
        3   4 5   6    -> storing priority for transitions

        Array type for storing:
        [0,1,2,3,4,5,6]
        """
        parent_idx = 0
        while True:  # the while loop is faster than the method in the reference code
            cl_idx = 2 * parent_idx + 1  # this leaf's left and right kids
            cr_idx = cl_idx + 1
            if cl_idx >= self.capacity + self.size - 1:
                # if cl_idx >= len(self.tree):        # reach bottom, end search
                leaf_idx = parent_idx
                break
            else:  # downward search, always search for a higher priority node
                if v <= self.tree[cl_idx]:
                    parent_idx = cl_idx
                else:
                    v -= self.tree[cl_idx]
                    parent_idx = cr_idx

        if leaf_idx < self.capacity - 1:
            # not leaf, banch node
            leaf_idx = self.capacity - 1 + self.size - 1
        data_idx = leaf_idx - self.capacity + 1
        return leaf_idx, self.tree[leaf_idx], self.data[data_idx]

    @property
    def total_p(self):
        return self.tree[0]  # the root

    @property
    def size(self):
        return len(self.data)


class PriReplayBuffer(object):  # stored as ( s, a, r, s_, i ) in SumTree
    """
    Prioritized experience replay
    This Memory class is modified based on the original code from:
    https://github.com/jaara/AI-blog/blob/master/Seaquest-DDQN-PER.py
    Detailed information:
    https://yulizi123.github.io/tutorials/machine-learning/reinforcement-learning/4-6-prioritized-replay/
    """
    epsilon = 0.01  # small amount to avoid zero priority
    alpha = 0.6  # [0~1] convert the importance of TD error to priority. If alpha = 0, there is no Importance Sampling.
    beta = 0.4  # importance-sampling, from initial value increasing to 1
    beta_increment_per_sampling = 0.001
    abs_err_upper = 1.  # clipped abs error

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def add(self, transition):
        max_p = np.max(self.tree.tree[-self.tree.capacity:])
        if max_p == 0:
            max_p = self.abs_err_upper
        self.tree.add(max_p, transition)  # set the max p for new p

    def sample(self, n):
        # assert self.tree.size==self.tree.capacity
        b_idx, ISWeights = np.empty((n,), dtype=np.int32), np.empty((n, 1))
        b_state, b_action, b_reward, b_next_state, b_truncated, b_done, b_info = [], [], [], [], [], [], []
        pri_seg = self.tree.total_p / n  # priority segment
        self.beta = np.min([1., self.beta + self.beta_increment_per_sampling])  # max = 1

        min_prob = np.min(
            self.tree.tree[self.tree.capacity - 1:self.tree.capacity - 1 + self.size()]) / self.tree.total_p
        # min_prob = np.min(self.tree.tree[-self.tree.capacity:]) / self.tree.total_p     # for later calculate ISweight
        for i in range(n):
            a, b = pri_seg * i, pri_seg * (i + 1)
            v = np.random.uniform(a, b)  # sample from  [a, b)
            idx, p, data = self.tree.get_leaf(v)
            prob = p / self.tree.total_p
            ISWeights[i, 0] = np.power(prob / min_prob, -self.beta)
            b_idx[i] = idx
            b_state.append(data[0])
            b_action.append(data[1])
            b_reward.append(data[2])
            b_next_state.append(data[3])
            b_truncated.append(data[4])
            b_done.append(data[5])
            b_info.append(data[6])

        # print(self.tree.tree)
        # print(b_idx)
        return b_idx, ISWeights, (np.array(b_state), np.array(b_action), np.array(b_reward), np.array(b_next_state),
                                  np.array(b_truncated), np.array(b_done), np.array(b_info))

    def batch_update(self, tree_idx, abs_errors):
        abs_errors += self.epsilon  # convert to abs and avoid 0
        clipped_errors = np.minimum(abs_errors, self.abs_err_upper)
        ps = np.power(clipped_errors, self.alpha)
        for ti, p in zip(tree_idx, ps):
            self.tree.update(ti, p)

    def size(self):
        return self.tree.size

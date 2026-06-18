"""
单元测试框架

为强化学习项目提供基础测试功能
"""

import unittest
import numpy as np
import sys
import os
from typing import Callable, Any, Tuple

sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))


class RLTestCase(unittest.TestCase):
    """强化学习测试基类"""
    
    def setUp(self):
        """测试前设置"""
        self.test_passed = 0
        self.test_failed = 0
    
    def tearDown(self):
        """测试后清理"""
        pass
    
    def assertArrayEqual(self, arr1: np.ndarray, arr2: np.ndarray, msg: str = ""):
        """断言数组相等"""
        self.assertTrue(
            np.allclose(arr1, arr2),
            f"{msg}\n期望: {arr2}\n实际: {arr1}"
        )
    
    def assertArrayShape(self, arr: np.ndarray, shape: Tuple, msg: str = ""):
        """断言数组形状"""
        self.assertEqual(
            arr.shape, shape,
            f"{msg}\n期望形状: {shape}\n实际形状: {arr.shape}"
        )


class TestQLearner(RLTestCase):
    """Q-Learning测试"""
    
    def test_q_table_initialization(self):
        """测试Q表初始化"""
        from rl.algorithms import QLearner
        
        states = 16
        actions = 4
        learner = QLearner(states, actions)
        
        self.assertArrayShape(learner.q_table, (states, actions))
        self.assertTrue(np.allclose(learner.q_table, 0))
        print("✓ Q表初始化测试通过")
    
    def test_action_selection(self):
        """测试动作选择"""
        from rl.algorithms import QLearner
        
        states = 16
        actions = 4
        learner = QLearner(states, actions)
        
        state = 0
        action = learner.get_next_action_without_Q_table_update(state)
        
        self.assertIn(action, [0, 1, 2, 3])
        print("✓ 动作选择测试通过")
    
    def test_q_table_update(self):
        """测试Q表更新"""
        from rl.algorithms import QLearner
        
        states = 16
        actions = 4
        learner = QLearner(states, actions)
        
        initial_value = learner.q_table[0, 0]
        state = 0
        learner.get_next_action_with_Q_table_update(state, reward=1.0)
        
        self.assertNotEqual(learner.q_table[0, 0], initial_value)
        print("✓ Q表更新测试通过")


class TestSARSALearner(RLTestCase):
    """SARSA测试"""
    
    def test_sarsa_initialization(self):
        """测试SARSA初始化"""
        from rl.algorithms import SARSALearner
        
        states = 16
        actions = 4
        learner = SARSALearner(states, actions)
        
        self.assertArrayShape(learner.q_table, (states, actions))
        self.assertTrue(np.allclose(learner.q_table, 0))
        print("✓ SARSA初始化测试通过")
    
    def test_on_policy_update(self):
        """测试在线更新"""
        from rl.algorithms import SARSALearner
        
        states = 16
        actions = 4
        learner = SARSALearner(states, actions, radr=0)
        
        state = 0
        action = learner.get_next_action_without_Q_table_update(state)
        learner.get_next_action_with_Q_table_update(state, reward=0.5)
        
        self.assertIsNotNone(learner.last_rar)
        print("✓ SARSA在线更新测试通过")


class TestDQNLearner(RLTestCase):
    """DQN测试"""
    
    def test_dqn_initialization(self):
        """测试DQN初始化"""
        from rl.deep_learning import DQNLearner
        
        state_dim = 4
        action_dim = 2
        learner = DQNLearner(state_dim, action_dim)
        
        self.assertEqual(learner.state_dim, state_dim)
        self.assertEqual(learner.action_dim, action_dim)
        self.assertEqual(learner.epsilon, 1.0)
        print("✓ DQN初始化测试通过")
    
    def test_replay_buffer(self):
        """测试经验回放"""
        from rl.deep_learning import ReplayBuffer
        
        buffer = ReplayBuffer(capacity=100)
        
        buffer.push(0, 0, 1.0, 1, False)
        buffer.push(1, 1, 0.5, 2, False)
        
        self.assertEqual(len(buffer), 2)
        
        batch = buffer.sample(2)
        self.assertEqual(len(batch), 5)
        print("✓ 经验回放测试通过")
    
    def test_action_selection_epsilon(self):
        """测试ε-贪心动作选择"""
        from rl.deep_learning import DQNLearner
        
        learner = DQNLearner(state_dim=4, action_dim=2, epsilon=0.0)
        
        state = np.array([1.0, 0.0, 0.0, 0.0])
        action = learner.get_action(state, training=False)
        
        self.assertIn(action, [0, 1])
        print("✓ DQN动作选择测试通过")


class TestREINFORCE(RLTestCase):
    """REINFORCE测试"""
    
    def test_reinforce_initialization(self):
        """测试REINFORCE初始化"""
        from rl.simple_policy import REINFORCEAgent
        
        state_dim = 4
        action_dim = 2
        agent = REINFORCEAgent(state_dim, action_dim)
        
        self.assertEqual(agent.policy.state_dim, state_dim)
        self.assertEqual(agent.policy.action_dim, action_dim)
        print("✓ REINFORCE初始化测试通过")
    
    def test_action_selection(self):
        """测试动作选择"""
        from rl.simple_policy import REINFORCEAgent
        
        agent = REINFORCEAgent(state_dim=4, action_dim=2)
        
        state = 0
        action, prob = agent.select_action(state)
        
        self.assertIn(action, [0, 1])
        self.assertTrue(0 <= prob <= 1)
        print("✓ REINFORCE动作选择测试通过")
    
    def test_trajectory_update(self):
        """测试轨迹更新"""
        from rl.simple_policy import REINFORCEAgent
        
        agent = REINFORCEAgent(state_dim=4, action_dim=2, learning_rate=0.1)
        
        trajectory = [(0, 0, 0.0), (1, 1, 1.0), (2, 0, 0.0)]
        agent.update(trajectory)
        
        self.assertIsNotNone(agent.policy.weights)
        print("✓ REINFORCE轨迹更新测试通过")


class TestEnvironmentTools(RLTestCase):
    """环境工具测试"""
    
    def test_episode_statistics(self):
        """测试回合统计"""
        from rl.env_tools import EpisodeStatistics
        
        stats = EpisodeStatistics()
        
        stats.update(1.0)
        stats.update(0.5)
        stats.update(0.0)
        stats.end_episode()
        
        self.assertEqual(len(stats.episode_rewards), 1)
        self.assertEqual(stats.current_reward, 0)
        print("✓ 回合统计测试通过")
    
    def test_action_scaler(self):
        """测试动作缩放"""
        from rl.env_tools import ActionScaler
        import gymnasium as gym
        
        env = gym.make("MountainCar-v0")
        scaler = ActionScaler(env.action_space)
        
        scaled = scaler.scale(np.array([0.0]))
        self.assertTrue(np.all(scaled >= env.action_space.low))
        self.assertTrue(np.all(scaled <= env.action_space.high))
        
        env.close()
        print("✓ 动作缩放测试通过")
    
    def test_reward_scaling(self):
        """测试奖励缩放"""
        from rl.env_tools import RewardScaling, NormalizedEnv
        import gymnasium as gym
        
        env = gym.make("CartPole-v1")
        wrapped = NormalizedEnv(env)
        
        state = wrapped.reset()
        next_state, reward, done, trunc, info = wrapped.step(0)
        
        self.assertEqual(reward, 1.0)
        
        wrapped.close()
        print("✓ 奖励缩放测试通过")


class TestMultiEnv(RLTestCase):
    """多环境测试"""
    
    def test_cartpole_trainer(self):
        """测试CartPole训练器"""
        from rl.multi_env import CartPoleTrainer
        
        trainer = CartPoleTrainer()
        env = trainer.create_env()
        
        state = env.reset()[0]
        discrete = trainer.discretize_state(state)
        
        self.assertIsInstance(discrete, tuple)
        self.assertEqual(len(discrete), 4)
        
        env.close()
        print("✓ CartPole训练器测试通过")
    
    def test_mountain_car_trainer(self):
        """测试MountainCar训练器"""
        from rl.multi_env import MountainCarTrainer
        
        trainer = MountainCarTrainer()
        env = trainer.create_env()
        
        state = env.reset()[0]
        discrete = trainer.discretize_state(state)
        
        self.assertIsInstance(discrete, tuple)
        self.assertEqual(len(discrete), 2)
        
        env.close()
        print("✓ MountainCar训练器测试通过")
    
    def test_create_trainer(self):
        """测试训练器工厂"""
        from rl.multi_env import create_trainer
        
        trainer = create_trainer("CartPole-v1")
        self.assertIsNotNone(trainer)
        
        with self.assertRaises(ValueError):
            create_trainer("InvalidEnv-v0")
        
        print("✓ 训练器工厂测试通过")


def run_tests(verbose: bool = True):
    """运行所有测试"""
    loader = unittest.TestLoader()
    suite = unittest.TestSuite()
    
    suite.addTests(loader.loadTestsFromTestCase(TestQLearner))
    suite.addTests(loader.loadTestsFromTestCase(TestSARSALearner))
    suite.addTests(loader.loadTestsFromTestCase(TestDQNLearner))
    suite.addTests(loader.loadTestsFromTestCase(TestREINFORCE))
    suite.addTests(loader.loadTestsFromTestCase(TestEnvironmentTools))
    suite.addTests(loader.loadTestsFromTestCase(TestMultiEnv))
    
    runner = unittest.TextTestRunner(verbosity=2 if verbose else 1)
    result = runner.run(suite)
    
    return result.wasSuccessful()


if __name__ == "__main__":
    success = run_tests()
    exit(0 if success else 1)

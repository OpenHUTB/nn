from six import StringIO
import sys
import gym
from gym import spaces
import numpy as np
from gym import error
from gym.utils import seeding

# 随机策略函数：为当前玩家随机选择合法落子动作
def make_random_policy(np_random):
    def random_policy(state, player_color):
        possible_places = ReversiEnv.get_possible_actions(state, player_color)
        # 没有可落子位置，返回"pass"动作
        if len(possible_places) == 0:
            d = state.shape[-1]  # 动态获取棋盘的边长
            return d**2 + 1     # pass动作
        # 随机选择一个可能的动作
        a = np_random.randint(len(possible_places))  # 生成随机索引
        return possible_places[a]  # 返回对应索引的放置位置
    return random_policy  # 返回定义好的随机策略函数

class ReversiEnv(gym.Env):
    """黑白棋环境，支持标准8x8棋盘和强化学习接口"""
    BLACK = 0  # 黑棋内部标识
    WHITE = 1  # 白棋内部标识
    metadata = {"render.modes": ["ansi", "human"]}

    def __init__(self, player_color, opponent, observation_type, illegal_place_mode, board_size=8):
        """初始化环境参数"""
        assert isinstance(board_size, int) and board_size >= 4, '棋盘大小必须≥4（偶数）'
        self.board_size = board_size

        # 映射颜色字符串到内部标识
        colormap = {'black': ReversiEnv.BLACK, 'white': ReversiEnv.WHITE}
        try:
            self.player_color = colormap[player_color]
        except KeyError:
            raise error.Error("player_color必须是'black'或'white'")

        self.opponent = opponent  # 对手策略
        assert observation_type in ['numpy3c'], "仅支持'numpy3c'观测类型"
        self.observation_type = observation_type
        assert illegal_place_mode in ['lose', 'raise'], "仅支持'lose'或'raise'非法处理方式"
        self.illegal_place_mode = illegal_place_mode

        # 动作空间：棋盘位置（0~size²-1）+ pass（size²）+ resign（size²+1）
        self.action_space = spaces.Discrete(self.board_size ** 2 + 2)
        # 初始化观测空间
        observation = self.reset()
        self.observation_space = spaces.Box(
            low=-1.0, high=1.0, shape=observation.shape, dtype=np.float32
        )

        self._seed()  # 初始化随机种子

    def reset(self):
        """重置游戏状态，返回初始观测"""
        # 初始化棋盘（0=空，1=黑棋，-1=白棋）
        self.board = np.zeros((self.board_size, self.board_size), dtype=int)
        # 中心4格初始布局
        mid = self.board_size // 2
        self.board[mid-1, mid-1] = 1  # 黑棋
        self.board[mid-1, mid] = -1   # 白棋
        self.board[mid, mid-1] = -1   # 白棋
        self.board[mid, mid] = 1      # 黑棋
        
        self.current_player = 1  # 1=黑棋回合，-1=白棋回合
        self.possible_actions = self._get_valid_moves()  # 合法动作
        self.done = False  # 游戏结束标记
        return self._get_observation()

    def _get_valid_moves(self):
        """获取当前玩家的合法落子位置（索引形式）"""
        valid_moves = []
        for i in range(self.board_size):
            for j in range(self.board_size):
                if self.board[i, j] == 0 and self._is_valid_move(i, j):
                    valid_moves.append(i * self.board_size + j)
        if not valid_moves:
            valid_moves.append(self.board_size ** 2 + 1)  # 无合法动作时添加pass
        return valid_moves

    def _is_valid_move(self, i, j):
        """判断(i,j)是否为当前玩家的合法落子位置"""
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        current_color = self.current_player
        opponent_color = -current_color

        for di, dj in directions:
            x, y = i + di, j + dj
            flipped = []
            while 0 <= x < self.board_size and 0 <= y < self.board_size:
                if self.board[x, y] == opponent_color:
                    flipped.append((x, y))
                    x += di
                    y += dj
                elif self.board[x, y] == current_color:
                    if flipped:
                        return True
                    break
                else:
                    break
        return False

    def _get_observation(self):
        """返回3通道观测：[黑棋位置, 白棋位置, 当前玩家]"""
        obs = np.zeros((3, self.board_size, self.board_size), dtype=np.float32)
        obs[0] = (self.board == 1).astype(np.float32)  # 黑棋通道
        obs[1] = (self.board == -1).astype(np.float32) # 白棋通道
        obs[2] = np.full((self.board_size, self.board_size), self.current_player, dtype=np.float32)
        return obs

    def _seed(self, seed=None):
        """设置随机种子"""
        self.np_random, seed = seeding.np_random(seed)
        # 初始化对手策略
        if isinstance(self.opponent, str):
            if self.opponent == 'random':
                self.opponent_policy = make_random_policy(self.np_random)
            else:
                raise error.Error('不支持的对手策略: {}'.format(self.opponent))
        else:
            self.opponent_policy = self.opponent
        return [seed]

    def step(self, action):
        """执行动作并更新环境状态"""
        if self.done:
            return self._get_observation(), 0.0, True, {}

        reward = 0.0
        # 检查动作合法性
        if not self._is_action_valid(action):
            if self.illegal_place_mode == 'raise':
                raise error.Error(f"非法动作: {action}（当前玩家：{self.current_player}）")
            else:
                reward = -1.0  # 非法动作判负
                self.done = True
                return self._get_observation(), reward, self.done, {}

        # 执行合法动作（落子/翻转）
        if not self._is_pass_or_resign(action):
            self._place_stone(*self._action_to_coords(action))

        # 切换到对手回合
        self.current_player *= -1
        self.possible_actions = self._get_valid_moves()

        # 对手行动
        if not self.done:
            opponent_action = self.opponent_policy(self._get_observation(), self.current_player)
            if not self._is_action_valid(opponent_action):
                reward = 1.0  # 对手非法动作，当前玩家胜
                self.done = True
            else:
                if not self._is_pass_or_resign(opponent_action):
                    self._place_stone(*self._action_to_coords(opponent_action))
                # 检查游戏是否结束（双方无合法动作）
                self.current_player *= -1
                self.possible_actions = self._get_valid_moves()
                if len(self.possible_actions) == 1 and self.possible_actions[0] == self.board_size**2 + 1:
                    # 计算最终得分
                    black_count = np.sum(self.board == 1)
                    white_count = np.sum(self.board == -1)
                    if black_count > white_count:
                        reward = 1.0 if self.player_color == ReversiEnv.BLACK else -1.0
                    elif black_count < white_count:
                        reward = 1.0 if self.player_color == ReversiEnv.WHITE else -1.0
                    else:
                        reward = 0.0  # 平局
                    self.done = True
                self.current_player *= -1

        return self._get_observation(), reward, self.done, {'board': self.board.copy()}

    def render(self, mode='human'):
        """渲染棋盘状态"""
        outfile = StringIO() if mode == 'ansi' else sys.stdout
        d = self.board_size

        # 打印列标
        outfile.write('   ')
        for j in range(d):
            outfile.write(f' {j:2} ')
        outfile.write('\n')

        # 打印棋盘
        for i in range(d):
            outfile.write(f'{i:2} ')  # 行标
            for j in range(d):
                if self.board[i, j] == 1:
                    outfile.write(' B ')  # 黑棋
                elif self.board[i, j] == -1:
                    outfile.write(' W ')  # 白棋
                else:
                    outfile.write(' . ')  # 空位
            outfile.write('\n')

        if mode == 'ansi':
            return outfile.getvalue()

    def close(self):
        """关闭环境"""
        pass

    # 辅助方法：动作转坐标
    def _action_to_coords(self, action):
        if self._is_pass_or_resign(action):
            return (-1, -1)  # 特殊动作返回无效坐标
        return (action // self.board_size, action % self.board_size)

    # 辅助方法：检查动作是否为pass或resign
    def _is_pass_or_resign(self, action):
        return action in [self.board_size**2, self.board_size**2 + 1]

    # 辅助方法：检查动作是否合法
    def _is_action_valid(self, action):
        if self._is_pass_or_resign(action):
            return True
        i, j = self._action_to_coords(action)
        return 0 <= i < self.board_size and 0 <= j < self.board_size and self._is_valid_move(i, j)

    # 辅助方法：落子并翻转对手棋子
    def _place_stone(self, i, j):
        current_color = self.current_player
        self.board[i, j] = current_color

        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        
        for di, dj in directions:
            x, y = i + di, j + dj
            flipped = []
            while 0 <= x < self.board_size and 0 <= y < self.board_size:
                if self.board[x, y] == -current_color:
                    flipped.append((x, y))
                    x += di
                    y += dj
                elif self.board[x, y] == current_color:
                    for (fx, fy) in flipped:
                        self.board[fx, fy] = current_color
                    break
                else:
                    break

    # 静态方法：供外部策略获取合法动作（关键修复）
    @staticmethod
    def get_possible_actions(state, player_color):
        """根据观测状态和玩家颜色返回合法动作"""
        board_size = state.shape[-1]
        board = np.zeros((board_size, board_size), dtype=int)
        board[state[0] == 1.0] = 1  # 黑棋位置
        board[state[1] == 1.0] = -1 # 白棋位置
        
        valid_moves = []
        for i in range(board_size):
            for j in range(board_size):
                if board[i, j] == 0 and ReversiEnv._static_is_valid_move(board, i, j, player_color):
                    valid_moves.append(i * board_size + j)
        
        if not valid_moves:
            valid_moves.append(board_size ** 2 + 1)
        return valid_moves

    @staticmethod
    def _static_is_valid_move(board, i, j, player_color):
        """静态方法：判断落子是否合法"""
        directions = [(-1, -1), (-1, 0), (-1, 1),
                      (0, -1),          (0, 1),
                      (1, -1),  (1, 0), (1, 1)]
        opponent_color = -player_color
        board_size = board.shape[0]

        for di, dj in directions:
            x, y = i + di, j + dj
            flipped = []
            while 0 <= x < board_size and 0 <= y < board_size:
                if board[x, y] == opponent_color:
                    flipped.append((x, y))
                    x += di
                    y += dj
                elif board[x, y] == player_color:
                    if flipped:
                        return True
                    break
                else:
                    break
        return False
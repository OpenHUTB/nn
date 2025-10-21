DISCOUNT = 0.99
# Discount factor for future rewards in the RL algorithm.

FPS = 60
# Frames per second in the simulation.

MEMORY_FRACTION = 0.35
# Fraction of GPU memory allocated for training.

REWARD_OFFSET = -100
# Stops the simulation when reached

MIN_REPLAY_MEMORY_SIZE = 1_000
# Minimum size of the replay memory before training starts.

REPLAY_MEMORY_SIZE = 5_000
# Maximum capacity of the replay memory.

MINIBATCH_SIZE = 64
# Number of experiences sampled from the replay memory for each training iteration.

PREDICTION_BATCH_SIZE = 1
# Batch size used during the prediction phase.

TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 8
# Batch size used during the training phase.

EPISODES = 451
# Number of episodes the agent will train on.

# SECONDS_PER_EPISODE = 60
SECONDS_PER_EPISODE = 45
# Duration of each episode in seconds.

MIN_EPSILON = 0.1
EPSILON = 1.0
# Exploration rates for the epsilon-greedy exploration strategy.

EPSILON_DECAY = 0.9975
# EPSILON_DECAY = 0.993
# Decay rate of the exploration rate over time.

MODEL_NAME = "YY"
# Name or identifier for the trained model.F

MIN_REWARD = 100
# MIN_REWARD = -1000
# Minimum reward required for an experience to be considered "good" or "positive."

UPDATE_TARGET_EVERY = 5
# Frequency at which the target network is updated.

AGGREGATE_STATS_EVERY = 10
# Frequency at which statistics (e.g., average scores, rewards) are computed and aggregated.

SHOW_PREVIEW = False
# Determines whether to show a preview window or not.

IM_WIDTH = 640
# Width of the image captured in the preview or simulation.

IM_HEIGHT = 480
# Height of the image captured in the preview or simulation.

SLOW_COUNTER = 330

LOW_REWARD_THRESHOLD = -4

SUCCESSFUL_THRESHOLD = 1

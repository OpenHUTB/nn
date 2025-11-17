# Hyperparameters.py
DISCOUNT = 0.95
# Discount factor for future rewards in the RL algorithm.

FPS = 60
# Frames per second in the simulation.

MEMORY_FRACTION = 0.35
# Fraction of GPU memory allocated for training.

REWARD_OFFSET = -100
# Stops the simulation when reached

MIN_REPLAY_MEMORY_SIZE = 2_000
# Minimum size of the replay memory before training starts.

REPLAY_MEMORY_SIZE = 10_000
# Maximum capacity of the replay memory.

MINIBATCH_SIZE = 32
# Number of experiences sampled from the replay memory for each training iteration.

PREDICTION_BATCH_SIZE = 1
# Batch size used during the prediction phase.

TRAINING_BATCH_SIZE = MINIBATCH_SIZE // 4
# Batch size used during the training phase.

EPISODES = 1000
# Number of episodes the agent will train on.

SECONDS_PER_EPISODE = 60
# Duration of each episode in seconds.

MIN_EPSILON = 0.01
EPSILON = 1.0
# Exploration rates for the epsilon-greedy exploration strategy.

EPSILON_DECAY = 0.995
# Decay rate of the exploration rate over time.

MODEL_NAME = "YY_Optimized"
# Name or identifier for the trained model.

MIN_REWARD = 5
# Minimum reward required for an experience to be considered "good" or "positive."

UPDATE_TARGET_EVERY = 10
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

LOW_REWARD_THRESHOLD = -2

SUCCESSFUL_THRESHOLD = 3

LEARNING_RATE = 0.0001
# Learning rate for the optimizer
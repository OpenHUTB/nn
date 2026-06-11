
Objective
This repo trains a Deep Reinforcement Learning agent in Carla for a vehicle to autonomusly follow a path using semantic segmentation sensor as the input.

Dependencies
This repo is tested on Carla 0.9.15

You can install the dependencies by running the following script.

pip3 install -r requirements.txt
Arguments
python3 train.py --host --port --town --total_timesteps --reload_model --fps --config --num_checkpoints --no_render
Configuration file
The configuration is located in config.py. It contains the following parameters:

algorithm: The RL algorithm to use. Algorithms with continuous action space are supported now.
algoritm_params: The parameters of the algorithm. See the Stable Baselines 3 documentation for more information.
action_smoothing: Whether to use action smoothing or not.
reward_fn: The reward function to use. See the agent/rewards.py file for more information.
reward_params: The parameters of the reward function.
obs_res: The resolution of the observation. It's recommended to use (160, 80)
Usage
# Clone the repo
git clone https://github.com/YuHang-Zhou/nn.git

# Go inside the repo
cd RL_SB3_carla

# Run the training script
# The default --host arg is IP of a different Host
python3 train.py
Run an experiment
# Run Carla on your system
./CarlaUE4.sh -RenderOffScreen

# Run the training and Carla on one host
python3 train.py --host "localhost"
Note
This repo was tested on two host machines :-

Host 1 - Running carla simulator(0.9.15)
Host 2 - running the RL agent
The --host argument is set to a different IP by default. Change this to localhost to run everything on your system.

The inspiration of the code was taken from this repo. Check it out.



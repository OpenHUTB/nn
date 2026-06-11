CrowdMARL
crowd simulation using multi-agent PPO algorithm.
2021 Fall CS492: Introduction to Deep Learning Team11 final project.

Result video
Youtube Link

Code descriptions
environment.py : Implementation of crowd simulation reinforcement learning environment.
model.py : The actor-critic model used for training.
train_model.py : Model traning code (with PPO algorithm implementation).
test_model.py : Model test code (including simulation visualization tool).
utils.py : A collection of utility functions.
Environments
Currently, we are supporting 6 environments:

basic : Basic environment with only one agent and target.
circle1 : 8 agents are located at the vertices dividing the circle into 8 equal parts, and targeting the opposite vertex.
circle2 : 8 agents are located at the vertices of square and circle, and targeting the opposite vertex.
crossing1 : Two groups of agents cross each other horizontally.
crossing2 : Two groups of agents cross each other vertically.
obstacles : 5 agents move toward the target while avoiding obstacles.
Dependencies
pip install -r requirements.txt
Train model
python .\train_model.py --env=<environment_name> --path=<model_save_location> --model=<model_load_location>

options
--dt (required) : The timestep of the environment.
-e/--env (optional) : The name of the environment to use.
-p/--path (optional) : Path to save trained parameters of the model.
-m/--model (optional) : Path of the model to resume.
Example
python .\train_model.py --dt --path .\checkpoints\
Test model
python .\test_model.py --model=<test_model_location> --env=<environment_name>

options
--dt (required) : The timestep of the environment.
-m/--model (optional) : Path of the model to test.
-e/--env (optional) : The name of the environment to use.
-r/--render (optional) : Whether to render result or not. (Default=True)
Example
 python .\test_model.py --dt 0.25 --env obstacles
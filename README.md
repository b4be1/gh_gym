# Grasshopper Gym
Rhino Grasshopper as OpenAI Gym Environment for Reinforcement Learning with Stable-Baselines3

## Installation
Install Anaconda.
Create Python 3.8 environment and install `stable-baselines3` by running the following commands
```bash
    conda create --name py38 python=3.8
    conda activate py38
    pip install stable-baselines3
````
To test socket communication, run `python server.py` and `python client.py` in separate terminals.
Server acts as the agent and client acts as the environment.
In the server terminal, you should see a printout of 10 environment steps;
the client script should terminate with `[Errno 9] Bad file descriptor`.

To test learning, run `python baseline.py` and `python client.py` in separate terminals.
The server should print out the learning progress and terminate by reporting the final return;
the client should terminate with `[Errno 9] Bad file descriptor`.

To test learning in Grasshopper, open `gh_env.gh` and run `python baseline.py` in a terminal.
Click Reset and then Loop on the HoopSnake component in Grasshopper
and observe the output in the terminal.
After 3000 time steps, the agent will execute the learned policy and print out its return.

## Usage
Scripts `python server.py` and `python client.py` are provided as an example of running
an OpenAI Gym environment over a socket.
These scripts should only be used for testing communication or by users familiar with Python
for implementing custom functionality.

File `gh_env.gh` shows how to implement an RL environment inside Grasshopper.
Environment `Pendulum-v0` from OpenAI Gym is implemented as an example.
Script `baseline.py` shows how to apply learning algorithms from the RL library `stable-baselines3`
using Grasshopper as the environment. Soft actor-critic (SAC) algorithm is used as an example.

Users should modify `gh_env.gh` to represent their task inside Grasshopper
and adjust `baseline.py` to choose a suitable RL algorithm and set learning parameters.
See `stable-baselines3` documentation for tips on optimizing the performance of RL algorithms.
 
### Implementing Environment in Grasshopper
There are two Python components in `gh_env.gh`: Agent and Environment.
Only Environment needs to be replaced by the user.
Environment receives two inputs---action and reset---and
it sets 4 global variables as a reaction to the inputs
```python
sc.sticky["observation"]
sc.sticky["reward"]
sc.sticky["done"]
sc.sticky["info"]
```
 When `reset=True`, the `info` field should contain the description of the action space
 and the observation space according to the OpenAI Gym interface,
 and the `observation` field should return the initial observation of the environment.
 Otherwise, `action` is provided by the agent and the environment should set all 4 global variables.
 
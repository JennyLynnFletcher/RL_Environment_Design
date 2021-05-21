# Using Reinforcement Learning to Design Environments for Autonomous Agents

This project is a collection of code used for using RL to train a policy to be able to improve the layout of obstacles in an environment in order to improve robot navigation efficiency.

## Required Libraries

All required libraries can be installed using:
```pip install -r requirements.txt```

The version of ```torch``` required may be different depending on available CUDA version.

## Usage

```python Simulator.py``` runs the RVO simulation and produces a pygame render of the robot navigation.

```python train_local.py``` and ```python train_remote.py``` run training with different settings and hyperparameters which are used for use locally or on a GPU cluster respectively. unning training saves checkpoints and progress data in ```~/ray_results```. The data analysis and testing code expects to find this data in ```~/ray_results```. 

```python Training/test_policy.py``` runs a policy on 500 intial environments, a seed value is used for these environments so intial environments are kept consistent between test runs.

```python RVO_time.py``` Tests the execution time of RVO on different environments using ```timeit``` and outputs the average in seconds.

## Future Extensions
There are many potential extensions to this project, if you would like to implement any (or any other extensions or improvements) feel free to submit a pull request.

- Implement non-holonomic robots. In many applications navigation is non-holonomic, for example cars using Ackermann steering. By adding further constraints, the learned policy has potential to show different behaviour depending on the motion constraints.

- Implement alternative robot navigation algorithms. As in the previous extension, this may show different learned policy behaviour, potentially some navigation algorithms may allow for better or worse environment design.

- Using the same RL environment design, different RL algorithms could be used and performance between them evaluated.

- Have more constraints on the environments the robots navigate, for example some cells could be more costly to navigate or more costly to put obstacles one. This is a better model for real life scenarios where there may be other considerations of environment layout beyond maximising robot throughput.

- Designing environments which for robots when start and goal locations may not always be known. Sometimes, the exact journeys that are taken are not known during the design phase, however likelihood of certain journeys is high. An RL agent could be given probability maps for start and goal locations.

- Using a heterogeneous multi-agent reinforcement learning algorithm with the environment designer as one agent and the navigation of the robots as another agent. This has the potential to produce novel behaviours where certain navigation strategies are employed in response to environment layouts, that would not otherwise be seen using a predefined navigation strategy.


## License
[MIT](https://choosealicense.com/licenses/mit/) 

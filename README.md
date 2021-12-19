# Reinforcement Learning
This project presents the solution for the **cart-pole** environment (available [here](https://gym.openai.com/envs/CartPole-v1/)) from [OpenAI Gym](https://gym.openai.com).

## Environment: cart-pole
A pole is attached by an un-actuated joint to a cart, which moves along a frictionless track. The system is controlled by applying a force of +1 or -1 to the cart. The pendulum starts upright, and the goal is to prevent it from falling over. A reward of +1 is provided for every timestep that the pole remains upright.
The episode ends when the pole is more than 15 degrees from vertical, or the cart moves more than 2.4 units from the center.

<kbd>![cart-pole environment](https://miro.medium.com/max/1200/1*hc84EDx6iNcrf0aLa4CuIw.gif)</kbd>

### Action space
The agent may only choose to go LEFT or RIGHT.

### Observation space
The observation available at each time step contains the following information:
* cart position [-4.8, +4,8]
* cart velocity [-Inf, +Inf]
* pole angle [-0.418, +0.418]
* pole angular velocity [-Inf, +Inf]

### Reward
A reward of +1 is provided for every timestep that the pole remains upright.

# Approximate solution methods
The [agents](https://github.com/giannpelle/RL-OpenAI-gym-cart-pole/blob/master/agents) directory contains the most popular algorithms to solve the environment using an approximation function to calculate the Q-values of each state-action pair.

## Double Deep Q-Network
It was developed following the algorithm available in *Deep Reinforcement Learning with Double Q-learning* (Hasselt et al., 2015), with the following optimization technique:
1. *Methods and Apparatus for Reinforcement Learning* (Mnih et al., 2017)
The code implementation is available [here](https://github.com/giannpelle/RL-OpenAI-gym-cart-pole/blob/main/agents/DoubleDQN_agent.py).

## Prioritized Double Deep Q-Network
It was developed following the algorithm available in *Deep Reinforcement Learning with Double Q-learning* (Hasselt et al., 2015), with the following 2 optimization techniques:
1. *Methods and Apparatus for Reinforcement Learning* (Mnih et al., 2017)
2. *Prioritized Experience Replay* (Schaul et al., 2016)
The code implementation is available [here](https://github.com/giannpelle/RL-OpenAI-gym-cart-pole/blob/main/agents/PerDoubleDQN_agent.py).

## Dueling Prioritized Double Deep Q-Network
It was developed following the algorithm available in *Dueling Network Architectures for Deep Reinforcement Learning* (Wang et al., 2016), with the following 2 optimization techniques:
1. *Methods and Apparatus for Reinforcement Learning* (Mnih et al., 2017)
2. *Prioritized Experience Replay* (Schaul et al., 2016)
The code implementation is available [here](https://github.com/giannpelle/RL-OpenAI-gym-cart-pole/blob/main/agents/DuelingPerDoubleDQN_agent.py).

## Installation

```bash
cd RL-OpenAI-gym-cart-pole
conda env create -f environment.yml
conda activate cart-pole

python cart_pole_player.py --method=doubledqn
python cart_pole_player.py --method=perdoubledqn
python cart_pole_player.py --method=duelingperdoubledqn
```
## References
* [Khev: RL-practice-keras](https://github.com/Khev/RL-practice-keras/blob/master/DDQN/write_up_for_openai.ipynb)
* [jaromuru: AI-blog](https://github.com/jaromiru/AI-blog/blob/master/SumTree.py)
* [jaromiru: AI-blog](https://github.com/jaromiru/AI-blog/blob/master/Seaquest-DDQN-PER.py)
* [Solving the Cartpole with Dueling Double Deep Q Network](https://python.plainenglish.io/solving-the-cartpole-with-dueling-double-deep-q-network-10a2040ecfc7)


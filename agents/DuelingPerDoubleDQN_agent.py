#!/usr/bin/env python

from typing import NamedTuple
from PER_utils import SumTree

import numpy as np
import random
import time
import statistics

import gym

from keras.models import Model
from tensorflow.keras.layers import Input, Dense, Lambda, Add
from tensorflow.keras.optimizers import Adam
from keras import backend as K

class StepExperience(NamedTuple):
    """
    represents a single step experience perceived from the agent
    """
    state: int
    action: int
    reward: float
    new_state: int
    done: bool

class Memory:
    """
    Implementation of Memory was taken from:
    https://github.com/jaromiru/AI-blog/blob/master/Seaquest-DDQN-PER.py

    experiences are stored as ( s, a, r, s_ ) in SumTree
    """
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append((idx, data))

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

    def get_size(self):
        return len(self.tree.data)

class DuelingPerDoubleDQNModel(object):
    """
    implementation of a neural network to derive a value approximation function
    for each (state, action) pair
    The hyper-parameters of the neural networks and
    the epsilon, decay values of the epsilon-greedy policy
    were taken from this implementation (ONLY the values of the parameters)
    https://github.com/Khev/RL-practice-keras/blob/master/DDQN/agent.py
    https://github.com/Khev/RL-practice-keras/blob/master/DDQN/write_up_for_openai.ipynb

    target network is a more stable model for the state-action value approximation, 
    whereas the online network is a more dynamic model updated at each time step. 
    We use it to get the behavior policy
    target network weights are updated at each time step towards the online network weigths 
    based on the tau parameter
    """
    def __init__(self, observation_space, action_space, memory_size=2_000, batch_size=32, alpha=0.001, gamma=0.99, tau=0.1, learning_start=500):
        self.memory = Memory(memory_size)
        self.batch_size = batch_size
        self.gamma = gamma
        self.tau = tau

        # collect this many experiences before learning
        self.learning_start = learning_start

        self.observation_space = observation_space
        self.action_space = action_space

        self.online_network = self.make_network(observation_space=observation_space, action_space=action_space, alpha=alpha)
        self.target_network = self.make_network(observation_space=observation_space, action_space=action_space, alpha=alpha)

    def make_network(self, *, observation_space, action_space, alpha):
        """
        implementation of dueling neural network taken from 
        https://python.plainenglish.io/solving-the-cartpole-with-dueling-double-deep-q-network-10a2040ecfc7
        """
        X_input = Input(observation_space)
        
        X = X_input
        X = Dense(64, input_shape=observation_space, activation="relu", kernel_initializer='he_uniform')(X)
        X = Dense(64, activation="relu", kernel_initializer='he_uniform')(X)

        state_value = Dense(1, kernel_initializer='he_uniform')(X)
        state_value = Lambda(lambda s: K.expand_dims(s[:, 0], -1), output_shape=(action_space,))(state_value)

        action_advantage = Dense(action_space, kernel_initializer='he_uniform')(X)
        action_advantage = Lambda(lambda a: a[:, :] - K.mean(a[:, :], keepdims=True), output_shape=(action_space,))(action_advantage)

        X = Add()([state_value, action_advantage])

        model = Model(inputs = X_input, outputs = X)
        model.compile(loss="mean_squared_error", optimizer=Adam(learning_rate=alpha), metrics=["accuracy"])

        model.summary()
        return model

    def get_greedy_action_for(self, *, state):
        q_values = self.online_network.predict(state)[0]
        return np.argmax(q_values)

    def remember(self, experience): 
        q_value = experience.reward
        if not experience.done:
            new_action = np.argmax(self.online_network.predict(experience.new_state)[0])
            q_value = experience.reward + self.gamma * self.target_network.predict(experience.new_state)[0][new_action]
        delta = abs(self.target_network.predict(experience.state)[0][experience.action] - q_value)
        self.memory.add(delta, experience)

    def experience_replay(self):
        if self.memory.get_size() < self.learning_start:
            return

        batches = self.memory.sample(self.batch_size)

        # vectorized approach for speed performance concernes
        state_batch = np.zeros((self.batch_size, self.observation_space[0]))
        action_batch = []
        reward_batch = []
        new_state_batch = np.zeros((self.batch_size, self.observation_space[0]))
        done_batch = []

        for index, (tree_index, experience) in enumerate(batches):
            state_batch[index] = experience.state
            action_batch.append(experience.action)
            reward_batch.append(experience.reward)
            new_state_batch[index] = experience.new_state
            done_batch.append(experience.done)

        online_network_predict_state_batch = self.online_network.predict(state_batch)
        online_network_predict_new_state_batch = self.online_network.predict(new_state_batch)
        target_network_predict_new_state_batch = self.target_network.predict(new_state_batch)
        
        index_experience_q_values = []

        for index, (tree_index, experience) in enumerate(batches):
            q_update = experience.reward
            if not experience.done:
                new_action = np.argmax(online_network_predict_new_state_batch[index])
                q_update = experience.reward + self.gamma * target_network_predict_new_state_batch[index][new_action]
            
            online_network_predict_state_batch[index][action_batch[index]] = q_update
            index_experience_q_values.append((tree_index, experience, q_update, index))

        self.online_network.fit(state_batch, online_network_predict_state_batch, batch_size=self.batch_size, verbose=0)
        self.update_target_network()

        target_network_predict_state_batch = self.target_network.predict(state_batch)

        for index, experience, q_value, batch_index in index_experience_q_values:
            delta = abs(target_network_predict_state_batch[batch_index][experience.action] - q_value)
            self.memory.update(index, delta)

    def update_target_network(self):
        online_network_weights = self.online_network.get_weights()
        target_network_weights = self.target_network.get_weights()
        new_weights = []

        for online_weight, target_weight in zip(online_network_weights, target_network_weights):
            new_weights.append(target_weight * (1 - self.tau) + online_weight * self.tau)
        self.target_network.set_weights(new_weights)

class DuelingPerDoubleDQNAgent(object):
    """
    Implementation of the Dueling Prioritized Double Deep Q-Learning RL technique
    """

    def __init__(self, *, env):
        self.env = env
        self.model = DuelingPerDoubleDQNModel(observation_space=env.observation_space.shape, action_space=env.action_space.n)

    def epsilon_greedy_policy(self, *, state, epsilon):
        """
        Creates an epsilon-greedy policy based on a given Q-function and epsilon
        """
        actions_count = self.env.action_space.n
        greedy_action = self.model.get_greedy_action_for(state=state)
        probs = np.full(actions_count, epsilon / actions_count)
        probs[greedy_action] += 1 - epsilon
        return int(np.random.choice(range(actions_count), p=probs))

    def off_policy_train(self):
        """
        agent training phase
        """
        print("\nTRAINING (with DuelingPerDoubleDQN learning method)")
        observations_count = self.env.observation_space.shape[0]
        actions_count = self.env.action_space.n

        episode = 0
        rewards = []

        epsilon = 0.5
        min_epsilon = 0.01
        decay = 0.99

        while True:
            print(f"running episode: {episode}")
            if episode % 10 == 0 and episode > 0:
                print(f"training rewards: {rewards[-10:]}")
                print(f"last 100 rewards mean: {statistics.mean(rewards[-100:])}")

            steps = 0
            observation = self.env.reset()
            state = np.reshape(observation, [1, observations_count])

            done = False
            episode_reward = 0

            while not done:
                steps += 1
                action = self.epsilon_greedy_policy(state=state, epsilon=epsilon)

                new_observation, reward, done, _ = self.env.step(action)
                new_state = np.reshape(new_observation, [1, observations_count])
                experience = StepExperience(state, action, reward, new_state, done)
                
                self.model.remember(experience)
                self.model.experience_replay()

                episode_reward += reward
                state = new_state

            epsilon = max(epsilon * decay, min_epsilon)

            rewards.append(episode_reward)
            episode += 1

            if statistics.mean(rewards[-100:]) > 195:
                break

        print()
        print("-" * 20 + " Training rewards " + "-" * 20)
        print(rewards)
    
        return rewards

    def run_optimal(self):
        """
        run the agent in the given environment following the policy being calculated
        """
        observations_count = self.env.observation_space.shape[0]
        observation = self.env.reset()
        state = np.reshape(observation, [1, observations_count])

        done = False
        episode_reward = 0

        while not done:
            action = self.model.get_greedy_action_for(state=state)

            new_observation, reward, done, _ = self.env.step(action)
            new_state = np.reshape(new_observation, [1, observations_count])
            self.env.render()
            time.sleep(0.3)

            episode_reward += reward
            state = new_state

        print()
        print(f"Reward following the optimal policy: {episode_reward}")
        self.env.close()


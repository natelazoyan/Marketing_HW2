"""
Multi-Armed Bandit (MAB) Problem Simulation

Overview:
---------
This Python code simulates the Multi-Armed Bandit (MAB) problem, a classic problem in probability theory and statistics. The problem consists of a gambler standing in front of several slot machines (bandits), each with an unknown probability distribution of giving rewards. The gambler's goal is to maximize the total reward obtained over a series of trials. Two algorithms, Epsilon-Greedy and Thompson Sampling, are implemented and compared in this simulation.

Libraries Used:
---------------
- `logging`: Used for logging messages.
- `abc (Abstract Base Classes)`: Used for defining abstract classes and methods.
- `CustomFormatter from logs`: A custom log formatter.
- `random`: Used for generating random numbers.
- `matplotlib.pyplot`: Used for creating plots.
- `numpy`: Used for numerical operations.
- `pandas`: Used for data manipulation and storage.

Classes and Methods:
--------------------
1. `Bandit` (Abstract Class)
   - `__init__(self, p)`: Constructor method initializing bandit parameters.
   - `__repr__(self)`: Abstract method representing the bandit object.
   - `pull(self)`: Abstract method simulating a bandit pull.
   - `update(self)`: Abstract method updating bandit parameters based on pull results.
   - `experiment(self)`: Abstract method simulating a bandit experiment.
   - `report(self)`: Abstract method reporting bandit experiment results.

2. `Visualization` Class
   - `plot_rewards(self, epsilon_greedy_rewards, thompson_rewards)`: Plots cumulative rewards of both algorithms.
   - `plot_regrets(self, epsilon_greedy_rewards, thompson_rewards)`: Plots cumulative regrets of both algorithms.
   - `store_rewards_to_csv(self, epsilon_greedy_rewards, thompson_rewards)`: Stores rewards data in a CSV file.
   - `report_cumulative_reward_and_regret(self, epsilon_greedy_rewards, thompson_rewards)`: Reports cumulative reward and regret for both algorithms.

Usage:
------
Ensure the required libraries are installed and execute the code to observe the MAB algorithms' performance.

Note: Modify Bandit and Visualization classes as per your specific use case if needed.
"""

import logging as log
from abc import ABC, abstractmethod
from logs import CustomFormatter
import random
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

log.basicConfig
logger = log.getLogger("MAB Application")

ch = log.StreamHandler()
ch.setLevel(log.DEBUG)
ch.setFormatter(CustomFormatter())
logger.addHandler(ch)

class Bandit(ABC):
    """Abstract class representing a Multi-Armed Bandit."""
    @abstractmethod
    def __init__(self, p):
        """Constructor method initializing bandit parameters."""
        self.true_means = p
        self.estimated_means = [0.0] * len(p)
        self.action_counts = [0] * len(p)

    @abstractmethod
    def __repr__(self):
        """Abstract method representing the bandit object."""
        pass

    @abstractmethod
    def pull(self):
        """Abstract method simulating a bandit pull."""
        pass

    @abstractmethod
    def update(self):
        """Abstract method updating bandit parameters based on pull results."""
        pass

    @abstractmethod
    def experiment(self):
        """Abstract method simulating a bandit experiment."""
        pass

    @abstractmethod
    def report(self):
        """Abstract method reporting bandit experiment results."""
        pass

class Visualization():
    """Class for visualizing MAB experiment results."""
    def __init__(self):
        """Constructor method initializing visualization parameters."""
        self.cumulative_rewards = {"Epsilon-Greedy": [], "Thompson Sampling": []}
        self.cumulative_regrets = {"Epsilon-Greedy": [], "Thompson Sampling": []}

    def plot_rewards(self, epsilon_greedy_rewards, thompson_rewards):
        """
        Plots cumulative rewards of both Epsilon-Greedy and Thompson Sampling algorithms.

        :param epsilon_greedy_rewards: List of cumulative rewards for Epsilon-Greedy algorithm.
        :param thompson_rewards: List of cumulative rewards for Thompson Sampling algorithm.
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(epsilon_greedy_rewards), label="Epsilon-Greedy")
        plt.plot(np.cumsum(thompson_rewards), label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()

        plt.subplot(1, 2, 2)
        plt.plot(np.log(np.cumsum(epsilon_greedy_rewards)), label="Epsilon-Greedy (log scale)")
        plt.plot(np.log(np.cumsum(thompson_rewards)), label="Thompson Sampling (log scale)")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward (log scale)")
        plt.legend()

        plt.show()

    def plot_regrets(self, epsilon_greedy_rewards, thompson_rewards):
        """
        Plots cumulative regrets of both Epsilon-Greedy and Thompson Sampling algorithms.

        :param epsilon_greedy_rewards: List of cumulative rewards for Epsilon-Greedy algorithm.
        :param thompson_rewards: List of cumulative rewards for Thompson Sampling algorithm.
        """
        plt.figure(figsize=(12, 6))
        plt.subplot(1, 2, 1)
        plt.plot(np.cumsum(epsilon_greedy_rewards), label="Epsilon-Greedy")
        plt.plot(np.cumsum(thompson_rewards), label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Reward")
        plt.legend()

        plt.subplot(1, 2, 2)
        cumulative_regret_epsilon = np.cumsum([max(Bandit_Reward) - r for r in epsilon_greedy_rewards])
        cumulative_regret_thompson = np.cumsum([max(Bandit_Reward) - r for r in thompson_rewards])
        plt.plot(cumulative_regret_epsilon, label="Epsilon-Greedy")
        plt.plot(cumulative_regret_thompson, label="Thompson Sampling")
        plt.xlabel("Trials")
        plt.ylabel("Cumulative Regret")
        plt.legend()

        plt.show()

    def store_rewards_to_csv(self, epsilon_greedy_rewards, thompson_rewards):
        """
        Stores the rewards data from Epsilon-Greedy and Thompson Sampling algorithms into a CSV file.

        :param epsilon_greedy_rewards: List of rewards for Epsilon-Greedy algorithm.
        :param thompson_rewards: List of rewards for Thompson Sampling algorithm.
        """
        data = {
            "Bandit": ["Epsilon-Greedy"] * len(epsilon_greedy_rewards) + ["Thompson Sampling"] * len(thompson_rewards),
            "Reward": epsilon_greedy_rewards + thompson_rewards,
            "Algorithm": ["Epsilon-Greedy"] * len(epsilon_greedy_rewards) + ["Thompson Sampling"] * len(thompson_rewards)
        }
        df = pd.DataFrame(data)
        df.to_csv("bandit_rewards.csv", index=False)

    def report_cumulative_reward_and_regret(self, epsilon_greedy_rewards, thompson_rewards):
        """
        Calculates and prints the cumulative rewards and regrets for Epsilon-Greedy and Thompson Sampling algorithms.

        :param epsilon_greedy_rewards: List of rewards for Epsilon-Greedy algorithm.
        :param thompson_rewards: List of rewards for Thompson Sampling algorithm.
        """
        cumulative_reward_epsilon = np.sum(epsilon_greedy_rewards)
        cumulative_reward_thompson = np.sum(thompson_rewards)
        cumulative_regret_epsilon = np.sum([max(Bandit_Reward) - r for r in epsilon_greedy_rewards])
        cumulative_regret_thompson = np.sum([max(Bandit_Reward) - r for r in thompson_rewards])

        print(f"Epsilon-Greedy Cumulative Reward: {cumulative_reward_epsilon:.2f}")
        print(f"Thompson Sampling Cumulative Reward: {cumulative_reward_thompson:.2f}")
        print(f"Epsilon-Greedy Cumulative Regret: {cumulative_regret_epsilon:.2f}")
        print(f"Thompson Sampling Cumulative Regret: {cumulative_regret_thompson:.2f}")

class EpsilonGreedy(Bandit):
    """
    EpsilonGreedy class implements the epsilon-greedy algorithm for multi-armed bandit problems.
    It explores with probability epsilon and exploits the best known arm with probability 1 - epsilon.
    
    Parameters:
    - true_rewards (list): List of true rewards for each arm.
    - epsilon (float): Exploration-exploitation tradeoff parameter (default is 0.1).
    """

    def __init__(self, true_rewards, epsilon=0.1):
        super().__init__(true_rewards)
        self.epsilon = epsilon
        self.true_rewards = true_rewards
        self.action_counts = [0] * len(true_rewards)
        self.action_values = [0.0] * len(true_rewards)

    def __repr__(self):
        return f"EpsilonGreedy Bandit with epsilon={self.epsilon}"

    def pull(self):
        """
        Selects an arm based on epsilon-greedy strategy.
        """
        if random.random() < self.epsilon:
            return random.randint(0, len(self.true_rewards) - 1)
        else:
            return self.action_values.index(max(self.action_values))

    def update(self, arm, reward):
        """
        Updates the action values based on the reward received from a selected arm.
        
        Parameters:
        - arm (int): Index of the selected arm.
        - reward (float): Reward obtained from the selected arm.
        """
        self.action_counts[arm] += 1
        n = self.action_counts[arm]
        self.action_values[arm] += (1 / n) * (reward - self.action_values[arm])

    def experiment(self, num_trials):
        """
        Runs the epsilon-greedy experiment for a specified number of trials.
        
        Parameters:
        - num_trials (int): Number of trials to run the experiment.
        
        Returns:
        - rewards (list): List of rewards obtained in each trial.
        """
        rewards = []
        for _ in range(num_trials):
            action = self.pull()
            reward = self.true_rewards[action]
            rewards.append(reward)
            self.update(action, reward)
        return rewards

    def report(self, rewards, name):
        """
        Reports the results of the epsilon-greedy experiment.
        
        Parameters:
        - rewards (list): List of rewards obtained in each trial.
        - name (str): Name of the algorithm used for the experiment.
        """
        avg_reward = sum(rewards) / len(rewards)
        avg_regret = max(self.true_rewards) - avg_reward
        print(f"{name} Results: Average Reward={avg_reward:.2f}, Average Regret={avg_regret:.2f}")


class ThompsonSampling(Bandit):
    """
    ThompsonSampling class implements the Thompson Sampling algorithm for multi-armed bandit problems.
    It uses a Bayesian approach to update beliefs about arm probabilities and selects arms based on samples
    from the posterior distribution.
    
    Parameters:
    - p (list): List of true probabilities for each arm.
    """

    def __init__(self, p):
        super().__init__(p)
        self.alpha = [1] * len(p)
        self.beta = [1] * len(p)

    def __repr__(self):
        return "ThompsonSampling Bandit"

    def pull(self):
        """
        Selects an arm using Thompson Sampling strategy.
        """
        sampled_means = [random.betavariate(self.alpha[i], self.beta[i]) for i in range(len(self.true_means))]
        return sampled_means.index(max(sampled_means))

    def update(self, arm, reward):
        """
        Updates the posterior distribution parameters based on the reward received from a selected arm.
        
        Parameters:
        - arm (int): Index of the selected arm.
        - reward (int): Reward obtained from the selected arm (1 for success, 0 for failure).
        """
        if reward == 1:
            self.alpha[arm] += 1
        else:
            self.beta[arm] += 1

    def experiment(self, num_trials):
        """
        Runs the Thompson Sampling experiment for a specified number of trials.
        
        Parameters:
        - num_trials (int): Number of trials to run the experiment.
        
        Returns:
        - rewards (list): List of rewards obtained in each trial.
        """
        rewards = []
        for _ in range(num_trials):
            arm = self.pull()
            reward = self.true_means[arm]
            self.update(arm, reward)
            rewards.append(reward)
        return rewards

    def report(self):
        """
        Reports the results of the Thompson Sampling experiment.
        
        Returns:
        - report (str): Formatted string containing the average reward and regret.
        """
        avg_reward = sum(self.alpha) / (sum(self.alpha) + sum(self.beta))
        avg_regret = max(self.true_means) - avg_reward
        return f"Thompson Sampling Results: Average Reward={avg_reward:.2f}, Average Regret={avg_regret:.2f}"


def comparison(num_trials):
    """
    Compares the performance of EpsilonGreedy and ThompsonSampling algorithms over a specified number of trials.
    
    Parameters:
    - num_trials (int): Number of trials for the comparison experiment.
    """
    epsilon = 0.1
    epsilon_greedy_bandit = EpsilonGreedy(Bandit_Reward, epsilon)
    thompson_bandit = ThompsonSampling(Bandit_Reward)

    epsilon_greedy_rewards = epsilon_greedy_bandit.experiment(num_trials)
    thompson_rewards = thompson_bandit.experiment(num_trials)

    vis = Visualization()  # Assuming Visualization class is defined elsewhere
    vis.plot_rewards(epsilon_greedy_rewards, thompson_rewards)
    vis.plot_regrets(epsilon_greedy_bandit.experiment(num_trials), thompson_bandit.experiment(num_trials))
    vis.store_rewards_to_csv(epsilon_greedy_rewards, thompson_rewards)
    vis.report_cumulative_reward_and_regret(epsilon_greedy_rewards, thompson_rewards)


if __name__=='__main__':
    num_trials = 20000
    comparison(num_trials)

    logger.debug("debug message")
    logger.info("info message")
    logger.warning("warning message")
    logger.error("error message")
    logger.critical("critical message")

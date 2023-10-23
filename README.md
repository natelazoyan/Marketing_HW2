# Multi-Armed Bandit (MAB) Problem Simulation

## Overview

This Python code simulates the Multi-Armed Bandit (MAB) problem, a classic problem in probability theory and statistics. The problem involves a gambler standing in front of several slot machines (bandits), each with an unknown probability distribution of giving rewards. The gambler's objective is to maximize the total reward obtained over a series of trials. In this simulation, two MAB algorithms, Epsilon-Greedy and Thompson Sampling, are implemented and compared for their performance.

## Libraries Used

- `logging`: Used for logging messages.
- `abc (Abstract Base Classes)`: Used for defining abstract classes and methods.
- `CustomFormatter from logs`: A custom log formatter.
- `random`: Used for generating random numbers.
- `matplotlib.pyplot`: Used for creating plots.
- `numpy`: Used for numerical operations.
- `pandas`: Used for data manipulation and storage.

## Classes and Methods

### `Bandit` (Abstract Class)
- `__init__(self, p)`: Constructor method initializing bandit parameters.
- `__repr__(self)`: Abstract method representing the bandit object.
- `pull(self)`: Abstract method simulating a bandit pull.
- `update(self)`: Abstract method updating bandit parameters based on pull results.
- `experiment(self)`: Abstract method simulating a bandit experiment.
- `report(self)`: Abstract method reporting bandit experiment results.

### `Visualization` Class
- `plot_rewards(self, epsilon_greedy_rewards, thompson_rewards)`: Plots cumulative rewards of both algorithms.
- `plot_regrets(self, epsilon_greedy_rewards, thompson_rewards)`: Plots cumulative regrets of both algorithms.
- `store_rewards_to_csv(self, epsilon_greedy_rewards, thompson_rewards)`: Stores rewards data in a CSV file.
- `report_cumulative_reward_and_regret(self, epsilon_greedy_rewards, thompson_rewards)`: Reports cumulative reward and regret for both algorithms.

## Usage

1. Ensure the required libraries are installed in your Python environment.
2. Execute the code to observe the performance of the Epsilon-Greedy and Thompson Sampling algorithms in a Multi-Armed Bandit problem.

**Note**: Modify the `Bandit` and `Visualization` classes as per your specific use case, if needed.

---

The code provided is a comprehensive simulation of the Multi-Armed Bandit problem, offering a comparison of two popular algorithms. It also includes logging and visualization functionalities for better understanding and analysis. You can use this code as a starting point for experimenting with different MAB scenarios or customizing the algorithms to suit your specific needs.

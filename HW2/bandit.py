import numpy as np
import csv
from abc import ABC, abstractmethod
from loguru import logger
import matplotlib.pyplot as plt


class Bandit(ABC):
    """
    Abstract base class for Bandit algorithms.

    Defines the interface and methods any bandit algorithm must implement.
    """

    @abstractmethod
    def __init__(self, p):
        """
        Initialize the bandit with a list of true expected rewards per arm.

        :param p: List[float], true expected reward for each arm.
        """
        pass

    @abstractmethod
    def __repr__(self):
        """
        Return a string representation of the bandit.
        """
        pass

    @abstractmethod
    def pull(self):
        """
        Selects which arm to pull.

        :return: int, index of chosen arm (zero-based).
        """
        pass

    @abstractmethod
    def update(self, chosen_arm, reward):
        """
        Update internal state based on chosen arm and reward.

        :param chosen_arm: int, zero-based index of chosen arm.
        :param reward: float, observed reward.
        """
        pass

    @abstractmethod
    def experiment(self):
        """
        Run the full experiment, iterating over all trials.
        """
        pass

    @abstractmethod
    def report(self):
        """
        Prints cumulative reward, regret, and saves detailed trial info to CSV
        with columns ["Bandit", "Reward", "Algorithm"].
        Bandit is reported as 1-based arm index for human-readability.
        """
        pass


class EpsilonGreedy(Bandit):
    def __init__(self, p, n_trials=20000):
        """
        Epsilon-Greedy bandit with decaying epsilon = 1/t.

        :param p: List[float], true expected reward for arms.
        :param n_trials: Number of trials to run.
        """
        self.p = p
        self.n_arms = len(p)
        self.n_trials = n_trials
        self.counts = np.zeros(self.n_arms)
        self.values = np.zeros(self.n_arms)
        self.epsilon = 1.0
        self.t = 1
        self.rewards = []
        self.chosen_arms = []
        self.cumulative_rewards = []
        self.best_arm = np.argmax(p)
        self.logger = logger
        self.logger.info(f"EpsilonGreedy initialized with arms: {self.p}")

    def __repr__(self):
        """
        Returns:
            str: Description of EpsilonGreedy bandit.
        """
        return f"EpsilonGreedy Bandit with arms {self.p}"

    def pull(self):
        """
        Selects arm to pull, using decayed epsilon for exploration.

        Returns:
            int: Index of chosen arm.
        """
        self.epsilon = 1 / self.t
        if np.random.rand() < self.epsilon:
            chosen_arm = np.random.randint(self.n_arms)
            self.logger.debug(f"Trial {self.t}: Random pull arm {chosen_arm} with epsilon={self.epsilon:.4f}")
        else:
            chosen_arm = np.argmax(self.values)
            self.logger.debug(f"Trial {self.t}: Greedy pull arm {chosen_arm} with epsilon={self.epsilon:.4f}")
        return chosen_arm

    def update(self, chosen_arm, reward):
        """
        Update posterior reward estimate for chosen arm.

        Args:
            chosen_arm (int): Index of arm pulled.
            reward (float): Reward received.
        """
        self.counts[chosen_arm] += 1
        n = self.counts[chosen_arm]
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n
        self.logger.debug(f"Trial {self.t}: Updated arm {chosen_arm} estimate to {self.values[chosen_arm]:.4f}")

    def experiment(self):
        """
        Run epsilon-greedy bandit for configured number of trials.
        """
        self.logger.info("Starting EpsilonGreedy experiment.")
        for self.t in range(1, self.n_trials + 1):
            arm = self.pull()
            reward = self.p[arm]
            self.update(arm, reward)
            self.rewards.append(reward)
            self.chosen_arms.append(arm)
            self.cumulative_rewards.append(sum(self.rewards))
        self.logger.info("EpsilonGreedy experiment finished.")

    def report(self):
        """
        Save experiment results to CSV and print cumulative statistics.

        Saves columns ["Bandit", "Reward", "Algorithm"], with 1-based arm indexing.
        Prints cumulative reward and regret.
        """
        self.logger.info("Reporting EpsilonGreedy results.")
        with open("epsilon_greedy_rewards.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Bandit", "Reward", "Algorithm"])
            for arm, reward in zip(self.chosen_arms, self.rewards):
                writer.writerow([arm + 1, reward, "EpsilonGreedy"])
        cum_reward = sum(self.rewards)
        best_possible = self.n_trials * self.p[self.best_arm]
        regret = best_possible - cum_reward
        print(f"EpsilonGreedy: Cumulative reward: {cum_reward}")
        print(f"EpsilonGreedy: Cumulative regret: {regret}")
        self.logger.info("EpsilonGreedy reporting complete.")


class ThompsonSampling(Bandit):
    def __init__(self, p, n_trials=20000, precision=1.0):
        """
        Thompson Sampling bandit with Gaussian conjugate prior and known precision.

        :param p: List[float], true expected reward for arms.
        :param n_trials: Number of trials to run.
        :param precision: float, known precision (inverse variance) of reward distribution.
        """
        self.p = p
        self.n_arms = len(p)
        self.n_trials = n_trials
        self.precision = precision
        self.prior_mean = np.zeros(self.n_arms)
        self.prior_precision = np.ones(self.n_arms) * 1e-6
        self.post_mean = self.prior_mean.copy()
        self.post_precision = self.prior_precision.copy()
        self.rewards = []
        self.chosen_arms = []
        self.cumulative_rewards = []
        self.best_arm = np.argmax(p)
        self.logger = logger
        self.logger.info(f"ThompsonSampling initialized with arms: {self.p}")

    def __repr__(self):
        """
        Returns:
            str: Description of ThompsonSampling bandit.
        """
        return f"ThompsonSampling Bandit with arms {self.p}"

    def pull(self):
        """
        Samples posterior mean from each arm; chooses arm with highest sample.

        Returns:
            int: Index of best sampled arm.
        """
        sampled_means = np.random.normal(self.post_mean, 1 / np.sqrt(self.post_precision))
        chosen_arm = np.argmax(sampled_means)
        self.logger.debug(f"Sampled means: {sampled_means}, Chosen arm: {chosen_arm}")
        return chosen_arm

    def update(self, chosen_arm, reward):
        """
        Update posterior mean/precision for selected arm.

        Args:
            chosen_arm (int): Index of arm pulled.
            reward (float): Observed reward.
        """
        pm = self.post_mean[chosen_arm]
        pp = self.post_precision[chosen_arm]
        post_prec = pp + self.precision
        post_mean = (pm * pp + self.precision * reward) / post_prec
        self.post_precision[chosen_arm] = post_prec
        self.post_mean[chosen_arm] = post_mean
        self.logger.debug(f"Updated posterior arm {chosen_arm}: mean={post_mean:.4f}, precision={post_prec:.4f}")

    def experiment(self):
        """
        Run ThompsonSampling bandit for configured number of trials.
        """
        self.logger.info("Starting ThompsonSampling experiment.")
        for t in range(1, self.n_trials + 1):
            arm = self.pull()
            reward = self.p[arm]
            self.update(arm, reward)
            self.rewards.append(reward)
            self.chosen_arms.append(arm)
            self.cumulative_rewards.append(sum(self.rewards))
        self.logger.info("ThompsonSampling experiment finished.")

    def report(self):
        """
        Save experiment results to CSV and print cumulative statistics.

        Saves columns ["Bandit", "Reward", "Algorithm"], with 1-based arm indexing.
        Prints cumulative reward and regret.
        """
        self.logger.info("Reporting ThompsonSampling results.")
        with open("thompson_sampling_rewards.csv", "w", newline="") as f:
            writer = csv.writer(f)
            writer.writerow(["Bandit", "Reward", "Algorithm"])
            for arm, reward in zip(self.chosen_arms, self.rewards):
                writer.writerow([arm + 1, reward, "ThompsonSampling"])
        cum_reward = sum(self.rewards)
        best_possible = self.n_trials * self.p[self.best_arm]
        regret = best_possible - cum_reward
        print(f"ThompsonSampling: Cumulative reward: {cum_reward}")
        print(f"ThompsonSampling: Cumulative regret: {regret}")
        self.logger.info("ThompsonSampling reporting complete.")


class Visualization:
    """
    Visualization utilities for comparing multi-armed bandit algorithms.
    """

    def __init__(self, eg_bandit=None, ts_bandit=None):
        """
        Initialize visualization.

        Args:
            eg_bandit (EpsilonGreedy): EpsilonGreedy bandit instance.
            ts_bandit (ThompsonSampling): ThompsonSampling bandit instance.
        """
        self.eg = eg_bandit
        self.ts = ts_bandit

    def plot1(self):
        """
        Plot learning curve and average reward for both algorithms.

        Left: Cumulative reward over trials.
        Right: Average reward per trial over time.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        ax1.plot(self.eg.cumulative_rewards, label='Epsilon Greedy')
        ax1.plot(self.ts.cumulative_rewards, label='Thompson Sampling')
        ax1.set_xlabel('Trials')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('Cumulative Reward Over Trials')
        ax1.legend()

        trials = np.arange(1, self.eg.n_trials + 1)
        avg_eg = np.array(self.eg.cumulative_rewards) / trials
        avg_ts = np.array(self.ts.cumulative_rewards) / trials
        ax2.plot(trials, avg_eg, label='Epsilon Greedy')
        ax2.plot(trials, avg_ts, label='Thompson Sampling')
        ax2.set_xlabel('Trials')
        ax2.set_ylabel('Average Reward')
        ax2.set_title('Average Reward Over Trials')
        ax2.legend()

        plt.tight_layout()
        plt.show()

    def plot2(self):
        """
        Plot cumulative rewards and cumulative regrets for both algorithms.

        Left: Cumulative rewards.
        Right: Cumulative regrets.
        """
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

        trials = np.arange(1, self.eg.n_trials + 1)
        ax1.plot(self.eg.cumulative_rewards, label='Epsilon Greedy')
        ax1.plot(self.ts.cumulative_rewards, label='Thompson Sampling')
        ax1.set_xlabel('Trials')
        ax1.set_ylabel('Cumulative Reward')
        ax1.set_title('Cumulative Rewards')
        ax1.legend()

        best_r = max(self.eg.p)
        eg_regret = best_r * trials - np.array(self.eg.cumulative_rewards)
        ts_regret = best_r * trials - np.array(self.ts.cumulative_rewards)
        ax2.plot(trials, eg_regret, label='Epsilon Greedy')
        ax2.plot(trials, ts_regret, label='Thompson Sampling')
        ax2.set_xlabel('Trials')
        ax2.set_ylabel('Cumulative Regret')
        ax2.set_title('Cumulative Regrets')
        ax2.legend()

        plt.tight_layout()
        plt.show()


def comparison():
    """
    Run both bandits, plot learning curves and cumulative statistics.

    Creates EpsilonGreedy and ThompsonSampling bandits,
    runs experiments, visualizes learning, and saves/report results.
    """
    Bandit_Reward = [1, 2, 3, 4]
    n_trials = 20000

    eg = EpsilonGreedy(Bandit_Reward, n_trials)
    ts = ThompsonSampling(Bandit_Reward, n_trials, precision=1.0)

    eg.experiment()
    ts.experiment()

    vis = Visualization(eg, ts)
    vis.plot1()
    vis.plot2()

    eg.report()
    ts.report()


if __name__ == "__main__":
    logger.debug("Starting bandit algorithms comparison")
    comparison()
    logger.debug("Finished bandit algorithms comparison")

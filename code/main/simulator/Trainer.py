import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import csv
import sys
import os
sys.path.append(os.path.abspath(".")) #Where Simulator.py is located

from Simulator import *
from datetime import datetime

class Trainer(object):

    def __init__(self, env, pilot, starting_point_strategy, e_greedy_strategy, seed):
        self.env = env
        self.pilot = pilot

        e_greedy_strategies = ["decay_most_visited", "constant", "deterministic"]
        assert any(e_greedy_strategy in x for x in e_greedy_strategies),\
            "Expecting e-greedy strategy one of %s. Got %s" % (e_greedy_strategies, e_greedy_strategy)
        self.e_greedy_strategy = e_greedy_strategy

        starting_point_strategies = ["top_left_corner", "corners", "borders"]
        assert any(starting_point_strategy in x for x in starting_point_strategies),\
            "Expecting starting point strategy one of %s. Got %s" % (starting_point_strategies, starting_point_strategy)
        self.starting_point_strategy = starting_point_strategy
        self.rs = np.random.RandomState(seed)

        # Metrics
        self.train_rewards = None
        self.test_rewards = None
        self.action_values = []
        self.episode_rewards = []

    def set_metrics(self, location_action_values, test_frequency=1, gather_train=False):
        """


        :param location_action_values:
        :param test_frequency: Run test on every <test_frequency> train episode. Be careful when choosing "corners" as
                               starting point strategy because then some corners might be missed. Default=1 for to
                               perform test run for every train episode. Will use the same starting point as just
                               finished train episode
        :param gather_train:
        :return:
        """
        self.location_action_values = Point.fromtuple(location_action_values)
        self.gather_train_metrics = gather_train
        self.test_frequency = test_frequency

        if self.starting_point_strategy == "corners":
            self.test_rewards = [[], [], [], []]
            if gather_train:
                self.train_rewards = [[], [], [], []]
        else:
            # We don't care about the starting position
            self.test_rewards = []
            if gather_train:
                self.train_rewards = []

        # State for action values
        # TODO: Adapt for different state definitions
        drone_location_flat = np.zeros(self.env.grid_size).ravel()
        drone_location_flat[location_action_values[1] * self.env.grid_size[0] + location_action_values[0]] = 1
        self.action_values_state = np.append(drone_location_flat, np.zeros(self.env.grid_size).ravel())

    def _evaluate_e_greedy_strategy(self, e_greedy_strategy, e_greedy=None):

        if e_greedy_strategy == "deterministic":
            def get_action(state, global_discovery_map=None, drone_location=None):
                return self.pilot.get_action(state=state)

        elif e_greedy_strategy == "constant":
            def get_action(state, global_discovery_map=None, drone_location=None):
                if self.rs.rand() < e_greedy:
                    return self.rs.randint(0, self.pilot.action_size)
                else:
                    return self.pilot.get_action(state=state)

        elif e_greedy_strategy == "decay_most_visited":
            def get_action(state, global_discovery_map, drone_location):
                e_greedy_decayed = e_greedy * (num_episodes / (num_episodes + global_discovery_map[drone_location]))
                if self.rs.rand() < e_greedy_decayed:
                    return self.rs.randint(0, self.pilot.action_size)
                else:
                    return self.pilot.get_action(state=state)
        else:
            raise ValueError("Specified e-greedy strategy not implemented <%s>" % e_greedy_strategy)

        return get_action

    def _evaluate_starting_point_strategy(self):
        # starting points

        if self.starting_point_strategy == "top_left_corner":
            def get_next_starting_point():
                return Point(0, 0)

        elif self.starting_point_strategy == "corners":
            # Iterate over the four corners sequentially
            def get_next_starting_point():
                size = self.env.grid.size
                points = [Point(0, 0), Point(0, size[0]), Point(size[1], 0), Point(size[1], size[0])]
                i = 0
                while True:
                    i += 1
                    yield points[i % len(points)]

        elif self.starting_point_strategy == "borders":
            # Start at a random border point
            def get_next_starting_point():
                border_side = self.rs.randint(0, 3)
                size = self.env.grid.size
                if border_side == 0:
                    return Point(0, self.rs.randint(0, size[1]))
                elif border_side == 1:
                    return Point(size[0], self.rs.randint(0, size[1]))
                elif border_side == 2:
                    return Point(0, self.rs.randint(0, size[0]))
                else:
                    return Point(size[1], self.rs.randint(0, size[0]))

        return get_next_starting_point

    def train(self, num_episodes, max_steps_per_episode, e_greedy,
              num_obstacles, target_seeds, obstacle_seeds):
        """


        :param num_episodes:
        :param max_steps_per_episode:
        :param e_greedy_strategy:
        :param e_greedy: value between 0 and 1 describing the probability of choosing a random action based on the strategy
        :param starting_point_strategy:
        _param num_runs:
        :return:
        """

        self.max_steps_per_episode = max_steps_per_episode
        self.num_obstacles = num_obstacles
        self.target_seeds = target_seeds
        self.obstacle_seeds = obstacle_seeds

        self.global_discovery_map = np.zeros(self.env.grid_size, dtype=np.int64).ravel()

        get_action = self._evaluate_e_greedy_strategy(self.e_greedy_strategy, e_greedy)
        get_next_starting_point = self._evaluate_starting_point_strategy()

        # Need for test
        get_deterministic_action = self._evaluate_e_greedy_strategy(e_greedy_strategy="deterministic")

        # For now just one drone
        id = self.env.drone_id

        for episode in range(num_episodes):

            if episode % 100 == 0:
                print("Episode", episode)

            starting_point = get_next_starting_point()

            # Train
            self.run_episode(starting_point, get_action, id)

            self.global_discovery_map += self.env.grid.discovery_map

            if self.gather_train_metrics:
                self.gather_metrics(episode, is_train=True)

            # Test
            if episode % self.test_frequency == 0:
                self.run_episode(starting_point, get_deterministic_action, id, is_train=False)
                self.gather_metrics(episode, is_train=False)

        return self.test_rewards, self.train_rewards, self.action_values

    def run_episode(self, starting_point, get_action, id, is_train=True):

        self.env.build_world(initial_position=starting_point,
                             num_obstacles=self.num_obstacles,
                             target_seed=self.target_seeds[0],
                             obstacles_seed=self.obstacle_seeds[0])

        state = self.env.get_state()

        for step in range(self.max_steps_per_episode):

            action_idx = get_action(state=state,
                                    global_discovery_map=self.global_discovery_map,
                                    drone_location=self.env.drone.get_position_flat())

            next_state, reward, done = self.env.step(action_idx)

            self.pilot.store_step(state=state,
                                  action_idx=action_idx,
                                  reward=reward,
                                  done=done,
                                  next_state=next_state)

            if not is_train:
                self.store_reward(reward)
            else:
                if self.gather_train_metrics:
                    self.store_reward(reward)

            if done:
                break

            state = next_state.copy()

        if is_train:
            self.pilot.learn()


    def store_reward(self, reward):
        self.episode_rewards.append(reward)

    def gather_metrics(self, episode, is_train=False):
        # store previous rewards and reset buffer
        if self.starting_point_strategy == "corners":
            if is_train:
                self.train_rewards[episode % 4].append(np.sum(self.episode_rewards))
            else:
                self.test_rewards[episode % 4].append(np.sum(self.episode_rewards))
        else:
            if is_train:
                self.train_rewards.append(np.sum(self.episode_rewards))
            else:
                self.test_rewards.append(np.sum(self.episode_rewards))
        self.episode_rewards = []

        # Store action values for test only
        if not is_train:
            self.action_values.append(self.pilot.get_action_values(self.action_values_state))

    def grid_search(self, runs, hp_ranges):

        file_name = datetime.now().strftime("%y%m%d-%H%M%S") + "/grid_search_result.csv"

        if not os.path.exists(os.path.dirname(file_name)):
                os.makedirs(os.path.dirname(file_name))

        with open(file_name, "w") as file:
            file.write("type,hyperparameters,values\n")



        labels = np.array(["r_test_mean", "r_test_std", "a_up_mean", "a_right_mean", "a_down_mean", "a_left_mean",
                           "a_up_std", "a_right_std", "a_down_std", "a_left_std"]).reshape(-1, 1)

        entropy_factors = hp_ranges["entropy_factors"]
        critic_factors = hp_ranges["critic_factors"]
        reward_decays = hp_ranges["reward_decays"]

        for entropy_factor in entropy_factors:
            self.pilot.hp["entropy_factor"] = entropy_factor
            for critic_factor in critic_factors:
                self.pilot.hp["critic_factor"] = critic_factor
                for reward_decay in reward_decays:
                    self.pilot.hp["reward_decay"] = reward_decay

                    test_rewards_runs = []
                    action_values_runs = []

                    hyperparameters = json.dumps(self.pilot.hp).replace('"', "'")  # use `json.loads` to do the reverse
                    print("Current hyperparameters", hyperparameters)

                    try:

                        for run in range(runs):

                            test_rewards, train_rewards, action_values = self.pilot.run()
                            test_rewards_runs.append(test_rewards)
                            action_values_runs.append(action_values)

                            self.pilot.reset()

                        test_rewards_mean = np.array(test_rewards_runs).mean(axis=0)
                        test_rewards_std = np.array(test_rewards_runs).std(axis=0)
                        action_values_mean = np.squeeze(np.array(action_values_runs)).mean(axis=0)
                        action_values_std = np.squeeze(np.array(action_values_runs)).std(axis=0)

                        data = np.vstack((test_rewards_mean, test_rewards_std, action_values_mean.T, action_values_std.T))
                        data = np.round(data, 4)

                        hyperparameters = np.array([hyperparameters for x in range(data.shape[0])]).reshape(-1, 1)

                        data = np.hstack((labels, hyperparameters, data))

                        with open(file_name, "a") as csvfile:
                            filewriter = csv.writer(csvfile)
                            filewriter.writerows(data)

                    except Exception as e:
                        data = "Exception occurred: %s with hyperparameters %s" % (e, hyperparameters)
                        with open(file_name, "a") as csvfile:
                            csvfile.write(data)

    @staticmethod
    def plot_data(csv_file):
        data = pd.read_csv(csv_file, quotechar='"', skiprows=1, header=None).values

        def rolling(data, window):
            return np.convolve(data, np.ones((window,)) / window, mode="valid")

        labels = data[:, 0]
        hyperparams = data[:, 1]
        data = data[:, 2:].astype(np.float64)

        window = 700
        for i in range(0, data.shape[0], 10):
            mean = rolling(data[i, :], window)
            std = rolling(data[i+1, :], window)
            x = range(len(mean))

            hp = json.loads(hyperparams[i].replace("'", '"'))

            plt.plot(x, mean, label="test_reward (rolling mean")
            plt.fill_between(x, mean - std, mean + std, color='#888888', alpha=0.4)
            plt.title("entropy_factor: %s, critic_factor %s, reward_decay %s" % (hp["entropy_factor"], hp["critic_factor"], hp["reward_decay"]))
            plt.legend()
            plt.xlabel("Episodes")
            plt.ylabel("test reward")
            plt.show()



if __name__ == "__main__":

    Trainer.plot_data("../pilots/grid_search_result.csv")

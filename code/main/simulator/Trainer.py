import numpy as np

import sys
import os
sys.path.append(os.path.abspath(".")) #Where Simulator.py is located

from Simulator import *

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
        self.global_discovery_map = np.zeros(self.env.grid_size, dtype=np.int8)
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
        self.location_action_values = location_action_values
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

        get_action = self._evaluate_e_greedy_strategy(self.e_greedy_strategy, e_greedy)
        get_next_starting_point = self._evaluate_starting_point_strategy()

        # Need for test
        get_deterministic_action = self._evaluate_e_greedy_strategy(e_greedy_strategy="deterministic")

        # For now just one drone
        id = self.env.drone_id

        for episode in range(num_episodes):

            starting_point = get_next_starting_point()

            # Train
            self.run_episode(starting_point, get_action, id)

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

            self.global_discovery_map += self.env.drone.get_position_flat()

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

    def gather_metrics(self, episode, state=None, is_train=False):
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
            self.action_values.append(self.pilot.get_action_values(state))

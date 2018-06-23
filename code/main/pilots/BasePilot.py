import numpy as np
import sys
import os
sys.path.append(os.path.abspath("../simulator"))  # Where Trainer.py is located

from Trainer import *
from Simulator import *


class BasePilot(object):
    """
    Base class for all pilots.

    variable <hp> represents the hyperparameters. Following are the options

    * drone_id:                                 int
    * grid_size:                                tuple, (int: len y-axis, int: len x-axis)
    * state_include_drone_location:             bool
    * state_include_discovery_map:              bool
    * state_include_drone_observed_obstacles:   bool
    * trainer_seed:                             int
    * metric_location_action_values:            tuple, (int: x, int: y)
    * metric_gather_train:                      bool
    * num_episodes:                             int
    * max_steps_per_episode:                    int, after how many steps episode should be forced to end
    * e_greedy_strategy:                        string, one of "decay_most_visited", "constant" or "deterministic"
    * e_greedy:                                 float, 1 for complete random action, 0 for deterministic action
    * starting_point_strategy:                  string, one of "top_left_corner", "corners" or "borders"
    * num_obstacles:                            int
    * target_seeds:                             list
    * obstacle_seeds:                           list

    """

    hp = {
        "drone_id": 99,
        "grid_size": (5, 5),
        "state_include_drone_location": True,
        "state_include_discovery_map": True,
        "state_include_drone_observed_obstacles": False,
        "trainer_seed": 1,
        "metric_location_action_values": (0, 0),
        "metric_gather_train": True,
        "num_episodes": 2,
        "max_steps_per_episode": 2,
        "e_greedy_strategy": "deterministic",
        "e_greedy": 0.6,
        "starting_point_strategy": "top_left_corner",
        "num_obstacles": 0,
        "target_seeds": [2],
        "obstacle_seeds": [2]
    }

    action_size = 4

    def get_action(self, state):
        """
        Returns the index of the optimal action for the given state.

        :param state: Flat numpy array. The state for which the optimal action should be determined
        :return: Integer. The action index of the optimal action
        """
        return int(np.argmax(self.get_action_values(state)))

    def get_action_values(self, state):
        """
        Returns the policy, or in other names, the full output of the policy for the given state.
        This are action values for all possible actions.

        :param state: Flat numpy array. The state for which the action values should be computed
        :return: Flat numpy array of action size
        """

        raise NotImplementedError("Using method of base class, please implement custom method!")

    def store_step(self, state, action_idx, reward, done, next_state):
        """
        Callback for the trainer to store the given values after each step. The handling across episodes (e.g.
        Experience Replay) must be handled by pilot itself.

        :param state: Flat numpy array. The initial state of current step
        :param action_idx: int. The index of action performed in current step
        :param reward: float. The reward obtained by drone <id> performing action <action_idx>
        :param done: bool. Flag to indicate whether the simulator entered exit state after performing action
        :param next_state: Flat numpy array. The next state after performed action <action_idx>. Will be equal to
                           <state> of next call if episode did not finish yet.
        """

        raise NotImplementedError("Using method of base class, please implement custom method!")

    def learn(self):
        """
        Called by trainer after each training episode.

        :return:
        """

        raise NotImplementedError("Using method of base class, please implement custom method!")

    @staticmethod
    def reward_fkt(drone, move_direction, discovery_map, step_num):
        """
        The reward function to be used by the simulator.

        :param drone:
        :param move_direction:
        :param discovery_map:
        :param step_num:
        :return:
        """

        raise NotImplementedError("Using method of base class, please implement custom method!")

    def reset(self, hp):
        """
        Resets the pilot and updates hyperparameters

        :param hp:
        :return:
        """

        raise NotImplementedError("Using method of base class, please implement custom method!")

    def run(self):
        pilot = self
        hp = pilot.hp

        # Setup environment
        self.env = Simulator(drone_id=hp["drone_id"],
                        grid_size=hp["grid_size"])
        self.env.set_reward_function(pilot.reward_fkt)
        self.env.define_state(drone_location=hp["state_include_drone_location"],
                         discovery_map=hp["state_include_discovery_map"],
                         drone_observed_obstacles=hp["state_include_drone_observed_obstacles"])

        self.trainer = Trainer(self.env, pilot,
                          starting_point_strategy=hp["starting_point_strategy"],
                          e_greedy_strategy=hp["e_greedy_strategy"],
                          seed=hp["trainer_seed"])

        self.trainer.set_metrics(location_action_values=hp["metric_location_action_values"],
                            gather_train=hp["metric_gather_train"])

        return self.trainer.train(
            num_episodes=hp["num_episodes"],
            max_steps_per_episode=hp["max_steps_per_episode"],
            e_greedy=hp["e_greedy"],
            num_obstacles=hp["num_obstacles"],
            target_seeds=hp["target_seeds"],
            obstacle_seeds=hp["obstacle_seeds"])


if __name__ == "__main__":
    pilot = BasePilot()
    pilot.run()

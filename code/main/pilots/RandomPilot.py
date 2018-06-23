import numpy as np
import sys
import os
sys.path.append(os.path.abspath("../simulator"))  # Where Trainer.py is located

from Trainer import *
from Simulator import *

from BasePilot import *


class RandomPilot(BasePilot):

    hp = {
        "drone_id": 99,
        "grid_size": (5, 5),
        "state_include_drone_location": True,
        "state_include_discovery_map": True,
        "state_include_drone_observed_obstacles": False,
        "trainer_seed": 1,
        "metric_location_action_values": (0, 0),
        "metric_gather_train": True,
        "num_episodes": 5,
        "max_steps_per_episode": 5,
        "e_greedy_strategy": "deterministic",
        "e_greedy": 0.6,
        "starting_point_strategy": "top_left_corner",
        "num_obstacles": 0,
        "target_seeds": [2],
        "obstacle_seeds": [2]
    }

    action_size = 4

    def get_action_values(self, state):
        return np.random.rand(self.action_size)

    def store_step(self, state, action_idx, reward, done, next_state):
        pass

    def learn(self):
        pass

    def reset(self, hp):
        for k, v in hp.items():
            self.hp[k] = v

    @staticmethod
    def reward_fkt(drone, move_direction, discovery_map, step_num):
        return 1, False


if __name__ == "__main__":
    pilot = RandomPilot()
    pilot.run()

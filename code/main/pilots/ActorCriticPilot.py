import numpy as np
import tensorflow as tf
import json
import sys
import os
sys.path.append(os.path.abspath("../simulator"))  # Where Trainer.py is located

os.environ['TF_CPP_MIN_LOG_LEVEL'] = "3"  # Shut up tensorflow warnings

from Trainer import *
from Simulator import *

from BasePilot import *


class ActorCriticPilot(BasePilot):

    hp = {
        "drone_id": 99,
        "grid_size": (8, 8),
        "state_include_drone_location": True,
        "state_include_discovery_map": True,
        "state_include_drone_observed_obstacles": False,
        "trainer_seed": 1,
        "metric_location_action_values": (0, 0),
        "metric_gather_train": True,
        "num_episodes": 5000,
        "max_steps_per_episode": 15,
        "e_greedy_strategy": "deterministic",
        "e_greedy": 0.6,
        "starting_point_strategy": "top_left_corner",
        "num_obstacles": 0,
        "target_seeds": [4],
        "obstacle_seeds": [2],
        "learning_rate":  1e-5,
        "entropy_factor": 0.1,
        "critic_factor": 0.1,
        "reward_decay": 0.99
    }

    action_size = 4

    def __init__(self):
        self.n_x = self.hp["grid_size"][0] * self.hp["grid_size"][1] * 2
        self.n_y = self.action_size

        self.episode_states = np.empty((0, self.n_x), np.float32)
        self.episode_actions = []
        self.episode_rewards = []
        self.done = False
        self.last_state = None

        self.build_network()

        self.sess = tf.Session()
        self.sess.run((tf.global_variables_initializer(), tf.local_variables_initializer()))

    def build_network(self):
        self.X = tf.placeholder(tf.float32, shape=[None, self.n_x], name="X")
        self.R = tf.placeholder(tf.float32, shape=[None, 1], name="discounted_rewards")
        self.A = tf.placeholder(tf.int32, shape=[None], name="actions")
        self.AVG = tf.placeholder(tf.float32, shape=[None, 1], name="advantages")

        with tf.name_scope("policy_network"):
            policy_nn_raw = self.policy_nn()
            self.policy_network = tf.nn.softmax(policy_nn_raw)

        print("Shape policy", self.policy_network)

        with tf.name_scope("critic_network"):
            self.critic_network = self.critic_nn()

        with tf.name_scope("entropy"):
            # We wanna maximize the entropy (then all outputs are even)
            entropy = -tf.reduce_sum(self.policy_network * tf.log(self.policy_network))

        with tf.name_scope("policy_loss"):
            # Touch only the action we took
            action_mask_one_hot = tf.one_hot(self.A,  # column index
                                             self.policy_network.shape[1],
                                             on_value=True,
                                             off_value=False,
                                             dtype=tf.bool)

            policy = -tf.boolean_mask(self.policy_network, action_mask_one_hot)

            # policy * advantage + entropy
            self.policy_loss = tf.reduce_sum(tf.multiply(policy, self.AVG)) - self.hp["entropy_factor"] * entropy

        with tf.name_scope("critic_loss"):
            self.critic_loss = tf.reduce_sum(
                tf.squared_difference(self.R, self.critic_network) + 0.1 * tf.square(self.critic_network))

        self.loss = self.policy_loss + self.hp["critic_factor"] * self.critic_loss


        trainer = tf.train.AdamOptimizer(self.hp["learning_rate"])
        self.train_ops = trainer.minimize(self.loss)

    ### Network contruction functions ###
    @staticmethod
    def dense(input, kernel_shape, activation=True):
        # Have to use get_variable to have the scope adjusted correctly
        kernel = tf.get_variable("kernel", kernel_shape, initializer=tf.variance_scaling_initializer(scale=2.0,
                                                                                                     mode="fan_avg",
                                                                                                     distribution="uniform"),
                                 trainable=True)
        biases = tf.get_variable("biases", kernel_shape[1], initializer=tf.constant_initializer(0.1))
        raw_dense = tf.matmul(input, kernel) + biases
        act = tf.nn.relu(raw_dense)
        #tf.summary.histogram("weights", kernel)
        #tf.summary.histogram("biases", biases)
        #tf.summary.histogram("activations", act)

        return act

    @staticmethod
    def dense_output(input, kernel_shape):
        # Have to use get_variable to have the scope adjusted correctly
        kernel = tf.get_variable("kernel", kernel_shape, initializer=tf.variance_scaling_initializer(scale=1.0,
                                                                                                     mode="fan_avg",
                                                                                                     distribution="uniform"))
        biases = tf.get_variable("biases", kernel_shape[1], initializer=tf.constant_initializer(-0.1))
        raw_dense = tf.matmul(input, kernel) + biases
        #tf.summary.histogram("weights", kernel)
        #tf.summary.histogram("biases", biases)

        return raw_dense

    def policy_nn(self):
        with tf.variable_scope("hidden1", reuse=tf.AUTO_REUSE):
            in_hidden1 = self.n_x
            out_hidden1 = self.n_x
            hidden1 = self.dense(self.X, [in_hidden1, out_hidden1])
        with tf.variable_scope("hidden2", reuse=tf.AUTO_REUSE):
            in_hidden2 = self.n_x
            out_hidden2 = int(self.n_x / 2)
            hidden2 = self.dense(hidden1, [in_hidden2, out_hidden2])
        with tf.variable_scope("action_raw"):
            out_policy = 4
            policy_raw = self.dense_output(hidden2, [out_hidden2, out_policy])
        return policy_raw

    def get_policy(self, input):
        policies = self.sess.run(self.policy_network, feed_dict={self.X: input.reshape(1, -1)})
        if np.isnan(policies).any():
            raise ValueError("NaN occured in policy network")
        return policies

    def critic_nn(self):
        with tf.variable_scope("hidden1", reuse=tf.AUTO_REUSE):
            in_hidden1 = self.n_x
            out_hidden1 = self.n_x
            hidden1 = self.dense(self.X, [in_hidden1, out_hidden1])
        with tf.variable_scope("hidden2", reuse=tf.AUTO_REUSE):
            in_hidden2 = self.n_x
            out_hidden2 = int(self.n_x / 2)
            hidden2 = self.dense(hidden1, [in_hidden2, out_hidden2])
        with tf.variable_scope("critic_raw"):
            out_critic = 1
            critic_raw = self.dense_output(hidden2, [out_hidden2, out_critic])
        return critic_raw

    def get_critic(self, input):
        critic_value = self.sess.run(self.critic_network, feed_dict={self.X: input})
        if np.isnan(critic_value).any():
            raise ValueError("NaN occured in policy network")

        return critic_value

    # Override
    def get_action_values(self, state):
        return self.get_policy(state)

    # Override
    def store_step(self, state, action_idx, reward, done, next_state):
        self.episode_states = np.vstack((self.episode_states, state))
        self.episode_rewards.append(reward)
        self.episode_actions.append(action_idx)
        self.last_state = next_state

        if done:
            self.done = True

    # Override
    def learn(self):

        # Final reference reward
        if self.done:
            discounted_reward = 0
        else:
            discounted_reward = self.get_critic(self.last_state.reshape(1, -1))

        episode_count = len(self.episode_rewards)

        discounted_rewards = []

        for i in reversed(range(episode_count)):
            discounted_reward = self.episode_rewards[i] + self.hp["reward_decay"] * discounted_reward
            discounted_rewards.append(discounted_reward)

        discounted_rewards = np.array(list(reversed(discounted_rewards))).ravel().reshape(-1, 1)

        values = self.get_critic(self.episode_states)

        advantages = discounted_rewards - values  # TODO increase value of critic with time

        feed_dict = {self.X: self.episode_states,
                     self.A: self.episode_actions,
                     self.R: discounted_rewards,
                     self.AVG: advantages}

        self.sess.run(self.train_ops, feed_dict=feed_dict)

        # Reset the episode data, clear memory
        self.episode_states = np.empty((0, self.n_x), np.float32)
        self.episode_actions = []
        self.episode_rewards = []

    # Override
    def reset(self, hp):
        tf.reset_default_graph()

        for k, v in hp.items():
            self.hp[k] = v

    # Override
    def reward_fkt(self, drone, move_direction, discovery_map, step_num):
        movement_cost = (step_num + 1) / (self.hp["max_steps_per_episode"] * 10)

        """Move the drone and get the reward."""
        try:
            drone.move(move_direction)
            if "T" in drone.observe_surrounding():
                # arbitrary reward for finding target
                return 1 - movement_cost, True
            else:
                # if a drone has been to a location multiple times,
                # the penalty will increase linearly with each visit
                return -discovery_map[drone.get_position_flat()] / (self.hp["max_steps_per_episode"] * 5) - movement_cost, False
        except (PositioningError, IndexError):
            # We hit an obstacle or tried to exit the grid
            return -discovery_map[drone.get_position_flat()] / (self.hp["max_steps_per_episode"] * 5) - movement_cost - 0.4, False

    # Override
    def run(self):
        test_rewards, train_rewards, action_values = super().run()

        labels = "test_rewards,action_value_up,action_value_right,action_value_down,action_value_left\r\n"
        hyperparameters = json.dumps(self.hp)  # use `json.loads` to do the reverse

        with open("hyperparameters.txt", "w") as file:
            file.write(hyperparameters)

        with open("result.csv", "w") as file:
            file.write(labels)
            for i in range(len(test_rewards)):
                line = str(test_rewards[i]) + "," + ",".join(str(x) for x in action_values[i][0])
                file.write(line + "\r\n")

        print(self.env.grid)

        print(self.trainer.global_discovery_map.reshape(self.hp["grid_size"]))





if __name__ == "__main__":
    pilot = ActorCriticPilot()
    pilot.run()

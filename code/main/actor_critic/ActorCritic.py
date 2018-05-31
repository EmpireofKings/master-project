"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt

from datetime import datetime



class ActorCritic:
    def __init__(self,
                 input_size,
                 output_size,
                 learning_rate,
                 reward_decay):
        self.n_x = input_size
        self.n_y = output_size
        self.lr = learning_rate
        self.discount_factor = reward_decay

        self.episode_states = np.empty((0,input_size), np.float32)
        self.episode_actions = []
        self.episode_rewards = []

        self.cost_history = []

        self.sess = tf.Session()

        modifier = datetime.now().strftime("%m%d-%H%M%S") + " lnr=,disc=%s" % (self.discount_factor)
        log_path = "_logs_/" + modifier

        # $ tensorboard --logdir=_logs_
        self.writer = tf.summary.FileWriter(log_path)

        #self.build_network()
        self.trainer = tf.train.AdamOptimizer(self.lr)

        self.X = tf.placeholder(tf.float32, shape=[None, self.n_x], name="X")
        self.R = tf.placeholder(tf.float32, shape=[None, 1], name="discounted_rewards")
        self.A = tf.placeholder(tf.float32, shape=[None, 4], name="actions")

        with tf.variable_scope("policy_network", reuse=tf.AUTO_REUSE):
            self.policy_network = tf.nn.softmax(self.policy_nn())

            tf.summary.scalar("direction_up", self.policy_network[1][0])
            tf.summary.scalar("direction_right", self.policy_network[1][1])
            tf.summary.scalar("direction_down", self.policy_network[1][2])
            tf.summary.scalar("direction_left", self.policy_network[1][3])

        with tf.variable_scope("critic_network", reuse=tf.AUTO_REUSE):
            self.critic_network = self.critic_nn()

        with tf.name_scope("entropy"):
            entropy = tf.reduce_sum(self.policy_network * tf.log(self.policy_network))

        # Touch only the action we took
        with tf.name_scope("policy_loss"):
            policy = tf.reduce_sum(tf.multiply(self.policy_network, self.A))
            self.policy_loss = policy*(self.R - self.critic_network) + 0.2 * entropy

            tf.summary.scalar("policy_loss", tf.reduce_sum(self.policy_loss))

        with tf.name_scope("critic_loss"):
            self.critic_loss = (self.R - self.critic_network)**2

            tf.summary.scalar("critic_loss", tf.reduce_sum(self.critic_loss))

        self.train_ops = (self.trainer.minimize(self.policy_loss),
                          self.trainer.minimize(self.critic_loss))

        self.sess.run(tf.global_variables_initializer())
        self.writer.add_graph(self.sess.graph)

        self.iterations = 0


    def store_transition(self, s, a, r):
        """
            Store play memory for training

            Arguments:
                s: state
                a: index of action taken
                r: reward after action
        """
        self.episode_states = np.vstack((self.episode_states, s))
        self.episode_rewards.append(r)

        # Store actions as list of arrays, hot-one encoding at each transition
        action = np.zeros(self.n_y, dtype=np.float32)
        action[a] = 1
        self.episode_actions.append(action)


    def choose_action(self, observation):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (,num_features)

            Returns: index of action we want to choose
        """

        prob_weights = self.get_policy(observation)

        return np.argmax(prob_weights)



    def learn(self, last_state, was_target_found):

        # Final reference reward
        if was_target_found:
            discounted_reward = 0
        else:
            discounted_reward = self.get_critic(last_state)

        #print(discounted_reward)

        discounted_rewards = []
        for i in reversed(range(len(self.episode_rewards))):
            discounted_reward = self.episode_rewards[i] + self.discount_factor * discounted_reward
            discounted_rewards.append(discounted_reward)
        discounted_rewards = np.array(list(reversed(discounted_rewards))).ravel().reshape(-1, 1)

        #print(self.episode_rewards)
        #print(discounted_rewards.ravel())

        feed_dict = {self.X: self.episode_states,
                     self.A: self.episode_actions,
                     self.R: discounted_rewards}


        #var_policy = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy_network")
        #var_critic = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="critic_network")

        #action_gradients = tf.gradients(self.policy_loss, var_policy)
        #value_gradients = tf.gradients(self.critic_loss, var_critic)

        # Does not work with Adam optimizer!
        #self.sess.run(self.trainer.apply_gradients(zip(action_gradients, var_policy)), feed_dict=feed_dict)
        #self.sess.run(self.trainer.apply_gradients(zip(value_gradients, var_critic)), feed_dict=feed_dict)



        # Will collect all summaries for tensorboard
        summ = tf.summary.merge_all()

        _, _, _, s = self.sess.run((self.train_ops,
                                    self.policy_loss,
                                    self.critic_loss,
                                    summ), feed_dict=feed_dict)

        self.writer.add_summary(s, self.iterations)

        self.iterations += 1


    def dense(self, input, kernel_shape):
        # Have to use get_variable to have the scope adjusted correctly
        kernel = tf.get_variable("kernel", kernel_shape, initializer=tf.random_normal_initializer(0.1, 0.2))
        biases = tf.get_variable("biases", kernel_shape[1], initializer=tf.constant_initializer(0.1))
        act = tf.nn.relu(tf.matmul(input, kernel) + biases)

        if not tf.is_variable_initialized(kernel).eval(session=self.sess):
            self.sess.run(tf.variables_initializer([kernel]))
        if not tf.is_variable_initialized(biases).eval(session=self.sess):
            self.sess.run(tf.variables_initializer([biases]))

        tf.summary.histogram("weights", kernel)
        tf.summary.histogram("biases", biases)
        tf.summary.histogram("activations", act)

        return act

    def policy_nn(self):
        with tf.variable_scope("hidden1"):
            in_hidden1 = self.n_x
            out_hidden1 = 30
            hidden1 = self.dense(self.X, [in_hidden1, out_hidden1])
        with tf.variable_scope("action_raw"):
            out_policy = 4
            policy_raw = self.dense(hidden1, [out_hidden1, out_policy])
        return policy_raw

    def get_policy(self, input):
        return self.sess.run(self.policy_network, feed_dict={self.X: input.reshape(1, -1)})

    def critic_nn(self):
        with tf.variable_scope("hidden1"):
            in_hidden1 = self.n_x
            out_hidden1 = 30
            hidden1 = self.dense(self.X, [in_hidden1, out_hidden1])
        with tf.variable_scope("critic_raw"):
            out_critic = 1
            critic_raw = self.dense(hidden1, [out_hidden1, out_critic])
        return critic_raw

    def get_critic(self, input):
        return self.sess.run(self.critic_network, feed_dict={self.X: input.reshape(1, -1)})





    def reset(self):
        # Reset the episode data, clear memory
        self.episode_states = np.empty((0,self.n_x), np.float32)
        self.episode_actions = []
        self.episode_rewards  = []


    def build_network(self):


        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_x], name="X")
            self.actions_taken = tf.placeholder(tf.float32,
                                                shape=[None, self.n_y],
                                                name="actions_taken")
            self.advantages = tf.placeholder(tf.float32,
                                             shape=[None, 1],
                                             name="advantages")
            self.true_values = tf.placeholder(tf.float32,
                                              shape=[None, 1],
                                              name="true_values")

        # First hidden layer
        self.layer_1 = tf.layers.Dense(units=30,  # num output nodes
                                  activation=tf.nn.relu,
                                  trainable=True,
                                  name="layer_1")

        # Raw action output layer, therefore no activation function
        # Also called logits sometimes, don't know why
        self.layer_actions = tf.layers.Dense(units=self.n_y,  # num output nodes
                                             trainable=True,
                                             name="layer_actions")

        self.action_NN = tf.nn.softmax(self.layer_actions(self.layer_1(self.X)), name="layer_softmax_out")

        # Raw critic output layer, therefore no activation function
        self.layer_critic = tf.layers.Dense(units=1,
                                            trainable=True,
                                            name="layer_critic")

        self.critic_NN = self.layer_critic(self.layer_1(self.X))

        ## Policy loss
        # Element-wise multiplication, to touch only values for which we actually took an action
        # Note that the discounded rewards are normalized between 0 and 1 in order to fit to the softmax
        #    output of the network
        # -> actions: [0, 1, 0, 0] * 0.1 => [0, 0.1, 0, 0]
        # If the softmmax outup is [0, 0.9, 0, 0] but we have a low (negative before normalization) reward
        #   this value (0.9) will be decreased
        action_truth = tf.multiply(self.actions_taken, self.advantages)

        # sum((y_hat-y_true)**2)
        # Touch only values for which we actually took an action
        action_prediction = tf.multiply(self.actions_taken, self.action_NN)

        self.policy_loss = tf.reduce_sum((action_prediction - action_truth)**2)

        ## Value (Critic) loss
        value_prediction = self.critic_NN
        self.value_loss = tf.reduce_sum((value_prediction - self.true_values)**2)

        self.loss = self.value_loss + self.policy_loss



    def print_value_estimation(self):
        values = self.sess.run(self.critic_NN, feed_dict={self.X: self.episode_states})
        print("values of the critic:\n", values)

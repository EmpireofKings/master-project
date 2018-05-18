"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network

"""
import tensorflow as tf
import numpy as np
from tensorflow.python.framework import ops


class PolicyGradient:
    def __init__(
        self,
        n_x,
        n_y,
        learning_rate,
        reward_decay
#        load_path=None,
#        save_path=None
    ):

        self.n_x = n_x
        self.n_y = n_y
        self.lr = learning_rate
        self.gamma = reward_decay

        self.episode_states, self.episode_actions, self.episode_rewards = [], [], []

        self.build_network()

        self.cost_history = []

        self.sess = tf.Session()

        # $ tensorboard --logdir=logs
        # http://0.0.0.0:6006/
        tf.summary.FileWriter("logs/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

    def store_transition(self, s, a, r):
        """
            Store play memory for training

            Arguments:
                s: state
                a: action taken
                r: reward after action
        """
        self.episode_states.append(np.array(s))
        self.episode_rewards.append(r)

        # Store actions as list of arrays
        # e.g. for n_y = 2 -> [ array([ 1.,  0.]), array([ 0.,  1.]), array([ 0.,  1.]), array([ 1.,  0.]) ]
        action = np.zeros(self.n_y)
        action[a] = 1
        self.episode_actions.append(action)


    def choose_action(self, observation):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (num_features)

            Returns: index of action we want to choose
        """
        # Reshape observation to (num_features, 1)
        observation = observation[np.newaxis,:]

        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.outputs_softmax, feed_dict = {self.X: observation})

        # Select action using a biased sample
        # this will return the index of the action we've sampled
        action = np.random.choice(range(len(prob_weights.ravel())), p=prob_weights.ravel())
        return action

    def learn(self):
        # Discount and normalize episode reward
        discounted_episode_rewards_norm = np.array(self.discount_and_norm_rewards()).reshape(-1,1)

        # Train on episode
        self.sess.run(self.train_op, feed_dict={
             self.X: self.episode_states,
             self.Y: np.array(self.episode_actions),
             self.discounted_episode_rewards_norm: np.array(discounted_episode_rewards_norm)
        })
        return discounted_episode_rewards_norm

    def reset(self):
        # Reset the episode data
        self.episode_states, self.episode_actions, self.episode_rewards  = [], [], []

        

    def discount_and_norm_rewards(self):
        discounted_episode_rewards = []#np.zeros_like(self.episode_rewards)
        cumulative = self.episode_rewards[-1]
        for t in reversed(range(len(self.episode_rewards))):
            discounted_episode_rewards.append(np.float(cumulative * self.gamma**t + self.episode_rewards[t]))
#
#        discounted_episode_rewards -= np.mean(discounted_episode_rewards)
#        discounted_episode_rewards /= np.std(discounted_episode_rewards)
        
        return discounted_episode_rewards


    def build_network(self):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=(None,self.n_x), name="X")
            self.Y = tf.placeholder(tf.float32, shape=(None,self.n_y), name="Y")
            self.discounted_episode_rewards_norm = tf.placeholder(tf.float32, [None, 1], name="actions_value")

        # Initialize parameters
        units_input_layer = self.n_x
        units_layer_1 = int(self.n_x*2)
        units_layer_2 = int(self.n_x/4)
        units_output_layer = self.n_y
        
        with tf.name_scope('parameters'):
            W1 = tf.Variable(tf.random_normal([units_input_layer, units_layer_1], stddev=1/units_input_layer**0.5), name="W1")
            b1 = tf.Variable(tf.random_normal([units_layer_1], stddev=1/units_layer_1**0.5), name="b1")
#            W2 = tf.Variable(tf.random_normal([units_layer_1, units_layer_2], stddev=1/units_layer_1**0.5), name="W2")
#            b2 = tf.Variable(tf.random_normal([units_layer_2], stddev=1/units_layer_2**0.5), name="b2")
            W3 = tf.Variable(tf.random_normal([units_layer_1, units_output_layer], stddev=1/units_layer_2**0.5), name="W3")
            b3 = tf.Variable(tf.random_normal([units_output_layer], stddev=1/units_output_layer**0.5), name="b3")

        # Forward prop
        with tf.name_scope('layer_1'):
            Z1 = tf.add(tf.matmul(tf.cast(self.X, tf.float32), W1),b1)
            A1 = tf.nn.relu(Z1)
#        with tf.name_scope('layer_2'):
#            Z2 = tf.add(tf.matmul(A1, W2),b2)
#            A2 = tf.nn.relu(Z2)
        with tf.name_scope('layer_3'):
#            Z3 = tf.add(tf.matmul(A2, W3),b3)
            Z3 = tf.add(tf.matmul(A1, W3),b3)

        # Softmax outputs, we need to transpose as tensorflow nn functions expects them in this shape
        logits = tf.transpose(Z3)
        labels = tf.transpose(self.Y)
        self.outputs_softmax = tf.nn.softmax(logits, axis=0)
        with tf.name_scope('loss'):
            neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=logits, labels=labels)
            loss = tf.reduce_mean(neg_log_prob * self.discounted_episode_rewards_norm)  # reward guided loss
            
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(loss)
    def learning_rate(self,multiplier):
        self.lr = self.lr*multiplier
        
    def plot_cost(self):
        import matplotlib.pyplot as plt
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
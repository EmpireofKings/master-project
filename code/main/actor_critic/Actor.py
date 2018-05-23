"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


class PolicyGradient:
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

        self.build_network()

        self.cost_history = []

        self.sess = tf.Session()

        # $ tensorboard --logdir=_logs_
        # http://0.0.0.0:6006/
        tf.summary.FileWriter("_logs_/", self.sess.graph)

        self.sess.run(tf.global_variables_initializer())

        # 'Saver' op to save and restore all the variables
        self.saver = tf.train.Saver()

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
        
        # Reshape necessary for as tesseract expects the rows as columns
        observation = observation.reshape(1, -1)
        
        # Run forward propagation to get softmax probabilities
        prob_weights = self.sess.run(self.network, feed_dict={self.X: observation})
        
        # Select action using a biased sample
        # this will return the index of the action we've sampled
        return np.argmax(prob_weights)
        #np.random.seed(1)
        #return np.random.choice(range(self.n_y), p=prob_weights.ravel())

    def learn(self):
        # Discount episode reward
        discounted_rewards = self.discount_rewards()

        ## DEBUG OUTPUT commented out
        #print("episode rewards", self.episode_rewards)
        #print("losses", self.sess.run(self.loss, feed_dict={
        #                        self.X: self.episode_states,
        #                        self.actions_taken: self.episode_actions,
        #                        self.discounted_rewards: discounted_rewards}))
        #print("discount rewards\n", discounted_rewards)
        #print("action values\n", self.episode_actions)
        #print("episode_states\n", self.episode_states)
        
        #print("policy loss", self.sess.run(self.policy_loss, feed_dict={
        #                            self.actions_taken: self.episode_actions,
        #                            self.discounted_rewards: discounted_rewards}))
        
        #test_state = np.zeros((1, self.n_x))
        #test_state[0,0] = 1
        
        #print("test_state", test_state)
        #print("Before learning:", self.sess.run(self.network, feed_dict={self.X: test_state}))
        ## DEBUG END
        
        # Train on available data    
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                            self.X: self.episode_states,
                            self.actions_taken: self.episode_actions,
                            self.discounted_rewards: discounted_rewards})

        #print("After learning:", self.sess.run(self.network, feed_dict={self.X: test_state}))
        
    def reset(self):
        # Reset the episode data, clear memory
        self.episode_states = np.empty((0,self.n_x), np.float32)
        self.episode_actions = []
        self.episode_rewards  = []

    def discount_rewards(self):
        count_steps = len(self.episode_rewards)
        # Not all rewads should be negative if the last is negative
        # TODO: Maybe not needed because of normalization
        last_reward = self.episode_rewards[-1] if self.episode_rewards[-1] > 0 else 0
        discounted_rewards = np.zeros((count_steps))
        discounted_rewards[-1] = self.episode_rewards[-1]
        
        for t in range(count_steps - 1):
            discounted_rewards[t] = last_reward * self.discount_factor**(count_steps-t) + self.episode_rewards[t]
        
        discounted_rewards = discounted_rewards.reshape(-1, 1)
        return MinMaxScaler().fit(discounted_rewards).transform(discounted_rewards)

        
    def build_network(self):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_x], name="X")
            self.actions_taken = tf.placeholder(tf.float32,
                                                shape=[None, self.n_y],
                                                name="actions_taken")
            self.discounted_rewards = tf.placeholder(tf.float32,
                                                     shape=[None, 1],
                                                     name="discounted_rewards")
            
        # First hidden layer
        layer_1 = tf.layers.Dense(units=30,  # num output nodes
                                  activation=tf.nn.relu,
                                  name="layer_1")

        # Raw output layer, therefore no activation function
        # Also called logits sometimes, don't know why
        layer_2 = tf.layers.Dense(units=self.n_y,  # num output nodes
                                  name="layer_2")

        self.network = tf.nn.softmax(layer_2(layer_1(self.X)), name="layer_softmax_out")
        
        ## Loss
        # Element-wise multiplication, to touch only values for which we actually took an action
        # Note that the discounded rewards are normalized between 0 and 1 in order to fit to the softmax
        #    output of the network
        # -> actions: [0, 1, 0, 0] * 0.1 => [0, 0.1, 0, 0]
        # If the softmmax outup is [0, 0.9, 0, 0] but we have a low (negative before normalization) reward
        #   this value (0.9) will be decreased
        self.policy_loss = tf.multiply(self.actions_taken, self.discounted_rewards)
        
        # sum((y_hat-y_true)**2)
        # Touch only values for which we actually took an action
        y_hat_single_action = tf.multiply(self.actions_taken, self.network)
        self.loss = tf.reduce_sum((y_hat_single_action - self.policy_loss)**2)
        
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
    
    def adjust_learning_rate(self,multiplier):
        self.lr = self.lr * multiplier
        
    def plot_cost(self):
        plt.plot(np.arange(len(self.cost_history)), self.cost_history)
        plt.ylabel('Cost')
        plt.xlabel('Training Steps')
        plt.show()
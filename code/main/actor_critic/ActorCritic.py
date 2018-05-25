"""
Policy Gradient Reinforcement Learning
Uses a 3 layer neural network as the policy network

"""
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


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
        prob_weights = self.sess.run(self.action_NN, feed_dict={self.X: observation})
        
        # Select action using a biased sample
        # this will return the index of the action we've sampled
        return np.argmax(prob_weights)
        #np.random.seed(1)
        #return np.random.choice(range(self.n_y), p=prob_weights.ravel())

    def learn(self, last_state, was_target_found):
        # Discount episode reward
        advantages, true_values = self.calc_advantages_and_true_values(last_state, was_target_found)

        ## DEBUG OUTPUT commented out
        #print("episode rewards", self.episode_rewards)
        #print("losses", self.sess.run(self.loss, feed_dict={
        #                        self.X: self.episode_states,
        #                        self.actions_taken: self.episode_actions,
        #                        self.advantages: advantages}))
        #print("discount rewards\n", advantages)
        #print("action values\n", self.episode_actions)
        #print("episode_states\n", self.episode_states)
        
        #print("policy loss", self.sess.run(self.policy_loss, feed_dict={
        #                            self.actions_taken: self.episode_actions,
        #                            self.advantages: advantages}))
        
        #test_state = np.zeros((1, self.n_x))
        #test_state[0,0] = 1
        
        #print("test_state", test_state)
        #print("Before learning:", self.sess.run(self.network, feed_dict={self.X: test_state}))
        ## DEBUG END
        
        # Train on available data    
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                            self.X: self.episode_states,
                            self.actions_taken: self.episode_actions,
                            self.advantages: advantages,
                            self.true_values: true_values})

        #print("After learning:", self.sess.run(self.network, feed_dict={self.X: test_state}))
        
    def reset(self):
        # Reset the episode data, clear memory
        self.episode_states = np.empty((0,self.n_x), np.float32)
        self.episode_actions = []
        self.episode_rewards  = []

    def calc_advantages_and_true_values(self, last_state, was_target_found):
        count_steps = len(self.episode_rewards)
        
        # get critic values
        values = self.sess.run(self.critic_NN, feed_dict={self.X: self.episode_states})
        
        # get value of last state
        if was_target_found:
            value_last_state = 10
        else:
            value_last_state = self.sess.run(self.critic_NN, feed_dict={self.X: last_state.reshape(1, -1)})
        
        # Calculate advantages
        advantages = np.zeros(count_steps, dtype=np.float32)
        true_values = np.zeros(count_steps, dtype=np.float32).reshape(-1, 1)
        
        true_values[-1] = value_last_state + self.episode_rewards[-1]
        
        for i in reversed(range(count_steps-1)):
            true_values[i] = true_values[i+1] + self.episode_rewards[i]
        
        advantages = true_values - values
        
        #print("values", values)
        #print("rewards", self.episode_rewards)
        #print("true_values", true_values)
        #print("advantages", advantages)
        
        
        return advantages, true_values

        
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
        layer_1 = tf.layers.Dense(units=30,  # num output nodes
                                  activation=tf.nn.relu,
                                  name="layer_1")

        # Raw action output layer, therefore no activation function
        # Also called logits sometimes, don't know why
        layer_actions = tf.layers.Dense(units=self.n_y,  # num output nodes
                                  name="layer_actions")

        self.action_NN = tf.nn.softmax(layer_actions(layer_1(self.X)), name="layer_softmax_out")
        
        # Raw critic output layer, therefore no activation function
        layer_critic = tf.layers.Dense(units=1,
                                       name="layer_critic")
        
        self.critic_NN = layer_critic(layer_1(self.X))
        
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
        
        
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
     
    def print_value_estimation(self):
        values = self.sess.run(self.critic_NN, feed_dict={self.X: self.episode_states})
        print("values of the critic:\n", values)
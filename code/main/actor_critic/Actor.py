"""
Deep Q-learning
Uses a 3 layer neural network as the policy network

"""
import tensorflow as tf
import numpy as np


class PolicyGradient:
    def __init__(self,
                 input_size,
                 output_size,
                 learning_rate,
                 grid_size_flat,
                 value_decay,
                 regul_factor):
        self.n_x = input_size
        self.n_y = output_size
        self.grid_size = grid_size_flat
        self.lr = learning_rate
        self.discount_factor = value_decay
        self.regul_factor = regul_factor
        
        self.episode_states = np.empty((0,input_size), np.float32)
        self.episode_rewards = []
        self.episode_next_values = []
        self.episode_actions = []

        self.build_network()

        self.sess = tf.Session()

        self.sess.run(tf.global_variables_initializer())

        # 'Saver' op to save and restore all the variables
#        self.saver = tf.train.Saver()


    def store_transition(self, s, r, a, v):
        """
            Store play memory for training

            Arguments:
                s: state
                r: reward after action
                a: action taken
                v: values for next state using previous weights
        """
        
        self.episode_states = np.vstack((self.episode_states, s))
        
        self.episode_rewards.append(r)
        
        # Store actions as list of arrays, one-hot encoding at each transition
        action = np.zeros(self.n_y, dtype=np.float32)
        action[a] = 1
        self.episode_actions.append(action)
        
        
        self.episode_next_values.append(v)


    def choose_action(self, observation):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (,num_features)

            Returns: index of action we want to choose
        """
        
        observation = observation.reshape(1, -1)
        
        # Run forward propagation to get values of the action outputs
        values = self.sess.run(self.network, feed_dict={self.X: observation})

        # return the index of the action which results in the best expected reward
        return np.argmax(values)


    def action_values(self, observation):
        
        observation = observation.reshape(1, -1)
        
        # Run forward propagation to get values of the action outputs
        values = self.sess.run(self.network, feed_dict={self.X: observation})

        # return the index of the action which results in the best expected reward
        return values
        

    def max_value_action(self, observation):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (,num_features)

            Returns: max value of possible actions
        """
        
        observation = observation.reshape(1, -1)
        
        # Run forward propagation to get Q-values of each action output
        values = self.sess.run(self.network, feed_dict={self.X: observation})
        
        # this will return the max value of the possible actions
        return np.max(values)


    def learn(self):
        # Train on available data    
        loss, _ = self.sess.run([self.loss, self.train_op], feed_dict={
                            self.X: self.episode_states,
                            self.values: self.discount_values(),
                            self.actions_taken: self.episode_actions,
                            self.rewards: np.array(self.episode_rewards).reshape(-1,1)})


    def reset(self):
        # Reset the episode data, clear memory
        self.episode_states = np.empty((0,self.n_x), np.float32)
        self.episode_rewards  = []
        self.episode_actions = []
        self.episode_next_values = []
        
        
    def discount_values(self):
        """Discount the value of the best action in the following state
        using previous parameters (calculated in the run_simulator).
        This is used in the loss function"""
        
        discounted_values = []
                
        for i in range(len(self.episode_next_values)):
            discounted_values.append((self.episode_next_values[i]*self.discount_factor))
        
        discounted_values = np.array(discounted_values).reshape(-1,1)
        
        return discounted_values
        
    
    def build_network(self):
        # Create placeholders
        with tf.name_scope('inputs'):
            self.X = tf.placeholder(tf.float32, shape=[None, self.n_x], name="X")

            self.values = tf.placeholder(tf.float32,
                                                shape=[None, 1],
                                                name="next_values") 
            
            self.rewards = tf.placeholder(tf.float32,
                                                shape=[None, 1],
                                                name="rewards") 
            
            self.actions_taken = tf.placeholder(tf.float32,
                                                shape=[None, self.n_y],
                                                name="actions_taken")
        """Variables"""
        inputs = self.n_x
        nodes_l1 = int(self.grid_size*2)
        nodes_l2 = int(self.grid_size)
        outputs = self.n_y
        
        
        
        
        """First hidden layer"""
        layer_1 = tf.layers.Dense(units=nodes_l1,
                                  activation=tf.nn.relu,
                                  kernel_initializer=tf.variance_scaling_initializer(
                                          scale=2.0,
                                          mode="fan_avg",
                                          distribution="uniform",
                                          dtype=tf.float32),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  name="layer_1")

        
        """Second hidden layer"""
        layer_2 = tf.layers.Dense(units=nodes_l2,
                                  activation = tf.nn.relu,
                                  kernel_initializer=tf.variance_scaling_initializer(
                                          scale=2.0,
                                          mode="fan_avg",
                                          distribution="uniform",
                                          dtype=tf.float32),
                                  bias_initializer=tf.constant_initializer(0.1),
                                  name="layer_2")
        
        """Output layer"""
        layer_3 = tf.layers.Dense(units = outputs,
                                  kernel_initializer=tf.variance_scaling_initializer(
                                          scale=1.0,
                                          mode="fan_avg",
                                          distribution="uniform",
                                          dtype=tf.float32),
                                  name = "layer_3")

#        self.network = layer_3(layer_1(self.X))
        self.network = layer_3(layer_2(layer_1(self.X)))
        
#        
#        
#        
#        """First hidden layer"""
#        weights_1 = tf.Variable(tf.random_uniform((inputs,nodes_l1),
#                                                  minval=-2/np.sqrt(inputs),
#                                                  maxval=2/np.sqrt(inputs)))
#        biases_1 = tf.Variable(tf.random_uniform((1,nodes_l1),
#                                                 minval=-1/np.sqrt(nodes_l1),
#                                                 maxval=1/np.sqrt(nodes_l1)))
#        h1_output = tf.nn.relu(tf.matmul(self.X,weights_1)+biases_1)
#        
#        
#        """Second hidden layer"""
#        weights_2 = tf.Variable(tf.random_uniform((nodes_l1,nodes_l2),
#                                                  minval=-2/np.sqrt(nodes_l1),
#                                                  maxval=2/np.sqrt(nodes_l1)))
#        biases_2 = tf.Variable(tf.random_uniform((1,nodes_l2),
#                                                 minval=-1/np.sqrt(nodes_l2),
#                                                 maxval=1/np.sqrt(nodes_l2)))
#        h2_output = tf.nn.relu(tf.matmul(h1_output,weights_2)+biases_2)
#        
#        
#        """Output layer"""
#        weights_output = tf.Variable(tf.random_uniform((nodes_l2,outputs),
#                                                  minval=-2/np.sqrt(nodes_l2),
#                                                  maxval=2/np.sqrt(nodes_l2)))
#        biases_output = tf.Variable(tf.random_uniform((1,outputs),
#                                                 minval=-1/np.sqrt(outputs),
#                                                 maxval=1/np.sqrt(outputs)))
#        nn_output = tf.matmul(h2_output,weights_output)+biases_output
#        
#        self.network = nn_output

        
        """Regularization L2 and Entropy"""
#        layer1_reg = tf.nn.l2_loss(weights_1)
#        layer2_reg = tf.nn.l2_loss(weights_2)
#        output_reg = tf.nn.l2_loss(weights_output)
        
        entropy = tf.reduce_sum(tf.log(tf.nn.softmax(self.network))*tf.nn.softmax(self.network))
        entropy *= self.regul_factor
#        regul =  layer1_reg + layer2_reg + output_reg + entropy


        """Loss"""
        #Reward + discounted value of best next action using 
        # previous weights - current best action and weights
        y_hat_1_act = tf.multiply(self.actions_taken, self.network)
        val_1_act = tf.multiply(self.actions_taken, self.values)
        rew_1_act = tf.multiply(self.actions_taken, self.rewards)
        entropy_1_act = tf.multiply(self.actions_taken, entropy)
        
        self.loss = tf.reduce_sum(tf.square(rew_1_act + val_1_act - y_hat_1_act)) + entropy_1_act
        
        with tf.name_scope('train'):
            self.train_op = tf.train.AdamOptimizer(self.lr).minimize(self.loss)
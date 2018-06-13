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

        self.entropy_factor = 0.8

        self.episode_states = np.empty((0,input_size), np.float32)
        self.episode_actions = []
        self.episode_rewards = []
        
        
        modifier = datetime.now().strftime("%m%d-%H%M%S") + " lnr=,disc=%s" % (self.discount_factor)
        log_path = "_logs_/" + modifier

        # $ tensorboard --logdir=_logs_
        self.writer = tf.summary.FileWriter(log_path)
        
        self.iterations = 0
        
        self.build_network()
        
        self.sess = tf.Session()
        
        #print([x.name for x in tf.global_variables()])
        self.sess.run(tf.global_variables_initializer())


    def build_network(self):
    
        self.X   = tf.placeholder(tf.float32, shape=[None, self.n_x], name="X")
        self.R   = tf.placeholder(tf.float32, shape=[None, 1], name="discounted_rewards")
        self.A   = tf.placeholder(tf.int32, shape=[None], name="actions")
        self.AVG = tf.placeholder(tf.float32, shape=[None, 1], name="advantages")

        #with tf.variable_scope("policy_network", reuse=tf.AUTO_REUSE):#
        with tf.name_scope("policy_network"):
            policy_nn_raw = self.policy_nn()
            self.policy_network = tf.nn.softmax(policy_nn_raw)
            
        print("Shape policy", self.policy_network)

        #with tf.variable_scope("critic_network", reuse=tf.AUTO_REUSE):
        with tf.name_scope("critic_network"):
            self.critic_network = self.critic_nn()
            

        with tf.name_scope("entropy"):
            entropy = -tf.reduce_sum(policy_nn_raw * tf.log(policy_nn_raw))

        
        with tf.name_scope("policy_loss"):
            # Touch only the action we took
            action_mask_one_hot = tf.one_hot(self.A,    # column index
                                             self.policy_network.shape[1],
                                             on_value = True,
                                             off_value = False,
                                             dtype = tf.bool)
                                             
            policy = tf.boolean_mask(self.policy_network, action_mask_one_hot)
            
            #policy = tf.log(policy)
            
            # policy * advantage + entropy
            self.policy_loss = tf.reduce_mean(policy*self.AVG)


        with tf.name_scope("critic_loss"):
            # advantage squared
            self.critic_loss = tf.reduce_mean(tf.square(self.R - self.critic_network))

            


        self.loss = -entropy#0.25*self.critic_loss + self.policy_loss - self.entropy_factor * entropy
            
        trainer = tf.train.GradientDescentOptimizer(self.lr)
        #trainer = tf.train.RMSPropOptimizer(learning_rate=self.lr, decay=0.99, epsilon=1e-5)
        #trainer = tf.train.AdamOptimizer(self.lr)
        
        #self.train_ops = (self.trainer.minimize(self.policy_loss),
        #                  self.trainer.minimize(self.critic_loss))
        
        self.train_ops = trainer.minimize(self.loss)
        
        #https://github.com/openai/baselines/blob/master/baselines/a2c/a2c.py
        
        #variables = tf.trainable_variables()
        #print(variables)
        #grads = tf.gradients(self.loss, variables)
        
        #max_grad_norm = 20
        #if max_grad_norm is not None:
        #    grads, grad_norm = tf.clip_by_global_norm(grads, max_grad_norm)
        
        #grads = list(zip(grads, variables))
        
        
        
        
        #self.train_ops = trainer.apply_gradients(grads)

        


    def learn(self, was_target_found):

        # Final reference reward
        if was_target_found:
            discounted_reward = 0
        else:
            last_state = self.episode_states[-1]
            discounted_reward = self.get_critic(last_state.reshape(1, -1))

        #print("discounted_reward", discounted_reward)
        #print("len episode rewards", len(self.episode_rewards))
        
        episode_count = len(self.episode_rewards)

        discounted_rewards = []
        for i in reversed(range(episode_count-1)):
            discounted_reward = self.episode_rewards[i] + self.discount_factor * discounted_reward
            discounted_rewards.append(discounted_reward)
        discounted_rewards = np.array(list(reversed(discounted_rewards))).ravel().reshape(-1, 1)
        
        values = self.get_critic(self.episode_states[0:-1])
        
        
        advantages = np.array(self.episode_rewards[:-1]).reshape(-1, 1)#discounted_rewards - values
        
        #if self.iterations %  4 == 0:
            #print(np.hstack((values, discounted_rewards)))
            #print("---", sum(np.abs(advantages)), "---")
            #input("Press Enter to continue...")
        

        #print("---------------------------")
        #print(self.episode_states)
        #print(self.episode_actions)
        #print(self.episode_rewards)
        #print(discounted_rewards)
        #print("---------------------------")
        
        
        
        feed_dict = {self.X: self.episode_states[:-1],
                     self.A: self.episode_actions[:-1],
                     self.R: discounted_rewards,
                     self.AVG: advantages}


        # TODO ?
        #var_policy = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="policy_network")
        #var_critic = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope="critic_network")
        

        #action_gradients = tf.gradients(self.policy_loss, var_policy)
        #value_gradients = tf.gradients(self.critic_loss, var_critic)
        
        
        self.sess.run(self.train_ops, feed_dict=feed_dict)

        # Does not work with Adam optimizer!
        #self.sess.run(self.trainer.apply_gradients(zip(action_gradients, var_policy)), feed_dict=feed_dict)
        #self.sess.run(self.trainer.apply_gradients(zip(value_gradients, var_critic)), feed_dict=feed_dict)


        # Will collect all summaries for tensorboard
        #summ = tf.summary.merge_all()

        #_, _, _, s = self.sess.run((self.train_ops,
        #                            self.policy_loss,
        #                            self.critic_loss,
        #                            summ), feed_dict=feed_dict)

        #self.writer.add_summary(s, self.iterations)

        self.iterations += 1

    ### Network contruction functions ###

    def dense(self, input, kernel_shape, activation=True):
        # Have to use get_variable to have the scope adjusted correctly
        kernel = tf.get_variable("kernel", kernel_shape, initializer=tf.variance_scaling_initializer(scale=2.0,
                                                                                                     mode="fan_avg",
                                                                                                     distribution="uniform"),
                                 trainable=True)
        biases = tf.get_variable("biases", kernel_shape[1], initializer=tf.constant_initializer(0.1))
        raw_dense = tf.matmul(input, kernel) + biases

        #if not tf.is_variable_initialized(kernel).eval(session=self.sess):
        #    self.sess.run(tf.variables_initializer([kernel]))
        #if not tf.is_variable_initialized(biases).eval(session=self.sess):
        #    self.sess.run(tf.variables_initializer([biases]))

        tf.summary.histogram("weights", kernel)
        tf.summary.histogram("biases", biases)
        
        if activation:
            act = tf.nn.relu(raw_dense)
            tf.summary.histogram("activations", act)
            return act
        else:
            return raw_dense
            
    def dense_output(self, input, kernel_shape):
        # Have to use get_variable to have the scope adjusted correctly
        kernel = tf.get_variable("kernel", kernel_shape, initializer=tf.variance_scaling_initializer(scale=1.0,
                                                                                                     mode="fan_avg",
                                                                                                     distribution="uniform"))
        biases = tf.get_variable("biases", kernel_shape[1], initializer=tf.constant_initializer(-0.1))
        raw_dense = tf.matmul(input, kernel) + biases

        #if not tf.is_variable_initialized(kernel).eval(session=self.sess):
        #    self.sess.run(tf.variables_initializer([kernel]))
        #if not tf.is_variable_initialized(biases).eval(session=self.sess):
        #    self.sess.run(tf.variables_initializer([biases]))

        tf.summary.histogram("weights", kernel)
        tf.summary.histogram("biases", biases)
        
        return raw_dense
        
        
    def shared_hidden_layers():
        # 1st: same number of nodes as input
        
        # 2nd: half number of inputs
        
        # kernel initialization: tf.variance_scaling_initializer
        # dist: uniform,
        # scale: 2.0 (due to relu)
        # mode="fan_avg"
        #
        # tf.constant_initializer(0.1)
        pass


    def policy_nn(self):
        with tf.variable_scope("hidden1", reuse=tf.AUTO_REUSE):
            in_hidden1 = self.n_x
            out_hidden1 = 100
            hidden1 = self.dense(self.X, [in_hidden1, out_hidden1])
        with tf.variable_scope("hidden2", reuse=tf.AUTO_REUSE):
            in_hidden2 = 100
            out_hidden2 = 50
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
            out_hidden1 = 100
            hidden1 = self.dense(self.X, [in_hidden1, out_hidden1])
        with tf.variable_scope("hidden2", reuse=tf.AUTO_REUSE):
            in_hidden2 = 100
            out_hidden2 = 50
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
        #action = np.zeros(self.n_y, dtype=np.float32)
        #action[a] = 1
        #self.episode_actions.append(action)
        self.episode_actions.append(a)
        


    def choose_action(self, observation):
        """
            Choose action based on observation

            Arguments:
                observation: array of state, has shape (,num_features)

            Returns: index of action we want to choose
        """

        return np.argmax(self.get_policy(observation))


    def reset(self):
        # Reset the episode data, clear memory
        self.episode_states = np.empty((0,self.n_x), np.float32)
        self.episode_actions = []
        self.episode_rewards  = []

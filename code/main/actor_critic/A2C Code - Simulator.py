import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# Import target module
import sys
import os
sys.path.append(os.path.abspath("../../main/simulator"))
from Simulator import *

np.random.seed(1)



# Our reward function

gridsize = [10,5]
grid_flat = gridsize[0]*gridsize[1]
target_found = False
num_actions = 4

x = tf.placeholder(tf.float32, [None, grid_flat])
y_true = tf.placeholder(tf.float32, [None, num_actions])
y_true_cls = tf.placeholder(tf.int64, [None])

#Weights and biases for each layer and output with sizes and initial values
weights1 = tf.Variable(tf.random_normal([grid_flat, grid_flat/2], stddev=1/grid_flat**0.5), name="H1 weights")
biases1 = tf.Variable(tf.random_normal([grid_flat/2], stddev=1/grid_flat**0.5), name="H1 bias")

weights2 = tf.Variable(tf.random_normal([grid_flat/2, int(grid_flat/4)], stddev=1/int(grid_flat/4)**0.5), name="H2 weights")
biases2 = tf.Variable(tf.random_normal([int(grid_flat/4)], stddev=1/int(grid_flat/4)**0.5), name="H2 bias")

policy_ws = tf.Variable(tf.random_normal([grid_flat/2, num_actions], stddev=1/int(grid_flat/4)**0.5), name="Pol weights")
policy_bs = tf.Variable(tf.random_normal([num_actions], stddev=1/int(grid_flat/4)**0.5), name="Pol bias")

value_ws = tf.Variable(tf.random_normal([int(grid_flat/4), 1], stddev=1/int(grid_flat/4)**0.5), name="Val weights")
value_bs = tf.Variable(tf.random_normal([1], stddev=1/int(grid_flat/4)**0.5), name="Val bias")


#Caclulation of each layer output
hidden1 = tf.matmul(x, weights1) + biases1
hidden1 = tf.nn.relu(hidden1)
hidden2 = tf.matmul(hidden1, weights2) + biases2
hidden2 = tf.nn.relu(hidden2)
#Value Function Output and Preferred Action
#policy = tf.nn.softmax(policy_f)
policy = tf.matmul(hidden2,policy_ws)+policy_bs

best_action = tf.argmax(policy, axis=1)


#Policy output calculation
value = tf.matmul(hidden2,value_ws)+value_bs

disc_map = np.zeros((1,grid_flat))

def reward(location):
    if target_found == True:
        return 10 #arbitrary reward for finding target
    else:
        return -disc_map[np.argmax(location)] #if a drone has been to a location multiple times, the penalty will increase linearly with each visit



l_rate = 0.01
e_greedy = 0.8
epochs = 100
max_iterations = 50
returns = []


obs_reward = reward(drone_loc) #will need to be changed to discounted reward later
act_reward = policy.get([best_action]) #policy indexed at the best action
advantage = obs_reward - act_reward

#Loss Functions:
value_loss = 0.5 * tf.reduce_sum(tf.square(obs_reward - tf.reshape(value,[-1])))
entropy = - tf.reduce_sum(policy * tf.log(policy))
policy_loss = -tf.reduce_sum(tf.log(act_reward)*advantage)
loss = 0.5 * value_loss + policy_loss# - entropy * 0.01

optimizer = tf.train.AdamOptimizer(learning_rate=l_rate).minimize(loss)

session = tf.Session()

session.run(tf.global_variables_initializer())

for epoch in range(epochs):
    # Setup our simulator
    grid = Grid(gridsize[0],gridsize[1], seed=grid_seed)
    grid.set_obstacles(0) #No obstacles to start
    drone = Drone(grid, position=Point(0,0), id=99)
    itr = 0
    disc_map = drone.get_drones_vector()
    x_batch = []
    y_true_batch = []
    rewards = 0
    while itr < max_iterations:
        
        # e-greedy
        if np.random.randint(1,100)/100 < e_greedy/(1+epoch):
            next_move = np.random.choice(list(Direction))
        else:
            next_move = tf.argmax(policy, axis=1)
        
        
        x_batch.append(drone.get_drones_vector())
        y_true_batch.append() #how do you calculate true policy and value results?
        grid.move_drone(drone,next_move)
        rewards.append(reward(drone.get_drones_vector()))
        disc_map += drone.get_drones_vector()
        
    returns.append(np.sum(rewards))
    feed_dict_train = {x: x_batch, y_true: y_true_batch}
    session.run(optimizer, feed_dict = feed_dict_train)

plt.plot(returns)
plt.xlabel('Epochs')
plt.ylabel('Return of Overall Reward')
plt.show()
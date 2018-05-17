from Actor import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import time
from Simulator import *


EPISODES = 300
rewards = []

gridsize = [5,10]
grid_flat = gridsize[0]*gridsize[1]
num_channels = 2 #1 for drone location, 1 for map of visited locations.  2 more will be needed for other drone and obstacle locations
input_size = grid_flat*num_channels

output_size = 4 #number of actions to choose from
action_size = 4 #number of actions to choose from
e_greedy = 0.8
grid_seed = 1
max_iterations = 50

def Try_Move(drone,action):
    try:
        drone.move(Direction(action))
    except (PositioningError, IndexError):
        return grid.get_drones_vector()
    else:
        return grid.get_drones_vector()
     
def Reward(location):
        if done == True:
            return 10 #arbitrary reward for finding target
        else:
            return -disc_map[np.argmax(location)]-1 #if a drone has been to a location multiple times, the penalty will increase linearly with each visit


if __name__ == "__main__":
    reward_list = []
    
    PG = PolicyGradient(
        n_x = input_size,
        n_y = output_size,
        learning_rate=0.01,
        reward_decay=0.99

    )
    
    for episode in range(EPISODES):
        done = False
        
        episode_reward = 0

        tic = time.clock()
        grid = Grid(gridsize[0],gridsize[1], seed=grid_seed)
        grid.set_obstacles(0) #No obstacles to start
        drone = Drone(grid, position=Point(0,0), id=99)
        drone_loc = grid.get_drones_vector()
        grid.set_target()
        disc_map = drone_loc
        itr = 1
        state = np.reshape(np.append(drone_loc,disc_map),(-1,input_size)).ravel()
        while True:


            
            # 1. Choose an action based on observation
            if np.random.randint(1,100)/100 < e_greedy/(1+episode):
                action = np.random.randint(0,action_size)
            else:
                action = PG.choose_action(state)
                

            # 2. Take action in the environment
            drone_loc = Try_Move(drone,action)
            disc_map = np.reshape(disc_map,(grid_flat,1))+np.reshape(drone_loc,(grid_flat,1))
            state_ = np.reshape(np.append(drone_loc,disc_map).ravel(),(input_size,1)).ravel()
            observed_surrounding = drone.observe_surrounding()
            
            
            #Check to see if target has been found
            if "T" in observed_surrounding:
                done = True
            reward = Reward(drone_loc)

            # 4. Store transition for training
            PG.store_transition(state, action, reward)
            
            toc = time.clock()
            elapsed_sec = toc - tic
            itr +=1
            
            if elapsed_sec > 120:
                done = True
            
            if itr > max_iterations:
                done = True         
            
            if done:
                episode_rewards_sum = sum(PG.episode_rewards)
                rewards.append(episode_rewards_sum)
                max_reward_so_far = np.amax(rewards)

                print("==========================================")
                print("Episode: ", episode)
                print("Seconds: ", elapsed_sec)
                print("Reward: ", episode_rewards_sum)
                print("Max reward so far: ", max_reward_so_far)
                print('Discovered map: \n', disc_map.reshape(gridsize[0],gridsize[1]))

                # 5. Train neural network
                discounted_episode_rewards_norm = PG.learn()
                reward_list.append(episode_rewards_sum)
                break

            # Save new state
            state = state_
    plt.plot(reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Cost')
    plt.show()
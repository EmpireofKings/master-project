from Actor import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import time
import sys, os
sys.path.append(os.path.abspath("../simulator"))#Change path to be wherever Simulator.py is stored locally

from Simulator import *

np.random.seed(1)

EPISODES = 400
rewards = []

gridsize = [5, 10]
grid_flat = gridsize[0] * gridsize[1]
# 1 for drone location, 1 for map of visited locations
num_channels = 2
input_size = grid_flat * num_channels

action_size = 4 #number of actions to choose from
e_greedy = 0.4
grid_seed = 1
max_iterations = 30#grid_flat #it will take this many actions to go to all cells in the grid


def move_and_get_reward(drone, action_idx, disc_map):
    """Move the drone and get the reward."""
    try:
        drone.move(Direction(action_idx))
        if "T" in drone.observe_surrounding():
            # arbitrary reward for finding target
            return 1
        else:
            # if a drone has been to a location multiple times,
            # the penalty will increase linearly with each visit
            location_point = drone.get_position()
            location = location_point.get_y() * gridsize[1] + location_point.get_x()
            return -disc_map[location]
            #return 0.1
    except (PositioningError, IndexError):
        # We hit an obsticle or tried to exit the grid
        return -10 # arbitrary
            


if __name__ == "__main__":
    reward_list = []
    
    PG = PolicyGradient(input_size, action_size,
                        learning_rate=0.01,
                        reward_decay=0.99)
    
    global_disc_map = np.zeros(grid_flat)
    
    for episode in range(EPISODES):
        print("Episode", episode, end="\r")
                
        # Setup simulator
        grid = Grid(gridsize[0],gridsize[1], seed=grid_seed)
        grid.set_obstacles(0) #No obstacles to start
        drone = Drone(grid, position=Point(0,0), id=99)
        grid.set_target()
        
        disc_map = np.zeros(grid_flat)

        # Generate traning data
        for itr in range(max_iterations):
            # 1. Get the current state
            drone_loc = grid.get_drones_vector()
            
            # Use smaller state space for the beginning
            disc_map += drone_loc
            state = np.append(drone_loc, disc_map)
            
            # 2. Choose an action based on observation
            if np.random.randint(1,100)/100 < e_greedy:
                action_idx = np.random.randint(0, action_size)
            else:
                action_idx = PG.choose_action(state)
                
            # 3. Take action in the environment
            reward = move_and_get_reward(drone, action_idx, disc_map)
            
            # 4. Store transition for training
            PG.store_transition(state, action_idx, reward)
            
            # 5. Check to see if target has been found
            if reward >= 1:
                disc_map += grid.get_drones_vector()
                break
        
        global_disc_map += disc_map
        #print("\n-------TRAINING------\n")
        
        # 5. Train neural network
        PG.learn()
        
        reward_list.append(np.sum(PG.episode_rewards))
        
        #Reset the action, state, and reward lists. 
        #Keeping this is like replay memory (full sampling)
        PG.reset()
    
    print("\n-------TRAINING------\n")
    print("Total training discovery map:\n", global_disc_map.reshape(gridsize[0], gridsize[1]))
    
    # EXPLOTATION
    print("\n---------EXPLOITATION--------\n")
    # Setup simulator
    grid = Grid(gridsize[0],gridsize[1], seed=grid_seed)
    grid.set_obstacles(0) #No obstacles to start
    drone = Drone(grid, position=Point(0,0), id=99)
    grid.set_target()
    
    disc_map = np.zeros(grid_flat)

    # Run simulator
    for itr in range(max_iterations):
        # 1. Get the current state
        drone_loc = grid.get_drones_vector()
        
        # Use smaller state space for the beginning
        disc_map = disc_map + drone_loc
        state = np.append(drone_loc, disc_map)
        
        # 2. Choose an action based on observation
        action_idx = PG.choose_action(state)
            
        # 3. Take action in the environment
        reward = move_and_get_reward(drone, action_idx, disc_map)
        
        # 5. Check to see if target has been found
        if reward >= 1:
            break
    
    print("Exploitation discovery map:\n", disc_map.reshape(gridsize[0], gridsize[1]))
    print(grid)
    
    plt.plot(reward_list)
    plt.xlabel('Episodes')
    plt.ylabel('Cost')
    plt.show()
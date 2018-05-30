from Actor import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import sys, os
sys.path.append(os.path.abspath("../simulator")) #This finds where Simulator.py is located

from Simulator import *


"""Hyperparameters"""
np.random.seed(1)

EPISODES = 1000

gridsize = [5, 10]
grid_flat = gridsize[0] * gridsize[1]

# 1 for drone location, 1 for map of visited locations
num_channels = 2
input_size = grid_flat * num_channels

action_size = 4 #number of actions to choose from
e_greedy = 0.4
grid_seed = 1
max_iterations = int(grid_flat/2)
val_decay = 0.25
l_rate = 0.001
reg_factor = 0.1

num_exec= 3

def move_and_get_reward(drone, action_idx, disc_map,itr):
    """Move the drone and get the reward."""
    cost = (itr+1)/(max_iterations*50)
    
    location_point = drone.get_position()
    location = location_point.get_y() * gridsize[1] + location_point.get_x()
    bad_move = disc_map[location]/(max_iterations*100)
    
    try:
        drone.move(Direction(action_idx))
        if "T" in drone.observe_surrounding():
            # arbitrary reward for finding target: This will always be positive
            return max_iterations/100 - cost
        else:
            # if a drone has been to a location multiple times,
            # the penalty will increase linearly with each visit
            return -bad_move - cost

    except (PositioningError, IndexError):
        # We hit an obstacle or tried to exit the grid (-max_itr / 10)
        # Penalty is arbitrary but includes the cost and the bad_move penalty
        return -1/10 - bad_move - cost # arbitrary
            

if __name__ == "__main__":
    
    reward_list = [] #Reward sum of each Test Episode
    
    for i in range(0,num_exec):

        
        action_values = []
    
        targets_found_test = 0 #counter for number of times the target has been found in Test Episodes
        targets_found = 0 #counter for number of times the target has been found in Train Episodes
        
        PG = PolicyGradient(input_size, action_size,
                            learning_rate=l_rate,
                            value_decay=val_decay,
                            regul_factor = reg_factor)
        
        global_disc_map = np.zeros(grid_flat)
        
        PG_old = PG
        
        for episode in range(EPISODES):
            
            """Training Environment"""
            # Setup simulator
            grid = Grid(gridsize[0],gridsize[1], seed=grid_seed)
            grid.set_obstacles(0) #No obstacles to start
            drone = Drone(grid, position=Point(0,0), id=99)
            grid.set_target()
            
            drone_loc = grid.get_drones_vector()
            disc_map = np.zeros(grid_flat)
            state = np.append(drone_loc, disc_map)
            initial_state = state
            target_found=False
            # Generate traning data
            for itr in range(max_iterations):
                
                # Choose an action based on observation
                if np.random.randint(1,100)/100 < e_greedy:
                    action_idx = np.random.randint(0, action_size)
                else:
                    action_idx = PG.choose_action(state)
                
                
                # Take action in the environment
                reward = move_and_get_reward(drone, action_idx, disc_map,itr)
                
                if reward >= 0:
                    target_found = True
                
                
                # Update the state
                drone_loc = grid.get_drones_vector()
                disc_map += drone_loc
                state_next = np.append(drone_loc, disc_map)
    
                
                # Calculate Next Action Value
                n_value = PG_old.max_value_action(state_next)
                if target_found:
                    n_value = 0
    
    
                # Store transition for training
                PG.store_transition(state, reward, action_idx, n_value)
                state = state_next
                
                
                """Enable the following to have iterative updates"""
                PG_Old = PG
                PG.learn()
                PG.reset() #Disabling this is episodic replay memory (full sampling)
                
                
                # End Episode if target has been found
                if target_found:
                    targets_found+=1
                    break
            
            action_values.append(np.array(PG.action_values(initial_state)).reshape(-1,4))
            global_disc_map += disc_map
    
    
            """Enable the following to have episodic updates"""
    #        PG_old = PG
    #        PG.learn()
    #        PG.reset() #Disabling this is replay memory (full sampling)
            
            
            
            """Testing Environment"""
            #Reset Grid and variables
            grid = Grid(gridsize[0],gridsize[1], seed=grid_seed)
            grid.set_obstacles(0) #No obstacles to start
            drone = Drone(grid, position=Point(0,0), id=99)
            grid.set_target()
            rewards = 0
            
            disc_map = np.zeros(grid_flat)
            drone_loc = grid.get_drones_vector()
            disc_map = disc_map + drone_loc
            state = np.append(drone_loc, disc_map)
            target_found = False
            
            # Run simulator
            for itr in range(max_iterations):
                # Choose an action based on observation
                action_idx = PG.choose_action(state)
                
                # Take action in the environment
                reward = move_and_get_reward(drone, action_idx, disc_map,itr)
                
                if reward >=0:
                    target_found = True
                    targets_found_test+=1
                    
                rewards += reward
                
                drone_loc = grid.get_drones_vector()
                disc_map = disc_map + drone_loc
                state = np.append(drone_loc, disc_map)
                
                # End Episode if target has been found
                if target_found:
                    break
             
            reward_list.append(rewards)
            
        action_values = np.array(action_values).reshape(-1,4)

        plt.plot(action_values[:,0],label="Up")
        plt.plot(action_values[:,1],label="Right")
        plt.plot(action_values[:,2],label="Down")
        plt.plot(action_values[:,3],label="Left")
        plt.legend()
        plt.title('Drone Search Values')
        plt.xlabel('Episodes')
        plt.ylabel('Values')
        plt.rcParams["figure.figsize"]=(10,5)
        plt.show()
            
    # EXPLORATION
    print("\n-------TRAINING------\n")
    print("Target Found in: ",int(10000*targets_found/EPISODES)/100,'%  of Training Episodes')
    print("Total training discovery map:\n", global_disc_map.reshape(gridsize[0], gridsize[1]))
    
    # EXPLOTATION
    print("\n---------EXPLOITATION--------\n")
    print("Target Found in: ",int(10000*targets_found_test/EPISODES)/100,'%  of Test Episodes')
    print("Exploitation discovery map:\n", disc_map.reshape(gridsize[0], gridsize[1]))
#        print('\n',grid)

    reward_lists = np.mean(np.array(reward_list).reshape(-1,num_exec),axis=1)
    plt.plot(reward_lists, label= ('Reward'))
    plt.legend()
    plt.show()

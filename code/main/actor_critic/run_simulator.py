from ActorCritic import ActorCritic
import matplotlib.pyplot as plt
import numpy as np
import time
import sys, os
sys.path.append(os.path.abspath("../simulator"))

from Simulator import *

np.random.seed(1)

EPISODES = 3000
rewards = []
learning_rate = 1e-3

gridsize = [5, 10]
grid_flat = gridsize[0] * gridsize[1]
# 1 for drone location, 1 for map of visited locations
num_channels = 2
input_size = grid_flat * num_channels

action_size = 4 #number of actions to choose from
e_greedy = 0.0
grid_seed = 1
max_iterations = int(grid_flat/2) #it will take this many actions to go to half of the cells in the grid


def move_and_get_reward(drone, action_idx, disc_map, itr):
    movement_cost = (itr+1) / (max_iterations * 10)
    
    """Move the drone and get the reward."""
    try:
        drone.move(Direction(action_idx))
        if "T" in drone.observe_surrounding():
            # arbitrary reward for finding target
            return 1 - movement_cost, True
            #return 1, True
        else:
            # if a drone has been to a location multiple times,
            # the penalty will increase linearly with each visit
            location_point = drone.get_position()
            location = location_point.get_y() * gridsize[1] + location_point.get_x()
    
            return 0, False#-disc_map[location] / (max_iterations * 10), False
            return -movement_cost
    except (PositioningError, IndexError):
        # We hit an obstacle or tried to exit the grid
        location_point = drone.get_position()
        location = location_point.get_y() * gridsize[1] + location_point.get_x()
        return -1 / max_iterations, False # disc_map[location] / (max_iterations * 5) - 0.4, False # arbitrary 



if __name__ == "__main__":
    reward_list = [[], [], [], []]
    reward_list_train = [[], [], [], []]
    action_values_tl = [[], [], [], []]
    advantages = []
    target_found = 0
    PG = ActorCritic(input_size, action_size,
                     learning_rate,
                     reward_decay=0.99)

    global_disc_map = np.zeros(grid_flat)

    
    positions = [Point(0,0), Point(0,4), Point(9,4), Point(9,0)]


    for episode in range(EPISODES):
        print("Episode", episode, end="\r")
        
        position = positions[0]#[episode % 4]

        # Setup simulator
        grid = Grid(gridsize[0],gridsize[1], seed=grid_seed)
        grid.set_obstacles(0) #No obstacles to start
        
        # TODO: Change drone start randomly somewhere at the border
        drone = Drone(grid, position, id=99)
        grid.set_target()

        disc_map = np.zeros(grid_flat)
        was_target_found = False
        rewards = 0

        # Generate traning data
        for itr in range(max_iterations):
        
            # 1. Get the current state
            drone_loc = grid.get_drones_vector()

            # Use smaller state space for the beginning
            #disc_map += drone_loc
            state = np.append(drone_loc, disc_map)
            

            # 2. Choose an action based on observation
            # TODO: Decrease e-greedy for frequently visited cells
            if np.random.randint(1,100)/100 < e_greedy:#*(EPISODES / (EPISODES + global_disc_map[np.argmax(drone_loc)])):
                action_idx = np.random.randint(0, action_size)
            else:
                action_idx = PG.choose_action(state)

            # 3. Take action in the environment
            reward, found = move_and_get_reward(drone, action_idx, disc_map, itr)
            rewards += reward

            # 4. Store transition for training
            PG.store_transition(state, action_idx, reward)
            

            # 5. Check to see if target has been found
            if found == True:
                disc_map += grid.get_drones_vector()
                target_found+=1
                was_target_found = True
                break

        #print(grid)
        reward_list_train[episode%4].append(rewards)

        global_disc_map += disc_map
        #print("\n-------TRAINING------\n")

        # Get last state for learning
        #disc_map += grid.get_drones_vector()
        last_state = np.append(grid.get_drones_vector(), disc_map)

        # 5. Train neural network
        avg_sum = PG.learn(was_target_found, last_state)
        
        advantages.append(avg_sum)

#        reward_list.append(np.sum(PG.episode_rewards))

        #Reset the action, state, and reward lists.
        #Keeping this is like replay memory (full sampling)
        PG.reset()

        grid = Grid(gridsize[0],gridsize[1], seed=grid_seed)
        grid.set_obstacles(0) #No obstacles to start
        
        drone = Drone(grid, position=position, id=99)
        grid.set_target()
        rewards = 0
        disc_map = np.zeros(grid_flat)

        # Run simulator
        for itr in range(max_iterations):
            # 1. Get the current state
            drone_loc = grid.get_drones_vector()

            # Use smaller state space for the beginning
            #disc_map = disc_map + drone_loc
            state = np.append(drone_loc, disc_map)

            # 2. Choose an action based on observation
            action_idx = PG.choose_action(state)

            # 3. Take action in the environment
            reward, found = move_and_get_reward(drone, action_idx, disc_map, itr)
            rewards += reward

            # 5. Check to see if target has been found
            if found:
                break

        reward_list[episode%4].append(rewards)
        
        
        drone_loc1 = np.zeros(grid_flat)
        drone_loc1[0] = 1
        state1 = np.append(drone_loc1, np.zeros(grid_flat))
        
        action_values = PG.get_policy(state1)[0]
        
        action_values_tl[0].append(action_values[0])
        action_values_tl[1].append(action_values[1])
        action_values_tl[2].append(action_values[2])
        action_values_tl[3].append(action_values[3])
        
        
        

    print("\n-------TRAINING------\n")
    print("Target Found in: ",int(100*target_found/EPISODES),'%  of Episodes')
    print("Total training discovery map:\n", global_disc_map.reshape(gridsize[0], gridsize[1]))

    # EXPLOTATION
    print("\n---------EXPLOITATION--------\n")
    # Setup simulator


    print("Exploitation discovery map:\n", disc_map.reshape(gridsize[0], gridsize[1]))
    print('\n',grid)
    
    plt.plot(advantages)
    plt.show()
    
    plt.plot(action_values_tl[0], label="up")
    plt.plot(action_values_tl[1], label="right")
    plt.plot(action_values_tl[2], label="down")
    plt.plot(action_values_tl[3], label="left")
    plt.legend()
    plt.show()
    
    
    plt.plot(reward_list_train[0], label="top left")
    plt.plot(reward_list_train[1], label="bottom left")
    plt.plot(reward_list_train[2], label="bottom right")
    plt.plot(reward_list_train[3], label="top right")
    plt.legend()
    plt.title('Drone Search TRAIN')
    plt.xlabel('Episodes')
    plt.ylabel('Cost')
    plt.rcParams["figure.figsize"]=(10,5)
    plt.show()


    #plt.plot(reward_list[0], label="top left")
    #plt.plot(reward_list[1], label="bottom left")
    #plt.plot(reward_list[2], label="bottom right")
    #plt.plot(reward_list[3], label="top right")
    #plt.legend()
    #plt.title('Drone Search')
    #plt.xlabel('Episodes')
    #plt.ylabel('Cost')
    #plt.rcParams["figure.figsize"]=(10,5)
    #plt.show()

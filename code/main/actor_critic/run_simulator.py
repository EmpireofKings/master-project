from Actor import PolicyGradient
import matplotlib.pyplot as plt
import numpy as np
import sys, os
from random import shuffle
import datetime
sys.path.append(os.path.abspath("../simulator")) #Where Simulator.py is located

from Simulator import *


"""Hyperparameters"""
np.random.seed(1)

gridsize = [5, 10]
grid_flat = gridsize[0] * gridsize[1]

# 1 for drone location, 1 for map of visited locations
num_channels = 2
input_size = grid_flat * num_channels

action_size = 4 #number of actions to choose from
grid_seed = 1
num_obs = 5 #Number of obstacles to generate the grid with

e_greedy = 0.8
max_iterations = int(grid_flat/2)
val_decay = 0.9
l_rate = 0.0001
reg_factor = 0.1
EPISODES = 7500 #must be a multiple of the number of starting locations (4)

print('--- HYPERPARAMETERS ---')
print(' e_greedy:       ',e_greedy)
print(' max_iterations: ',max_iterations)
print(' val_decay:      ',val_decay)
print(' l_rate:         ',l_rate)
print(' reg_factor:     ',reg_factor)
print(' Episodes:       ',EPISODES)
print(' Grid seed:      ',grid_seed)
print(' Num Obstacles:  ',num_obs)

num_exec= 3



def move_and_get_reward(drone, action_idx, disc_map,itr):
    """Move the drone and get the reward."""
    cost = (itr+1)/(max_iterations*10)
    
    location_point = drone.get_position()
    location = location_point.get_y() * gridsize[1] + location_point.get_x()
    bad_move = disc_map[location]/(max_iterations*5)
    
    try:
        drone.move(Direction(action_idx))
        if "T" in drone.observe_surrounding():
            # arbitrary reward for finding target: This will always be positive
            return 1 - cost,True
        else:
            # if a drone has been to a location multiple times,
            # the penalty will increase linearly with each visit
            return -bad_move-cost,False

    except (PositioningError, IndexError):
        # We hit an obstacle or tried to exit the grid
        # Penalty is arbitrary but includes the cost and the bad_move penalty
        return -0.4-bad_move-cost,False
            



def test_simulator(PG,max_iterations,start_loc):
    #Reset Grid and variables
    grid = Grid(gridsize[0],gridsize[1], seed=grid_seed)
    grid.set_obstacles(num_obs) 
    drone = Drone(grid, position=Point(start_loc[1],start_loc[0]), id=99)
    grid.set_target()
    rewards = 0
    
    disc_map = np.zeros(grid_flat)
    drone_loc = grid.get_drones_vector()
    disc_map = disc_map + drone_loc
    state = np.append(drone_loc, disc_map)
    target_found = False
    iters = 0
    losses = 0
    targets_found_test = 0
    # Run simulator
    for itr in range(max_iterations):
        # Choose an action based on observation
        action_idx = PG.choose_action(state)
        iters+=1
        # Take action in the environment
        reward,target_found = move_and_get_reward(drone, action_idx, disc_map,itr)
        
        rewards += reward
        
        drone_loc = grid.get_drones_vector()
        disc_map = disc_map + drone_loc
        state_next = np.append(drone_loc, disc_map)
        
        n_value = PG_old.max_value_action(state_next)
        if target_found:
            n_value = 0
        
        qval = PG.max_value_action(state)
        loss = (reward + n_value*val_decay - qval)**2
        losses+=loss
        
        state = state_next.copy()
        
        # End Episode if target has been found
        if target_found:
            targets_found_test+=1
            break
    
    return rewards,losses,iters


def iter_ep(iter_episode):
    #Averages over the number of executions and number of start locations
    iter_ep_test = np.mean(np.array(iter_episode).reshape(num_exec,-1).T,axis=1)
    iter_ep_test = np.mean(np.array(iter_ep_test).reshape(-1,len(drone_start_locs)),axis=1)
    return iter_ep_test


def run_format(data):
    #Average over the number of executions
    return np.mean(np.array(data).reshape(num_exec,-1).T,axis=1)


if __name__ == "__main__":
    
    grid = Grid(gridsize[0],gridsize[1], seed=grid_seed)
    grid.set_obstacles(num_obs)
    
    drone = Drone(grid, position=Point(0,1), id=99) #0,1 if obstacles are implemented
    
    drone_loc = grid.get_drones_vector()
    disc_map = np.zeros(grid_flat)
    disc_map += drone_loc
    state = np.append(drone_loc, disc_map)
    initial_state = state
    targ_state = [0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 0, 0, 0, 0,0, 1, 0, 0, 0]
    
    target_state = np.append(targ_state,disc_map)
    
    start_point_order = np.array([0,1,2,3])
    drone_start_locs = [(0,1),(gridsize[0]-1,0),(0,gridsize[1]-1),(gridsize[0]-1,gridsize[1]-1)]
    
    
    """Plot Parameters"""
    plt.rcParams.update({'font.size': 20})
    fig_size = [20,10]    
    
    """Lists for capturing plot data"""
    reward_list0 = [] #Reward sum of each Test Episode
    reward_list1 = [] 
    reward_list2 = [] 
    reward_list3 = [] 
    
    losses_train = []
    losses_test0 = []
    losses_test1 = []
    losses_test2 = []
    losses_test3 = []
    
    iter_ep_train = []
    iter_ep_test0 = []
    iter_ep_test1 = []
    iter_ep_test2 = []
    iter_ep_test3 = []
    
    rewards_itr_test0 = []
    rewards_itr_test1 = []
    rewards_itr_test2 = []
    rewards_itr_test3 = []
    
    for i in range(0,num_exec):

        action_values_init = []
        action_values_targ = []
        
        targets_found_test = 0 #counter for number of times the target has been found in Test Episodes
        targets_found_train = 0 #counter for number of times the target has been found in Train Episodes
        opt_act = 0 #Number of optimal actions chosen at targ_state
        
        PG = PolicyGradient(input_size, action_size,
                            learning_rate=l_rate,
                            value_decay=val_decay,
                            regul_factor = reg_factor)
        
        global_disc_map = np.zeros(grid_flat)
        
        PG_old = PG #change to copy not ref
        
        for episode in range(EPISODES):
            
            """Training Environment"""
            # Setup simulator
            grid = Grid(gridsize[0],gridsize[1], seed=grid_seed)
            grid.set_obstacles(num_obs) 
            
            if episode%len(start_point_order)==0:
                shuffle(start_point_order)
            
            start_point = drone_start_locs[start_point_order[episode%len(start_point_order)]]
                
            drone = Drone(grid, position=Point(start_point[1],start_point[0]), id=99)
            grid.set_target()
            
            drone_loc = grid.get_drones_vector()
            disc_map = np.zeros(grid_flat)
            disc_map += drone_loc
            state = np.append(drone_loc, disc_map)
            
            target_found=False
            losses = 0
            iters = 0
            # Generate traning data
            for itr in range(max_iterations):
                iters+=1
                # Choose an action based on observation
                if np.random.randint(1,100)/100 < e_greedy*(EPISODES / (EPISODES + global_disc_map[np.argmax(drone_loc)])):
                    action_idx = np.random.randint(0, action_size)
                else:
                    action_idx = PG.choose_action(state)
                
                # Take action in the environment
                reward,target_found = move_and_get_reward(drone, action_idx, disc_map,itr)
                
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
                state = state_next.copy()
                
                if np.argmax(drone_loc) == np.argmax(targ_state) and action_idx == 1:
                    opt_act+=1
                    
                
                """Enable the following to have iterative updates"""
                PG_Old = PG # change ref to copy
                PG.learn()
                PG.reset() #Disabling this is episodic replay memory (full sampling)
                
                qval = PG.max_value_action(state)
                loss = (reward + n_value*val_decay - qval)**2
                losses+=loss
                
                # End Episode if target has been found
                if target_found:
                    targets_found_train+=1
                    break
            
            iter_ep_train.append(iters)
            if losses > 10:
                losses=10
            losses_train.append(losses)
            losses = 0
            action_values_init.append(np.array(PG.action_values(initial_state)).reshape(-1,4))
            action_values_targ.append(np.array(PG.action_values(target_state)).reshape(-1,4))
            global_disc_map += disc_map
            
            
            """Enable the following to have episodic updates"""
#            PG_old = PG # change ref to copy
#            PG.learn()
#            PG.reset() #Disabling this is replay memory (full sampling)
            
            
            
            """Testing Environment"""
            
            rewards,losses,iters=test_simulator(PG,max_iterations,drone_start_locs[0])
            iter_ep_test0.append(iters)
            reward_list0.append(rewards)
            rewards_itr_test0.append(rewards/iters)
            if losses > 10:
                losses=10
            losses_test0.append(losses)
            
            rewards,losses,iters=test_simulator(PG,max_iterations,drone_start_locs[1])
            iter_ep_test1.append(iters)
            reward_list1.append(rewards)
            rewards_itr_test1.append(rewards/iters)
            if losses > 10:
                losses=10
            losses_test1.append(losses)
            
            rewards,losses,iters=test_simulator(PG,max_iterations,drone_start_locs[2])
            iter_ep_test2.append(iters)
            reward_list2.append(rewards)
            rewards_itr_test2.append(rewards/iters)
            if losses > 10:
                losses=10
            losses_test2.append(losses)
            
            rewards,losses,iters=test_simulator(PG,max_iterations,drone_start_locs[3])
            iter_ep_test3.append(iters)
            reward_list3.append(rewards)
            rewards_itr_test3.append(rewards/iters)
            if losses > 10:
                losses=10
            losses_test3.append(losses)
            
        

        action_values = np.array(action_values_init).reshape(-1,4)
        plt.rcParams["figure.figsize"]=(fig_size[0],fig_size[1])
        plt.plot(action_values[:,0],label="Up")
        plt.plot(action_values[:,1],label="Right")
        plt.plot(action_values[:,2],label="Down")
        plt.plot(action_values[:,3],label="Left")
        plt.legend()
        plt.title('Drone Search Values Initial')
        plt.xlabel('Episodes')
        plt.ylabel('Values')
        
        plt.show()
        
        
        action_values = np.array(action_values_targ).reshape(-1,4)
        plt.rcParams["figure.figsize"]=(fig_size[0],fig_size[1])
        plt.plot(action_values[:,0],label="Up")
        plt.plot(action_values[:,1],label="Right")
        plt.plot(action_values[:,2],label="Down")
        plt.plot(action_values[:,3],label="Left")
        plt.legend()
        plt.title('Drone Search Values Near Target')
        plt.xlabel('Episodes')
        plt.ylabel('Values')
        plt.rcParams["figure.figsize"]=(10,5)
        plt.show()
            
    # EXPLORATION
    print("\n-------TRAINING------\n")
    print("Target Found in: {0:.1%}  of Train Episodes".format(targets_found_train/EPISODES))
    print("Total training discovery map:\n", global_disc_map.reshape(gridsize[0], gridsize[1]))
    
    # EXPLOTATION
    print("\n---------EXPLOITATION--------\n")
#    print("Target Found in: {0:.1%}  of Test Episodes'".format(targets_found_test/EPISODES))
#    print("Exploitation discovery map:\n", disc_map.reshape(gridsize[0], gridsize[1]))
#        print('\n',grid)
    
    
    
    """Test Reward Plot"""
    
    reward_list0 = run_format(reward_list0)
    reward_list1 = run_format(reward_list1)
    reward_list2 = run_format(reward_list2)
    reward_list3 = run_format(reward_list3)
    
    plt.rcParams["figure.figsize"]=(fig_size[0],fig_size[1])
    plt.plot(reward_list0,label = 'Starting Point(0,0)')
    plt.plot(reward_list1,label = 'Starting Point(4,0)')
    plt.plot(reward_list2,label = 'Starting Point(0,9)')
    plt.plot(reward_list3,label = 'Starting Point(4,9)')
    
    plt.legend()
    plt.title('Test Reward Sum')
    plt.show()
    
    
    
    """Test Reward per iteration Plot"""
    
    rewards_itr_test0 = run_format(rewards_itr_test0)
    rewards_itr_test1 = run_format(rewards_itr_test1)
    rewards_itr_test2 = run_format(rewards_itr_test2)
    rewards_itr_test3 = run_format(rewards_itr_test3)
    
    plt.rcParams["figure.figsize"]=(fig_size[0],fig_size[1])
    plt.plot(rewards_itr_test0,label = 'Starting Point(0,0)')
    plt.plot(rewards_itr_test1,label = 'Starting Point(4,0)')
    plt.plot(rewards_itr_test2,label = 'Starting Point(0,9)')
    plt.plot(rewards_itr_test3,label = 'Starting Point(4,9)')
    plt.title('Test Reward Normalized by Iteration per Episode')
    plt.legend()
    plt.show()
    
    
    
    
    """Test Reward Plot"""
   
    losses_train = run_format(losses_train)
    losses_test0 = run_format(losses_test0)
    losses_test1 = run_format(losses_test1)
    losses_test2 = run_format(losses_test2)
    losses_test3 = run_format(losses_test3)
    
    plt.rcParams["figure.figsize"]=(fig_size[0],fig_size[1])
    plt.plot(losses_train,label = 'Train')
    plt.plot(losses_test0,label = 'Test Starting Point(0,0)')
    plt.plot(losses_test1,label = 'Test Starting Point(4,0)')
    plt.plot(losses_test2,label = 'Test Starting Point(0,9)')
    plt.plot(losses_test3,label = 'Test Starting Point(4,9)')
    plt.legend()
    plt.title('Drone Search Losses')
    plt.xlabel('Episodes')
    plt.ylabel('Loss')
    plt.show()
    
    
    
    """Iterations per Episode plot"""
    #Every starting location will provide a different normal number of iterations, this will sum across all starting locations for the train.  The Test will still be separated.
    iter_ep_train = iter_ep(iter_ep_train)
    iter_ep_test0 = iter_ep(iter_ep_test0)
    iter_ep_test1 = iter_ep(iter_ep_test1)
    iter_ep_test2 = iter_ep(iter_ep_test2)
    iter_ep_test3 = iter_ep(iter_ep_test3)
    
    plt.rcParams["figure.figsize"]=(fig_size[0],fig_size[1])
    plt.plot(iter_ep_train,label = 'Train')
    plt.plot(iter_ep_test0,label = 'Test Starting Point(0,0)')
    plt.plot(iter_ep_test1,label = 'Test Starting Point(4,0)')
    plt.plot(iter_ep_test2,label = 'Test Starting Point(0,9)')
    plt.plot(iter_ep_test3,label = 'Test Starting Point(4,9)')
    plt.legend()
    plt.title('Average Iterations in Four Episodes')
    plt.xlabel('Episodes (x4)')
    plt.ylabel('Iterations')
    plt.show()


save_data = False

if save_data:
    save_path = os.path.join(os.getcwd(),"Results")
    run_folder = os.path.join(save_path, datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S'))
    os.makedirs(run_folder)
    
    
    """Hyperparameters"""
    hyperp = []
    hyperp.append('--- HYPERPARAMETERS ---')
    hyperp.append(' e_greedy:       '+str(e_greedy))
    hyperp.append(' max_iterations: '+str(max_iterations))
    hyperp.append(' val_decay:      '+str(val_decay))
    hyperp.append(' l_rate:         '+str(l_rate))
    hyperp.append(' reg_factor:     '+str(reg_factor))
    hyperp.append(' Episodes:       '+str(EPISODES))
    hyperp.append(' Grid seed:      '+str(grid_seed))
    hyperp.append(' Num Obstacles:  '+str(num_obs))

    file_name = 'Hyperparameters'
    save_file = os.path.join(run_folder,file_name+".txt")
    file = open(save_file,"w")
    for item in hyperp:
        file.write("%s\n" % item)
#    file.write(input(hyperp))
    file.close()
#    np.savetxt(save_file,np.array(hyperp))
    
    
    """Reward Lists"""
    rewards_dir = os.path.join(run_folder,"Rewards")
    os.makedirs(rewards_dir)
    
    file_name = 'reward_list0'
    save_file = os.path.join(rewards_dir,file_name+".txt")
    np.savetxt(save_file,reward_list0)
    
    file_name = 'reward_list1'
    save_file = os.path.join(rewards_dir,file_name+".txt")
    np.savetxt(save_file,reward_list1)
    
    file_name = 'reward_list2'
    save_file = os.path.join(rewards_dir,file_name+".txt")
    np.savetxt(save_file,reward_list2)
    
    file_name = 'reward_list3'
    save_file = os.path.join(rewards_dir,file_name+".txt")
    np.savetxt(save_file,reward_list3)
    
    
    
    """Rewards / Iterations Lists"""   
    rewards_itr = os.path.join(run_folder,"Rewards_Itr")
    os.makedirs(rewards_itr)
    
    file_name = 'rewards_itr_test0'
    save_file = os.path.join(rewards_itr,file_name+".txt")
    np.savetxt(save_file,rewards_itr_test0)
    
    file_name = 'rewards_itr_test1'
    save_file = os.path.join(rewards_itr,file_name+".txt")
    np.savetxt(save_file,rewards_itr_test1)
    
    file_name = 'rewards_itr_test2'
    save_file = os.path.join(rewards_itr,file_name+".txt")
    np.savetxt(save_file,rewards_itr_test2)
    
    file_name = 'rewards_itr_test3'
    save_file = os.path.join(rewards_itr,file_name+".txt")
    np.savetxt(save_file,rewards_itr_test3)



    """Loss Lists"""   
    loss_dir = os.path.join(run_folder,"Losses")
    os.makedirs(loss_dir)
    
    file_name = 'losses_train'
    save_file = os.path.join(loss_dir,file_name+".txt")
    np.savetxt(save_file,losses_train)
    
    file_name = 'losses_test0'
    save_file = os.path.join(loss_dir,file_name+".txt")
    np.savetxt(save_file,losses_test0)
    
    file_name = 'losses_test1'
    save_file = os.path.join(loss_dir,file_name+".txt")
    np.savetxt(save_file,losses_test1)
    
    file_name = 'losses_test2'
    save_file = os.path.join(loss_dir,file_name+".txt")
    np.savetxt(save_file,losses_test2)
    
    file_name = 'losses_test3'
    save_file = os.path.join(loss_dir,file_name+".txt")
    np.savetxt(save_file,losses_test3)
        

    
    """Iter Lists"""   
    itr_dir = os.path.join(run_folder,"Iterations")
    os.makedirs(itr_dir)
    
    file_name = 'iter_ep_train'
    save_file = os.path.join(itr_dir,file_name+".txt")
    np.savetxt(save_file,iter_ep_train)
    
    file_name = 'iter_ep_test0'
    save_file = os.path.join(itr_dir,file_name+".txt")
    np.savetxt(save_file,iter_ep_test0)
    
    file_name = 'iter_ep_test1'
    save_file = os.path.join(itr_dir,file_name+".txt")
    np.savetxt(save_file,iter_ep_test1)
    
    file_name = 'iter_ep_test2'
    save_file = os.path.join(itr_dir,file_name+".txt")
    np.savetxt(save_file,iter_ep_test2)
    
    file_name = 'iter_ep_test3'
    save_file = os.path.join(itr_dir,file_name+".txt")
    np.savetxt(save_file,iter_ep_test3)
    
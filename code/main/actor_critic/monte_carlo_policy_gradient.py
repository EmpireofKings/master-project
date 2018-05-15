import numpy as np
import matplotlib.pyplot as plt

# Import target module
import sys
import os
sys.path.append(os.path.abspath("../../main/simulator"))
from Simulator import *

np.random.seed(1)


#####                                                                  #####
# Use algorithm from Sutton & Batro (2017), chapter 13.3 REINFORCE, p. 271 #
#####                                                                  #####

######          REINFORCE: MONTE CARLO POLICY GRADIENT                ######

num_features = 3
def x(s, a):
    """vector of features visible when in state s taking action a"""
    next_s = s + a
    value = 1 if grid.get_value(next_s) == "O" else 0
    return np.array([next_s.getX(), next_s.getY(), value], dtype=np.float64)

def h(s, a, theta):
    """Parameterized numerical action preferences, equation 13.3.
       This will be replaced by a NN."""
    # Trivial
    return theta.T.dot(x(s, a))
    
def pi(s, a, theta):
    """policy parametrisation. Exponential soft-max over h(s, a, theta)"""
    return np.exp(h(s, a, theta)) / np.sum([np.exp(h(s, a_i, theta)) for a_i in Direction])

def pi_gradient(s, a, theta):
    """Gradient of policy parametrisation, equation 13.7"""
    return x(s, a) - np.sum([pi(s, a_i, theta) * x(s, a_i) for a_i in Direction])

# Our reward function

def reward(a):
    # Trivial
    try:
        drone.move(a)
    except (PositioningError, IndexError):
        return -1
    observed_surrounding = drone.observe_surrounding()
    if "T" in observed_surrounding:
        return 1
    else:
        return 0

def return_fkt(data, t):
    return np.sum(data[:t+1 ,2])

## HYPERPARAMETERS
steplenght = 1e-6
epochs = 200
train_steps = 100    # How many moves the drone does before updating weights
e_greedy = 0.2      # e_greedy percent of the times drone makes random move
    
# Initialize weights
theta = np.random.random_sample((num_features,))

# Example seeds:
#   Not converging: 1, 4, 7, 8, 10
#   Converging: 2, 3, 6
grid_seed = 3

theta_0 = theta.copy()
returns = []

for i in range(epochs):
    # Setup our simulator
    grid = Grid(10, 5, seed=grid_seed)
    grid.set_obsticles(10)
    drone = Drone(grid, position=Point(0,0), id=99)
    grid.set_target()
    if i == 0:
        print(grid)
    
    # Generate training data
    data = np.array([]) # will contain [state_t, action_t, reward_t+1, state_t+1]
    T = train_steps
    for t in range(T):
        s_t0 = drone.get_position()
        #print("policy:", [pi(s_t0, a_i, theta) for a_i in Direction])
        
        # e-greedy
        if np.random.randint(1 / e_greedy) == 0:
            preferred_action = np.random.choice(list(Direction))
        else:
            preferred_action = Direction(int(np.argmax([pi(s_t0, a_i, theta) for a_i in Direction])))
        reward_t1 = reward(preferred_action)
        s_t1 = drone.get_position()
        if t == 0:
            data = np.array([[s_t0, preferred_action, reward_t1, s_t1]])
        else:
            data = np.append(data, [[s_t0, preferred_action, reward_t1, s_t1]], axis=0)
        
        # Found target
        if reward_t1 == 1:
            T = t
            break
    
    # Calculate the total return as convergance metric
    returns.append(np.sum(data[:, 2]))
    #print(data)
    
    # Update the weights using stochstic gradient ascent. Equation 13.6
    for t in range(T-1):
        G = return_fkt(data, t)
        s = data[t, 0]
        a = data[t, 1]
        #print("cumulated return", G)
        theta = theta + steplenght * G * pi_gradient(s, a, theta)
        
    
print("theta 0", theta_0)
print("theta  ", theta)

plt.plot(returns)
plt.show()

print("\n========  EXPLOITATION PHASE  =======\n")

grid = Grid(10, 5, seed=grid_seed)
grid.set_obsticles(10)
drone = Drone(grid, position=Point(0,0), id=99)
grid.set_target()

for t in range(10):
    print(grid)
    s_t0 = drone.get_position()
    preferred_action = Direction(int(np.argmax([pi(s_t0, a_i, theta) for a_i in Direction])))
    print("\nNext direction:", preferred_action, "\n")
    reward_t1 = reward(preferred_action)
    if reward_t1 == 1:
        print("==================\nFinal situation:\n\n", grid)
        break


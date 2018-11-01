import gym
import numpy as np
from matplotlib import pyplot as plt
import random
import math

from rbf_approximation import update_estimation

# Use LunarLander-v2 for the second part.
env = gym.make('LunarLander-v2')

episodes = 20000
test_episodes = 10
action_dim = 2

# Reasonable values for Cartpole discretization
discr = 16
x_min, x_max = -2.4, 2.4
v_min, v_max = -3, 3
th_min, th_max = -0.3, 0.3
av_min, av_max = -4, 4

# Parameters
gamma = 0.99
alpha = 0.1
a = 1000

# Create discretization grid
x_grid = np.linspace(x_min, x_max, discr)
y_grid = np.linspace(x_min, x_max, discr)
v_x_grid = np.linspace(v_min, v_max, discr)
v_y_grid = np.linspace(v_min, v_max, discr)
th_grid = np.linspace(th_min, th_max, discr)
av_grid = np.linspace(av_min, av_max, discr)
c1_grid = np.linspace(0, 1, 2)
c2_grid = np.linspace(0, 1, 2)

def find_nearest_value_idx(array,value):
    idx = np.searchsorted(array, value, side="left")
    if idx > 0 and (idx == len(array) or math.fabs(value - array[idx-1]) < math.fabs(value - array[idx])):
        return idx-1
    else:
        return idx

def discretize_state(state):
    d_state = np.zeros(8, np.int8) 
    d_state[0] = find_nearest_value_idx(x_grid, state[0])
    d_state[1] = find_nearest_value_idx(y_grid, state[1])
    d_state[2] = find_nearest_value_idx(v_x_grid, state[2])
    d_state[3] = find_nearest_value_idx(v_y_grid, state[3])
    d_state[4] = find_nearest_value_idx(th_grid, state[4])
    d_state[5] = find_nearest_value_idx(av_grid, state[5])
    d_state[6] = find_nearest_value_idx(c1_grid, state[6])
    d_state[7] = find_nearest_value_idx(c2_grid, state[7])

    return d_state

def get_action_values(state, grid):
    return grid[state[0]][state[1]][state[2]][state[3]][state[4]][state[5]][state[6]][state[7]]

# Table for Q values
q_grid = np.zeros((discr, discr, discr, discr, discr, discr, 2, 2, action_dim))

# Training loop
ep_lengths, epl_avg = [], []

# Initate epsilon min, max, decay rate
epsilon = 1
max_epsilon = 1
min_epsilon = 0.01
decay_rate = 0.0002
for ep in range(episodes+test_episodes):
    # Do a greedy test run in the last few episodes and render it
    test = ep > episodes

    # Initialize things
    state, done, steps = env.reset(), False, 0

    # Loop through the episode
    while not done:
        # Pick a random action (change it!)
        # action = int(np.random.random()*action_dim)
        
        # Act greedy to epsilon
        is_exploitation = random.uniform(0, 1) > epsilon

        # state discretization
        d_state = discretize_state(state)
        actions_values = get_action_values(d_state, q_grid)

        if is_exploitation:
            action = np.argmax(actions_values)
        else:
            action = int(np.random.random()*action_dim)

        # Perform the action
        state, reward, done, _ = env.step(action)

        # Update Q value
        d_state_new = discretize_state(state)
        new_actions_values = get_action_values(d_state_new, q_grid)
        prev_q = actions_values[action]

        new_q = prev_q + alpha * (reward + gamma * np.max(new_actions_values) - prev_q)
        # actions_values[action] = new_q
        actions_values[action] = update_estimation(d_state_new, action, new_q)

        # Draw if testing
        if test:
            env.render()

        steps += 1

    # Reduce epsilon to emphasize exploitation over time
    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * ep)

    # Bookkeeping for plots
    ep_lengths.append(steps)
    epl_avg.append(np.mean(ep_lengths[min(0, ep-500):]))
    if ep % 200 == 0:
        print("Episode {}, average timesteps: {:.2f}, current epsilon is {:.3f}".format(ep, np.mean(ep_lengths[min(0, ep-200):]), epsilon))

# Save the Q-value array
np.save("q_values.npy", q_grid)

# Draw plots
plt.plot(ep_lengths)
plt.plot(epl_avg)
plt.legend(["Episode length", "500 episode average"])
plt.title("Episode lengths")
plt.show()

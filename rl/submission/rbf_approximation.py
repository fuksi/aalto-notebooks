import gym
import numpy as np
from sklearn.linear_model import SGDRegressor
from sklearn.kernel_approximation import RBFSampler
import sklearn.preprocessing
from sklearn.preprocessing import MinMaxScaler
from sklearn.pipeline import FeatureUnion
from matplotlib import pyplot as plt
import LunarLander
import pickle

# Create the environment
env = LunarLander.LunarLander()

featurizer = FeatureUnion([
    ("rbf1", RBFSampler(gamma=5.0, n_components=100)),
    ("rbf2", RBFSampler(gamma=2.0, n_components=100)),
    ("rbf3", RBFSampler(gamma=1.0, n_components=100)),
    ("rbf4", RBFSampler(gamma=0.5, n_components=100)),
])

observation_examples = np.array([env.observation_space.sample() for x in range(5000)])
scaler = sklearn.preprocessing.StandardScaler()
scaler.fit(observation_examples)

featurizer.fit(scaler.transform(observation_examples))


action_space = 4
state_space = 8


# Return scaled and featurized state
def preprocess(state):
    return featurizer.transform(scaler.transform([state]))


# Keep a separate Q function for each action
# (implementation detail)
q_functions = []
for a in range(action_space):
    m = SGDRegressor(learning_rate="constant")
    m.partial_fit(preprocess(np.zeros(state_space)), [0])
    q_functions.append(m)


# Return an estimation of Q(s, a)
def get_q_estimation(state, action):
    preprocessed = preprocess(state)
    return q_functions[action].predict(preprocessed)[0]


# Perform an SGD step to bring Q(s, a) closer to the given value
def update_estimation(state, action, value):
    preprocessed = preprocess(state)
    q_functions[action].partial_fit(preprocessed, [value])


# Get action for given state
def get_action(state, epsilon):
    is_exploitation = np.random.uniform(0, 1) > epsilon
    if is_exploitation:
        actions_values = [get_q_estimation(state, action) for action in range(action_space)]
        action = np.argmax(actions_values)
    else:
        action = int(np.random.random()*action_space)
    
    return action


# Main training loop
episodes = 5000
gamma = 0.99
total_rewards, smoothed_rewards = [], []
for ep in range(episodes):
    # Reset env and compute epsilon for this episode
    state, done, total_reward = env.reset(), False, 0
    current_state = state
    epsilon = 0.9*0.997**ep
    steps = 0

    # Loop through the episode
    while not done:
        action = get_action(state, epsilon)
        prev_state = state
        state, reward, done, _ = env.step(action)

        value = reward + gamma * get_q_estimation(state, action)

        total_reward += reward
        steps += 1

        update_estimation(prev_state, action, value)
        env.render()

        # Sometimes it gets stuck...
        if steps > 1000:
            done = True


    # Bookkeeping, again
    total_rewards.append(total_reward)
    smoothed_rewards.append(np.mean(total_rewards[min(0, ep-200):]))
    print("Episode {}, total reward: {:.3f}".format(ep, total_reward))

file = open("qfunction.mdl", "wb")
pickle.dump(q_functions, file)

plt.plot(total_rewards)
plt.plot(smoothed_rewards)
plt.title("Total reward per episode")
plt.show()

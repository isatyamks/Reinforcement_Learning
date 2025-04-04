import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

env = gym.make('CartPole-v1', render_mode='human')

pos_space = np.linspace(-2.4, 2.4, 10)
vel_space = np.linspace(-3, 3, 10)
ang_space = np.linspace(-.2095, .2095, 10)
ang_vel_space = np.linspace(-4, 4, 10)

with open('cartpole1.pkl', 'rb') as f:
    q = pickle.load(f)

learning_rate_a = 0.1
discount_factor_g = 0.99
epsilon = 1
epsilon_decay_rate = 0.00001
rng = np.random.default_rng()
rewards_per_episode = []
i = 0

state = env.reset()[0]
state_p = np.digitize(state[0], pos_space)
state_v = np.digitize(state[1], vel_space)
state_a = np.digitize(state[2], ang_space)
state_av = np.digitize(state[3], ang_vel_space)
terminated = False
rewards = 0

while not terminated and rewards < 10000:
    action = np.argmax(q[state_p, state_v, state_a, state_av, :])
    new_state, reward, terminated, _, _ = env.step(action)
    new_state_p = np.digitize(new_state[0], pos_space)
    new_state_v = np.digitize(new_state[1], vel_space)
    new_state_a = np.digitize(new_state[2], ang_space)
    new_state_av = np.digitize(new_state[3], ang_vel_space)
    state_p, state_v, state_a, state_av = new_state_p, new_state_v, new_state_a, new_state_av
    rewards += reward
    print(f'Episode: {i}  Rewards: {rewards}')

env.close()
plt.plot(rewards_per_episode)
plt.xlabel('Episode')
plt.ylabel('Rewards')
plt.title('Rewards per Episode')
plt.savefig('cartpole_rewards.png')

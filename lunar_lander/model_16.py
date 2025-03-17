import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
from collections import defaultdict

def discretize_state(state, bins):
    """Convert continuous state into a discrete index tuple."""
    return tuple(np.digitize(state[i], bins[i]) for i in range(len(state)))

def run(is_training=True, render=False):

    env = gym.make('LunarLander-v3', render_mode='human' if render else None)

    # Define discretization bins for state space (assumed ranges)
    state_bins = [
        np.linspace(-1.5, 1.5, 10),   # x position
        np.linspace(-1.5, 1.5, 10),   # y position
        np.linspace(-2.0, 2.0, 10),   # x velocity
        np.linspace(-2.0, 2.0, 10),   # y velocity
        np.linspace(-3.14, 3.14, 10), # angle
        np.linspace(-2.0, 2.0, 10),   # angular velocity
        [0, 1],                       # left leg contact
        [0, 1]                        # right leg contact

        [1,1]          #line space for the both legs

        
    ]

    action_space_size = env.action_space.n  # Discrete actions (0,1,2,3)

    if is_training:
        # Use defaultdict to initialize missing states dynamically
        q = defaultdict(lambda: np.zeros(action_space_size))
    else:
        try:
            with open('lunar_lander\\q_tables100000.pkl', 'rb') as f:
                q = pickle.load(f)
        except FileNotFoundError:
            print("Q-table file not found! Running in training mode.")
            q = defaultdict(lambda: np.zeros(action_space_size))

    # Hyperparameters
    learning_rate = 0.1  
    discount_factor = 0.99  
    epsilon = 1.0  
    epsilon_decay_rate = 0.0001  

    rng = np.random.default_rng()
    rewards_per_episode = []
    i = 0

    while True:
        state = env.reset()[0]
        state_d = discretize_state(state, state_bins)  # Convert to discrete indices
        terminated = False  
        rewards = 0  

        while not terminated:
            # Ensure the state exists in the Q-table
            if state_d not in q:
                q[state_d] = np.zeros(action_space_size)

            # Choose action using epsilon-greedy policy
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_d])

            # Take step in environment
            new_state, reward, terminated, _, _ = env.step(action)
            new_state_d = discretize_state(new_state, state_bins)

            if is_training:
                # Ensure new state exists in Q-table
                if new_state_d not in q:
                    q[new_state_d] = np.zeros(action_space_size)

                # Q-learning update
                q[state_d][action] += learning_rate * (
                    reward + discount_factor * np.max(q[new_state_d]) - q[state_d][action]
                )

            state_d = new_state_d
            rewards += reward

            if not is_training:
                print(f'Episode: {i}  Rewards: {rewards:.2f}')

        rewards_per_episode.append(rewards)
        mean_rewards = np.mean(rewards_per_episode[-100:])  # Moving average

        if is_training and i % 500 == 0:
            print(f'Episode: {i}  Mean Reward: {mean_rewards:.1f}')

        # Stop training if consistently high rewards are achieved
        if mean_rewards > 0:
            break

        epsilon = max(epsilon - epsilon_decay_rate, 0)  # Decay epsilon
        i += 1

    env.close()

    # Save Q-table
    if is_training:
        with open('lunar_lander_q.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Plot rewards per episode
    plt.plot(rewards_per_episode)
    plt.xlabel('Episode')
    plt.ylabel('Rewards')
    plt.title('Rewards per Episode')
    plt.savefig('lunar_lander_rewards.png')

if __name__ == '__main__':
    # Train the model
    run(is_training=True, render=False)

    # Test the trained model
    # run(is_training=False, render=True)

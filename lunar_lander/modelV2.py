import numpy as np
import gymnasium as gym
import pickle
import argparse
import matplotlib.pyplot as plt
from collections import deque

# Define Q-learning parameters
ALPHA = 0.1   # Learning rate
GAMMA = 0.99  # Discount factor
EPSILON = 1.0 # Initial exploration probability
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# Create the environment
env = gym.make('LunarLander-v3')
n_actions = env.action_space.n
obs_space = env.observation_space.shape[0]

# Discretization parameters
DISCRETE_BINS = 10  # Number of bins per dimension
obs_high = env.observation_space.high
obs_low = env.observation_space.low
obs_bins = [np.linspace(obs_low[i], obs_high[i], DISCRETE_BINS) for i in range(obs_space)]

# Initialize Q-table
Q = np.zeros([DISCRETE_BINS] * obs_space + [n_actions])


def discretize(state):
    """Convert continuous state into a discrete state for the Q-table."""
    indices = [np.digitize(state[i], obs_bins[i]) - 1 for i in range(obs_space)]
    return tuple(indices)


def choose_action(state, epsilon):
    """Choose an action using epsilon-greedy strategy."""
    if np.random.rand() < epsilon:
        return np.random.choice(n_actions)  # Explore
    return np.argmax(Q[state])  # Exploit


def train_model(num_episodes=1000):
    """Train the Q-learning model."""
    global EPSILON
    rewards = []
    rolling_mean = deque(maxlen=100)

    plt.ion()  # Turn on interactive mode for live plotting
    fig, ax = plt.subplots()
    
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0
        
        while True:
            action = choose_action(state, EPSILON)
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_state)
            
            # Q-learning update
            best_next_action = np.argmax(Q[next_state])
            Q[state + (action,)] += ALPHA * (reward + GAMMA * Q[next_state + (best_next_action,)] - Q[state + (action,)])
            
            state = next_state
            total_reward += reward
            
            if terminated or truncated:
                break
        
        rewards.append(total_reward)
        rolling_mean.append(total_reward)
        mean_reward = np.mean(rolling_mean)

        if episode % 10 == 0:
            print(f"Episode {episode}: Reward = {total_reward}, Mean Reward (last 100) = {mean_reward:.2f}")

        # Live plot update
        ax.clear()
        ax.set_title("Training Progress")
        ax.set_xlabel("Episode")
        ax.set_ylabel("Mean Reward (Last 100 Episodes)")
        ax.plot(range(len(rewards)), rewards, label="Episode Reward", alpha=0.5)
        ax.plot(range(len(rewards)), [np.mean(rewards[max(0, i-100):i+1]) for i in range(len(rewards))], label="Mean Reward (100)", color="red")
        ax.legend()
        plt.pause(0.01)

        # Decay epsilon
        EPSILON = max(EPSILON * EPSILON_DECAY, EPSILON_MIN)

    plt.ioff()
    plt.show()

    print("\nTraining completed.")
    return Q


def save_model(Q, filename="qq_table.pkl"):
    """Save the trained Q-table to a file."""
    with open(filename, "wb") as f:
        pickle.dump(Q, f)
    print("Model saved.")


def load_model(filename="qq_table.pkl"):
    """Load a trained Q-table from a file."""
    with open(filename, "rb") as f:
        return pickle.load(f)

import time

def test_model(Q, num_episodes=10):
    """Test the trained agent."""
    env = gym.make("LunarLander-v3", render_mode="human")  # ✅ Corrected here

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0

        while True:
            env.render()  
            time.sleep(0.05)  
            
            action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_state)
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()




if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train or test the Q-learning model.")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the model")
    parser.add_argument("--episodes", type=int, default=1000, help="Number of episodes for training/testing")
    args = parser.parse_args()

    if args.train:
        Q = train_model(args.episodes)
        save_model(Q)

    if args.test:
        Q = load_model()
        test_model(Q, args.episodes)

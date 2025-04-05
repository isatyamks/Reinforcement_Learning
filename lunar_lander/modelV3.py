import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
import argparse
import time
import math
from collections import deque, defaultdict

ALPHA = 0.05
GAMMA = 0.99
EPSILON = 1.0
EPSILON_MIN = 0.01
EPSILON_DECAY = 0.999
NUM_EPISODES = 5000

state_bounds = [
    (-1.0, 1.0),
    (0.0, 1.5),
    (-2.0, 2.0),
    (-2.0, 2.0),
    (-math.pi, math.pi),
    (-5, 5)
]

NUM_BINS = 10
bins = [np.linspace(low, high, NUM_BINS - 1) for low, high in state_bounds]

def discretize(state):
    discrete_state = []
    for i in range(6):
        low, high = state_bounds[i]
        value = np.clip(state[i], low, high)
        discrete_index = int(np.digitize(value, bins[i]))
        discrete_state.append(discrete_index)
    discrete_state.append(int(state[6]))
    discrete_state.append(int(state[7]))
    return tuple(discrete_state)

def choose_action(state, Q, epsilon):
    if np.random.random() < epsilon:
        return np.random.randint(0, 4)
    else:
        return np.argmax(Q[state])

def train_model(num_episodes):
    env = gym.make("LunarLander-v3", render_mode=None)
    Q = defaultdict(lambda: np.zeros(4))
    epsilon = EPSILON
    rewards_history = []
    rolling_mean = deque(maxlen=100)
    plt.ion()
    fig, ax = plt.subplots()

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0
        done = False

        while not done:
            action = choose_action(state, Q, epsilon)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_disc = discretize(next_state)
            best_next = np.argmax(Q[next_state_disc])
            td_target = reward + GAMMA * Q[next_state_disc][best_next]
            td_error = td_target - Q[state][action]
            Q[state][action] += ALPHA * td_error
            state = next_state_disc
            total_reward += reward

        rewards_history.append(total_reward)
        rolling_mean.append(total_reward)
        mean_reward = np.mean(rolling_mean)

        if episode % 100 == 0:
            print(f"Episode {episode}: Reward = {total_reward:.2f}, Mean Reward (last 100) = {mean_reward:.2f}")

        ax.clear()
        ax.plot(range(len(rewards_history)), rewards_history, label="Episode Reward", alpha=0.5)
        moving_avg = [np.mean(rewards_history[max(0, i-100):i+1]) for i in range(len(rewards_history))]
        ax.plot(range(len(rewards_history)), moving_avg, label="Moving Avg (100)", color='red')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Training Progress")
        ax.legend()
        plt.pause(0.01)

        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

    plt.ioff()
    plt.show()
    env.close()
    return Q, rewards_history

def test_model(Q, num_episodes):
    env = gym.make("LunarLander-v3", render_mode="human")
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0
        done = False

        while not done:
            time.sleep(0.02)
            action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = discretize(next_state)
            total_reward += reward

        print(f"Test Episode {episode+1}: Total Reward = {total_reward:.2f}")
    env.close()

def save_model(Q, filename="qqq_table.pkl"):
    with open(filename, "wb") as f:
        pickle.dump(dict(Q), f)
    print("Model saved to", filename)

def load_model(filename="q_tables100000.pkl"):
    with open(filename, "rb") as f:
        Q_dict = pickle.load(f)
    Q = defaultdict(lambda: np.zeros(4), Q_dict)
    print("Model loaded from", filename)
    return Q

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Enhanced Q-learning for LunarLander-v3")
    parser.add_argument("--train", action="store_true", help="Train the model")
    parser.add_argument("--test", action="store_true", help="Test the saved model")
    parser.add_argument("--episodes", type=int, default=NUM_EPISODES, help="Number of episodes")
    args = parser.parse_args()

    if args.train:
        Q, rewards_history = train_model(args.episodes)
        save_model(Q)
    if args.test:
        Q = load_model()
        test_model(Q, args.episodes)

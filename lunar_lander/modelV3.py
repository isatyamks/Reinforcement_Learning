import numpy as np
import gymnasium as gym
import matplotlib.pyplot as plt
import pickle
import argparse
import time
import math
from collections import deque, defaultdict

# ---------------------------
# Hyperparameters and Settings
# ---------------------------
ALPHA = 0.05          # Learning rate (tuned lower for smoother updates)
GAMMA = 0.99          # Discount factor
EPSILON = 1.0         # Starting exploration probability
EPSILON_MIN = 0.01    # Minimum epsilon
EPSILON_DECAY = 0.999 # Slower decay for more exploration
NUM_EPISODES = 5000   # Total training episodes

# ---------------------------
# Discretization Settings
# ---------------------------
# Lunar Lander state: [x_pos, y_pos, x_vel, y_vel, angle, angular_vel, left_leg, right_leg]
# We'll define custom bounds for the first 6 continuous dimensions.
# These are heuristic ranges based on domain knowledge.
state_bounds = [
    (-1.0, 1.0),              # x position
    (0.0, 1.5),               # y position
    (-2.0, 2.0),              # x velocity
    (-2.0, 2.0),              # y velocity
    (-math.pi, math.pi),      # angle (radians)
    (-5, 5)                   # angular velocity
    # The last two dimensions (leg contacts) are binary.
]

NUM_BINS = 10  # Number of bins for each continuous dimension

# Create bins for each continuous dimension
bins = [np.linspace(low, high, NUM_BINS - 1) for low, high in state_bounds]

def discretize(state):
    """
    Convert the continuous state into a discrete tuple.
    For the first six dimensions, we clip and digitize using predefined bins.
    The leg contacts (last two dimensions) are kept as is.
    """
    discrete_state = []
    # Process first 6 dimensions
    for i in range(6):
        low, high = state_bounds[i]
        # Clip the value to remain within the defined bounds
        value = np.clip(state[i], low, high)
        discrete_index = int(np.digitize(value, bins[i]))
        discrete_state.append(discrete_index)
    # Append binary leg contact values
    discrete_state.append(int(state[6]))
    discrete_state.append(int(state[7]))
    return tuple(discrete_state)

def choose_action(state, Q, epsilon):
    """Select an action using the epsilon-greedy policy."""
    if np.random.random() < epsilon:
        return np.random.randint(0, 4)  # 4 discrete actions
    else:
        return np.argmax(Q[state])

# ---------------------------
# Training Function
# ---------------------------
def train_model(num_episodes):
    # Create environment without rendering during training.
    env = gym.make("LunarLander-v3", render_mode=None)
    Q = defaultdict(lambda: np.zeros(4))  # Q-table: state -> [q(a=0), q(a=1), q(a=2), q(a=3)]
    epsilon = EPSILON
    rewards_history = []
    rolling_mean = deque(maxlen=100)

    # Set up live plotting.
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

            # Q-learning update
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

        # Update live plot: Reward per episode and moving average over 100 episodes.
        ax.clear()
        ax.plot(range(len(rewards_history)), rewards_history, label="Episode Reward", alpha=0.5)
        moving_avg = [np.mean(rewards_history[max(0, i-100):i+1]) for i in range(len(rewards_history))]
        ax.plot(range(len(rewards_history)), moving_avg, label="Moving Avg (100)", color='red')
        ax.set_xlabel("Episode")
        ax.set_ylabel("Reward")
        ax.set_title("Training Progress")
        ax.legend()
        plt.pause(0.01)

        # Decay epsilon after each episode
        epsilon = max(epsilon * EPSILON_DECAY, EPSILON_MIN)

    plt.ioff()
    plt.show()
    env.close()
    return Q, rewards_history

# ---------------------------
# Testing Function
# ---------------------------
def test_model(Q, num_episodes):
    # Create environment with human rendering.
    env = gym.make("LunarLander-v3", render_mode="human")
    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0
        done = False

        while not done:
            # Rendering is handled automatically in 'human' mode.
            time.sleep(0.02)  # Slow down for visualization clarity
            action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            state = discretize(next_state)
            total_reward += reward

        print(f"Test Episode {episode+1}: Total Reward = {total_reward:.2f}")
    env.close()

# ---------------------------
# Model Save/Load Functions
# ---------------------------
def save_model(Q, filename="qqq_table.pkl"):
    with open(filename, "wb") as f:
        # Convert defaultdict to a normal dict before saving.
        pickle.dump(dict(Q), f)
    print("Model saved to", filename)

def load_model(filename="qqq_table.pkl"):
    with open(filename, "rb") as f:
        Q_dict = pickle.load(f)
    # Convert loaded dict back to defaultdict.
    Q = defaultdict(lambda: np.zeros(4), Q_dict)
    print("Model loaded from", filename)
    return Q

# ---------------------------
# Main: Train or Test based on command-line args
# ---------------------------
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

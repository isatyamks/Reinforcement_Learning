import gymnasium as gym
import numpy as np
import pickle
from collections import defaultdict, deque
import argparse
import random
import csv  # Import the csv module

ne = 10000

# --- Discretization Helper Functions ---
def create_bins(low, high, num_bins):
    """
    Create equally spaced bins for discretization.
    """
    return np.linspace(low, high, num_bins - 1)

# Define bins for the first 6 continuous state variables.
# LunarLander-v2 observation: [x_position, y_position, x_velocity, y_velocity, angle, angular_velocity, left_contact, right_contact]
bins = [
    create_bins(-1.5, 1.5, 20),     # x position
    create_bins(0.0, 2.0, 20),      # y position
    create_bins(-3.0, 3.0, 20),     # x velocity
    create_bins(-3.0, 3.0, 20),     # y velocity
    create_bins(-np.pi, np.pi, 20), # angle
    create_bins(-6.0, 6.0, 20)      # angular velocity
]

def discretize_state(state):
    """
    Discretize a continuous state into a tuple of integers.
    The first six dimensions are binned; the last two are already binary.
    """
    discrete_state = []
    for i in range(6):
        discrete_index = int(np.digitize(state[i], bins[i]))
        discrete_state.append(discrete_index)
    # Append leg contact states (already binary)
    discrete_state.append(int(state[6]))
    discrete_state.append(int(state[7]))
    return tuple(discrete_state)

# --- Q-Learning Agent Functions ---
def zero_q():
    return np.zeros(4)

def train_model(Q=None, num_episodes=10000, alpha=0.1, gamma=0.99, 
                epsilon=1.0, epsilon_min=0.01, epsilon_decay=0.995, 
                render=False, replay_buffer_size=10000, batch_size=32):
    # If no Q-table is provided, initialize a new one.
    if Q is None:
        Q = defaultdict(zero_q)
    rewards = []
    replay_buffer = deque(maxlen=replay_buffer_size)
    env = gym.make('LunarLander-v3')
    
    # Open a CSV file to log episode and reward data
    with open('training_log.csv', mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        csv_writer.writerow(['Episode', 'Reward'])  # Write header
    
        for episode in range(num_episodes):
            state, _ = env.reset()
            state_disc = discretize_state(state)
            total_reward = 0
            done = False
            
            while not done:
                # Epsilon-greedy action selection.
                if np.random.rand() < epsilon:
                    action = np.random.randint(0, 4)
                else:
                    action = np.argmax(Q[state_disc])
                
                next_state, reward, done, truncated, info = env.step(action)
                done = done or truncated
                next_state_disc = discretize_state(next_state)
                
                # Store experience in replay buffer.
                replay_buffer.append((state_disc, action, reward, next_state_disc, done))
                
                # Sample a batch from the replay buffer.
                if len(replay_buffer) >= batch_size:
                    batch = random.sample(replay_buffer, batch_size)
                    for s, a, r, ns, d in batch:
                        # Q-learning update.
                        best_next_action = np.argmax(Q[ns])
                        td_target = r + gamma * Q[ns][best_next_action] * (not d)
                        td_error = td_target - Q[s][a]
                        Q[s][a] += alpha * td_error
                
                state_disc = next_state_disc
                total_reward += reward
                
                if render:
                    env.render()
                    
            rewards.append(total_reward)
            
            # Log episode and reward to CSV
            csv_writer.writerow([episode + 1, total_reward])
            
            if (episode + 1) % 200 == 0:
                mean_reward_200 = np.mean(rewards[-200:])
                print(f"Episode {episode+1}/{num_episodes}, Mean Reward (last 200 episodes): {mean_reward_200:.2f}")
            if (episode + 1) % 1000 == 0:
                mean_reward_1000 = np.mean(rewards[-1000:])
                print(f"Episode {episode+1}/{num_episodes}, Mean Reward (last 1000 episodes): {mean_reward_1000:.2f}")
            # Decay the exploration rate.
            epsilon = max(epsilon_min, epsilon * epsilon_decay)
        
    env.close()
    return Q

def test_model(Q, num_episodes=10, render=True):
    """
    Test the agent using the provided Q-table.
    """
    # Use render_mode="human" for visual testing.
    env = gym.make('LunarLander-v3', render_mode="human")
    for episode in range(num_episodes):
        state, _ = env.reset()
        state_disc = discretize_state(state)
        total_reward = 0
        done = False
        while not done:
            # Always pick the best action.
            action = np.argmax(Q[state_disc])
            next_state, reward, done, truncated, info = env.step(action)
            done = done or truncated
            state_disc = discretize_state(next_state)
            total_reward += reward
            if render:
                env.render()
        print(f"Test Episode {episode+1}/{num_episodes}, Total Reward: {total_reward:.2f}")
    env.close()
ne=20000
def save_model(Q, filename=f"q_table{ne}.pkl"):
    with open(filename, 'wb') as f:
        pickle.dump(dict(Q), f)
    print(f"Model saved to {filename}")
ne = 10000
def load_model(filename=f"q_table{ne}.pkl"):
    with open(filename, 'rb') as f:
        Q_dict = pickle.load(f)
    Q = defaultdict(zero_q, Q_dict)
    print(f"Model loaded from {filename}")
    return Q

# --- Main Program ---
if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Train or test Q-learning model for LunarLander-v3")
    parser.add_argument('--train', action='store_true', help='Train the model')
    parser.add_argument('--test', action='store_true', help='Test the saved model')
    args = parser.parse_args()
    
    if args.train:
        try:
            Q = load_model()
            print("Resuming training from saved model...")
        except FileNotFoundError:
            Q = None
            print("No saved model found. Training from scratch...")
        Q = train_model(Q=Q, num_episodes=10000)
        save_model(Q)
    elif args.test:
        # Load and test the saved model.
        Q = load_model()
        test_model(Q, num_episodes=10)
    else:
        print("Please specify --train to train the model or --test to test the saved model.")
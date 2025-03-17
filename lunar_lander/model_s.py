import gymnasium as gym
import numpy as np
import pickle
from collections import defaultdict
import argparse

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
    create_bins(-1.0, 1.0, 10),     # x position
    create_bins(0.0, 1.5, 10),      # y position
    create_bins(-2.0, 2.0, 10),     # x velocity
    create_bins(-2.0, 2.0, 10),     # y velocity
    create_bins(-np.pi, np.pi, 10), # angle
    create_bins(-5.0, 5.0, 10)      # angular velocity
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
# Instead of a lambda (which is hard to pickle), we use a named function.
def zero_q():
    return np.zeros(4)

def train_model(Q=None, num_episodes=1000, alpha=0.1, gamma=0.99, 
                epsilon=1.0, epsilon_min=0.05, epsilon_decay=0.995, 
                render=False):
    # If no Q-table is provided, initialize a new one.
    if Q is None:
        Q = defaultdict(zero_q)
    rewards = []
    env = gym.make('LunarLander-v3')
    
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
            
            # Q-learning update.
            best_next_action = np.argmax(Q[next_state_disc])
            td_target = reward + gamma * Q[next_state_disc][best_next_action]
            td_error = td_target - Q[state_disc][action]
            Q[state_disc][action] += alpha * td_error
            
            state_disc = next_state_disc
            total_reward += reward
            
            if render:
                env.render()
                
        rewards.append(total_reward)
        
        if (episode + 1) % 200 == 0:
            mean_reward_200 = np.mean(rewards[-200:])
            print(f"Episode {episode+1}/{num_episodes}, Mean Reward (last 200 episodes): {mean_reward_200:.2f}")
        if (episode + 1) % 10000 == 0:
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




def save_model(Q, filename=f"q_table{ne}.pkl"):

    with open(filename, 'wb') as f:
        pickle.dump(dict(Q), f)
    print(f"Model saved to {filename}")

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

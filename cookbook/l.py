import gymnasium as gym
import numpy as np
import pickle
import matplotlib.pyplot as plt

def q_learning_frozenlake(is_training=True, render=False, num_episodes=2000):
    # Create the FrozenLake environment. "is_slippery=True" adds randomness.
    env = gym.make("FrozenLake-v1", is_slippery=True,render_mode='human' if render else None)

    # The environment's state and action spaces are discrete.
    n_states = env.observation_space.n  # e.g., 16 for a 4x4 grid
    n_actions = env.action_space.n      # e.g., 4 possible actions (left, down, right, up)

    # Initialize the Q-table to all zeros. Its shape is [n_states x n_actions].
    Q = np.zeros((n_states, n_actions))

    # Hyperparameters for Q-learning
    alpha = 0.8           # Learning rate: controls how much new experiences override old ones.
    gamma = 0.99          # Discount factor: emphasizes future rewards.
    epsilon = 1.0         # Starting value for epsilon (exploration rate).
    epsilon_decay = 0.995 # Decay factor for epsilon after each episode.
    min_epsilon = 0.01    # Minimum value for epsilon to ensure some exploration.

    rewards_per_episode = []

    # Run Q-learning for a number of episodes
    for episode in range(num_episodes):
        # Reset the environment at the start of each episode
        state, _ = env.reset()
        total_reward = 0
        done = False

        # Run the episode until termination (done becomes True)
        while not done:
            # Epsilon-greedy action selection:
            # With probability epsilon, choose a random action (exploration)
            # Otherwise, choose the action with the highest Q-value for the current state (exploitation)
            if is_training and np.random.rand() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(Q[state, :])
            
            # Take the action and receive the new state and reward
            next_state, reward, done, truncated, _ = env.step(action)
            
            # Q-learning update rule:
            # Q(s, a) = Q(s, a) + alpha * (reward + gamma * max_a' Q(s', a') - Q(s, a))
            Q[state, action] = Q[state, action] + alpha * (reward + gamma * np.max(Q[next_state, :]) - Q[state, action])
            
            # Update state and accumulate reward
            state = next_state
            total_reward += reward

            if render:
                env.render()

        # Decay epsilon to reduce the exploration rate over time
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        rewards_per_episode.append(total_reward)

        # Print some info every 100 episodes
        if episode:
            print(f"Episode {episode}: Total Reward = {total_reward}, Epsilon = {epsilon:.2f}")

    env.close()

    # Plot the rewards per episode to see the learning progress
    plt.plot(rewards_per_episode)
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.title("Rewards per Episode in FrozenLake")
    plt.show()

    # If training, save the Q-table for later use
    if is_training:
        with open("frozenlake_q.pkl", "wb") as f:
            pickle.dump(Q, f)

    return Q

if __name__ == "__main__":
    # Run the training process (set is_training=False to use a pre-trained Q-table)
    q_table = q_learning_frozenlake(is_training=False, render=True, num_episodes=20000)

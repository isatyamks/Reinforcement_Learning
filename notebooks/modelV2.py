import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(is_training=True, render=False):
    env = gym.make('Pendulum-v1', render_mode='human' if render else None)

    # Hyperparameters
    learning_rate_a = 0.1        # Learning rate (alpha)
    discount_factor_g = 0.9      # Discount factor (gamma)
    epsilon = 1.0                # Initial epsilon (for epsilon-greedy policy)
    epsilon_decay_rate = 0.0005  # Epsilon decay rate
    epsilon_min = 0.05           # Minimum epsilon
    divisions = 15               # Discretization divisions

    # Divide observation space into discrete segments
    x = np.linspace(env.observation_space.low[0], env.observation_space.high[0], divisions)
    y = np.linspace(env.observation_space.low[1], env.observation_space.high[1], divisions)
    w = np.linspace(env.observation_space.low[2], env.observation_space.high[2], divisions)

    # Divide action space into discrete segments
    a = np.linspace(env.action_space.low[0], env.action_space.high[0], divisions)

    if is_training:
        # Initialize Q-table with zeros
        q = np.zeros((len(x) + 1, len(y) + 1, len(w) + 1, len(a) + 1))
    else:
        # Load pretrained Q-table
        with open('pendulum.pkl', 'rb') as f:
            q = pickle.load(f)

    best_reward = -float('inf')
    rewards_per_episode = []  # List to store rewards for each episode
    i = 0

    while True:
        state, _ = env.reset()
        s_i0 = np.digitize(state[0], x)
        s_i1 = np.digitize(state[1], y)
        s_i2 = np.digitize(state[2], w)

        rewards = 0
        steps = 0

        # Episode
        while steps < 1000 or not is_training:
            if is_training and np.random.rand() < epsilon:
                # Choose random action
                action_idx = np.random.randint(0, len(a))
                action = a[action_idx]
            else:
                # Choose best action based on Q-table
                action_idx = np.argmax(q[s_i0, s_i1, s_i2, :])
                action = a[action_idx]

            # Take action and observe new state and reward
            new_state, reward, terminated, truncated, _ = env.step([action])

            # Discretize new state
            ns_i0 = np.digitize(new_state[0], x)
            ns_i1 = np.digitize(new_state[1], y)
            ns_i2 = np.digitize(new_state[2], w)

            # Update Q-table
            if is_training:
                q[s_i0, s_i1, s_i2, action_idx] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[ns_i0, ns_i1, ns_i2, :]) - q[s_i0, s_i1, s_i2, action_idx]
                )

            state = new_state
            s_i0, s_i1, s_i2 = ns_i0, ns_i1, ns_i2

            rewards += reward
            steps += 1

            if terminated or truncated:
                break

        # Update best reward and save Q-table
        if rewards > best_reward:
            best_reward = rewards
            if is_training:
                with open('pendulum.pkl', 'wb') as f:
                    pickle.dump(q, f)

        # Store rewards per episode
        rewards_per_episode.append(rewards)

        # Print stats
        if is_training and i % 100 == 0 and i != 0:
            mean_reward = np.mean(rewards_per_episode[-100:])
            print(f'Episode: {i}, Epsilon: {epsilon:.2f}, Best Reward: {best_reward:.2f}, Mean Rewards: {mean_reward:.2f}')

            # Graph mean rewards
            mean_rewards = [np.mean(rewards_per_episode[max(0, t - 100):t + 1]) for t in range(len(rewards_per_episode))]
            plt.plot(mean_rewards)
            plt.xlabel("Episodes")
            plt.ylabel("Mean Reward")
            plt.title("Training Performance")
            plt.savefig(f'pendulum.png')
            plt.close()

        elif not is_training:
            print(f'Episode: {i}, Reward: {rewards:.2f}')

        # Decay epsilon
        epsilon = max(epsilon - epsilon_decay_rate, epsilon_min)

        i += 1

        # Stop condition for testing (to prevent infinite loop)
        if not is_training and i >= 10:
            break

if __name__ == '__main__':
    # Training mode
    # run(is_training=True, render=False)

    # Testing mode
    run(is_training=False, render=True)

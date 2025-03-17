import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle

def run(is_training=True, render=False):
    env = gym.make('CartPole-v1', render_mode='human' if render else None)
    pos_space = np.linspace(-2.4, 2.4, 10)
    vel_space = np.linspace(-3, 3, 10)
    ang_space = np.linspace(-.2095, .2095, 10)
    ang_vel_space = np.linspace(-4, 4, 10)

    if is_training:
        # Q-table dimensions: discretized states x number of actions
        q = np.zeros((len(pos_space)+1, len(vel_space)+1,
                      len(ang_space)+1, len(ang_vel_space)+1,
                      env.action_space.n))
    else:
        with open('cartpole.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.1 
    discount_factor_g = 0.99 
    epsilon = 1         # Starting epsilon (used only during training)
    epsilon_decay_rate = 0.00001
    rng = np.random.default_rng()
    rewards_per_episode = []

    # --- Setup live plotting for training ---
    if is_training:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], '-o', label='Mean Reward per 100 Episodes')
        ax.set_xlabel('Episode Block (# of 100 Episodes)')
        ax.set_ylabel('Mean Reward (Last 100 Episodes)')
        ax.set_title('Training Progress: Mean Reward per 100 Episodes')
        ax.legend()
        plt.show()
    else:
        # --- Setup live plotting for testing ---
        plt.ion()
        fig, axs = plt.subplots(2, 2, figsize=(10, 8))
        fig.suptitle("Testing Metrics")
        # Initialize lists for metrics
        test_episodes = 100  # run testing for 100 episodes
        episodes_list = []
        rewards_list = []      # Total reward per episode
        lengths_list = []      # Episode lengths (number of steps)
        rolling_means = []     # Rolling mean over last N episodes
        epsilons_list = []     # Epsilon values (will be constant 0 in testing)
        window_size = 10       # window size for rolling mean

    episode = 0
    while True:
        state, _ = env.reset()
        # Discretize initial state
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False
        total_reward = 0
        step_count = 0  # track steps in current episode

        while not terminated and total_reward < 10000:
            if is_training and rng.random() < epsilon:
                action = env.action_space.sample()
            else:
                action = np.argmax(q[state_p, state_v, state_a, state_av, :])

            new_state, reward, terminated, _, _ = env.step(action)
            new_state_p = np.digitize(new_state[0], pos_space)
            new_state_v = np.digitize(new_state[1], vel_space)
            new_state_a = np.digitize(new_state[2], ang_space)
            new_state_av = np.digitize(new_state[3], ang_vel_space)

            if is_training:
                # Q-learning update rule
                q[state_p, state_v, state_a, state_av, action] += learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :])
                    - q[state_p, state_v, state_a, state_av, action]
                )

            state = new_state
            state_p, state_v, state_a, state_av = new_state_p, new_state_v, new_state_a, new_state_av
            total_reward += reward
            step_count += 1

            # if not is_training:
            #     print(f'Episode: {episode}  Reward: {total_reward}  Steps: {step_count}')

        rewards_per_episode.append(total_reward)

        if is_training:
            # Decay epsilon after each episode
            epsilon = max(epsilon - epsilon_decay_rate, 0)
            recent_mean = np.mean(rewards_per_episode[-100:])
            print(f'Episode: {episode}  Reward: {total_reward}  Epsilon: {epsilon:0.2f}  Mean (last 100): {recent_mean:0.1f}')

            # Every 100 episodes update the live plot for training
            if (episode + 1) % 100 == 0:
                block = (episode + 1) // 100
                block_mean = np.mean(rewards_per_episode[-100:])
                line.set_xdata(np.append(line.get_xdata(), block))
                line.set_ydata(np.append(line.get_ydata(), block_mean))
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)
                
                # Stop training early if performance criterion is met
                if block_mean > 1000:
                    break
        else:
            # --- Update testing metrics and live plots ---
            episodes_list.append(episode)
            rewards_list.append(total_reward)
            lengths_list.append(step_count)
            # Compute rolling mean reward over the last "window_size" episodes
            current_window = rewards_list[-window_size:] if len(rewards_list) >= window_size else rewards_list
            rolling_means.append(np.mean(current_window))
            # In testing we use a greedy policy, so epsilon is 0.
            epsilons_list.append(0)
            
            # Update each subplot
            axs[0, 0].cla()
            axs[0, 0].plot(episodes_list, rewards_list, '-o', color='blue')
            axs[0, 0].set_title("Episode vs Total Reward")
            axs[0, 0].set_xlabel("Episode")
            axs[0, 0].set_ylabel("Reward")
            
            axs[0, 1].cla()
            axs[0, 1].plot(episodes_list, lengths_list, '-o', color='green')
            axs[0, 1].set_title("Episode vs Episode Length")
            axs[0, 1].set_xlabel("Episode")
            axs[0, 1].set_ylabel("Steps")
            
            axs[1, 0].cla()
            axs[1, 0].plot(episodes_list, rolling_means, '-o', color='magenta')
            axs[1, 0].set_title(f"Rolling Mean Reward (window={window_size})")
            axs[1, 0].set_xlabel("Episode")
            axs[1, 0].set_ylabel("Mean Reward")
            
            axs[1, 1].cla()
            axs[1, 1].plot(episodes_list, epsilons_list, '-o', color='red')
            axs[1, 1].set_title("Epsilon vs Episode")
            axs[1, 1].set_xlabel("Episode")
            axs[1, 1].set_ylabel("Epsilon")
            
            plt.tight_layout(rect=[0, 0.03, 1, 0.95])
            plt.draw()
            plt.pause(0.001)
            
            # Stop testing after a fixed number of episodes
            if episode >= test_episodes - 1:
                break

        episode += 1

    env.close()

    # Save Q table to file if training
    if is_training:
        with open('cartpole.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Finalize the plot (turn off interactive mode)
    plt.ioff()
    if is_training:
        plt.savefig('cartpole_rewards.png')
    plt.show()

if __name__ == '__main__':
    # Set render=False during training to speed up episodes.
    # Change is_training to False to enter testing mode with live plots.
    run(is_training=False, render=False)

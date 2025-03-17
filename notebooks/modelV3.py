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
        q = np.zeros((len(pos_space)+1, len(vel_space)+1,
                      len(ang_space)+1, len(ang_vel_space)+1,
                      env.action_space.n))
    else:
        with open('cartpole.pkl', 'rb') as f:
            q = pickle.load(f)

    learning_rate_a = 0.1 
    discount_factor_g = 0.99 
    epsilon = 1         
    epsilon_decay_rate = 0.00001
    rng = np.random.default_rng()
    rewards_per_episode = []

    # Lists to store the block index and the corresponding mean rewards
    block_indices = []
    mean_rewards_100 = []

    # Setup live plotting for training
    if is_training:
        plt.ion()
        fig, ax = plt.subplots()
        line, = ax.plot([], [], '-o', label='Mean Reward per 100 Episodes')
        ax.set_xlabel('Episode Block (# of 100 Episodes)')
        ax.set_ylabel('Mean Reward (Last 100 Episodes)')
        ax.set_title('Training Progress: Mean Reward per 100 Episodes')
        ax.legend()
        plt.show()

    episode = 0
    while True:
        state, _ = env.reset()
        state_p = np.digitize(state[0], pos_space)
        state_v = np.digitize(state[1], vel_space)
        state_a = np.digitize(state[2], ang_space)
        state_av = np.digitize(state[3], ang_vel_space)

        terminated = False
        total_reward = 0

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

            if not is_training:
                print(f'Episode: {episode}  Reward: {total_reward}')

        rewards_per_episode.append(total_reward)
        
        if is_training:
            # Decay epsilon after each episode
            epsilon = max(epsilon - epsilon_decay_rate, 0)
            # Print the reward info for this episode
            recent_mean = np.mean(rewards_per_episode[-100:])
            print(f'Episode: {episode}  Reward: {total_reward}  Epsilon: {epsilon:0.2f}  Mean (last 100): {recent_mean:0.1f}')

            # Every 100 episodes update the live plot
            if (episode + 1) % 100 == 0:
                block = (episode + 1) // 100
                block_indices.append(block)
                block_mean = np.mean(rewards_per_episode[-100:])
                mean_rewards_100.append(block_mean)

                # Update plot data
                line.set_data(block_indices, mean_rewards_100)
                ax.relim()
                ax.autoscale_view()
                plt.draw()
                plt.pause(0.001)
                
                # Stop training if performance criterion is met
                if block_mean > 1000:
                    break

        episode += 1

    env.close()

    # Save Q table to file if training
    if is_training:
        with open('cartpole.pkl', 'wb') as f:
            pickle.dump(q, f)

    # Finalize the plot (turn off interactive mode)
    if is_training:
        plt.ioff()
        plt.savefig('cartpole_rewards.png')
        plt.show()

if __name__ == '__main__':
    # Set render=False during training to speed up episodes
    run(is_training=True, render=False)

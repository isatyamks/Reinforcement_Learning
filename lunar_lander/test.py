import time
import gymnasium as gym
import numpy as np

def test_model(Q, num_episodes=10):
    """Test the trained agent."""
    env = gym.make("LunarLander-v2", render_mode="human")  # ✅ Corrected here

    for episode in range(num_episodes):
        state, _ = env.reset()
        state = discretize(state)
        total_reward = 0

        while True:
            env.render()  # ✅ Now works!
            time.sleep(0.05)  # Optional: slows down rendering for visibility
            
            action = np.argmax(Q[state])
            next_state, reward, terminated, truncated, _ = env.step(action)
            next_state = discretize(next_state)
            state = next_state
            total_reward += reward

            if terminated or truncated:
                break

        print(f"Test Episode {episode + 1}: Total Reward = {total_reward}")

    env.close()


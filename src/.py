# import pygame
# import numpy as np
# import matplotlib.pyplot as plt
# import pickle
# import math
# import sys

# # Custom CartPole environment
# import pygame
# import numpy as np
# import math

# class CartPoleEnv:
#     def __init__(self, render=False, screen=None):
#         # Physics parameters
#         self.g = 9.8       # gravitational acceleration (m/s^2)
#         self.M = 1.0       # mass of the cart (kg)
#         self.m = 0.1       # mass of the pole (kg)
#         self.l = 0.5       # half-length of the pole (m)
#         self.force_mag = 10.0  # magnitude of force applied (N)
#         self.dt = 0.02     # time step (s)

#         # Rendering related parameters
#         self.render_enabled = render
#         self.screen = screen
#         self.screen_width = 800
#         self.screen_height = 600
#         self.scale = 100   # 1 meter = 100 pixels
#         self.cart_width = 50
#         self.cart_height = 30
#         self.cart_y = self.screen_height - 100 - self.cart_height

#         # Setup fonts if rendering is enabled
#         if self.render_enabled and self.screen is not None:
#             self.font = pygame.font.SysFont("Arial", 24)
#             self.large_font = pygame.font.SysFont("Arial", 48)

#         self.reset()

#     def reset(self):
#         # Initialize state: [cart position, cart velocity, pole angle, pole angular velocity]
#         self.state = np.array([0.0, 0.0, 0.05, 0.0])
#         self.steps = 0
#         self.game_over = False
#         return self.state

#     def step(self, action):
#         # Map action to force: 0 -> push left, 1 -> push right
#         F = self.force_mag if action == 1 else -self.force_mag
#         x, x_dot, theta, theta_dot = self.state

#         # Compute dynamics using Euler integration
#         total_mass = self.M + self.m
#         polemass_length = self.m * self.l
#         costheta = math.cos(theta)
#         sintheta = math.sin(theta)

#         temp = (F + polemass_length * theta_dot**2 * sintheta) / total_mass
#         theta_acc = (self.g * sintheta - costheta * temp) / (self.l * (4.0/3 - self.m * costheta**2 / total_mass))
#         x_acc = temp - polemass_length * theta_acc * costheta / total_mass

#         x_dot += x_acc * self.dt
#         x += x_dot * self.dt
#         theta_dot += theta_acc * self.dt
#         theta += theta_dot * self.dt

#         self.state = np.array([x, x_dot, theta, theta_dot])
#         self.steps += 1

#         # Termination conditions: if cart goes out-of-bound or pole falls too far (about 12°)
#         self.game_over = bool(abs(x) > 2.4 or abs(theta) > 0.2095)
#         reward = 1.0 if not self.game_over else 0.0

#         return self.state, reward, self.game_over

#     def render(self):
#         if not self.render_enabled or self.screen is None:
#             return

#         # Draw a pleasant light-blue background
#         self.screen.fill((200, 220, 255))

#         # Draw ground
#         ground_y = self.screen_height - 100
#         pygame.draw.line(self.screen, (50, 50, 50), (0, ground_y), (self.screen_width, ground_y), 4)

#         # Retrieve state for drawing
#         x, _, theta, _ = self.state
#         cart_screen_x = self.screen_width // 2 + int(x * self.scale) - self.cart_width // 2

#         # Draw the cart (black rectangle) with wheels
#         cart_rect = pygame.Rect(cart_screen_x, self.cart_y, self.cart_width, self.cart_height)
#         pygame.draw.rect(self.screen, (0, 0, 0), cart_rect)
#         wheel_radius = 8
#         left_wheel_center = (cart_screen_x + 10, self.cart_y + self.cart_height)
#         right_wheel_center = (cart_screen_x + self.cart_width - 10, self.cart_y + self.cart_height)
#         pygame.draw.circle(self.screen, (100, 100, 100), left_wheel_center, wheel_radius)
#         pygame.draw.circle(self.screen, (100, 100, 100), right_wheel_center, wheel_radius)

#         # Draw the pole (rod) with enhanced visual effects
#         pivot_x = cart_screen_x + self.cart_width // 2
#         pivot_y = self.cart_y
#         pole_length_pixels = int(2 * self.l * self.scale)  # full pole length
#         end_x = pivot_x + pole_length_pixels * math.sin(theta)
#         end_y = pivot_y - pole_length_pixels * math.cos(theta)
#         # Draw the rod with a thicker line
#         pygame.draw.line(self.screen, (255, 0, 0), (pivot_x, pivot_y), (end_x, end_y), 6)
#         # Draw a blue weight at the end of the pole
#         pygame.draw.circle(self.screen, (0, 0, 255), (int(end_x), int(end_y)), 10)

#         # Display the live score (number of steps survived)
#         score_text = self.font.render(f"Score: {self.steps}", True, (0, 0, 0))
#         self.screen.blit(score_text, (10, 10))

#         # If game over, display a large GAME OVER message in the center
#         if self.game_over:
#             game_over_text = self.large_font.render("GAME OVER", True, (255, 0, 0))
#             text_rect = game_over_text.get_rect(center=(self.screen_width // 2, self.screen_height // 2))
#             self.screen.blit(game_over_text, text_rect)

#         pygame.display.flip()

# # Main Q-learning loop integrated with the custom Pygame environment
# def run(is_training=True, render=False):

#     # Set up Pygame if rendering is enabled
#     if render:
#         pygame.init()
#         screen = pygame.display.set_mode((800, 600))
#         pygame.display.set_caption("CartPole Q-Learning")
#         clock = pygame.time.Clock()
#     else:
#         screen = None

#     # Create the environment
#     env = CartPoleEnv(render=render, screen=screen)

#     # Discretize state spaces (for cart position, velocity, pole angle, and angular velocity)
#     pos_space = np.linspace(-2.4, 2.4, 10)
#     vel_space = np.linspace(-3, 3, 10)
#     ang_space = np.linspace(-0.2095, 0.2095, 10)
#     ang_vel_space = np.linspace(-4, 4, 10)

#     # Initialize Q-table
#     if is_training:
#         q = np.zeros((len(pos_space)+1, len(vel_space)+1, len(ang_space)+1, len(ang_vel_space)+1, 2))
#     else:
#         with open('cartpole.pkl', 'rb') as f:
#             q = pickle.load(f)

#     learning_rate_a = 0.1 
#     discount_factor_g = 0.99 
#     epsilon = 1.0         
#     epsilon_decay_rate = 0.00001
#     rng = np.random.default_rng()
#     rewards_per_episode = []
#     i = 0

#     # Main loop: each iteration is an episode
#     while True:
#         state = env.reset()
#         state_p = np.digitize(state[0], pos_space)
#         state_v = np.digitize(state[1], vel_space)
#         state_a = np.digitize(state[2], ang_space)
#         state_av = np.digitize(state[3], ang_vel_space)

#         terminated = False
#         rewards = 0

#         while not terminated:
#             # Process Pygame events if rendering
#             if render:
#                 for event in pygame.event.get():
#                     if event.type == pygame.QUIT:
#                         pygame.quit()
#                         sys.exit()

#             # Choose an action using an epsilon-greedy policy
#             if is_training and rng.random() < epsilon:
#                 action = rng.integers(0, 2)  # randomly choose 0 or 1
#             else:
#                 action = np.argmax(q[state_p, state_v, state_a, state_av, :])

#             new_state, reward, terminated = env.step(action)
#             new_state_p = np.digitize(new_state[0], pos_space)
#             new_state_v = np.digitize(new_state[1], vel_space)
#             new_state_a = np.digitize(new_state[2], ang_space)
#             new_state_av = np.digitize(new_state[3], ang_vel_space)

#             # Update Q-table if training
#             if is_training:
#                 q[state_p, state_v, state_a, state_av, action] += learning_rate_a * (
#                     reward + discount_factor_g * np.max(q[new_state_p, new_state_v, new_state_a, new_state_av, :]) -
#                     q[state_p, state_v, state_a, state_av, action]
#                 )

#             state_p, state_v, state_a, state_av = new_state_p, new_state_v, new_state_a, new_state_av
#             rewards += reward

#             if render:
#                 env.render()
#                 clock.tick(50)  # Limit frame rate

#         rewards_per_episode.append(rewards)
#         mean_rewards = np.mean(rewards_per_episode[-100:])
#         if is_training:
#             print(f'Episode: {i}  Rewards: {rewards}  Epsilon: {epsilon:.2f}  Mean Rewards: {mean_rewards:.1f}')
#         else:
#             print(f'Episode: {i}  Rewards: {rewards}')

#         # Break if the average reward exceeds a threshold (indicating good performance)
#         if mean_rewards > 1000:
#             break

#         epsilon = max(epsilon - epsilon_decay_rate, 0)
#         i += 1

#     # Save Q-table if training
#     if is_training:
#         with open('cartpole.pkl', 'wb') as f:
#             pickle.dump(q, f)

#     # Plot rewards over episodes
#     plt.plot(rewards_per_episode)
#     plt.xlabel('Episode')
#     plt.ylabel('Rewards')
#     plt.title('Rewards per Episode')
#     plt.savefig('cartpole_rewards.png')

#     if render:
#         pygame.quit()

# if __name__ == '__main__':
#     # Uncomment one of the following lines:
#     # For training with visualization:
#     run(is_training=True, render=False)
#     # For testing with visualization:
#     # run(is_training=False, render=True)

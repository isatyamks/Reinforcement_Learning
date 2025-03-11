import pygame
import math

# Initialize pygame
pygame.init()

# World and observation dimensions
WORLD_WIDTH, WORLD_HEIGHT = 800, 600           # Full world size
CAMERA_WIDTH, CAMERA_HEIGHT = 96, 96            # Observation space (top-down view)
WINDOW_WIDTH, WINDOW_HEIGHT = 600, 600           # Display window size
FPS = 60

# Colors
GRASS_COLOR = (50, 150, 50)      # Background/grass
TRACK_COLOR = (100, 100, 100)    # Track fill color
BOUNDARY_COLOR = (255, 255, 255) # Track boundaries
CAR_COLOR = (255, 0, 0)          # Car color

# Create display window and world surface
screen = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
pygame.display.set_caption("Car Racing - Inspired by Gymnasium")
world = pygame.Surface((WORLD_WIDTH, WORLD_HEIGHT))

# --- Generate a Curved Track ---
# Define center points for a closed-loop track
track_center = [
    (200, 300),
    (300, 250),
    (500, 200),
    (600, 300),
    (650, 400),
    (600, 500),
    (500, 550),
    (300, 500),
    (200, 400)
]
TRACK_WIDTH = 40  # Offset distance for inner/outer boundaries

def compute_track_boundaries(center_points, offset):
    """For each center point, compute inner and outer boundary points by averaging the
    directions from the previous and next points and offsetting perpendicular to the path."""
    inner_points = []
    outer_points = []
    n = len(center_points)
    for i in range(n):
        prev_point = center_points[i-1]
        curr_point = center_points[i]
        next_point = center_points[(i+1) % n]
        # Compute tangent direction by taking vector from previous to next point
        tx = next_point[0] - prev_point[0]
        ty = next_point[1] - prev_point[1]
        length = math.hypot(tx, ty)
        if length != 0:
            tx /= length
            ty /= length
        # Left normal: (-ty, tx)
        nx, ny = -ty, tx
        outer_points.append((curr_point[0] + nx * offset, curr_point[1] + ny * offset))
        inner_points.append((curr_point[0] - nx * offset, curr_point[1] - ny * offset))
    return inner_points, outer_points

inner_boundary, outer_boundary = compute_track_boundaries(track_center, TRACK_WIDTH)
# Create a polygon that represents the track area by combining outer and inner boundaries
track_polygon = outer_boundary + inner_boundary[::-1]

# --- Car Properties and Physics ---
# Start the car near the first center point
car_x, car_y = track_center[0]
car_angle = 0       # Angle in degrees (0 faces right)
car_speed = 0

acceleration = 0.2       # Speed increase per frame when accelerating
brake_deceleration = 0.3 # Speed decrease per frame when braking
max_speed = 8
friction = 0.05          # Natural deceleration when no input is given
turn_speed = 3           # Turning factor (scaled by speed)

# Car dimensions (simple rectangle)
car_width, car_height = 20, 10

clock = pygame.time.Clock()
running = True

while running:
    dt = clock.tick(FPS) / 1000.0  # Time elapsed in seconds (not used explicitly here)
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

    # --- Process Input ---
    # Mapping arrow keys to discrete actions:
    # LEFT: steer left, RIGHT: steer right, UP: gas, DOWN: brake
    keys = pygame.key.get_pressed()
    steering = 0   # -1 for left, +1 for right
    gas = 0        # 1 if accelerating
    brake = 0      # 1 if braking

    if keys[pygame.K_LEFT]:
        steering = -1
    if keys[pygame.K_RIGHT]:
        steering = 1
    if keys[pygame.K_UP]:
        gas = 1
    if keys[pygame.K_DOWN]:
        brake = 1

    # --- Update Car Physics ---
    if gas:
        car_speed += acceleration
    if brake:
        car_speed -= brake_deceleration

    # Limit speed (allowing a reduced reverse speed)
    car_speed = max(-max_speed / 2, min(max_speed, car_speed))

    # Apply friction when no input is pressed
    if not gas and not brake:
        if car_speed > 0:
            car_speed -= friction
            if car_speed < 0:
                car_speed = 0
        elif car_speed < 0:
            car_speed += friction
            if car_speed > 0:
                car_speed = 0

    # Update car's orientation based on steering and current speed
    if car_speed != 0:
        car_angle += steering * turn_speed * (car_speed / max_speed)

    # Update car position based on its speed and direction
    rad = math.radians(car_angle)
    car_x += math.cos(rad) * car_speed
    car_y -= math.sin(rad) * car_speed  # Note: y axis increases downward in pygame

    # Ensure car stays within world bounds
    car_x = max(0, min(WORLD_WIDTH, car_x))
    car_y = max(0, min(WORLD_HEIGHT, car_y))

    # --- Render the World ---
    # Draw the grass background
    world.fill(GRASS_COLOR)
    # Draw the track area
    pygame.draw.polygon(world, TRACK_COLOR, track_polygon)
    # Draw the track boundaries for clarity
    pygame.draw.lines(world, BOUNDARY_COLOR, True, outer_boundary, 3)
    pygame.draw.lines(world, BOUNDARY_COLOR, True, inner_boundary, 3)

    # Draw the car as a rotated rectangle
    car_surface = pygame.Surface((car_width, car_height), pygame.SRCALPHA)
    car_surface.fill(CAR_COLOR)
    rotated_car = pygame.transform.rotate(car_surface, car_angle)
    car_rect = rotated_car.get_rect(center=(car_x, car_y))
    world.blit(rotated_car, car_rect.topleft)

    # --- Create Observation Space ---
    # Extract a 96x96 region centered on the car
    cam_x = int(car_x - CAMERA_WIDTH // 2)
    cam_y = int(car_y - CAMERA_HEIGHT // 2)
    # Clamp the camera to remain within world bounds
    cam_x = max(0, min(WORLD_WIDTH - CAMERA_WIDTH, cam_x))
    cam_y = max(0, min(WORLD_HEIGHT - CAMERA_HEIGHT, cam_y))
    camera_view = world.subsurface(pygame.Rect(cam_x, cam_y, CAMERA_WIDTH, CAMERA_HEIGHT))

    # Scale the observation view to fill the display window
    scaled_view = pygame.transform.scale(camera_view, (WINDOW_WIDTH, WINDOW_HEIGHT))
    screen.blit(scaled_view, (0, 0))
    
    pygame.display.flip()

pygame.quit()

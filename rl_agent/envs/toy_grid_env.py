import gymnasium as gym
from gymnasium import spaces
import numpy as np
from energy_model.ina219_energy_model import INA219EnergyModel

class ToyGridEnv(gym.Env):
    """
    Energy-Aware Custom Grid Environment with Exploration Matrix State.
    """
    def __init__(self, grid_size=5):
        super(ToyGridEnv, self).__init__()
        self.grid_size = grid_size
        
        # Actions: 0:Up, 1:Down, 2:Left, 3:Right
        self.action_space = spaces.Discrete(4)
        
        # Observation Space: Robot Pos (2) + Full Grid Coverage Matrix (5x5 = 25) = 27 Values
        self.observation_space = spaces.Box(
            low=0, high=1, shape=(2 + grid_size * grid_size,), dtype=np.float32
        )
        
        self.energy_sensor = INA219EnergyModel(voltage_nom=12.0)
        self.reset()

    def _get_obs(self):
        # Normalize robot position between 0 and 1
        pos_norm = self.robot_pos / (self.grid_size - 1.0)
        # Flatten visited matrix (0 for unvisited, 1 for visited)
        grid_flat = self.grid_matrix.flatten()
        return np.concatenate([pos_norm, grid_flat]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.robot_pos = np.array([0, 0], dtype=np.int32)
        
        # 5x5 Matrix to track visited cells (0 = Unvisited, 1 = Visited)
        self.grid_matrix = np.zeros((self.grid_size, self.grid_size), dtype=np.float32)
        self.grid_matrix[0, 0] = 1.0  # Starting cell is visited
        
        self.visited_count = 1
        self.steps_taken = 0
        self.max_steps = 50
        
        return self._get_obs(), {}

    def step(self, action):
        self.steps_taken += 1
        hit_wall = False
        
        # Movement Logic with Boundary Checks
        if action == 0:   # Up
            if self.robot_pos[1] < self.grid_size - 1:
                self.robot_pos[1] += 1
            else: hit_wall = True
        elif action == 1: # Down
            if self.robot_pos[1] > 0:
                self.robot_pos[1] -= 1
            else: hit_wall = True
        elif action == 2: # Left
            if self.robot_pos[0] > 0:
                self.robot_pos[0] -= 1
            else: hit_wall = True
        elif action == 3: # Right
            if self.robot_pos[0] < self.grid_size - 1:
                self.robot_pos[0] += 1
            else: hit_wall = True

        x, y = self.robot_pos
        
        # Energy Calculation
        energy_spent = self.energy_sensor.get_energy_consumption(action_type="flat", step_duration_sec=1.0)
        energy_penalty = 0.05 * energy_spent  # 0.60 Joules cost

        # Reward Logic
        if hit_wall:
            reward = -5.0 - energy_penalty  # Deewar se takrane ki bhari penalty (-5.6)
        elif self.grid_matrix[x, y] == 0.0:
            self.grid_matrix[x, y] = 1.0
            self.visited_count += 1
            reward = 10.0 - energy_penalty  # Naya cell explore karne ka (+9.4)
        else:
            reward = -1.0 - energy_penalty  # Purane cell par wapas aane ki penalty (-1.6)

        # Check Done Condition
        terminated = self.visited_count == (self.grid_size * self.grid_size)
        truncated = self.steps_taken >= self.max_steps
        
        return self._get_obs(), reward, terminated, truncated, {"energy_spent": energy_spent}
import gymnasium as gym
from gymnasium import spaces
import numpy as np
import math
from .base import RescueBotEnv
from .models import RescueAction

class RescueBotGym(gym.Env):
    """
    Standard Gymnasium wrapper for RescueBotEnv.
    Used for local RL training with Stable Baselines3.
    """
    def __init__(self, task_id="easy", enable_sync=False):
        super(RescueBotGym, self).__init__()
        self.internal_env = RescueBotEnv(task_id=task_id, enable_sync=enable_sync)
        
        # Action space: [move_val, rotate_val, interact_val, extinguish_val, scan_val]
        # interact_val: > 0.5 (Grab), < -0.5 (Drop)
        # extinguish_val: > 0.5 (Extinguish)
        # scan_val: > 0.5 (Scan)
        self.action_space = spaces.Box(
            low=np.array([-1.0, -1.0, -1.0, -1.0, -1.0]), 
            high=np.array([1.0, 1.0, 1.0, 1.0, 1.0]), 
            dtype=np.float32
        )
        
        # Observation space: 8 lidar sensors + x, y, heading, battery, integrity, carrying
        self.observation_space = spaces.Box(
            low=0.0, 
            high=1.0, 
            shape=(14,), 
            dtype=np.float32
        )

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        obs_pydantic = self.internal_env.reset()
        return self._preprocess_obs(obs_pydantic), {}

    def step(self, action):
        total_reward = 0.0
        
        # 1. Map continuous RL actions to OpenEnv structured actions
        move_val = float(action[0])
        rot_val = float(action[1])
        interact_val = float(action[2])
        extinguish_val = float(action[3])
        scan_val = float(action[4])
        
        # Accumulate rewards from all internal steps to ensure agent sees feedback
        
        # Rotate
        _, r, d, _ = self.internal_env.step(RescueAction(action_type="rotate", value=rot_val))
        total_reward += r
        if d: return self._return_step(total_reward, d)
        
        # Grab/Drop
        if interact_val > 0.5:
            _, r, d, _ = self.internal_env.step(RescueAction(action_type="grab"))
            total_reward += r
            if d: return self._return_step(total_reward, d)
        elif interact_val < -0.5:
            _, r, d, _ = self.internal_env.step(RescueAction(action_type="drop"))
            total_reward += r
            if d: return self._return_step(total_reward, d)
            
        # Extinguish
        if extinguish_val > 0.5:
            _, r, d, _ = self.internal_env.step(RescueAction(action_type="extinguish"))
            total_reward += r
            if d: return self._return_step(total_reward, d)
            
        # Scan
        if scan_val > 0.5:
            _, r, d, _ = self.internal_env.step(RescueAction(action_type="scan"))
            total_reward += r
            if d: return self._return_step(total_reward, d)

        # Move (Final step of the macro-action)
        obs_pydantic, r, d, info = self.internal_env.step(RescueAction(action_type="move", value=move_val))
        total_reward += r
        
        return self._preprocess_obs(obs_pydantic), float(total_reward), d, False, info

    def _return_step(self, reward, done):
        """Helper to return early if a sub-action triggers terminal state."""
        obs_pydantic = self.internal_env._get_observation()
        return self._preprocess_obs(obs_pydantic), float(reward), done, False, {}

    def _preprocess_obs(self, obs):
        lidar = [s.distance / 15.0 for s in obs.sensors] # Normalize by max possible scan dist
        
        # Robot state
        pos_x = obs.robot.position.x / self.internal_env.grid_size
        pos_y = obs.robot.position.y / self.internal_env.grid_size
        heading = obs.robot.heading / (2 * math.pi)
        battery = obs.robot.battery
        integrity = obs.robot.hull_integrity
        carrying = 1.0 if obs.robot.carrying else 0.0
        
        arr = np.array(lidar + [pos_x, pos_y, heading, battery, integrity, carrying], dtype=np.float32)
        return np.clip(arr, 0.0, 1.0)

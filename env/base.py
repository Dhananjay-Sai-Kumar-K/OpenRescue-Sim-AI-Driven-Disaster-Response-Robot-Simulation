import math
import random
from typing import List, Tuple, Dict, Any, Optional
from .models import (
    RescueObservation,
    RescueAction,
    RescueReward,
    RobotState,
    Position,
    SensorReading
)

class RescueBotEnv:
    """
    OpenEnv Rescue Robot Environment.
    Simulates a robot navigating a 2D disaster zone.
    """

    def __init__(self, task_id: str = "easy", enable_sync: bool = False):
        self.task_id = task_id
        self.enable_sync = enable_sync
        self.grid_size = 20.0  # units (e.g., meters)
        self.max_steps = 1000
        self.reset()

    def reset(self) -> RescueObservation:
        """Resets the environment and returns the initial observation."""
        self.steps = 0
        self.robot_pos = Position(x=1.0, y=1.0)
        self.robot_heading = 0.0
        self.battery = 1.0
        self.hull_integrity = 1.0
        self.carrying = None
        self.done = False

        # --- Dynamic Environment Setup Based on Task ---
        if self.task_id == "easy":
            # Simple Navigation: Reach safe zone (18, 18) from (1, 1). No victims or fires.
            self.victims = []
            self.fire_zones = []
            self.obstacles = [
                Position(x=5.0, y=5.0),
                Position(x=6.0, y=5.0),
                Position(x=12.0, y=12.0),
                Position(x=13.0, y=12.0),
            ]

        elif self.task_id == "medium":
            # Victim Locating: Find thermal signature and move to safe zone.
            # Adds more victims to search for.
            self.victims = [
                {"id": "v1", "pos": Position(x=10.0, y=10.0), "rescued": False, "type": "victim"},
                {"id": "v2", "pos": Position(x=15.0, y=5.0), "rescued": False, "type": "victim"},
                {"id": "v3", "pos": Position(x=5.0, y=15.0), "rescued": False, "type": "victim"}
            ]
            self.fire_zones = [
                {"pos": Position(x=12.0, y=12.0), "intensity": 0.3}
            ]
            self.obstacles = [
                Position(x=7.0, y=3.0),
                Position(x=8.0, y=3.0),
                Position(x=2.0, y=12.0),
                Position(x=16.0, y=14.0),
            ]

        elif self.task_id == "hard":
            # Hazardous Extraction: Multiple fires and debris layout.
            self.victims = [
                {"id": "v1", "pos": Position(x=8.0, y=17.0), "rescued": False, "type": "victim"},
                {"id": "v2", "pos": Position(x=17.0, y=8.0), "rescued": False, "type": "victim"}
            ]
            self.fire_zones = [
                {"pos": Position(x=5.0, y=15.0), "intensity": 0.6},
                {"pos": Position(x=15.0, y=5.0), "intensity": 0.8},
                {"pos": Position(x=18.0, y=15.0), "intensity": 0.4}
            ]
            # Create a "debris corridor"
            self.obstacles = [Position(x=x, y=10.0) for x in range(2, 18, 2)]

        self.safe_zone = Position(x=18.0, y=18.0)
        self.prev_dist_to_target = self._dist(self.robot_pos, self.safe_zone)
        
        return self._get_observation()

    def step(self, action: RescueAction) -> Tuple[RescueObservation, float, bool, Dict[str, Any]]:
        if self.done:
            return self._get_observation(), 0.0, True, {"info": "Episode already finished."}

        self.steps += 1
        reward_breakdown = {"progress": 0.0, "safety": 0.0, "rescue": 0.0, "penalty": 0.0}
        battery_drain = 0.002  # Default base drain
        self.scan_active = False
        
        # 1. Process Action
        if action.action_type == "move":
            battery_drain += abs(action.value) * 0.01
            # Simple physics-lite move
            dist = max(-0.5, min(0.5, action.value * 0.5))  # Clipped speed
            new_x = self.robot_pos.x + dist * math.cos(self.robot_heading)
            new_y = self.robot_pos.y + dist * math.sin(self.robot_heading)
            
            # Boundary & Collision Check
            if 0 <= new_x <= self.grid_size and 0 <= new_y <= self.grid_size:
                if not self._check_collision(new_x, new_y):
                    self.robot_pos.x = new_x
                    self.robot_pos.y = new_y
                else:
                    reward_breakdown["penalty"] -= 0.5  # Heavy collision penalty
                    self.hull_integrity -= 0.05
            else:
                reward_breakdown["penalty"] -= 0.2  # Boundary penalty
                self.hull_integrity -= 0.02
        
        elif action.action_type == "rotate":
            battery_drain += abs(action.value) * 0.005
            # Apply rotation with clipping
            rot_val = max(-0.5, min(0.5, action.value))
            self.robot_heading = (self.robot_heading + rot_val) % (2 * math.pi)

        elif action.action_type == "grab":
            battery_drain += 0.005
            # Try to grab a nearby victim if not already carrying one
            if self.carrying is None:
                for v in self.victims:
                    if not v["rescued"] and self._dist(self.robot_pos, v["pos"]) < 1.0:
                        self.carrying = v["id"]
                        reward_breakdown["rescue"] += 2.0 # Discovery bonus
                        break
            
        elif action.action_type == "drop":
            battery_drain += 0.005
            if self.carrying:
                # Check if we are in the safe zone
                if self._dist(self.robot_pos, self.safe_zone) < 2.0:
                    for v in self.victims:
                        if v["id"] == self.carrying:
                            v["rescued"] = True
                            reward_breakdown["rescue"] += 15.0 # Mission success bonus
                            break
                    self.carrying = None
                else:
                    # Dropped outside safe zone - minor penalty
                    self.carrying = None
                    reward_breakdown["penalty"] -= 0.5
                    
        elif action.action_type == "extinguish":
            battery_drain += 0.02
            for fire in self.fire_zones:
                if self._dist(self.robot_pos, fire["pos"]) < 2.5:
                    fire["intensity"] -= 0.3
                    reward_breakdown["progress"] += 2.0
            self.fire_zones = [f for f in self.fire_zones if f["intensity"] > 0]
            
        elif action.action_type == "scan":
            battery_drain += 0.015
            self.scan_active = True

        # 2. Environmental Effects & Dense Reward Shaping
        self.battery -= battery_drain
        reward_breakdown["penalty"] -= 0.01 # Small step penalty (encourages speed)
        
        # Proximity to fire reduces hull integrity
        for fire in self.fire_zones:
            dist_to_fire = self._dist(self.robot_pos, fire["pos"])
            if dist_to_fire < 2.5:
                dmg = (2.5 - dist_to_fire) * 0.05
                self.hull_integrity -= dmg
                reward_breakdown["safety"] -= dmg * 30.0 # High penalty for fire exposure

        # 3. Dense Progress Reward (Task-Specific)
        if self.task_id == "easy":
            current_dist_to_target = self._dist(self.robot_pos, self.safe_zone)
            progress_delta = self.prev_dist_to_target - current_dist_to_target
            reward_breakdown["progress"] += progress_delta * 2.0 
            self.prev_dist_to_target = current_dist_to_target
            
            if current_dist_to_target < 1.0:
                self.done = True
                reward_breakdown["progress"] += 5.0 # Goal bonus

        elif self.task_id == "medium" or self.task_id == "hard":
            # Find the active target (either a victim or the safe zone)
            if self.carrying is None:
                # Target is the nearest non-rescued victim
                active_victims = [v for v in self.victims if not v["rescued"]]
                if active_victims:
                    target = active_victims[0]["pos"]
                else:
                    target = self.safe_zone
                    self.done = True # All rescued
            else:
                # Target is the Safe Zone
                target = self.safe_zone
                
            current_dist_to_target = self._dist(self.robot_pos, target)
            progress_delta = self.prev_dist_to_target - current_dist_to_target
            reward_breakdown["progress"] += progress_delta * 3.0 # Stronger incentive
            self.prev_dist_to_target = current_dist_to_target

            # Final check for all victims rescued
            if all(v["rescued"] for v in self.victims):
                self.done = True

        # Check terminal conditions
        if self.battery <= 0 or self.hull_integrity <= 0:
            self.done = True
            reward_breakdown["penalty"] -= 5.0 # Failure penalty
        elif self.steps >= self.max_steps:
            self.done = True

        obs = self._get_observation()
        total_reward = sum(reward_breakdown.values())
        
        reward_obj = RescueReward(
            total_reward=total_reward,
            breakdown=reward_breakdown,
            done=self.done,
            info=f"Step {self.steps} completed. Health: {self.hull_integrity:.2f}, Battery: {self.battery:.2f}"
        )
        
        # SYNC TO DISK FOR VISUALIZATION (Only if enabled)
        if self.enable_sync:
            self._sync_to_disk()

        return obs, total_reward, self.done, reward_obj.dict()

    def _sync_to_disk(self):
        """Saves current state to a file for the visualization server to read."""
        import json
        import os
        temp_file = ".env_state.json.tmp"
        try:
            with open(temp_file, "w") as f:
                json.dump(self.state(), f)
            # Atomic rename (Windows supports this with os.replace)
            os.replace(temp_file, ".env_state.json")
        except Exception:
            pass # Avoid crashing during IO wait

    def state(self) -> Dict[str, Any]:
        """Returns the full internal ground truth."""
        def to_dict(obj):
            if hasattr(obj, 'dict'):
                return obj.dict()
            if isinstance(obj, list):
                return [to_dict(i) for i in obj]
            if isinstance(obj, dict):
                return {k: to_dict(v) for k, v in obj.items()}
            # Float rounding for faster JSON
            if isinstance(obj, float):
                return round(obj, 3)
            return obj

        return {
            "robot": to_dict(self.robot_pos),
            "heading": round(self.robot_heading, 3),
            "victims": to_dict(self.victims),
            "fire_zones": to_dict(self.fire_zones),
            "integrity": round(self.hull_integrity, 3),
            "battery": round(self.battery, 3)
        }

    def _get_observation(self) -> RescueObservation:
        # Generate simulated lidar/camera readings
        sensors = []
        for angle in range(0, 360, 45):
            rad = math.radians(angle + math.degrees(self.robot_heading))
            sensors.append(self._simulate_radar(rad))
            
        return RescueObservation(
            robot=RobotState(
                position=self.robot_pos,
                heading=self.robot_heading,
                battery=self.battery,
                hull_integrity=self.hull_integrity,
                carrying=self.carrying
            ),
            sensors=sensors,
            current_task=self.task_id,
            time_remaining=self.max_steps - self.steps
        )

    def _dist(self, p1: Position, p2: Position) -> float:
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _check_collision(self, x: float, y: float) -> bool:
        for obs_pos in self.obstacles:
            if math.sqrt((x - obs_pos.x)**2 + (y - obs_pos.y)**2) < 0.8:
                return True
        for fire in self.fire_zones:
            if math.sqrt((x - fire["pos"].x)**2 + (y - fire["pos"].y)**2) < 0.8:
                return True
        for v in self.victims:
            if not v["rescued"] and math.sqrt((x - v["pos"].x)**2 + (y - v["pos"].y)**2) < 0.5:
                return True
        return False

    def _simulate_radar(self, rad: float) -> SensorReading:
        # Ray-casting simulation checking obstacles, fire, and victims
        max_dist = 15.0 if getattr(self, "scan_active", False) else 8.0
        step_sz = 0.5
        for step in range(1, int(max_dist / step_sz)):
            cx = self.robot_pos.x + step * step_sz * math.cos(rad)
            cy = self.robot_pos.y + step * step_sz * math.sin(rad)
            
            if not (0 <= cx <= self.grid_size and 0 <= cy <= self.grid_size):
                return SensorReading(angle=rad, distance=step*step_sz, type="obstacle")
            
            pos_cx_cy = Position(x=cx, y=cy)
            for obs in self.obstacles:
                if self._dist(pos_cx_cy, obs) < 0.8:
                    return SensorReading(angle=rad, distance=step*step_sz, type="obstacle")
            for f in self.fire_zones:
                if self._dist(pos_cx_cy, f["pos"]) < 1.0:
                    return SensorReading(angle=rad, distance=step*step_sz, type="fire")
            for v in self.victims:
                if not v["rescued"] and self._dist(pos_cx_cy, v["pos"]) < 0.8:
                    return SensorReading(angle=rad, distance=step*step_sz, type="victim")
                    
        return SensorReading(angle=rad, distance=max_dist, type="safe_zone")
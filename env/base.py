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

    Fixes applied (v3):
      A. escape_steps: after a successful grab, collision checks are
         suspended for 5 steps so the bot can pull away from the pickup
         position without instantly bouncing off the surrounding geometry.
      B. Wall-proximity speed clamp: before applying a move the env
         checks the nearest obstacle distance in the travel direction and
         scales the effective speed down to 0.25 (50 % of normal) when
         within 1.2 m.
      C. Safe grab approach: the grab action now selects the best open
         cell around the victim and teleports the robot 0.9 m toward that
         cell before locking the carry.
      D. [NEW] Anti-oscillation on grab:
           - prev_dist_to_target is reset to the distance from the NEW
             robot position to the safe_zone immediately after a grab.
             Without this, the very next progress_delta is computed against
             the old victim-target distance, producing a large spurious
             negative reward that causes the agent to reverse direction.
           - robot_heading is snapped toward safe_zone after grab so the
             agent doesn't spend battery oscillating to reconcile its
             pre-grab heading with the new carry-target direction.
      E. [NEW] Target-switch guard in progress reward: on the step where
         carrying flips from None → victim_id the target changes from
         victim_pos → safe_zone. prev_dist_to_target is now always
         rewritten to the current distance to whatever the new target is
         before the delta is computed, preventing a one-step spike.
    """

    # ------------------------------------------------------------------ #
    # Constants                                                            #
    # ------------------------------------------------------------------ #
    GRAB_RADIUS        = 1.5   # distance at which grab succeeds
    COLLISION_RADIUS   = 0.8   # hard-obstacle / victim blocking radius
    ESCAPE_STEPS       = 5     # steps of collision immunity after grab
    WALL_CLAMP_DIST    = 1.2   # metres – begin speed reduction
    WALL_CLAMP_SPEED   = 0.25  # reduced effective speed near walls
    SAFE_APPROACH_DIST = 0.9   # metres – offset from victim for grab

    def __init__(self, task_id: str = "easy", enable_sync: bool = False):
        self.task_id = task_id
        self.enable_sync = enable_sync
        self.grid_size = 20.0
        self.max_steps = 1000
        self.debris = []
        self.escape_steps_remaining = 0
        self.reset()

    # ------------------------------------------------------------------ #
    # reset                                                                #
    # ------------------------------------------------------------------ #
    def reset(self) -> RescueObservation:
        self.steps = 0
        self.robot_pos = Position(x=1.0, y=1.0)
        self.robot_heading = 0.0
        self.battery = 1.0
        self.hull_integrity = 1.0
        self.carrying = None
        self.done = False
        self.escape_steps_remaining = 0

        if self.task_id == "easy":
            self.victims = []
            self.fire_zones = []
            self.obstacles = [
                Position(x=5.0,  y=5.0),
                Position(x=6.0,  y=5.0),
                Position(x=12.0, y=12.0),
                Position(x=13.0, y=12.0),
            ]

        elif self.task_id == "medium":
            self.victims = [
                {"id": "v1", "pos": Position(x=10.0, y=10.0), "rescued": False, "type": "victim"},
                {"id": "v2", "pos": Position(x=15.0, y=5.0),  "rescued": False, "type": "victim"},
                {"id": "v3", "pos": Position(x=5.0,  y=15.0), "rescued": False, "type": "victim"},
            ]
            self.fire_zones = [{"pos": Position(x=12.0, y=12.0), "intensity": 0.3}]
            self.obstacles = [
                Position(x=7.0,  y=3.0),
                Position(x=8.0,  y=3.0),
                Position(x=2.0,  y=12.0),
                Position(x=16.0, y=14.0),
            ]

        elif self.task_id == "hard":
            self.victims = [
                {"id": "v1", "pos": Position(x=8.0,  y=17.0), "rescued": False, "type": "victim"},
                {"id": "v2", "pos": Position(x=17.0, y=8.0),  "rescued": False, "type": "victim"},
            ]
            self.fire_zones = [
                {"pos": Position(x=5.0,  y=15.0), "intensity": 0.6},
                {"pos": Position(x=15.0, y=5.0),  "intensity": 0.8},
                {"pos": Position(x=18.0, y=15.0), "intensity": 0.4},
            ]
            self.debris = [
                {"id": f"d{i}", "pos": Position(x=x, y=10.0), "type": "debris"}
                for i, x in enumerate(range(2, 18, 2))
            ]
            self.obstacles = [Position(x=5.0, y=15.0), Position(x=15.0, y=5.0)]

        self.safe_zone = Position(x=18.0, y=18.0)
        self.prev_dist_to_target = self._dist(self.robot_pos, self.safe_zone)
        self.target_dist = self.prev_dist_to_target
        self.target_angle = 0.0
        self.stuck_steps = 0
        return self._get_observation()

    # ------------------------------------------------------------------ #
    # step                                                                 #
    # ------------------------------------------------------------------ #
    def step(self, action: RescueAction) -> Tuple[RescueObservation, float, bool, Dict[str, Any]]:
        if self.done:
            return self._get_observation(), 0.0, True, {"info": "Episode already finished."}

        self.steps += 1
        reward_breakdown = {"progress": 0.0, "safety": 0.0, "rescue": 0.0, "penalty": 0.0}
        battery_drain = 0.002
        self.scan_active = False
        last_pos_snapshot = Position(x=self.robot_pos.x, y=self.robot_pos.y)

        # Tick down escape immunity
        if self.escape_steps_remaining > 0:
            self.escape_steps_remaining -= 1

        # Snapshot carrying state BEFORE processing action so we can
        # detect a grab transition for FIX E.
        carrying_before = self.carrying

        # ---------------------------------------------------------------- #
        # 1. Process Action                                                  #
        # ---------------------------------------------------------------- #
        if action.action_type == "move":
            battery_drain += abs(action.value) * 0.01

            raw_speed = max(-0.5, min(0.5, action.value * 0.5))

            # FIX B: wall-proximity speed clamp
            effective_speed = self._clamp_speed_near_wall(raw_speed)

            new_x = self.robot_pos.x + effective_speed * math.cos(self.robot_heading)
            new_y = self.robot_pos.y + effective_speed * math.sin(self.robot_heading)

            if 0 <= new_x <= self.grid_size and 0 <= new_y <= self.grid_size:
                # FIX A: skip collision check during escape window
                in_escape = self.escape_steps_remaining > 0
                collision  = False if in_escape else self._check_collision(new_x, new_y)

                if not collision:
                    self.robot_pos.x = new_x
                    self.robot_pos.y = new_y
                    if self.carrying:
                        for v in self.victims:
                            if v["id"] == self.carrying:
                                v["pos"].x = new_x
                                v["pos"].y = new_y
                        for d in self.debris:
                            if d["id"] == self.carrying:
                                d["pos"].x = new_x
                                d["pos"].y = new_y
                else:
                    reward_breakdown["penalty"] -= 0.5
                    is_hard_obstacle = any(
                        math.sqrt((new_x - o.x)**2 + (new_y - o.y)**2) < 0.8
                        for o in self.obstacles
                    )
                    if is_hard_obstacle:
                        self.hull_integrity -= 0.05
            else:
                reward_breakdown["penalty"] -= 0.2
                self.hull_integrity -= 0.02

        elif action.action_type == "rotate":
            battery_drain += abs(action.value) * 0.005
            rot_val = max(-0.5, min(0.5, action.value))
            self.robot_heading = (self.robot_heading + rot_val) % (2 * math.pi)

        elif action.action_type == "grab":
            battery_drain += 0.005
            if self.carrying is None:
                for v in self.victims:
                    if not v["rescued"] and self._dist(self.robot_pos, v["pos"]) < self.GRAB_RADIUS:
                        # FIX C: reposition to safe approach point
                        safe_pos = self._find_safe_approach(v["pos"])
                        if safe_pos:
                            self.robot_pos.x = safe_pos.x
                            self.robot_pos.y = safe_pos.y

                        self.carrying = v["id"]
                        # FIX A: grant escape immunity
                        self.escape_steps_remaining = self.ESCAPE_STEPS
                        reward_breakdown["rescue"] += 5.0

                        # ---- FIX D: reset distance baseline & heading ----
                        # prev_dist_to_target was pointing at the victim;
                        # target is now safe_zone.  Resetting here means the
                        # VERY NEXT progress delta is computed correctly and
                        # won't produce a large spurious negative that sends
                        # the agent oscillating back toward the victim.
                        self.prev_dist_to_target = self._dist(self.robot_pos, self.safe_zone)

                        # Snap heading toward safe_zone so the agent exits
                        # the grab position already facing its new goal.
                        dx = self.safe_zone.x - self.robot_pos.x
                        dy = self.safe_zone.y - self.robot_pos.y
                        self.robot_heading = math.atan2(dy, dx)
                        break

                if self.carrying is None:
                    for d in self.debris:
                        if self._dist(self.robot_pos, d["pos"]) < self.GRAB_RADIUS:
                            safe_pos = self._find_safe_approach(d["pos"])
                            if safe_pos:
                                self.robot_pos.x = safe_pos.x
                                self.robot_pos.y = safe_pos.y
                            self.carrying = d["id"]
                            self.escape_steps_remaining = self.ESCAPE_STEPS

                            # FIX D: same baseline + heading reset for debris
                            self.prev_dist_to_target = self._dist(self.robot_pos, self.safe_zone)
                            dx = self.safe_zone.x - self.robot_pos.x
                            dy = self.safe_zone.y - self.robot_pos.y
                            self.robot_heading = math.atan2(dy, dx)
                            break

        elif action.action_type == "drop":
            battery_drain += 0.005
            if self.carrying:
                if self._dist(self.robot_pos, self.safe_zone) < 2.0:
                    for v in self.victims:
                        if v["id"] == self.carrying:
                            v["rescued"] = True
                            reward_breakdown["rescue"] += 15.0
                            break
                    self.carrying = None
                else:
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

        # ---------------------------------------------------------------- #
        # 2. Environmental Effects & Dense Reward Shaping                   #
        # ---------------------------------------------------------------- #
        self.battery -= battery_drain * 0.1
        reward_breakdown["penalty"] -= 0.001

        for fire in self.fire_zones:
            dist_to_fire = self._dist(self.robot_pos, fire["pos"])
            if dist_to_fire < 1.5:
                self.hull_integrity -= 0.01
                reward_breakdown["safety"] -= 5.0

        # ---------------------------------------------------------------- #
        # 3. Dense Progress Reward (Task-Specific Target Selection)         #
        # ---------------------------------------------------------------- #
        if self.task_id == "easy":
            target = self.safe_zone
            if self._dist(self.robot_pos, self.safe_zone) < 1.0:
                self.done = True
                reward_breakdown["progress"] += 10.0 # Clear goal bonus

        else: # medium / hard
            if self.carrying is None:
                active_victims = [v for v in self.victims if not v["rescued"]]
                if active_victims:
                    target = active_victims[0]["pos"]
                else:
                    target = self.safe_zone
                    if self._dist(self.robot_pos, self.safe_zone) < 1.5:
                        self.done = True
            else:
                target = self.safe_zone

        # STUCK DETECTOR
        if action.action_type == "move" and abs(action.value) > 0.1:
            dist_moved = self._dist(self.robot_pos, last_pos_snapshot)
            if dist_moved < 0.01:
                self.stuck_steps += 1
            else:
                self.stuck_steps = 0
            
            if self.stuck_steps > 5:
                reward_breakdown["penalty"] -= 2.0 # Significant penalty for being stuck
        else:
            self.stuck_steps = 0

        # ---- COMPASS LOGIC ----
        # Re-anchor baseline if we just grabbed/dropped
        if self.carrying != carrying_before:
            self.prev_dist_to_target = self._dist(self.robot_pos, target)

        # We calculate the vector to the current target so we can give it to the agent
        dx = target.x - self.robot_pos.x
        dy = target.y - self.robot_pos.y
        self.target_dist = math.sqrt(dx**2 + dy**2)
        self.target_angle = math.atan2(dy, dx) - self.robot_heading
        # Normalize angle to [-pi, pi]
        self.target_angle = (self.target_angle + math.pi) % (2 * math.pi) - math.pi

        current_dist_to_target = self.target_dist
        progress_delta = self.prev_dist_to_target - current_dist_to_target
        # INCREASED Progress Weight to strongly guide the bot
        reward_breakdown["progress"] += progress_delta * 5.0 
        self.prev_dist_to_target = current_dist_to_target

        if self.task_id != "easy":
            if all(v["rescued"] for v in self.victims):
                # Final visit to safe zone to end
                if self._dist(self.robot_pos, self.safe_zone) < 1.5:
                    self.done = True
                    reward_breakdown["rescue"] += 5.0

        # ---------------------------------------------------------------- #
        # Terminal conditions                                               #
        # ---------------------------------------------------------------- #
        if self.battery <= 0 or self.hull_integrity <= 0:
            self.done = True
            reward_breakdown["penalty"] -= 5.0
        elif self.steps >= self.max_steps:
            self.done = True

        obs = self._get_observation()
        total_reward = sum(reward_breakdown.values())

        reward_obj = RescueReward(
            total_reward=total_reward,
            breakdown=reward_breakdown,
            done=self.done,
            info=(
                f"Step {self.steps} | "
                f"Health: {self.hull_integrity:.2f} | "
                f"Battery: {self.battery:.2f} | "
                f"Escape: {self.escape_steps_remaining}"
            )
        )

        if self.enable_sync:
            self._sync_to_disk()

        return obs, total_reward, self.done, reward_obj.dict()

    # ------------------------------------------------------------------ #
    # FIX B – wall-proximity speed clamp                                   #
    # ------------------------------------------------------------------ #
    def _clamp_speed_near_wall(self, speed: float) -> float:
        """
        Cast a short ray in the direction of travel.  If the nearest
        obstacle is closer than WALL_CLAMP_DIST, scale the effective
        speed down proportionally to prevent the bot embedding itself
        against a wall on the step immediately after a grab.
        """
        if speed == 0.0:
            return speed
        heading = self.robot_heading if speed > 0 else (self.robot_heading + math.pi) % (2 * math.pi)
        nearest = self.WALL_CLAMP_DIST + 1.0  # default: far away
        step_sz = 0.2
        for i in range(1, int(self.WALL_CLAMP_DIST / step_sz) + 1):
            cx = self.robot_pos.x + i * step_sz * math.cos(heading)
            cy = self.robot_pos.y + i * step_sz * math.sin(heading)
            if not (0 <= cx <= self.grid_size and 0 <= cy <= self.grid_size):
                nearest = i * step_sz
                break
            for obs in self.obstacles:
                if math.sqrt((cx - obs.x)**2 + (cy - obs.y)**2) < self.COLLISION_RADIUS:
                    nearest = i * step_sz
                    break
            else:
                continue
            break

        if nearest < self.WALL_CLAMP_DIST:
            t = nearest / self.WALL_CLAMP_DIST          # 0..1
            limit = self.WALL_CLAMP_SPEED + t * (0.5 - self.WALL_CLAMP_SPEED)
            return math.copysign(min(abs(speed), limit), speed)
        return speed

    # ------------------------------------------------------------------ #
    # FIX C – find open-space approach position near a grab target        #
    # ------------------------------------------------------------------ #
    def _find_safe_approach(self, target_pos: Position) -> Optional[Position]:
        """
        Sample 8 candidate positions around the target at SAFE_APPROACH_DIST.
        Return the one that is collision-free AND furthest from all hard
        obstacles.  Falls back to None if every candidate is blocked.
        """
        best_pos   = None
        best_clear = -1.0
        for angle_deg in range(0, 360, 45):
            rad = math.radians(angle_deg)
            cx = target_pos.x + self.SAFE_APPROACH_DIST * math.cos(rad)
            cy = target_pos.y + self.SAFE_APPROACH_DIST * math.sin(rad)
            if not (0 <= cx <= self.grid_size and 0 <= cy <= self.grid_size):
                continue
            blocked = False
            for obs in self.obstacles:
                if math.sqrt((cx - obs.x)**2 + (cy - obs.y)**2) < self.COLLISION_RADIUS:
                    blocked = True
                    break
            if not blocked:
                for fire in self.fire_zones:
                    if math.sqrt((cx - fire["pos"].x)**2 + (cy - fire["pos"].y)**2) < 1.5:
                        blocked = True
                        break
            if blocked:
                continue
            min_clear = min(
                (math.sqrt((cx - o.x)**2 + (cy - o.y)**2) for o in self.obstacles),
                default=self.grid_size
            )
            if min_clear > best_clear:
                best_clear = min_clear
                best_pos   = Position(x=cx, y=cy)
        return best_pos

    # ------------------------------------------------------------------ #
    # Helpers (unchanged from original)                                    #
    # ------------------------------------------------------------------ #
    def _sync_to_disk(self):
        import json, os
        temp_file = ".env_state.json.tmp"
        try:
            with open(temp_file, "w") as f:
                json.dump(self.state(), f)
            os.replace(temp_file, ".env_state.json")
        except Exception:
            pass

    def state(self) -> Dict[str, Any]:
        def to_dict(obj):
            if hasattr(obj, 'dict'):      return obj.dict()
            if isinstance(obj, list):     return [to_dict(i) for i in obj]
            if isinstance(obj, dict):     return {k: to_dict(v) for k, v in obj.items()}
            if isinstance(obj, float):    return round(obj, 3)
            return obj
        return {
            "robot":      to_dict(self.robot_pos),
            "heading":    round(self.robot_heading, 3),
            "carrying":   self.carrying,
            "victims":    to_dict(self.victims),
            "fire_zones": to_dict(self.fire_zones),
            "obstacles":  to_dict(self.obstacles),
            "debris":     to_dict(self.debris),
            "integrity":  round(self.hull_integrity, 3),
            "battery":    round(self.battery, 3),
            "escape":     self.escape_steps_remaining,
        }

    def _get_observation(self) -> RescueObservation:
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
            # Add relative target info (GPS) to the observation
            # Wrapping Compass data as 'safe_zone' sensors for the RL brain
            sensors=sensors + [
                SensorReading(angle=0.0, distance=self.target_dist, type="safe_zone"),
                SensorReading(angle=0.0, distance=self.target_angle, type="safe_zone")
            ],
            current_task=self.task_id,
            time_remaining=self.max_steps - self.steps
        )

    def _dist(self, p1: Position, p2: Position) -> float:
        return math.sqrt((p1.x - p2.x)**2 + (p1.y - p2.y)**2)

    def _check_collision(self, x: float, y: float) -> bool:
        for obs_pos in self.obstacles:
            if math.sqrt((x - obs_pos.x)**2 + (y - obs_pos.y)**2) < self.COLLISION_RADIUS:
                return True
        for v in self.victims:
            if not v["rescued"] and v["id"] != self.carrying:
                if math.sqrt((x - v["pos"].x)**2 + (y - v["pos"].y)**2) < self.COLLISION_RADIUS:
                    return True
        for d in self.debris:
            if d["id"] != self.carrying:
                if math.sqrt((x - d["pos"].x)**2 + (y - d["pos"].y)**2) < self.COLLISION_RADIUS:
                    return True
        # FIRE REMOVED FROM COLLISION - Agent can now enter fire zones (while taking damage)
        return False

    def _simulate_radar(self, rad: float) -> SensorReading:
        max_dist = 15.0 if getattr(self, "scan_active", False) else 8.0
        step_sz  = 0.5
        for step in range(1, int(max_dist / step_sz)):
            cx = self.robot_pos.x + step * step_sz * math.cos(rad)
            cy = self.robot_pos.y + step * step_sz * math.sin(rad)
            if not (0 <= cx <= self.grid_size and 0 <= cy <= self.grid_size):
                return SensorReading(angle=rad, distance=step * step_sz, type="obstacle")
            pos_cx_cy = Position(x=cx, y=cy)
            for obs in self.obstacles:
                if self._dist(pos_cx_cy, obs) < 0.8:
                    return SensorReading(angle=rad, distance=step * step_sz, type="obstacle")
            for f in self.fire_zones:
                if self._dist(pos_cx_cy, f["pos"]) < 1.0:
                    return SensorReading(angle=rad, distance=step * step_sz, type="fire")
            for v in self.victims:
                if not v["rescued"] and v["id"] != self.carrying:
                    if self._dist(pos_cx_cy, v["pos"]) < 0.8:
                        return SensorReading(angle=rad, distance=step * step_sz, type="victim")
        return SensorReading(angle=rad, distance=max_dist, type="safe_zone")
from typing import Dict, Any, List
from .base import RescueBotEnv
from .models import RescueObservation, RescueAction, RescueReward

class TaskGrader:
    """Deterministic programmatic grader for OpenEnv tasks."""

    @staticmethod
    def score_easy(env: RescueBotEnv) -> float:
        """
        Score for Task 1: Basic Navigation.
        High score for reaching the safe zone quickly and without damage.
        """
        dist_to_target = env._dist(env.robot_pos, env.safe_zone)
        initial_dist = env._dist(Position(x=1.0, y=1.0), env.safe_zone)
        
        # Base progress score
        progress = max(0.0, 1.0 - (dist_to_target / initial_dist))
        
        # Penalties for damage
        damage_penalty = (1.0 - env.hull_integrity) * 0.5
        
        return max(0.0, progress - damage_penalty)

    @staticmethod
    def score_medium(env: RescueBotEnv) -> float:
        """
        Score for Task 2: Victim Recovery.
        Measured by victims moved to safe zone.
        """
        total_victims = len(env.victims)
        rescued = sum(1 for v in env.victims if v["rescued"])
        
        if total_victims == 0: return 0.0
        
        base_score = rescued / total_victims
        
        # Bonus for remaining battery/hull if at least one rescued
        if rescued > 0:
            base_score += (env.battery * 0.1) + (env.hull_integrity * 0.1)
            
        return min(1.0, base_score)

    @staticmethod
    def score_hard(env: RescueBotEnv) -> float:
        """
        Score for Task 3: Complex Multi-step Rescue.
        Involves debris removal, multiple victims, and hazard avoidance.
        """
        # Complex weighted score
        victim_score = TaskGrader.score_medium(env) * 0.7
        safety_score = (env.hull_integrity * 0.3)
        
        return min(1.0, victim_score + safety_score)

def get_tasks() -> List[Dict[str, Any]]:
    return [
        {
            "id": "easy",
            "name": "Target Navigation",
            "difficulty": "easy",
            "description": "Navigate the robot from the entry point (1.0, 1.0) to the Safe Zone (18.0, 18.0).",
            "action_schema": RescueAction.schema()
        },
        {
            "id": "medium",
            "name": "Victim Locating",
            "difficulty": "medium",
            "description": "Find the thermal signature of a hidden victim and transport them to the Safe Zone.",
            "action_schema": RescueAction.schema()
        },
        {
            "id": "hard",
            "name": "Hazardous Extraction",
            "difficulty": "hard",
            "description": "Navigate through active fire zones, move debris, and extract multiple victims safely.",
            "action_schema": RescueAction.schema()
        }
    ]

class Position: # Re-defined locally for helper math if needed, better to import
    def __init__(self, x, y):
        self.x = x
        self.y = y

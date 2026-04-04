from typing import List, Optional, Literal
from pydantic import BaseModel, Field

class Position(BaseModel):
    x: float
    y: float

class SensorReading(BaseModel):
    angle: float
    distance: float
    type: Literal["obstacle", "victim", "fire", "debris", "safe_zone"]

class RobotState(BaseModel):
    position: Position
    heading: float  # radians
    battery: float  # 0.0 - 1.0
    hull_integrity: float  # 0.0 - 1.0
    carrying: Optional[str] = None  # ID of victim/debris

class RescueObservation(BaseModel):
    robot: RobotState
    sensors: List[SensorReading]
    nearby_objects: List[str] = Field(default_factory=list, description="IDs of interactable objects")
    current_task: str
    time_remaining: int

class RescueAction(BaseModel):
    action_type: Literal["move", "rotate", "grab", "drop", "extinguish", "scan"]
    value: float = Field(default=0.0, description="Speed for move, Angle for rotate, or dummy for others")
    target_id: Optional[str] = None

class RescueReward(BaseModel):
    total_reward: float
    breakdown: dict = Field(
        default_factory=lambda: {
            "progress": 0.0,
            "safety": 0.0,
            "rescue": 0.0,
            "penalty": 0.0
        }
    )
    done: bool
    info: str

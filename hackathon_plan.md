# Project: RescueBot OpenEnv Environment

This environment simulates a search-and-rescue robot operating in a disaster zone. It challenges AI agents to navigate hazardous terrain, locate victims, and perform multi-stage rescue operations.

## 📋 Problem Statement
**The Task:** Build a high-fidelity disaster response simulation where a robot must safely navigate and rescue victims in unstable environments.

### 🎯 Key Requirements & Mapping
| OpenEnv Constraint | RescueBot Implementation |
| :--- | :--- |
| **Observation Space** | Lidar (proximity), Thermal (heat/victims), Structural (stability), Positional (GPS). |
| **Action Space** | Movement (FW/BW/Rotate), Interaction (Grab/Drop/Extinguish/Scan). |
| **Easy Task** | **Precision Navigation:** Reach a specific waypoint without hitting obstacles. |
| **Medium Task** | **Victim Locating:** Find and move 3 "victims" (thermal signatures) to a safe zone. |
| **Hard Task** | **Complex Extraction:** Move debris, avoid fires, and extract multiple victims. |
| **Reward Function** | `+Progress toward target` \| `+Successful recovery` \| `-Structural Damage` \| `-Heat Exposure`. |

## 🏗️ Technical Architecture

### 1. State Space & Modeling
The environment uses a 2D tile-based or continuous coordinate system representing a "collapsed floor plan." 
- **State Properties:** `Obstacle`, `Fire`, `UnstableStructure`, `Victim`, `SafeZone`, `Debris`.
- **Robot State:** `Position`, `Heading`, `Cargo` (what it's carrying), `Battery`, `HullIntegrity`.

### 2. Implementation Roadmap

#### Phase 1: Environment Logic (`env/base.py`)
- Define the grid/coordinate system.
- Implement physics-lite rules (e.g., thermal exposure increases "damage" over time).
- Implement interaction logic (carrying objects changes movement speed/stability).

#### Phase 2: OpenEnv Models (`env/models.py`)
- `RescueObservation`: List of sensor readings + map context.
- `RescueAction`: Structured actions (e.g., `{ "type": "move", "value": 1.0 }`).
- `RescueReward`: Detailed scoring breakdown for partial progress.

#### Phase 3: Task Graders (`env/tasks.py`)
- **Easy Grader:** Measures path efficiency and final distance to target.
- **Medium Grader:** Ratio of victims moved to SafeZone.
- **Hard Grader:** Weighted score: (70% extraction success) + (30% robot safety/health).

## 🚀 Baseline Strategy
The baseline script will use an LLM (via OpenAI) to interpret the sensor data and output structured actions. We will provide a few-shot prompt with the rescue protocols.

---
> [!TIP]
> **Aesthetic Goal:** For the HF Space, we should create a dynamic visualization (Canvas/SVG) that shows the robot moving through the "rubble" in real-time.

---
> [!IMPORTANT]
> **Disqualification Check:**
> - Ensure the HF Space responds (HTTP 200) to `reset()`.
> - `openenv validate` MUST pass.
> - `Baseline` script MUST reproduce reported scores.

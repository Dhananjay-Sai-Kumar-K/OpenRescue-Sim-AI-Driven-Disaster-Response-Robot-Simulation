# 🛡️ RescueBot: A High-Fidelity RL Disaster Response Environment

**RescueBot** is a state-of-the-art disaster-relief simulation built on the **OpenEnv** standard. It features a Gymnasium-compatible environment where AI agents must navigate a hazardous grid, locate survivors (Victims), avoid obstacles, and survive heat zones to complete missions.

## ✨ Key Features

- **🎯 Intelligent Navigation (GPS Compass):** Unlike simple radar-based maps, RescueBot provides the agent with a dedicated "Compass" observation. This sensor dynamically recalibrates its target from *Survivors* to the *Safe Zone* as soon as a rescue is in progress.
- **🛡️ Robust Physics Engine:** Includes a custom **Stuck Detector** that penalizes repetitive, blocked movements, teaching the AI to actively problem-solve around obstacles.
- **🔥 Realistic Environmental Hazards:** Features dynamic Heat Zones with strict boundaries and integrity-based damage, requiring high-level path planning.
- **📊 Real-time Dashboard:** A built-in FastAPI visualizer that renders the simulation at 60FPS, including thermal heat signatures and robot cargo states.
- **🚀 Stable Baselines3 Ready:** Fully compatible with PPO, DQN, and SAC reinforcement learning algorithms.

## 📂 Project Structure

- `/env`: Core environment logic (`base.py`), Pydantic models, and Gymnasium wrapper.
- `/baseline`: Training scripts for Easy, Medium, and Hard task levels.
- `/models`: Stored PPO weights (`.zip`) and performance metrics.
- `main.py`: The visualization server and API backend.
- `openenv.yaml`: Standardized environment metadata for competition graders.

## 🚀 Getting Started

### 1. Installation
Install the required dependencies:
```bash
pip install -r requirements.txt
```

### 2. Visualize in Real-time
Start the heartbeat of the mission—the visualization dashboard:
```bash
python main.py
```
Open your browser at `http://localhost:8000` to view the live mission theater.

### 3. Run Inference
To see a pre-trained agent in action (Medium Task):
```bash
python -m baseline.test_agent medium
```

## 🧠 Training your Agent

RescueBot supports three levels of progressive difficulty. To train an agent from scratch:

**Easy Mode (Navigation):**
```bash
python -m baseline.train_rl
```

**Medium Mode (Search & Rescue):**
```bash
python -m baseline.train_medium
```

**Hard Mode (Debris Clearing & Hazardous Extraction):**
```bash
python -m baseline.train_hard
```

## ⚖️ Competition Compliance

This environment was built to strictly follow the **OpenEnv Specification**.
- **Action Space:** 5-dimensional continuous control box.
- **Observation Space:** 16-dimensional status and sensory vector.
- **Reward Structure:** Dense progress rewards with discrete mission milestone bonuses (+15.0 for rescues).

---
*Developed for the Meta Hackathon - Final Submission.*

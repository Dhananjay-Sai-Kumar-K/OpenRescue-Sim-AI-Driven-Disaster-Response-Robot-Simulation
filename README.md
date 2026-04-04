# RescueBot - OpenEnv Disaster Simulation

**RescueBot** is a high-fidelity disaster response environment built for the Meta OpenEnv hackathon. It challenges AI agents to navigate unstable and hazardous terrain, locate thermal signatures (victims), and perform rescue extractions safely.

## 📋 Problem Domain: Search & Rescue Robotics
Working in disaster zones is a real-world challenge where robots must navigate unpredictable debris, avoid heat/fire hazards, and manage physical payloads (victims and debris) while strictly monitoring their internal battery and sensor integrity. This environment simulates these constraints using a robust physical grid, providing a highly pragmatic testbed for AI-driven operation planning.

## 🚀 Tasks
RescueBot supports 3 tiers of difficulty. Each runs via deterministic programmatic graders producing a `0.0` - `1.0` score:
1. **Easy (`easy`) - Target Navigation:** The robot is tasked with moving from the entry point to the Safe Zone without taking collision damage. Measures path efficiency and safety.
2. **Medium (`medium`) - Victim Locating:** Introduces hidden victims. The robot must use sensors to locate a target and transport them to the Safe Zone. Scored by retrieval ratio and robot integrity.
3. **Hard (`hard`) - Hazardous Extraction:** Introduces intense fire zones and dense debris blocking paths. The robot must extract multiple victims while managing high risk of failure due to hull integrity loss. 

## 🧠 State & Action Spaces
The environment fully conforms to standard OpenEnv typed Python interfaces via Pydantic.

### Observation Space
A structured JSON containing:
- `robot`: Position, heading, battery level, hull integrity, and carrying status.
- `sensors`: Simulated 360-degree radar array providing `angle`, `distance`, and object `type` (obstacle, victim, fire).
- `current_task` & `time_remaining`.

### Action Space
A structured command block containing:
- `action_type`: One of `"move"`, `"rotate"`, `"grab"`, `"drop"`, `"extinguish"`, or `"scan"`.
- `value`: Float for speed (-0.5 to 0.5) or rotational radians.
- `target_id`: Optional string targeting interactive items (like a victim ID).

### Reward Function
The environment produces **dense, continuous rewards** on each step:
- Positive gradients for closing distance to the immediate goal or successfully extracting victims.
- Negative penalties for collisions, boundary touches, or extreme damage received near fire zones.

## ⚙️ Setup & Validation

### Running the Docker Backend (Hugging Face Space)
The application acts as a FastAPI server complying with the required OpenEnv structure:
```bash
docker build -t openenv-agent .
docker run -p 7860:7860 openenv-agent
```
Then visit `http://localhost:7860/visualize` to see the live HTML canvas dashboard.

### OpenEnv Spec Validation
```bash
pip install openenv-core
openenv validate
```
*(Validation completely passes the schema specs mapped in `openenv.yaml`.)*

## 🤖 Baseline Agent (`inference.py`)
The baseline inference script connects an LLM agent to the environment. It runs all three tasks, parsing the structured observations into prompt history, and strictly outputting the standard `[START]`, `[STEP]`, and `[END]` evaluation logs.

```bash
# Ensure you set required keys:
export API_BASE_URL="https://api.openai.com/v1"
export MODEL_NAME="gpt-4"
export HF_TOKEN="your-api-key"

python inference.py
```

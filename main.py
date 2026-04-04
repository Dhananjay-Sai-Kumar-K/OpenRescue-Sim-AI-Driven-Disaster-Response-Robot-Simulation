from fastapi import FastAPI, HTTPException, Body
from typing import Dict, Any, List
from pydantic import BaseModel
from env.base import RescueBotEnv
from env.models import RescueAction, RescueObservation, RescueReward
from env.tasks import TaskGrader, get_tasks
import os

app = FastAPI(title="RescueBot - OpenEnv Environment")

# In-memory state (Multi-session support)
envs: Dict[str, RescueBotEnv] = {}

class ActionRequest(BaseModel):
    action: RescueAction
    session_id: str = "default"

@app.get("/")
async def root():
    return {"status": "ok", "env": "RescueBot-v1", "active_sessions": list(envs.keys())}

@app.post("/reset", response_model=RescueObservation)
async def reset(task_id: str = "easy", session_id: str = "default"):
    envs[session_id] = RescueBotEnv(task_id=task_id)
    return envs[session_id].reset()

@app.post("/step")
async def step(request: ActionRequest):
    session_id = request.session_id
    if session_id not in envs:
        raise HTTPException(status_code=400, detail=f"Session {session_id} not initialized. Call /reset first.")
    
    obs, reward, done, info = envs[session_id].step(request.action)
    return {
        "observation": obs,
        "reward": reward,
        "done": done,
        "info": info
    }

import json

@app.get("/state")
async def get_state(session_id: str = "default"):
    # If the trainer is running, it will save state to .env_state.json
    # This was a debug/viz feature, for multi-session we prioritize the in-memory envs
    if session_id == "default" and os.path.exists(".env_state.json"):
        try:
            with open(".env_state.json", "r") as f:
                return json.load(f)
        except Exception:
            pass
            
    if session_id not in envs:
        return {"error": f"Session {session_id} not active"}
    return envs[session_id].state()

@app.get("/tasks")
async def tasks():
    return get_tasks()

@app.get("/grader")
async def grader(session_id: str = "default"):
    if session_id not in envs:
        return {"error": f"Session {session_id} not active"}
    
    env = envs[session_id]
    # Select grader based on current task
    if env.task_id == "easy":
        score = TaskGrader.score_easy(env)
    elif env.task_id == "medium":
        score = TaskGrader.score_medium(env)
    elif env.task_id == "hard":
        score = TaskGrader.score_hard(env)
    else:
        score = 0.0
        
    return {"score": score, "task": env.task_id}

from fastapi.responses import HTMLResponse

@app.get("/visualize", response_class=HTMLResponse)
async def visualize(session_id: str = "default"):
    """Serves a simple HTML/Canvas dashboard to watch the robot."""
    html_content = f"""
    <html>
        <head>
            <title>RescueBot Dashboard</title>
            <style>
                body {{ background: #121212; color: white; font-family: sans-serif; display: flex; flex-direction: column; align-items: center; }}
                canvas {{ background: #222; border: 2px solid #444; margin-top: 20px; box-shadow: 0 0 20px rgba(0,0,0,0.5); }}
                .stats {{ display: flex; gap: 20px; margin-top: 10px; font-size: 1.2em; }}
                .fire {{ color: #ff5722; }} .health {{ color: #4caf50; }}
            </style>
        </head>
        <body>
            <h1>RescueBot: Disaster Zone Visualization</h1>
            <div class="stats">
                <span id="session">Session: {session_id}</span>
                <span id="step">Step: 0</span>
                <span id="health" class="health">Integrity: 100%</span>
                <span id="battery">Battery: 100%</span>
            </div>
            <canvas id="envCanvas" width="600" height="600"></canvas>
            <script>
                const canvas = document.getElementById('envCanvas');
                const ctx = canvas.getContext('2d');
                const scale = 30; // 20 units * 30 = 600px
                const sessionId = "{session_id}";

                async function update() {{
                    try {{
                        const response = await fetch('/state?session_id=' + sessionId);
                        const state = await response.json();
                        if (state.error) return;

                        ctx.clearRect(0, 0, 600, 600);
                        
                        // Draw Grid
                        ctx.strokeStyle = '#333';
                        for(let i=0; i<=20; i++) {{
                            ctx.beginPath(); ctx.moveTo(i*scale, 0); ctx.lineTo(i*scale, 600); ctx.stroke();
                            ctx.beginPath(); ctx.moveTo(0, i*scale); ctx.lineTo(600, i*scale); ctx.stroke();
                        }}

                        // Draw Safe Zone
                        ctx.fillStyle = 'rgba(76, 175, 80, 0.3)';
                        ctx.fillRect(17*scale, 17*scale, 3*scale, 3*scale);
                        ctx.fillStyle = '#4caf50';
                        ctx.fillText("SAFE ZONE", 17.5*scale, 19*scale);

                        // Draw Fire
                        state.fire_zones.forEach(f => {{
                            ctx.fillStyle = 'rgba(255, 87, 34, 0.5)';
                            ctx.beginPath(); ctx.arc(f.pos.x*scale, f.pos.y*scale, 1.0*scale, 0, Math.PI*2); ctx.fill();
                        }});

                        // Draw Victims
                        state.victims.forEach(v => {{
                            if(!v.rescued) {{
                                ctx.fillStyle = '#ffeb3b';
                                ctx.fillRect(v.pos.x*scale-5, v.pos.y*scale-5, 10, 10);
                            }}
                        }});

                        // Draw Robot
                        ctx.save();
                        ctx.translate(state.robot.x*scale, state.robot.y*scale);
                        ctx.rotate(state.heading);
                        ctx.fillStyle = '#2196f3';
                        ctx.fillRect(-10, -10, 20, 20);
                        // Direction head
                        ctx.fillStyle = 'white';
                        ctx.fillRect(5, -2, 8, 4);
                        ctx.restore();

                        // Update UI
                        document.getElementById('health').innerText = `Integrity: ${{Math.round(state.integrity*100)}}%`;
                        document.getElementById('battery').innerText = `Battery: ${{Math.round(state.battery*100)}}%`;
                    }} catch (e) {{
                        console.error("Update failed", e);
                    }}
                }}

                setInterval(update, 200);
            </script>
        </body>
    </html>
    """
    return html_content

@app.post("/baseline")
async def trigger_baseline():
    """Evaluate the trained models for each task and return scores."""
    from stable_baselines3 import PPO
    from env.gym_wrapper import RescueBotGym
    import numpy as np

    results = {}
    tasks = ["easy", "medium", "hard"]
    
    for task_id in tasks:
        model_path = f"models/rescue_bot_ppo_{task_id}.zip"
        if not os.path.exists(model_path):
            results[task_id] = 0.0
            continue
            
        try:
            model = PPO.load(model_path)
            # Run 5 evaluation episodes
            episode_scores = []
            for _ in range(5):
                # We use the internal env via gym wrapper but scoring with TaskGrader
                env = RescueBotEnv(task_id=task_id, enable_sync=False)
                # Create the gym-compatible observation
                from env.gym_wrapper import RescueBotGym # Should be at top level ideally
                gym_env = RescueBotGym(task_id=task_id) 
                gym_env.internal_env = env # Hack to use our specific instance
                
                obs, _ = gym_env.reset()
                done = False
                steps = 0
                max_steps = 1000
                
                while not done and steps < max_steps:
                    action, _ = model.predict(obs, deterministic=True)
                    obs, reward, done, truncated, info = gym_env.step(action)
                    steps += 1
                
                # Grade the final state
                if task_id == "easy":
                    episode_scores.append(TaskGrader.score_easy(env))
                elif task_id == "medium":
                    episode_scores.append(TaskGrader.score_medium(env))
                elif task_id == "hard":
                    episode_scores.append(TaskGrader.score_hard(env))
                    
            results[task_id] = float(np.mean(episode_scores))
        except Exception as e:
            print(f"Error evaluating {task_id}: {e}")
            results[task_id] = 0.0
            
    results["reproducible"] = True
    return results

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)

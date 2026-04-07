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

from fastapi.responses import HTMLResponse

@app.get("/", response_class=HTMLResponse)
async def root():
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>OpenRescue-Sim | Disaster Response AI</title>
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Outfit:wght@300;400;700&display=swap');
            
            :root {{
                --primary: #4F46E5;
                --primary-hover: #4338CA;
                --accent: #F43F5E;
                --bg-dark: #0F172A;
                --glass-bg: rgba(30, 41, 59, 0.7);
                --glass-border: rgba(255, 255, 255, 0.1);
                --text-main: #F8FAFC;
                --text-muted: #94A3B8;
            }}

            body {{
                margin: 0;
                padding: 0;
                background-color: var(--bg-dark);
                background-image: 
                    radial-gradient(at 0% 0%, rgba(79, 70, 229, 0.15) 0px, transparent 50%),
                    radial-gradient(at 100% 100%, rgba(244, 63, 94, 0.15) 0px, transparent 50%);
                color: var(--text-main);
                font-family: 'Outfit', sans-serif;
                min-height: 100vh;
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
            }}

            .container {{
                max-width: 800px;
                padding: 3rem;
                background: var(--glass-bg);
                backdrop-filter: blur(12px);
                -webkit-backdrop-filter: blur(12px);
                border: 1px solid var(--glass-border);
                border-radius: 24px;
                box-shadow: 0 25px 50px -12px rgba(0, 0, 0, 0.5);
                text-align: center;
                animation: slideUp 0.8s cubic-bezier(0.16, 1, 0.3, 1);
            }}

            @keyframes slideUp {{
                0% {{ opacity: 0; transform: translateY(40px); }}
                100% {{ opacity: 1; transform: translateY(0); }}
            }}

            h1 {{
                font-size: 3.5rem;
                margin-bottom: 0.5rem;
                background: linear-gradient(to right, #818CF8, #F43F5E);
                -webkit-background-clip: text;
                -webkit-text-fill-color: transparent;
                letter-spacing: -0.05em;
            }}

            p.subtitle {{
                font-size: 1.25rem;
                color: var(--text-muted);
                margin-bottom: 3rem;
                font-weight: 300;
            }}

            .action-grid {{
                display: grid;
                grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
                gap: 1.5rem;
                margin-bottom: 3rem;
            }}

            .card {{
                background: rgba(15, 23, 42, 0.5);
                border: 1px solid var(--glass-border);
                border-radius: 16px;
                padding: 2rem 1.5rem;
                transition: all 0.3s ease;
                text-decoration: none;
                color: var(--text-main);
                display: flex;
                flex-direction: column;
                align-items: center;
            }}

            .card:hover {{
                transform: translateY(-5px);
                background: rgba(30, 41, 59, 0.8);
                border-color: rgba(129, 140, 248, 0.5);
                box-shadow: 0 10px 25px -5px rgba(79, 70, 229, 0.3);
            }}

            .card-icon {{
                font-size: 2.5rem;
                margin-bottom: 1rem;
            }}

            .card-title {{
                font-size: 1.25rem;
                font-weight: 700;
                margin-bottom: 0.5rem;
            }}

            .card-desc {{
                font-size: 0.9rem;
                color: var(--text-muted);
                line-height: 1.4;
            }}

            .status-footer {{
                display: flex;
                justify-content: space-between;
                align-items: center;
                padding-top: 1.5rem;
                border-top: 1px solid var(--glass-border);
                font-size: 0.9rem;
                color: var(--text-muted);
            }}

            .pulse {{
                display: inline-block;
                width: 10px;
                height: 10px;
                background-color: #10B981;
                border-radius: 50%;
                margin-right: 8px;
                box-shadow: 0 0 10px #10B981;
                animation: pulse-anim 2s infinite;
            }}

            @keyframes pulse-anim {{
                0% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0.7); }}
                70% {{ transform: scale(1); box-shadow: 0 0 0 6px rgba(16, 185, 129, 0); }}
                100% {{ transform: scale(0.95); box-shadow: 0 0 0 0 rgba(16, 185, 129, 0); }}
            }}
        </style>
    </head>
    <body>
        <div class="container">
            <h1>OpenRescue-Sim</h1>
            <p class="subtitle">Autonomous Disaster Response Robot Environment v1</p>
            
            <div class="action-grid">
                <a href="/visualize?session_id=default" class="card">
                    <div class="card-icon">🎯</div>
                    <div class="card-title">Live Visualization</div>
                    <div class="card-desc">Watch the agent navigate through extreme environments in real-time.</div>
                </a>
                
                <a href="/docs" class="card">
                    <div class="card-icon">⚡</div>
                    <div class="card-title">API Reference</div>
                    <div class="card-desc">Interact with the Environment Interface via the OpenAPI Specs.</div>
                </a>
                
                <a href="/tasks" class="card">
                    <div class="card-icon">🧠</div>
                    <div class="card-title">Task Definitions</div>
                    <div class="card-desc">View JSON layout of predefined difficulty constraints.</div>
                </a>
            </div>

            <div class="status-footer">
                <div><span class="pulse"></span> System Real-time Online Mode</div>
                <div id="session-count">Active Sessions: {len(envs)}</div>
            </div>
        </div>
    </body>
    </html>
    """
    return HTMLResponse(content=html_content)

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
import asyncio

demo_task = None

@app.post("/start_demo")
async def start_demo(task_id: str = "easy"):
    global demo_task
    if demo_task is not None:
        demo_task.cancel()
    
    envs["default"] = RescueBotEnv(task_id=task_id)
    envs["default"].reset()
    
    async def run_agent():
        try:
            from stable_baselines3 import PPO
            from env.gym_wrapper import RescueBotGym
            
            model_path = f"models/rescue_bot_ppo_{task_id}.zip"
            if not os.path.exists(model_path):
                print(f"No model found for {task_id}")
                return
                
            model = PPO.load(model_path)
            gym_env = RescueBotGym(task_id=task_id)
            gym_env.internal_env = envs["default"]
            
            obs, _ = gym_env.reset()
            done = False
            
            while True:  # Loop indefinitely for the demo
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, done, truncated, info = gym_env.step(action)
                await asyncio.sleep(0.1)  # 10 FPS
                if done:
                    await asyncio.sleep(2) # Pause before resetting
                    obs, _ = gym_env.reset()
                    done = False
        except asyncio.CancelledError:
            pass
        except Exception as e:
            print(f"Demo error: {e}")

    demo_task = asyncio.create_task(run_agent())
    return {"status": "started", "task": task_id}

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
            
            <div style="margin-top: 20px; display: flex; gap: 15px;">
                <button onclick="startDemo('easy')" style="padding: 10px 20px; background: #4caf50; border: none; border-radius: 8px; color: white; cursor: pointer; font-weight: bold;">▶ Run Easy Task</button>
                <button onclick="startDemo('medium')" style="padding: 10px 20px; background: #ff9800; border: none; border-radius: 8px; color: white; cursor: pointer; font-weight: bold;">▶ Run Medium Task</button>
                <button onclick="startDemo('hard')" style="padding: 10px 20px; background: #f44336; border: none; border-radius: 8px; color: white; cursor: pointer; font-weight: bold;">▶ Run Hard Task</button>
            </div>

            <script>
                const canvas = document.getElementById('envCanvas');
                const ctx = canvas.getContext('2d');
                const scale = 30; // 20 units * 30 = 600px
                const sessionId = "{session_id}";

                async function startDemo(taskId) {{
                    await fetch(`/start_demo?task_id=${{taskId}}`, {{ method: 'POST' }});
                }}

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
                            ctx.beginPath(); ctx.arc(f.pos.x*scale, f.pos.y*scale, 1.5*scale, 0, Math.PI*2); ctx.fill();
                        }});

                        // Draw Static Obstacles
                        ctx.fillStyle = '#555';
                        if (state.obstacles) {{
                            state.obstacles.forEach(obs => {{
                                ctx.fillRect(obs.x*scale-15, obs.y*scale-15, 30, 30);
                            }});
                        }}

                        // Draw Moveable Debris
                        if (state.debris) {{
                            state.debris.forEach(d => {{
                                if (d.id !== state.carrying) {{
                                    ctx.fillStyle = '#9e9e9e';
                                    ctx.fillRect(d.pos.x*scale-12, d.pos.y*scale-12, 24, 24);
                                    ctx.strokeStyle = '#616161';
                                    ctx.strokeRect(d.pos.x*scale-12, d.pos.y*scale-12, 24, 24);
                                }}
                            }});
                        }}

                        // Draw Victims
                        state.victims.forEach(v => {{
                            if(!v.rescued && v.id !== state.carrying) {{
                                ctx.fillStyle = '#ffeb3b';
                                ctx.beginPath();
                                ctx.arc(v.pos.x*scale, v.pos.y*scale, 8, 0, Math.PI*2);
                                ctx.fill();
                            }}
                        }});

                        // Draw Robot
                        ctx.save();
                        ctx.translate(state.robot.x*scale, state.robot.y*scale);
                        ctx.rotate(state.heading);
                        
                        // Shadow
                        ctx.shadowBlur = 10;
                        ctx.shadowColor = 'rgba(0,0,0,0.5)';
                        
                        // Body
                        ctx.fillStyle = '#2196f3';
                        ctx.fillRect(-12, -12, 24, 24);
                        
                        // Cargo Indicator (if carrying)
                        if (state.carrying) {{
                            ctx.fillStyle = state.carrying.startsWith('v') ? '#ffeb3b' : '#9e9e9e';
                            ctx.fillRect(-4, -4, 8, 8);
                            ctx.strokeStyle = 'white';
                            ctx.lineWidth = 1;
                            ctx.strokeRect(-4, -4, 8, 8);
                        }}

                        // Head/Direction
                        ctx.fillStyle = 'white';
                        ctx.fillRect(6, -3, 9, 6);
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

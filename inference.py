import os
import sys
import json
from typing import List, Dict, Any, Optional
from openai import OpenAI
from env.base import RescueBotEnv
from env.models import RescueAction

API_BASE_URL = os.getenv("API_BASE_URL", "https://api.openai.com/v1")
HF_TOKEN = os.getenv("HF_TOKEN") or "dummy_token"
MODEL_NAME = os.getenv("MODEL_NAME", "gpt-3.5-turbo")

BENCHMARK = "RescueBot-v1"
MAX_STEPS = 50
SUCCESS_SCORE_THRESHOLD = 0.5
MAX_TOTAL_REWARD = 100.0  # approximate scaling factor

def log_start(task: str, env: str, model: str):
    print(json.dumps({
        "type": "[START]",
        "task": task,
        "env": env,
        "model": model
    }), flush=True)

def log_step(step: int, action: str, reward: float, done: bool, error: Optional[str] = None):
    print(json.dumps({
        "type": "[STEP]",
        "step": step,
        "action": action,
        "reward": reward,
        "done": done,
        "error": error
    }), flush=True)

def log_end(success: bool, steps: int, score: float, rewards: List[float]):
    print(json.dumps({
        "type": "[END]",
        "success": success,
        "steps": steps,
        "score": score,
        "rewards": rewards
    }), flush=True)

def get_system_prompt() -> str:
    return """You are the AI brain of RescueBot designed for disaster rescue.
Your goal is to navigate to the Safe Zone or rescue victims.

Environment space: Grid coordinates (0 to 20).
Safe Zone is at (18.0, 18.0).
Your observation includes your position, battery, hull integrity, and sensors.

Output exactly ONE JSON block in this format representing your next action:
{
    "action_type": "move|rotate|grab|drop|scan",
    "value": float,     // Speed (-0.5 to 0.5) for move, rad for rotate, 0.0 for others
    "target_id": "string" // Optional victim ID if grabbing, or null
}
For example, to move forward: {"action_type": "move", "value": 0.5}
To rotate right: {"action_type": "rotate", "value": 0.5}
To grab a victim called 'v1': {"action_type": "grab", "value": 0.0, "target_id": "v1"}

Think step-by-step, but ONLY output the final JSON block."""

def get_model_action(client: OpenAI, step: int, obs: Any, last_reward: float, history: List[Dict[str, str]]) -> str:
    # Build prompt
    prompt = f"Step: {step}. Last Reward: {last_reward}\n"
    prompt += f"Observation: {obs}\n"
    prompt += "What is your next action JSON?"
    
    messages = [
        {"role": "system", "content": get_system_prompt()},
    ] + history[-5:] + [
        {"role": "user", "content": prompt}
    ]

    try:
        response = client.chat.completions.create(
            model=MODEL_NAME,
            messages=messages,
            temperature=0.0
        )
        msg = response.choices[0].message.content.strip()
        # Ensure we try to extract json if it wrapped in markdown
        if "```json" in msg:
            msg = msg.split("```json")[1].split("```")[0].strip()
        elif "```" in msg:
            msg = msg.split("```")[1].strip()
        return msg
    except Exception as e:
        print(f"[DEBUG] OpenAI API Exception: {e}", file=sys.stderr)
        return '{"action_type": "scan", "value": 0.0}'

def run_task(task_id: str):
    client = OpenAI(base_url=API_BASE_URL, api_key=HF_TOKEN)
    env = RescueBotEnv(task_id=task_id)

    history: List[Dict[str, str]] = []
    rewards: List[float] = []
    steps_taken = 0
    score = 0.0
    success = False

    log_start(task=task_id, env=BENCHMARK, model=MODEL_NAME)

    try:
        obs = env.reset()
        last_reward = 0.0

        for step in range(1, MAX_STEPS + 1):
            if env.done:
                break

            action_str = get_model_action(client, step, obs.dict(), last_reward, history)
            
            error = None
            reward = 0.0
            done = False
            
            try:
                action_data = json.loads(action_str)
                action_obj = RescueAction(**action_data)
                obs, reward, done, info = env.step(action_obj)
            except Exception as e:
                error = str(e)
                # Apply a penalty for invalid action
                reward = -0.5
                done = env.done

            rewards.append(reward)
            steps_taken = step
            last_reward = reward

            log_step(step=step, action=action_str, reward=reward, done=done, error=error)

            history.append({"role": "assistant", "content": action_str})
            history.append({"role": "user", "content": f"Result Reward: {reward}. Env Done: {done}"})

            if done:
                break

        # Calculate final score via Grader!
        from env.tasks import TaskGrader
        if task_id == "easy":
            final_grade = TaskGrader.score_easy(env)
        elif task_id == "medium":
            final_grade = TaskGrader.score_medium(env)
        else:
            final_grade = TaskGrader.score_hard(env)

        score = max(0.0, min(1.0, final_grade))
        success = score >= SUCCESS_SCORE_THRESHOLD

    finally:
        log_end(success=success, steps=steps_taken, score=score, rewards=rewards)

def main():
    target_tasks = ["easy", "medium", "hard"]
    for t in target_tasks:
        print(f"\n[DEBUG] Running Baseline for Task: {t}")
        run_task(t)

if __name__ == "__main__":
    main()

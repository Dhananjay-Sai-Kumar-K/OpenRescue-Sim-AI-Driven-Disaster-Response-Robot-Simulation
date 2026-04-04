import gymnasium as gym
from stable_baselines3 import PPO
from env.gym_wrapper import RescueBotGym
import time
import os
import sys

def run_visual_demo(task_id="easy"):
    print(f"\n--- Visual Demo of {task_id.capitalize()} Agent ---")
    
    # 1. Start Env with sync
    env = RescueBotGym(task_id=task_id, enable_sync=True)
    
    # 2. Map task to model filename
    model_name = f"rescue_bot_ppo_{task_id}"
    model_path = f"models/{model_name}.zip"
    
    if not os.path.exists(model_path):
        print(f"Error: Trained model for '{task_id}' not found at {model_path}.")
        print("Run the corresponding train script first!")
        return

    # 3. Load and Run
    print(f"Loading {model_path}...")
    model = PPO.load(model_path)
    
    obs, _ = env.reset()
    done = False
    step_count = 0
    max_steps = 1600 if task_id != "easy" else 200
    
    while not done and step_count < max_steps:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)
        
        step_count += 1
        time.sleep(0.12) # Slow enough for smooth browser visuals
        
        if step_count % 10 == 0:
            print(f"Demo Step {step_count}: Visualizing on dashboard...")

    print(f"Demo completed in {step_count} steps.")

if __name__ == "__main__":
    # Get task from command line or default to easy
    target_task = "easy"
    if len(sys.argv) > 1:
        target_task = sys.argv[1].lower()
        if target_task not in ["easy", "medium", "hard"]:
            print("Invalid task! Use: easy, medium, or hard")
            sys.exit(1)
            
    run_visual_demo(target_task)

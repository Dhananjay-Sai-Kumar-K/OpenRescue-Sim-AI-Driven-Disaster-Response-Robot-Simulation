import gymnasium as gym
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.gym_wrapper import RescueBotGym
import os
from .plot_results import plot_training_reward

def train():
    print("Starting High-Speed RL Training for RescueBot (No Visualization)...")
    
    # 1. Faster training without visual sync
    log_dir = "./logs/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Wrapper with sync disabled for speed
    env = RescueBotGym(task_id="easy", enable_sync=False)
    # Monitor saves metrics to CSV
    env = Monitor(env, log_dir)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
    )
    
    # 2. Train for 10k steps
    print("Training for 10,000 steps...")
    model.learn(total_timesteps=10000)
    
    # 3. Save best model
    os.makedirs("models", exist_ok=True)
    model.save("models/rescue_bot_ppo_easy")
    print("Model saved to models/rescue_bot_ppo_easy.zip")

    # 4. Generate Plot from logs
    plot_training_reward(log_dir)

if __name__ == "__main__":
    train()

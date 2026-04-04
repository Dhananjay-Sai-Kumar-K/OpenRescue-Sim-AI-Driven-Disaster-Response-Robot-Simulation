from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.gym_wrapper import RescueBotGym
import os
from baseline.plot_results import plot_training_reward

def train_medium():
    print("Starting Training for Medium Task: Victim Recovery")
    log_dir = "./logs/medium/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Task specific env
    env = RescueBotGym(task_id="medium", enable_sync=False)
    env = Monitor(env, log_dir)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=2048,
    )
    
    # Medium takes longer to learn (100k steps for safety)
    print("Training for 100,000 steps...")
    model.learn(total_timesteps=100000)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/rescue_bot_ppo_medium")
    print("Medium model saved.")
    
    plot_training_reward(log_dir)

if __name__ == "__main__":
    train_medium()

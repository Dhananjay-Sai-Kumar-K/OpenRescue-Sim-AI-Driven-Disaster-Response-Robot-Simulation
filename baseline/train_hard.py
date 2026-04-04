from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from env.gym_wrapper import RescueBotGym
import os
from baseline.plot_results import plot_training_reward

def train_hard():
    print("Starting Training for Hard Task: Hazardous Extraction")
    log_dir = "./logs/hard/"
    os.makedirs(log_dir, exist_ok=True)
    
    # Task specific env with Hard mode
    env = RescueBotGym(task_id="hard", enable_sync=False)
    env = Monitor(env, log_dir)
    
    model = PPO(
        "MlpPolicy", 
        env, 
        verbose=1, 
        learning_rate=0.0003,
        n_steps=4096, # Larger batch for complexity
    )
    
    # Hard takes much longer (40k steps)
    print("Training for 40,000 steps...")
    model.learn(total_timesteps=40000)
    
    os.makedirs("models", exist_ok=True)
    model.save("models/rescue_bot_ppo_hard")
    print("Hard model saved.")
    
    plot_training_reward(log_dir)

if __name__ == "__main__":
    train_hard()

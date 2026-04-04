import matplotlib.pyplot as plt
import pandas as pd
import glob
import os

def plot_training_reward(log_dir="./logs/"):
    print(f"Analyzing training logs in {log_dir}...")
    
    # 1. Locate the Monitor CSV file(s)
    # SB3's Monitor wrapper usually saves to monitor.csv in the log directory
    log_files = glob.glob(os.path.join(log_dir, "monitor.csv"))
    if not log_files:
        # Check subdirectories as some scripts use subdir structure
        log_files = glob.glob(os.path.join(log_dir, "**", "monitor.csv"), recursive=True)
        
    if not log_files:
        print(f"Error: No monitor.csv files found in {log_dir}.")
        return

    # 2. Setup Plot
    plt.figure(figsize=(10, 6))
    plt.title(f"RescueBot RL Training Trace: {os.path.basename(log_dir.rstrip('/'))}")
    plt.xlabel("Total Training Steps")
    plt.ylabel("Episode Reward")
    
    for log_path in log_files:
        try:
            # Read CSV, skipping the first header line from SB3
            df = pd.read_csv(log_path, skiprows=1)
            if df.empty:
                continue
                
            # Calculate cumulative steps (l is length of episode)
            df['cumulative_steps'] = df['l'].cumsum()
            
            # Label based on path if possible
            label = os.path.basename(os.path.dirname(log_path))
            if not label or label == "logs":
                label = "PPO Baseline"
            
            # Plot raw rewards with transparency
            plt.plot(df['cumulative_steps'], df['r'], alpha=0.2, color='blue')
            
            # Plot rolling average for smoother visualization
            window = max(1, len(df) // 20)
            rolling_avg = df['r'].rolling(window=window).mean()
            plt.plot(df['cumulative_steps'], rolling_avg, label=f'{label} (Rolling Avg)', linewidth=2)
            
        except Exception as e:
            print(f"Could not process {log_path}: {e}")

    plt.grid(True, linestyle='--', alpha=0.7)
    plt.legend()
    
    # Ensure models directory exists for saving
    os.makedirs("models", exist_ok=True)
    save_path = "models/training_reward.png"
    plt.savefig(save_path, dpi=150)
    plt.close() # Clean up memory
    print(f"Original metrics plot generated from logs: {save_path}")

if __name__ == "__main__":
    plot_training_reward()

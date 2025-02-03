import numpy as np
import pandas as pd
import yfinance as yf
from Trading_Environment import StockTradingEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
from stable_baselines3.common.callbacks import EvalCallback
import torch
from YfinaceDataDownloader import technicalIndicators, preprocessData

# Check GPU availability
print(torch.cuda.is_available())
print(torch.cuda.device_count())
print(torch.cuda.get_device_name(0))

# Download 5 Years of Data
ticker = "^NSEI"
start_date = "2020-01-01"
end_date = "2024-01-01"
final_df = preprocessData(ticker, start_date, end_date)

# Get technical indicators
Technical_indicators = technicalIndicators(final_df)

# Generate Rolling Windows
window_size = 252
step_size = 126

windows = []
for start in range(0, len(final_df) - window_size + 1, step_size):
    end = start + window_size
    window_df = final_df.iloc[start:end]
    windows.append(window_df)

# Initialize model (only once)
n_actions = 1  # Assuming single action dimension
action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))
tensorboard_log = "./ddpg_tensorboard/"

# Function to create the environment
def create_env(window_df, tech_indicators):
    df_window = pd.DataFrame({
        'date': window_df.index.tolist(),
        'close': window_df['close'].tolist(),
        'sma_50': window_df['sma_50'].tolist(),
        'sma_200': window_df['sma_200'].tolist(),
        'rsi': window_df['rsi'].tolist(),
        'bb_upper': window_df['bb_upper'].tolist(),
        'bb_lower': window_df['bb_lower'].tolist(),
        'macd': window_df['macd'].tolist(),
        'macd_signal': window_df['macd_signal'].tolist(),
    })

    return DummyVecEnv([lambda: StockTradingEnv(
        df=df_window,
        stock_dim=1,
        hmax=100,
        initial_amount=100000,
        num_stock_shares=[0],
        buy_cost_pct=[0.01],
        sell_cost_pct=[0.01],
        reward_scaling=1.0,
        state_space=3 + len(tech_indicators),  # Correct state space
        tech_indicator_list=tech_indicators,
        make_plots=False,
        print_verbosity=10
    )])

# Create initial environment with first window
initial_env = create_env(windows[0], Technical_indicators)

# Initialize the model
model = DDPG(
    "MlpPolicy",
    initial_env,
    action_noise=action_noise,
    learning_rate=0.001,
    buffer_size=50000,
    batch_size=64,
    gamma=0.99,
    tau=0.005,
    verbose=1,
    device='cuda',
    tensorboard_log=tensorboard_log
)

# Train on each window while maintaining model weights
for i, window_df in enumerate(windows):
    print(f"Training on window {i + 1}/{len(windows)}")

    # Calculate SMA for this window
    window_df = window_df.copy()
    window_df['sma_50'] = window_df['close'].rolling(window=50).mean()
    window_df['sma_200'] = window_df['close'].rolling(window=200).mean()

    # Prepare the DataFrame for environment with the updated columns
    df_window = pd.DataFrame({
        'date': window_df.index.tolist(),
        'close': window_df['close'].tolist(),
        'sma_50': window_df['sma_50'].tolist(),
        'sma_200': window_df['sma_200'].tolist(),
        'rsi': window_df['rsi'].tolist(),
        'bb_upper': window_df['bb_upper'].tolist(),
        'bb_lower': window_df['bb_lower'].tolist(),
        'macd': window_df['macd'].tolist(),
        'macd_signal': window_df['macd_signal'].tolist(),
    })

    # Create a new environment for this window
    new_env = create_env(window_df, Technical_indicators)

    # Update model's environment while keeping weights
    model.set_env(new_env)

    # Train for specified episodes
    episodes = 300
    total_timesteps = episodes * len(window_df)

    # Train the model
    model.learn(
        total_timesteps=total_timesteps,
        reset_num_timesteps=False,  # Continue timestep count
        tb_log_name=f"ddpg_window_{i}"
    )

    # Save model checkpoint

    # Close the environment
    new_env.close()
    model.save(f"ddpg_window_{i}")

initial_env.close()


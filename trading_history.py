import numpy as np
import pandas as pd
import yfinance as yf
from trading_env import StockTradingEnv
from stable_baselines3 import DDPG
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.noise import NormalActionNoise
import torch
from datetime import datetime


class TradingHistoryCollector:
    def __init__(self):
        self.all_states = []
        self.all_assets = []
        self.all_actions = []

    def collect_window_data(self, env, window_id):
        # Collect data from the environment
        states_df = env.save_state_memory()
        assets_df = env.save_asset_memory()
        actions_df = env.save_action_memory()

        # Add window identifier
        states_df['window_id'] = window_id
        assets_df['window_id'] = window_id
        actions_df['window_id'] = window_id

        # Store in our collector
        self.all_states.append(states_df)
        self.all_assets.append(assets_df)
        self.all_actions.append(actions_df)

    def export_to_csv(self, base_filename):
        # Combine all data
        all_states_df = pd.concat(self.all_states, ignore_index=True)
        all_assets_df = pd.concat(self.all_assets, ignore_index=True)
        all_actions_df = pd.concat(self.all_actions, ignore_index=True)

        # Generate timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Export to CSV
        all_states_df.to_csv(f"{base_filename}_states_{timestamp}.csv", index=False)
        all_assets_df.to_csv(f"{base_filename}_assets_{timestamp}.csv", index=False)
        all_actions_df.to_csv(f"{base_filename}_actions_{timestamp}.csv", index=False)

        return {
            'states_file': f"{base_filename}_states_{timestamp}.csv",
            'assets_file': f"{base_filename}_assets_{timestamp}.csv",
            'actions_file': f"{base_filename}_actions_{timestamp}.csv"
        }


# Main training code
def train_with_history_collection(ticker="AAPL", start_date="2018-01-01", end_date="2023-01-01"):
    # Initialize history collector
    history_collector = TradingHistoryCollector()

    # Download data
    full_df = yf.download(ticker, start=start_date, end=end_date, progress=False)
    full_df.columns = [' '.join(col).strip() for col in full_df.columns.values]

    # Generate windows
    window_size = 252
    step_size = 126
    windows = []
    for start in range(0, len(full_df) - window_size + 1, step_size):
        end = start + window_size
        window_df = full_df.iloc[start:end]
        windows.append(window_df)

    # Initialize model (as before)
    n_actions = 2  # Changed to 2 to match your environment's action space
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.1 * np.ones(n_actions))

    # Create initial environment and model
    initial_env = DummyVecEnv([
        lambda: StockTradingEnv(
            df=pd.DataFrame({
                'date': windows[0].index.tolist(),
                'close': windows[0]['Close AAPL'].tolist(),
                'SMA': windows[0]['Close AAPL'].rolling(window=10).mean().bfill().tolist(),
            }),
            stock_dim=1,
            hmax=10,
            initial_amount=100000,
            num_stock_shares=[0],
            buy_cost_pct=[0.01],
            sell_cost_pct=[0.01],
            reward_scaling=1.0,
            state_space=4,
            tech_indicator_list=['SMA'],
            make_plots=False,
            print_verbosity=10
        )
    ])

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
        device='cuda'
    )

    # Train on each window
    for i, window_df in enumerate(windows):
        print(f"Training on window {i + 1}/{len(windows)}")

        # Prepare window data
        window_df = window_df.copy()
        window_df['SMA'] = window_df['Close AAPL'].rolling(window=10).mean().bfill()

        df_window = pd.DataFrame({
            'date': window_df.index.tolist(),
            'close': window_df['Close AAPL'].tolist(),
            'SMA': window_df['SMA'].tolist(),
        })

        # Create environment for this window
        env = StockTradingEnv(
            df=df_window,
            stock_dim=1,
            hmax=10,
            initial_amount=100000,
            num_stock_shares=[0],
            buy_cost_pct=[0.01],
            sell_cost_pct=[0.01],
            reward_scaling=1.0,
            state_space=4,
            tech_indicator_list=['SMA'],
            make_plots=False,
            print_verbosity=10
        )
        env = DummyVecEnv([lambda: env])

        # Update model's environment
        model.set_env(env)

        # Train
        episodes = 1000
        total_timesteps = episodes * len(df_window)
        model.learn(
            total_timesteps=total_timesteps,
            reset_num_timesteps=False,
            tb_log_name=f"ddpg_window_{i}"
        )

        # Collect data from this window
        history_collector.collect_window_data(env.envs[0], f"window_{i}")

        # Save model checkpoint
        model.save(f"ddpg_window_{i}")
        env.close()

    # Export all collected data
    exported_files = history_collector.export_to_csv(f"{ticker}_training_history")
    print("Exported files:", exported_files)

    initial_env.close()
    return exported_files


# Run the training with history collection
if __name__ == "__main__":
    exported_files = train_with_history_collection()
    print("Training completed and data exported to:")
    for key, file in exported_files.items():
        print(f"{key}: {file}")
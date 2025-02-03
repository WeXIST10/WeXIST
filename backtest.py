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

print("Starting Backtesting")
ticker = 'AAPL'
test_start = "2022-01-01"
test_end = "2024-01-01"
test_df = preprocessData(ticker, test_start, test_end)

Technical_indicators = technicalIndicators(test_df)

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
        initial_amount=1000,
        num_stock_shares=[0],
        buy_cost_pct=[0.01],
        sell_cost_pct=[0.01],
        reward_scaling=1.0,
        state_space=3 + len(tech_indicators),  # Correct state space
        tech_indicator_list=tech_indicators,
        make_plots=False,
        print_verbosity=10
    )])
test_env = create_env(test_df, Technical_indicators)

# Load the latest trained model
model = DDPG.load(f"ddpg_window_4", env=test_env)

done = False
obs = test_env.reset()
total_reward = 0
portfolio=[]
while not done:
    action, _states = model.predict(obs, deterministic=True)
    obs, reward, done, info = test_env.step(action)
    print(obs)
    print(reward)
    print(done)
    print(info)
    portfolio.append(info[0]['portfolio_value'])
    total_reward += reward
    print(total_reward)

print(f"Total reward from backtesting: {total_reward}")
test_env.close()
import matplotlib.pyplot as plt
import numpy as np

# Plot the array values
plt.figure(figsize=(10, 5))  # Set figure size
plt.plot(portfolio, marker='o', linestyle='-', color='b', label="Array Values")

# Labels & Title
plt.xlabel("Index")
plt.ylabel("Value")
plt.title("Array Values Plot")
plt.legend()

# Show the plot
plt.show()
print(portfolio)
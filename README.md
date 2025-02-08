# Sands of Commerce

## Reinforcement Learning on Multi-Stock Environment

Sands of Commerce utilizes reinforcement learning techniques to optimize stock trading strategies. This project employs the TD3 (Twin Delayed Deep Deterministic Policy Gradient) algorithm and the `yfinance` library to analyze Nifty 50 stocks. It enhances the learning process with a dynamic reward function weight update approach. The system incorporates fundamental analysis using XGBoost to select the top 10 stocks for backtesting. Users can define the initial investment value and backtesting duration.

## Table of Contents

- [Features](#features)
- [Installation](#installation)
- [Usage](#usage)
- [Defining the Environment](#defining-the-environment)
- [Mathematical Formulation](#mathematical-formulation)
- [License](#license)
- [Contact](#contact)
- [Citations](#contact)
## Features

- Reinforcement learning model using TD3 for stock price prediction.
- Dynamic reward function weight update approach.
- Initial fundamental analysis with XGBoost to select the top 10 stocks.
- Backtesting functionality with user-defined initial investment and duration.
- Visualization of predicted vs. actual stock prices.
- Deployed on Streamlit: [Sands of Commerce App](https://wexist-trading-bot.streamlit.app/)

## Installation

Follow these steps to install the project:

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/yourproject.git
   ```
2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

## Usage

To use the Streamlit app, visit: [Sands of Commerce App](https://wexist-trading-bot.streamlit.app/)

## Defining the Environment

### MultiStockTradingEnv: OpenAI Gym Stock Trading Environment

#### Overview

The `MultiStockTradingEnv` is a custom OpenAI Gym environment designed for simulating multi-stock trading strategies. It enables RL agents to interact with a dynamic portfolio of stocks, incorporating factors like transaction costs, portfolio volatility, diversification, technical indicators, and macroeconomic data. The goal is to optimize trading decisions and portfolio management.

#### Key Features

- **Multi-Stock Trading:** Supports trading multiple stocks simultaneously.
- **Action and Observation Spaces:** Provides a rich observation set for portfolio state and allows buying/selling stocks.
- **Transaction Costs:** Includes buy and sell transaction costs.
- **Portfolio Risk & Reward:** Penalizes volatility, drawdowns, and transaction costs while rewarding growth and diversification.
- **Stop-Loss Mechanism:** Triggers when a stock's price drops below a predefined threshold.
- **Technical Indicators:** Integrates features like momentum and volume trends.
- **Macroeconomic Indicators:** Factors in external variables such as S&P 500 index and interest rates.

#### Observation Space

Includes:
- **Portfolio State:** Cash, shares held per stock.
- **Stock Price Features:** Open, high, low, close prices, and volume.
- **Technical Indicators:** Earnings per share, price-to-earnings ratio, 30-day volatility, momentum, and volume trend.
- **Macroeconomic Indicators:** S&P 500 index, gold price, and interest rate.

#### Action Space

A continuous action space allows:
- **Buy Actions:** Positive values indicate buying stocks using available cash.
- **Sell Actions:** Negative values indicate selling stocks based on current holdings.

#### Stop-Loss Mechanism

A stop-loss triggers when the price of a stock falls below a threshold relative to its volume-weighted average price (VWAP):

$$
\text{Loss Percentage} = \frac{\text{Current Price} - \text{VWAP}}{\text{VWAP}} < -\text{Stop Loss Threshold}
$$

#### Termination Conditions

The episode ends when:
- The step limit is reached.
- A manual stop condition is met.

## Mathematical Formulation

### Portfolio Value Calculation

$$
\text{Portfolio Value}_t = \text{Cash}_t + \sum_{i=1}^{n} (\text{Shares}_i \times \text{Price}_i)
$$

Where:
- \( \text{Cash}_t \) is the available cash.
- \( \text{Shares}_i \) represents held shares of stock \( i \).
- \( \text{Price}_i \) is the stock’s market price.

# Reinforcement Learning for Trading

## Overview
This repository implements a Reinforcement Learning (RL) agent for trading financial instruments. The agent optimizes a trading strategy using a custom reward function that balances profit maximization, risk management, and drawdown control.

## Features
- **Custom Reward Function**: Encourages profitable trades while penalizing excessive risk and drawdown.
- **Action Space**: Supports buying, selling, and holding assets.
- **State Representation**: Uses technical indicators and price history.
- **Risk Management**: Incorporates volatility and drawdown constraints.

## Installation
To set up the environment, run:
```bash
pip install -r requirements.txt
```

## Usage
Run the training script:
```bash
python train.py
```
For evaluation:
```bash
python evaluate.py
```

## Reward Function
The reward function is defined as follows:

### 1. **Profit-Based Term:**
\[\
f_{\text{profit}} = \begin{cases}
10 k_p |r_t|, & \text{if } r_t > 0 \\
-10 k_l |r_t|, & \text{if } r_t \leq 0
\end{cases}
\]
where:
- \( r_t \) is the portfolio return at time \( t \), scaled by 100.
- \( k_p \) and \( k_l \) are coefficients for profit and loss.
- \( f_{\text{profit}} \) is clamped between \([-100, 100]\).

### 2. **Risk-Based Penalty:**
\[\
\text{penalty}_{\text{risk}} = w_{\text{risk}} \cdot \left( k_r \cdot \frac{\sigma^\beta}{\log(1 + e^{\lambda (\mu - \sigma)})} \right)
\]
where:
- \( \sigma \) is the standard deviation of recent returns (volatility).
- \( \mu \) is the mean return.
- \( \lambda \) and \( \beta \) are risk sensitivity parameters.

### 3. **Drawdown-Based Penalty:**
\[\
\text{penalty}_{\text{drawdown}} = w_{\text{drawdown}} \cdot \left( k_d \cdot \frac{D^\delta}{\log(1 + e^{\rho (V - (V(1 - D)))})} \right)
\]
where:
- \( D \) is the drawdown.
- \( V \) is the portfolio value.
- \( k_d, \delta, \rho \) are drawdown sensitivity parameters.

### 4. **Action-Based Penalty:**
\[\
\text{penalty}_{\text{action}} = \begin{cases}
w_{\text{action}} \cdot k_a ||a||^2 e^{-\eta |r_t|}, & \text{if } |r_t| < r_{\text{threshold}} \\
0, & \text{otherwise}
\end{cases}
\]
where:
- \( a \) is the action vector.
- \( \eta \) controls the effect of portfolio return on the penalty.

### 5. **Total Penalty:**
\[\
\text{total penalty} = \min(\text{penalty}_{\text{risk}} + \text{penalty}_{\text{drawdown}} + \text{penalty}_{\text{action}}, 0.99)
\]

### 6. **Final Reward Function:**
\[\
R_t = \text{sign}(r_t) \cdot w_{\text{profit}} \cdot f_{\text{profit}} \cdot (1 - \text{total penalty})
\]

### 7. **Weight Updates:**
The weight update rule for each component is:
\[\
\Delta w = \text{clip}(\phi \cdot \text{gradient}, -\epsilon, \epsilon)
\]
where:
- \( \phi \) are the learning rates.
- \( \epsilon \) prevents instability.
- The weight updates ensure they remain above a minimum threshold.

## Contribution
Feel free to open an issue or submit a pull request if you have any suggestions or improvements.

## License
This is Licensed By us..

## Citations
Algorithmic Trading Using Continuous Action Space
Deep Reinforcement Learning
Naseh Majidia
(naseh.majidi@ee.sharif.edu), Mahdi Shamsia
(shamsi.mahdi@ee.sharif.edu), Farokh Marvastia
(marvasti@sharif.edu)
https://datascience.stackexchange.com/questions/26342/reasoning-for-temporal-difference-update-rule


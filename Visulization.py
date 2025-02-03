import pygame
import numpy as np
from typing import List, Tuple


class TradingVisualizer:
    def __init__(self, width=1280, height=720):
        pygame.init()
        pygame.font.init()

        self.width = width
        self.height = height
        self.screen = pygame.display.set_mode((width, height))
        pygame.display.set_caption("Real-time Trading Training Visualization")

        # Colors
        self.WHITE = (255, 255, 255)
        self.BLACK = (0, 0, 0)
        self.RED = (255, 50, 50)
        self.GREEN = (50, 255, 50)
        self.BLUE = (50, 50, 255)
        self.GRAY = (128, 128, 128)

        # Fonts
        self.font_large = pygame.font.SysFont('Arial', 36)
        self.font_medium = pygame.font.SysFont('Arial', 24)
        self.font_small = pygame.font.SysFont('Arial', 18)

        # Chart settings
        self.chart_margin = 50
        self.chart_width = width - 2 * self.chart_margin
        self.chart_height = height * 0.6

        # Training metrics
        self.episode_rewards = []
        self.price_history = []
        self.portfolio_history = []
        self.action_history = []
        self.training_metrics = {
            'episode': 0,
            'window': 0,
            'total_reward': 0,
            'avg_reward': 0
        }

    def update(self, state, action, portfolio_value, current_price, metrics):
        """Update visualization with new state and training metrics"""
        self.price_history.append(current_price)
        self.portfolio_history.append(portfolio_value)
        self.action_history.append(action)
        self.training_metrics.update(metrics)

        # Keep only last 100 points
        max_history = 100
        if len(self.price_history) > max_history:
            self.price_history = self.price_history[-max_history:]
            self.portfolio_history = self.portfolio_history[-max_history:]
            self.action_history = self.action_history[-max_history:]

        self._draw_frame(state, action, portfolio_value, current_price)

        # Handle pygame events
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                return False
        return True

    def _draw_frame(self, state, action, portfolio_value, current_price):
        """Draw visualization frame"""
        self.screen.fill(self.WHITE)
        self._draw_chart()
        self._draw_portfolio_info(portfolio_value, current_price, state, action)
        self._draw_training_metrics()
        pygame.display.flip()

    def _draw_chart(self):
        """Draw price and portfolio charts"""
        if len(self.price_history) < 2:
            return

        # Calculate scaling factors
        price_min = min(self.price_history) if self.price_history else 0
        price_max = max(self.price_history) if self.price_history else 1
        price_range = max(price_max - price_min, 0.01)

        portfolio_min = min(self.portfolio_history) if self.portfolio_history else 0
        portfolio_max = max(self.portfolio_history) if self.portfolio_history else 1
        portfolio_range = max(portfolio_max - portfolio_min, 0.01)

        # Draw price line
        points = []
        for i, price in enumerate(self.price_history):
            x = self.chart_margin + (i / len(self.price_history)) * self.chart_width
            y = self.chart_margin + ((price - price_min) / price_range) * self.chart_height
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(self.screen, self.BLUE, False, points, 2)

        # Draw portfolio value line
        points = []
        for i, value in enumerate(self.portfolio_history):
            x = self.chart_margin + (i / len(self.portfolio_history)) * self.chart_width
            y = self.chart_margin + ((value - portfolio_min) / portfolio_range) * self.chart_height
            points.append((x, y))

        if len(points) > 1:
            pygame.draw.lines(self.screen, self.GREEN, False, points, 2)

        # Draw actions
        for i, action in enumerate(self.action_history):
            x = self.chart_margin + (i / len(self.action_history)) * self.chart_width
            y = self.chart_margin + ((self.price_history[i] - price_min) / price_range) * self.chart_height

            if isinstance(action, (tuple, list)):
                direction = action[0]
            else:
                direction = action

            if direction > 0:  # Buy
                color = self.GREEN
                size = 5
            elif direction < 0:  # Sell
                color = self.RED
                size = 5
            else:  # Hold
                continue

            pygame.draw.circle(self.screen, color, (int(x), int(y)), size)

    def _draw_portfolio_info(self, portfolio_value, current_price, state, action):
        """Draw portfolio information"""
        y_start = self.height - 200

        # Portfolio Value
        text = self.font_large.render(f"Portfolio: ${portfolio_value:,.2f}", True, self.BLACK)
        self.screen.blit(text, (self.chart_margin, y_start))

        # Current Price
        text = self.font_medium.render(f"Price: ${current_price:,.2f}", True, self.BLACK)
        self.screen.blit(text, (self.chart_margin, y_start + 50))

        # Action
        if isinstance(action, (tuple, list)):
            direction = action[0]
        else:
            direction = action
        action_text = "BUY" if direction > 0 else "SELL" if direction < 0 else "HOLD"
        text = self.font_medium.render(f"Action: {action_text}", True, self.BLACK)
        self.screen.blit(text, (self.chart_margin, y_start + 100))

    def _draw_training_metrics(self):
        """Draw training metrics"""
        y_start = 20
        metrics = [
            f"Window: {self.training_metrics['window']}",
            f"Episode: {self.training_metrics['episode']}",
            f"Total Reward: {self.training_metrics['total_reward']:.2f}",
            f"Avg Reward: {self.training_metrics['avg_reward']:.2f}"
        ]

        for i, metric in enumerate(metrics):
            text = self.font_medium.render(metric, True, self.BLACK)
            self.screen.blit(text, (self.width - 300, y_start + i * 30))

    def close(self):
        """Close visualization"""
        pygame.quit()
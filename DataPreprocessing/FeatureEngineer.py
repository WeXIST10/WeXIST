import pandas as pd
import os
import logging
import ta

class StockFeatureEngineer:
    """
    Class for adding financial features to stock data using technical indicators.
    """
    
    def __init__(self, log_file='feature_engineering.log', log_level=logging.INFO):
        """
        Initialize the feature engineer with logging configuration.

        Parameters:
            log_file (str): Log file name
            log_level (int): Logging level (e.g., logging.INFO)
        """
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(log_level)
        
        formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
        
        # File handler
        file_handler = logging.FileHandler(log_file)
        file_handler.setFormatter(formatter)
        file_handler.setLevel(log_level)
        
        # Console handler
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        console_handler.setLevel(log_level)
        
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)

    def calculate_sma(self, df, period, column='Close'):
        """Calculate Simple Moving Average using ta library"""
        return ta.trend.sma_indicator(df[column], window=period)

    def calculate_ema(self, df, period, column='Close'):
        """Calculate Exponential Moving Average using ta library"""
        return ta.trend.ema_indicator(df[column], window=period)

    def calculate_macd(self, df):
        """Calculate MACD indicator using ta library"""
        return ta.trend.macd(df['Close'])

    def calculate_rsi(self, df, period=14, column='Close'):
        """Calculate Relative Strength Index using ta library"""
        return ta.momentum.rsi(df[column], window=period)

    def calculate_cci(self, df, period=14):
        """Calculate Commodity Channel Index using ta library"""
        return ta.trend.cci(df['High'], df['Low'], df['Close'], window=period)

    def calculate_adx(self, df, period=14):
        """Calculate Average Directional Index using ta library"""
        return ta.trend.adx(df['High'], df['Low'], df['Close'], window=period)

    def calculate_donchian_bands(self, df, period=20):
        """Calculate Donchian Bands"""
        df['DU'] = df['High'].rolling(window=period).max()
        df['DL'] = df['Low'].rolling(window=period).min()
        return df

    def calculate_volatility_metrics(self, df, period=14):
        """Calculate Volatility Metrics (ATR-based) using ta library"""
        df['VM'] = ta.volatility.average_true_range(df['High'], df['Low'], 
                                                     df['Close'], window=period)
        return df

    def calculate_bollinger_bands(self, df, period=20, column='Close'):
        """Calculate Bollinger Bands using ta library"""
        indicator_bb = ta.volatility.BollingerBands(df[column], window=period, window_dev=2)
        df['BB_upper'] = indicator_bb.bollinger_hband()
        df['BB_lower'] = indicator_bb.bollinger_lband()
        df['BB_middle'] = indicator_bb.bollinger_mavg()  # Middle band is SMA
        return df

    def calculate_stochastic_oscillator(self, df, period=14):
        """Calculate Stochastic Oscillator using ta library"""
        df['SOk'] = ta.momentum.stoch(df['High'], df['Low'], df['Close'], window=period)
        df['SOd'] = ta.momentum.stoch_signal(df['High'], df['Low'], df['Close'], window=period)
        return df

    def calculate_obv(self, df):
        """Calculate On-Balance Volume using ta library"""
        df['OBV'] = ta.volume.on_balance_volume(df['Close'], df['Volume'])
        return df

    def run_feature_engineering(self, input_dir, output_dir):
        """
        Main method to process all files in a directory and add features.

        Parameters:
            input_dir (str): Directory containing input CSV files
            output_dir (str): Directory to save feature-enhanced data
        """
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            self.logger.info(f"Created output directory: {output_dir}")

        for filename in os.listdir(input_dir):
            if not filename.endswith('.csv'):
                self.logger.warning(f"Skipping non-CSV file: {filename}")
                continue

            try:
                self.logger.info(f"Processing file: {filename}")
                file_path = os.path.join(input_dir, filename)
                
                # Read data
                df = pd.read_csv(file_path, index_col='Date', parse_dates=True)
                
                # Calculate and add features
                df['SMA50'] = self.calculate_sma(df, 50)
                df['SMA200'] = self.calculate_sma(df, 200)
                df['EMA12'] = self.calculate_ema(df, 12)
                df['EMA26'] = self.calculate_ema(df, 26)
                
                df['MACD'] = self.calculate_macd(df)
                df['RSI'] = self.calculate_rsi(df)
                df['CCI'] = self.calculate_cci(df)
                df['ADX'] = self.calculate_adx(df)
                self.calculate_stochastic_oscillator(df)
                
                self.calculate_donchian_bands(df)
                self.calculate_volatility_metrics(df)
                self.calculate_bollinger_bands(df)
                
                self.calculate_obv(df)
                
                # Handle missing values
                df.fillna(method='ffill', inplace=True)
                df.fillna(method='bfill', inplace=True)
                
                # Save processed file
                output_file = os.path.join(output_dir, f"features_{filename}")
                df.to_csv(output_file)
                self.logger.info(f"Saved to {output_file}")

            except Exception as e:
                self.logger.error(f"Error processing {filename}: {str(e)}")


if __name__ == "__main__":
    input_directory = 'preprocessed_data'
    output_directory = 'feature_added_data'
    
    # Initialize the feature engineer
    feature_engineer = StockFeatureEngineer()
    
    # Run feature engineering pipeline
    feature_engineer.run_feature_engineering(input_directory, output_directory)
import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb
from sklearn.metrics import mean_squared_error

class StockPipeline:
    def __init__(self, nifty_url, num_stocks=200):
        self.nifty_url = nifty_url
        self.num_stocks = num_stocks
        self.features = [
            "MarketCap", "RevGrowth", "EPSGrowth", "ROE",
            "DebtToEquity", "PERatio", "PBRatio", "DivYield"
        ]
        self.predictions = None
        self.model = None
        self.historical_data = None

    def fetch_nifty_list(self):
        """Step 1: Fetch NIFTY 500 list and process symbols"""
        df_nse = pd.read_csv(self.nifty_url)
        df_nse.rename(columns={"Symbol": "SYMBOL"}, inplace=True)
        df_nse["Yahoo_Ticker"] = df_nse["SYMBOL"].apply(lambda x: f"{x.strip()}.NS")
        return df_nse

    def collect_historical_metrics(self, tickers):
        """Step 2: Collect historical metrics for each ticker"""
        all_data = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                hist = stock.history(period="max")
                
                # Calculate 3-year rolling returns and collect metrics
                for i in range(len(hist) - 3*252):
                    start_price = hist.iloc[i]['Close']
                    end_price = hist.iloc[i+3*252]['Close']
                    return_pct = (end_price - start_price) / start_price
                    
                    metrics = {
                        "Ticker": ticker,
                        "MarketCap": stock.info.get("marketCap", np.nan),
                        "RevGrowth": stock.info.get("revenueGrowth", np.nan),
                        "EPSGrowth": stock.info.get("earningsGrowth", np.nan),
                        "ROE": stock.info.get("returnOnEquity", np.nan),
                        "DebtToEquity": stock.info.get("debtToEquity", np.nan),
                        "PERatio": stock.info.get("trailingPE", np.nan),
                        "PBRatio": stock.info.get("priceToBook", np.nan),
                        "DivYield": stock.info.get("dividendYield", np.nan),
                        "3YReturn": return_pct
                    }
                    all_data.append(metrics)
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
        return pd.DataFrame(all_data).dropna()

    def prepare_split_data(self, df):
        """Step 3: Prepare and split data into train/test sets"""
        df = df.dropna()
        X = df[self.features]
        y = df["3YReturn"]
        
        train_size = int(len(X) * 0.8)
        X_train, y_train = X.iloc[:train_size], y.iloc[:train_size]
        X_test, y_test = X.iloc[train_size:], y.iloc[train_size:]
        
        return X_train, y_train, X_test, y_test

    def train_model(self, X_train, y_train):
        """Step 4: Train XGBoost model"""
        model = xgb.XGBRegressor(
            objective="reg:squarederror",
            n_estimators=200,
            learning_rate=0.05,
            max_depth=6
        )
        model.fit(X_train, y_train)
        return model

    def evaluate_model(self, model, X_test, y_test):
        """Step 5: Evaluate model performance"""
        predictions = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, predictions))
        print(f"RMSE on test data: {rmse:.2f}")
        return rmse

    def collect_latest_metrics(self, tickers):
        """Step 6: Collect latest metrics for predictions"""
        latest_features = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                latest_features.append({
                    "Ticker": ticker,
                    "MarketCap": stock.info.get("marketCap", np.nan),
                    "RevGrowth": stock.info.get("revenueGrowth", np.nan),
                    "EPSGrowth": stock.info.get("earningsGrowth", np.nan),
                    "ROE": stock.info.get("returnOnEquity", np.nan),
                    "DebtToEquity": stock.info.get("debtToEquity", np.nan),
                    "PERatio": stock.info.get("trailingPE", np.nan),
                    "PBRatio": stock.info.get("priceToBook", np.nan),
                    "DivYield": stock.info.get("dividendYield", np.nan)
                })
            except:
                continue
        return pd.DataFrame(latest_features).dropna()

    def predict_returns(self, model, latest_df):
        """Step 7: Generate future return predictions"""
        latest_X = latest_df[self.features]
        latest_df["PredictedReturn"] = model.predict(latest_X)
        return latest_df

    def run_pipeline(self):
        # Step 1: Fetch NIFTY 500 list
        print("Fetching NIFTY 500 list...")
        df_nse = self.fetch_nifty_list()
        
        # Step 2: Collect historical data
        print("Collecting historical metrics...")
        tickers = df_nse["Yahoo_Ticker"].tolist()[:self.num_stocks]
        self.historical_data = self.collect_historical_metrics(tickers)
        
        # Step 3: Prepare and split data
        print("Preparing and splitting data...")
        X_train, y_train, X_test, y_test = self.prepare_split_data(self.historical_data)
        
        # Step 4: Train model
        print("Training model...")
        self.model = self.train_model(X_train, y_train)
        
        # Save the trained model
        self.model.save_model("xgboost_model.bin")  # <== Save the model
        
        # Step 5: Evaluate model
        print("Evaluating model...")
        self.evaluate_model(self.model, X_test, y_test)
        
        # Step 6: Collect latest data
        print("Collecting latest metrics...")
        latest_df = self.collect_latest_metrics(tickers)
        
        # Step 7: Generate predictions
        print("Generating predictions...")
        predictions_df = self.predict_returns(self.model, latest_df)
        
        # Step 8: Return top 10 stocks
        self.predictions = predictions_df.sort_values(
            by="PredictedReturn", ascending=False).head(10)
        
        return self.predictions[["Ticker", "PredictedReturn"]]


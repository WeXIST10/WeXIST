import pandas as pd
import numpy as np
import yfinance as yf
import xgboost as xgb

class BacktestFundamental:
    def __init__(self, nifty_url, model_path, num_stocks=300):
        self.nifty_url = nifty_url
        self.model_path = model_path
        self.num_stocks = num_stocks
        self.features = [
            "MarketCap", "RevGrowth", "EPSGrowth", "ROE",
            "DebtToEquity", "PERatio", "PBRatio", "DivYield"
        ]
        self.model = None
        
    def load_model(self):
        """Load the pre-trained XGBoost model"""
        model = xgb.XGBRegressor()
        model.load_model(self.model_path)
        return model
        
    def fetch_nifty_list(self):
        """Fetch NIFTY 500 list and process symbols"""
        df_nse = pd.read_csv(self.nifty_url)
        df_nse.rename(columns={"Symbol": "SYMBOL"}, inplace=True)
        df_nse["Yahoo_Ticker"] = df_nse["SYMBOL"].apply(lambda x: f"{x.strip()}.NS")
        return df_nse
        
    def collect_current_metrics(self, tickers):
        """Collect current fundamental metrics for predictions"""
        current_features = []
        for ticker in tickers:
            try:
                stock = yf.Ticker(ticker)
                metrics = {
                    "Ticker": ticker,
                    "MarketCap": stock.info.get("marketCap", np.nan),
                    "RevGrowth": stock.info.get("revenueGrowth", np.nan),
                    "EPSGrowth": stock.info.get("earningsGrowth", np.nan),
                    "ROE": stock.info.get("returnOnEquity", np.nan),
                    "DebtToEquity": stock.info.get("debtToEquity", np.nan),
                    "PERatio": stock.info.get("trailingPE", np.nan),
                    "PBRatio": stock.info.get("priceToBook", np.nan),
                    "DivYield": stock.info.get("dividendYield", np.nan)
                }
                current_features.append(metrics)
            except Exception as e:
                print(f"Error processing {ticker}: {str(e)}")
                continue
        return pd.DataFrame(current_features).dropna()

    def predict_returns(self, df):
        """Generate predictions using the pre-trained model"""
        X = df[self.features]
        df["PredictedReturn"] = self.model.predict(X)
        return df.sort_values("PredictedReturn", ascending=False)

    def run_backtest(self):
        """Execute the backtesting pipeline"""
        print("Loading pre-trained model...")
        self.model = self.load_model()
        
        print("Fetching NIFTY 500 list...")
        df_nse = self.fetch_nifty_list()
        
        print("Collecting current metrics...")
        tickers = df_nse["Yahoo_Ticker"].tolist()[:self.num_stocks]
        current_data = self.collect_current_metrics(tickers)
        
        print("Generating predictions...")
        predictions = self.predict_returns(current_data)
        
        return predictions[["Ticker", "PredictedReturn"]].head(10)

# Example usage
if __name__ == "__main__":
    nifty_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
    model_path = r"C:\Darsh\WeXIST_FINAL\xgboost_model.bin"
    
    backtest = BacktestFundamental(nifty_url, model_path)
    top_stocks = backtest.run_backtest()
    top_10_tickers = top_stocks.head(10)[['Ticker']]
    print("\nTop 10 Stocks from Backtest:")
    print(top_stocks)
    top_stocks.to_csv("backtest_results.csv", index=False)
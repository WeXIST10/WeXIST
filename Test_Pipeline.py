from Fundamental.test_fundamental import BacktestFundamental
from DataPreprocessing.DataPreprocessingPipeline import StockPreProcessPipeline
from Environment.backtest import TD3Backtester   

nifty_url = "https://archives.nseindia.com/content/indices/ind_nifty500list.csv"
model_path = r"C:\Darsh\WeXIST_FINAL\xgboost_model.bin"
    
backtest = BacktestFundamental(nifty_url, model_path , 200)
top_stocks = backtest.run_backtest()
print("\nTop 10 Stocks from Backtest:")
print(top_stocks)


top_10_tickers = top_stocks.head(10)[['Ticker']]
 
top_stocks.to_csv("top_10_stocks.csv", index=False)
print("\nTop 10 Predicted Stocks:")
print(top_10_tickers)

pipeline = StockPreProcessPipeline()
ticker_list = top_10_tickers['Ticker'].tolist()
start_date = "2021-01-01"
end_date = "2024-01-01"

combined_csv_path = pipeline.run_data_pipeline(ticker_list, start_date, end_date)
print(f"Combined data saved to: {combined_csv_path}")
backtester = TD3Backtester(
        model_path=r"C:\Darsh\WeXIST_FINAL\td3_stock_trading.zip",
        env_path=r"C:\Darsh\WeXIST_FINAL\td3_stock_trading_env.pkl",
        output_dir=r"C:\Darsh\WeXIST_FINAL\backtest_results"
    )
    
results = backtester.run_backtest(r"C:\Darsh\WeXIST_FINAL\combined_stock_data.csv")
for key, value in results.items():
    print(f"{key}: {value}")

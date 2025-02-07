from DataPreprocessing.DataPreprocessingPipeline import StockPreProcessPipeline
from Environment.New_testing import TD3TradingBot
import math
import os

def train_stock_models(tickers, start_date, end_date, batch_size=7, checkpoint_dir="./checkpoints"):
    
    pipeline = StockPreProcessPipeline()
    bot = TD3TradingBot()  
    
    os.makedirs(checkpoint_dir, exist_ok=True)
    
    n_batches = math.ceil(len(tickers) / batch_size)
    current_model_path = None
    current_env_path = None
    
    total_steps = 0
    
    for batch_idx in range(n_batches):
        # Get current batch of tickers
        start_idx = batch_idx * batch_size
        end_idx = min((batch_idx + 1) * batch_size, len(tickers))
        current_tickers = tickers[start_idx:end_idx]
        
        print(f"\nProcessing batch {batch_idx + 1}/{n_batches}")
        print(f"Tickers: {current_tickers}")
        
        try:
            combined_csv_path = pipeline.run_data_pipeline(
                current_tickers, 
                start_date, 
                end_date
            )
            print(f"Combined data saved to: {combined_csv_path}")
            
            if batch_idx == 0:
                print("Training initial model from scratch...")
                current_model_path, current_env_path = bot.train_from_scratch(
                    csv_file_path=combined_csv_path,
                    total_timesteps=75000,
                    save_path=checkpoint_dir
                )
                print(f"Initial model saved to: {current_model_path}")
                
            else:
                print("Training with pretrained model...")
                current_model_path, current_env_path = bot.train_pretrained(
                    csv_file_path=combined_csv_path,
                    model_path=current_model_path,
                    env_path=current_env_path,
                    total_timesteps=50000,
                    save_path=checkpoint_dir
                )
                print(f"Updated model saved to: {current_model_path}")
            
            batch_checkpoint_path = os.path.join(
                checkpoint_dir, 
                f"batch_{batch_idx + 1}_model"
            )
            batch_env_path = os.path.join(
                checkpoint_dir,
                f"batch_{batch_idx + 1}_env.pkl"
            )
            
            if current_model_path and current_env_path:
                os.system(f"cp {current_model_path}.zip {batch_checkpoint_path}.zip")
                os.system(f"cp {current_env_path} {batch_env_path}")
                print(f"Batch checkpoint saved: {batch_checkpoint_path}")
                
        except Exception as e:
            print(f"Error processing batch {batch_idx + 1}: {str(e)}")
            # Save emergency checkpoint if we have a model
            if current_model_path and current_env_path:
                emergency_path = os.path.join(
                    checkpoint_dir,
                    f"emergency_batch_{batch_idx + 1}"
                )
                os.system(f"cp {current_model_path}.zip {emergency_path}_model.zip")
                os.system(f"cp {current_env_path} {emergency_path}_env.pkl")
                print(f"Emergency checkpoint saved: {emergency_path}")
            continue
            
    return current_model_path, current_env_path

if __name__ == "__main__":
    tickers = [ 
        "RELIANCE.NS", "TCS.NS", "INFY.NS", "HDFCBANK.NS", "ICICIBANK.NS",
        "HINDUNILVR.NS", "HDFC.NS", "KOTAKBANK.NS", "ITC.NS", "LT.NS",
        "AXISBANK.NS", "SBIN.NS", "BHARTIARTL.NS", "ASIANPAINT.NS", "MARUTI.NS",
        "BAJFINANCE.NS", "M&M.NS", "NESTLEIND.NS", "SUNPHARMA.NS", "WIPRO.NS",
        "ULTRACEMCO.NS", "HCLTECH.NS", "TITAN.NS", "BAJAJFINSV.NS", "POWERGRID.NS",
        "NTPC.NS", "TATASTEEL.NS", "INDUSINDBK.NS", "JSWSTEEL.NS", "TECHM.NS",
        "ADANIPORTS.NS", "GRASIM.NS", "CIPLA.NS", "BAJAJ-AUTO.NS", "TATAMOTORS.NS",
        "DIVISLAB.NS", "DRREDDY.NS", "BRITANNIA.NS", "HEROMOTOCO.NS", "EICHERMOT.NS",
        "HINDALCO.NS", "SBILIFE.NS", "TATACONSUM.NS", "UPL.NS", "APOLLOHOSP.NS",
        "COALINDIA.NS", "BPCL.NS", "ADANIENT.NS", "ONGC.NS", "HDFCLIFE.NS"
    ]
    
    start_date = "2015-01-01"
    end_date = "2021-01-01"
    checkpoint_dir = "./model_checkpoints"  
    
    # Run the training pipeline
    try:
        final_model_path, final_env_path = train_stock_models(
            tickers=tickers,
            start_date=start_date,
            end_date=end_date,
            batch_size=7,
            checkpoint_dir=checkpoint_dir
        )
        
        print("\nTraining complete!")
        print(f"Final model path: {final_model_path}")
        print(f"Final environment path: {final_env_path}")
        
    except Exception as e:
        print(f"Error in training pipeline: {str(e)}")
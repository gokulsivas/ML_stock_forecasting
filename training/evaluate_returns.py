import torch
import numpy as np
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt
import sys
sys.path.append('..')

from data.data_loader import StockDataLoader
from models.hybrid_lstm_gru import HybridLSTMGRU
from training.dataset import StockSequenceDataset
from torch.utils.data import DataLoader
from sklearn.preprocessing import StandardScaler
import joblib


def evaluate_returns_model(test_symbol):
    """Evaluate returns-based model"""
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load data
    loader = StockDataLoader()
    df = loader.load_stock_data(test_symbol)
    
    # Calculate returns
    df['target_return'] = df['close_price'].pct_change().shift(-1)
    
    # Load feature columns
    feature_cols = joblib.load('saved_models/returns_feature_cols.pkl')
    
    df = df.dropna()
    
    # Normalize features
    scaler = StandardScaler()
    for lag in [1, 2, 3, 5, 10]:
        df[f'return_lag_{lag}'] = df['close_price'].pct_change().shift(lag)
    df['volume_spike'] = df['volume'] / (df['volume'].rolling(20).mean() + 1e-9)
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna()
    X = scaler.fit_transform(df[feature_cols].values)
    y = df['target_return'].values.reshape(-1, 1)
    
    # Create sequences
    sequence_length = 90
    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])
    
    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)
    
    # Use last 15% as test
    test_size = int(len(X_seq) * 0.15)
    X_test = X_seq[-test_size:]
    y_test = y_seq[-test_size:]
    
    # Get corresponding prices for plotting
    test_prices = df['close_price'].values[-(test_size+1):]  # +1 for reconstruction
    
    # Load model
    input_size = X_seq.shape[2]
    model = HybridLSTMGRU(input_size=input_size, hidden_size=128, num_layers=2)
    model.load_state_dict(torch.load('saved_models/returns_model.pth', map_location=device))
    model.to(device)
    model.eval()
    
    # Predict
    test_dataset = StockSequenceDataset(X_test, y_test)
    test_loader = DataLoader(test_dataset, batch_size=32)
    
    predicted_returns = []
    actual_returns = []
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch = X_batch.to(device)
            outputs = model(X_batch)
            predicted_returns.extend(outputs.cpu().numpy())
            actual_returns.extend(y_batch.numpy())
    
    predicted_returns = np.array(predicted_returns).flatten()
    actual_returns = np.array(actual_returns).flatten()
    
    # Calculate metrics on returns
    rmse = np.sqrt(mean_squared_error(actual_returns, predicted_returns))
    mae = mean_absolute_error(actual_returns, predicted_returns)
    r2 = r2_score(actual_returns, predicted_returns)
    mape = np.mean(np.abs((actual_returns - predicted_returns) / (actual_returns + 1e-10))) * 100
    
    # Directional accuracy (most important for trading)
    actual_direction = np.sign(actual_returns)
    pred_direction = np.sign(predicted_returns)
    directional_accuracy = np.mean(actual_direction == pred_direction) * 100
    
    print(f"\n{'='*60}")
    print(f"Returns Model - Evaluation Results for {test_symbol}")
    print(f"{'='*60}")
    print(f"RMSE (returns): {rmse:.6f}")
    print(f"MAE (returns): {mae:.6f}")
    print(f"R² (returns): {r2:.4f}")
    print(f"MAPE (returns): {mape:.2f}%")
    print(f"Directional Accuracy: {directional_accuracy:.2f}%")
    print(f"{'='*60}")
    
    # Reconstruct prices from returns for visualization
    reconstructed_prices = [test_prices[0]]
    for ret in predicted_returns:
        next_price = reconstructed_prices[-1] * (1 + ret)
        reconstructed_prices.append(next_price)
    
    actual_prices = test_prices[1:len(predicted_returns)+1]
    reconstructed_prices = reconstructed_prices[1:]
    
    # Plot price reconstruction
    plt.figure(figsize=(14, 7))
    
    plt.subplot(2, 1, 1)
    plt.plot(actual_prices, label='Actual Prices', alpha=0.7, linewidth=2)
    plt.plot(reconstructed_prices, label='Predicted Prices', alpha=0.7, linewidth=2)
    plt.title(f'{test_symbol} - Returns Model: Price Predictions')
    plt.ylabel('Price (₹)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.subplot(2, 1, 2)
    plt.plot(actual_returns * 100, label='Actual Returns', alpha=0.7, linewidth=1)
    plt.plot(predicted_returns * 100, label='Predicted Returns', alpha=0.7, linewidth=1)
    plt.title('Daily Returns Comparison')
    plt.xlabel('Time Steps')
    plt.ylabel('Return (%)')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig(f'experiments/{test_symbol}_returns_model.png', dpi=150)
    print(f"\nPlot saved to experiments/{test_symbol}_returns_model.png")
    
    return {
        'rmse': rmse,
        'mae': mae,
        'r2': r2,
        'mape': mape,
        'directional_accuracy': directional_accuracy
    }


if __name__ == "__main__":
    metrics = evaluate_returns_model('20MICRONS')

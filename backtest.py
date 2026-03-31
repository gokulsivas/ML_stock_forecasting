import sys
sys.path.append('.')

import torch
import numpy as np
import pandas as pd
import joblib
from datetime import datetime, timedelta
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error

from models.hybrid_lstm_gru import HybridLSTMGRU
from data.data_loader import StockDataLoader

# ── Config ──────────────────────────────────────────────
SYMBOLS      = ['RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK']
MODEL_PATH   = 'saved_models/returns_model.pth'
FEATURE_PATH = 'saved_models/returns_feature_cols.pkl'
HIDDEN_SIZE  = 512   # ✅ CHANGED: was 256
NUM_LAYERS   = 4     # ✅ CHANGED: was 3
DAYS_TO_TEST = 7
LOOKBACK     = 90
# ────────────────────────────────────────────────────────

device       = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
feature_cols = joblib.load(FEATURE_PATH)

model = HybridLSTMGRU(
    input_size  = len(feature_cols),
    hidden_size = HIDDEN_SIZE,
    num_layers  = NUM_LAYERS,
    dropout     = 0.3             # ✅ ADDED: match training architecture
)

# ✅ CHANGED: handles full checkpoint dict saved by new train.py
checkpoint = torch.load(MODEL_PATH, map_location=device, weights_only=False)
if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
    model.load_state_dict(checkpoint['model_state_dict'])
else:
    model.load_state_dict(checkpoint)

model.to(device)
model.eval()

loader  = StockDataLoader()
results = []

for symbol in SYMBOLS:
    print(f"\n{'='*50}")
    print(f"  Backtesting: {symbol}")
    print(f"{'='*50}")

    df = loader.load_stock_data(symbol)
    if df is None or len(df) < LOOKBACK + DAYS_TO_TEST + 10:
        print(f"  Not enough data for {symbol}, skipping.")
        continue

    df['target_return'] = df['close_price'].pct_change().shift(-1)
    df = df.dropna()

    cutoff_idx = len(df) - DAYS_TO_TEST
    history_df = df.iloc[:cutoff_idx].copy()
    actual_df  = df.iloc[cutoff_idx:cutoff_idx + DAYS_TO_TEST].copy()

    if len(history_df) < LOOKBACK:
        print(f"  Not enough history, skipping.")
        continue

    scaler = StandardScaler()
    scaler.fit_transform(history_df[feature_cols].values)
    scaled = scaler.transform(history_df[feature_cols].values)

    sequence         = scaled[-LOOKBACK:].copy()
    last_price       = float(history_df['close_price'].iloc[-1])
    predicted_prices = []

    with torch.no_grad():
        for i in range(DAYS_TO_TEST):
            X           = torch.FloatTensor(sequence).unsqueeze(0).to(device)
            pred_return = model(X).cpu().numpy()[0, 0]
            predicted_price = last_price * (1 + pred_return)
            predicted_prices.append(predicted_price)
            last_price = predicted_price

            new_row    = sequence[-1].copy()
            new_row[0] = pred_return
            for lag in range(min(5, len(new_row) - 1), 0, -1):
                new_row[lag] = new_row[lag - 1]
            sequence = np.vstack([sequence[1:], new_row.reshape(1, -1)])

    actual_prices  = actual_df['close_price'].values[:DAYS_TO_TEST]

    mae  = mean_absolute_error(actual_prices, predicted_prices)
    rmse = np.sqrt(mean_squared_error(actual_prices, predicted_prices))
    mape = np.mean(np.abs((np.array(actual_prices) - np.array(predicted_prices)) / np.array(actual_prices))) * 100

    pred_directions   = [1 if predicted_prices[i] > predicted_prices[i-1] else -1 for i in range(1, len(predicted_prices))]
    actual_directions = [1 if actual_prices[i]    > actual_prices[i-1]    else -1 for i in range(1, len(actual_prices))]
    dir_acc = np.mean(np.array(pred_directions) == np.array(actual_directions)) * 100

    print(f"\n  {'Day':<6} {'Predicted':>12} {'Actual':>12} {'Error %':>10} {'Direction':>12}")
    print(f"  {'-'*55}")
    for i in range(len(actual_prices)):
        err_pct    = abs(predicted_prices[i] - actual_prices[i]) / actual_prices[i] * 100
        pred_dir   = 'U' if (i > 0 and predicted_prices[i] > predicted_prices[i-1]) else ('D' if i > 0 else '-')
        actual_dir = 'U' if (i > 0 and actual_prices[i]    > actual_prices[i-1])    else ('D' if i > 0 else '-')
        match      = 'OK' if pred_dir == actual_dir else 'X'   # ✅ ASCII only — no Unicode crashes on Windows
        print(f"  {i+1:<6} {predicted_prices[i]:>12.2f} {actual_prices[i]:>12.2f} {err_pct:>9.2f}% {pred_dir} {actual_dir} {match}")

    print(f"\n  MAE:                  Rs.{mae:.2f}")
    print(f"  RMSE:                 Rs.{rmse:.2f}")
    print(f"  MAPE:                 {mape:.2f}%")
    print(f"  Directional Accuracy: {dir_acc:.1f}%")

    results.append({
        'Symbol':           symbol,
        'MAE':              round(mae, 2),
        'RMSE':             round(rmse, 2),
        'MAPE (%)':         round(mape, 2),
        'Dir. Accuracy (%)': round(dir_acc, 1)
    })

print(f"\n\n{'='*60}")
print("  SUMMARY")
print(f"{'='*60}")
summary = pd.DataFrame(results)
print(summary.to_string(index=False))
print(f"\n  Avg MAPE:             {summary['MAPE (%)'].mean():.2f}%")
print(f"  Avg Directional Acc:  {summary['Dir. Accuracy (%)'].mean():.1f}%")
import torch
import numpy as np
import sys
import os

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import StockDataLoader
from data.feature_engineering import FeatureEngineer
from models.hybrid_lstm_gru import HybridLSTMGRU
from sklearn.preprocessing import StandardScaler
import joblib


def evaluate(test_symbol='20MICRONS', model_path='saved_models/returns_model.pth',
             feature_cols_path='saved_models/returns_feature_cols.pkl',
             sequence_length=60):

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\nUsing device: {device}")
    print(f"Evaluating: {test_symbol}")
    print("=" * 50)

    loader   = StockDataLoader()
    df       = loader.load_stock_data(test_symbol)
    engineer = FeatureEngineer()
    df       = engineer.add_technical_indicators(df)

    df['target_return'] = df['close_price'].pct_change().shift(-1)
    df = df.dropna()

    feature_cols = joblib.load(feature_cols_path)
    feature_cols = [c for c in feature_cols if c in df.columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    y = df['target_return'].values

    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])

    X_seq = np.array(X_seq)
    y_seq = np.array(y_seq)

    test_size = int(len(X_seq) * 0.15)
    X_test    = torch.FloatTensor(X_seq[-test_size:]).to(device)
    y_test    = y_seq[-test_size:]

        # ── Load checkpoint ───────────────────────────────────
    checkpoint = torch.load(model_path, map_location=device)
    state_dict = checkpoint['model_state_dict']
    cfg        = checkpoint.get('config', {})
    hidden     = cfg.get('hidden_size', 256)
    layers     = cfg.get('num_layers', 3)
    print(f"Checkpoint epoch  : {checkpoint.get('epoch', 'N/A')}")
    print(f"Model type        : {cfg.get('model_type', 'regression')}")

    model = HybridLSTMGRU(input_size=X_seq.shape[2],
                          hidden_size=hidden, num_layers=layers)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    with torch.no_grad():
        logits = model(X_test).cpu().numpy().flatten()

    # Classification: sigmoid > 0.5 = UP
    probs    = 1 / (1 + np.exp(-logits))        # sigmoid
    pred_dir = (probs > 0.5).astype(float)      # 1=UP, 0=DOWN
    actual_dir = (y_test > 0).astype(float)     # 1=UP, 0=DOWN

    directional_acc = np.mean(actual_dir == pred_dir) * 100
    up_acc   = np.mean(pred_dir[actual_dir == 1]  == 1)  * 100
    down_acc = np.mean(pred_dir[actual_dir == -1] == -1) * 100

    correct_up   = int(np.sum((pred_dir == 1)  & (actual_dir == 1)))
    correct_down = int(np.sum((pred_dir == -1) & (actual_dir == -1)))
    total_up     = int(np.sum(actual_dir == 1))
    total_down   = int(np.sum(actual_dir == -1))

    print(f"Test samples      : {test_size}")
    print(f"Features used     : {len(feature_cols)}")
    print("-" * 50)
    print(f"Directional Acc   : {directional_acc:.2f}%")
    print(f"  UP  correct     : {correct_up}/{total_up}  ({up_acc:.1f}%)")
    print(f"  DOWN correct    : {correct_down}/{total_down}  ({down_acc:.1f}%)")
    print("-" * 50)

    if directional_acc >= 58:
        print("✅ GREAT — Model is strong (>=58%)")
    elif directional_acc >= 55:
        print("✅ GOOD  — Model is profitable (>=55%)")
    elif directional_acc >= 52:
        print("⚠️  WEAK  — Marginal, needs more training")
    else:
        print("❌ BAD   — Below random, retrain needed")

    print("=" * 50)
    return directional_acc


if __name__ == '__main__':
    symbols = ['20MICRONS', 'RELIANCE', 'INFY']

    results = {}
    for sym in symbols:
        try:
            acc = evaluate(test_symbol=sym)
            results[sym] = acc
        except Exception as e:
            print(f"Skipped {sym}: {e}")

    if results:
        print(f"\n{'='*50}")
        print(f"OVERALL AVERAGE: {np.mean(list(results.values())):.2f}%")
        print(f"{'='*50}")
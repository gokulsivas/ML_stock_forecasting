import torch
import numpy as np
import sys
sys.path.append('..')

from models.hybrid_lstm_gru import HybridLSTMGRU
from data.data_loader import StockDataLoader
from sklearn.preprocessing import StandardScaler
from datetime import datetime, timedelta
import joblib
import pandas as pd


class PredictionService:
    """Handle model inference for API"""

    def __init__(self, model_path='saved_models/returns_model.pth'):
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"Loading model on device: {self.device}")

        self.feature_cols = joblib.load('saved_models/returns_feature_cols.pkl')

        input_size = len(self.feature_cols)
        self.model = HybridLSTMGRU(
            input_size  = input_size,
            hidden_size = 512,    # ✅ CHANGED: was 256
            num_layers  = 4,      # ✅ CHANGED: was 3
            dropout     = 0.3
        )

        # ✅ CHANGED: handles both full checkpoint dict and bare state_dict
        checkpoint = torch.load(model_path, map_location=self.device, weights_only=False)
        if isinstance(checkpoint, dict) and 'model_state_dict' in checkpoint:
            self.model.load_state_dict(checkpoint['model_state_dict'])
        else:
            self.model.load_state_dict(checkpoint)

        self.model.to(self.device)
        self.model.eval()

        self.data_loader = StockDataLoader()
        print("Model loaded successfully")   # ✅ removed Unicode checkmark (Windows encoding safe)

    def predict(self, symbol, days_ahead=5):
        """Generate predictions for N days ahead"""

        if days_ahead < 1 or days_ahead > 365:
            return None

        try:
            df = self.data_loader.load_stock_data(symbol)
            if df is None or len(df) < 90:   # ✅ CHANGED: was 60
                return None

            df['target_return'] = df['close_price'].pct_change().shift(-1)
            df = df.dropna()

            if len(df) < 90:                  # ✅ CHANGED: was 60
                return None

            recent_df = df.tail(90).copy()    # ✅ CHANGED: was 60

            scaler = StandardScaler()
            scaled_features = scaler.fit_transform(recent_df[self.feature_cols].values)

            current_price = float(df['close_price'].iloc[-1])
            last_date     = df['trade_date'].iloc[-1]

            predictions = []
            last_price  = current_price
            sequence    = scaled_features[-90:].copy()   # ✅ CHANGED: was 60

            with torch.no_grad():
                for i in range(days_ahead):
                    X_tensor   = torch.FloatTensor(sequence).unsqueeze(0).to(self.device)
                    pred_return = self.model(X_tensor).cpu().numpy()[0, 0]

                    predicted_price = last_price * (1 + pred_return)

                    pred_date = last_date + timedelta(days=i + 1)
                    while pred_date.weekday() >= 5:
                        pred_date += timedelta(days=1)

                    predictions.append({
                        'date':              pred_date.strftime('%Y-%m-%d'),
                        'predicted_price':   round(float(predicted_price), 2),
                        'predicted_return':  round(float(pred_return * 100), 2)
                    })

                    new_row    = sequence[-1].copy()
                    new_row[0] = pred_return

                    for lag in range(min(5, len(new_row) - 1), 0, -1):
                        new_row[lag] = new_row[lag - 1]

                    sequence   = np.vstack([sequence[1:], new_row.reshape(1, -1)])
                    last_price = predicted_price

            return {
                'symbol':        symbol,
                'current_price': round(current_price, 2),
                'current_date':  last_date.strftime('%Y-%m-%d'),
                'predictions':   predictions
            }

        except Exception as e:
            print(f"Prediction error for {symbol}: {e}")
            import traceback
            traceback.print_exc()
            return None


_predictor = None


def get_predictor():
    global _predictor
    if _predictor is None:
        _predictor = PredictionService()
    return _predictor
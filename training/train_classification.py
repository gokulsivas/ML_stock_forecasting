import os
import sys
import torch
import torch.nn as nn
import numpy as np
import joblib
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from data.data_loader import StockDataLoader
from data.feature_engineering import FeatureEngineer
from models.hybrid_lstm_gru import HybridLSTMGRU, count_parameters
from training.dataset import StockSequenceDataset
from sklearn.preprocessing import StandardScaler


SEQUENCE_LENGTH = 90
HIDDEN_SIZE     = 256
NUM_LAYERS      = 3
DROPOUT         = 0.3
BATCH_SIZE      = 128
LEARNING_RATE   = 0.0005
EPOCHS          = 150
PATIENCE        = 20
TRAIN_SPLIT     = 0.70
VAL_SPLIT       = 0.15
NUM_STOCKS      = 300
BEST_PATH       = 'saved_models/classification_model.pth'
LATEST_PATH     = 'saved_models/classification_latest.pth'
FEAT_COLS_PATH  = 'saved_models/classification_feature_cols.pkl'


def make_sequences(df, feature_cols, sequence_length=SEQUENCE_LENGTH):
    engineer = FeatureEngineer()
    df = engineer.add_technical_indicators(df)

    df['target'] = (df['close_price'].pct_change().shift(-1) > 0).astype(float)
    df = df.dropna()

    cols = [c for c in feature_cols if c in df.columns]

    scaler = StandardScaler()
    X = scaler.fit_transform(df[cols].values)
    y = df['target'].values

    X_seq, y_seq = [], []
    for i in range(len(X) - sequence_length):
        X_seq.append(X[i:i + sequence_length])
        y_seq.append(y[i + sequence_length])

    return np.array(X_seq, dtype=np.float32), np.array(y_seq, dtype=np.float32)


def load_all_data(num_stocks=NUM_STOCKS):
    loader   = StockDataLoader()
    engineer = FeatureEngineer()
    symbols  = loader.get_stocks_with_min_history(min_days=1500)[:num_stocks]
    print(f"Loading {len(symbols)} stocks...")

    feature_cols = None
    for sym in symbols:
        df = loader.load_stock_data(sym)
        if df is not None and len(df) > 1500:
            df_feat = engineer.add_technical_indicators(df)
            feature_cols = [c for c in df_feat.columns
                            if c not in ['symbol', 'trade_date', 'close_price',
                                         'open_price', 'high_price', 'low_price',
                                         'volume', 'target']]
            break

    joblib.dump(feature_cols, FEAT_COLS_PATH)
    print(f"Feature columns: {len(feature_cols)}")

    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []

    for sym in symbols:
        try:
            df = loader.load_stock_data(sym)
            if df is None or len(df) < 1500:
                continue

            X_seq, y_seq = make_sequences(df, feature_cols)
            if len(X_seq) < 100:
                continue

            train_size = int(len(X_seq) * TRAIN_SPLIT)
            val_size   = int(len(X_seq) * VAL_SPLIT)

            all_X_train.append(X_seq[:train_size])
            all_y_train.append(y_seq[:train_size])
            all_X_val.append(X_seq[train_size:train_size + val_size])
            all_y_val.append(y_seq[train_size:train_size + val_size])
            print(f"  ✓ {sym}: {train_size} train, {val_size} val")

        except Exception as e:
            print(f"  ✗ {sym}: {e}")

    X_train = np.concatenate(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_val   = np.concatenate(all_X_val)
    y_val   = np.concatenate(all_y_val)

    print(f"\nTotal train: {len(X_train)}, val: {len(X_val)}")
    up_pct = y_train.mean() * 100
    print(f"Class balance — UP: {up_pct:.1f}%  DOWN: {100-up_pct:.1f}%")

    return X_train, y_train, X_val, y_val, feature_cols


def train():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}\n")

    X_train, y_train, X_val, y_val, feature_cols = load_all_data()

    train_loader = DataLoader(
        StockSequenceDataset(X_train, y_train),
        batch_size=BATCH_SIZE, shuffle=True, num_workers=4, pin_memory=True
    )
    val_loader = DataLoader(
        StockSequenceDataset(X_val, y_val),
        batch_size=BATCH_SIZE, num_workers=4, pin_memory=True
    )

    model = HybridLSTMGRU(
        input_size=X_train.shape[2],
        hidden_size=HIDDEN_SIZE,
        num_layers=NUM_LAYERS,
        dropout=DROPOUT
    ).to(device)
    print(f"Parameters: {count_parameters(model):,}\n")

    up_ratio   = y_train.mean()
    pos_weight = torch.tensor([(1 - up_ratio) / (up_ratio + 1e-9)]).to(device)
    criterion  = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

    optimizer  = torch.optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=1e-4)
    scheduler  = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=8   # ✅ CHANGED: mode min → max
    )
    scaler_amp = GradScaler('cuda')

    best_val_acc     = 0.0    # ✅ CHANGED: was best_val_loss = float('inf')
    patience_counter = 0

    for epoch in range(1, EPOCHS + 1):
        # ── Train ──────────────────────────────────────────
        model.train()
        train_loss    = 0
        train_correct = 0
        train_total   = 0

        for X_batch, y_batch in tqdm(train_loader, desc=f"Epoch {epoch}/{EPOCHS}"):
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device).unsqueeze(1)

            optimizer.zero_grad()
            with autocast('cuda'):
                logits = model(X_batch)
                loss   = criterion(logits, y_batch)

            scaler_amp.scale(loss).backward()
            scaler_amp.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler_amp.step(optimizer)
            scaler_amp.update()

            train_loss    += loss.item()
            preds          = (torch.sigmoid(logits) > 0.5).float()
            train_correct += (preds == y_batch).sum().item()
            train_total   += y_batch.size(0)

        # ── Validate ────────────────────────────────────────
        model.eval()
        val_loss    = 0
        val_correct = 0
        val_total   = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(device)
                y_batch = y_batch.to(device).unsqueeze(1)
                with autocast('cuda'):
                    logits = model(X_batch)
                    loss   = criterion(logits, y_batch)
                val_loss    += loss.item()
                preds        = (torch.sigmoid(logits) > 0.5).float()
                val_correct += (preds == y_batch).sum().item()
                val_total   += y_batch.size(0)

        avg_train_loss = train_loss / len(train_loader)
        avg_val_loss   = val_loss   / len(val_loader)
        train_acc      = train_correct / train_total * 100
        val_acc        = val_correct   / val_total   * 100

        scheduler.step(val_acc)    # ✅ CHANGED: was scheduler.step(avg_val_loss)

        print(f"\nEpoch {epoch}/{EPOCHS}")
        print(f"  Train — Loss: {avg_train_loss:.4f}  Acc: {train_acc:.2f}%")
        print(f"  Val   — Loss: {avg_val_loss:.4f}  Acc: {val_acc:.2f}%")

        # Save latest every epoch
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'val_acc': val_acc,
            'config': {
                'input_size': X_train.shape[2],
                'hidden_size': HIDDEN_SIZE,
                'num_layers': NUM_LAYERS,
                'dropout': DROPOUT,
                'model_type': 'classification'
            }
        }, LATEST_PATH)

        if val_acc > best_val_acc:          # ✅ CHANGED: was avg_val_loss < best_val_loss
            best_val_acc     = val_acc      # ✅ CHANGED: was best_val_loss = avg_val_loss
            patience_counter = 0
            torch.save({
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'val_acc': val_acc,
                'config': {
                    'input_size': X_train.shape[2],
                    'hidden_size': HIDDEN_SIZE,
                    'num_layers': NUM_LAYERS,
                    'dropout': DROPOUT,
                    'model_type': 'classification'
                }
            }, BEST_PATH)
            print(f"  ✅ Best model saved! Val Acc: {val_acc:.2f}%")
        else:
            patience_counter += 1
            print(f"  No improvement ({patience_counter}/{PATIENCE})")
            if patience_counter >= PATIENCE:
                print(f"Early stopping at epoch {epoch}")
                break

        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("\nTraining complete!")
    print(f"Best model saved: {BEST_PATH}")


if __name__ == '__main__':
    train()
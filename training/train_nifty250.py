import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
import joblib
from tqdm import tqdm
sys.path.append('.')

from data.data_loader import StockDataLoader
from models.hybrid_lstm_gru import HybridLSTMGRU, count_parameters
from training.dataset import StockSequenceDataset
from sklearn.preprocessing import StandardScaler

# ── GPU Optimisation flags ─────────────────────────────
torch.backends.cudnn.benchmark = True       # auto-tune CUDA kernels
torch.backends.cuda.matmul.allow_tf32 = True # TF32 for matmul (Ampere+)
torch.backends.cudnn.allow_tf32 = True

# ── Config — tuned for large GPU (RTX 3080/3090/4090/A100) ──
CONFIG = dict(
    batch_size            = 512,   # was 128 on RTX 3060
    sequence_length       = 90,
    hidden_size           = 512,   # was 256
    num_layers            = 4,     # was 3
    dropout               = 0.3,
    learning_rate         = 0.0005,
    epochs                = 200,   # more room to converge
    early_stopping_patience = 30,
    train_split           = 0.7,
    val_split             = 0.15,
    num_workers           = 8,     # was 4
    grad_clip             = 1.0,   # gradient clipping
)

TARGET_STOCKS = 800  # was 400

PRIORITY_SYMBOLS = [
    # NIFTY 50
    'RELIANCE','TCS','HDFCBANK','INFY','ICICIBANK','HINDUNILVR','SBIN',
    'BHARTIARTL','KOTAKBANK','ITC','LT','AXISBANK','ASIANPAINT','MARUTI',
    'SUNPHARMA','TITAN','BAJFINANCE','NESTLEIND','WIPRO','HCLTECH','TECHM',
    'ULTRACEMCO','POWERGRID','NTPC','ONGC','COALINDIA','DIVISLAB','DRREDDY',
    'CIPLA','EICHERMOT','BAJAJFINSV','BPCL','HEROMOTOCO','HINDALCO','GRASIM',
    'INDUSINDBK','JSWSTEEL','M&M','TATAMOTORS','TATASTEEL','ADANIPORTS',
    'APOLLOHOSP','BRITANNIA','SBILIFE','HDFCLIFE','TATACONSUM','PIDILITIND',
    'BAJAJ-AUTO','LTIM','ADANIENT',
    # NIFTY Next 50
    'VEDL','SIEMENS','HAVELLS','DABUR','MARICO','AMBUJACEM','GODREJCP',
    'BERGEPAINT','MUTHOOTFIN','LICI','BANKBARODA','CANBK','PNB','UNIONBANK',
    'FEDERALBNK','CHOLAFIN','BAJAJHLDNG','PGHH','COLPAL','ICICIPRULI',
    'SBICARD','HDFCAMC','NAUKRI','DMART','TRENT','ZOMATO','IRCTC','IRFC',
    'HUDCO','RVNL','HAL','BEL','BHEL','SAIL','NMDC','GAIL','IOC',
    'HINDPETRO','PETRONET','CONCOR','ADANIGREEN','ADANIPOWER','ATGL',
    # NIFTY Midcap top picks
    'VOLTAS','MPHASIS','PERSISTENT','COFORGE','LTTS','TATAELXSI','KPITTECH',
    'CYIENT','CUMMINSIND','THERMAX','ABB','ASTRAL','POLYCAB','MOTHERSON',
    'BOSCHLTD','TVSMOTOR','AUROPHARMA','TORNTPHARM','ALKEM','IPCALAB',
    'LALPATHLAB','RBLBANK','IDFCFIRSTB','BANDHANBNK','MANAPPURAM',
    'LICHSGFIN','COROMANDEL','CGPOWER','GODREJPROP','OBEROIRLTY',
    'PHOENIXLTD','PRESTIGE','PAGEIND','ZYDUSLIFE','GLENMARK','AJANTPHARM',
]


# ── Loss ──────────────────────────────────────────────
class DirectionalLoss(nn.Module):
    def __init__(self, mse_weight=0.5, dir_weight=0.5):  # balanced for better GPU
        super().__init__()
        self.mse_weight = mse_weight
        self.dir_weight = dir_weight

    def forward(self, predictions, targets):
        mse_loss = F.mse_loss(predictions, targets)
        pred_dir = torch.sign(predictions)
        true_dir = torch.sign(targets)
        dir_loss = torch.mean((pred_dir != true_dir).float())
        return self.mse_weight * mse_loss + self.dir_weight * dir_loss


# ── Trainer ───────────────────────────────────────────
class StockTrainer:
    def __init__(self, model, device, config):
        self.model   = model.to(device)
        self.device  = device
        self.config  = config

        # bfloat16 on Ampere+, float16 on older GPUs
        gpu_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ''
        self.amp_dtype = torch.bfloat16 if any(x in gpu_name for x in ['a100','a10','3080','3090','4090','4080','4070','h100','h200']) else torch.float16
        print(f"AMP dtype: {self.amp_dtype}")

        self.scaler    = GradScaler('cuda')
        self.criterion = DirectionalLoss(mse_weight=0.5, dir_weight=0.5)
        self.optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-4
        )
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )
        self.best_val_loss   = float('inf')
        self.patience_counter = 0
        self.start_epoch     = 1

    def save_checkpoint(self, epoch, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save({
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss':        self.best_val_loss,
            'patience_counter':     self.patience_counter,
        }, save_path)

    def load_checkpoint(self, save_path):
        if os.path.exists(save_path):
            checkpoint = torch.load(save_path, map_location=self.device)
            # Only resume if architecture matches
            try:
                self.model.load_state_dict(checkpoint['model_state_dict'])
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                if 'scheduler_state_dict' in checkpoint:
                    self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
                self.best_val_loss    = checkpoint.get('best_val_loss', float('inf'))
                self.patience_counter = checkpoint.get('patience_counter', 0)
                self.start_epoch      = checkpoint.get('epoch', 0) + 1
                print(f"Resuming from epoch {self.start_epoch} | Best val loss: {self.best_val_loss:.6f}")
            except Exception:
                print("Checkpoint architecture mismatch — starting fresh.")
        else:
            print("No checkpoint found. Starting from epoch 1.")

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        for X_batch, y_batch in tqdm(train_loader, desc='Training', leave=False):
            X_batch = X_batch.to(self.device, non_blocking=True)
            y_batch = y_batch.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)  # faster than zero_grad()
            with autocast('cuda', dtype=self.amp_dtype):
                outputs = self.model(X_batch)
                loss    = self.criterion(outputs, y_batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total_loss += loss.item()
        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device, non_blocking=True)
                y_batch = y_batch.to(self.device, non_blocking=True)
                with autocast('cuda', dtype=self.amp_dtype):
                    outputs = self.model(X_batch)
                    loss    = self.criterion(outputs, y_batch)
                total_loss += loss.item()
        return total_loss / len(val_loader)

    def train(self, train_loader, val_loader, epochs, save_path='saved_models/returns_model.pth'):
        self.load_checkpoint(save_path)
        for epoch in range(self.start_epoch, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.validate(val_loader)
            self.scheduler.step(epoch)

            lr = self.optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}", end='')
            if torch.cuda.is_available():
                print(f" | GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB", end='')
            print()

            if val_loss < self.best_val_loss:
                self.best_val_loss    = val_loss
                self.patience_counter = 0
                self.save_checkpoint(epoch, save_path)
                print(f"  Model saved! Best val loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
                if self.patience_counter >= self.config['early_stopping_patience']:
                    print(f"Early stopping at epoch {epoch}")
                    break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


# ── Sequence builder ──────────────────────────────────
def create_return_sequences(df, seq_len=90):
    df = df.copy()
    df['target_return'] = df['close_price'].pct_change().shift(-1)
    feature_cols = [c for c in df.columns if c not in
                    ['symbol','trade_date','close_price','open_price',
                     'high_price','low_price','volume','target_return']]
    df = df.dropna()
    if len(df) < seq_len + 100:
        return None, None, None, None
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    y = df['target_return'].values
    X_seq = np.array([X[i:i+seq_len] for i in range(len(X)-seq_len)])
    y_seq = np.array([y[i+seq_len]   for i in range(len(X)-seq_len)])
    return X_seq, y_seq.reshape(-1,1), scaler, feature_cols


# ── Main ──────────────────────────────────────────────
def train_returns_model():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    loader     = StockDataLoader()
    all_symbols = loader.get_stocks_with_min_history(min_days=1500)
    priority   = [s for s in PRIORITY_SYMBOLS if s in all_symbols]
    others     = [s for s in all_symbols if s not in priority]
    selected   = priority + others[:max(0, TARGET_STOCKS - len(priority))]

    print(f"\nPriority: {len(priority)} | Others: {len(selected)-len(priority)} | Total: {len(selected)}")
    print('='*60)

    all_X_train, all_y_train = [], []
    all_X_val,   all_y_val   = [], []
    successful, failed       = [], []
    feature_cols             = None

    for i, symbol in enumerate(selected):
        print(f"  {i+1}/{len(selected)} {symbol}...", end=' ')
        try:
            df = loader.load_stock_data(symbol)
            if df is None or len(df) < 1500:
                print("SKIP")
                failed.append(symbol)
                continue
            X_seq, y_seq, _, fc = create_return_sequences(df, CONFIG['sequence_length'])
            if X_seq is None or len(X_seq) < 200:
                print("SKIP (few sequences)")
                failed.append(symbol)
                continue
            if feature_cols is None:
                feature_cols = fc
            train_size = int(len(X_seq) * CONFIG['train_split'])
            val_size   = int(len(X_seq) * CONFIG['val_split'])
            all_X_train.append(X_seq[:train_size])
            all_y_train.append(y_seq[:train_size])
            all_X_val.append(X_seq[train_size:train_size+val_size])
            all_y_val.append(y_seq[train_size:train_size+val_size])
            successful.append(symbol)
            print(f"OK ({train_size} train seqs)")
        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(symbol)

    print('='*60)
    print(f"Processed: {len(successful)} | Failed: {len(failed)}")

    if not all_X_train:
        print("ERROR: No valid data!")
        return

    X_train = np.concatenate(all_X_train)
    y_train = np.concatenate(all_y_train)
    X_val   = np.concatenate(all_X_val)
    y_val   = np.concatenate(all_y_val)

    print(f"Total train: {len(X_train):,} | Val: {len(X_val):,} | Features: {X_train.shape[2]}")

    joblib.dump(feature_cols, 'saved_models/returns_feature_cols.pkl')
    joblib.dump(successful,   'saved_models/trained_stocks.pkl')

    train_dataset = StockSequenceDataset(X_train, y_train)
    val_dataset   = StockSequenceDataset(X_val,   y_val)
    train_loader  = DataLoader(train_dataset, batch_size=CONFIG['batch_size'],
                               shuffle=True,  num_workers=CONFIG['num_workers'],
                               pin_memory=True, persistent_workers=True)
    val_loader    = DataLoader(val_dataset,   batch_size=CONFIG['batch_size'],
                               num_workers=CONFIG['num_workers'],
                               pin_memory=True, persistent_workers=True)

    model = HybridLSTMGRU(
        input_size  = X_train.shape[2],
        hidden_size = CONFIG['hidden_size'],
        num_layers  = CONFIG['num_layers'],
        dropout     = CONFIG['dropout']
    )

    # torch.compile — massive speedup on PyTorch 2.0+ with Ampere GPUs
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    print(f"Model parameters: {count_parameters(model):,}")
    trainer = StockTrainer(model, device, CONFIG)
    trainer.train(train_loader, val_loader, CONFIG['epochs'],
                  save_path='saved_models/returns_model.pth')

    print('='*60)
    print(f"DONE! Model saved: saved_models/returns_model.pth")
    print(f"Trained on {len(successful)} stocks")
    print('='*60)


if __name__ == '__main__':
    train_returns_model()
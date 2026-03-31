import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torch.amp import autocast, GradScaler
import numpy as np
from tqdm import tqdm

sys.path.append('..')

from data.data_loader import StockDataLoader
from data.preprocessing import TimeSeriesPreprocessor
from models.hybrid_lstm_gru import HybridLSTMGRU, count_parameters
from training.dataset import StockSequenceDataset

# ── GPU optimisation flags ─────────────────────────────────────────────────
torch.backends.cudnn.benchmark = True
torch.backends.cuda.matmul.allow_tf32 = True
torch.backends.cudnn.allow_tf32 = True


class DirectionalLoss(nn.Module):
    def __init__(self, alpha=0.5):   # ✅ CHANGED: 0.7 → 0.5 (balanced for large GPU)
        super().__init__()
        self.alpha = alpha

    def forward(self, predictions, targets):
        mse_loss         = F.mse_loss(predictions, targets)
        pred_dir         = torch.sign(predictions)
        true_dir         = torch.sign(targets)
        direction_penalty = torch.mean((pred_dir != true_dir).float())
        return self.alpha * mse_loss + (1 - self.alpha) * direction_penalty


class StockTrainer:
    """Training pipeline — updated for advanced GPU"""

    def __init__(self, model, device, config):
        self.model  = model.to(device)
        self.device = device
        self.config = config

        # ✅ Auto-detect bfloat16 support (Ampere+ GPUs)
        gpu_name = torch.cuda.get_device_name(0).lower() if torch.cuda.is_available() else ''
        self.amp_dtype = torch.bfloat16 if any(x in gpu_name for x in [
            'a100', 'a10', '3080', '3090', '4090', '4080', '4070', 'h100', 'h200'
        ]) else torch.float16
        print(f"AMP dtype: {self.amp_dtype}")

        self.scaler    = GradScaler('cuda')
        self.criterion = DirectionalLoss(alpha=0.5)

        # ✅ CHANGED: Adam → AdamW with weight decay
        self.optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=config['learning_rate'],
            weight_decay=1e-4
        )

        # ✅ CHANGED: ReduceLROnPlateau → CosineAnnealingWarmRestarts
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            self.optimizer, T_0=50, T_mult=2, eta_min=1e-6
        )

        self.best_val_loss    = float('inf')
        self.patience_counter = 0
        self.start_epoch      = 1

    # ── Checkpoint helpers ─────────────────────────────────────────────────

    def _checkpoint_dict(self, epoch):
        return {
            'epoch':                epoch,
            'model_state_dict':     self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss':        self.best_val_loss,
            'patience_counter':     self.patience_counter,
            'config':               self.config,
        }

    def save_best_checkpoint(self, epoch, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self._checkpoint_dict(epoch), save_path)

    def save_latest_checkpoint(self, epoch, save_path):
        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        torch.save(self._checkpoint_dict(epoch), save_path)

    def _same_shape(self, state_a, state_b):
        if state_a.keys() != state_b.keys():
            return False
        for k in state_a:
            if state_a[k].shape != state_b[k].shape:
                return False
        return True

    def load_checkpoint(self, latest_path):
        if not os.path.exists(latest_path):
            print("No checkpoint found. Starting from epoch 1.")
            return

        checkpoint       = torch.load(latest_path, map_location=self.device)
        ckpt_model_state = checkpoint.get('model_state_dict', None)

        if ckpt_model_state is None or not self._same_shape(self.model.state_dict(), ckpt_model_state):
            print("Checkpoint shape mismatch — starting fresh.")
            return

        self.model.load_state_dict(ckpt_model_state)

        try:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        except Exception:
            print("Warning: optimizer state not loaded.")

        try:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        except Exception:
            print("Warning: scheduler state not loaded.")

        self.best_val_loss    = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.start_epoch      = checkpoint.get('epoch', 0) + 1

        print(f"Resuming from epoch {self.start_epoch}")
        print(f"Best val loss so far: {self.best_val_loss:.6f}")

    # ── Training loop ──────────────────────────────────────────────────────

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in tqdm(train_loader, desc="Training", leave=False):
            X_batch = X_batch.to(self.device, non_blocking=True)  # ✅ non_blocking
            y_batch = y_batch.to(self.device, non_blocking=True)

            self.optimizer.zero_grad(set_to_none=True)  # ✅ faster than zero_grad()

            with autocast('cuda', dtype=self.amp_dtype):
                outputs = self.model(X_batch)
                loss    = self.criterion(outputs, y_batch)

            self.scaler.scale(loss).backward()

            # ✅ gradient clipping
            self.scaler.unscale_(self.optimizer)
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.get('grad_clip', 1.0))

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

    def train(
        self,
        train_loader,
        val_loader,
        epochs,
        best_save_path   = 'saved_models/returns_model.pth',
        latest_save_path = 'saved_models/latest_checkpoint.pth'
    ):
        self.load_checkpoint(latest_save_path)

        for epoch in range(self.start_epoch, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss   = self.validate(val_loader)

            # ✅ CosineAnnealingWarmRestarts takes epoch, not val_loss
            self.scheduler.step(epoch)
            lr = self.optimizer.param_groups[0]['lr']

            print(f"Epoch {epoch}/{epochs} | Train: {train_loss:.6f} | Val: {val_loss:.6f} | LR: {lr:.2e}", end='')
            if torch.cuda.is_available():
                print(f" | GPU: {torch.cuda.memory_allocated()/1e9:.2f}GB", end='')
            print()

            self.save_latest_checkpoint(epoch, latest_save_path)

            if val_loss < self.best_val_loss:
                self.best_val_loss    = val_loss
                self.patience_counter = 0
                self.save_best_checkpoint(epoch, best_save_path)
                print(f"  Model saved! Best val loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
                print(f"  No improvement ({self.patience_counter}/{self.config['early_stopping_patience']})")

            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping at epoch {epoch}")
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    # ── Config updated for advanced GPU ──────────────────────────────────
    config = {
        'batch_size':              512,   # ✅ was 128
        'sequence_length':         90,
        'hidden_size':             512,   # ✅ was 256
        'num_layers':              4,     # ✅ was 3
        'dropout':                 0.3,
        'learning_rate':           0.0005,
        'epochs':                  200,   # ✅ was 150
        'early_stopping_patience': 30,    # ✅ was 25
        'train_split':             0.7,
        'val_split':               0.15,
        'num_workers':             8,     # ✅ was 4
        'grad_clip':               1.0,   # ✅ new
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"VRAM: {torch.cuda.get_device_properties(0).total_memory/1e9:.1f} GB")

    loader  = StockDataLoader()
    symbols = loader.get_stocks_with_min_history(min_days=1500)
    print(f"Found {len(symbols)} stocks with sufficient history")

    test_symbol = symbols[0]
    print(f"Training on: {test_symbol}")

    df           = loader.load_stock_data(test_symbol)
    preprocessor = TimeSeriesPreprocessor(sequence_length=config['sequence_length'])
    preprocessor.fit_scalers(df)
    X, y         = preprocessor.transform(df)
    X_seq, y_seq = preprocessor.create_sequences(X, y)

    train_size = int(len(X_seq) * config['train_split'])
    val_size   = int(len(X_seq) * config['val_split'])

    train_dataset = StockSequenceDataset(X_seq[:train_size], y_seq[:train_size])
    val_dataset   = StockSequenceDataset(
        X_seq[train_size:train_size + val_size],
        y_seq[train_size:train_size + val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True   # ✅ new
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=config['num_workers'],
        pin_memory=True,
        persistent_workers=True   # ✅ new
    )

    input_size = X_seq.shape[2]
    model = HybridLSTMGRU(
        input_size  = input_size,
        hidden_size = config['hidden_size'],
        num_layers  = config['num_layers'],
        dropout     = config['dropout']
    )

    # ✅ torch.compile — PyTorch 2.0+ kernel fusion speedup
    if hasattr(torch, 'compile'):
        print("Compiling model with torch.compile()...")
        model = torch.compile(model)

    print(f"Model parameters: {count_parameters(model):,}")

    trainer = StockTrainer(model, device, config)
    trainer.train(train_loader, val_loader, config['epochs'])
    preprocessor.save_scalers()
    print("Training completed!")


if __name__ == "__main__":
    main()
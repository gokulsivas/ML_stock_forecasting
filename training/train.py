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


class DirectionalLoss(nn.Module):
    def __init__(self, mse_weight=0.6, dir_weight=0.4):
        super().__init__()
        self.mse_weight = mse_weight
        self.dir_weight = dir_weight

    def forward(self, predictions, targets):
        mse_loss = F.mse_loss(predictions, targets)
        pred_dir = torch.sign(predictions)
        true_dir = torch.sign(targets)
        dir_loss = torch.mean((pred_dir != true_dir).float())
        return self.mse_weight * mse_loss + self.dir_weight * dir_loss


class StockTrainer:
    """Training pipeline optimized for RTX 3060 6GB"""

    def __init__(self, model, device, config):
        self.model = model.to(device)
        self.device = device
        self.config = config

        self.scaler = GradScaler('cuda')
        self.criterion = DirectionalLoss(mse_weight=0.6, dir_weight=0.4)
        self.optimizer = torch.optim.Adam(
            self.model.parameters(),
            lr=config['learning_rate']
        )
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=8
        )

        self.best_val_loss = float('inf')
        self.patience_counter = 0
        self.start_epoch = 1

    # ── checkpoint helpers ────────────────────────────────────────────────

    def _checkpoint_dict(self, epoch):
        return {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'best_val_loss': self.best_val_loss,
            'patience_counter': self.patience_counter,
            'config': self.config,
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

        checkpoint = torch.load(latest_path, map_location=self.device)
        ckpt_model_state = checkpoint.get('model_state_dict', None)

        if ckpt_model_state is None or not self._same_shape(self.model.state_dict(), ckpt_model_state):
            print("Checkpoint found but model shape does not match. Starting from epoch 1.")
            return

        self.model.load_state_dict(ckpt_model_state)

        if 'optimizer_state_dict' in checkpoint:
            try:
                self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            except Exception:
                print("Warning: optimizer state could not be loaded. Continuing without it.")

        if 'scheduler_state_dict' in checkpoint:
            try:
                self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
            except Exception:
                print("Warning: scheduler state could not be loaded. Continuing without it.")

        self.best_val_loss = checkpoint.get('best_val_loss', float('inf'))
        self.patience_counter = checkpoint.get('patience_counter', 0)
        self.start_epoch = checkpoint.get('epoch', 0) + 1

        print(f"Resuming from epoch {self.start_epoch}")
        print(f"Best val loss so far: {self.best_val_loss:.6f}")

    # ── training loop ─────────────────────────────────────────────────────

    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0

        for X_batch, y_batch in tqdm(train_loader, desc="Training"):
            X_batch = X_batch.to(self.device)
            y_batch = y_batch.to(self.device)

            self.optimizer.zero_grad()

            with autocast('cuda'):
                outputs = self.model(X_batch)
                loss = self.criterion(outputs, y_batch)

            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()

            total_loss += loss.item()

        return total_loss / len(train_loader)

    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0

        with torch.no_grad():
            for X_batch, y_batch in val_loader:
                X_batch = X_batch.to(self.device)
                y_batch = y_batch.to(self.device)

                with autocast('cuda'):
                    outputs = self.model(X_batch)
                    loss = self.criterion(outputs, y_batch)

                total_loss += loss.item()

        return total_loss / len(val_loader)

    def train(
        self,
        train_loader,
        val_loader,
        epochs,
        best_save_path='saved_models/best_model.pth',
        latest_save_path='saved_models/latest_checkpoint.pth'
    ):
        self.load_checkpoint(latest_save_path)

        for epoch in range(self.start_epoch, epochs + 1):
            train_loss = self.train_epoch(train_loader)
            val_loss = self.validate(val_loader)

            self.scheduler.step(val_loss)

            print(f"Epoch {epoch}/{epochs}")
            print(f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}")
            if torch.cuda.is_available():
                print(f"GPU Memory: {torch.cuda.memory_allocated()/1e9:.2f}GB")

            # Always save latest so we can resume from here
            self.save_latest_checkpoint(epoch, latest_save_path)

            if val_loss < self.best_val_loss:
                self.best_val_loss = val_loss
                self.patience_counter = 0
                self.save_best_checkpoint(epoch, best_save_path)
                print(f"Model saved! Best val loss: {val_loss:.6f}")
            else:
                self.patience_counter += 1
                print(f"No improvement ({self.patience_counter}/{self.config['early_stopping_patience']})")

            if self.patience_counter >= self.config['early_stopping_patience']:
                print(f"Early stopping triggered after {epoch} epochs")
                break

            if torch.cuda.is_available():
                torch.cuda.empty_cache()


def main():
    config = {
        'batch_size': 128,
        'sequence_length': 90,
        'hidden_size': 256,
        'num_layers': 3,
        'dropout': 0.3,
        'learning_rate': 0.0005,
        'epochs': 150,
        'early_stopping_patience': 25,
        'train_split': 0.7,
        'val_split': 0.15
    }

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    loader = StockDataLoader()
    symbols = loader.get_stocks_with_min_history(min_days=1500)
    print(f"Found {len(symbols)} stocks with sufficient history")

    test_symbol = symbols[0]
    print(f"Training on: {test_symbol}")

    df = loader.load_stock_data(test_symbol)
    preprocessor = TimeSeriesPreprocessor(sequence_length=config['sequence_length'])
    preprocessor.fit_scalers(df)
    X, y = preprocessor.transform(df)
    X_seq, y_seq = preprocessor.create_sequences(X, y)

    train_size = int(len(X_seq) * config['train_split'])
    val_size = int(len(X_seq) * config['val_split'])

    train_dataset = StockSequenceDataset(X_seq[:train_size], y_seq[:train_size])
    val_dataset = StockSequenceDataset(
        X_seq[train_size:train_size + val_size],
        y_seq[train_size:train_size + val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=config['batch_size'],
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=config['batch_size'],
        num_workers=4,
        pin_memory=True
    )

    input_size = X_seq.shape[2]
    model = HybridLSTMGRU(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    print(f"Model parameters: {count_parameters(model):,}")

    trainer = StockTrainer(model, device, config)
    trainer.train(train_loader, val_loader, config['epochs'])
    preprocessor.save_scalers()
    print("Training completed!")


if __name__ == "__main__":
    main()

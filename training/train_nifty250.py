import os                          # ← ADDED
import torch
import sys
sys.path.append('..')
from data.data_loader import StockDataLoader
from models.hybrid_lstm_gru import HybridLSTMGRU, count_parameters
from training.dataset import StockSequenceDataset
from training.train import StockTrainer
from torch.utils.data import DataLoader
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
import joblib


# NIFTY 50 first, then fill remaining slots from DB
PRIORITY_SYMBOLS = [
    # NIFTY 50
    'RELIANCE', 'TCS', 'HDFCBANK', 'INFY', 'ICICIBANK',
    'HINDUNILVR', 'SBIN', 'BHARTIARTL', 'KOTAKBANK', 'ITC',
    'LT', 'AXISBANK', 'ASIANPAINT', 'MARUTI', 'SUNPHARMA',
    'TITAN', 'BAJFINANCE', 'NESTLEIND', 'WIPRO', 'HCLTECH',
    'TECHM', 'ULTRACEMCO', 'POWERGRID', 'NTPC', 'ONGC',
    'COALINDIA', 'DIVISLAB', 'DRREDDY', 'CIPLA', 'EICHERMOT',
    'BAJAJFINSV', 'BPCL', 'HEROMOTOCO', 'HINDALCO', 'GRASIM',
    'INDUSINDBK', 'JSWSTEEL', 'M&M', 'TATAMOTORS', 'TATASTEEL',
    'ADANIPORTS', 'APOLLOHOSP', 'BRITANNIA', 'SBILIFE', 'HDFCLIFE',
    'TATACONSUM', 'PIDILITIND', 'BAJAJ-AUTO', 'LTIM', 'ADANIENT',

    # NIFTY Next 50
    'VEDL', 'SIEMENS', 'HAVELLS', 'DABUR', 'MARICO',
    'AMBUJACEM', 'GODREJCP', 'BERGEPAINT', 'MUTHOOTFIN', 'LICI',
    'BANKBARODA', 'CANBK', 'PNB', 'UNIONBANK', 'FEDERALBNK',
    'CHOLAFIN', 'BAJAJHLDNG', 'PGHH', 'COLPAL', 'ICICIPRULI',
    'SBICARD', 'HDFCAMC', 'NAUKRI', 'DMART', 'TRENT',
    'ZOMATO', 'PAYTM', 'NYKAA', 'POLICYBZR', 'IRCTC',
    'IRFC', 'HUDCO', 'RVNL', 'RAILTEL', 'COCHINSHIP',
    'HAL', 'BEL', 'BHEL', 'SAIL', 'NMDC',
    'GAIL', 'IOC', 'HINDPETRO', 'PETRONET', 'MGL',
    'CONCOR', 'ADANIGREEN', 'ADANITRANS', 'ADANIPOWER', 'ATGL',

    # NIFTY Midcap 150 (top picks)
    'VOLTAS', 'MPHASIS', 'PERSISTENT', 'COFORGE', 'LTTS',
    'TATAELXSI', 'KPITTECH', 'CYIENT', 'SONACOMS', 'SCHAEFFLER',
    'TIINDIA', 'CUMMINSIND', 'THERMAX', 'ABB', 'AIAENG',
    'KAJARIACER', 'ASTRAL', 'POLYCAB', 'FINOLEX', 'HLEG',
    'SUNDRMFAST', 'MOTHERSON', 'BOSCHLTD', 'EXIDEIND', 'AMARAJABAT',
    'TVSMOTOR', 'BAJAJCON', 'ESCORTS', 'ASHOKLEY', 'FORCEMOT',
    'AUROPHARMA', 'TORNTPHARM', 'ALKEM', 'IPCALAB', 'NATCOPHARM',
    'LALPATHLAB', 'METROPOLIS', 'MAXHEALTH', 'FORTIS', 'ASTER',
    'INDIAMART', 'JUSTDIAL', 'ZENSARTECH', 'RBLBANK', 'IDFCFIRSTB',
    'BANDHANBNK', 'CUB', 'KARURVYSYA', 'DCBBANK', 'SOUTHBANK',
    'CHOLAHLDNG', 'MANAPPURAM', 'IIFL', 'MFSL', 'SUNDARAM',
    'LICHSGFIN', 'PNBHOUSING', 'CANFINHOME', 'APTUS', 'HOMEFIRST',
    'PIIND', 'UPL', 'RALLIS', 'BAYER', 'SUMICHEM',
    'COROMANDEL', 'CHAMBLFERT', 'GNFC', 'DEEPAKFERT', 'ATUL',
    'NAVINFLUOR', 'FLUOROCHEM', 'CLEAN', 'FINEORG', 'SUDARSCHEM',
    'CGPOWER', 'APLAPOLLO', 'RATNAMANI', 'WELCORP', 'JINDALSAW',
    'GODREJPROP', 'OBEROIRLTY', 'PHOENIXLTD', 'BRIGADE', 'PRESTIGE',
    'LUXIND', 'PAGEIND', 'VSTIND', 'RADICO', 'UNITDSPR',
    'ZYDUSLIFE', 'GLENMARK', 'JUBLPHARM', 'ERIS', 'AJANTPHARM',
]

TARGET_STOCKS = 400   # change to 500 if you want more


def create_return_sequences(df, sequence_length=90):
    df = df.copy()
    df['target_return'] = df['close_price'].pct_change().shift(-1)
    feature_cols = [col for col in df.columns if col not in
                    ['symbol', 'trade_date', 'close_price', 'open_price',
                     'high_price', 'low_price', 'volume', 'target_return']]
    df = df.dropna()
    if len(df) < sequence_length + 100:
        return None, None, None, None
    scaler = StandardScaler()
    X = scaler.fit_transform(df[feature_cols].values)
    y = df['target_return'].values.reshape(-1, 1)
    Xseq, yseq = [], []
    for i in range(len(X) - sequence_length):
        Xseq.append(X[i:i + sequence_length])
        yseq.append(y[i + sequence_length])
    return np.array(Xseq), np.array(yseq), scaler, feature_cols


def train_large_model():
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
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.1f} GB")

    loader = StockDataLoader()

    all_symbols = loader.get_stocks_with_min_history(min_days=1500)
    print(f"\nTotal symbols with 1500+ days history: {len(all_symbols)}")

    priority = [s for s in PRIORITY_SYMBOLS if s in all_symbols]
    others   = [s for s in all_symbols if s not in priority]
    selected = priority + others[:max(0, TARGET_STOCKS - len(priority))]

    print(f"NIFTY 50 (priority): {len(priority)}")
    print(f"Other stocks:        {len(selected) - len(priority)}")
    print(f"Total selected:      {len(selected)}")
    print("=" * 60)

    all_Xtrain, all_ytrain = [], []
    all_Xval,   all_yval   = [], []
    successful = []
    failed     = []

    for i, symbol in enumerate(selected):
        print(f"[{i+1}/{len(selected)}] {symbol}...", end=" ")
        try:
            df = loader.load_stock_data(symbol)
            if df is None or len(df) < 1500:
                print("SKIP (no data)")
                failed.append(symbol)
                continue

            Xseq, yseq, _, feature_cols = create_return_sequences(df, config['sequence_length'])
            if Xseq is None or len(Xseq) < 200:
                print("SKIP (too few sequences)")
                failed.append(symbol)
                continue

            train_size = int(len(Xseq) * config['train_split'])
            val_size   = int(len(Xseq) * config['val_split'])

            all_Xtrain.append(Xseq[:train_size])
            all_ytrain.append(yseq[:train_size])
            all_Xval.append(Xseq[train_size:train_size + val_size])
            all_yval.append(yseq[train_size:train_size + val_size])
            successful.append(symbol)
            print(f"OK ({train_size} train seqs)")

        except Exception as e:
            print(f"ERROR: {e}")
            failed.append(symbol)

    print("=" * 60)
    print(f"Successfully processed: {len(successful)} stocks")
    print(f"Failed/skipped:         {len(failed)} stocks")

    if not all_Xtrain:
        print("ERROR: No valid data!")
        return

    Xtrain = np.concatenate(all_Xtrain)
    ytrain = np.concatenate(all_ytrain)
    Xval   = np.concatenate(all_Xval)
    yval   = np.concatenate(all_yval)

    print(f"Total training samples: {len(Xtrain):,}")
    print(f"Total val samples:      {len(Xval):,}")
    print(f"Input features:         {Xtrain.shape[2]}")
    print("=" * 60)

    # ← ADDED: create folder before saving
    os.makedirs('saved_models', exist_ok=True)

    # Save metadata
    joblib.dump(feature_cols, 'saved_models/returns_feature_cols.pkl')
    joblib.dump(successful,   'saved_models/trained_stocks.pkl')

    train_dataset = StockSequenceDataset(Xtrain, ytrain)
    val_dataset   = StockSequenceDataset(Xval,   yval)
    train_loader = DataLoader(train_dataset, batch_size=config['batch_size'],
                              shuffle=True, num_workers=0, pin_memory=False)
    val_loader   = DataLoader(val_dataset,   batch_size=config['batch_size'],
                              num_workers=0, pin_memory=False)

    input_size = Xtrain.shape[2]
    model = HybridLSTMGRU(
        input_size=input_size,
        hidden_size=config['hidden_size'],
        num_layers=config['num_layers'],
        dropout=config['dropout']
    )
    print(f"Model parameters: {count_parameters(model):,}")

    trainer = StockTrainer(model, device, config)
    trainer.train(train_loader, val_loader, config['epochs'],
                  save_path='saved_models/returns_model.pth')

    print("=" * 60)
    print(f"DONE! Model saved: saved_models/returns_model.pth")
    print(f"Trained on {len(successful)} stocks")
    print("=" * 60)


if __name__ == "__main__":
    train_large_model()
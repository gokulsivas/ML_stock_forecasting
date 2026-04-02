import pandas as pd
import numpy as np
from ta.trend import SMAIndicator, EMAIndicator, MACD
from ta.momentum import RSIIndicator, StochasticOscillator, ROCIndicator, WilliamsRIndicator
from ta.volatility import BollingerBands, AverageTrueRange
from ta.volume import OnBalanceVolumeIndicator


class FeatureEngineer:
    """Compute technical indicators for stock data — 58 features"""

    def __init__(self):
        self.feature_columns = []

    def add_technical_indicators(self, df):
        df = df.copy()
        close  = df['close_price']
        high   = df['high_price']
        low    = df['low_price']
        volume = df['volume']

        # ── Price-based features ──────────────────────────
        df['returns']           = close.pct_change()
        df['log_returns']       = np.log(close / close.shift(1))
        df['high_low_spread']   = (high - low) / close
        df['open_close_spread'] = (df['open_price'] - close) / close

        # ── Moving Averages ───────────────────────────────
        df['sma_20']  = SMAIndicator(close, window=20).sma_indicator()
        df['sma_50']  = SMAIndicator(close, window=50).sma_indicator()
        df['sma_200'] = SMAIndicator(close, window=200).sma_indicator()
        df['ema_12']  = EMAIndicator(close, window=12).ema_indicator()
        df['ema_26']  = EMAIndicator(close, window=26).ema_indicator()

        # ── Price-to-MA ratios ────────────────────────────
        df['price_to_sma20']  = close / df['sma_20']  - 1
        df['price_to_sma50']  = close / df['sma_50']  - 1
        df['price_to_sma200'] = close / df['sma_200'] - 1
        df['ema_cross']       = df['ema_12'] / df['ema_26'] - 1

        # ── MACD ──────────────────────────────────────────
        macd = MACD(close)
        df['macd']        = macd.macd()
        df['macd_signal'] = macd.macd_signal()
        df['macd_diff']   = macd.macd_diff()

        # ── RSI ───────────────────────────────────────────
        df['rsi']    = RSIIndicator(close, window=14).rsi()
        df['rsi_7']  = RSIIndicator(close, window=7).rsi()
        df['rsi_21'] = RSIIndicator(close, window=21).rsi()

        # ── Stochastic + Williams %R ──────────────────────
        stoch = StochasticOscillator(high, low, close)
        df['stoch_k']    = stoch.stoch()
        df['stoch_d']    = stoch.stoch_signal()
        df['williams_r'] = WilliamsRIndicator(high, low, close, lbp=14).williams_r()

        # ── Rate of Change ────────────────────────────────
        df['roc_5']  = ROCIndicator(close, window=5).roc()
        df['roc_10'] = ROCIndicator(close, window=10).roc()
        df['roc_20'] = ROCIndicator(close, window=20).roc()

        # ── Bollinger Bands ───────────────────────────────
        bb = BollingerBands(close)
        df['bb_high']     = bb.bollinger_hband()
        df['bb_low']      = bb.bollinger_lband()
        df['bb_mid']      = bb.bollinger_mavg()
        df['bb_width']    = (df['bb_high'] - df['bb_low']) / df['bb_mid']
        df['bb_position'] = (close - df['bb_low']) / (df['bb_high'] - df['bb_low'] + 1e-9)

        # ── ATR + Volatility ──────────────────────────────
        df['atr']          = AverageTrueRange(high, low, close).average_true_range()
        df['atr_pct']      = df['atr'] / close
        df['volatility_10']= df['returns'].rolling(window=10).std()
        df['volatility_20']= df['returns'].rolling(window=20).std()

        # ── OBV ───────────────────────────────────────────
        df['obv'] = OnBalanceVolumeIndicator(close, volume).on_balance_volume()

        # ── Volume features ───────────────────────────────
        df['volume_change']  = volume.pct_change()
        df['volume_sma_20']  = volume.rolling(window=20).mean()
        df['volume_ratio']   = volume / (df['volume_sma_20'] + 1e-9)
        df['volume_spike']   = (df['volume_ratio'] > 2.0).astype(float)

        # ── Momentum ─────────────────────────────────────
        df['momentum_5']  = close / close.shift(5)  - 1
        df['momentum_10'] = close / close.shift(10) - 1
        df['momentum_20'] = close / close.shift(20) - 1

        # ── Multi-period returns ──────────────────────────
        df['return_1d']  = close.pct_change(1)
        df['return_3d']  = close.pct_change(3)
        df['return_5d']  = close.pct_change(5)
        df['return_10d'] = close.pct_change(10)
        df['return_20d'] = close.pct_change(20)

        # ── Price position features ───────────────────────
        df['hl_range_pct']   = (high - low) / (close + 1e-9)
        df['close_position'] = (close - low) / (high - low + 1e-9)  # 0=at low, 1=at high
        df['open_gap']       = (df['open_price'] - close.shift(1)) / (close.shift(1) + 1e-9)

        # ── Calendar features ─────────────────────────────
        df['day_of_week'] = pd.to_datetime(df['trade_date']).dt.dayofweek
        df['month']       = pd.to_datetime(df['trade_date']).dt.month

        # ── 52-week position ──────────────────────────────
        rolling_min = close.rolling(window=252).min()
        rolling_max = close.rolling(window=252).max()
        df['52w_position'] = (close - rolling_min) / (rolling_max - rolling_min + 1e-9)

        # ── Lagged returns ────────────────────────────────
        df['return_lag_1']  = df['returns'].shift(1)
        df['return_lag_2']  = df['returns'].shift(2)
        df['return_lag_3']  = df['returns'].shift(3)
        df['return_lag_5']  = df['returns'].shift(5)
        df['return_lag_10'] = df['returns'].shift(10)

        # ── Clean up ──────────────────────────────────────
        df = df.replace([np.inf, -np.inf], np.nan)
        df = df.dropna()

        return df

    def get_feature_columns(self, df):
        exclude = ['symbol', 'trade_date', 'close_price']
        return [col for col in df.columns if col not in exclude]
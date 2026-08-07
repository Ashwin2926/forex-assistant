import pandas as pd
import numpy as np
from app.models.schemas import RuleConfig


def ema(series: pd.Series, period: int) -> pd.Series:
    return series.ewm(span=period, adjust=False).mean()


def rsi(series: pd.Series, period: int = 14) -> pd.Series:
    delta = series.diff()
    gain = delta.where(delta > 0, 0.0)
    loss = -delta.where(delta < 0, 0.0)
    avg_gain = gain.rolling(window=period).mean()
    avg_loss = loss.rolling(window=period).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def macd(series: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9):
    ema_fast = ema(series, fast)
    ema_slow = ema(series, slow)
    macd_line = ema_fast - ema_slow
    signal_line = ema(macd_line, signal)
    histogram = macd_line - signal_line
    return macd_line, signal_line, histogram


def atr(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """Average True Range - measures volatility. df needs high, low, close columns."""
    high_low = df["high"] - df["low"]
    high_close = (df["high"] - df["close"].shift()).abs()
    low_close = (df["low"] - df["close"].shift()).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.rolling(window=period).mean()


def add_all_indicators(df: pd.DataFrame, config: RuleConfig = RuleConfig()) -> pd.DataFrame:
    """
    Expects df sorted ascending by timestamp with columns: open, high, low, close, volume.
    Returns df with indicator columns appended. Column names are generic (ema_fast/ema_slow/
    rsi/atr) rather than baking in a period, so rule logic stays decoupled from whatever
    periods a given RuleConfig uses — needed for parameter sweeps.
    """
    df = df.copy()
    df["ema_fast"] = ema(df["close"], config.ema_fast)
    df["ema_slow"] = ema(df["close"], config.ema_slow)
    df["rsi"] = rsi(df["close"], config.rsi_period)
    macd_line, signal_line, hist = macd(df["close"], config.macd_fast, config.macd_slow, config.macd_signal)
    df["macd"] = macd_line
    df["macd_signal"] = signal_line
    df["macd_hist"] = hist
    df["atr"] = atr(df, config.atr_period)
    return df

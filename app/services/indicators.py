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


def bollinger_bands(series: pd.Series, period: int = 20, std_mult: float = 2.0):
    """Middle band is the SMA (not EMA, unlike the trend rules elsewhere) -- Bollinger Bands
    are conventionally SMA-based since the band width itself needs a stable, non-decaying
    baseline for standard deviation to be measured against."""
    middle = series.rolling(window=period).mean()
    std = series.rolling(window=period).std()
    upper = middle + std_mult * std
    lower = middle - std_mult * std
    return upper, middle, lower


def stochastic(df: pd.DataFrame, k_period: int = 14, d_period: int = 3):
    """%K/%D momentum oscillator, 0-100 bounded. df needs high, low, close columns."""
    lowest_low = df["low"].rolling(window=k_period).min()
    highest_high = df["high"].rolling(window=k_period).max()
    percent_k = 100 * (df["close"] - lowest_low) / (highest_high - lowest_low)
    percent_d = percent_k.rolling(window=d_period).mean()
    return percent_k, percent_d


def adx(df: pd.DataFrame, period: int = 14) -> pd.Series:
    """
    Average Directional Index - trend STRENGTH (0-100), direction-agnostic by design; pairs
    with a direction-giving rule (stochastic here) rather than a trend-direction rule like
    trend_ema, since ADX alone can't say which way a strong trend is pointing.
    """
    up_move = df["high"].diff()
    down_move = -df["low"].diff()
    plus_dm = up_move.where((up_move > down_move) & (up_move > 0), 0.0)
    minus_dm = down_move.where((down_move > up_move) & (down_move > 0), 0.0)

    tr = atr(df, period=1)  # single-period true range, reuses atr()'s TR calc at period=1
    atr_smooth = tr.rolling(window=period).mean()
    plus_di = 100 * (plus_dm.rolling(window=period).mean() / atr_smooth)
    minus_di = 100 * (minus_dm.rolling(window=period).mean() / atr_smooth)

    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di)
    return dx.rolling(window=period).mean()


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
    bb_upper, bb_middle, bb_lower = bollinger_bands(df["close"])
    df["bb_upper"] = bb_upper
    df["bb_middle"] = bb_middle
    df["bb_lower"] = bb_lower
    stoch_k, stoch_d = stochastic(df)
    df["stoch_k"] = stoch_k
    df["stoch_d"] = stoch_d
    df["adx"] = adx(df)
    return df

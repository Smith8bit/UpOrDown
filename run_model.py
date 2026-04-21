import torch
import torch.nn as nn
import pandas as pd
import yfinance as yf
import ta
import pandas_market_calendars as mcal
from datetime import date, timedelta

# ── 1. model architecture ─────────────────────────────
class StockModel(nn.Module):
    def __init__(self, input_dim, embed_dim, dropout):
        super().__init__()
        self.input_proj = nn.Linear(input_dim, embed_dim)
        self.LSTM = nn.LSTM(embed_dim, embed_dim, 2, batch_first=True, bidirectional=False)
        self.batch_norm = nn.BatchNorm1d(embed_dim)
        self.norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Sequential(
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, embed_dim),
            nn.ReLU(),
            nn.Linear(embed_dim, 2),
            nn.Softmax(dim=-1)
        )

    def forward(self, x):
        x = self.input_proj(x)
        x, (h, c) = self.LSTM(x)
        h = h[1:].squeeze(0)
        h = self.batch_norm(h)
        h = self.norm(h)
        h = self.dropout(h)
        return self.out(h)

# ── 2. check if today is a trading day ────────────────
nyse = mcal.get_calendar("NYSE")
today = date.today()

today_schedule = nyse.schedule(start_date=today, end_date=today)
if today_schedule.empty:
    print(f"{today} is not a trading day. Skipping.")
    exit()

# ── 3. find next trading day (what we're predicting) ──
next_market_day = nyse.schedule(
    start_date=today + timedelta(1),
    end_date=today + timedelta(7)
).index[0].date()

print(f"Today: {today} | Predicting for: {next_market_day}")

# ── 4. check if prediction already exists ─────────────
try:
    results = pd.read_csv("results.csv")
    results["date"] = pd.to_datetime(results["date"]).dt.date
    if next_market_day in results["date"].values:
        print(f"Prediction for {next_market_day} already exists. Skipping.")
        exit()
except FileNotFoundError:
    results = pd.DataFrame(columns=["date", "prediction", "actual", "correct"])

# ── 5. download data ──────────────────────────────────
SEQUENCE_LENGTH = 60
LOOKBACK = SEQUENCE_LENGTH + 300

def download_and_clean(ticker):
    df = yf.download(ticker, period="2y", auto_adjust=True, progress=False)
    df.columns = df.columns.droplevel(1)
    df.index.name = "Date"
    df = df.reset_index()
    df["Date"] = pd.to_datetime(df["Date"]).dt.normalize()
    return df

print("Downloading market data...")
df_sp500 = download_and_clean("^GSPC")
df_gold  = download_and_clean("GC=F")
df_oil   = download_and_clean("CL=F")
df_bond  = download_and_clean("^TNX")

# ── 6. build features (exactly as in training) ────────
df = df_sp500.copy()
high, low, close, volume = df["High"], df["Low"], df["Close"], df["Volume"]

# Returns
df["Return"]       = df["Close"].pct_change()
df["PerOpen"]      = df["Open"].pct_change()
df["PerHigh"]      = df["High"].pct_change()
df["PerLow"]       = df["Low"].pct_change()

# Volume
df["Volume_Ratio"] = df["Volume"] / df["Volume"].rolling(20).mean()

# SMA distance
for w in [5, 10, 20, 50, 200]:
    sma = ta.trend.sma_indicator(df["Close"], window=w)
    df[f"SMA_{w}_dist"] = (df["Close"] - sma) / df["Close"]

# EMA distance
for w in [5, 10, 20, 50, 200]:
    ema = ta.trend.ema_indicator(df["Close"], window=w)
    df[f"EMA_{w}_dist"] = (df["Close"] - ema) / df["Close"]

# EMA crosses
df["EMA_5_20_cross"]  = df["EMA_5_dist"]  - df["EMA_20_dist"]
df["EMA_20_50_cross"] = df["EMA_20_dist"] - df["EMA_50_dist"]

# MACD
df["MACD"] = ta.trend.macd(df["Close"])

# Parabolic SAR
df["PSAR_DOWN"] = ta.trend.psar_down(high, low, close) / close
df["PSAR_UP"]   = ta.trend.psar_up(high, low, close)   / close

# Bollinger Bands
df["Bollband_Hband"] = ta.volatility.bollinger_hband(close) / close
df["Bollband_Lband"] = ta.volatility.bollinger_lband(close) / close
df["Bollband_Pband"] = ta.volatility.bollinger_pband(close)
df["Bollband_Wband"] = ta.volatility.bollinger_wband(close)

# Stochastic RSI
df["Stochastic_d"] = ta.momentum.stochrsi_d(close)
df["Stochastic_k"] = ta.momentum.stochrsi_k(close)

# Momentum
df["RSI"] = ta.momentum.rsi(df["Close"], window=14) / 100
df["MFI"] = ta.volume.money_flow_index(high, low, close, volume) / 100
df["CMF"] = ta.volume.chaikin_money_flow(high, low, close, volume)

# Volatility
df["Volatility"] = df["Return"].rolling(10).std()

# External features
df_gold["Gold_Return"] = df_gold["Close"].pct_change()
df_gold["Gold_RSI"]    = ta.momentum.rsi(df_gold["Close"], window=14) / 100

df_oil["Oil_Return"]       = df_oil["Close"].pct_change()
df_oil["Oil_Volume_Ratio"] = df_oil["Volume"] / df_oil["Volume"].rolling(20).mean()
df_oil["Oil_RSI"]          = ta.momentum.rsi(df_oil["Close"], window=14) / 100

df_bond["Bond_Return"] = df_bond["Close"].pct_change()
df_bond["Bond_Yield"]  = df_bond["Close"] / 10
df_bond["Bond_RSI"]    = ta.momentum.rsi(df_bond["Close"], window=14) / 100

df = df.merge(df_gold[["Date", "Gold_Return", "Gold_RSI"]], on="Date", how="left") \
       .merge(df_oil[["Date", "Oil_Return", "Oil_Volume_Ratio", "Oil_RSI"]], on="Date", how="left") \
       .merge(df_bond[["Date", "Bond_Return", "Bond_Yield", "Bond_RSI"]], on="Date", how="left")

df = df.ffill().dropna().reset_index(drop=True)

# ── 7. prepare features ───────────────────────────────
DROP_COLS = ["Volume", "Close", "Open", "High", "Low", "Date"]
feature_cols = [col for col in df.columns if col not in DROP_COLS]

df = df.tail(LOOKBACK).reset_index(drop=True)

# ── 8. normalize ──────────────────────────────────────
features = torch.tensor(df[feature_cols].values, dtype=torch.float32)
mean = features.mean(dim=0)
std  = features.std(dim=0)
std[std == 0] = 1

features_norm = (features - mean) / std
input_tensor = features_norm[-SEQUENCE_LENGTH:].unsqueeze(0)  # (1, 60, 38)

print(f"Input shape: {input_tensor.shape}")

# ── 9. load model and predict ─────────────────────────
model = StockModel(input_dim=41, embed_dim=256, dropout=0.25)
model.load_state_dict(torch.load("LSTM.pth", map_location="cpu", weights_only=True))
model.eval()

with torch.no_grad():
    output = model(input_tensor)
    prediction = output.argmax(dim=1).item()
    confidence = output[0][prediction].item()

print(f"Prediction for {next_market_day}: {'UP' if prediction == 1 else 'DOWN'} (confidence: {confidence:.2%})")

# ── 10. save to results.csv ───────────────────────────
new_row = pd.DataFrame([{
    "date": next_market_day,   # ← the day being predicted, not today
    "prediction": prediction,
    "actual": None,
    "correct": None
}])

results = pd.concat([results, new_row], ignore_index=True)
results.to_csv("results.csv", index=False)
print(f"Saved — predicting {next_market_day} will go {'UP' if prediction == 1 else 'DOWN'}")

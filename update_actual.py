import pandas as pd
import yfinance as yf
import pandas_market_calendars as mcal
from datetime import date

# ── 1. check if today is a trading day ────────────────
nyse = mcal.get_calendar("NYSE")
today = date.today()

today_schedule = nyse.schedule(start_date=today, end_date=today)
if today_schedule.empty:
    print(f"{today} is not a trading day. Skipping.")
    exit()

# ── 2. load results.csv ───────────────────────────────
try:
    df = pd.read_csv("results.csv")
    df["date"] = pd.to_datetime(df["date"]).dt.date
except FileNotFoundError:
    print("results.csv not found. Run backtest.py first.")
    exit()

# ── 3. check if today has a pending prediction ────────
mask = (df["date"] == today) & (df["actual"].isna())
if not mask.any():
    print(f"No pending prediction found for {today}. Already updated or doesn't exist.")
    exit()

# ── 4. get actual outcome from yfinance ───────────────
print(f"Fetching actual outcome for {today}...")

spy = yf.download("^GSPC", period="5d", auto_adjust=True, progress=False)
spy.columns = spy.columns.droplevel(1)
closes = spy["Close"]

if len(closes) < 2:
    print("Not enough data from yfinance. Market may still be open.")
    exit()

actual = int(closes.iloc[-1] > closes.iloc[-2])

# ── 5. update results.csv ─────────────────────────────
prediction = int(df.loc[mask, "prediction"].values[0])
correct    = int(prediction == actual)

df.loc[mask, "actual"]  = actual
df.loc[mask, "correct"] = correct

df.to_csv("results.csv", index=False)
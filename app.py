import streamlit as st
import pandas as pd
import streamlit.components.v1 as components
import pandas_market_calendars as mcal
import pytz
from datetime import date, timedelta, datetime

st.set_page_config(page_title="Daily Stock Prediction", layout="centered")
st.markdown("""
    <style>
        /* Reduce the default bottom padding */
        .block-container {
            padding-bottom: 1rem; 
        }
        /* Hide the Streamlit footer */
        footer {
            visibility: hidden;
        }
        .stApp a:first-child {
            display: none;
        /* disable title link on markdown */
}
    </style>
""", unsafe_allow_html=True)

# ── load data ─────────────────────────────────────────
df = pd.read_csv("results.csv")
df["date"] = pd.to_datetime(df["date"])
latest = df.iloc[-1]

# ── next market day ───────────────────────────────────
ny_tz = pytz.timezone('US/Eastern')
today_ny = datetime.now(ny_tz).date()

nyse = mcal.get_calendar("NYSE")
next_market_day = nyse.schedule(
    start_date=today_ny, 
    end_date=today_ny + timedelta(7)
).index[0].date()

# ── header ────────────────────────────────────────────
st.markdown(f"""
    <h1 style="font-size: 2rem; font-weight: 600;">
        On {next_market_day.strftime('%A, %b %d')} — S&P 500 will...
    </h1>   
""", unsafe_allow_html=True)

if latest["prediction"] == 1:
    st.markdown("""<p style="
                    font-size: 8rem;
                    font-weight: 800;
                    font-family: Source Sans;
                    color: rgb(92, 228, 136);
                    background-color: rgba(92, 228, 136, 0.2);
                    border-radius: 1.5rem;
                    margin: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                ">UP</p>""", unsafe_allow_html=True)
else:
    st.markdown("""<p style="
                    font-size: 8rem;
                    font-weight: 800;
                    font-family: Source Sans;
                    color: rgb(255, 108, 108);
                    background-color: rgba(255, 108, 108, 0.2);
                    border-radius: 1.5rem;
                    margin: 0;
                    display: flex;
                    justify-content: center;
                    align-items: center;
                ">DOWN</p>""", unsafe_allow_html=True)

st.divider()

# ── accuracy ──────────────────────────────────────────
st.subheader("Accuracy")

totalAcc = df["correct"].mean() * 100
st.metric("Total Accuracy", f"{totalAcc:.2f}%")

# ── last 60 bar ───────────────────────────────────────
last_60 = df.tail(61).iloc[:-1].copy()
acc_60 = last_60["correct"].mean() * 100
bars_data = []
for _, row in last_60.iterrows():
    if pd.isna(row["correct"]):
        bars_data.append("null")
    elif row["correct"] == 1.0:
        bars_data.append("1")
    else:
        bars_data.append("0")

bars_json = "[" + ",".join(bars_data) + "]"

components.html(f"""
<div style="font-family: sans-serif; padding: 4px 0;">
    <div id="strip" style="display: flex; gap: 2px; height: 48px;"></div>
    <div style="display: flex; justify-content: space-between; margin-top: 8px; font-size: 12px; color: #888;">
        <span>Last 60 days</span>
        <span style="font-size: 13px; font-weight: 600;">{acc_60:.2f}% Correct</span>
        <span>Today</span>
    </div>
</div>
<script>
const data = {bars_json};
const strip = document.getElementById('strip');
data.forEach(val => {{
    const bar = document.createElement('div');
    bar.style.flex = '1';
    bar.style.borderRadius = '2px';
    if (val === null) {{
        bar.style.background = '#e0e0e0';
    }} else if (val === 1) {{
        bar.style.background = '#28ef49';
    }} else {{
        bar.style.background = '#f2332f';
    }}
    strip.appendChild(bar);
}});
</script>
""", height=80)

st.divider()

# ── TradingView ───────────────────────────────────────
st.subheader("SPY Chart")

components.html("""
<div class="tradingview-widget-container">
  <div class="tradingview-widget-container__widget"></div>
  <script type="text/javascript" src="https://s3.tradingview.com/external-embedding/embed-widget-advanced-chart.js" async>
  {
    "width": "100%",
    "height": "480",
    "allow_symbol_change": true,
    "calendar": false,
    "details": false,
    "hide_side_toolbar": true,
    "hide_top_toolbar": false,
    "hide_legend": false,
    "hide_volume": false,
    "hotlist": false,
    "interval": "D",
    "locale": "en",
    "save_image": true,
    "style": "1",
    "symbol": "AMEX:SPY",
    "theme": "dark",
    "timezone": "Asia/Bangkok",
    "backgroundColor": "#0F0F0F",
    "gridColor": "rgba(242, 242, 242, 0.06)",
    "watchlist": [],
    "withdateranges": false,
    "compareSymbols": [],
    "studies": []
  }
  </script>
</div>
""", height=500, scrolling=False)

st.divider()

st.markdown('<p style=" text-align: justify;"><b>Disclaimer:</b> The information on this website is for general information and educational purposes only. It does not constitute investment advice, a solicitation to buy or sell securities, or a recommendation. All forecasts, projections, and opinions are forward-looking statements based on subjective assumptions that may not be realized. Actual results may differ materially.<br><br>We do not guarantee the accuracy, completeness, or timeliness of the information. Using this information is at your own risk. Past performance is not a guide to future performance. We accept no liability for any direct or consequential loss arising from any use of this information.</p>', unsafe_allow_html=True)
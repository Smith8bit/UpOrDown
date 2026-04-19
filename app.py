import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import pandas_market_calendars as mcal
from datetime import date, timedelta

st.set_page_config(page_title="Daily Stock Prediction", layout="centered")

# ── load data ─────────────────────────────────────────
df = pd.read_csv("results.csv")
df["date"] = pd.to_datetime(df["date"])
latest = df.iloc[-1]
completed = df.dropna(subset=["actual"]).copy()

# ── next market day ───────────────────────────────────
nyse = mcal.get_calendar("NYSE")
next_market_day = nyse.schedule(
    start_date=date.today(),
    end_date=date.today() + timedelta(7)
).index[0].date()

# ── header ────────────────────────────────────────────
st.markdown(f"""
    <h1 style="font-size: 2rem; font-weight: 600;">
        On {next_market_day.strftime('%A, %b %d')} — S&P 500 will...
    </h1>
""", unsafe_allow_html=True)

if latest["prediction"] == 1:
    st.success("## UP")
else:
    st.error("## DOWN")

st.divider()

# ── accuracy metrics ──────────────────────────────────
if len(completed) > 0:
    total = len(completed)
    correct_count = int(completed["correct"].sum())
    wrong_count = total - correct_count
    accuracy = completed["correct"].mean() * 100

    # current streak
    streak = 0
    for val in reversed(completed["correct"].tolist()):
        if val == 1:
            streak += 1
        else:
            break

    col1, col2, col3     = st.columns(3)
    col1.metric("Total", total)
    col2.metric("Correct", correct_count)
    col3.metric("Accuracy", f"{accuracy:.1f}%")
else:
    st.info("No completed predictions yet.")

st.divider()

# ── heatmap ───────────────────────────────────────────
st.subheader("Prediction History")

year = date.today().year

st.markdown("🟥 Wrong &nbsp; 🟩 Correct")

# only use NYSE trading days — skip weekends and holidays
schedule = nyse.schedule(
    start_date=f"{year}-01-01",
    end_date=f"{year}-12-31"
)
trading_days = pd.DataFrame({"date": schedule.index.normalize()})
trading_days["date"] = trading_days["date"].dt.date
trading_days["date"] = pd.to_datetime(trading_days["date"])
trading_days["week"] = trading_days["date"].dt.isocalendar().week.astype(int)
trading_days["weekday"] = trading_days["date"].dt.weekday
trading_days["date_str"] = trading_days["date"].dt.strftime("%Y-%m-%d")

# merge correctness
trading_days = trading_days.merge(
    completed[["date", "correct"]],
    on="date",
    how="left"
)

# pivot
pivot = trading_days.pivot_table(
    index="weekday",
    columns="week",
    values="correct",
    aggfunc="first"
)

pivot_dates = trading_days.pivot_table(
    index="weekday",
    columns="week",
    values="date_str",
    aggfunc="first"
)

fig = go.Figure(go.Heatmap(
    z=pivot.values,
    text=pivot_dates.values,
    colorscale=[
        [0.0, "#f2332f"],   # wrong = red
        [1.0, "#28ef49"],   # correct = green
    ],
    zmin=0, zmax=1,
    showscale=False,
    xgap=3,
    ygap=3,
    hovertemplate="%{text}<extra></extra>",
))

fig.update_layout(
    height=180,
    margin=dict(t=10, b=10, l=50, r=10),
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    yaxis=dict(
        tickvals=[0, 1, 2, 3, 4],
        ticktext=["Mon", "Tue", "Wed", "Thu", "Fri"],
        showgrid=False
    ),
    xaxis=dict(showgrid=False, showticklabels=False)
)

st.plotly_chart(fig, use_container_width=True)
import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd

# 1. Config and Title
st.set_page_config(page_title="Taiwan Stock Dashboard", layout="wide")
st.title("🇹🇼 Taiwan Market")

raw_tickers = [
    "2330",
    "2317",
    "2454",
    "2308",
    "3711",
    "2891",
    "2881",
    "2382",
    "2882",
    "2345",
    "2303",
    "2412",
    "2884",
    "6669",
    "2886",
    "3231",
    "2885",
    "2887",
    "3017",
    "2383",
    "1216",
    "2327",
    "2357",
    "2890",
    "2360",
    "1303",
    "2892",
    "2301",
    "2880",
    "3661",
    "2883",
    "2408",
    "2379",
    "3665",
    "5880",
    "3008",
    "3034",
    "2002",
    "2603",
    "3653",
    "1301",
    "2059",
    "4904",
    "2207",
    "3045",
    "6919",
    "2395",
    "2912",
    "2615",
    "6505",
]

# Sidebar for controls
st.sidebar.header("Settings")
vol_sma_length = st.sidebar.slider("Volume SMA Length", 20, 120, 60)
stoch_k_length = st.sidebar.slider("Stochastic K Length", 5, 20, 9)
k_threshold = st.sidebar.slider("K Threshold", 10, 50, 20)


def flatten_multiindex_columns(df):
    """Flatten MultiIndex columns from yfinance."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    return df


@st.cache_data(ttl=3600)  # Cache raw data for 1 hour
def fetch_raw_data(ticker_list):
    """Fetch raw stock data from yfinance (cached)."""
    raw_data = {}
    for symbol in ticker_list:
        full_symbol = f"{symbol}.TW"
        df = yf.download(full_symbol, period="2y", progress=False)
        if not df.empty:
            df = flatten_multiindex_columns(df)
            raw_data[symbol] = df
    return raw_data


# Force refresh button
st.sidebar.divider()
if st.sidebar.button("🔄 Force Refresh Data", use_container_width=True):
    fetch_raw_data.clear()
    st.rerun()


def calculate_indicators(df, vol_sma_len, stoch_k_len):
    """Calculate all technical indicators."""
    indicators = {}
    indicators["vol_sma"] = ta.sma(df["Volume"], length=vol_sma_len)

    # Stochastic K
    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=stoch_k_len)
    indicators["stoch_k"] = (
        stoch[f"STOCHk_{stoch_k_len}_3_3"] if stoch is not None else None
    )

    return indicators


def extract_latest_values(df, indicators):
    """Extract latest values from dataframe and indicators."""
    values = {
        "close": float(df["Close"].iloc[-1]),
        "volume": int(df["Volume"].iloc[-1]),
    }

    indicator_keys = ["vol_sma", "stoch_k"]
    for key in indicator_keys:
        indicator = indicators.get(key)
        if indicator is not None:
            val = indicator.iloc[-1]
            if key == "vol_sma":
                values[key] = int(val) if pd.notna(val) else None
            else:
                values[key] = float(val) if pd.notna(val) else None
        else:
            values[key] = None

    return values


def check_conditions(values, k_thresh):
    """Check which conditions are met."""
    k_val = values.get("stoch_k")
    vol_sma_val = values.get("vol_sma")

    k_below_threshold = (
        k_val < k_thresh if k_val is not None and pd.notna(k_val) else False
    )
    vol_above_sma = (
        values["volume"] > vol_sma_val
        if vol_sma_val is not None and pd.notna(vol_sma_val)
        else False
    )

    return k_below_threshold, vol_above_sma


def build_result_dict(symbol, values, vol_sma_len, stoch_k_len, conditions):
    """Build the result dictionary for a single ticker."""
    k_below_threshold, vol_above_sma = conditions
    match_count = sum([k_below_threshold, vol_above_sma])

    return {
        "Ticker": symbol,
        "Close": round(values["close"], 2),
        "Volume": values["volume"],
        f"VolSMA{vol_sma_len}": (
            int(values["vol_sma"])
            if values["vol_sma"] is not None and pd.notna(values["vol_sma"])
            else None
        ),
        f"K({stoch_k_len})": (
            round(values["stoch_k"], 2)
            if values["stoch_k"] is not None and pd.notna(values["stoch_k"])
            else None
        ),
        "K<Thresh": k_below_threshold,
        "Vol>SMA": vol_above_sma,
        "Matches": match_count,
    }


# Function to process data with indicators
def process_data(raw_data, vol_sma_len, stoch_k_len, k_thresh):
    """Process raw stock data with technical indicators."""
    processed_results = []

    for symbol, df in raw_data.items():
        indicators = calculate_indicators(df, vol_sma_len, stoch_k_len)
        values = extract_latest_values(df, indicators)
        conditions = check_conditions(values, k_thresh)
        result = build_result_dict(symbol, values, vol_sma_len, stoch_k_len, conditions)
        processed_results.append(result)

    return pd.DataFrame(processed_results)


# 3. App Execution
# Fetch raw data (cached, only happens once per hour)
with st.spinner("Fetching data from Yahoo Finance..."):
    raw_data = fetch_raw_data(raw_tickers)
    refresh_time = pd.Timestamp.now(tz="Asia/Taipei").strftime("%Y-%m-%d %H:%M:%S")

if raw_data:
    # Get the latest date from the first stock for Close column header
    latest_date = list(raw_data.values())[0].index[-1].strftime("%Y-%m-%d")

    # Process data with current settings (runs every time settings change)
    data_table = process_data(
        raw_data,
        vol_sma_length,
        stoch_k_length,
        k_threshold,
    )

    # Sort by matching conditions (descending)
    data_table = data_table.sort_values(by="Matches", ascending=False).reset_index(
        drop=True
    )

    # Adjust index to start from 1 for display
    data_table.index = data_table.index + 1

    # Create TradingView URLs for Ticker column
    data_table["Ticker"] = data_table["Ticker"].apply(
        lambda x: f"https://www.tradingview.com/chart/?symbol=TWSE%3A{x}"
    )

    # Format volume columns with thousands separators
    data_table["Volume"] = data_table["Volume"].apply(
        lambda x: f"{x:,}" if pd.notna(x) else None
    )
    data_table[f"VolSMA{vol_sma_length}"] = data_table[f"VolSMA{vol_sma_length}"].apply(
        lambda x: f"{int(x):,}" if pd.notna(x) else None
    )

    # Configure column formatting
    column_config = {
        "Ticker": st.column_config.LinkColumn(
            "Ticker",
            help="Click to view on TradingView",
            display_text=r"https://www\.tradingview\.com/chart/\?symbol=TWSE%3A(.*)",
            width="small",
            pinned=True,
        ),
        "Close": st.column_config.NumberColumn(
            f"Close ({latest_date})",
            format="%.2f",
        ),
        "Volume": st.column_config.TextColumn(
            "Volume",
            width="small",
        ),
        f"VolSMA{vol_sma_length}": st.column_config.TextColumn(
            f"VolSMA{vol_sma_length}",
            width="small",
        ),
        f"K({stoch_k_length})": st.column_config.NumberColumn(
            f"K({stoch_k_length})",
            format="%.2f",
            width="small",
        ),
        "K<Thresh": st.column_config.CheckboxColumn(
            "K<Thresh",
            width="small",
        ),
        "Vol>SMA": st.column_config.CheckboxColumn(
            "Vol>SMA",
            width="small",
        ),
        "Matches": st.column_config.NumberColumn(
            "Matches",
            format="%d",
            width="small",
        ),
    }

    st.subheader("Summary")
    st.caption(f"Last refreshed: {refresh_time}")
    st.dataframe(
        data_table,
        width="stretch",
        height="content",
        column_config=column_config,
    )
else:
    st.error("Could not fetch data. Please check your internet connection.")

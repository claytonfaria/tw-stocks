import streamlit as st
import yfinance as yf
import pandas_ta as ta
import pandas as pd
import requests
from datetime import datetime, timedelta
import altair as alt

# 1. Config and Title
st.set_page_config(page_title="Taiwan Stock Dashboard", layout="wide")
st.title("🇹🇼 Taiwan Market")

# Initialize session state and query params
if "selected_ticker" not in st.session_state:
    st.session_state.selected_ticker = None

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

# Default settings
default_vol_sma_length = 60
default_stoch_k_length = 9
default_k_threshold = 20


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


@st.cache_data(ttl=3600)
def fetch_institutional_data(stock_id, days=90):
    """Fetch institutional investor data from FinMind API."""
    end_date = datetime.now()
    start_date = end_date - timedelta(days=days)

    url = "https://api.finmindtrade.com/api/v4/data"
    parameter = {
        "dataset": "TaiwanStockInstitutionalInvestorsBuySell",
        "data_id": stock_id,
        "start_date": start_date.strftime("%Y-%m-%d"),
        "end_date": end_date.strftime("%Y-%m-%d"),
    }

    try:
        response = requests.get(url, params=parameter, timeout=10)
        response.raise_for_status()
        data = response.json()

        if data.get("data"):
            df = pd.DataFrame(data["data"])
            # Convert buy/sell to integers and format
            df["buy"] = pd.to_numeric(df["buy"], errors="coerce").fillna(0).astype(int)
            df["sell"] = (
                pd.to_numeric(df["sell"], errors="coerce").fillna(0).astype(int)
            )
            df["net"] = df["buy"] - df["sell"]

            # Map investor types to proper categories
            investor_map = {
                "Foreign_Investor": "Foreign Investors (外資)",
                "Investment_Trust": "Investment Trust (投信)",
                "Dealer_self": "Dealers (自營商)",
                "Dealer_Hedging": "Dealers (自營商)",
                "Foreign_Dealer_Self": "Foreign Investors (外資)",
            }

            df["category"] = df["name"].map(investor_map)

            # Group by date and category (merge Dealer types and Foreign types)
            grouped = (
                df.groupby(["date", "stock_id", "category"])
                .agg({"buy": "sum", "sell": "sum", "net": "sum"})
                .reset_index()
            )

            # Calculate Three Major Institutional Investors total
            three_major = (
                grouped.groupby(["date", "stock_id"])
                .agg({"buy": "sum", "sell": "sum", "net": "sum"})
                .reset_index()
            )
            three_major["category"] = "Three Major Institutional Investors (三大法人)"

            # Combine grouped data with three major total
            result = pd.concat([grouped, three_major], ignore_index=True)

            return result.sort_values(["date", "category"], ascending=[False, True])
        return pd.DataFrame()
    except Exception as e:
        st.error(f"Error fetching institutional data: {str(e)}")
        return pd.DataFrame()


def create_candlestick_chart(stock_id, raw_data, days=90):
    """Create an Altair candlestick chart with volume and stochastic K indicator."""
    if stock_id not in raw_data:
        return None

    df = raw_data[stock_id].copy()

    # Calculate Stochastic K and D
    stoch = ta.stoch(df["High"], df["Low"], df["Close"], k=9)
    df["stoch_k"] = stoch["STOCHk_9_3_3"] if stoch is not None else None
    df["stoch_d"] = stoch["STOCHd_9_3_3"] if stoch is not None else None

    # Calculate Volume SMA 60
    df["vol_sma_60"] = ta.sma(df["Volume"], length=60)

    # Calculate Price SMAs
    df["sma_20"] = ta.sma(df["Close"], length=20)
    df["sma_200"] = ta.sma(df["Close"], length=200)

    # Reset index to get Date as a column (keeps original OHLC column names)
    df = df.reset_index()
    df = df.rename(columns={"index": "Date"})

    # Remove any rows with NaN values in OHLC data
    df = df.dropna(subset=["Open", "High", "Low", "Close"])

    # Take last N days based on parameter
    df = df.tail(days)
    # Price candlestick chart - Taiwan style: red = up, green = down
    open_close_color = (
        alt.when("datum.Close >= datum.Open")
        .then(alt.value("#ae1325"))  # Red for up (close >= open)
        .otherwise(alt.value("#06982d"))  # Green for down (close < open)
    )

    base_price = alt.Chart(df).encode(
        alt.X("yearmonthdate(Date):O").axis(labelAngle=-45, title=None),
        color=open_close_color,
        tooltip=[
            alt.Tooltip("Date:T", title="Date", format="%Y-%m-%d"),
            alt.Tooltip("Open:Q", title="Open", format=".2f"),
            alt.Tooltip("High:Q", title="High", format=".2f"),
            alt.Tooltip("Low:Q", title="Low", format=".2f"),
            alt.Tooltip("Close:Q", title="Close", format=".2f"),
        ],
    )

    price_rule = base_price.mark_rule().encode(
        alt.Y("Low:Q").title("Price").scale(zero=False), alt.Y2("High:Q")
    )

    price_bar = base_price.mark_bar().encode(alt.Y("Open:Q"), alt.Y2("Close:Q"))

    # Prepare SMA data for proper legend
    sma_data = df[["Date", "sma_20", "sma_200"]].melt(
        id_vars=["Date"], var_name="SMA", value_name="value"
    )
    sma_data["SMA"] = sma_data["SMA"].map({"sma_20": "SMA 20", "sma_200": "SMA 200"})

    # SMA lines with legend
    sma_lines = (
        alt.Chart(sma_data)
        .mark_line(size=2)
        .encode(
            alt.X("yearmonthdate(Date):O"),
            alt.Y("value:Q"),
            color=alt.Color(
                "SMA:N",
                scale=alt.Scale(
                    domain=["SMA 20", "SMA 200"], range=["#FFA500", "#0000FF"]
                ),
                legend=alt.Legend(title="Moving Averages", orient="top"),
            ),
            strokeDash=alt.StrokeDash(
                "SMA:N",
                scale=alt.Scale(domain=["SMA 20", "SMA 200"], range=[[5, 5], [2, 2]]),
                legend=None,
            ),
        )
    )

    price_chart = (price_rule + price_bar + sma_lines).properties(height=250)

    # Volume chart with SMA 60
    volume_bars = (
        alt.Chart(df)
        .mark_bar(opacity=0.7)
        .encode(
            alt.X("yearmonthdate(Date):O").axis(labelAngle=-45, title=None),
            alt.Y("Volume:Q").title("Volume"),
            color=open_close_color,
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("Volume:Q", title="Volume", format=",.0f"),
            ],
        )
    )

    volume_sma_line = (
        alt.Chart(df)
        .mark_line(color="orange", size=2)
        .encode(
            alt.X("yearmonthdate(Date):O"),
            alt.Y("vol_sma_60:Q"),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("vol_sma_60:Q", title="Volume SMA 60", format=",.0f"),
            ],
        )
    )

    volume_chart = (volume_bars + volume_sma_line).properties(height=100)

    # Stochastic K and D chart
    # Prepare stochastic data for proper legend
    stoch_data = df[["Date", "stoch_k", "stoch_d"]].melt(
        id_vars=["Date"], var_name="Line", value_name="value"
    )
    stoch_data["Line"] = stoch_data["Line"].map({"stoch_k": "%K", "stoch_d": "%D"})

    stoch_lines = (
        alt.Chart(stoch_data)
        .mark_line(size=2)
        .encode(
            alt.X("yearmonthdate(Date):O").axis(labelAngle=-45, title="Date"),
            alt.Y("value:Q").title("Stochastic K9/D3").scale(domain=[0, 100]),
            color=alt.Color(
                "Line:N",
                scale=alt.Scale(domain=["%K", "%D"], range=["#1f77b4", "#ff7f0e"]),
                legend=alt.Legend(title="Stochastic", orient="top"),
            ),
            tooltip=[
                alt.Tooltip("Date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("Line:N", title="Line"),
                alt.Tooltip("value:Q", title="Value", format=".2f"),
            ],
        )
        .properties(height=100)
    )

    # Add reference lines for stochastic
    oversold_line = (
        alt.Chart(pd.DataFrame({"y": [20]}))
        .mark_rule(color="red", strokeDash=[5, 5])
        .encode(y="y:Q")
    )

    overbought_line = (
        alt.Chart(pd.DataFrame({"y": [80]}))
        .mark_rule(color="green", strokeDash=[5, 5])
        .encode(y="y:Q")
    )

    stoch_with_lines = stoch_lines + oversold_line + overbought_line

    # Combine all charts vertically
    combined_chart = (
        alt.vconcat(price_chart, volume_chart, stoch_with_lines)
        .resolve_scale(color="independent")
        .properties(title=f"Price Chart - {stock_id} (Last {days} Days)")
        .configure_view(strokeWidth=0)
    )

    return combined_chart


def create_institutional_chart(inst_data, stock_id, raw_data):
    """Create an Altair chart showing institutional investor buy/sell with stock price overlay."""
    if inst_data.empty or stock_id not in raw_data:
        return None

    # Get stock price data
    stock_df = raw_data[stock_id].copy()
    stock_df = stock_df.reset_index()
    stock_df.columns = ["date", "open", "high", "low", "close", "volume"]
    stock_df["date"] = pd.to_datetime(stock_df["date"])

    # Prepare institutional data
    inst_pivot = inst_data.pivot_table(
        index="date",
        columns="category",
        values="net",
        aggfunc="sum",
    ).reset_index()
    inst_pivot["date"] = pd.to_datetime(inst_pivot["date"])

    # Filter stock data to match institutional data date range
    min_date = inst_pivot["date"].min()
    max_date = inst_pivot["date"].max()
    stock_df = stock_df[(stock_df["date"] >= min_date) & (stock_df["date"] <= max_date)]

    # Prepare data for Altair - need to melt for proper layering
    inst_melted = inst_pivot.melt(
        id_vars=["date"],
        value_vars=[
            "Foreign Investors (外資)",
            "Investment Trust (投信)",
            "Dealers (自營商)",
        ],
        var_name="category",
        value_name="net",
    )

    # Create color scale
    color_scale = alt.Scale(
        domain=[
            "Foreign Investors (外資)",
            "Investment Trust (投信)",
            "Dealers (自營商)",
        ],
        range=["#1f77b4", "#9467bd", "#ff7f0e"],
    )

    # Create bar chart for institutional investors
    bars = (
        alt.Chart(inst_melted)
        .mark_bar(opacity=0.7)
        .encode(
            x=alt.X(
                "yearmonthdate(date):O", axis=alt.Axis(labelAngle=-45), title="Date"
            ),
            y=alt.Y("net:Q", title="Net Buy/Sell (shares)"),
            color=alt.Color(
                "category:N",
                scale=color_scale,
                legend=alt.Legend(title="Investor Type"),
            ),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("category:N", title="Investor Type"),
                alt.Tooltip("net:Q", title="Net Buy/Sell", format=",.0f"),
            ],
        )
    )

    # Create line chart for stock price on secondary axis
    # We'll scale the price to fit within the same visual space
    line = (
        alt.Chart(stock_df)
        .mark_line(color="#2ca02c", size=2)
        .encode(
            x=alt.X("yearmonthdate(date):O"),
            y=alt.Y("close:Q", title="Stock Price (TWD)"),
            tooltip=[
                alt.Tooltip("date:T", title="Date", format="%Y-%m-%d"),
                alt.Tooltip("close:Q", title="Stock Price", format=".2f"),
            ],
        )
    )

    # Layer charts with resolve to have independent y-axes
    chart = (
        alt.layer(bars, line)
        .resolve_scale(y="independent")
        .properties(
            title=f"Institutional Investor Activity - {stock_id}",
            width="container",
            height=400,
        )
        .configure_legend(orient="top")
        .interactive()
    )

    return chart


# Force refresh button
if st.sidebar.button("🔄 Force Refresh Data", width="stretch"):
    fetch_raw_data.clear()
    fetch_institutional_data.clear()
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


def build_result_dict(symbol, values, vol_sma_len, stoch_k_len, conditions, k_thresh):
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
        f"K < {k_thresh}": k_below_threshold,
        f"Vol > {vol_sma_len} SMA": vol_above_sma,
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
        result = build_result_dict(
            symbol, values, vol_sma_len, stoch_k_len, conditions, k_thresh
        )
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

    # Get ticker IDs
    ticker_ids = list(raw_data.keys())

    st.caption(f"Last refreshed: {refresh_time}")

    # Define page functions
    def historical_data_page():
        st.header("📈 Historical data")

        # Sidebar settings for this page
        st.sidebar.header("Table Settings")
        vol_sma_length = st.sidebar.slider(
            "Volume SMA Length", 20, 120, default_vol_sma_length
        )
        stoch_k_length = st.sidebar.slider(
            "Stochastic K Length", 5, 20, default_stoch_k_length
        )
        k_threshold = st.sidebar.slider("K Threshold", 10, 50, default_k_threshold)

        # Process data with current settings
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

        # Format volume columns with thousands separators
        data_table["Volume"] = data_table["Volume"].apply(
            lambda x: f"{x:,}" if pd.notna(x) else None
        )
        data_table[f"VolSMA{vol_sma_length}"] = data_table[
            f"VolSMA{vol_sma_length}"
        ].apply(lambda x: f"{int(x):,}" if pd.notna(x) else None)

        # Convert Ticker column to URL format for navigation
        data_table["Ticker"] = data_table["Ticker"].apply(
            lambda x: f"details?ticker={x}"
        )

        # Configure column formatting
        column_config = {
            "Ticker": st.column_config.LinkColumn(
                "Ticker",
                help="Click to view ticker details",
                display_text=r"details\?ticker=(.*)",
                width="small",
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
            f"K < {k_threshold}": st.column_config.CheckboxColumn(
                f"K < {k_threshold}",
                width="small",
            ),
            f"Vol > {vol_sma_length} SMA": st.column_config.CheckboxColumn(
                f"Vol > {vol_sma_length} SMA",
                width="small",
            ),
            "Matches": st.column_config.NumberColumn(
                "Matches",
                format="%d",
                width="small",
            ),
        }

        # Display the formatted dataframe
        st.dataframe(
            data_table,
            width="stretch",
            height="content",
            column_config=column_config,
            hide_index=True,
        )

    def ticker_details_page():
        st.header("📊 Ticker details")

        # Sidebar settings for this page
        st.sidebar.header("Chart Settings")
        price_chart_days = st.sidebar.slider(
            "Price Chart Days", 30, 365, 90, key="price_days"
        )
        institutional_days = st.sidebar.slider(
            "Institutional History Days", 30, 180, 30, key="inst_days"
        )

        # Get ticker from query params or default to first
        selected_ticker = st.query_params.get("ticker", ticker_ids[0])
        if selected_ticker not in ticker_ids:
            selected_ticker = ticker_ids[0]

        default_index = ticker_ids.index(selected_ticker)

        col1, col2 = st.columns([2, 1])

        with col1:
            # Update query param when selection changes
            def on_ticker_change():
                st.query_params.ticker = st.session_state.ticker_details_select

            selected_ticker = st.selectbox(
                "Select Ticker",
                options=ticker_ids,
                index=default_index,
                key="ticker_details_select",
                on_change=on_ticker_change,
            )

        with col2:
            st.write("")  # Empty space for layout

        if selected_ticker:
            # Display candlestick chart
            st.subheader(f"Price Chart - {selected_ticker}")
            candle_chart = create_candlestick_chart(
                selected_ticker, raw_data, price_chart_days
            )
            if candle_chart:
                st.altair_chart(candle_chart, width="stretch")

            st.divider()

            # Display institutional investor data
            st.subheader("Institutional Investor Activity")
            with st.spinner(f"Fetching institutional data for {selected_ticker}..."):
                inst_data = fetch_institutional_data(
                    selected_ticker, institutional_days
                )

            if not inst_data.empty:
                # Create and display chart
                chart = create_institutional_chart(inst_data, selected_ticker, raw_data)
                if chart:
                    st.altair_chart(chart, width="stretch")

                # st.divider()

                # col_left, col_right = st.columns([3, 1])

                # with col_left:
                #     st.caption(f"Latest: {inst_data['date'].max()}")
                #     st.dataframe(
                #         inst_data,
                #         width="stretch",
                #         height=400,
                #         column_config={
                #             "date": "Date",
                #             "stock_id": "Ticker",
                #             "category": "Investor Type",
                #             "buy": st.column_config.NumberColumn("Buy", format="%d"),
                #             "sell": st.column_config.NumberColumn("Sell", format="%d"),
                #             "net": st.column_config.NumberColumn("Net", format="%d"),
                #         },
                #         hide_index=True,
                #     )

                # with col_right:
                #     st.caption("Summary (Latest Date)")
                #     latest_data = inst_data[
                #         inst_data["date"] == inst_data["date"].max()
                #     ]

                #     # Show Three Major first
                #     three_major = latest_data[
                #         latest_data["category"]
                #         == "Three Major Institutional Investors (三大法人)"
                #     ]
                #     if not three_major.empty:
                #         row = three_major.iloc[0]
                #         st.metric("Three Major Buy", f"{row['buy']:,}")
                #         st.metric("Three Major Sell", f"{row['sell']:,}")
                #         st.metric(
                #             "Three Major Net", f"{row['net']:,}", delta=row["net"]
                #         )

                #     st.divider()

                #     # Show breakdown by investor type (excluding the total)
                #     st.caption("Breakdown by Type")
                #     breakdown = latest_data[
                #         latest_data["category"]
                #         != "Three Major Institutional Investors (三大法人)"
                #     ].sort_values("category")

                #     for _, row in breakdown.iterrows():
                #         with st.expander(row["category"]):
                #             st.metric("Buy", f"{row['buy']:,}")
                #             st.metric("Sell", f"{row['sell']:,}")
                #             st.metric("Net", f"{row['net']:,}")
            else:
                st.info(f"No institutional data available for {selected_ticker}")

    # Set up navigation with simple list
    pages = [
        st.Page(historical_data_page, title="Historical data", icon="📈", default=True),
        st.Page(
            ticker_details_page, title="Ticker details", icon="📊", url_path="details"
        ),
    ]

    pg = st.navigation(pages, position="sidebar")
    pg.run()

else:
    st.error("Could not fetch data. Please check your internet connection.")

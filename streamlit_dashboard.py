# streamlit_fmcg_dashboard.py
# ----------------------------------------
# Optimized FMCG Stock Analysis Dashboard
# ----------------------------------------
import streamlit as st
from mlxtend.frequent_patterns import apriori, association_rules
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import os
from datetime import datetime,timedelta
from sklearn.linear_model import LinearRegression
# -----------------------------
# ⚙️ STREAMLIT CONFIG
# -----------------------------
st.set_page_config(page_title="FMCG Stocks Dashboard", layout="wide")
st.title("Indian FMCG Stock Analysis Dashboard")
st.markdown("Analyze NIFTY FMCG company data interactively with efficient caching and optimized visualizations.")
# -----------------------------
# 📂 PATH CONFIGURATION
# -----------------------------
DATA_PATH = os.path.abspath(r"D:\DMBI\Project\fmcg_data")  # Change path if needed

if not os.path.exists(DATA_PATH):
    st.error(f"The folder path is invalid: {DATA_PATH}")
    st.stop()
else:
    st.sidebar.success(f"Data folder loaded successfully: {DATA_PATH}")

# -----------------------------
# 🧾 FILE DETECTION
# -----------------------------
files = [f for f in os.listdir(DATA_PATH) if f.endswith(".csv")]

if not files:
    st.error("No CSV files found in the given folder. Please check your dataset path.")
    st.stop()

companies = sorted(set([f.split("_")[0] for f in files]))
st.sidebar.header("Data Filters")

company = st.sidebar.selectbox("Select Company", companies)

# Default: show only daily data first for performance
interval_options = [f for f in files if f.startswith(company) and "day" in f.lower()]
if not interval_options:
    interval_options = [f for f in files if f.startswith(company)]

selected_file = st.sidebar.selectbox("Select Interval File", interval_options)

# -----------------------------
# 🧭 DATA LOADING FUNCTION
# -----------------------------
@st.cache_data(ttl=600)
def load_data(file):
    """Load and preprocess the selected stock file."""
    df = pd.read_csv(os.path.join(DATA_PATH, file))
    df.columns = [c.strip().capitalize() for c in df.columns]

    date_col = 'Datetime' if 'Datetime' in df.columns else 'Date'
    df[date_col] = pd.to_datetime(df[date_col], errors='coerce')
    df = df.dropna(subset=[date_col]).sort_values(date_col)

    # Downsample for large intraday data
    if len(df) > 10000:
        df = df.iloc[::5, :]

    return df, date_col

# Manual load button (avoids auto-refresh lag)
if st.sidebar.button("Load Selected File"):
    df, date_col = load_data(selected_file)

    st.success(f"Loaded {selected_file} ({len(df)} rows)")
    st.write(f"### Viewing data for **{company}** — *{selected_file.replace('.csv','')}*")
    st.dataframe(df.head(5000))
else:
    st.warning("Select a company and interval, then click **'Load Selected File'** to begin.")
    st.stop()

# -----------------------------
# 📊 BASIC STATS
# -----------------------------
st.subheader("Descriptive Statistics")
col1, col2, col3 = st.columns(3)
col1.metric("Mean Close Price", f"₹{df['Close'].mean():.2f}")
col2.metric("Max Price", f"₹{df['High'].max():.2f}")
col3.metric("Min Price", f"₹{df['Low'].min():.2f}")

# -----------------------------
# 🕒 PRICE TREND
# -----------------------------
st.subheader("Price Movement Over Time")
fig = go.Figure()
fig.add_trace(go.Candlestick(
    x=df[date_col],
    open=df['Open'], high=df['High'],
    low=df['Low'], close=df['Close'],
    name="Candlestick"
))
fig.update_layout(
    xaxis_title="Date",
    yaxis_title="Price (₹)",
    template="plotly_dark",
    height=500
)
st.plotly_chart(fig, use_container_width=True)

# -----------------------------
# 📉 RETURN & VOLATILITY
# -----------------------------
st.subheader("Daily Returns & Volatility")
df['Return'] = df['Close'].pct_change()
avg_return = df['Return'].mean() * 100
volatility = df['Return'].std() * 100

st.write(f"**Average Return:** {avg_return:.2f}% | **Volatility:** {volatility:.2f}%")
st.line_chart(df.set_index(date_col)['Return'])

# -----------------------------
# 📦 VOLUME ANALYSIS
# -----------------------------
st.subheader("Trading Volume Over Time")
st.area_chart(df.set_index(date_col)['Volume'])


@st.cache_data
def load_all_daily(files, data_path):
    """Load all daily FMCG data for correlation analysis."""
    close_dict = {}
    for f in files:
        try:
            # Only include daily files
            if "day" not in f.lower():
                continue

            comp = f.split("_")[0]
            df_temp = pd.read_csv(os.path.join(data_path, f))

            # Try both 'Date' and 'Datetime' as possible date columns
            date_col = None
            for col in df_temp.columns:
                if col.strip().lower() in ["date", "datetime"]:
                    date_col = col
                    break

            if not date_col:
                continue  # Skip files without a valid date column

            df_temp[date_col] = pd.to_datetime(df_temp[date_col], errors="coerce")
            df_temp = df_temp.dropna(subset=[date_col])

            if "Close" not in df_temp.columns:
                # Try lowercase or variant names
                close_cols = [c for c in df_temp.columns if "close" in c.lower()]
                if close_cols:
                    df_temp.rename(columns={close_cols[0]: "Close"}, inplace=True)
                else:
                    continue

            # Keep only necessary columns
            df_temp = df_temp[[date_col, "Close"]].dropna()
            close_dict[comp] = df_temp.set_index(date_col)["Close"]

        except Exception as e:
            st.warning(f"Skipped {f} due to: {e}")

    # If no valid data found, return empty DataFrame
    if not close_dict:
        st.info("No valid daily data found for correlation.")
        return pd.DataFrame()

    corr_df = pd.concat(close_dict, axis=1)
    corr_df.columns = close_dict.keys()  # rename to company names
    return corr_df


corr_df = load_all_daily(files, DATA_PATH)
if corr_df is not None and not corr_df.empty:
    st.subheader("Correlation Between FMCG Stocks (Daily Close)")
    corr_matrix = corr_df.corr()
    fig_corr = px.imshow(corr_matrix, text_auto=True, color_continuous_scale="RdBu_r",
                         title="Stock Correlation Matrix")
    st.plotly_chart(fig_corr, use_container_width=True)
else:
    st.info("No valid daily data found for correlation analysis.")

# -----------------------------
# 🧠 LINEAR REGRESSION FORECASTING
# -----------------------------
st.subheader("Linear Regression – Price Prediction & Accuracy")

# Prepare data
df_model = df[[date_col, 'Close']].dropna().copy()
df_model['Days_Since_Start'] = (df_model[date_col] - df_model[date_col].min()).dt.days  # convert dates to numbers

# Define X and y
X = df_model[['Days_Since_Start']]
y = df_model['Close']

# Train Linear Regression
model = LinearRegression()
model.fit(X, y)

# Predict for training data (for trendline and residuals)
y_pred = model.predict(X)

# Calculate R² score (model accuracy)
r2 = model.score(X, y)

# Predict for next 30 days
future_days = 30
last_time = df_model['Days_Since_Start'].max()
future_times = np.arange(last_time + 1, last_time + future_days + 1)
future_dates = [df_model[date_col].max() + timedelta(days=i) for i in range(1, future_days + 1)]
future_preds = model.predict(future_times.reshape(-1, 1))
pred_df = pd.DataFrame({'Date': future_dates, 'Predicted_Close': future_preds})

# Combine for plotting
fig_pred = go.Figure()

# Actual close prices
fig_pred.add_trace(go.Scatter(
    x=df_model[date_col],
    y=df_model['Close'],
    mode='lines',
    name='Actual Close',
    line=dict(color='cyan')
))

# Regression trendline (fitted)
fig_pred.add_trace(go.Scatter(
    x=df_model[date_col],
    y=y_pred,
    mode='lines',
    name='Fitted Trendline',
    line=dict(color='orange', dash='dot')
))

# Future prediction line
fig_pred.add_trace(go.Scatter(
    x=pred_df['Date'],
    y=pred_df['Predicted_Close'],
    mode='lines',
    name='Predicted Future',
    line=dict(color='lime', dash='dash')
))

fig_pred.update_layout(
    title=f"Linear Regression Forecast – {company}",
    xaxis_title="Date",
    yaxis_title="Price",
    template="plotly_dark",
    height=500
)

st.plotly_chart(fig_pred, use_container_width=True)

# -----------------------------
# 📉 RESIDUAL PLOT
# -----------------------------
st.subheader("Residual Analysis")

# Residual = Actual - Predicted
residuals = y - y_pred
fig_resid = go.Figure()
fig_resid.add_trace(go.Scatter(
    x=df_model[date_col],
    y=residuals,
    mode='markers',
    marker=dict(color='lightcoral', size=5),
    name='Residuals'
))
fig_resid.add_hline(y=0, line_dash="dot", line_color="white")
fig_resid.update_layout(
    title="Residual Plot (Actual - Predicted)",
    xaxis_title="Date",
    yaxis_title="Residual",
    template="plotly_dark",
    height=400
)
st.plotly_chart(fig_resid, use_container_width=True)

# -----------------------------
# 📊 MODEL PERFORMANCE SUMMARY
# -----------------------------
slope = model.coef_[0]
intercept = model.intercept_
st.markdown("Model Summary")
st.write(f"**Model Equation:**  `Price = {slope:.2f} × Days_Since_Start + {intercept:.2f}`")
st.write(f"**R² Score (Accuracy):**  `{r2*100:.2f}%`")
st.write(f"**Predicted Price after 30 days:**  ₹{future_preds[-1]:.2f}")
if r2 >= 0.9:
    st.success("Excellent model fit! (R² > 0.9)")
elif r2 >= 0.7:
    st.info("Good fit, though some variance remains unexplained.")
else:
    st.warning("Weak fit — the trend may not be purely linear.")
    
    
# -----------------------------
# 🧩 APRIORI ASSOCIATION RULE MINING
# -----------------------------
st.subheader("🧠 Association Rule Mining (Apriori Algorithm)")

if corr_df is not None and not corr_df.empty:
    st.markdown("Identify co-movement patterns between FMCG stocks using daily price directions (Up/Down).")

    min_sup = st.sidebar.slider("Minimum Support", 0.1, 0.9, 0.3)
    min_conf = st.sidebar.slider("Minimum Confidence", 0.5, 1.0, 0.7)

    # Convert to Up (1) / Down (0)
    returns = corr_df.pct_change().dropna()
    up_down_df = (returns > 0).astype(int)

    # Apply Apriori
    freq_items = apriori(up_down_df, min_support=min_sup, use_colnames=True)
    rules = association_rules(freq_items, metric="confidence", min_threshold=min_conf)

    if not rules.empty:
        # ✅ Convert frozensets → readable strings
        rules['antecedents'] = rules['antecedents'].apply(lambda x: ', '.join(list(x)))
        rules['consequents'] = rules['consequents'].apply(lambda x: ', '.join(list(x)))

        st.markdown("### 📜 Strong Association Rules")
        st.dataframe(rules[['antecedents', 'consequents', 'support', 'confidence', 'lift']].head(10))

        fig_rules = px.scatter(
            rules,
            x='support', y='confidence', size='lift',
            hover_data=['antecedents', 'consequents'],
            title='Association Rules (Support vs Confidence)',
            template='plotly_dark'
        )
        st.plotly_chart(fig_rules, use_container_width=True)
    else:
        st.info("ℹ️ No strong association rules found for the chosen thresholds.")
else:
    st.warning("⚠️ Load valid daily data for Apriori analysis.")




# -----------------------------
# 📤 EXPORT BUTTON
# -----------------------------
st.download_button(
    label="Download Filtered Data (CSV)",
    data=df.to_csv(index=False).encode('utf-8'),
    file_name=f"{company}_{selected_file}",
    mime='text/csv'
)

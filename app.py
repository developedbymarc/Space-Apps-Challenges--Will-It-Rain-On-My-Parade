import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import bnlearn as bn
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px

db_path = "large_merra2_data.db"

# Page configuration
st.set_page_config(
    page_title="Weather Prediction - NASA MERRA-2",
    page_icon="🌤️",
    layout="wide"
)

# Cache the model to avoid retraining on every interaction
@st.cache_resource
def load_and_train_model():
    """Load data and train Bayesian Network"""
    conn = sqlite3.connect(db_path)
    
    query = """
    SELECT timestamp, latitude, longitude, 
           T2M_Celsius, QV2M, QV10M, 
           PRECSNO, PRECTOT, 
           wind_speed_2m, wind_direction_2m,
           wind_speed_10m, wind_direction_10m
    FROM weather_data
    ORDER BY timestamp
    """
    
    df = pd.read_sql_query(query, conn)
    conn.close()
    
    # Convert timestamp - already in date format
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    
    # Engineer features
    df['temp_category'] = pd.cut(
        df['T2M_Celsius'], 
        bins=[-np.inf, 0, 10, 20, 30, np.inf],
        labels=['freezing', 'cold', 'mild', 'warm', 'hot']
    )
    
    df['wind_category'] = pd.cut(
        df['wind_speed_10m'],
        bins=[-np.inf, 2, 5, 10, 15, np.inf],
        labels=['calm', 'light', 'moderate', 'strong', 'very_strong']
    )
    
    df['precip_category'] = pd.cut(
        df['PRECTOT'],
        bins=[-np.inf, 0.1, 2.5, 10, 50, np.inf],
        labels=['none', 'light', 'moderate', 'heavy', 'very_heavy']
    )
    
    df['snow_category'] = pd.cut(
        df['PRECSNO'],
        bins=[-np.inf, 0.01, 1, 5, np.inf],
        labels=['none', 'light', 'moderate', 'heavy']
    )
    
    df['humidity_category'] = pd.cut(
        df['QV2M'],
        bins=[-np.inf, 0.005, 0.010, 0.015, np.inf],
        labels=['low', 'moderate', 'high', 'very_high']
    )
    
    # Weather condition
    conditions = []
    for _, row in df.iterrows():
        if row['PRECSNO'] > 0.1:
            condition = 'snowy'
        elif row['PRECTOT'] > 2.5:
            condition = 'rainy'
        elif row['wind_speed_10m'] > 10:
            condition = 'windy'
        elif row['T2M_Celsius'] < 0:
            condition = 'freezing'
        elif row['T2M_Celsius'] > 30:
            condition = 'hot'
        else:
            condition = 'clear'
        conditions.append(condition)
    
    df['weather_condition'] = conditions
    
    # Prepare for Bayesian Network
    bn_features = [
        'temp_category',
        'wind_category',
        'precip_category',
        'snow_category',
        'humidity_category',
        'weather_condition'
    ]
    
    df_discretized = df[bn_features].copy().dropna()
    
    for col in df_discretized.columns:
        df_discretized[col] = df_discretized[col].astype(str)
    
    # Train Bayesian Network
    model = bn.structure_learning.fit(df_discretized, methodtype='hc', scoretype='bic')
    model = bn.parameter_learning.fit(model, df_discretized)
    
    return model, df

@st.cache_data
def get_available_locations():
    """Get unique locations from database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT DISTINCT latitude, longitude FROM weather_data"
    locations = pd.read_sql_query(query, conn)
    conn.close()
    return locations

@st.cache_data
def get_date_range():
    """Get available date range from database"""
    conn = sqlite3.connect(db_path)
    query = "SELECT MIN(timestamp) as min_date, MAX(timestamp) as max_date FROM weather_data"
    dates = pd.read_sql_query(query, conn)
    conn.close()
    
    min_date = pd.to_datetime(dates.iloc[0]['min_date'], unit='s')
    max_date = pd.to_datetime(dates.iloc[0]['max_date'], unit='s')
    
    return min_date, max_date


def get_weather_data_for_datetime(target_datetime, tolerance_hours=1):
    """Get actual weather data for a specific datetime"""
    conn = sqlite3.connect(db_path)
    
    # Convert to Unix timestamp
    target_ts = int(target_datetime.timestamp())
    tolerance_seconds = tolerance_hours * 3600
    
    query = f"""
    SELECT * FROM weather_data
    WHERE timestamp BETWEEN {target_ts - tolerance_seconds} AND {target_ts + tolerance_seconds}
    ORDER BY ABS(timestamp - {target_ts})
    LIMIT 1
    """
    
    result = pd.read_sql_query(query, conn)
    conn.close()
    
    if len(result) > 0:
        result['timestamp'] = pd.to_datetime(result['timestamp'], unit='s')
        return result.iloc[0]
    return None


# Main App
def main():
    # Header
    st.title("🌤️ Weather Prediction System")
    st.markdown("### NASA MERRA-2 Dataset - Bayesian Network Predictions")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model... This may take a moment on first run."):
        model, historical_df = load_and_train_model()
    
    # Sidebar - Input Section
    st.sidebar.header("📍 Input Parameters")
    
    # Location selection
    locations = get_available_locations()
    
    if len(locations) == 1:
        lat = locations.iloc[0]['latitude']
        lon = locations.iloc[0]['longitude']
        st.sidebar.info(f"📍 Location: {lat:.4f}°, {lon:.4f}°")
        st.sidebar.markdown(f"[View on Map](https://www.google.com/maps?q={lat},{lon})")
    else:
        location_idx = st.sidebar.selectbox(
            "Select Location",
            range(len(locations)),
            format_func=lambda i: f"{locations.iloc[i]['latitude']:.4f}°, {locations.iloc[i]['longitude']:.4f}°"
        )
        lat = locations.iloc[location_idx]['latitude']
        lon = locations.iloc[location_idx]['longitude']
    
    # Date selection
    min_date, max_date = get_date_range()
    
    st.sidebar.markdown("### 📅 Date & Time")
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=min_date.date(),
        min_value=min_date.date(),
        max_value=max_date.date()
    )
    
    selected_time = st.sidebar.time_input(
        "Select Time",
        value=datetime.now().time()
    )
    
    target_datetime = datetime.combine(selected_date, selected_time)
    
    # Input method selection
    st.sidebar.markdown("---")
    input_method = st.sidebar.radio(
        "Input Method",
        ["Use Historical Data", "Manual Input"]
    )
    
    # Get data based on input method
    if input_method == "Use Historical Data":
        weather_data = get_weather_data_for_datetime(target_datetime)
        
        if weather_data is not None:
            st.sidebar.success("✅ Historical data found!")
            
            temp = weather_data['T2M_Celsius']
            wind = weather_data['wind_speed_10m']
            humidity = weather_data['QV2M']
            precip = weather_data['PRECTOT']
            
        else:
            st.sidebar.error("[ERROR] No data for this date/time")
            st.stop()
    
    else:  # Manual Input
        st.sidebar.markdown("### 🎛️ Weather Parameters")
        
        # temporarily hardcoded values for sliders
        temp = st.sidebar.slider("Temperature (°C)", -20.0, 50.0, 20.0, 0.5)
        wind = st.sidebar.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0, 0.5)
        humidity = st.sidebar.slider("Humidity (kg/kg)", 0.0, 0.03, 0.01, 0.001)
        precip = st.sidebar.slider("Precipitation (mm)", 0.0, 50.0, 0.0, 0.5)
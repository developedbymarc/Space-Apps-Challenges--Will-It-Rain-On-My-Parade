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
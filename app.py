from io import StringIO
import streamlit as st
import sqlite3
import pandas as pd
import numpy as np
import bnlearn as bn
from datetime import datetime, timedelta
import plotly.graph_objects as go
import plotly.express as px
from streamlit_folium import st_folium
import folium

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
    
    min_date = pd.to_datetime(dates.iloc[0]['min_date'])
    max_date = pd.to_datetime(dates.iloc[0]['max_date'])
    
    return min_date, max_date

def get_weather_data_for_date(target_date):
    """Get average weather data for a specific date"""
    conn = sqlite3.connect(db_path)
    
    # Format date as string for SQL query (YYYY-MM-DD)
    date_str = target_date.strftime('%Y-%m-%d')
    
    query = f"""
    SELECT AVG(T2M_Celsius) as T2M_Celsius,
           AVG(QV2M) as QV2M,
           AVG(wind_speed_10m) as wind_speed_10m,
           AVG(PRECTOT) as PRECTOT,
           COUNT(*) as count
    FROM weather_data
    WHERE timestamp = '{date_str}'
    """
    
    result = pd.read_sql_query(query, conn)
    conn.close()
    
    if result.iloc[0]['count'] > 0:
        return result.iloc[0]
    return None

def categorize_value(value, variable):
    """Convert raw value to category"""
    if variable == 'temperature':
        if value < 0: return 'freezing'
        elif value < 10: return 'cold'
        elif value < 20: return 'mild'
        elif value < 30: return 'warm'
        else: return 'hot'
    
    elif variable == 'wind':
        if value < 2: return 'calm'
        elif value < 5: return 'light'
        elif value < 10: return 'moderate'
        elif value < 15: return 'strong'
        else: return 'very_strong'
    
    elif variable == 'humidity':
        if value < 0.005: return 'low'
        elif value < 0.010: return 'moderate'
        elif value < 0.015: return 'high'
        else: return 'very_high'

def predict_weather(model, evidence):
    """Make prediction using Bayesian Network"""
    # Filter evidence to only include variables in the network
    network_vars = set()
    for edge in model['model_edges']:
        network_vars.add(edge[0])
        network_vars.add(edge[1])
    
    valid_evidence = {k: v for k, v in evidence.items() if k in network_vars}
    
    if not valid_evidence:
        valid_evidence = None
    
    # Perform inference
    query_result = bn.inference.fit(
        model,
        variables=['weather_condition'],
        evidence=valid_evidence,
        verbose=0
    )
    
    result_df = query_result.df if hasattr(query_result, 'df') else query_result
    return result_df.sort_values('p', ascending=False)


def find_best_days_in_range(model, target_date, preferred_condition, evidence, days_range=7, min_date=None, max_date=None):
    """Find best days within ±days_range that match preferred condition"""
    results = []
    
    for day_offset in range(-days_range, days_range + 1):
        check_date = target_date + timedelta(days=day_offset)
        
        # Check if this date is in historical range
        if min_date and max_date:
            is_historical = min_date.date() <= check_date <= max_date.date()
        else:
            is_historical = False
        
        if is_historical:
            # Use historical data
            weather_data = get_weather_data_for_date(check_date)
            
            if weather_data is None or weather_data['count'] == 0:
                continue
            
            # Categorize the historical data
            day_evidence = {
                'temp_category': categorize_value(weather_data['T2M_Celsius'], 'temperature'),
                'wind_category': categorize_value(weather_data['wind_speed_10m'], 'wind'),
                'humidity_category': categorize_value(weather_data['QV2M'], 'humidity')
            }
            
            temp = weather_data['T2M_Celsius']
            wind = weather_data['wind_speed_10m']
            humidity = weather_data['QV2M']
            precip = weather_data['PRECTOT']
            data_source = 'historical'
        else:
            # Use user's preferred conditions for future dates
            day_evidence = evidence
            temp = None  # Will be shown as "predicted based on your conditions"
            wind = None
            humidity = None
            precip = None
            data_source = 'predicted'
        
        # Get predictions
        predictions = predict_weather(model, day_evidence)
        
        # Find probability of preferred condition
        preferred_prob = predictions[predictions['weather_condition'] == preferred_condition]['p'].values
        
        if len(preferred_prob) > 0:
            results.append({
                'date': check_date,
                'probability': preferred_prob[0] * 100,
                'temp': temp,
                'wind': wind,
                'humidity': humidity,
                'precip': precip,
                'data_source': data_source
            })
    
    # Sort by probability
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('probability', ascending=False)
    
    return results_df


def create_probability_chart(predictions_df):
    """Create interactive probability bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=predictions_df['weather_condition'],
            y=predictions_df['p'] * 100,
            text=[f"{p:.1f}%" for p in predictions_df['p'] * 100],
            textposition='auto',
            marker=dict(
                color=predictions_df['p'] * 100,
                colorscale='RdYlGn',
                showscale=False
            )
        )
    ])
    
    fig.update_layout(
        title="Weather Condition Probabilities",
        xaxis_title="Weather Condition",
        yaxis_title="Probability (%)",
        height=400,
        yaxis=dict(range=[0, 100])
    )
    
    return fig

def create_best_days_chart(best_days_df, preferred_condition):
    """Create chart showing probabilities across date range"""
    fig = go.Figure()
    
    fig.add_trace(go.Scatter(
        x=best_days_df['date'],
        y=best_days_df['probability'],
        mode='lines+markers',
        name=f'{preferred_condition.capitalize()} Probability',
        line=dict(color='#2E86AB', width=3),
        marker=dict(size=10)
    ))
    
    # Highlight best day
    best_day = best_days_df.iloc[0]
    fig.add_trace(go.Scatter(
        x=[best_day['date']],
        y=[best_day['probability']],
        mode='markers',
        name='Best Day',
        marker=dict(size=20, color='#F24236', symbol='star')
    ))
    
    fig.update_layout(
        title=f"Probability of {preferred_condition.capitalize()} Weather (±7 Days)",
        xaxis_title="Date",
        yaxis_title="Probability (%)",
        height=400,
        hovermode='x unified'
    )
    
    return fig

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

    lat = 0
    lon = 0
    with st.sidebar:
        mp = folium.Map(location=[0, 0], zoom_start=12)
        mp.add_child(folium.LatLngPopup())
        out = st_folium(mp, width=500, height=500)
        if out and out.get("last_clicked"):
            lat = out["last_clicked"]["lat"]
            lon = out["last_clicked"]["lng"]
    
    # Date selection
    min_date, max_date = get_date_range()
    
    st.sidebar.markdown("### 📅 Date Selection")
    
    # Allow future dates for prediction
    today = datetime.now().date()
    max_future_date = today + timedelta(days=365 * 2)  # Allow 2 years into future
    
    selected_date = st.sidebar.date_input(
        "Select Date",
        value=today,
        min_value=min_date.date(),
        max_value=max_future_date,
        help="Select any date - historical data or future predictions"
    )
    
    # Check if selected date is in historical range
    is_historical = min_date.date() <= selected_date <= max_date.date()
    is_future = selected_date > max_date.date()
    
    if is_future:
        st.sidebar.info(f"🔮 Future prediction mode - {(selected_date - max_date.date()).days} days ahead")
    elif is_historical:
        st.sidebar.success(f"📊 Historical data available")
    
    # Analysis mode
    st.sidebar.markdown("---")
    analysis_mode = st.sidebar.radio(
        "Analysis Mode",
        ["Single Day Prediction", "Find Best Day (±7 days)"]
    )
    
    # Weather parameters input
    st.sidebar.markdown("### 🎛️ Preferred Weather Conditions")
    
    temp = st.sidebar.slider("Temperature (°C)", -20.0, 50.0, 20.0, 0.5)
    wind = st.sidebar.slider("Wind Speed (m/s)", 0.0, 30.0, 5.0, 0.5)
    humidity = st.sidebar.slider("Humidity (kg/kg)", 0.0, 0.03, 0.01, 0.001)
    
    # Categorize inputs
    evidence = {
        'temp_category': categorize_value(temp, 'temperature'),
        'wind_category': categorize_value(wind, 'wind'),
        'humidity_category': categorize_value(humidity, 'humidity')
    }
    
    # Get prediction for selected date
    with st.spinner("Analyzing weather data..."):
        predictions = predict_weather(model, evidence)
    
    most_likely_condition = predictions.iloc[0]['weather_condition']
    
    # Main content based on mode
    if analysis_mode == "Single Day Prediction":
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📊 Prediction Results")
            st.markdown(f"**Date:** {selected_date.strftime('%Y-%m-%d')}")
            st.markdown(f"**Location:** {lat:.4f}°, {lon:.4f}°")
            
            if is_future:
                st.warning("🔮 **Future Prediction** - Based on your preferred weather conditions")
            elif is_historical:
                st.success("📊 **Historical Data** - Based on actual recorded conditions")
            
            # Display input conditions
            st.markdown("### 🌡️ Preferred Conditions")
            
            metrics_col1, metrics_col2, metrics_col3 = st.columns(3)
            
            with metrics_col1:
                st.metric("Temperature", f"{temp:.1f}°C", delta=evidence['temp_category'])
            
            with metrics_col2:
                st.metric("Wind Speed", f"{wind:.1f} m/s", delta=evidence['wind_category'])
            
            with metrics_col3:
                st.metric("Humidity", f"{humidity:.4f} kg/kg", delta=evidence['humidity_category'])
            
            st.markdown("---")
            st.markdown("### 🎯 Weather Condition Probabilities")
            
            # Display probabilities
            for idx, row in predictions.iterrows():
                condition = row['weather_condition']
                prob = row['p'] * 100
                
                # Emoji mapping
                emoji_map = {
                    'clear': '☀️',
                    'hot': '🔥',
                    'windy': '💨',
                    'rainy': '🌧️',
                    'snowy': '❄️',
                    'freezing': '🥶'
                }
                
                emoji = emoji_map.get(condition, '🌤️')
                
                st.markdown(f"**{emoji} {condition.capitalize()}**")
                st.progress(prob / 100)
                st.markdown(f"**{prob:.1f}%**")
                st.markdown("")
            
            # Chart
            fig = create_probability_chart(predictions)
            st.plotly_chart(fig, use_container_width=True)
        
        with col2:
            st.header("ℹ️ Information")
            
            # Most likely condition
            st.markdown("### 🏆 Most Likely")
            st.info(f"**{most_likely_condition.capitalize()}**\n\n{predictions.iloc[0]['p']*100:.1f}% probability")
            
            if is_future:
                st.markdown("### 🔮 Prediction Mode")
                st.markdown("""
                This is a **future date prediction** based on:
                - Your preferred weather conditions
                - Historical weather patterns
                - Bayesian Network inference
                
                The prediction assumes conditions similar to your inputs.
                """)
            
            # Network info
            st.markdown("### 🧠 Model Info")
            st.markdown(f"""
            - **Model Type:** Bayesian Network
            - **Algorithm:** Hill Climbing
            - **Edges:** {len(model['model_edges'])}
            - **Training Data:** {len(historical_df):,} records
            - **Date Range:** {min_date.date()} to {max_date.date()}
            """)
    
    else:  # Find Best Day mode
        st.header("🔍 Finding Best Day for Your Preferred Weather")
        
        # Select preferred condition
        preferred_condition = st.selectbox(
            "What weather condition are you looking for?",
            options=predictions['weather_condition'].tolist(),
            index=0,
            help="Select the weather condition you prefer"
        )
        
        with st.spinner(f"Analyzing ±7 days for best {preferred_condition} weather..."):
            best_days = find_best_days_in_range(
                model, 
                selected_date, 
                preferred_condition, 
                evidence,
                days_range=7,
                min_date=min_date,
                max_date=max_date
            )
        
        if len(best_days) > 0:
            # Best day info
            best_day = best_days.iloc[0]
            
            col1, col2 = st.columns([2, 1])
            
            with col1:
                st.success(f"### 🏆 Best Day Found!")
                st.markdown(f"## 📅 {best_day['date'].strftime('%A, %B %d, %Y')}")
                st.markdown(f"### {best_day['probability']:.1f}% probability of {preferred_condition} weather")
                
                # Data source indicator
                if best_day['data_source'] == 'historical':
                    st.info("📊 Based on historical weather data")
                else:
                    st.warning("🔮 Based on future prediction with your preferred conditions")
                
                # Days difference
                days_diff = (best_day['date'] - selected_date).days
                if days_diff > 0:
                    st.info(f"ℹ️ This is {days_diff} days after your selected date")
                elif days_diff < 0:
                    st.info(f"ℹ️ This is {abs(days_diff)} days before your selected date")
                else:
                    st.info(f"ℹ️ This is your selected date!")
                
                # Expected conditions (only show if historical data available)
                if best_day['data_source'] == 'historical':
                    st.markdown("### 🌡️ Expected Conditions on Best Day")
                    
                    metrics_col1, metrics_col2, metrics_col3, metrics_col4 = st.columns(4)
                    
                    with metrics_col1:
                        st.metric("Temperature", f"{best_day['temp']:.1f}°C")
                    
                    with metrics_col2:
                        st.metric("Wind Speed", f"{best_day['wind']:.1f} m/s")
                    
                    with metrics_col3:
                        st.metric("Humidity", f"{best_day['humidity']:.4f}")
                    
                    with metrics_col4:
                        st.metric("Precipitation", f"{best_day['precip']:.2f} mm")
                else:
                    st.markdown("### 🌡️ Prediction Based On")
                    st.markdown(f"""
                    - Temperature: {temp:.1f}°C ({evidence['temp_category']})
                    - Wind Speed: {wind:.1f} m/s ({evidence['wind_category']})
                    - Humidity: {humidity:.4f} kg/kg ({evidence['humidity_category']})
                    """)
                
                # Chart showing all days
                st.markdown("---")
                fig = create_best_days_chart(best_days, preferred_condition)
                st.plotly_chart(fig, use_container_width=True)
            
            with col2:
                st.markdown("### 📋 All Days Ranked")
                
                for idx, row in best_days.head(7).iterrows():
                    date_str = row['date'].strftime('%b %d')
                    prob = row['probability']
                    source_emoji = "📊" if row['data_source'] == 'historical' else "🔮"
                    
                    # Medal emojis for top 3
                    if idx == 0:
                        medal = "🥇"
                    elif idx == 1:
                        medal = "🥈"
                    elif idx == 2:
                        medal = "🥉"
                    else:
                        medal = f"{idx + 1}."
                    
                    st.markdown(f"{medal} **{date_str}** {source_emoji} - {prob:.1f}%")
                    st.progress(prob / 100)
                    st.markdown("")
                
                st.markdown("---")
                st.markdown("### 💡 Legend")
                st.markdown("""
                - 📊 Historical data
                - 🔮 Future prediction
                """)
                st.info("Rankings show highest probability of your preferred weather condition.")
        
        else:
            st.error("No data available for analysis.")
    
    # Footer
    st.markdown("---")
    st.markdown("**NASA Space Apps Challenge 2025** | Built with Streamlit & bnlearn")

    if analysis_mode == "Single Day Prediction":
    # Convert predictions DataFrame to CSV
        csv = predictions.to_csv(index=False).encode('utf-8')
        st.sidebar.download_button(
            label="📥 Download Predictions (CSV)",
            data=csv,
            file_name=f"weather_predictions_{selected_date}.csv",
            mime="text/csv"
        )

    else:  # Best Day mode
        if len(best_days) > 0:
            csv = best_days.to_csv(index=False).encode('utf-8')
            st.sidebar.download_button(
                label="📥 Download Best Days Data (CSV)",
                data=csv,
                file_name=f"best_days_{selected_date}.csv",
                mime="text/csv"
        )

if __name__ == "__main__":
    main()
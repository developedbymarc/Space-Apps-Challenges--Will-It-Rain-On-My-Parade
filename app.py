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
import google.generativeai as genai

db_path = "large_merra2_data.db"

def day_to_date(day_str, year=2024):
    """Convert day_### string to actual date
    
    Args:
        day_str: String in format 'day_###' where ### is day of year (1-365)
        year: Year to use for the date (default: 2024)
    
    Returns:
        datetime object representing that day
    """
    day_num = int(day_str.split('_')[1])
    # Start from Jan 1 and add days
    base_date = datetime(year, 1, 1)
    target_date = base_date + timedelta(days=day_num - 1)
    return target_date

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
    
    # Extract day of year (1-366)
    # Normalize leap days: Feb 29 (day 60) becomes Feb 28, and shift all days after
    df['day_of_year'] = df['timestamp'].dt.dayofyear
    df['is_leap_year'] = df['timestamp'].dt.is_leap_year
    
    # Adjust for leap years: if it's after Feb 29 in a leap year, subtract 1
    df['day_of_year_normalized'] = df.apply(
        lambda row: row['day_of_year'] - 1 if row['is_leap_year'] and row['day_of_year'] > 60 else row['day_of_year'],
        axis=1
    )
    
    # Now we have days 1-365 consistently
    # Convert to string categories for BN (day_001, day_002, ..., day_365)
    df['day_category'] = df['day_of_year_normalized'].apply(lambda x: f'day_{int(x):03d}')
    
    # Each unique lat/long gets its own bin
    # Round to 2 decimal places for consistency and convert to string categories
    df['lat_category'] = df['latitude'].round(2).apply(lambda x: f'lat_{x:.2f}')
    df['lon_category'] = df['longitude'].round(2).apply(lambda x: f'lon_{x:.2f}')
    
    # Engineer weather features
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
    
    # Prepare for Bayesian Network
    # Include location and time as evidence variables
    bn_features = [
        'day_category',
        'lat_category',
        'lon_category',
        'temp_category',
        'wind_category',
        'precip_category',
        'snow_category',
        'humidity_category',
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

def categorize_day_of_year(day_of_year, is_leap_year=False):
    """Convert day of year to normalized day category (handling leap years)
    
    Leap year handling: Feb 29 (day 60) is treated as Feb 28,
    and all days after are shifted back by 1 to maintain consistency.
    This gives us 365 consistent day bins across all years.
    """
    if is_leap_year and day_of_year > 60:
        day_of_year = day_of_year - 1
    
    # Ensure we're in range 1-365
    day_of_year = max(1, min(365, day_of_year))
    
    return f'day_{int(day_of_year):03d}'

def categorize_latitude(lat):
    """Convert latitude to category bin
    
    Each unique latitude value gets its own bin.
    Rounded to 2 decimal places for consistency.
    """
    return f'lat_{lat:.2f}'

def categorize_longitude(lon):
    """Convert longitude to category bin
    
    Each unique longitude value gets its own bin.
    Rounded to 2 decimal places for consistency.
    """
    return f'lon_{lon:.2f}'

def predict_weather(model, evidence, target_variables):
    """Make prediction using Bayesian Network
    
    Args:
        model: Trained Bayesian Network
        evidence: Dictionary of evidence variables (day, lat, lon)
        target_variables: List of weather variables to predict
    
    Returns:
        Dictionary with predictions for each variable, or error message
    """
    # Filter evidence to only include variables in the network
    network_vars = set()
    for edge in model['model_edges']:
        network_vars.add(edge[0])
        network_vars.add(edge[1])
    
    valid_evidence = {k: v for k, v in evidence.items() if k in network_vars}
    
    if not valid_evidence:
        return None
    
    # Filter target variables to only include those in the network
    valid_targets = [v for v in target_variables if v in network_vars]
    
    if not valid_targets:
        return None
    
    # Perform inference for each target variable
    results = {}
    try:
        for target in valid_targets:
            query_result = bn.inference.fit(
                model,
                variables=[target],
                evidence=valid_evidence,
                verbose=0
            )
            
            result_df = query_result.df if hasattr(query_result, 'df') else query_result
            results[target] = result_df.sort_values('p', ascending=False)
        
        return results
    except KeyError as e:
        # This happens when the evidence combination doesn't exist in training data
        return {"error": "location_not_in_data", "message": str(e)}

def create_predictions_dataframe(predictions, evidence, selected_date, lat, lon):
    """Create a comprehensive DataFrame from predictions for CSV export"""
    
    # Create base info
    export_data = []
    
    # Get top prediction for each variable
    for var_name, pred_df in predictions.items():
        top_pred = pred_df.iloc[0]
        top_category = top_pred[var_name]
        top_prob = top_pred['p']
        
        export_data.append({
            'date': selected_date,
            'latitude': lat,
            'longitude': lon,
            'day_category': evidence['day_category'],
            'lat_category': evidence['lat_category'],
            'lon_category': evidence['lon_category'],
            'variable': var_name,
            'predicted_category': top_category,
            'probability': top_prob
        })
    
    df = pd.DataFrame(export_data)
    
    # Calculate joint probability (multiply all top predictions)
    joint_probability = df['probability'].prod()
    
    # Add joint probability row
    joint_row = {
        'date': selected_date,
        'latitude': lat,
        'longitude': lon,
        'day_category': evidence['day_category'],
        'lat_category': evidence['lat_category'],
        'lon_category': evidence['lon_category'],
        'variable': 'JOINT_PROBABILITY',
        'predicted_category': 'all_conditions_together',
        'probability': joint_probability
    }
    
    df = pd.concat([df, pd.DataFrame([joint_row])], ignore_index=True)
    
    return df, joint_probability

def create_detailed_predictions_dataframe(predictions, evidence, selected_date, lat, lon):
    """Create a detailed DataFrame with ALL probabilities for each variable"""
    
    export_data = []
    
    # Add all predictions for each variable
    for var_name, pred_df in predictions.items():
        for idx, row in pred_df.iterrows():
            export_data.append({
                'date': selected_date,
                'latitude': lat,
                'longitude': lon,
                'day_category': evidence['day_category'],
                'lat_category': evidence['lat_category'],
                'lon_category': evidence['lon_category'],
                'variable': var_name,
                'category': row[var_name],
                'probability': row['p'],
                'rank': idx + 1
            })
    
    return pd.DataFrame(export_data)

def find_best_days_in_range(model, evidence: dict, target_variables, preferred_categories, days_range=7, min_date=None, max_date=None):
    """Find best days within +/- days_range that match preferred condition"""
    results = []
    
    # evidence = {
    #     'day_category': categorize_day_of_year(day_of_year, is_leap_year),
    #     'lat_category': categorize_latitude(lat),
    #     'lon_category': categorize_longitude(lon)
    # }
    # day_category format : day_###, we extract the last 3 digits
    target_date = int(evidence['day_category'][-3:])

    def day_offsetter(day, offset):
        """Formula to calculate offsets for the days (days value range: 1 to 365)"""
        return (((day - 1 + offset) % 365) + 365) % 365 + 1

    results = {
        'probability': [],
        'day': []
    }

    for day_offset in range(-days_range, days_range + 1):
        offset_date = day_offsetter(target_date, day_offset)
        offset_date = "day_" + f"{offset_date}".zfill(3)

        day_evidence = evidence.copy()
        day_evidence['day_category'] = offset_date
        
        # Get predictions
        predictions = predict_weather(model, day_evidence, target_variables)
        
        print(results)
        results['probability'].append(1)
        results['day'].append(offset_date)
        for category in preferred_categories:
            # voodoo magic stuff.. i really donno what the hell i did here but it works. dont touch pls
            print(preferred_categories[category], predictions[category][predictions[category][category] == preferred_categories[category]])
            results['probability'][day_offset + days_range] *= predictions[category][predictions[category][category] == preferred_categories[category]]['p'].values[0]
    
    # Sort by probability
    results_df = pd.DataFrame(results)
    if len(results_df) > 0:
        results_df = results_df.sort_values('probability', ascending=False)
    
    print(results_df)
    return results_df

def create_probability_chart(predictions_df, variable_name):
    """Create interactive probability bar chart"""
    fig = go.Figure(data=[
        go.Bar(
            x=predictions_df[variable_name],
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
        title=f"{variable_name.replace('_', ' ').title()} Probabilities",
        xaxis_title=variable_name.replace('_', ' ').title(),
        yaxis_title="Probability (%)",
        height=350,
        yaxis=dict(range=[0, 100])
    )
    
    return fig
    
def categorize_value(value, variable):
    """
    Convert raw value to category

    ### Params
    - `value` - raw value
    - `variable` - 'temp' | 'wind' | 'humidity' | 'prectot' | 'precsno'
    """
    if variable == 'temp':
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

    elif variable == 'prectot':
        if value < 0.1: return 'none'
        elif value < 2.5: return 'light'
        elif value < 10: return 'moderate'
        elif value < 50: return 'heavy'
        else: return 'very_heavy'
    
    elif variable == 'precsno':
        if value < 0.01: return 'none'
        elif value < 1: return 'light'
        elif value < 5: return 'moderate'
        elif value < 25: return 'heavy'
        else: return 'very_heavy'


def generate_weather_summary(predictions, joint_prob, temp_cat, wind_cat, precip_cat, snow_cat, humidity_cat, location, date):
    """Generate friendly weather summary using Gemini API"""
    try:
        genai.configure(api_key=st.secrets["GEMINI_API_KEY"])
        model = genai.GenerativeModel('gemini-2.5-flash-lite')
        
        prompt = f"""You are a friendly weather assistant. Create a brief, conversational summary (2-3 sentences) of this weather prediction:

Date: {date}
Location: {location}

Predicted conditions:
- Temperature: {temp_cat}
- Wind: {wind_cat}
- Precipitation: {precip_cat}
- Snow: {snow_cat}
- Humidity: {humidity_cat}
- Overall confidence: {joint_prob:.1f}%

Make it sound natural and helpful, like talking to a friend about the weather."""
        
        response = model.generate_content(prompt)
        return response.text
    except Exception as e:
        return f"Unable to generate summary: {str(e)}"
# Main App
def main():
    # Header
    st.title("🌤️ Weather Prediction System")
    st.markdown("### NASA MERRA-2 Dataset - Bayesian Network Predictions")
    st.markdown("Predict weather conditions given **location** and **time**")
    st.markdown("---")
    
    # Load model
    with st.spinner("Loading model... This may take a moment on first run."):
        model, historical_df = load_and_train_model()
    
    # Sidebar - Input Section
    st.sidebar.header("📍 Input Parameters")
    
    st.sidebar.markdown("### 🌍 Location")
    
    # Get available locations
    locations = get_available_locations()
    
    # Dropdown selector for reliable selection
    location_idx = st.sidebar.selectbox(
        "Select Location from Database",
        range(len(locations)),
        format_func=lambda i: f"{locations.iloc[i]['latitude']:.2f}°, {locations.iloc[i]['longitude']:.2f}°"
    )
    
    lat = locations.iloc[location_idx]['latitude']
    lon = locations.iloc[location_idx]['longitude']
    
    # Display map showing selected location
    st.sidebar.markdown("**Selected Location:**")
    
    # Create folium map centered on selected location
    m = folium.Map(location=[lat, lon], zoom_start=8)
    
    # Add marker for selected location
    folium.Marker(
        location=[lat, lon],
        popup=f"Selected: {lat:.2f}°, {lon:.2f}°",
        tooltip="Selected Location",
        icon=folium.Icon(color='red', icon='info-sign')
    ).add_to(m)
    
    # Add all other available locations
    for idx, row in locations.iterrows():
        if idx != location_idx:
            folium.CircleMarker(
                location=[row['latitude'], row['longitude']],
                radius=4,
                color='blue',
                fill=True,
                fillColor='lightblue',
                fillOpacity=0.6,
                popup=f"Lat: {row['latitude']:.2f}, Lon: {row['longitude']:.2f}"
            ).add_to(m)
    
    # Display map in sidebar
    st_folium(m, width=300, height=300)
    
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
        help="Select any date - predictions based on day of year"
    )
    
    # Calculate day of year
    selected_datetime = datetime(selected_date.year, selected_date.month, selected_date.day)
    day_of_year = selected_datetime.timetuple().tm_yday
    is_leap_year = selected_datetime.year % 4 == 0 and (selected_datetime.year % 100 != 0 or selected_datetime.year % 400 == 0)
    
    # Show original and normalized day
    normalized_day = day_of_year - 1 if is_leap_year and day_of_year > 60 else day_of_year
    
    if is_leap_year and selected_date.month == 2 and selected_date.day == 29:
        st.sidebar.info(f"Day of year: {day_of_year} → {normalized_day} (Feb 29 → Feb 28)")
    elif is_leap_year and day_of_year > 60:
        st.sidebar.info(f"Day of year: {day_of_year} → {normalized_day} (leap year adjusted)")
    else:
        st.sidebar.info(f"Day of year: {day_of_year}")
    
    # Check if selected date is in historical range
    is_historical = min_date.date() <= selected_date <= max_date.date()
    is_future = selected_date > max_date.date()
    
    if is_future:
        st.sidebar.info(f"🔮 Future prediction mode")
    elif is_historical:
        st.sidebar.success(f"📊 Can compare with historical data")
    
    # Weather parameters input
    st.sidebar.markdown("### 🎛️ Preferred Weather Conditions")

    temp_category = st.sidebar.selectbox("Temperature Category", ['freezing', 'cold', 'mild', 'warm', 'hot'])
    wind_category = st.sidebar.selectbox("Windspeed Category", ['calm', 'light', 'moderate', 'strong', 'very_strong'])
    humd_category = st.sidebar.selectbox("Humidity Category", ['low', 'moderate', 'high', 'very_high'])
    rain_category = st.sidebar.selectbox("Rain Category", ['none', 'light', 'moderate', 'heavy', 'very_heavy'])
    snow_category = st.sidebar.selectbox("Snow Category", ['none', 'light', 'moderate', 'heavy'])

    # Build evidence dictionary
    evidence = {
        'day_category': categorize_day_of_year(day_of_year, is_leap_year),
        'lat_category': categorize_latitude(lat),
        'lon_category': categorize_longitude(lon)
    }
    
    # Define target weather variables to predict
    target_variables = [
        'temp_category',
        'wind_category',
        'precip_category',
        'snow_category',
        'humidity_category'
    ]

    # Build preferred_conditions dictionary
    preferred_categories = {
        'temp_category': temp_category,
        'wind_category': wind_category,
        'precip_category': rain_category,
        'snow_category': snow_category,
        'humidity_category': humd_category
    }

    # Best days search range
    st.sidebar.markdown("### 🔍 Best Days Search")
    days_range = st.sidebar.slider(
        "Search Range (± days)", 
        min_value=1, 
        max_value=30, 
        value=7,
        help="Search for best days within this range of the selected date"
    )
    
    # Get predictions
    st.sidebar.markdown("---")
    if st.sidebar.button("🔮 Predict Weather", type="primary"):
        with st.spinner("Analyzing weather patterns..."):
            predictions = predict_weather(model, evidence, target_variables)

        if predictions is None:
            st.error("Unable to make predictions. Please check the model structure.")
            return
        
        # Check if there was an error (location not in training data)
        if isinstance(predictions, dict) and "error" in predictions:
            st.error("⚠️ No data available for this location")
            st.warning(f"""
            **The selected location ({lat:.2f}°, {lon:.2f}°) was not in the training dataset.**
            
            Please select one of the blue markers on the map, which represent locations with historical data.
            
            Available locations have data from {min_date.date()} to {max_date.date()}.
            """)
            return
        
        # Create DataFrames for export
        summary_df, joint_prob = create_predictions_dataframe(predictions, evidence, selected_date, lat, lon)
        detailed_df = create_detailed_predictions_dataframe(predictions, evidence, selected_date, lat, lon)
        
        # Find best days (automatically run this)
        with st.spinner("Finding best days for your preferred conditions..."):
            best_days_df = find_best_days_in_range(
                model, 
                evidence, 
                target_variables, 
                preferred_categories,
                days_range=days_range
            )
        
        # ===== DISPLAY BEST DAYS SECTION =====
        st.header("🏆 Best Days for Your Preferred Conditions")
        
        st.markdown(f"""
        **Searching around:** {selected_date.strftime('%B %d, %Y')}  
        **Range:** ± {days_range} days  
        **Location:** {lat:.2f}°, {lon:.2f}°
        """)
        
        if best_days_df is not None and len(best_days_df) > 0:
            # Show preferred conditions at the top
            st.markdown("### 🎯 Your Preferred Conditions")
            pref_cols = st.columns(5)
            
            with pref_cols[0]:
                st.metric("Temperature", temp_category.replace('_', ' ').title())
            with pref_cols[1]:
                st.metric("Wind", wind_category.replace('_', ' ').title())
            with pref_cols[2]:
                st.metric("Humidity", humd_category.replace('_', ' ').title())
            with pref_cols[3]:
                st.metric("Rain", rain_category.replace('_', ' ').title())
            with pref_cols[4]:
                st.metric("Snow", snow_category.replace('_', ' ').title())
            
            st.markdown("---")
            st.markdown("### 📅 Top 5 Best Days")
            
            # Display top 5 results
            top_5 = best_days_df.head(5)
            
            for idx, row in top_5.iterrows():
                day_str = row['day']
                prob = row['probability'] * 100
                
                # Convert day_### to actual date
                actual_date = day_to_date(day_str, selected_date.year)
                
                # Calculate days difference
                days_diff = (actual_date.date() - selected_date).days
                
                # Create color coding based on probability
                if prob >= 50:
                    color = "🟢"
                elif prob >= 30:
                    color = "🟡"
                elif prob >= 10:
                    color = "🟠"
                else:
                    color = "🔴"
                
                # Highlight if it's the selected day
                is_selected = days_diff == 0
                
                # Display result
                with st.container():
                    col1, col2, col3 = st.columns([3, 1, 1])
                    
                    with col1:
                        date_str = actual_date.strftime('%A, %B %d, %Y')
                        if is_selected:
                            st.markdown(f"### {color} {date_str} ⭐ **SELECTED**")
                        else:
                            st.markdown(f"### {color} {date_str}")
                    
                    with col2:
                        if days_diff == 0:
                            st.markdown("**Today's Date**")
                        elif days_diff > 0:
                            st.markdown(f"**+{days_diff} days**")
                        else:
                            st.markdown(f"**{days_diff} days**")
                    
                    with col3:
                        st.metric("Match", f"{prob:.1f}%")
                    
                    st.progress(prob / 100)
                    st.markdown("---")
            
            # Show visualization
            st.markdown("### 📈 Probability Timeline")
            
            # Create timeline chart
            chart_df = best_days_df.copy()
            chart_df['date'] = chart_df['day'].apply(lambda x: day_to_date(x, selected_date.year))
            chart_df['probability_pct'] = chart_df['probability'] * 100
            
            # Sort by date to ensure proper line plotting
            chart_df = chart_df.sort_values('date').reset_index(drop=True)
            
            # Mark which row is the selected date
            selected_dt = datetime(selected_date.year, selected_date.month, selected_date.day)
            chart_df['is_selected'] = chart_df['date'].apply(lambda x: x.date() == selected_date)
            
            fig = px.line(
                chart_df, 
                x='date', 
                y='probability_pct',
                markers=True,
                title="Match Probability Over Time"
            )
            
            # Add a red marker for the selected date
            selected_row = chart_df[chart_df['is_selected']]
            if not selected_row.empty:
                fig.add_scatter(
                    x=selected_row['date'],
                    y=selected_row['probability_pct'],
                    mode='markers',
                    marker=dict(size=15, color='red', symbol='star'),
                    name='Selected Date',
                    showlegend=True
                )
            
            fig.update_layout(
                xaxis_title="Date",
                yaxis_title="Match Probability (%)",
                hovermode='x unified',
                height=400
            )
            
            st.plotly_chart(fig, use_container_width=True)
            
            # Show all results in expandable section
            with st.expander("📊 View All Results"):
                # Create display dataframe
                display_df = best_days_df.copy()
                display_df['date'] = display_df['day'].apply(lambda x: day_to_date(x, selected_date.year))
                display_df['date_formatted'] = display_df['date'].apply(lambda x: x.strftime('%Y-%m-%d (%A)'))
                display_df['probability_pct'] = (display_df['probability'] * 100).round(2)
                display_df['days_offset'] = display_df['date'].apply(lambda x: (x.date() - selected_date).days)
                
                # Select and rename columns for display
                display_cols = display_df[['date_formatted', 'days_offset', 'probability_pct']].copy()
                display_cols.columns = ['Date', 'Days from Selected', 'Match Probability (%)']
                
                st.dataframe(display_cols, use_container_width=True, hide_index=True)
                
                # Download button
                csv = display_df[['date_formatted', 'day', 'probability', 'days_offset']].to_csv(index=False).encode('utf-8')
                st.download_button(
                    label="📥 Download Best Days (CSV)",
                    data=csv,
                    file_name=f"best_days_{selected_date.strftime('%Y%m%d')}_lat{lat:.2f}_lon{lon:.2f}.csv",
                    mime="text/csv"
                )
        else:
            st.warning("No results found for the best days analysis.")
        
        st.markdown("---")
        st.markdown("---")
        
        # ===== ORIGINAL PREDICTIONS SECTION =====
        # Main content
        col1, col2 = st.columns([2, 1])
        
        with col1:
            st.header("📊 Weather Predictions")
            st.markdown(f"**Date:** {selected_date.strftime('%A, %B %d, %Y')} (Day {day_of_year})")
            st.markdown(f"**Location:** {lat:.4f}°, {lon:.4f}°")
            
            # Display joint probability prominently
            st.markdown("---")
            st.markdown("### 🎲 Joint Probability")
            st.info(f"**Probability of all top predictions occurring together: {joint_prob*100:.2f}%**")
            st.caption("This is calculated by multiplying the probabilities of the most likely category for each weather variable.")
            
            st.markdown("---")
            st.markdown("### 🎯 Predicted Weather Conditions")
            st.markdown("**Given:** Location and Time of Year")

            # AI Summary
            st.markdown("### 🤖 AI Weather Summary")
            with st.spinner("Generating friendly summary..."):
                summary = generate_weather_summary(
                    predictions,
                    joint_prob,
                    list(predictions['temp_category']['temp_category'])[0],
                    list(predictions['wind_category']['wind_category'])[0],
                    list(predictions['precip_category']['precip_category'])[0],
                    list(predictions['snow_category']['snow_category'])[0],
                    list(predictions['humidity_category']['humidity_category'])[0],
                    f"{lat:.2f}°, {lon:.2f}°",
                    selected_date.strftime('%A, %B %d, %Y')
                )
            st.info(summary)
            st.markdown("---")
                
            # Display predictions for each variable
            for var_name, pred_df in predictions.items():
                st.markdown(f"#### {var_name.replace('_', ' ').title()}")
                
                # Show top prediction
                top_pred = pred_df.iloc[0]
                top_category = top_pred[var_name]
                top_prob = top_pred['p'] * 100
                
                # Emoji mapping
                emoji_map = {
                    'temp_category': {'freezing': '🥶', 'cold': '❄️', 'mild': '🌤️', 'warm': '☀️', 'hot': '🔥'},
                    'wind_category': {'calm': '🍃', 'light': '💨', 'moderate': '🌬️', 'strong': '🌪️', 'very_strong': '🌀'},
                    'precip_category': {'none': '☀️', 'light': '🌦️', 'moderate': '🌧️', 'heavy': '⛈️', 'very_heavy': '🌊'},
                    'snow_category': {'none': '🌤️', 'light': '🌨️', 'moderate': '❄️', 'heavy': '⛄'},
                    'humidity_category': {'low': '🏜️', 'moderate': '🌤️', 'high': '💧', 'very_high': '💦'}
                }
                
                emoji = emoji_map.get(var_name, {}).get(top_category, '🌤️')
                
                st.markdown(f"**Most Likely:** {emoji} {top_category.replace('_', ' ').title()} ({top_prob:.1f}%)")
                
                # Show all probabilities
                for idx, row in pred_df.iterrows():
                    category = row[var_name]
                    prob = row['p'] * 100
                    
                    st.progress(prob / 100, text=f"{category.replace('_', ' ').title()}: {prob:.1f}%")
                
                # Chart
                fig = create_probability_chart(pred_df, var_name)
                st.plotly_chart(fig, use_container_width=True)
                
                st.markdown("---")
        
        with col2:
            st.header("ℹ️ Prediction Info")
            
            # Evidence used
            st.markdown("### 📌 Given Evidence")
            st.markdown(f"""
            - **Day of Year:** {normalized_day} ({selected_date.strftime('%B %d')})
            - **Latitude:** {lat:.2f}°
            - **Longitude:** {lon:.2f}°
            """)
            
            if is_leap_year:
                st.markdown("**Note:** Leap year adjustment applied for consistency")
            
            st.markdown("### 🧠 Model Structure")
            st.markdown("""
            The Bayesian Network predicts weather conditions based on:
            
            **P(Weather | Day, Latitude, Longitude)**
            
            This gives us the probability of specific weather conditions given the location and time of year.
            """)
            
            # Network info
            st.markdown("### 📊 Model Details")
            st.markdown(f"""
            - **Model Type:** Bayesian Network
            - **Algorithm:** Hill Climbing
            - **Edges:** {len(model['model_edges'])}
            - **Training Data:** {len(historical_df):,} records
            - **Date Range:** {min_date.date()} to {max_date.date()}
            """)
            
            st.markdown("### 🎯 Predicted Variables")
            st.markdown("""
            - Temperature Category
            - Wind Speed Category
            - Precipitation Category
            - Snow Category
            - Humidity Category
            """)
            
            # Download section
            st.markdown("---")
            st.markdown("### 📥 Download Predictions")
            
            # Summary CSV (top predictions + joint probability)
            summary_csv = summary_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📄 Download Summary (CSV)",
                data=summary_csv,
                file_name=f"weather_predictions_summary_{selected_date.strftime('%Y%m%d')}_lat{lat:.2f}_lon{lon:.2f}.csv",
                mime="text/csv",
                help="Download top predictions for each variable plus joint probability"
            )
            
            # Detailed CSV (all probabilities)
            detailed_csv = detailed_df.to_csv(index=False).encode('utf-8')
            st.download_button(
                label="📊 Download Detailed (CSV)",
                data=detailed_csv,
                file_name=f"weather_predictions_detailed_{selected_date.strftime('%Y%m%d')}_lat{lat:.2f}_lon{lon:.2f}.csv",
                mime="text/csv",
                help="Download all probabilities for all categories"
            )
    
    else:
        # Initial view
        st.info("👈 Click **Predict Weather** in the sidebar to get predictions for the selected location and date")
        
        st.markdown("### 🧠 How It Works")
        st.markdown("""
        This weather prediction system uses a **Bayesian Network** to model relationships between:
        
        1. **Location** (latitude, longitude)
        2. **Time** (day of year)
        3. **Weather Conditions** (temperature, wind, precipitation, snow, humidity)
        
        The model calculates: **P(Weather Conditions | Location, Time)**
        
        This allows you to predict what weather conditions are most likely for any location and date based on historical patterns.
        
        ### 🎲 Joint Probability
        
        The system calculates the **joint probability** by multiplying together the probabilities of the most likely category for each weather variable. This gives you the probability that all these conditions occur together.
        
        For example:
        - Temperature: Warm (60%)
        - Wind: Light (70%)
        - Precipitation: None (80%)
        - Snow: None (95%)
        - Humidity: Moderate (65%)
        
        **Joint Probability = 0.60 × 0.70 × 0.80 × 0.95 × 0.65 = 20.8%**
        """)
        
        # Show network structure
        st.markdown("### 🔗 Network Structure")
        if len(model['model_edges']) > 0:
            edges_df = pd.DataFrame(model['model_edges'], columns=['From', 'To'])
            st.dataframe(edges_df, use_container_width=True)
    
    # Footer
    st.markdown("---")
    st.markdown("**NASA Space Apps Challenge 2025** | Built with Streamlit & bnlearn")

if __name__ == "__main__":
    main()
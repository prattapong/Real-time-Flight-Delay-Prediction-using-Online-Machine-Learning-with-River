import json
import streamlit as st
from confluent_kafka import Consumer, KafkaException, KafkaError
from river import preprocessing, tree, linear_model
from datetime import datetime, timedelta
import pandas as pd
import altair as alt
import time
import numpy as np
from typing import Dict, Optional, List, Tuple

pd.set_option('display.max_columns', None)

# Page config
st.set_page_config(
    page_title="✈️ Flight Delay Predictor",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for modern design
st.markdown("""
<style>
    /* Main theme colors */
    :root {
        --primary-color: #FF6B6B;
        --secondary-color: #4ECDC4;
        --accent-color: #45B7D1;
        --success-color: #96CEB4;
        --warning-color: #FECA57;
        --danger-color: #FF9FF3;
    }
    
    /* Hide default streamlit elements */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Custom header */
    .main-header {
        background: linear-gradient(90deg, #667eea 0%, #764ba2 100%);
        padding: 2rem 0;
        margin: -1rem -1rem 2rem -1rem;
        border-radius: 0 0 20px 20px;
        text-align: center;
        color: white;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    
    .main-header h1 {
        font-size: 3rem;
        margin: 0;
        font-weight: 700;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.3);
    }
    
    .main-header p {
        font-size: 1.2rem;
        margin: 0.5rem 0 0 0;
        opacity: 0.9;
    }
    
    /* Metric cards */
    .metric-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        border: 1px solid #f0f0f0;
        text-align: center;
        margin: 0.5rem 0;
        transition: transform 0.2s ease;
    }
    
    .metric-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 20px rgba(0,0,0,0.15);
    }
    
    .metric-value {
        font-size: 2.5rem;
        font-weight: bold;
        color: #2c3e50;
        margin: 0;
    }
    
    .metric-label {
        font-size: 0.9rem;
        color: #7f8c8d;
        margin: 0.5rem 0 0 0;
        text-transform: uppercase;
        letter-spacing: 1px;
    }
    
    .metric-delta {
        font-size: 0.8rem;
        margin-top: 0.25rem;
    }
    
    .metric-delta.positive {
        color: #27ae60;
    }
    
    .metric-delta.negative {
        color: #e74c3c;
    }
    
    /* Status indicators */
    .status-indicator {
        display: inline-block;
        width: 12px;
        height: 12px;
        border-radius: 50%;
        margin-right: 8px;
    }
    
    .status-connected {
        background-color: #27ae60;
        animation: pulse 2s infinite;
    }
    
    .status-disconnected {
        background-color: #e74c3c;
    }
    
    @keyframes pulse {
        0% { opacity: 1; }
        50% { opacity: 0.5; }
        100% { opacity: 1; }
    }
    
    /* Section headers */
    .section-header {
        background: linear-gradient(90deg, #f093fb 0%, #f5576c 100%);
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin: 2rem 0 1rem 0;
        font-weight: 600;
        text-align: center;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Alert styles */
    .alert-success {
        background-color: #d4edda;
        border: 1px solid #c3e6cb;
        color: #155724;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .alert-warning {
        background-color: #fff3cd;
        border: 1px solid #ffeaa7;
        color: #856404;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    .alert-danger {
        background-color: #f8d7da;
        border: 1px solid #f5c6cb;
        color: #721c24;
        padding: 1rem;
        border-radius: 10px;
        margin: 1rem 0;
    }
    
    /* Custom dataframe styling */
    .dataframe {
        border-radius: 10px;
        overflow: hidden;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: linear-gradient(180deg, #667eea 0%, #764ba2 100%);
    }
    
    /* Chart containers */
    .chart-container {
        background: white;
        padding: 1rem;
        border-radius: 15px;
        box-shadow: 0 2px 10px rgba(0,0,0,0.1);
        margin: 1rem 0;
    }
    
    /* Flight status badges */
    .status-badge {
        padding: 0.25rem 0.75rem;
        border-radius: 20px;
        font-size: 0.85rem;
        font-weight: 600;
        display: inline-block;
    }
    
    .status-ontime {
        background-color: #d4edda;
        color: #155724;
    }
    
    .status-delayed {
        background-color: #f8d7da;
        color: #721c24;
    }
    
    .status-predicted {
        background-color: #cce5ff;
        color: #004085;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'data' not in st.session_state:
    st.session_state.data = []
if 'model_initialized' not in st.session_state:
    st.session_state.model_initialized = False
if 'kafka_connected' not in st.session_state:
    st.session_state.kafka_connected = False
if 'processed_count' not in st.session_state:
    st.session_state.processed_count = 0
if 'model_version' not in st.session_state:
    st.session_state.model_version = 0

# Setup model
@st.cache_resource
def initialize_model():
    """Initialize the online learning model pipeline"""
    return (
        preprocessing.OneHotEncoder() |
        preprocessing.StandardScaler() |
        linear_model.LogisticRegression()
        # tree.HoeffdingTreeClassifier(
        #     grace_period=50,
        #     # split_confidence=0.01,
        #     leaf_prediction='nb'
        # )
    )

model = initialize_model()

def extract_features(message_json: Dict) -> Dict:
    """Extract features from flight message for ML model"""
    dep = message_json.get("departure", {})
    arr = message_json.get("arrival", {})
    airline = message_json.get("airline", {})
    
    # Extract hour from scheduled departure time
    try:
        dep_scheduled = dep.get("scheduled", "")
        if dep_scheduled:
            dep_time = datetime.fromisoformat(dep_scheduled.replace("Z", "+00:00"))
            hour = dep_time.hour
            day_of_week = dep_time.weekday()
        else:
            hour = 0
            day_of_week = 0
    except:
        hour = 0
        day_of_week = 0
    
    # Extract features with proper null handling
    features = {
        "dep_airport": dep.get("iata", "UNK") or "UNK",
        "arr_airport": arr.get("iata", "UNK") or "UNK",
        "dep_terminal": str(dep.get("terminal", "UNK") or "UNK"),
        "arr_terminal": str(arr.get("terminal", "UNK") or "UNK"),
        "airline": airline.get("name", "UNK") or "UNK",
        "hour": hour,
        "day_of_week": day_of_week,
        "route": f"{dep.get('iata', 'UNK')}_{arr.get('iata', 'UNK')}"
    }
    
    return features

def extract_label(message_json: Dict) -> int:
    """Extract label (delayed or not) from flight message"""
    arr_delay = message_json.get("arrival", {}).get("delay")
    
    # Flight is delayed if either departure or arrival has delay
    return 1 if arr_delay is not None and arr_delay > 0 else 0

def process_flight_message(flight_data: Dict) -> Dict:
    """Process a single flight message and return formatted row data"""
    dep = flight_data.get("departure", {})
    arr = flight_data.get("arrival", {})
    airline = flight_data.get("airline", {})
    
    # Extract features for prediction
    features = extract_features(flight_data)
    
    # Make prediction
    y_pred_proba = model.predict_proba_one(features)
    y_pred = y_pred_proba.get(1, 0.0) if y_pred_proba else 0.0
    
    # Extract actual label if flight has already occurred
    actual_dep = dep.get("actual")
    actual_arr = arr.get("actual")
    estimated_arr = arr.get("estimated")
    
    # Determine if we should show prediction (only for future flights)
    flight_has_occurred = actual_dep is not None or actual_arr is not None or estimated_arr is not None
    
    # If flight has occurred, train the model
    if flight_has_occurred:
        y_actual = extract_label(flight_data)
        model.learn_one(features, y_actual)
        st.session_state.model_version += 1
    
    # Calculate delay time
    dep_delay = dep.get("delay", 0)
    arr_delay = arr.get("delay", 0)
    # delay_time = max(dep_delay or 0, arr_delay or 0) if (dep_delay or arr_delay) else None
    delay_time = arr_delay
    
    # Create row for dataframe
    row = {
        "flight_number": flight_data.get("flight", {}).get("iata", "N/A"),
        "airline": airline.get("name", "Unknown"),
        "departure_airport": dep.get("iata", "N/A"),
        "arrival_airport": arr.get("iata", "N/A"),
        "scheduled_departure": dep.get("scheduled"),
        "scheduled_arrival": arr.get("scheduled"),
        "actual_departure": actual_dep,
        "estimated_arrival": arr.get("estimated"),
        "actual_arrival": actual_arr,
        "departure_terminal": dep.get("terminal"),
        "arrival_terminal": arr.get("terminal"),
        "delay_time": delay_time,
        "probability_of_delay": y_pred if not flight_has_occurred else None,
        "status": "delayed" if delay_time else ("predicted" if not flight_has_occurred else "ontime"),
        "model_version": st.session_state.model_version
    }
    
    return row

def calculate_metrics(df: pd.DataFrame) -> Dict:
    """Calculate key performance metrics"""
    if len(df) == 0:
        return {
            'total_flights': 0,
            'delayed_flights': 0,
            'delay_rate': 0,
            'avg_delay': 0,
            'accuracy': 0,
            'precision': 0,
            'recall': 0,
            'flights_predicted': 0
        }
    
    # Basic metrics
    delayed_flights = (df['delay_time'] > 0).sum()
    delay_rate = (delayed_flights / len(df)) * 100 if len(df) > 0 else 0
    avg_delay = df[df['delay_time'] > 0]['delay_time'].mean() if delayed_flights > 0 else 0
    
    # Count predicted flights
    flights_predicted = df['status'].eq('predicted').sum() # df['probability_of_delay'].notna().sum()

    # Calculate model performance metrics
    # Only calculate for flights that have both prediction and actual outcome
    evaluated_flights = df[(df['probability_of_delay'].notna()) & (df['scheduled_departure'].notna())]
    
    if len(evaluated_flights) > 0:
        # Calculate predictions vs actuals
        evaluated_flights['predicted_delay'] = evaluated_flights['probability_of_delay'] > 0.5
        evaluated_flights['actual_delay'] = evaluated_flights['delay_time'] > 0
        
        # True/False Positives/Negatives
        tp = ((evaluated_flights['predicted_delay'] == True) & (evaluated_flights['actual_delay'] == True)).sum()
        tn = ((evaluated_flights['predicted_delay'] == False) & (evaluated_flights['actual_delay'] == False)).sum()
        fp = ((evaluated_flights['predicted_delay'] == True) & (evaluated_flights['actual_delay'] == False)).sum()
        fn = ((evaluated_flights['predicted_delay'] == False) & (evaluated_flights['actual_delay'] == True)).sum()
        
        # Metrics
        accuracy = ((tp + tn) / len(evaluated_flights)) * 100 if len(evaluated_flights) > 0 else 0
        precision = (tp / (tp + fp)) * 100 if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) * 100 if (tp + fn) > 0 else 0
    else:
        accuracy = precision = recall = 0
    
    return {
        'total_flights': len(df),
        'delayed_flights': delayed_flights,
        'delay_rate': delay_rate,
        'avg_delay': avg_delay,
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'flights_predicted': flights_predicted
    }

def create_metrics_cards(metrics: Dict):
    """Create modern metric cards with improved information"""
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">✈️ {metrics['total_flights']:,}</div>
            <div class="metric-label">Total Flights</div>
            <div class="metric-delta">{metrics['flights_predicted']} predicted</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        delta_class = "positive" if metrics['delay_rate'] < 20 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">⏰ {metrics['delayed_flights']:,}</div>
            <div class="metric-label">Delayed Flights</div>
            <div class="metric-delta {delta_class}">{metrics['delay_rate']:.1f}% delay rate</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        accuracy_class = "positive" if metrics['accuracy'] > 70 else "negative"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">🎯 {metrics['accuracy']:.0f}%</div>
            <div class="metric-label">Model Accuracy</div>
            <div class="metric-delta {accuracy_class}">P: {metrics['precision']:.0f}% R: {metrics['recall']:.0f}%</div>
        </div>
        """, unsafe_allow_html=True)
    
    with col4:
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">⏱️ {metrics['avg_delay']:.0f}m</div>
            <div class="metric-label">Avg Delay Time</div>
            <div class="metric-delta">When delayed</div>
        </div>
        """, unsafe_allow_html=True)

def create_airport_charts(df: pd.DataFrame):
    """Create airport traffic visualization"""
    if len(df) < 5:
        st.info("📊 Collecting data... Charts will appear when we have enough flights.")
        return
    
    # Top airports analysis
    top5_dep = df["departure_airport"].value_counts().head(5).reset_index()
    top5_dep.columns = ["airport", "count"]
    
    top5_arr = df["arrival_airport"].value_counts().head(5).reset_index()
    top5_arr.columns = ["airport", "count"]
    
    # Calculate delay rates by airport
    delay_by_dep = df[df['delay_time'] > 0].groupby('departure_airport').size()
    total_by_dep = df.groupby('departure_airport').size()
    delay_rate_dep = (delay_by_dep / total_by_dep * 100).fillna(0).sort_values(ascending=False).head(5)
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🛫 Top Departure Airports")
        
        # Add delay rate to tooltip
        top5_dep['delay_rate'] = top5_dep['airport'].map(
            lambda x: delay_rate_dep.get(x, 0)
        )
        
        bar_chart_dep = alt.Chart(top5_dep).mark_bar(
            cornerRadiusTopRight=3,
            cornerRadiusBottomRight=3,
            color='#667eea'
        ).encode(
            x=alt.X("count:Q", title="Number of Flights"),
            y=alt.Y("airport:N", sort='-x', title="Airport Code"),
            tooltip=[
                alt.Tooltip('airport:N', title='Airport'),
                alt.Tooltip('count:Q', title='Flights'),
                alt.Tooltip('delay_rate:Q', title='Delay Rate %', format='.1f')
            ]
        ).properties(
            height=250
        )
        
        st.altair_chart(bar_chart_dep, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)
    
    with col2:
        st.markdown('<div class="chart-container">', unsafe_allow_html=True)
        st.markdown("### 🛬 Top Arrival Airports")
        
        bar_chart_arr = alt.Chart(top5_arr).mark_bar(
            cornerRadiusTopRight=3,
            cornerRadiusBottomRight=3,
            color='#764ba2'
        ).encode(
            x=alt.X("count:Q", title="Number of Flights"),
            y=alt.Y("airport:N", sort='-x', title="Airport Code"),
            tooltip=['airport:N', 'count:Q']
        ).properties(
            height=250
        )
        
        st.altair_chart(bar_chart_arr, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

def create_time_series_chart(df: pd.DataFrame):
    """Create delay probability time series with confidence intervals"""
    if len(df) < 10:
        return
    
    # Get recent predictions
    recent_data = df.tail(50).copy()
    recent_data['index'] = range(len(recent_data))
    recent_data['timestamp'] = pd.to_datetime('now') - pd.to_timedelta(
        (len(recent_data) - recent_data['index']) * 30, unit='s'
    )
    
    # Filter for predicted flights
    prediction_data = recent_data[recent_data['probability_of_delay'].notna()]
    
    if len(prediction_data) == 0:
        return
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 📈 Real-time Delay Predictions")
    
    # Main line chart
    line_chart = alt.Chart(prediction_data).mark_line(
        point=True,
        strokeWidth=3,
        color='#FF6B6B'
    ).encode(
        x=alt.X('timestamp:T', title='Time'),
        y=alt.Y('probability_of_delay:Q', 
                title='Delay Probability',
                scale=alt.Scale(domain=[0, 1])),
        tooltip=[
            alt.Tooltip('flight_number:N', title='Flight'),
            alt.Tooltip('departure_airport:N', title='From'),
            alt.Tooltip('arrival_airport:N', title='To'),
            alt.Tooltip('probability_of_delay:Q', title='Delay Prob', format='.2%')
        ]
    ).properties(
        height=200
    )
    
    # Threshold line at 50%
    threshold_line = alt.Chart(pd.DataFrame({'y': [0.5]})).mark_rule(
        strokeDash=[5, 5],
        color='red',
        opacity=0.7
    ).encode(y='y:Q')
    
    # Combine charts
    combined_chart = line_chart + threshold_line
    st.altair_chart(combined_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_delay_heatmap(df: pd.DataFrame):
    """Create hourly delay pattern heatmap"""
    if len(df) < 20:
        return
    
    # Extract hour and calculate delay statistics
    df_copy = df.copy()
    df_copy['hour'] = pd.to_datetime(df_copy['scheduled_departure'], errors='coerce').dt.hour
    df_copy['is_delayed'] = (df_copy['delay_time'] > 0).astype(int)
    
    # Group by hour
    hourly_stats = df_copy.groupby('hour').agg({
        'is_delayed': ['count', 'sum']
    }).round(2)
    
    hourly_stats.columns = ['total_flights', 'delayed_flights']
    hourly_stats['delay_rate'] = (hourly_stats['delayed_flights'] / hourly_stats['total_flights'] * 100).fillna(0)
    hourly_stats = hourly_stats.reset_index()
    
    if len(hourly_stats) == 0:
        return
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### 🕐 Delay Patterns by Hour of Day")
    
    # Create heatmap
    heatmap = alt.Chart(hourly_stats).mark_rect(
        cornerRadius=3
    ).encode(
        x=alt.X('hour:O', title='Hour of Day'),
        y=alt.Y('delay_rate:Q', title='', axis=None),
        color=alt.Color('delay_rate:Q',
                       scale=alt.Scale(scheme='redyellowgreen', reverse=True),
                       title='Delay Rate (%)'),
        tooltip=[
            alt.Tooltip('hour:O', title='Hour'),
            alt.Tooltip('delay_rate:Q', title='Delay Rate', format='.1f'),
            alt.Tooltip('total_flights:Q', title='Total Flights'),
            alt.Tooltip('delayed_flights:Q', title='Delayed')
        ]
    ).properties(
        height=100
    )
    
    st.altair_chart(heatmap, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

def create_airline_performance_chart(df: pd.DataFrame):
    """Create airline performance comparison"""
    if len(df) < 10:
        return
    
    # Calculate airline statistics
    airline_stats = df.groupby('airline').agg({
        'delay_time': ['count', lambda x: (x > 0).sum(), 'mean']
    })
    
    airline_stats.columns = ['total_flights', 'delayed_flights', 'avg_delay']
    airline_stats['delay_rate'] = (airline_stats['delayed_flights'] / airline_stats['total_flights'] * 100)
    airline_stats = airline_stats[airline_stats['total_flights'] >= 3]  # Min 3 flights
    airline_stats = airline_stats.sort_values('delay_rate', ascending=True).head(10)
    
    if len(airline_stats) == 0:
        return
    
    st.markdown('<div class="chart-container">', unsafe_allow_html=True)
    st.markdown("### ✈️ Airline Performance")
    
    airline_stats_reset = airline_stats.reset_index()
    
    bar_chart = alt.Chart(airline_stats_reset).mark_bar(
        cornerRadiusTopRight=3,
        cornerRadiusBottomRight=3
    ).encode(
        x=alt.X('delay_rate:Q', title='Delay Rate (%)'),
        y=alt.Y('airline:N', sort='-x', title=''),
        color=alt.Color('delay_rate:Q',
                       scale=alt.Scale(scheme='redyellowgreen', reverse=True),
                       legend=None),
        tooltip=[
            alt.Tooltip('airline:N', title='Airline'),
            alt.Tooltip('delay_rate:Q', title='Delay Rate', format='.1f'),
            alt.Tooltip('total_flights:Q', title='Total Flights'),
            alt.Tooltip('avg_delay:Q', title='Avg Delay (min)', format='.0f')
        ]
    ).properties(
        height=250
    )
    
    st.altair_chart(bar_chart, use_container_width=True)
    st.markdown('</div>', unsafe_allow_html=True)

# Main App
st.markdown("""
<div class="main-header">
    <h1>✈️ Flight Delay Predictor</h1>
    <p>Real-time Machine Learning with River & Kafka Streaming</p>
</div>
""", unsafe_allow_html=True)

# Sidebar Configuration
with st.sidebar:
    st.markdown("## ⚙️ Settings")
    
    # Connection status
    status_class = "status-connected" if st.session_state.kafka_connected else "status-disconnected"
    status_text = "Connected" if st.session_state.kafka_connected else "Disconnected"
    st.markdown(f"""
    <div style="margin-bottom: 1rem;">
        <span class="status-indicator {status_class}"></span>
        Kafka Status: <strong>{status_text}</strong>
    </div>
    """, unsafe_allow_html=True)
    
    # Show model stats
    st.metric("Model Version", st.session_state.model_version)
    st.metric("Flights Processed", st.session_state.processed_count)
    
    st.markdown("---")
    
    # Kafka Configuration
    kafka_enabled = st.checkbox("Enable Kafka Connection", True)
    kafka_servers = st.text_input(
        "Kafka Servers", 
        "localhost:8097,localhost:8098,localhost:8099"
    )
    
    st.markdown("---")
    
    # Display Filters
    st.markdown("### 🔍 Display Filters")
    show_predictions_only = st.checkbox("Show predictions only", False)
    show_delayed_only = st.checkbox("Show delayed flights only", False)
    min_delay_prob = st.slider("Min delay probability", 0.0, 1.0, 0.0, 0.1)
    max_records = st.select_slider(
        "Max records to display",
        options=[10, 25, 50, 100, 200, 500],
        value=100
    )
    
    st.markdown("---")
    
    # Refresh Settings
    auto_refresh = st.checkbox("Auto-refresh (5s)", True)
    if st.button("🔄 Refresh Now"):
        st.rerun()
    
    st.markdown("---")
    
    # Model Information
    st.markdown("### 📊 Model Information")
    with st.expander("View Details"):
        st.info("""
        **Algorithm**: Hoeffding Tree Classifier
        
        **Features**:
        - Departure/Arrival airports
        - Terminals
        - Airlines
        - Hour of day
        - Day of week
        - Route (airport pair)
        
        **Learning**: Online/Incremental
        - Updates with each flight
        - No retraining needed
        - Adapts to patterns
        """)

# Main Content Area
data = st.session_state.data

# Convert to DataFrame and apply filters
if data:
    df = pd.DataFrame(data)
    
    # Apply filters
    if show_predictions_only:
        df = df[df['probability_of_delay'].notna()]
    
    if show_delayed_only:
        df = df[df['delay_time'] > 0]
    
    if min_delay_prob > 0:
        df = df[(df['probability_of_delay'].isna()) | (df['probability_of_delay'] >= min_delay_prob)]
    
    # Limit records
    df = df.tail(max_records)
else:
    df = pd.DataFrame()

# Calculate and display metrics


# Alert System
if len(df) > 0:
    # print('Prob col')
    # print(df[df['probability_of_delay'].notna()])
    metrics = calculate_metrics(df)
    create_metrics_cards(metrics)
    # High risk flights alert
    high_risk_flights = df[(df['probability_of_delay'] > 0.8) & (df['probability_of_delay'].notna())]
    if len(high_risk_flights) > 0:
        st.markdown(f"""
        <div class="alert-danger">
            <strong>⚠️ High Risk Alert!</strong> 
            {len(high_risk_flights)} flight(s) have >80% delay probability
        </div>
        """, unsafe_allow_html=True)
    
    # Recent delays alert
    recent_delays = df[(df['delay_time'] > 30) & (df['actual_departure'].notna())].tail(5)
    if len(recent_delays) > 0:
        avg_recent_delay = recent_delays['delay_time'].mean()
        st.markdown(f"""
        <div class="alert-warning">
            <strong>📊 Recent Delays:</strong> 
            Last {len(recent_delays)} delayed flights averaged {avg_recent_delay:.0f} minutes delay
        </div>
        """, unsafe_allow_html=True)

# Charts Section
st.markdown('<div class="section-header">📊 Real-time Analytics Dashboard</div>', unsafe_allow_html=True)

# Create visualization tabs
tab1, tab2, tab3, tab4 = st.tabs(["📈 Predictions", "✈️ Airports", "⏰ Patterns", "🏢 Airlines"])

with tab1:
    create_time_series_chart(df)

with tab2:
    create_airport_charts(df)

with tab3:
    create_delay_heatmap(df)

with tab4:
    create_airline_performance_chart(df)

# Recent Flights Table
st.markdown('<div class="section-header">📋 Recent Flights Monitor</div>', unsafe_allow_html=True)

if len(df) > 0:
    # Prepare display dataframe
    display_df = df.copy()

    # Add status badges
    def format_status(row):
        if row['status'] == 'delayed':
            return f'<span class="status-badge status-delayed">Delayed {row["delay_time"]:.0f}m</span>'
        elif row['status'] == 'predicted':
            prob = row['probability_of_delay']
            if prob > 0.7:
                return f'<span class="status-badge status-delayed">High Risk</span>'
            elif prob > 0.5:
                return f'<span class="status-badge status-predicted">Medium Risk</span>'
            else:
                return f'<span class="status-badge status-ontime">Low Risk</span>'
        else:
            return '<span class="status-badge status-ontime">On Time</span>'
    
    # Format columns for display
    display_df['Flight Status'] = display_df.apply(format_status, axis=1)
    
    # Format times
    for col in ['scheduled_departure', 'scheduled_arrival', 'actual_departure', 'estimated_arrival', 'actual_arrival']:
        display_df[col] = pd.to_datetime(display_df[col], errors='coerce').dt.strftime('%#d %b %H:%M')
    
    # Select and rename columns
    display_columns = {
        'flight_number': '✈️ Flight',
        'airline': '🏢 Airline',
        'departure_airport': '🛫 From',
        'scheduled_departure': '📅 Sched Dep',
        'actual_departure': '✓ Actual Dep',
        'arrival_airport': '🛬 To',
        'scheduled_arrival': '📅 Sched Arr',
        'estimated_arrival': '⏳ Est Arr',
        'actual_arrival': '✓ Actual Arr',
        'Flight Status': '📊 Status'
    }
    
    display_df = display_df.rename(columns=display_columns)
    selected_columns = list(display_columns.values())
    
    # Display with custom HTML
    st.markdown(
        display_df[selected_columns].tail(20).to_html(escape=False, index=False),
        unsafe_allow_html=True
    )
else:
    st.info("🔄 Waiting for flight data from Kafka stream...")

# Kafka Consumer Section
if 'consumer' not in st.session_state:
    st.session_state.consumer = None

if kafka_enabled:
    try:
        # Initialize consumer if needed
        if st.session_state.consumer is None:
            try:
                st.session_state.consumer = Consumer({
                    'bootstrap.servers': kafka_servers,
                    'group.id': f'flight_predictor_{int(time.time())}',
                    'auto.offset.reset': 'latest',
                    'enable.auto.commit': True,
                    'session.timeout.ms': 10000,
                    'heartbeat.interval.ms': 3000,
                    'max.poll.interval.ms': 300000,
                    'socket.keepalive.enable': True,
                    'api.version.request': True
                })
                st.session_state.consumer.subscribe(['flightDelay'])
                st.session_state.kafka_connected = True
            except Exception as e:
                st.session_state.kafka_connected = False
        
        # Poll for messages
        if st.session_state.consumer:
            messages_processed = 0
            errors_count = 0
            
            # Poll multiple times for better real-time performance
            for _ in range(30):  # Increased polling
                try:
                    msg = st.session_state.consumer.poll(timeout=0.3)
                    
                    if msg is None:
                        continue
                    
                    if msg.error():
                        if msg.error().code() == KafkaError._PARTITION_EOF:
                            continue
                        else:
                            errors_count += 1
                            continue
                    
                    # Process message
                    try:
                        value = msg.value().decode('utf-8')
                        flight_data = json.loads(value)
                        
                        # Process flight and add to data
                        row = process_flight_message(flight_data)
                        st.session_state.data.append(row)
                        st.session_state.processed_count += 1
                        messages_processed += 1
                        
                        # Keep only recent data (memory management)
                        if len(st.session_state.data) > 1000:
                            st.session_state.data = st.session_state.data[-1000:]
                        
                    except json.JSONDecodeError:
                        pass
                    except Exception:
                        pass
                
                except Exception:
                    pass
            
            # Update status
            if messages_processed > 0:
                st.session_state.kafka_connected = True
                
    except Exception:
        st.session_state.kafka_connected = False
        
        # Clean up consumer
        if st.session_state.consumer:
            try:
                st.session_state.consumer.close()
            except:
                pass
            st.session_state.consumer = None

else:
    # Kafka disabled - clean up
    if st.session_state.consumer:
        try:
            st.session_state.consumer.close()
        except:
            pass
        st.session_state.consumer = None
    st.session_state.kafka_connected = False
    
    # Demo Mode
    if st.sidebar.button("🎲 Generate Demo Data") or len(st.session_state.data) < 5:
        # Realistic airport pairs and airlines
        routes = [
            ("BKK", "NRT", "Thai Airways", 6.5),
            ("BKK", "SIN", "Singapore Airlines", 2.5),
            ("BKK", "HKG", "Cathay Pacific", 3.0),
            ("BKK", "ICN", "Korean Air", 5.5),
            ("NRT", "LAX", "ANA", 10.0),
            ("SIN", "SYD", "Qantas", 7.5),
            ("HKG", "LHR", "British Airways", 12.0),
            ("BKK", "DXB", "Emirates", 6.0),
            ("ICN", "SFO", "United Airlines", 11.0),
            ("BKK", "KUL", "Malaysia Airlines", 2.0)
        ]
        
        # Generate flights
        for _ in range(10):
            route = random.choice(routes)
            dep_airport, arr_airport, airline, flight_hours = route
            
            # Random times
            hour = random.randint(6, 22)
            scheduled_dep = datetime.now().replace(
                hour=hour,
                minute=random.randint(0, 59),
                second=0,
                microsecond=0
            )
            scheduled_arr = scheduled_dep + timedelta(hours=flight_hours)
            
            # Create flight data structure matching Kafka format
            flight_data = {
                "flight": {"iata": f"{airline[:2]}{random.randint(100, 999)}"},
                "airline": {"name": airline},
                "departure": {
                    "iata": dep_airport,
                    "terminal": random.choice(["1", "2", "3", "A", "B"]),
                    "scheduled": scheduled_dep.isoformat() + "Z",
                    "actual": None,
                    "delay": None
                },
                "arrival": {
                    "iata": arr_airport,
                    "terminal": random.choice(["1", "2", "3", "A", "B"]),
                    "scheduled": scheduled_arr.isoformat() + "Z",
                    "actual": None,
                    "delay": None
                }
            }
            
            # Simulate some flights as completed with delays
            if random.random() < 0.3:  # 30% of flights have occurred
                actual_dep = scheduled_dep + timedelta(minutes=random.randint(0, 30))
                delay_minutes = random.randint(15, 90) if random.random() < 0.3 else 0
                
                flight_data["departure"]["actual"] = actual_dep.isoformat() + "Z"
                flight_data["departure"]["delay"] = delay_minutes if delay_minutes > 0 else None
                
                if delay_minutes > 0:
                    actual_arr = scheduled_arr + timedelta(minutes=delay_minutes)
                    flight_data["arrival"]["actual"] = actual_arr.isoformat() + "Z"
                    flight_data["arrival"]["delay"] = delay_minutes
            
            # Process and add to data
            row = process_flight_message(flight_data)
            st.session_state.data.append(row)
            st.session_state.processed_count += 1
        
        # Clean old data
        if len(st.session_state.data) > 500:
            st.session_state.data = st.session_state.data[-500:]

# Auto-refresh
if auto_refresh:
    time.sleep(5)
    st.rerun()

# Footer
st.markdown("---")
st.markdown("""
<div style="text-align: center; color: #7f8c8d; margin: 2rem 0;">
    <p>🤖 Powered by River ML • 📊 Built with Streamlit • ⚡ Real-time Kafka Streaming</p>
    <p><small>Flight Delay Prediction System v2.0 - Online Machine Learning Demo</small></p>
    <p><small>Model learns from {model_flights} flights • Last update: {last_update}</small></p>
</div>
""".format(
    model_flights=st.session_state.processed_count,
    last_update=datetime.now().strftime("%H:%M:%S")
), unsafe_allow_html=True)
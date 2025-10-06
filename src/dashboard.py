"""
Dashboard view for energy consumption analytics
built with Streamlit
"""
import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
import joblib
from datetime import datetime

# Page configuration
st.set_page_config(
    page_title="Australian Energy Analytics",
    page_icon="🔋",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    h1 {
        color: #1f77b4;
        padding-bottom: 1rem;
    }
    .metric-container {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    </style>
    """, unsafe_allow_html=True)

@st.cache_data
def load_data():
    """Load processed energy data"""
    data_path = Path('data/processed/processed_energy_data.csv')
    if not data_path.exists():
        return None
    
    df = pd.read_csv(data_path)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    return df

@st.cache_resource
def load_models():
    """Load trained ML models"""
    model_path = Path('data/models/energy_forecast_models.pkl')
    if not model_path.exists():
        return None
    return joblib.load(model_path)

def create_consumption_chart(df, states):
    """Create time series chart for energy consumption"""
    filtered_df = df[df['state'].isin(states)].copy()
    
    # Sample data for performance (if too many points)
    if len(filtered_df) > 5000:
        filtered_df = filtered_df.sample(n=5000, random_state=42).sort_values('timestamp')
    
    fig = px.line(
        filtered_df,
        x='timestamp',
        y='consumption_mw',
        color='state',
        title='Energy Consumption Over Time',
        labels={'consumption_mw': 'Consumption (MW)', 'timestamp': 'Time'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(
        hovermode='x unified',
        height=450,
        showlegend=True,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1)
    )
    
    return fig

def create_state_comparison(df, selected_states):
    """Create bar chart comparing states"""
    state_data = df[df['state'].isin(selected_states)].groupby('state').agg({
        'consumption_mw': ['mean', 'max', 'min']
    }).round(2)
    
    state_data.columns = ['Average', 'Peak', 'Minimum']
    state_data = state_data.reset_index()
    
    fig = go.Figure()
    
    fig.add_trace(go.Bar(
        name='Average',
        x=state_data['state'],
        y=state_data['Average'],
        marker_color='#1f77b4'
    ))
    
    fig.add_trace(go.Bar(
        name='Peak',
        x=state_data['state'],
        y=state_data['Peak'],
        marker_color='#ff7f0e'
    ))
    
    fig.update_layout(
        title='Consumption Comparison by State',
        xaxis_title='State',
        yaxis_title='Consumption (MW)',
        barmode='group',
        height=400,
        showlegend=True
    )
    
    return fig

def create_hourly_pattern(df, selected_states):
    """Create heatmap showing consumption patterns"""
    filtered_df = df[df['state'].isin(selected_states)].copy()
    
    filtered_df['hour'] = filtered_df['timestamp'].dt.hour
    filtered_df['day_name'] = filtered_df['timestamp'].dt.day_name()
    
    pivot_data = filtered_df.groupby(['day_name', 'hour'])['consumption_mw'].mean().reset_index()
    pivot_table = pivot_data.pivot(index='day_name', columns='hour', values='consumption_mw')
    
    # Reorder days
    day_order = ['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday']
    pivot_table = pivot_table.reindex([day for day in day_order if day in pivot_table.index])
    
    fig = go.Figure(data=go.Heatmap(
        z=pivot_table.values,
        x=pivot_table.columns,
        y=pivot_table.index,
        colorscale='RdYlBu_r',
        hovertemplate='<b>%{y}</b><br>Hour: %{x}<br>Consumption: %{z:.0f} MW<extra></extra>',
        colorbar=dict(title="MW")
    ))
    
    fig.update_layout(
        title='Average Consumption Patterns: Hour vs Day of Week',
        xaxis_title='Hour of Day',
        yaxis_title='Day of Week',
        height=400
    )
    
    return fig

def create_distribution_chart(df, selected_states):
    """Create distribution comparison"""
    filtered_df = df[df['state'].isin(selected_states)]
    
    fig = px.box(
        filtered_df,
        x='state',
        y='consumption_mw',
        color='state',
        title='Consumption Distribution by State',
        labels={'consumption_mw': 'Consumption (MW)', 'state': 'State'},
        color_discrete_sequence=px.colors.qualitative.Set2
    )
    
    fig.update_layout(height=400, showlegend=False)
    
    return fig

def display_metrics(df, selected_states):
    """Display key metrics"""
    filtered_df = df[df['state'].isin(selected_states)]
    
    col1, col2, col3, col4 = st.columns(4)
    
    total_records = len(filtered_df)
    avg_consumption = filtered_df['consumption_mw'].mean()
    max_consumption = filtered_df['consumption_mw'].max()
    min_consumption = filtered_df['consumption_mw'].min()
    
    with col1:
        st.metric(
            "Total Records",
            f"{total_records:,}",
            help="Number of data points in selected period"
        )
    
    with col2:
        st.metric(
            "Average Consumption",
            f"{avg_consumption:,.0f} MW",
            help="Mean consumption across selected states"
        )
    
    with col3:
        st.metric(
            "Peak Consumption",
            f"{max_consumption:,.0f} MW",
            help="Highest recorded consumption"
        )
    
    with col4:
        st.metric(
            "Minimum Consumption",
            f"{min_consumption:,.0f} MW",
            help="Lowest recorded consumption"
        )

def forecast_section(df, models):
    """Display forecasting section"""
    st.header("Energy Consumption Forecast")
    
    if models is None:
        st.warning("!!Models not found. Please run the training pipeline first.")
        st.code("python src/train_model.py", language="bash")
        return
    
    st.markdown("**Model Performance Summary:**")
    
    # Create performance table
    perf_data = []
    for state, model_data in models.items():
        perf_data.append({
            'State': state,
            'Model Type': 'Random Forest',
            'Trees': 100,
            'Status': 'Trained'
        })
    
    perf_df = pd.DataFrame(perf_data)
    st.dataframe(perf_df, use_container_width=True, hide_index=True)
    
    st.markdown("---")
    
    # Get latest data for demonstration
    st.subheader("Latest Consumption Data")
    
    latest_data = []
    for state in sorted(df['state'].unique()):
        state_df = df[df['state'] == state].sort_values('timestamp')
        latest_row = state_df.iloc[-1]
        
        latest_data.append({
            'State': state,
            'Latest Reading': f"{latest_row['consumption_mw']:.0f} MW",
            'Temperature': f"{latest_row['temperature']:.1f}°C",
            'Time': latest_row['timestamp'].strftime('%Y-%m-%d %H:%M')
        })
    
    st.dataframe(pd.DataFrame(latest_data), use_container_width=True, hide_index=True)
    
    st.info("💡 **Note:** Models are trained and ready for real-time predictions. In production, these would forecast next-hour consumption based on current conditions.")

def main():
    """Main dashboard application"""
    
    # Header
    st.title("Australian Energy Consumption Analytics")
    st.markdown("**Real-time monitoring and ML-powered forecasting of energy consumption across Australia**")
    
    # Load data
    with st.spinner("Loading data..."):
        df = load_data()
        models = load_models()
    
    if df is None:
        st.error("! No data found. Please run the pipeline first:")
        st.code("python src/main.py", language="bash")
        return
    
    # Sidebar
    st.sidebar.header("⚙️ Filters & Settings")
    
    # Date range filter
    min_date = df['timestamp'].min().date()
    max_date = df['timestamp'].max().date()
    
    st.sidebar.subheader("📅 Date Range")
    date_range = st.sidebar.date_input(
        "Select dates",
        value=(min_date, max_date),
        min_value=min_date,
        max_value=max_date,
        help="Choose the time period to analyze"
    )
    
    # State filter
    st.sidebar.subheader("📍 States")
    all_states = sorted(df['state'].unique())
    selected_states = st.sidebar.multiselect(
        "Select states to display",
        options=all_states,
        default=all_states,
        help="Choose which states to include in analysis"
    )
    
    if not selected_states:
        st.warning("!! Please select at least one state from the sidebar.")
        return
    
    # Filter data
    if len(date_range) == 2:
        mask = (
            (df['timestamp'].dt.date >= date_range[0]) &
            (df['timestamp'].dt.date <= date_range[1]) &
            (df['state'].isin(selected_states))
        )
        filtered_df = df[mask]
    else:
        filtered_df = df[df['state'].isin(selected_states)]
    
    if len(filtered_df) == 0:
        st.warning("!! No data available for selected filters.")
        return
    
    # Display metrics
    display_metrics(filtered_df, selected_states)
    
    st.markdown("---")
    
    # Main visualizations
    st.header("📈 Consumption Analysis")
    
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Time Series", "📐 Comparison", "🔥 Patterns", "📦 Distribution"])
    
    with tab1:
        st.plotly_chart(
            create_consumption_chart(filtered_df, selected_states),
            use_container_width=True
        )
        
        with st.expander("ℹ️ About this chart"):
            st.markdown("""
            This time series chart shows energy consumption trends over time. Key observations:
            - **Daily patterns**: Higher consumption during business hours (9 AM - 5 PM)
            - **Weekly patterns**: Lower consumption on weekends
            - **Seasonal variations**: Changes based on temperature and weather
            """)
    
    with tab2:
        st.plotly_chart(
            create_state_comparison(filtered_df, selected_states),
            use_container_width=True
        )
        
        with st.expander("ℹ️ About this chart"):
            st.markdown("""
            Comparison of average, peak, and minimum consumption across states:
            - **NSW & QLD**: Highest consumption (larger populations)
            - **SA**: Lowest consumption (smaller population)
            - **Peak values**: Typically occur during hot summer afternoons
            """)
    
    with tab3:
        st.plotly_chart(
            create_hourly_pattern(filtered_df, selected_states),
            use_container_width=True
        )
        
        with st.expander("ℹ️ About this chart"):
            st.markdown("""
            Heatmap showing typical consumption patterns:
            - **Darkest red**: Peak consumption hours (afternoon)
            - **Light colors**: Low consumption (early morning, weekends)
            - **Business days vs weekends**: Clear difference in patterns
            """)
    
    with tab4:
        st.plotly_chart(
            create_distribution_chart(filtered_df, selected_states),
            use_container_width=True
        )
        
        with st.expander("ℹ️ About this chart"):
            st.markdown("""
            Box plot showing consumption distribution:
            - **Box**: 50% of data (25th to 75th percentile)
            - **Line in box**: Median consumption
            - **Whiskers**: Typical range of values
            - **Dots**: Outliers (unusual consumption events)
            """)
    
    st.markdown("---")
    
    # Forecasting section
    forecast_section(filtered_df, models)
    
    st.markdown("---")
    
    # Insights section
    st.header("💡 Key Insights")
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader("🕐 Peak Consumption Hours")
        hourly_avg = filtered_df.groupby(
            filtered_df['timestamp'].dt.hour
        )['consumption_mw'].mean().sort_values(ascending=False)
        
        peak_hours_data = []
        for hour, consumption in hourly_avg.head(5).items():
            peak_hours_data.append({
                'Hour': f"{hour:02d}:00",
                'Avg Consumption': f"{consumption:.0f} MW"
            })
        
        st.dataframe(pd.DataFrame(peak_hours_data), use_container_width=True, hide_index=True)
    
    with col2:
        st.subheader("🏆 State Rankings")
        state_totals = filtered_df.groupby('state')['consumption_mw'].sum().sort_values(ascending=False)
        
        ranking_data = []
        for idx, (state, total) in enumerate(state_totals.items(), 1):
            ranking_data.append({
                'Rank': idx,
                'State': state,
                'Total Consumption': f"{total:,.0f} MW"
            })
        
        st.dataframe(pd.DataFrame(ranking_data), use_container_width=True, hide_index=True)
    
    # Footer
    st.markdown("---")
    st.markdown(f"""
    <div style='text-align: center; color: #666; padding: 1rem;'>
        <p><strong>Australian Energy Analytics Dashboard</strong></p>
        <p>Data updated: {df['timestamp'].max().strftime("%Y-%m-%d %H:%M")} | 
        Total records: {len(df):,} | 
        States monitored: {len(df['state'].unique())}</p>
        <p>Built with Streamlit • Powered by Random Forest ML</p>
        <p>📧 Contact: gladysaureliaa@gmail.com | 📍 Brisbane, QLD</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
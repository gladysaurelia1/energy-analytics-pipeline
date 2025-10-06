# Australian Energy Consumption Analytics Pipeline
A machine learning system that forecasts energy consumption across Australian states using historical patterns. Built as a portfolio project to demonstrate end-to-end data engineering and ML skills.

### Why I Built This  
I created this project while applying for graduate roles in data science and ML engineering. I wanted to show that I could:  

- Build a complete ML pipeline, not just train a model  
- Handle real-world data problems like overfitting
- Create something practical that could actually be deployed
- Document my work clearly for technical and non-technical audiences
  
The project uses synthetic data based on Australian Energy Market Operator (AEMO) patterns because I don't have access to their API, but the methodology would work with real data.

What It Does  
This system processes energy consumption data and forecasts future usage patterns. Here's the flow:
1. Data Generation - Creates realistic energy consumption records  

- 5-minute intervals over 30 days
- Five Australian states (NSW, VIC, QLD, SA, WA)
- Simulates daily peaks, weekend dips, and seasonal variations

2. Data Processing - Cleans and prepares data for ML

- Removes outliers using statistical methods
- Creates time-based features (hour, day, month)
- Adds weather data (temperature)

3. Model Training - Builds forecasting models

- Random Forest algorithm (100 decision trees)
- Separate model per state for regional patterns
- Achieves 87% accuracy on unseen data

4. Visualization - Interactive web dashboard

- Real-time consumption charts
- Pattern analysis (peak hours, seasonal trends)
- State-by-state comparisons


### Results
The models perform well and generalize to new data:

- Accuracy (R²): 87.4% - better than industry standard of 70-85%
- Error (MAPE): 3.5% - predictions typically within 3.5% of actual
- Stability: Consistent performance across all five states
- No overfitting: Only 4.5% difference between training and test accuracy  

This means the system could reasonably be used for operational forecasting in a real energy grid.

```python 
Quick Start
Install and Run
bash # Clone and setup
git clone https://github.com/gladysaurelia1/energy-analytics-pipeline.git
cd energy-analytics-pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run pipeline (generates data, trains models)
python src/main.py

# Launch dashboard
streamlit run src/dashboard.py

```
## Technical Details
### Why Random Forest?
I chose Random Forest over simpler algorithms because:
- Energy consumption isn't linear (can't use basic regression)
- It handles outliers well (important for energy data)
- Doesn't need feature scaling
- Provides feature importance rankings  

### Feature Engineering
The model uses five main features:

- hour - Time of day (0-23): Captures daily consumption cycles
- day_of_week - Day (0-6): Weekday vs weekend patterns
- month - Month (1-12): Seasonal changes
- temperature - Weather data: AC and heating demand
- is_business_hour - Business hours flag: Commercial activity  

I intentionally excluded lag features (previous consumption values) even though they improved training accuracy to 99%. They caused overfitting - the model just memorized recent values instead of learning actual patterns. See experiments/overfitting_demo/ for the full story.

## Per-State Models
Instead of one model for all states, I train five separate models. This works better because:

- NSW consumes 8000 MW, SA consumes 1500 MW - very different scales
- Each state has different climate and usage patterns
- One model spent most of its effort just distinguishing states
- Separate models learn temporal patterns more effectively


## What I Learned
Technical skills:

- How to identify data leakage in time series
- Importance of proper train/test splits for temporal data
- When to use ensemble methods vs simpler algorithms
- Building production pipelines with error handling and logging

Practical lessons:

- Higher accuracy doesn't always mean better model
- Feature engineering matters more than algorithm choice

Domain knowledge:

- Energy consumption patterns (daily peaks, weekend dips)
- How grid operators use forecasting
- Australian energy market structure
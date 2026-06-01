# Australian Energy Consumption Analytics Pipeline
A machine learning pipeline that forecasts energy consumption across 5 Australian states. Built as a portfolio project to demonstrate real-world data engineering and ML skills, from raw data through to a live dashboard.

### Why I Built This  
I built this while applying for graduate roles in data science and other digital roles. I wanted a project that went beyond Jupyter notebooks, something I could actually run end-to-end and show recruiters as a working system.
The data is synthetic, modelled on AEMO consumption patterns, because live API access requires an enterprise agreement. The methodology translates directly to real data.

What It Does  
This system processes energy consumption data and forecasts future usage patterns.
1. Data Generation - Creates synthetic energy consumption records at 5-minute intervals for NSW, VIC, QLD, SA and WA. The base loads, daily shapes and seasonal patterns are calibrated against publicly available AEMO historical figures. Real data would slot in here with minimal changes.

2. Data Processing - Cleans duplicates and outliers, then builds the feature set. Time features (hour, day of week, month, business hours flag) and temperature are extracted. Lag features are generated for exploratory analysis but intentionally excluded from model training.

3. Model Training - Trains a Random Forest model separately for each state. Five separate models rather than one shared model with a state encoding feature.

4. Visualization - Streamlit app showing consumption trends, state comparisons, hourly patterns and model performance.


### Results
The per-state models hit the target range after fixing the issues in earlier versions:
 
| State | Test R2 | MAPE | Train/test gap |
|-------|---------|------|----------------|
| NSW   | 85.2%   | 3.9% | 5.0%           |
| VIC   | 85.3%   | 3.9% | 5.0%           |
| QLD   | 84.4%   | 3.9% | 5.2%           |
| SA    | 84.4%   | 4.0% | 5.6%           |
| WA    | 84.0%   | 4.1% | 6.1%           |
 
Average test R2: **84.7%** -- in the 70-85% range cited in grid forecasting literature for short-horizon consumption models. The train/test gap of 5.4% is low enough that I'm comfortable the models are generalising rather than memorising.

```python 
Quick Start
Install and Run
bash # Clone and setup
git clone https://github.com/gladysaurelia1/energy-analytics-pipeline.git
cd energy-analytics-pipeline
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate
pip install -r requirements.txt

# Run full pipeline 
python src/main.py

# Launch dashboard
streamlit run src/dashboard.py

```
## Project structure
 
```
src/
  main.py             pipeline orchestrator
  extract_data.py     synthetic data generation
  transform_data.py   cleaning and feature engineering
  train_model.py      per-state Random Forest training
  dashboard.py        Streamlit analytics dashboard
 
experiments/
  v1_extreme_overfitting.py   99.4% R2 -- data leakage via lag features
  v2_statedominance.py        97.0% R2 -- state encoding dominated features
  README.md                   walkthrough of what went wrong and why
 
data/
  raw/                generated CSVs (gitignored)
  processed/          cleaned, feature-engineered CSV (gitignored)
  models/             trained model PKLs (gitignored)
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
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
| State | Test R2 | MAPE | Train/test gap |
|-------|---------|------|----------------|
| NSW   | 89.2%   | 3.4% | 1.9%           |
| VIC   | 89.0%   | 3.4% | 1.9%           |
| QLD   | 89.3%   | 3.4% | 1.5%           |
| SA    | 89.0%   | 3.5% | 1.7%           |
| WA    | 88.5%   | 3.5% | 2.4%           |
 
Average test R2: **89.0%** -- above the 70-85% range cited in grid forecasting literature for short-horizon consumption models. The train/test gap of 1.9% is low, which means the models are generalising well rather than memorising training data.

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
## Technical Details
### Why Random Forest?
I chose Random Forest over simpler algorithms because:
- Energy consumption isn't linear (can't use basic regression)
- It handles outliers well (important for energy data)
- Doesn't need feature scaling
- Provides feature importance rankings 

### Why no lag features?
Inflated test accuracy to 99.4% in the first version. The lag features (previous hour, previous 24h) were carrying so much signal that the temporal features (hour, day, month) had near-zero importance. In production, you won't always have recent ground-truth lag values available, and even when you do, building a model that depends on them creates a brittle dependency. The five non-leaky features produce 87% R2 and represent patterns the model can generalise from.

### Feature Engineering
Five features make it into the final model:
 
- `hour` (0-23): Captures the daily consumption cycle. Peak around 14:00, trough around 03:00.
- `day_of_week` (0-6): Weekends run around 15% lighter than weekdays across all states.
- `month` (1-12): Seasonal signal -- higher in summer (cooling) and winter (heating), lower in spring/autumn.
- `is_business_hour` (binary): Commercial load switches on and off at 09:00 and 17:00. Adds signal beyond hour alone.
- `temperature`: HVAC load is the main driver of demand spikes. The correlation is nonlinear (U-shaped), which is part of why a tree-based model fits better than linear regression here.

## Per-State Models
Instead of one model for all states, I train five separate models. This works better because:

- NSW consumes 8000 MW, SA consumes 1500 MW - very different scales
- Each state has different climate and usage patterns
- One model spent most of its effort just distinguishing states
- Separate models learn temporal patterns more effectively


## What I Learned
Technical skills: how data leakage shows up in time series (inflated test scores, near-zero importance on the features that should matter most), why temporal train/test splits matter, and how feature importance rankings are a useful diagnostic tool not just a reporting metric.

Practical lessons: a 99% R2 is a warning sign, not a goal. The right question is whether the model is learning the right things, which requires looking at feature importance and the train/test gap alongside the headline metric.
 
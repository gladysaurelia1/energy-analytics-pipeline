# Overfitting Analysis

This folder documents my process of identifying and fixing overfitting issues in the energy forecasting models. It's a record of mistakes made and lessons learned. 

## The Problem

When I first trained the model, I got 99.4% accuracy. That should have been my first red flag, but honestly, I was excited to see such high numbers. It took me a while to realize something was fundamentally wrong.

---

## Version 1: The 99% Model (Extreme Overfitting)

File: train_model_v1_lag_features.py

### What I Did

Used these features:
```python
features = [
    'hour',
    'day_of_week',
    'month',
    'temperature',
    'consumption_lag_1h',      # Previous hour's consumption
    'consumption_lag_24h',     # Yesterday's consumption
    'consumption_rolling_mean' # Rolling average
]

R² Score: 99.44%
MAPE: 2.98%
Train-Test Gap: 0.3%

The lag features created data leakage. I was essentially asking the model "what will consumption be in the next hour?" and giving it "what was consumption in the previous hour?" as a feature.
consumption_rolling_mean: 41%
consumption_lag_1h: 24%
consumption_lag_24h: 22%
hour: 0.5%
The actual temporal features (hour, day, month) had almost no importance. That made no sense for energy consumption, which has clear daily and weekly patterns. 

```

##  Version 2: The 97% Model (State Encoding Dominance)
File: train_model_v2_state_encoding.py
## What I Changed
Removed all lag features and trained one model for all states:
```python
features = [
    'hour',
    'day_of_week',
    'month',
    'temperature',
    'is_business_hour',
    'state_encoded'  # NSW=0, VIC=1, QLD=2, SA=3, WA=4
]
Results
R² Score: 97.03%
MAPE: 10.96%
Train-Test Gap: 1.5%
Why This Still Had Issues
Better than version 1, but state encoding dominated everything:
Feature importance:

state_encoded: 93.22%
hour: 2.18%
All other features: <5% combined

The model was basically learning "NSW uses about 8000 MW, VIC uses 5500 MW, QLD uses 6500 MW" and ignoring when or why consumption varies.
I wanted one model to handle all states efficiently, but different states have vastly different consumption scales:

NSW: 6,000-10,000 MW
SA: 1,000-2,000 MW

A single model spent most of its capacity just remembering which state is which, rather than learning temporal patterns that apply across all states.
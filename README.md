# Startup Revenue Predictor — Hybrid XGBoost + LSTM Forecasting

A hybrid machine learning system that forecasts startup revenue by combining **XGBoost** (for structured, static features) with an **LSTM** network (for time-based funding sequences). The LSTM's output is fed as an input feature into XGBoost, creating an integrated hybrid model that outperforms either approach alone.

📄 Published in IEEE — *"Forecasting Startup Revenue Using XGBoost–LSTM Integrated Architecture"* (Lakshasree R, Shigee J, G Praveen Kumar — St. Joseph's Institute of Technology)

📄 Read the full paper: [IEEE Xplore](https://ieeexplore.ieee.org/document/11251949)

## Problem

Startups have irregular growth, volatile funding, and unpredictable market conditions — conditions where traditional financial forecasting models (linear regression, ARIMA) tend to fall short. This project addresses that gap using a hybrid approach that captures both static company attributes and time-dependent funding trends.

## Approach

- **XGBoost** — trained on structured, engineered features (funding amount, investor reputation, sector growth, market size, startup age, etc.) to capture non-linear feature interactions
- **LSTM** — a 2-layer network (64 units/layer, 0.2 dropout) trained on sliding 12-month funding sequences to capture temporal patterns
- **Hybrid integration** — the LSTM's predicted output is added as an input feature to the XGBoost model, combining time-series signal with structured business metrics

## Dataset

- **Structured dataset**: startup metadata — industry, funding type/stage, total investment, investor count, location, age, investor reputation, market size
- **Time-series dataset**: month-by-month/quarter-by-quarter funding history per startup
- Sourced from public startup funding data (Crunchbase public dumps, Kaggle repositories)
- Split: 80% training / 20% test, fixed random seed for reproducibility

## Results

| Model | MAE | RMSE | Accuracy |
|---|---|---|---|
| Linear Regression | 0.089 | 0.13 | 75% |
| XGBoost (alone) | 0.045 | 0.070 | 88% |
| LSTM (alone) | 0.038 | 0.060 | 90% |
| **Hybrid (XGBoost + LSTM)** | **0.032** | **0.050** | **92%** |

The hybrid model outperformed every individual baseline. Feature importance analysis showed **funding amount** as the strongest predictor, followed by investor reputation and sector growth rate.

## Tech Stack

Python 3.9 · pandas, NumPy · scikit-learn · XGBoost · TensorFlow/Keras · matplotlib, seaborn · joblib

## Project Structure
├── data/
│ ├── preprocessed_startup_dataset.csv
│ └── Preprocessed_time_series_data.csv
├── models/
│ ├── lstm_funding_model.keras
│ └── xgboost_startup_model_v3.json
├── predict_batch.py # Batch prediction from CSV input
├── predict_interactive.py # CLI tool for single manual prediction
└── requirements.txt

## How to Run

**Batch prediction (from CSV):**
```bash
pip install -r requirements.txt
python predict_batch.py
```

**Interactive single prediction:**
```bash
python predict_interactive.py
```

## Sample Run

**Interactive prediction example:**

```
Please enter the following startup details:

Total Funding (in M$): 2
Number of Investors: 2
Investor Reputation (0-10): 5
Growth Rate (%): 50
Current Revenue (in M$): 3
Market Size (in M$): 5
Years Since Founded: 4
Success Score (0-100): 80

Select Industry from the following:
1. AgriTech
2. Autonomous Vehicles
3. Cybersecurity
4. EdTech
5. FinTech
6. GreenTech
7. Healthcare
8. Quantum Computing
Enter the number corresponding to the industry: 5

Final Predicted Revenue (M$): 2.8155
```

## Future Improvements

- Real-time data integration (live funding news, macroeconomic indicators)
- Explore GRU/Transformer/attention-based architectures
- Explainable AI (XAI) techniques for greater model transparency
- Interactive dashboard for visualization and decision support
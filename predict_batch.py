import numpy as np
import pandas as pd
import xgboost as xgb
from tensorflow.keras.models import load_model
from sklearn.preprocessing import MinMaxScaler

# Load models
lstm_model = load_model("models/lstm_funding_model.keras")
xgb_model = xgb.Booster()
xgb_model.load_model("models/xgboost_startup_model_v3.json")

# ---- LSTM: predict next funding value from time series ----
df_lstm = pd.read_csv("data/Preprocessed_time_series_data.csv")

lstm_features = [
    "Funding Amount (M$)", "Number of Investors", "Investor Reputation",
    "Economic Indicator", "Sector Growth Rate (%)", "Estimated Valuation (M$)"
]

lstm_scaler = MinMaxScaler()
df_lstm[lstm_features] = lstm_scaler.fit_transform(df_lstm[lstm_features])

def create_sequences(data, seq_length=12):
    return np.array([data[i:i + seq_length] for i in range(len(data) - seq_length)])

X_lstm_seq = create_sequences(df_lstm[lstm_features].values)
lstm_prediction = lstm_model.predict(X_lstm_seq[-1:])[0][0]
print(f"LSTM Predicted Funding (M$): {lstm_prediction:.4f}")

# ---- XGBoost: predict revenue using structured features + LSTM output ----
df_xgb = pd.read_csv("data/preprocessed_startup_dataset.csv")
df_xgb["Predicted Funding (M$)"] = lstm_prediction

features_used = [
    'Total Funding (M$)', 'Number of Investors', 'Investor Reputation',
    'Growth Rate (%)', 'Revenue (M$)', 'Market Size', 'Years Since Founded',
    'Success Score', 'Industry_AgriTech', 'Industry_Autonomous Vehicles',
    'Industry_Cybersecurity', 'Industry_EdTech', 'Industry_FinTech',
    'Industry_GreenTech', 'Industry_Healthcare', 'Industry_Quantum Computing'
]

valid_features = [col for col in features_used if col in df_xgb.columns]
missing = set(features_used) - set(valid_features)
if missing:
    print(f"⚠️ Missing expected columns, skipping: {missing}")

dtest = xgb.DMatrix(df_xgb[valid_features])
df_xgb["Predicted Revenue (M$)"] = xgb_model.predict(dtest)

print(f"Final Predicted Revenue (M$) using hybrid model: {df_xgb['Predicted Revenue (M$)'].iloc[0]:.4f}")
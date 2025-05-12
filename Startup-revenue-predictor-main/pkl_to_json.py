# import xgboost as xgb
# xgb_model = xgb.Booster()
# xgb_model.load_model("xgboost_startup_model_v3.pkl")  # Ensure correct file path
# xgb_model.save_model("xgboost_startup_model.json")  # Save in new format

import xgboost as xgb

model = xgb.XGBRegressor()  # Change to XGBClassifier if needed
model.load_model("xgboost_startup_model_v3.json")
print("Model loaded successfully!")

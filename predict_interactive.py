import pandas as pd
import xgboost as xgb

xgb_model = xgb.Booster()
xgb_model.load_model("models/xgboost_startup_model_v3.json")

features_used = [
    'Total Funding (M$)', 'Number of Investors', 'Investor Reputation',
    'Growth Rate (%)', 'Revenue (M$)', 'Market Size', 'Years Since Founded',
    'Success Score', 'Industry_AgriTech', 'Industry_Autonomous Vehicles',
    'Industry_Cybersecurity', 'Industry_EdTech', 'Industry_FinTech',
    'Industry_GreenTech', 'Industry_Healthcare', 'Industry_Quantum Computing'
]

industries = [
    'AgriTech', 'Autonomous Vehicles', 'Cybersecurity', 'EdTech',
    'FinTech', 'GreenTech', 'Healthcare', 'Quantum Computing'
]

def get_user_input():
    print("Please enter the following startup details:\n")
    data = {
        'Total Funding (M$)': float(input("Total Funding (in M$): ")),
        'Number of Investors': int(input("Number of Investors: ")),
        'Investor Reputation': float(input("Investor Reputation (0-10): ")),
        'Growth Rate (%)': float(input("Growth Rate (%): ")),
        'Revenue (M$)': float(input("Current Revenue (in M$): ")),
        'Market Size': float(input("Market Size (in M$): ")),
        'Years Since Founded': int(input("Years Since Founded: ")),
        'Success Score': float(input("Success Score (0-100): ")),
    }

    print("\nSelect Industry from the following:")
    for idx, name in enumerate(industries):
        print(f"{idx + 1}. {name}")
    choice = int(input("Enter the number corresponding to the industry: "))

    for i, industry in enumerate(industries):
        data[f'Industry_{industry}'] = 1 if (i + 1) == choice else 0

    return data

def main():
    user_input = get_user_input()
    input_df = pd.DataFrame([user_input])
    dtest = xgb.DMatrix(input_df[features_used])
    predicted_revenue = xgb_model.predict(dtest)[0]
    print(f"\n✅ Final Predicted Revenue (M$): {predicted_revenue:.4f}")

if __name__ == "__main__":
    main()
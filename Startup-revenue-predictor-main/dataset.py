import pandas as pd
import random

# Unique startup name generator
adjectives = ["Smart", "Next", "Tech", "Inno", "Hyper", "Quantum", "Neuro", "Green", "Agri", "Fin"]
nouns = ["AI", "Cloud", "Data", "Finance", "Health", "Growth", "Sphere", "Lab", "Edge", "Connect"]
suffixes = ["Inc", "Solutions", "Tech", "Systems", "Hub", "Network", "Ventures", "AI", "Analytics", "Innovations"]

# Generate 10,000 unique startup names
startup_names = list([f"{random.choice(adjectives)}{random.choice(nouns)}{random.choice(suffixes)}" for _ in range(10000)])

industries = ["FinTech", "AI", "Healthcare", "AgriTech", "EdTech", "GreenTech", "Cybersecurity", "Autonomous Vehicles", "Quantum Computing"]
market_sizes = ["Small", "Medium", "Large"]
investors = ["Sequoia Capital", "Andreessen Horowitz", "Accel", "SoftBank", "Lightspeed", "Y Combinator", "GV", "Bessemer", "General Catalyst", "Insight Partners"]

# Create dataset
num_samples = len(startup_names)
data = []

for i in range(num_samples):
    startup = startup_names[i]  # Unique startup name
    industry = random.choice(industries)
    funding = round(random.uniform(0.5, 500), 2)  # Funding in million dollars
    num_investors = random.randint(1, 15)
    investor_rep = round(random.uniform(1, 10), 2)  # Reputation scale 1-10
    growth_rate = round(random.uniform(0.1, 80), 2)  # Growth rate in %
    revenue = round(random.uniform(0.1, 100), 2)  # Revenue in million dollars
    market_size = random.choice(market_sizes)
    years_since_founded = random.randint(1, 20)
    success_score = round(random.uniform(0, 100), 2)  # Target variable
    
    data.append([startup, industry, funding, num_investors, investor_rep, growth_rate, revenue, market_size, years_since_founded, success_score])
# Convert to DataFrame
columns = ["Startup Name", "Industry", "Total Funding (M$)", "Number of Investors", "Investor Reputation", 
           "Growth Rate (%)", "Revenue (M$)", "Market Size", "Years Since Founded", "Success Score"]

df = pd.DataFrame(data, columns=columns)

# Save dataset
df.to_csv("realistic_startups_dataset.csv", index=False)

print("Dataset saved as realistic_startups_dataset.csv ✅")

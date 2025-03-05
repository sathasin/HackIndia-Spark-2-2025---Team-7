import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("flights.csv")  # Change if needed

# Convert duration into total minutes
import pandas as pd

def convert_duration(duration):
    duration = duration.strip()
    hours, minutes = duration.split('h')
    minutes = minutes.replace('m', '').strip()
    if hours != '' and minutes != '':
        return float(hours) * 60 + float(minutes)
    elif hours != '':
        return float(hours) * 60
    elif minutes != '':
        return float(minutes)
    else:
        return 0
import pandas as pd
import numpy as np

def convert_time_to_minutes(time_str):
    """Converts 'HH:MM' format to total minutes since midnight."""
    if not time_str or pd.isna(time_str):  # Handle empty values
        return 0
    try:
        hours, minutes = map(int, time_str.split(':'))
        return hours * 60 + minutes
    except Exception:
        return 0  # Default to 0 in case of error

# Example Usage
df['departure_time'] = df['departure_time'].apply(convert_time_to_minutes)
df['arrival_time'] = df['arrival_time'].apply(convert_time_to_minutes)



df["duration"] = df["duration"].apply(convert_duration)

# Convert price (remove commas)
df["price"] = df["price"].astype(str).str.replace(",", "").astype(int)

# Encode categorical variables (airline, class, from, to, stops)
df = pd.get_dummies(df, columns=["airline", "class", "from", "to", "stops"])

# Features & Target
X = df.drop(columns=["flight_date", "flight_num", "price"])  # Remove non-numeric columns
y = df["price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Save model
with open("flight_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model trained and saved successfully!")

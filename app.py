from flask import Flask, request, jsonify, render_template
import pandas as pd
import pickle

app = Flask(__name__)

# Load the trained model
with open("flight_model.pkl", "rb") as f:
    model = pickle.load(f)

# Load flight data and preprocess
df = pd.read_csv("flights.csv")

# Convert price column (remove commas and convert to int)
df["price"] = df["price"].astype(str).str.replace(",", "").astype(int)

# Create a dictionary for fast route lookup
flight_cache = {}
for _, row in df.iterrows():
    key = (row["from"], row["to"])
    if key not in flight_cache:
        flight_cache[key] = []
    flight_cache[key].append(row.to_dict())

# Sort flights by price in advance for each route
for key in flight_cache:
    flight_cache[key] = sorted(flight_cache[key], key=lambda x: x["price"])


@app.route("/")
def home():
    return render_template("index.html")


@app.route("/search", methods=["POST"])
def search_flights():
    departure = request.form["departure"]
    destination = request.form["destination"]

    key = (departure, destination)
    if key not in flight_cache:
        return jsonify({"message": "No flights found!"})

    return jsonify(flight_cache[key])


if __name__ == "__main__":
    app.run(debug=True)

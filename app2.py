import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report
import pickle
from flask import Flask, request, render_template
import webbrowser
import sqlite3
import re
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)

# Load the dataset
data = pd.read_csv(r'C:\Users\pagid\OneDrive\Desktop\real time 2024 project\train_delay_prediction.csv')

# Convert categorical variables to numeric
data['day_of_week'] = data['day_of_week'].astype('category').cat.codes
data['weather'] = data['weather'].astype('category').cat.codes
data['departure_station'] = data['departure_station'].astype('category').cat.codes
data['arrival_station'] = data['arrival_station'].astype('category').cat.codes
data['departure_time'] = data['departure_time'].astype('category').cat.codes
data['arrival_time'] = data['arrival_time'].astype('category').cat.codes

# Feature and target variables 
X = data.drop('delay', axis=1)
y = data['delay']

# Save feature names
feature_names = X.columns.tolist()
with open(r'C:\Users\pagid\OneDrive\Desktop\real time 2024 project\feature_names.pkl', 'wb') as f:
    pickle.dump(feature_names, f)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the features
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Save the scaler object
with open(r'C:\Users\pagid\OneDrive\Desktop\real time 2024 project\scalar.pkl', 'wb') as f:
    pickle.dump(scaler, f)

# Initialize the Random Forest Classifier
model = RandomForestClassifier(n_estimators=100, random_state=42)

# Train the model
model.fit(X_train, y_train)

# Evaluate the model
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
conf_matrix = confusion_matrix(y_test, y_pred)
class_report = classification_report(y_test, y_pred, zero_division=1)
print(f'Accuracy: {accuracy}')
print('Confusion Matrix:')
print(conf_matrix)
print('Classification Report:')
print(class_report)

# Save the model to a file
with open(r'C:\Users\pagid\OneDrive\Desktop\real time 2024 project\train_delay_predictor.pkl', 'wb') as f:
    pickle.dump(model, f)

# Flask application setup
app = Flask(__name__, template_folder=r'C:\Users\pagid\OneDrive\Desktop\real time 2024 project')

# Load the model, scaler, and feature names
with open(r'C:\Users\pagid\OneDrive\Desktop\real time 2024 project\train_delay_predictor.pkl', 'rb') as f:
    loaded_model = pickle.load(f)
with open(r'C:\Users\pagid\OneDrive\Desktop\real time 2024 project\scalar.pkl', 'rb') as f:
    scaler = pickle.load(f)
with open(r'C:\Users\pagid\OneDrive\Desktop\real time 2024 project\feature_names.pkl', 'rb') as f:
    feature_names = pickle.load(f)

# Create the SQLite database and table
conn = sqlite3.connect('schedules.db')
cursor = conn.cursor()
cursor.execute('''CREATE TABLE IF NOT EXISTS train_schedule (
    departure_time TEXT NOT NULL,
    departure_station TEXT NOT NULL,
    arrival_station TEXT NOT NULL,
    arrival_time TEXT NOT NULL,
    weather TEXT NOT NULL,
    day_of_week TEXT NOT NULL,
    predicted_delay REAL
)''')
conn.commit()

def insert_schedule(departure_time, departure_station, arrival_station, arrival_time, weather, day_of_week, predicted_delay):
    conn = sqlite3.connect('schedules.db')
    cursor = conn.cursor()
    cursor.execute("INSERT INTO train_schedule (departure_time, departure_station, arrival_station, arrival_time, weather, day_of_week, predicted_delay) VALUES (?, ?, ?, ?, ?, ?, ?)", 
                   (departure_time, departure_station, arrival_station, arrival_time, weather, day_of_week, predicted_delay))
    conn.commit()
    conn.close()

@app.route("/",methods=['GET'])
def home():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    if request.method == "POST":
        # Get user inputs from the form
        departure_time = request.form.get("departure_time")
        departure_station = request.form.get("departure_station")
        arrival_station = request.form.get("arrival_station")
        arrival_time = request.form.get("arrival_time")
        weather = request.form.get("weather")
        day_of_week = request.form.get("day_of_week")

        if not all([departure_time, departure_station, arrival_station, arrival_time, weather, day_of_week]):
            return render_template("index.html", error_message="Please fill out all fields.")

        # Data validation
        valid_days = range(7)  # Assuming day_of_week is coded from 0 (Sunday) to 6 (Saturday)
        
        time_format = re.compile(r'^\d{2}:\d{2}$')  # Regex for HH:MM format
        if not (time_format.match(departure_time) and time_format.match(arrival_time)):
            logging.debug("Invalid time format: departure_time = %s, arrival_time = %s", departure_time, arrival_time)
            return render_template("index.html", error_message="Invalid time format. Use HH:MM.")

        if not (int(day_of_week) in valid_days):
            logging.debug("Invalid day of the week: day_of_week = %s", day_of_week)
            return render_template("index.html", error_message="Invalid day of the week.")
        
        # Convert input data to appropriate types and match feature names
        input_data = pd.DataFrame({
            'departure_time': [departure_time],
            'departure_station': [departure_station],
            'arrival_station': [arrival_station],
            'arrival_time': [arrival_time],
            'weather': [weather],
            'day_of_week': [day_of_week]
        })

        input_data['departure_time'] = input_data['departure_time'].astype('category').cat.codes
        input_data['departure_station'] = input_data['departure_station'].astype('category').cat.codes
        input_data['arrival_station'] = input_data['arrival_station'].astype('category').cat.codes
        input_data['arrival_time'] = input_data['arrival_time'].astype('category').cat.codes
        input_data['weather'] = input_data['weather'].astype('category').cat.codes
        input_data['day_of_week'] = input_data['day_of_week'].astype('category').cat.codes

        # Reorder input data to match the feature names
        input_data = input_data[feature_names]

        # Scale the input data
        input_data_scaled = scaler.transform(input_data)

        # Predict using the loaded model
        predicted_delay = loaded_model.predict(input_data_scaled)[0]

        # Cross-verification with historical data
        conn = sqlite3.connect('schedules.db')
        cursor = conn.cursor()
        cursor.execute("""SELECT AVG(predicted_delay) FROM train_schedule 
                          WHERE departure_station = ? AND arrival_station = ? AND day_of_week = ? AND weather = ?""",
                       (departure_station, arrival_station, day_of_week, weather))
        historical_avg_delay = cursor.fetchone()[0]

        if historical_avg_delay is not None:
            confidence = abs(predicted_delay - historical_avg_delay) / historical_avg_delay
            if confidence > 0.5:  # Adjusted threshold for inconsistency
                logging.debug("Prediction inconsistency: predicted_delay = %s, historical_avg_delay = %s", predicted_delay, historical_avg_delay)
                inconsistency_message = f"Prediction is inconsistent with historical data. Historical average delay: {historical_avg_delay:.2f}. Predicted delay: {predicted_delay:.2f}."
                return render_template("result.html", prediction=predicted_delay, inconsistency_message=inconsistency_message)

        # Insert data into the database
        insert_schedule(departure_time, departure_station, arrival_station, arrival_time, weather, day_of_week, predicted_delay)

        return render_template("result.html", prediction=predicted_delay, inconsistency_message="")

if __name__ == "__main__":
    # Open the URL in a new tab
    webbrowser.open('http://127.0.0.1:5000', new=0)

    # Run the Flask app
    app.run(debug=True, use_reloader=False)

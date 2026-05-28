Real-Time Flight Delay Prediction Using OpenSky Network API & Machine Learning
A real-time flight delay prediction project that fetches live ADS-B aircraft states from the OpenSky Network API, cleans and processes the data, and applies machine learning models to predict flight delay and future velocity.

Project Overview
This project demonstrates an end-to-end data science and machine learning pipeline for aviation analytics. It continuously collects live aircraft data, engineers features from flight telemetry, and uses Random Forest regression models to generate predictions in near real time.

The system is designed as a proof of concept for real-time flight monitoring and delay estimation, with potential for scaling into a production-grade streaming architecture.

Features
Live data ingestion from the OpenSky Network REST API.

Real-time processing of aircraft state vectors.

Data cleaning and missing value handling.

Synthetic delay label generation for model simulation.

Dual Random Forest regressors for:

Future velocity prediction

Flight delay prediction

Streaming loop with repeated fetch-predict-display cycles.

Console-based live output with top aircraft summaries.

Aircraft change tracking between iterations.

Simple and interpretable architecture for aviation analytics.

Data Source
The project uses the OpenSky Network API:

Endpoint: https://opensky-network.org/api/states/all

Each response contains live aircraft states with attributes such as:

icao24

callsign

origin_country

longitude

latitude

baro_altitude

geo_altitude

velocity

true_track

vertical_rate

on_ground

Core Features Used
The machine learning model primarily uses:

velocity

baro_altitude

true_track

Additional attributes are used for identification, filtering, and display.

How It Works
Fetch live flight data from the OpenSky API.

Convert JSON response into a Pandas DataFrame.

Clean missing values in key columns.

Generate simulated delay labels using a clipped normal distribution.

Train two Random Forest regressors:

one for delay prediction

one for velocity prediction

Predict output for the current batch of flights.

Display the top 10 aircraft in a live table.

Repeat the process in a streaming loop.

Example Output
text
[14:32:05] Live Table Update (Loop 1):
---------------------------------------------------------------
Callsign     | Altitude   | Current Vel  | Pred Vel     | Delay (min)
---------------------------------------------------------------
BAW123       | 34000      | 250.35       | 261.90       | 13.7
DLH456       | 28050      | 230.10       | 241.42       | 9.8
AFR789       | 12000      | 190.55       | 199.96       | 21.4
...
📈 Avg Velocity: 312.42 m/s | Max Altitude: 38000 ft | Total: 8234 aircraft
Project Structure
text
.
├── opensky.py
├── README.md
└── report/
    └── project_report.pdf
Requirements
Python 3.8+

requests

pandas

numpy

scikit-learn

Installation
bash
git clone https://github.com/hemanthv358-svg/Big-Data-Analytics
cd flight-delay-prediction
pip install -r requirements.txt
Usage
Run the Python script:

bash
python opensky.py
The script will:

connect to the OpenSky API,

fetch live aircraft states,

train the models on the current batch,

and print live predictions to the console.

Code Highlights
Fetch live data
python
resp = requests.get(OPENSKY_URL, timeout=15)
data = resp.json()
states = data.get("states", [])
df = pd.DataFrame(states, columns=columns)
Train and predict
python
model_delay = RandomForestRegressor(n_estimators=10, random_state=42)
model_vel = RandomForestRegressor(n_estimators=10, random_state=42)
model_delay.fit(X, y_delay)
model_vel.fit(X, y_vel)
Streaming loop
python
for iteration in range(3):
    df = fetch_opensky_states()
    df = predict_flight_metrics(df)
    print(df.head(10))
    time.sleep(5)
Limitations
Delay labels are simulated, not historical real-world delays.

Models are trained on each batch rather than persisted.

Output is console-based instead of a graphical dashboard.

No weather, airport congestion, or route history is included.

Future Improvements
Integrate real delay data from aviation sources.

Add weather and airport congestion features.

Persist trained models for reuse.

Build an interactive dashboard using Streamlit or Dash.

Scale the architecture using Kafka and Spark Streaming.

Benefits
This project is useful for:

aviation analytics,

real-time monitoring,

delay estimation,

machine learning experimentation on live data,

and demonstrating streaming data pipelines.

Author
Your Name
M.Tech in Data Science
JSS Science and Technology University

License
This project is for academic and research purposes.

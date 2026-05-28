import os
import time
from datetime import datetime

import requests
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
import warnings

warnings.filterwarnings("ignore")

# ==========================
# 1. AUTH & CONFIG
# ==========================

AUTH_URL = "https://auth.opensky-network.org/auth/realms/opensky-network/protocol/openid-connect/token"
OPENSKY_URL = "https://opensky-network.org/api/states/all"

# READ FROM ENVIRONMENT (terminal)
CLIENT_ID = os.getenv("OPENSKY_CLIENT_ID")
CLIENT_SECRET = os.getenv("OPENSKY_CLIENT_SECRET")


def get_opensky_token():
    """
    Get OAuth2 access token using client_credentials flow.
    """
    if not CLIENT_ID or not CLIENT_SECRET:
        raise RuntimeError(
            "OPENSKY_CLIENT_ID / OPENSKY_CLIENT_SECRET not set.\n"
            "Set them in terminal before running:\n"
            "  Windows PowerShell:  $env:OPENSKY_CLIENT_ID='...' ; $env:OPENSKY_CLIENT_SECRET='...'\n"
            "  CMD:                 set OPENSKY_CLIENT_ID=... & set OPENSKY_CLIENT_SECRET=..."
        )

    data = {
        "grant_type": "client_credentials",
        "client_id": CLIENT_ID,
        "client_secret": CLIENT_SECRET,
    }
    headers = {"Content-Type": "application/x-www-form-urlencoded"}

    resp = requests.post(AUTH_URL, data=data, headers=headers, timeout=15)

    if resp.status_code != 200:
        # Print useful debug info when 400 happens
        print("Auth error:")
        print("Status:", resp.status_code)
        print("Body:", resp.text)
        raise requests.HTTPError(f"Auth failed with status {resp.status_code}")

    token = resp.json()["access_token"]
    return token


# ==========================
# 2. DATA FETCH
# ==========================

def fetch_opensky_states(token: str) -> pd.DataFrame:
    """
    Fetch global flight states from OpenSky API using Bearer token auth.
    """
    headers = {
        "Authorization": f"Bearer {token}"
    }

    try:
        resp = requests.get(OPENSKY_URL, headers=headers, timeout=20)
        resp.raise_for_status()
        data = resp.json()
        states = data.get("states", [])

        if not states:
            return pd.DataFrame()

        columns = [
            "icao24", "callsign", "origin_country", "time_position",
            "last_contact", "longitude", "latitude", "baro_altitude",
            "on_ground", "velocity", "true_track", "vertical_rate",
            "sensors", "geo_altitude", "squawk", "spi", "position_source"
        ]

        df = pd.DataFrame(states, columns=columns)
        return df

    except Exception as e:
        print(f"Error fetching data: {e}")
        return pd.DataFrame()


# ==========================
# 3. ML: PREDICT DELAY & FUTURE VELOCITY
# ==========================

def predict_flight_metrics(df: pd.DataFrame) -> pd.DataFrame:
    """
    Clean data, simulate delay labels, train 2 Random Forest regressors
    and predict delay + future velocity for each aircraft.
    """
    if df.empty:
        return df

    df["callsign"] = df["callsign"].fillna("").astype(str)
    df = df[df["callsign"].str.strip() != ""]

    if df.empty:
        return df

    if df["velocity"].isna().all():
        df["velocity"] = 0
    else:
        df["velocity"] = df["velocity"].fillna(df["velocity"].median())

    if df["baro_altitude"].isna().all():
        df["baro_altitude"] = 0
    else:
        df["baro_altitude"] = df["baro_altitude"].fillna(df["baro_altitude"].median())

    df["true_track"] = df["true_track"].fillna(0)

    np.random.seed(42)
    df["sim_delay"] = np.random.normal(15, 10, len(df)).clip(0, 120)

    features = ["velocity", "baro_altitude", "true_track"]
    X = df[features]
    y_delay = df["sim_delay"]
    y_vel_future = df["velocity"] * (1 + np.random.normal(0, 0.05, len(df)))

    model_delay = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)
    model_vel = RandomForestRegressor(n_estimators=100, max_depth=10, random_state=42)

    model_delay.fit(X, y_delay)
    model_vel.fit(X, y_vel_future)

    df["pred_delay"] = model_delay.predict(X)
    df["pred_vel"] = model_vel.predict(X)

    return df


# ==========================
# 4. STREAMING LOOP
# ==========================

def run_prediction_stream():
    print("=" * 80)
    print(f"LIVE FLIGHT PREDICTION STREAM (AUTHENTICATED) | START: {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)

    # Get token (will print auth error body if 400)
    token = get_opensky_token()

    max_iterations = 3
    iteration = 1
    prev_df = None

    while iteration <= max_iterations:
        try:
            print(f"\n[Iteration {iteration}] Fetching live data with API token...")
            df = fetch_opensky_states(token)

            if df.empty:
                print("No data received. Retrying in 5 seconds...")
                time.sleep(5)
                continue

            df = predict_flight_metrics(df)
            print(f"✓ Success! Processed {len(df)} aircraft states.")

            if prev_df is not None and not prev_df.empty:
                prev_callsigns = set(prev_df["callsign"].str.strip())
                curr_callsigns = set(df["callsign"].str.strip())
                new_aircraft = curr_callsigns - prev_callsigns
                removed_aircraft = prev_callsigns - curr_callsigns
                if new_aircraft or removed_aircraft:
                    print(f"📊 Changes: {len(new_aircraft)} new aircraft, {len(removed_aircraft)} departed")

            prev_df = df.copy()

            print(f"\n[{datetime.now().strftime('%H:%M:%S')}] Live Prediction Table:")
            header = f"{'Callsign':<12} | {'Altitude(ft)':<12} | {'Current Vel':<12} | {'Pred Vel':<12} | {'Delay(min)':<10}"
            print("-" * len(header))
            print(header)
            print("-" * len(header))

            for _, row in df.head(10).iterrows():
                c = str(row["callsign"]).strip()
                alt = float(row["baro_altitude"]) if pd.notna(row["baro_altitude"]) else 0
                cv = float(row["velocity"]) if pd.notna(row["velocity"]) else 0
                pv = float(row["pred_vel"]) if pd.notna(row["pred_vel"]) else 0
                pd_val = float(row["pred_delay"]) if pd.notna(row["pred_delay"]) else 0
                print(f"{c:<12} | {alt:<12.0f} | {cv:<12.2f} | {pv:<12.2f} | {pd_val:<10.1f}")

            avg_vel = df["velocity"].mean()
            max_alt = df["baro_altitude"].max()
            avg_delay = df["pred_delay"].mean()

            print(f"\n📈 Avg Velocity: {avg_vel:.2f} m/s | Max Altitude: {max_alt:.0f} ft "
                  f"| Avg Pred Delay: {avg_delay:.1f} min | Total Aircraft: {len(df)}")

            iteration += 1
            print("\n⏳ Updating in 5 seconds...\n")
            time.sleep(5)

        except Exception as e:
            print(f"Stream Error: {e}")
            print("Retrying in 5 seconds...")
            time.sleep(5)

    print("\n" + "=" * 80)
    print(f"✓ Completed {max_iterations} iterations at {datetime.now().strftime('%H:%M:%S')}")
    print("=" * 80)


if __name__ == "__main__":
    run_prediction_stream()

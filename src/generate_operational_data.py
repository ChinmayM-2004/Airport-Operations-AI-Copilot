from pathlib import Path

import numpy as np
import pandas as pd


# ============================================================
# PROJECT PATHS
# ============================================================

PROJECT_ROOT = Path(__file__).resolve().parent.parent

OUTPUT_DIR = PROJECT_ROOT / "data" / "operational_data"
OUTPUT_FILE = OUTPUT_DIR / "airport_telemetry.csv"


# ============================================================
# CONFIGURATION
# ============================================================

RANDOM_SEED = 42

AIRPORTS = {
    "SFO": {
        "base_completion_rate": 0.94,
        "base_eta": 8.0,
        "base_active_drivers": 145,
        "base_cancellation_rate": 0.04,
        "base_queue_size": 65,
        "base_surge": 1.2,
        "base_request_volume": 420,
    },
    "LAX": {
        "base_completion_rate": 0.92,
        "base_eta": 10.0,
        "base_active_drivers": 180,
        "base_cancellation_rate": 0.05,
        "base_queue_size": 82,
        "base_surge": 1.4,
        "base_request_volume": 510,
    },
    "JFK": {
        "base_completion_rate": 0.90,
        "base_eta": 11.0,
        "base_active_drivers": 165,
        "base_cancellation_rate": 0.06,
        "base_queue_size": 75,
        "base_surge": 1.5,
        "base_request_volume": 480,
    },
}

NUM_HOURS = 24


# ============================================================
# DATA GENERATION
# ============================================================

def generate_airport_telemetry():
    """
    Generate synthetic hourly airport operational telemetry
    for SFO, LAX, and JFK.
    """

    rng = np.random.default_rng(RANDOM_SEED)

    # Fixed starting timestamp makes the synthetic dataset
    # reproducible.
    timestamps = pd.date_range(
        start="2026-09-15 00:00:00",
        periods=NUM_HOURS,
        freq="h",
    )

    records = []

    for airport_code, config in AIRPORTS.items():

        for timestamp in timestamps:

            hour = timestamp.hour

            # ------------------------------------------------
            # Demand pattern
            # ------------------------------------------------
            # Morning and evening periods have higher demand.
            if 6 <= hour <= 9:
                demand_factor = 1.20
            elif 16 <= hour <= 20:
                demand_factor = 1.30
            elif 10 <= hour <= 15:
                demand_factor = 1.05
            else:
                demand_factor = 0.75

            request_volume = int(
                config["base_request_volume"]
                * demand_factor
                * rng.uniform(0.90, 1.10)
            )

            # ------------------------------------------------
            # Driver availability
            # ------------------------------------------------
            active_drivers = int(
                config["base_active_drivers"]
                * rng.uniform(0.90, 1.10)
            )

            # ------------------------------------------------
            # Queue size
            # ------------------------------------------------
            queue_size = int(
                config["base_queue_size"]
                * demand_factor
                * rng.uniform(0.85, 1.15)
            )

            # ------------------------------------------------
            # ETA
            # ------------------------------------------------
            # More demand and queue pressure can increase ETA.
            eta = (
                config["base_eta"]
                * demand_factor
                * rng.uniform(0.90, 1.12)
            )

            average_eta = round(max(3.0, eta), 2)

            # ------------------------------------------------
            # Driver cancellation rate
            # ------------------------------------------------
            cancellation_rate = (
                config["base_cancellation_rate"]
                + max(0, demand_factor - 1.0) * 0.015
                + rng.uniform(-0.008, 0.008)
            )

            driver_cancellation_rate = round(
                np.clip(cancellation_rate, 0.01, 0.15),
                4,
            )

            # ------------------------------------------------
            # Completion rate
            # ------------------------------------------------
            completion_rate = (
                config["base_completion_rate"]
                - max(0, demand_factor - 1.0) * 0.025
                - max(0, driver_cancellation_rate - 0.05) * 0.20
                + rng.uniform(-0.01, 0.01)
            )

            completion_rate = round(
                np.clip(completion_rate, 0.75, 0.99),
                4,
            )

            # ------------------------------------------------
            # Surge multiplier
            # ------------------------------------------------
            surge_multiplier = (
                config["base_surge"]
                + max(0, demand_factor - 1.0) * 0.30
                + rng.uniform(-0.10, 0.10)
            )

            surge_multiplier = round(
                np.clip(surge_multiplier, 1.0, 2.5),
                2,
            )

            records.append(
                {
                    "airport_code": airport_code,
                    "completion_rate": completion_rate,
                    "average_eta": average_eta,
                    "active_drivers": active_drivers,
                    "driver_cancellation_rate": driver_cancellation_rate,
                    "queue_size": queue_size,
                    "surge_multiplier": surge_multiplier,
                    "request_volume": request_volume,
                    "timestamp": timestamp,
                }
            )

    return pd.DataFrame(records)


# ============================================================
# VALIDATION
# ============================================================

def validate_dataset(df):
    """
    Validate the generated operational dataset.
    """

    required_columns = [
        "airport_code",
        "completion_rate",
        "average_eta",
        "active_drivers",
        "driver_cancellation_rate",
        "queue_size",
        "surge_multiplier",
        "request_volume",
        "timestamp",
    ]

    # Check required columns.
    missing_columns = [
        column
        for column in required_columns
        if column not in df.columns
    ]

    if missing_columns:
        raise ValueError(
            f"Missing required columns: {missing_columns}"
        )

    # Check airports.
    expected_airports = set(AIRPORTS.keys())
    actual_airports = set(df["airport_code"].unique())

    if actual_airports != expected_airports:
        raise ValueError(
            f"Unexpected airport codes: {actual_airports}"
        )

    # Check numeric ranges.
    if not df["completion_rate"].between(0, 1).all():
        raise ValueError("Completion rate contains invalid values.")

    if not df["driver_cancellation_rate"].between(0, 1).all():
        raise ValueError(
            "Driver cancellation rate contains invalid values."
        )

    if (df["average_eta"] <= 0).any():
        raise ValueError("Average ETA must be positive.")

    if (df["active_drivers"] < 0).any():
        raise ValueError("Active driver count cannot be negative.")

    if (df["queue_size"] < 0).any():
        raise ValueError("Queue size cannot be negative.")

    if (df["request_volume"] < 0).any():
        raise ValueError("Request volume cannot be negative.")

    if not df["surge_multiplier"].between(1.0, 2.5).all():
        raise ValueError("Surge multiplier is outside the allowed test range.")

    if df.isnull().any().any():
        raise ValueError("Dataset contains null values.")

    print("Dataset validation: PASSED")


# ============================================================
# SAVE DATASET
# ============================================================

def save_dataset(df):
    """
    Save the operational telemetry dataset as CSV.
    """

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    df.to_csv(
        OUTPUT_FILE,
        index=False,
    )

    print(f"Dataset saved to: {OUTPUT_FILE}")


# ============================================================
# MAIN
# ============================================================

def main():
    print("=" * 70)
    print("AIRPORT OPERATIONAL TELEMETRY GENERATION")
    print("=" * 70)

    df = generate_airport_telemetry()

    print(f"\nTotal records generated: {len(df)}")

    print("\nRecords by airport:")
    print(df["airport_code"].value_counts().sort_index())

    validate_dataset(df)

    save_dataset(df)

    print("\nDataset preview:")
    print(df.head(10).to_string(index=False))

    print("\nDataset information:")
    print(df.info())

    print("\n" + "=" * 70)
    print("OPERATIONAL DATASET GENERATION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
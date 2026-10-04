"""
ml/train_model.py

Trains the study-effectiveness prediction model.

Pipeline:
    1. Load data (synthetic CSV + any real study_sessions logged in SQLite).
    2. Preprocess: encode categoricals (subject, topic), handle missing values.
    3. Train/test split.
    4. Train a RandomForestRegressor.
    5. Evaluate with MAE, RMSE, R^2.
    6. Save the model AND the fitted encoders/feature order with joblib, so
       ml/predict.py can reproduce the exact same preprocessing at inference time.

RETRAINING:
    train_and_save() can be called at any time (e.g. from app.py once enough
    new study sessions exist - see RETRAIN_THRESHOLD). It always re-reads the
    latest data (CSV + DB), so each retrain incorporates every new session
    logged since the last training run.
"""

import os

# Must be set before numpy/scipy/scikit-learn import - see app.py for why.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import sys
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import joblib

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from ml.generate_dataset import generate_dataset  # noqa: E402

HERE = os.path.dirname(os.path.abspath(__file__))
DATA_CSV = os.path.join(HERE, "..", "data", "study_data.csv")
MODEL_PATH = os.path.join(HERE, "study_model.pkl")

FEATURE_COLUMNS = [
    "subject_enc",
    "topic_enc",
    "previous_marks",
    "topic_mastery",
    "difficulty_level",
    "day_of_week",
    "study_hour",
    "study_duration",
    "focus_level",
    "sleep_hours",
    "previous_session_score",
    "days_until_exam",
    "number_of_previous_sessions",
    "revision_count",
    "quiz_score",
]
TARGET_COLUMN = "study_effectiveness"

# Once this many new (real) study sessions have accumulated since the last
# training run, app.py will trigger an automatic retrain.
RETRAIN_THRESHOLD = 15


def load_real_sessions_from_db():
    """Pull real logged study sessions from SQLite, shaped like the CSV."""
    try:
        from database.database import get_all_sessions_df_ready, init_db
        init_db()  # make sure tables exist even if this script runs standalone
        rows = get_all_sessions_df_ready()
    except Exception:
        return pd.DataFrame()

    if not rows:
        return pd.DataFrame()

    records = []
    for r in rows:
        if r["effectiveness"] is None:
            continue  # skip sessions without a recorded outcome
        study_date = r["study_date"]
        try:
            dow = pd.to_datetime(study_date).dayofweek
        except Exception:
            dow = 0
        try:
            hour = int(str(r["start_time"]).split(":")[0])
        except Exception:
            hour = 18

        records.append({
            "student_id": r["student_id"],
            "subject": r["subject_name"],
            "topic": r["topic_name"],
            "previous_marks": r["previous_marks"] if r["previous_marks"] is not None else 60,
            "topic_mastery": r["topic_mastery"] if r["topic_mastery"] is not None else 40,
            "difficulty_level": r["difficulty_level"],
            "day_of_week": dow,
            "study_hour": hour,
            "study_duration": r["duration"],
            "focus_level": r["focus_level"],
            "sleep_hours": r["sleep_hours"] if r["sleep_hours"] is not None else 7,
            "previous_session_score": r["quiz_score"] if r["quiz_score"] is not None else 60,
            "days_until_exam": 14,  # unknown retroactively; neutral default
            "number_of_previous_sessions": 5,
            "revision_count": 1,
            "quiz_score": r["quiz_score"] if r["quiz_score"] is not None else 60,
            "study_effectiveness": r["effectiveness"],
            "is_synthetic": 0,
        })
    return pd.DataFrame(records)


def build_training_dataframe():
    """Combine synthetic CSV data with any real logged sessions."""
    if not os.path.exists(DATA_CSV):
        generate_dataset(n_students=40, sessions_per_student=60, out_path=DATA_CSV)

    df_synthetic = pd.read_csv(DATA_CSV)
    df_real = load_real_sessions_from_db()

    if not df_real.empty:
        df = pd.concat([df_synthetic, df_real], ignore_index=True, sort=False)
    else:
        df = df_synthetic

    return df


def preprocess(df):
    """Handle missing values and encode categorical columns."""
    df = df.copy()

    # --- missing value handling ---
    numeric_defaults = {
        "previous_marks": df["previous_marks"].median() if "previous_marks" in df else 60,
        "topic_mastery": df["topic_mastery"].median() if "topic_mastery" in df else 40,
        "sleep_hours": 7,
        "previous_session_score": df["previous_session_score"].median() if "previous_session_score" in df else 60,
        "quiz_score": df["quiz_score"].median() if "quiz_score" in df else 60,
        "number_of_previous_sessions": 0,
        "revision_count": 0,
        "days_until_exam": 14,
        "study_duration": 45,
        "focus_level": 5,
        "difficulty_level": 3,
        "day_of_week": 0,
        "study_hour": 18,
    }
    for col, default in numeric_defaults.items():
        if col in df.columns:
            df[col] = df[col].fillna(default)

    df["subject"] = df["subject"].fillna("Unknown")
    df["topic"] = df["topic"].fillna("Unknown")

    df = df.dropna(subset=[TARGET_COLUMN])

    # --- categorical encoding ---
    subject_encoder = LabelEncoder()
    topic_encoder = LabelEncoder()
    df["subject_enc"] = subject_encoder.fit_transform(df["subject"].astype(str))
    df["topic_enc"] = topic_encoder.fit_transform(df["topic"].astype(str))

    encoders = {"subject": subject_encoder, "topic": topic_encoder}
    return df, encoders


def train_and_save():
    df = build_training_dataframe()
    df, encoders = preprocess(df)

    X = df[FEATURE_COLUMNS]
    y = df[TARGET_COLUMN]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=12,
        min_samples_leaf=3,
        random_state=42,
        n_jobs=-1,
    )
    model.fit(X_train, y_train)

    preds = model.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    metrics = {"mae": round(mae, 3), "rmse": round(rmse, 3), "r2": round(r2, 3),
               "n_train": len(X_train), "n_test": len(X_test)}

    feature_importances = dict(
        sorted(zip(FEATURE_COLUMNS, model.feature_importances_),
               key=lambda kv: kv[1], reverse=True)
    )

    bundle = {
        "model": model,
        "encoders": encoders,
        "feature_columns": FEATURE_COLUMNS,
        "metrics": metrics,
        "feature_importances": feature_importances,
        "n_rows_trained_on": len(df),
    }
    joblib.dump(bundle, MODEL_PATH)

    print("=== Training complete ===")
    print(f"Rows used: {len(df)}  (train={len(X_train)}, test={len(X_test)})")
    print(f"MAE:  {mae:.3f}")
    print(f"RMSE: {rmse:.3f}")
    print(f"R^2:  {r2:.3f}")
    print("Top features:", list(feature_importances.items())[:5])
    print(f"Model saved to {MODEL_PATH}")

    return bundle


if __name__ == "__main__":
    train_and_save()

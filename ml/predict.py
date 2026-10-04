"""
ml/predict.py

Loads the trained model bundle (model + encoders + feature order) and exposes
predict_effectiveness() for use by the Flask app and the recommendation engine.
"""

import os
import joblib
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(HERE, "study_model.pkl")

_bundle_cache = None


def get_bundle(force_reload=False):
    """Load (and cache) the trained model bundle. Trains one if missing."""
    global _bundle_cache
    if _bundle_cache is not None and not force_reload:
        return _bundle_cache

    if not os.path.exists(MODEL_PATH):
        from ml.train_model import train_and_save
        _bundle_cache = train_and_save()
    else:
        _bundle_cache = joblib.load(MODEL_PATH)

    return _bundle_cache


def _encode_value(encoder, value):
    """Encode a category, falling back to the most common class if unseen."""
    value = str(value)
    if value in encoder.classes_:
        return int(encoder.transform([value])[0])
    # Unseen label (e.g. a brand-new subject/topic name): fall back to class 0
    return 0


def predict_effectiveness(subject, topic, previous_marks, topic_mastery,
                           difficulty_level, day_of_week, study_hour,
                           study_duration, focus_level, sleep_hours,
                           previous_session_score, days_until_exam,
                           number_of_previous_sessions, revision_count,
                           quiz_score=None):
    """
    Run the trained RandomForestRegressor on a single study-condition input
    and return a predicted effectiveness score in [0, 100].
    """
    bundle = get_bundle()
    model = bundle["model"]
    encoders = bundle["encoders"]
    feature_columns = bundle["feature_columns"]

    if quiz_score is None:
        quiz_score = previous_session_score  # reasonable proxy if not yet known

    row = {
        "subject_enc": _encode_value(encoders["subject"], subject),
        "topic_enc": _encode_value(encoders["topic"], topic),
        "previous_marks": previous_marks,
        "topic_mastery": topic_mastery,
        "difficulty_level": difficulty_level,
        "day_of_week": day_of_week,
        "study_hour": study_hour,
        "study_duration": study_duration,
        "focus_level": focus_level,
        "sleep_hours": sleep_hours,
        "previous_session_score": previous_session_score,
        "days_until_exam": days_until_exam,
        "number_of_previous_sessions": number_of_previous_sessions,
        "revision_count": revision_count,
        "quiz_score": quiz_score,
    }

    X = pd.DataFrame([row])[feature_columns]
    pred = model.predict(X)[0]
    return float(max(0.0, min(100.0, pred)))


def get_metrics():
    bundle = get_bundle()
    return bundle.get("metrics", {})


def get_feature_importances():
    bundle = get_bundle()
    return bundle.get("feature_importances", {})

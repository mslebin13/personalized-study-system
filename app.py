"""
app.py

Flask backend for the Personalized Study Recommendation System.
Wires together: SQLite (database/database.py), the ML model
(ml/predict.py, ml/train_model.py) and the recommendation/timetable engine
(ml/recommend.py) behind a simple session-based login.
"""

import os

# Must be set BEFORE numpy/scipy/scikit-learn are imported anywhere below.
# Works around a known scikit-learn/SciPy hang caused by nested OpenBLAS-inside-
# OpenMP threading, which shows up more often on newer Python builds where the
# thread pool sizing heuristics haven't been tuned/tested as much yet.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from datetime import datetime
from functools import wraps

from flask import (Flask, render_template, request, redirect, url_for,
                    session, flash, jsonify)
from werkzeug.security import generate_password_hash, check_password_hash

from database import database as db
from ml.predict import predict_effectiveness, get_metrics, get_feature_importances, get_bundle
from ml.train_model import train_and_save, RETRAIN_THRESHOLD
from ml.recommend import generate_recommendations, generate_timetable

app = Flask(__name__)

_DEV_DEFAULT_SECRET = "dev-secret-change-me"
app.secret_key = os.environ.get("STUDY_APP_SECRET", _DEV_DEFAULT_SECRET)
if app.secret_key == _DEV_DEFAULT_SECRET:
    print(
        "WARNING: STUDY_APP_SECRET is not set - using an insecure default. "
        "Set a real STUDY_APP_SECRET environment variable in production.",
        flush=True,
    )
else:
    # Only enforce HTTPS-only cookies once a real secret is configured, so
    # local development (no env var set) still works over plain http://.
    app.config["SESSION_COOKIE_SECURE"] = True
    app.config["SESSION_COOKIE_HTTPONLY"] = True
    app.config["SESSION_COOKIE_SAMESITE"] = "Lax"

db.init_db()

# Train an initial model on first run if one doesn't exist yet
get_bundle()


# ---------------------------------------------------------------------------
# Auth helpers
# ---------------------------------------------------------------------------

def login_required(view):
    @wraps(view)
    def wrapped(*args, **kwargs):
        if "student_id" not in session:
            flash("Please log in first.", "warning")
            return redirect(url_for("login"))
        return view(*args, **kwargs)
    return wrapped


def current_student():
    if "student_id" in session:
        return db.get_student_by_id(session["student_id"])
    return None


@app.context_processor
def inject_student():
    return {"current_student": current_student()}


# ---------------------------------------------------------------------------
# Auth routes
# ---------------------------------------------------------------------------

@app.route("/")
def index():
    if "student_id" in session:
        return redirect(url_for("dashboard"))
    return redirect(url_for("login"))


@app.route("/register", methods=["GET", "POST"])
def register():
    if request.method == "POST":
        name = request.form.get("name", "").strip()
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")

        if not name or not email or not password:
            flash("All fields are required.", "danger")
            return render_template("register.html")

        if db.get_student_by_email(email):
            flash("An account with that email already exists.", "danger")
            return render_template("register.html")

        password_hash = generate_password_hash(password)
        student_id = db.create_student(name, email, password_hash)
        session["student_id"] = student_id
        flash("Account created! Let's set up your subjects.", "success")
        return redirect(url_for("subjects"))

    return render_template("register.html")


@app.route("/login", methods=["GET", "POST"])
def login():
    if request.method == "POST":
        email = request.form.get("email", "").strip().lower()
        password = request.form.get("password", "")
        student = db.get_student_by_email(email)

        if student and check_password_hash(student["password_hash"], password):
            session["student_id"] = student["id"]
            flash(f"Welcome back, {student['name']}!", "success")
            return redirect(url_for("dashboard"))

        flash("Invalid email or password.", "danger")

    return render_template("login.html")


@app.route("/logout")
def logout():
    session.clear()
    flash("Logged out successfully.", "info")
    return redirect(url_for("login"))


# ---------------------------------------------------------------------------
# Dashboard
# ---------------------------------------------------------------------------

@app.route("/dashboard")
@login_required
def dashboard():
    student_id = session["student_id"]
    subjects = db.get_subjects(student_id)
    sessions = db.get_study_sessions(student_id, limit=5)
    upcoming_exams = db.get_upcoming_exams(student_id)
    all_topics = db.get_all_topics_for_student(student_id)

    weakest_topic = min(all_topics, key=lambda t: t["mastery"]) if all_topics else None
    weakest_subject = None
    if subjects:
        subj_mastery = {}
        for t in all_topics:
            subj_mastery.setdefault(t["subject_name"], []).append(t["mastery"])
        if subj_mastery:
            weakest_subject = min(subj_mastery.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))[0]

    today_str = datetime.now().date().isoformat()
    all_sessions = db.get_study_sessions(student_id)
    today_minutes = sum(s["duration"] for s in all_sessions if s["study_date"] == today_str)

    focus_vals = [s["focus_level"] for s in all_sessions if s["focus_level"] is not None]
    avg_focus = round(sum(focus_vals) / len(focus_vals), 1) if focus_vals else 0

    eff_vals = [s["effectiveness"] for s in all_sessions if s["effectiveness"] is not None]
    avg_effectiveness = round(sum(eff_vals) / len(eff_vals), 1) if eff_vals else 0

    recommendations = generate_recommendations(student_id, top_n=5) if all_topics else []

    return render_template(
        "dashboard.html",
        subjects=subjects,
        sessions=sessions,
        upcoming_exams=upcoming_exams,
        weakest_topic=weakest_topic,
        weakest_subject=weakest_subject,
        today_minutes=today_minutes,
        avg_focus=avg_focus,
        avg_effectiveness=avg_effectiveness,
        recommendations=recommendations,
        total_sessions=len(all_sessions),
    )


# ---------------------------------------------------------------------------
# Subjects & Topics
# ---------------------------------------------------------------------------

@app.route("/subjects", methods=["GET", "POST"])
@login_required
def subjects():
    student_id = session["student_id"]

    if request.method == "POST":
        form_type = request.form.get("form_type")

        if form_type == "add_subject":
            name = request.form.get("name", "").strip()
            difficulty = int(request.form.get("difficulty_level", 3))
            marks = float(request.form.get("marks", 0) or 0)
            exam_date = request.form.get("exam_date", "").strip()

            if name:
                subject_id = db.add_subject(student_id, name, difficulty, marks)
                if exam_date:
                    db.add_exam(subject_id, exam_date)
                flash(f"Added subject '{name}'.", "success")
            return redirect(url_for("subjects"))

        elif form_type == "add_topic":
            subject_id = int(request.form.get("subject_id"))
            topic_name = request.form.get("topic_name", "").strip()
            mastery = float(request.form.get("mastery", 30) or 30)
            if topic_name:
                db.add_topic(subject_id, topic_name, mastery)
                flash(f"Added topic '{topic_name}'.", "success")
            return redirect(url_for("subjects"))

        elif form_type == "update_marks":
            subject_id = int(request.form.get("subject_id"))
            marks = float(request.form.get("marks", 0) or 0)
            db.update_subject_marks(subject_id, marks)
            flash("Marks updated.", "success")
            return redirect(url_for("subjects"))

        elif form_type == "update_mastery":
            topic_id = int(request.form.get("topic_id"))
            mastery = float(request.form.get("mastery", 0) or 0)
            db.update_topic_mastery(topic_id, mastery)
            flash("Topic mastery updated.", "success")
            return redirect(url_for("subjects"))

    subject_rows = db.get_subjects(student_id)
    subjects_with_topics = []
    for s in subject_rows:
        topics = db.get_topics(s["id"])
        exam = db.get_next_exam(s["id"])
        subjects_with_topics.append({"subject": s, "topics": topics, "exam": exam})

    return render_template("subjects.html", subjects_with_topics=subjects_with_topics)


# ---------------------------------------------------------------------------
# Study session logger
# ---------------------------------------------------------------------------

@app.route("/study-session", methods=["GET", "POST"])
@login_required
def study_session():
    student_id = session["student_id"]
    subject_rows = db.get_subjects(student_id)

    if request.method == "POST":
        subject_id = int(request.form.get("subject_id"))
        topic_id = int(request.form.get("topic_id"))
        study_date = request.form.get("study_date")
        start_time = request.form.get("start_time")
        duration = int(request.form.get("duration"))
        focus_level = int(request.form.get("focus_level"))
        quiz_score_raw = request.form.get("quiz_score", "").strip()
        quiz_score = float(quiz_score_raw) if quiz_score_raw else None
        difficulty_level = int(request.form.get("difficulty_level"))
        sleep_hours = float(request.form.get("sleep_hours", 7) or 7)
        notes = request.form.get("notes", "").strip()

        subj = db.get_subject(subject_id)
        topic = db.get_topic(topic_id)
        exam = db.get_next_exam(subject_id)
        days_until_exam = 30
        if exam:
            try:
                exam_date = datetime.strptime(exam["exam_date"], "%Y-%m-%d").date()
                days_until_exam = max(0, (exam_date - datetime.now().date()).days)
            except Exception:
                pass

        prev_sessions = db.get_sessions_for_topic(student_id, topic_id)
        number_of_previous_sessions = len(prev_sessions)
        revision_count = number_of_previous_sessions

        study_hour = int(start_time.split(":")[0]) if start_time else 18
        day_of_week = datetime.strptime(study_date, "%Y-%m-%d").weekday() if study_date else 0

        # Use the ML model to estimate effectiveness for this actual session.
        # If the student later records a quiz_score, that becomes part of the
        # ground truth used in the next retrain (see /retrain and train_model.py).
        predicted = predict_effectiveness(
            subject=subj["name"],
            topic=topic["name"],
            previous_marks=subj["marks"] or 60,
            topic_mastery=topic["mastery"],
            difficulty_level=difficulty_level,
            day_of_week=day_of_week,
            study_hour=study_hour,
            study_duration=duration,
            focus_level=focus_level,
            sleep_hours=sleep_hours,
            previous_session_score=quiz_score if quiz_score is not None else 60,
            days_until_exam=days_until_exam,
            number_of_previous_sessions=number_of_previous_sessions,
            revision_count=revision_count,
            quiz_score=quiz_score,
        )

        # The stored "effectiveness" label blends the model estimate with the
        # actual quiz_score (when given) - this is the real-world outcome that
        # future retraining will learn from.
        if quiz_score is not None:
            effectiveness = round(0.4 * predicted + 0.6 * quiz_score, 1)
        else:
            effectiveness = round(predicted, 1)

        db.add_study_session(
            student_id=student_id,
            subject_id=subject_id,
            topic_id=topic_id,
            study_date=study_date,
            start_time=start_time,
            duration=duration,
            focus_level=focus_level,
            quiz_score=quiz_score,
            difficulty_level=difficulty_level,
            sleep_hours=sleep_hours,
            notes=notes,
            effectiveness=effectiveness,
        )

        # Nudge topic mastery gently based on how the session went
        mastery_delta = (effectiveness - 50) / 20.0  # roughly -2.5 to +2.5
        new_mastery = max(0, min(100, topic["mastery"] + mastery_delta))
        db.update_topic_mastery(topic_id, new_mastery)

        flash(f"Session logged! Predicted/actual effectiveness: {effectiveness:.0f}%", "success")

        # Auto-retrain when enough new real sessions have accumulated
        if db.count_study_sessions(student_id) % RETRAIN_THRESHOLD == 0:
            train_and_save()
            flash("Enough new data collected - the ML model was retrained!", "info")

        return redirect(url_for("study_session"))

    subjects_json = []
    for s in subject_rows:
        topics = db.get_topics(s["id"])
        subjects_json.append({
            "id": s["id"], "name": s["name"], "difficulty_level": s["difficulty_level"],
            "topics": [{"id": t["id"], "name": t["name"]} for t in topics],
        })

    recent_sessions = db.get_study_sessions(student_id, limit=10)
    return render_template(
        "study_session.html",
        subjects=subject_rows,
        subjects_json=subjects_json,
        recent_sessions=recent_sessions,
    )


# ---------------------------------------------------------------------------
# Timetable
# ---------------------------------------------------------------------------

@app.route("/timetable")
@login_required
def timetable():
    student_id = session["student_id"]
    entries = db.get_timetable(student_id)
    return render_template("timetable.html", entries=entries)


@app.route("/timetable/generate", methods=["POST"])
@login_required
def timetable_generate():
    student_id = session["student_id"]
    topics = db.get_all_topics_for_student(student_id)
    if not topics:
        flash("Add subjects and topics before generating a timetable.", "warning")
        return redirect(url_for("subjects"))

    generate_timetable(student_id, days_ahead=7, sessions_per_day=2)
    flash("Adaptive timetable generated from your latest study data.", "success")
    return redirect(url_for("timetable"))


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------

@app.route("/analytics")
@login_required
def analytics():
    student_id = session["student_id"]
    sessions = db.get_study_sessions(student_id)
    metrics = get_metrics()
    importances = get_feature_importances()

    # Insights derived from historical data
    insights = {}
    if sessions:
        hour_perf = {}
        day_perf = {}
        for s in sessions:
            try:
                hour = int(str(s["start_time"]).split(":")[0])
            except Exception:
                continue
            outcome = s["effectiveness"] if s["effectiveness"] is not None else s["quiz_score"]
            if outcome is None:
                continue
            hour_perf.setdefault(hour, []).append(outcome)
            try:
                dow = datetime.strptime(s["study_date"], "%Y-%m-%d").strftime("%A")
                day_perf.setdefault(dow, []).append(outcome)
            except Exception:
                pass

        if hour_perf:
            best_hour = max(hour_perf.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
            insights["best_hour"] = f"{best_hour[0]}:00"
        if day_perf:
            best_day = max(day_perf.items(), key=lambda kv: sum(kv[1]) / len(kv[1]))
            insights["best_day"] = best_day[0]

        durations = [s["duration"] for s in sessions]
        insights["avg_duration"] = round(sum(durations) / len(durations), 1)

        focus_vals = [s["focus_level"] for s in sessions if s["focus_level"] is not None]
        insights["avg_focus"] = round(sum(focus_vals) / len(focus_vals), 1) if focus_vals else 0

    return render_template(
        "analytics.html",
        sessions=sessions,
        metrics=metrics,
        importances=importances,
        insights=insights,
    )


@app.route("/api/chart-data")
@login_required
def chart_data():
    """JSON data feed for Chart.js on the analytics/dashboard pages."""
    student_id = session["student_id"]
    sessions = db.get_study_sessions(student_id)

    by_date = {}
    subject_perf = {}
    focus_vs_perf = []
    hour_perf = {}
    topic_mastery_data = []

    for s in sessions:
        by_date.setdefault(s["study_date"], 0)
        by_date[s["study_date"]] += s["duration"]

        outcome = s["effectiveness"] if s["effectiveness"] is not None else s["quiz_score"]
        if outcome is not None:
            subject_perf.setdefault(s["subject_name"], []).append(outcome)
            if s["focus_level"] is not None:
                focus_vs_perf.append({"x": s["focus_level"], "y": outcome})
            try:
                hour = int(str(s["start_time"]).split(":")[0])
                hour_perf.setdefault(hour, []).append(outcome)
            except Exception:
                pass

    subject_perf_avg = {k: round(sum(v) / len(v), 1) for k, v in subject_perf.items()}
    hour_perf_avg = {k: round(sum(v) / len(v), 1) for k, v in sorted(hour_perf.items())}

    topics = db.get_all_topics_for_student(student_id)
    for t in topics:
        topic_mastery_data.append({"topic": t["topic_name"], "mastery": t["mastery"]})

    sorted_dates = sorted(by_date.items())

    return jsonify({
        "study_hours_over_time": {
            "labels": [d for d, _ in sorted_dates],
            "values": [round(v / 60, 2) for _, v in sorted_dates],
        },
        "subject_performance": {
            "labels": list(subject_perf_avg.keys()),
            "values": list(subject_perf_avg.values()),
        },
        "focus_vs_performance": focus_vs_perf,
        "effectiveness_by_hour": {
            "labels": [f"{h}:00" for h in hour_perf_avg.keys()],
            "values": list(hour_perf_avg.values()),
        },
        "topic_mastery": topic_mastery_data,
    })


# ---------------------------------------------------------------------------
# Profile & model retraining
# ---------------------------------------------------------------------------

@app.route("/profile")
@login_required
def profile():
    student = current_student()
    total_sessions = db.count_study_sessions(student["id"])
    metrics = get_metrics()
    sessions_until_retrain = RETRAIN_THRESHOLD - (total_sessions % RETRAIN_THRESHOLD)
    return render_template(
        "profile.html",
        student=student,
        total_sessions=total_sessions,
        metrics=metrics,
        retrain_threshold=RETRAIN_THRESHOLD,
        sessions_until_retrain=sessions_until_retrain,
    )


@app.route("/retrain", methods=["POST"])
@login_required
def retrain():
    train_and_save()
    flash("Model retrained on the latest data.", "success")
    return redirect(url_for("profile"))


if __name__ == "__main__":
    app.run(debug=True, port=5000)

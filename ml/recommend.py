"""
ml/recommend.py

The personalized recommendation engine and adaptive timetable generator.

This module is intentionally NOT if/else hard-coded logic for "what to study".
Instead, for every candidate (topic, hour, duration) combination it asks the
trained ML model (ml.predict.predict_effectiveness) what effectiveness to
expect, and combines that prediction with a priority score built from the
student's real data (weak topics, exam proximity, difficulty, revision need).
The candidate hours themselves are also learned from the student's own
history (see best_hours_by_subject), so as a student logs more sessions and
the *actual outcomes* shift, both retraining AND the historical-hour lookup
shift the plan - this is what makes the timetable "adaptive": rerun
generate_timetable() and it reflects the latest data automatically.
"""

from datetime import datetime, timedelta
from collections import defaultdict
import statistics

from database import database as db
from ml.predict import predict_effectiveness

CANDIDATE_HOURS = [7, 9, 11, 14, 16, 18, 19, 20, 21]
CANDIDATE_DURATIONS = [30, 45, 60, 90]


def best_hours_by_subject(student_id):
    """
    Learn, per subject, which study hour has historically produced the best
    average effectiveness/quiz_score for THIS student. Falls back to no
    preference (None) if there isn't enough history yet.
    """
    sessions = db.get_study_sessions(student_id)
    by_subject_hour = defaultdict(list)

    for s in sessions:
        try:
            hour = int(str(s["start_time"]).split(":")[0])
        except Exception:
            continue
        outcome = s["effectiveness"] if s["effectiveness"] is not None else s["quiz_score"]
        if outcome is None:
            continue
        by_subject_hour[(s["subject_name"], hour)].append(outcome)

    subject_hour_avg = {
        key: statistics.mean(vals) for key, vals in by_subject_hour.items() if len(vals) >= 1
    }

    best = {}
    for (subject, hour), avg in subject_hour_avg.items():
        if subject not in best or avg > best[subject][1]:
            best[subject] = (hour, avg)

    return {subject: hour for subject, (hour, avg) in best.items()}


def _days_until(exam_date_str):
    try:
        exam_date = datetime.strptime(exam_date_str, "%Y-%m-%d").date()
        return max(0, (exam_date - datetime.now().date()).days)
    except Exception:
        return 30  # unknown exam -> treat as not urgent


def build_candidates_for_topic(student_id, topic_row, preferred_hour=None):
    """
    For one topic, ask the ML model to predict effectiveness across a grid of
    plausible (hour, duration) combinations, and return the best one plus its
    predicted effectiveness and a human-readable reason.
    """
    subject_id = topic_row["subject_id"]
    subject_name = topic_row["subject_name"]
    topic_id = topic_row["topic_id"]
    topic_name = topic_row["topic_name"]
    mastery = topic_row["mastery"]
    difficulty = topic_row["difficulty_level"]
    marks = topic_row["marks"] or 60

    exam = db.get_next_exam(subject_id)
    days_until_exam = _days_until(exam["exam_date"]) if exam else 30

    prev_sessions = db.get_sessions_for_topic(student_id, topic_id)
    number_of_previous_sessions = len(prev_sessions)
    revision_count = sum(1 for s in prev_sessions)
    previous_session_score = (
        statistics.mean([s["quiz_score"] for s in prev_sessions if s["quiz_score"] is not None])
        if any(s["quiz_score"] is not None for s in prev_sessions) else marks
    )
    avg_sleep = (
        statistics.mean([s["sleep_hours"] for s in prev_sessions if s["sleep_hours"]])
        if any(s["sleep_hours"] for s in prev_sessions) else 7
    )

    day_of_week = datetime.now().weekday()

    hours_to_try = list(CANDIDATE_HOURS)
    if preferred_hour is not None and preferred_hour not in hours_to_try:
        hours_to_try.append(preferred_hour)

    best = None
    for hour in hours_to_try:
        for duration in CANDIDATE_DURATIONS:
            pred = predict_effectiveness(
                subject=subject_name,
                topic=topic_name,
                previous_marks=marks,
                topic_mastery=mastery,
                difficulty_level=difficulty,
                day_of_week=day_of_week,
                study_hour=hour,
                study_duration=duration,
                focus_level=7,  # assume a reasonably focused session when planning
                sleep_hours=avg_sleep,
                previous_session_score=previous_session_score,
                days_until_exam=days_until_exam,
                number_of_previous_sessions=number_of_previous_sessions,
                revision_count=revision_count,
            )
            if best is None or pred > best["predicted_effectiveness"]:
                best = {
                    "hour": hour,
                    "duration": duration,
                    "predicted_effectiveness": pred,
                }

    # --- priority score (used to rank topics against each other) ----------
    weakness_score = max(0, 100 - mastery)           # weaker topic -> higher
    urgency_score = max(0, 30 - days_until_exam) * 2  # closer exam -> higher
    difficulty_score = difficulty * 8
    revision_need_score = max(0, 5 - number_of_previous_sessions) * 6

    priority = (
        weakness_score * 0.4
        + urgency_score * 0.3
        + difficulty_score * 0.15
        + revision_need_score * 0.15
    )

    reasons = []
    if mastery < 50:
        reasons.append(f"topic mastery is low ({mastery:.0f}%)")
    if exam:
        reasons.append(f"exam in {days_until_exam} day(s)")
    if preferred_hour is not None and best["hour"] == preferred_hour:
        reasons.append(f"you historically perform well around {preferred_hour}:00")
    if number_of_previous_sessions < 3:
        reasons.append("needs more revision - few past sessions")
    if difficulty >= 4:
        reasons.append("high difficulty topic")
    if not reasons:
        reasons.append("keeps steady progress on this topic")
    reasons.append(f"predicted effectiveness: {best['predicted_effectiveness']:.0f}%")

    return {
        "subject_id": subject_id,
        "subject_name": subject_name,
        "topic_id": topic_id,
        "topic_name": topic_name,
        "hour": best["hour"],
        "duration": best["duration"],
        "predicted_effectiveness": round(best["predicted_effectiveness"], 1),
        "priority": round(priority, 1),
        "reason": "; ".join(reasons).capitalize() + ".",
    }


def generate_recommendations(student_id, top_n=5):
    """Rank every topic the student has and return the top_n candidates."""
    topics = db.get_all_topics_for_student(student_id)
    if not topics:
        return []

    hour_prefs = best_hours_by_subject(student_id)

    candidates = []
    for t in topics:
        preferred_hour = hour_prefs.get(t["subject_name"])
        candidate = build_candidates_for_topic(student_id, t, preferred_hour)
        candidates.append(candidate)

    candidates.sort(key=lambda c: c["priority"], reverse=True)
    return candidates[:top_n]


def generate_timetable(student_id, days_ahead=7, sessions_per_day=2):
    """
    Build a fresh weekly timetable, replacing any existing one for this
    student. Spreads the highest-priority topics across the coming days so
    the same topic isn't necessarily repeated every single day, while still
    favouring the weakest / most urgent topics overall.
    """
    topics = db.get_all_topics_for_student(student_id)
    if not topics:
        return []

    hour_prefs = best_hours_by_subject(student_id)

    ranked = []
    for t in topics:
        preferred_hour = hour_prefs.get(t["subject_name"])
        ranked.append(build_candidates_for_topic(student_id, t, preferred_hour))
    ranked.sort(key=lambda c: c["priority"], reverse=True)

    db.clear_timetable(student_id)

    entries = []
    today = datetime.now().date()
    slots_needed = days_ahead * sessions_per_day
    # Cycle through ranked topics (repeating weaker/more urgent ones more often)
    pool = []
    while len(pool) < slots_needed and ranked:
        pool.extend(ranked)
    pool = pool[:slots_needed]

    idx = 0
    for day_offset in range(days_ahead):
        entry_date = today + timedelta(days=day_offset)
        used_hours_today = set()
        for _ in range(sessions_per_day):
            if idx >= len(pool):
                break
            cand = pool[idx]
            idx += 1

            hour = cand["hour"]
            while hour in used_hours_today and hour < 22:
                hour += 1
            used_hours_today.add(hour)

            start_time = f"{hour:02d}:00"
            end_hour = hour + cand["duration"] // 60
            end_minute = cand["duration"] % 60
            end_time = f"{min(end_hour, 23):02d}:{end_minute:02d}"

            db.add_timetable_entry(
                student_id=student_id,
                entry_date=entry_date.isoformat(),
                start_time=start_time,
                end_time=end_time,
                subject_id=cand["subject_id"],
                topic_id=cand["topic_id"],
                duration=cand["duration"],
                priority=cand["priority"],
                predicted_effectiveness=cand["predicted_effectiveness"],
                reason=cand["reason"],
            )
            entries.append(cand)

    return db.get_timetable(student_id)

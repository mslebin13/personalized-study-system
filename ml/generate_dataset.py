"""
ml/generate_dataset.py

Generates a realistic SYNTHETIC dataset of student study sessions to bootstrap
the ML model before real usage data exists.

WHY SYNTHETIC DATA:
A brand-new deployment has zero study sessions, and a supervised model needs
labelled examples to learn from. We simulate thousands of plausible study
sessions with an underlying (but noisy) relationship between study conditions
and "study_effectiveness", so the model has something real to learn from on
day one.

HOW REAL DATA REPLACES IT OVER TIME:
Every time an actual student logs a study session in the app, a new row is
appended to data/study_data.csv (see database.get_all_sessions_df_ready +
train_model.build_training_dataframe). Once enough genuine rows accumulate
(see RETRAIN_THRESHOLD in ml/train_model.py), the model is retrained on the
combined real + synthetic pool. As the ratio of real rows grows, you can
simply stop calling this generator again and, later, drop the synthetic rows
entirely (filter on the `is_synthetic` column) once you have enough real data
to train on its own - the training script already supports both.
"""

import numpy as np
import pandas as pd
import os

RNG = np.random.default_rng(42)

SUBJECTS = {
    "Mathematics": {"difficulty": 4, "topics": ["Algebra", "Calculus", "Integration", "Geometry", "Trigonometry"]},
    "Physics": {"difficulty": 4, "topics": ["Kinematics", "Thermodynamics", "Optics", "Electromagnetism"]},
    "Chemistry": {"difficulty": 3, "topics": ["Organic Chemistry", "Periodic Table", "Chemical Bonding", "Acids and Bases"]},
    "Biology": {"difficulty": 2, "topics": ["Cell Biology", "Genetics", "Human Physiology", "Ecology"]},
    "English": {"difficulty": 2, "topics": ["Grammar", "Essay Writing", "Comprehension", "Literature"]},
    "Computer Science": {"difficulty": 3, "topics": ["Data Structures", "Algorithms", "Databases", "Networking"]},
}


def _clip(x, lo, hi):
    return max(lo, min(hi, x))


def generate_dataset(n_students=40, sessions_per_student=60, out_path=None):
    """
    Build a synthetic study-session dataset.

    The generating process encodes believable relationships so the model has
    real signal to find, plus per-student idiosyncrasies (e.g. some students
    are morning people) and random noise:

      - Effectiveness rises with focus_level and adequate sleep.
      - Effectiveness has a sweet spot for duration (~45-75 min); very short
        or very long sessions are less effective (fatigue / not enough depth).
      - Effectiveness rises when topic_mastery is low-to-mid and difficulty is
        not too high relative to the student's previous_marks (i.e. the topic
        is "learnable" right now, not hopelessly hard).
      - Effectiveness rises as days_until_exam shrinks (mild urgency helps,
        within reason) and with more prior revision (number_of_previous_sessions,
        revision_count).
      - Each synthetic student has a personal "best study hour" - sessions
        closer to that hour are more effective. This is what lets the
        recommendation engine later learn per-student time-of-day patterns.
      - Gaussian noise is added so the relationship is realistic, not exact.
    """
    rows = []
    subject_names = list(SUBJECTS.keys())

    for student_id in range(1, n_students + 1):
        # Each synthetic student has personal traits
        preferred_hour = RNG.integers(6, 23)          # best study hour, 6am-10pm
        base_sleep = RNG.normal(7, 1.0)
        skill_multiplier = RNG.uniform(0.85, 1.15)     # general studying ability

        for _ in range(sessions_per_student):
            subject = RNG.choice(subject_names)
            info = SUBJECTS[subject]
            topic = RNG.choice(info["topics"])
            difficulty_level = info["difficulty"] + RNG.integers(-1, 2)
            difficulty_level = int(_clip(difficulty_level, 1, 5))

            previous_marks = _clip(RNG.normal(65, 15), 20, 100)
            topic_mastery = _clip(RNG.normal(45, 20), 0, 100)

            study_hour = int(_clip(RNG.normal(preferred_hour, 2.5), 6, 23))
            day_of_week = int(RNG.integers(0, 7))  # 0=Mon
            study_duration = int(_clip(RNG.normal(60, 25), 15, 150))

            focus_level = int(_clip(RNG.normal(6.5, 2.0), 1, 10))
            sleep_hours = _clip(RNG.normal(base_sleep, 1.0), 3, 10)

            previous_session_score = _clip(RNG.normal(60, 18), 0, 100)
            number_of_previous_sessions = int(_clip(RNG.poisson(5), 0, 40))
            revision_count = int(_clip(RNG.poisson(1.5), 0, 15))
            days_until_exam = int(_clip(RNG.exponential(12), 0, 90))

            # --- underlying "true" effectiveness function ------------------
            score = 50.0

            # focus is the strongest driver
            score += (focus_level - 5) * 4.5

            # duration sweet spot around 45-75 minutes
            duration_penalty = abs(study_duration - 60) * 0.25
            score -= duration_penalty

            # sleep matters
            score += (sleep_hours - 6) * 3.0

            # time-of-day match to this student's personal best hour
            hour_distance = abs(study_hour - preferred_hour)
            score -= hour_distance * 1.8

            # mild exam urgency helps focus, too close becomes stressful
            if days_until_exam <= 1:
                score -= 6
            elif days_until_exam <= 10:
                score += (10 - days_until_exam) * 0.6
            else:
                score -= 2

            # topic mastery: too low = harder to be effective yet, too high = little to gain
            score += (topic_mastery - 50) * -0.15 + 8

            # difficulty vs previous_marks: hard topic + weak marks = tougher session
            skill_gap = difficulty_level * 10 - previous_marks * 0.3
            score -= skill_gap * 0.4

            # revision and repeated exposure help
            score += min(number_of_previous_sessions, 15) * 0.5
            score += min(revision_count, 8) * 0.8

            # previous performance carries over somewhat
            score += (previous_session_score - 60) * 0.15

            # personal ability
            score *= skill_multiplier

            # noise
            score += RNG.normal(0, 6)

            effectiveness = round(_clip(score, 0, 100), 1)

            # quiz_score correlates with effectiveness but is its own noisy signal
            quiz_score = round(_clip(effectiveness + RNG.normal(0, 10), 0, 100), 1)

            rows.append({
                "student_id": student_id,
                "subject": subject,
                "topic": topic,
                "previous_marks": round(previous_marks, 1),
                "topic_mastery": round(topic_mastery, 1),
                "difficulty_level": difficulty_level,
                "day_of_week": day_of_week,
                "study_hour": study_hour,
                "study_duration": study_duration,
                "focus_level": focus_level,
                "sleep_hours": round(sleep_hours, 1),
                "previous_session_score": round(previous_session_score, 1),
                "days_until_exam": days_until_exam,
                "number_of_previous_sessions": number_of_previous_sessions,
                "revision_count": revision_count,
                "quiz_score": quiz_score,
                "study_effectiveness": effectiveness,
                "is_synthetic": 1,
            })

    df = pd.DataFrame(rows)

    if out_path:
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
        df.to_csv(out_path, index=False)
        print(f"Saved {len(df)} synthetic study sessions to {out_path}")

    return df


if __name__ == "__main__":
    here = os.path.dirname(__file__)
    out_path = os.path.join(here, "..", "data", "study_data.csv")
    df = generate_dataset(n_students=40, sessions_per_student=60, out_path=out_path)
    print(df.describe(include="all").transpose())

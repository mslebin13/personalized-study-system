# Personalized Study Recommendation System with Adaptive Timetable

A full-stack web app that uses a **trained scikit-learn machine learning model**
(RandomForestRegressor) to predict study effectiveness and generate personalized,
adaptive study recommendations and timetables — not rule-based if/else logic.

## Architecture

```
Browser (Bootstrap + Chart.js)
        │
        ▼
Flask app (app.py) ── session-based auth
        │
        ├── database/database.py ─── SQLite (students, subjects, topics,
        │                             exams, study_sessions, recommendations,
        │                             timetable)
        │
        └── ml/
             ├── generate_dataset.py  → synthetic training data (bootstrap)
             ├── train_model.py       → preprocessing + RandomForestRegressor
             │                          training + evaluation + joblib save
             ├── predict.py           → loads the saved model, exposes
             │                          predict_effectiveness()
             └── recommend.py         → recommendation engine + adaptive
                                        timetable builder (uses predict.py +
                                        each student's real history)
```

### How the ML actually drives the app (not fake / not rule-based)

1. **Training** (`ml/train_model.py`): loads the synthetic dataset (plus any
   real study sessions already logged in SQLite), encodes `subject`/`topic`
   with `LabelEncoder`, fills missing values, splits 80/20 train/test, trains
   a `RandomForestRegressor`, evaluates with **MAE, RMSE, R²**, and saves the
   model + encoders + feature order with **joblib** to `ml/study_model.pkl`.
2. **Inference** (`ml/predict.py`): every prediction — when you log a session,
   when the dashboard shows "today's recommendations", when the timetable is
   generated — calls `model.predict(...)` on the actual trained model.
3. **Recommendation engine** (`ml/recommend.py`): for each topic the student
   has, it asks the model for the predicted effectiveness across a grid of
   candidate study hours/durations, and combines that with a data-driven
   priority score (topic mastery, days until exam, difficulty, revision
   need). **No recommendation text is hard-coded** — the reasons are built
   dynamically from the student's real numbers.
4. **Adaptivity**: `best_hours_by_subject()` looks at the student's own past
   sessions to find which hour has actually produced the best outcomes for
   each subject, and feeds that into the model as a candidate. So if a
   student performs poorly at 6 PM but well at 8 PM, later timetable
   regenerations will shift toward 8 PM — automatically, from data.
5. **Retraining**: every `RETRAIN_THRESHOLD` (15) new study sessions logged
   by a student, `app.py` automatically calls `train_and_save()` again, which
   re-reads the **latest real study_sessions from SQLite** combined with the
   synthetic data and retrains the model, saving a fresh `study_model.pkl`.
   You can also click "Retrain Model Now" on the Profile page.

### About the synthetic dataset

`ml/generate_dataset.py` creates ~2,400 realistic synthetic study sessions
(40 simulated students × 60 sessions) with a believable — but noisy —
underlying relationship between focus, sleep, duration, topic mastery, exam
proximity, revision count, and effectiveness, including per-student
"preferred study hour" patterns. This gives the model real signal to learn
from before any real student has used the app.

As real students log sessions, `train_model.py`'s `load_real_sessions_from_db()`
pulls those rows straight from SQLite and blends them into every retrain
(marked `is_synthetic = 0` vs `1`). Over time you can:
- Keep blending both (default — real data gradually dominates as it grows), or
- Filter `is_synthetic == 0` in `build_training_dataframe()` once you have
  enough real sessions to train on genuine data alone.

## Setup

```bash
cd personalized-study-system
pip install -r requirements.txt

# (Optional) explicitly regenerate the dataset and retrain from scratch:
python ml/generate_dataset.py
python ml/train_model.py

# Run the app (auto-trains a model on first run if study_model.pkl is missing)
python app.py
```

Then open **http://127.0.0.1:5000** in your browser.

A pre-trained `ml/study_model.pkl` and `data/study_data.csv` are already
included, so `python app.py` works immediately without any extra steps.

## Using the app

1. **Register** an account.
2. Go to **Subjects** → add a subject (with difficulty, marks, exam date),
   then add topics under it with a starting mastery %.
3. Go to **Log Session** → record a real study session (subject, topic,
   date/time, duration, focus level, optional quiz score). The ML model
   predicts/blends an effectiveness score for that session and nudges the
   topic's mastery.
4. Go to **Timetable** → click "Regenerate Timetable" to get a 7-day plan
   built from ML predictions across candidate times/durations.
5. Go to **Analytics** → see charts (study hours over time, subject
   performance, focus vs. performance, effectiveness by hour, topic mastery)
   plus the model's MAE/RMSE/R² and top feature importances.
6. Go to **Profile** → see how many sessions until the next automatic
   retrain, or trigger one manually.

## Project structure

```
personalized-study-system/
├── app.py                  # Flask app (routes, auth, wiring)
├── requirements.txt
├── README.md
├── database/
│   └── database.py         # SQLite schema + all DB access functions
├── ml/
│   ├── generate_dataset.py # synthetic data generator
│   ├── train_model.py      # preprocessing, training, evaluation, saving
│   ├── predict.py          # loads model, runs predictions
│   ├── recommend.py        # recommendation engine + timetable builder
│   └── study_model.pkl     # trained model bundle (joblib)
├── templates/               # Jinja2 + Bootstrap templates
├── static/
│   ├── css/style.css
│   └── js/script.js
└── data/
    └── study_data.csv       # synthetic training dataset
```

## Notes / next steps for production use

- Passwords are hashed with Werkzeug's `generate_password_hash` — fine for a
  learning project; for production, add rate limiting, HTTPS, and stronger
  session security.
- `app.secret_key` should be set via the `STUDY_APP_SECRET` environment
  variable in production instead of the default dev key.
- `RETRAIN_THRESHOLD` in `ml/train_model.py` controls how often the model
  auto-retrains; tune it based on real usage volume.

## Deploying to a public URL (Render)

This project is deployment-ready: it includes `requirements.txt` (with
Gunicorn + psycopg2), a `Procfile`, a `render.yaml` Blueprint, `.gitignore`,
and `.env.example`. The database layer (`database/database.py`) auto-detects
a `DATABASE_URL` environment variable and uses PostgreSQL when present,
falling back to local SQLite when it's absent - no code changes needed to
switch between them.

### 1. Push this code to GitHub

```bash
cd personalized-study-system
git init
git add .
git commit -m "Production-ready: Gunicorn, PostgreSQL support, deployment config"
git branch -M main
git remote add origin https://github.com/<your-username>/<your-repo>.git
git push -u origin main
```

### 2. Deploy on Render

**Option A - one-click Blueprint (recommended, uses render.yaml):**
1. Go to https://dashboard.render.com → **New** → **Blueprint**.
2. Connect your GitHub account and select this repository.
3. Render reads `render.yaml` automatically and sets up both the web service
   and a free PostgreSQL database, wiring `DATABASE_URL` between them and
   generating a random `STUDY_APP_SECRET` for you.
4. Click **Apply** - it builds and deploys automatically.

**Option B - manual setup:**
1. **New** → **Web Service** → connect your GitHub repo.
2. **Build Command:** `pip install -r requirements.txt`
3. **Start Command:** `gunicorn app:app --bind 0.0.0.0:$PORT`
4. Under **Environment**, add:
   - `STUDY_APP_SECRET` - a long random string (generate with
     `python -c "import secrets; print(secrets.token_hex(32))"`)
   - `DATABASE_URL` - leave unset to use SQLite (fine for a demo, but Render's
     disk is ephemeral so data can be wiped on redeploy), or create a
     **PostgreSQL** instance in Render and paste its **Internal Database URL**
     here for persistent storage.
5. Click **Create Web Service**.

### 3. After deploying

- Render gives you a public URL like `https://smart-study-ai.onrender.com`.
- Open it, register an account, and use the app exactly as locally.
- Each registered user only ever sees their own subjects/sessions/timetable -
  everything is scoped by `student_id` from their login session, and Flask's
  session cookie is signed with your `STUDY_APP_SECRET`, so no two users can
  see each other's data.
- The pre-trained `ml/study_model.pkl` ships in the repo, so the ML model
  works immediately on first deploy - no separate training step needed.

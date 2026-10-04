"""
database/database.py

Data access for the Personalized Study Recommendation System.

Supports two backends, chosen automatically from the environment:
  - PostgreSQL, when a DATABASE_URL environment variable is set (production -
    e.g. Render's managed Postgres add-on sets this for you).
  - SQLite, as a local-development fallback when DATABASE_URL is not set
    (this is the original behavior, unchanged).

Every other module (app.py, ml/*.py) talks to the database only through the
functions in this file - nothing else should write raw SQL outside this module.
The functions below are written once against a small sqlite3-shaped interface
(conn.execute()/cur.execute(), .fetchone(), .fetchall(), .lastrowid) - a thin
wrapper around psycopg2 (_PGConnection/_PGCursor below) reproduces that same
interface for Postgres, so none of the 30+ query functions below had to change.
"""

import sqlite3
import os
from datetime import datetime

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "study_system.db")
DB_PATH = os.path.abspath(DB_PATH)

DATABASE_URL = os.environ.get("DATABASE_URL")
USE_POSTGRES = bool(DATABASE_URL)


def _today_str():
    """Current date as an ISO string (YYYY-MM-DD) - portable across both DBs,
    used instead of SQLite's date('now') or Postgres's CURRENT_DATE."""
    return datetime.now().date().isoformat()


class _PGCursor:
    """Wraps a psycopg2 cursor so it behaves like a sqlite3 cursor for our
    call sites: '?' placeholders, and .lastrowid after an INSERT."""

    def __init__(self, raw_cursor):
        self._cur = raw_cursor
        self.lastrowid = None

    @staticmethod
    def _translate(query):
        # Our queries never contain a literal '?' in string data, only as a
        # placeholder, so a straight swap to psycopg2's '%s' style is safe.
        translated = query.replace("?", "%s")
        stripped = translated.strip().upper()
        if stripped.startswith("INSERT") and "RETURNING" not in stripped:
            translated = translated.rstrip().rstrip(";") + " RETURNING id"
        return translated

    def execute(self, query, params=()):
        translated = self._translate(query)
        self._cur.execute(translated, params)
        if translated.strip().upper().startswith("INSERT") and self._cur.description:
            row = self._cur.fetchone()
            if row:
                # RealDictCursor row -> dict with an "id" key
                self.lastrowid = row["id"] if isinstance(row, dict) else row[0]
        return self

    def fetchone(self):
        return self._cur.fetchone()

    def fetchall(self):
        return self._cur.fetchall()


class _PGConnection:
    """Wraps a psycopg2 connection so conn.execute(...) and conn.cursor()
    behave like their sqlite3 equivalents."""

    def __init__(self, raw_conn):
        self._conn = raw_conn

    def cursor(self):
        return _PGCursor(self._conn.cursor())

    def execute(self, query, params=()):
        cur = self.cursor()
        cur.execute(query, params)
        return cur

    def executescript(self, script):
        # psycopg2 happily executes a multi-statement string in one call.
        self._conn.cursor().execute(script)

    def commit(self):
        self._conn.commit()

    def close(self):
        self._conn.close()


def get_connection():
    """Return a DB connection - Postgres if DATABASE_URL is set, else SQLite."""
    if USE_POSTGRES:
        import psycopg2
        from psycopg2.extras import RealDictCursor

        raw_conn = psycopg2.connect(DATABASE_URL, cursor_factory=RealDictCursor)
        return _PGConnection(raw_conn)

    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA foreign_keys = ON")
    conn.row_factory = sqlite3.Row
    return conn


def init_db():
    """Create all tables if they do not already exist (either backend)."""
    conn = get_connection()
    cur = conn.cursor()
    pk = "SERIAL PRIMARY KEY" if USE_POSTGRES else "INTEGER PRIMARY KEY AUTOINCREMENT"

    cur.executescript(
        f"""
        CREATE TABLE IF NOT EXISTS students (
            id {pk},
            name TEXT NOT NULL,
            email TEXT UNIQUE NOT NULL,
            password_hash TEXT NOT NULL,
            created_at TEXT NOT NULL
        );

        CREATE TABLE IF NOT EXISTS subjects (
            id {pk},
            student_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            difficulty_level INTEGER NOT NULL DEFAULT 3, -- 1 (easy) - 5 (hard)
            marks REAL DEFAULT 0,
            created_at TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS topics (
            id {pk},
            subject_id INTEGER NOT NULL,
            name TEXT NOT NULL,
            mastery REAL NOT NULL DEFAULT 30, -- 0-100, how well the topic is known
            created_at TEXT NOT NULL,
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS exams (
            id {pk},
            subject_id INTEGER NOT NULL,
            exam_date TEXT NOT NULL,
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS study_sessions (
            id {pk},
            student_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            topic_id INTEGER NOT NULL,
            study_date TEXT NOT NULL,
            start_time TEXT NOT NULL,      -- HH:MM 24h
            duration INTEGER NOT NULL,      -- minutes
            focus_level INTEGER NOT NULL,   -- 1-10
            quiz_score REAL,                -- 0-100, optional
            difficulty_level INTEGER NOT NULL,
            sleep_hours REAL DEFAULT 7,
            notes TEXT,
            effectiveness REAL,             -- 0-100, ML target / actual outcome
            created_at TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
            FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS recommendations (
            id {pk},
            student_id INTEGER NOT NULL,
            subject_id INTEGER NOT NULL,
            topic_id INTEGER NOT NULL,
            recommended_date TEXT NOT NULL,
            recommended_time TEXT NOT NULL,
            duration INTEGER NOT NULL,
            priority REAL NOT NULL,
            predicted_effectiveness REAL NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
            FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
        );

        CREATE TABLE IF NOT EXISTS timetable (
            id {pk},
            student_id INTEGER NOT NULL,
            entry_date TEXT NOT NULL,
            start_time TEXT NOT NULL,
            end_time TEXT NOT NULL,
            subject_id INTEGER NOT NULL,
            topic_id INTEGER NOT NULL,
            duration INTEGER NOT NULL,
            priority REAL NOT NULL,
            predicted_effectiveness REAL NOT NULL,
            reason TEXT NOT NULL,
            created_at TEXT NOT NULL,
            FOREIGN KEY (student_id) REFERENCES students(id) ON DELETE CASCADE,
            FOREIGN KEY (subject_id) REFERENCES subjects(id) ON DELETE CASCADE,
            FOREIGN KEY (topic_id) REFERENCES topics(id) ON DELETE CASCADE
        );
        """
    )
    conn.commit()
    conn.close()


def now():
    return datetime.now().isoformat(timespec="seconds")


# ---------------------------------------------------------------------------
# Students
# ---------------------------------------------------------------------------

def create_student(name, email, password_hash):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO students (name, email, password_hash, created_at) VALUES (?, ?, ?, ?)",
        (name, email, password_hash, now()),
    )
    conn.commit()
    student_id = cur.lastrowid
    conn.close()
    return student_id


def get_student_by_email(email):
    conn = get_connection()
    row = conn.execute("SELECT * FROM students WHERE email = ?", (email,)).fetchone()
    conn.close()
    return row


def get_student_by_id(student_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM students WHERE id = ?", (student_id,)).fetchone()
    conn.close()
    return row


# ---------------------------------------------------------------------------
# Subjects / Topics / Exams
# ---------------------------------------------------------------------------

def add_subject(student_id, name, difficulty_level, marks):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO subjects (student_id, name, difficulty_level, marks, created_at) VALUES (?, ?, ?, ?, ?)",
        (student_id, name, difficulty_level, marks, now()),
    )
    conn.commit()
    subject_id = cur.lastrowid
    conn.close()
    return subject_id


def update_subject_marks(subject_id, marks):
    conn = get_connection()
    conn.execute("UPDATE subjects SET marks = ? WHERE id = ?", (marks, subject_id))
    conn.commit()
    conn.close()


def get_subjects(student_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM subjects WHERE student_id = ? ORDER BY name", (student_id,)
    ).fetchall()
    conn.close()
    return rows


def get_subject(subject_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM subjects WHERE id = ?", (subject_id,)).fetchone()
    conn.close()
    return row


def add_topic(subject_id, name, mastery):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO topics (subject_id, name, mastery, created_at) VALUES (?, ?, ?, ?)",
        (subject_id, name, mastery, now()),
    )
    conn.commit()
    topic_id = cur.lastrowid
    conn.close()
    return topic_id


def update_topic_mastery(topic_id, mastery):
    conn = get_connection()
    conn.execute("UPDATE topics SET mastery = ? WHERE id = ?", (mastery, topic_id))
    conn.commit()
    conn.close()


def get_topics(subject_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM topics WHERE subject_id = ? ORDER BY mastery ASC", (subject_id,)
    ).fetchall()
    conn.close()
    return rows


def get_topic(topic_id):
    conn = get_connection()
    row = conn.execute("SELECT * FROM topics WHERE id = ?", (topic_id,)).fetchone()
    conn.close()
    return row


def get_all_topics_for_student(student_id):
    """Every topic belonging to the student, joined with its subject info."""
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT t.id as topic_id, t.name as topic_name, t.mastery,
               s.id as subject_id, s.name as subject_name, s.difficulty_level, s.marks
        FROM topics t
        JOIN subjects s ON t.subject_id = s.id
        WHERE s.student_id = ?
        """,
        (student_id,),
    ).fetchall()
    conn.close()
    return rows


def add_exam(subject_id, exam_date):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        "INSERT INTO exams (subject_id, exam_date) VALUES (?, ?)", (subject_id, exam_date)
    )
    conn.commit()
    exam_id = cur.lastrowid
    conn.close()
    return exam_id


def get_next_exam(subject_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT * FROM exams WHERE subject_id = ? AND exam_date >= ? "
        "ORDER BY exam_date ASC LIMIT 1",
        (subject_id, _today_str()),
    ).fetchone()
    conn.close()
    return row


def get_upcoming_exams(student_id):
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT e.*, s.name as subject_name
        FROM exams e
        JOIN subjects s ON e.subject_id = s.id
        WHERE s.student_id = ? AND e.exam_date >= ?
        ORDER BY e.exam_date ASC
        """,
        (student_id, _today_str()),
    ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Study sessions
# ---------------------------------------------------------------------------

def add_study_session(student_id, subject_id, topic_id, study_date, start_time,
                       duration, focus_level, quiz_score, difficulty_level,
                       sleep_hours, notes, effectiveness):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO study_sessions
        (student_id, subject_id, topic_id, study_date, start_time, duration,
         focus_level, quiz_score, difficulty_level, sleep_hours, notes,
         effectiveness, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (student_id, subject_id, topic_id, study_date, start_time, duration,
         focus_level, quiz_score, difficulty_level, sleep_hours, notes,
         effectiveness, now()),
    )
    conn.commit()
    session_id = cur.lastrowid
    conn.close()
    return session_id


def get_study_sessions(student_id, limit=None):
    conn = get_connection()
    query = """
        SELECT ss.*, sub.name as subject_name, t.name as topic_name
        FROM study_sessions ss
        JOIN subjects sub ON ss.subject_id = sub.id
        JOIN topics t ON ss.topic_id = t.id
        WHERE ss.student_id = ?
        ORDER BY ss.study_date DESC, ss.start_time DESC
    """
    if limit:
        query += f" LIMIT {int(limit)}"
    rows = conn.execute(query, (student_id,)).fetchall()
    conn.close()
    return rows


def count_study_sessions(student_id):
    conn = get_connection()
    row = conn.execute(
        "SELECT COUNT(*) as c FROM study_sessions WHERE student_id = ?", (student_id,)
    ).fetchone()
    conn.close()
    return row["c"]


def get_sessions_for_topic(student_id, topic_id):
    conn = get_connection()
    rows = conn.execute(
        "SELECT * FROM study_sessions WHERE student_id = ? AND topic_id = ? "
        "ORDER BY study_date DESC",
        (student_id, topic_id),
    ).fetchall()
    conn.close()
    return rows


def get_all_sessions_df_ready(student_id=None):
    """Return raw rows across all students (or one), used for ML retraining."""
    conn = get_connection()
    if student_id:
        rows = conn.execute(
            """
            SELECT ss.*, sub.name as subject_name, sub.difficulty_level as subj_difficulty,
                   t.name as topic_name, t.mastery as topic_mastery, sub.marks as previous_marks
            FROM study_sessions ss
            JOIN subjects sub ON ss.subject_id = sub.id
            JOIN topics t ON ss.topic_id = t.id
            WHERE ss.student_id = ?
            """,
            (student_id,),
        ).fetchall()
    else:
        rows = conn.execute(
            """
            SELECT ss.*, sub.name as subject_name, sub.difficulty_level as subj_difficulty,
                   t.name as topic_name, t.mastery as topic_mastery, sub.marks as previous_marks
            FROM study_sessions ss
            JOIN subjects sub ON ss.subject_id = sub.id
            JOIN topics t ON ss.topic_id = t.id
            """
        ).fetchall()
    conn.close()
    return rows


# ---------------------------------------------------------------------------
# Recommendations / Timetable
# ---------------------------------------------------------------------------

def clear_timetable(student_id):
    conn = get_connection()
    conn.execute("DELETE FROM timetable WHERE student_id = ?", (student_id,))
    conn.commit()
    conn.close()


def add_timetable_entry(student_id, entry_date, start_time, end_time, subject_id,
                         topic_id, duration, priority, predicted_effectiveness, reason):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO timetable
        (student_id, entry_date, start_time, end_time, subject_id, topic_id,
         duration, priority, predicted_effectiveness, reason, created_at)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """,
        (student_id, entry_date, start_time, end_time, subject_id, topic_id,
         duration, priority, predicted_effectiveness, reason, now()),
    )
    conn.commit()
    entry_id = cur.lastrowid
    conn.close()
    return entry_id


def get_timetable(student_id):
    conn = get_connection()
    rows = conn.execute(
        """
        SELECT tt.*, sub.name as subject_name, t.name as topic_name
        FROM timetable tt
        JOIN subjects sub ON tt.subject_id = sub.id
        JOIN topics t ON tt.topic_id = t.id
        WHERE tt.student_id = ?
        ORDER BY tt.entry_date ASC, tt.start_time ASC
        """,
        (student_id,),
    ).fetchall()
    conn.close()
    return rows

import sqlite3
from datetime import datetime

class DB:
    def __init__(self, db_path):
        self.db_path = db_path
        self._init_db()

    def get_connection(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self):
        with self.get_connection() as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS students (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    name TEXT NOT NULL,
                    turma TEXT DEFAULT '1ECA',
                    embedding BLOB
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS attendance (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    rm TEXT,
                    name TEXT,
                    turma TEXT,
                    camera_id TEXT,
                    confidence REAL,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.execute("""
                CREATE TABLE IF NOT EXISTS events (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    camera_id TEXT,
                    label TEXT,
                    timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            conn.commit()

    def get_students(self):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT id, name, turma, CASE WHEN embedding IS NOT NULL THEN 1 ELSE 0 END as samples FROM students")
            return [dict(row) for row in cursor.fetchall()]

    def get_students_with_samples(self):
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT id, name, turma, embedding FROM students WHERE embedding IS NOT NULL")
            return [dict(row) for row in cursor.fetchall()]

    def add_student(self, name, turma="1ECA"):
        with self.get_connection() as conn:
            cursor = conn.execute("INSERT INTO students (name, turma) VALUES (?, ?)", (name, turma))
            conn.commit()
            return cursor.lastrowid

    def update_embedding(self, student_id, embedding_bytes):
        with self.get_connection() as conn:
            conn.execute("UPDATE students SET embedding = ? WHERE id = ?", (embedding_bytes, student_id))
            conn.commit()

    def delete_student(self, student_id):
        with self.get_connection() as conn:
            conn.execute("DELETE FROM students WHERE id = ?", (student_id,))
            conn.commit()

    def log_attendance(self, rm, name, turma, camera_id, confidence):
        today = datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            cursor = conn.execute(
                "SELECT id FROM attendance WHERE rm = ? AND date(timestamp) = ?", (rm, today)
            )
            if not cursor.fetchone():
                conn.execute(
                    "INSERT INTO attendance (rm, name, turma, camera_id, confidence) VALUES (?, ?, ?, ?, ?)",
                    (rm, name, turma, camera_id, confidence)
                )
                conn.commit()

    def log_event(self, camera_id, label):
        today = datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            # Evita spam de eventos duplicados para o mesmo tipo de alerta no próprio dia
            cursor = conn.execute(
                "SELECT id FROM events WHERE camera_id = ? AND label = ? AND date(timestamp) = ?",
                (camera_id, label, today)
            )
            if not cursor.fetchone():
                conn.execute(
                    "INSERT INTO events (camera_id, label) VALUES (?, ?)",
                    (camera_id, label)
                )
                conn.commit()

    def get_today_attendance(self):
        today = datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM attendance WHERE date(timestamp) = ?", (today,))
            return [dict(row) for row in cursor.fetchall()]

    def get_pending_events(self):
        today = datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            cursor = conn.execute("SELECT * FROM events WHERE date(timestamp) = ? ORDER BY timestamp DESC LIMIT 10", (today,))
            return [dict(row) for row in cursor.fetchall()]

    def reset_session(self):
        today = datetime.now().strftime("%Y-%m-%d")
        with self.get_connection() as conn:
            conn.execute("DELETE FROM attendance WHERE date(timestamp) = ?", (today,))
            conn.execute("DELETE FROM events WHERE date(timestamp) = ?", (today,))
            conn.commit()
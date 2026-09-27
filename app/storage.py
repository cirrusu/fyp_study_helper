import hashlib
import json
import secrets
import sqlite3
import uuid
from contextlib import contextmanager
from pathlib import Path


class SubmissionConflict(ValueError):
    pass


class Store:
    def __init__(self, path: Path, assessment_fingerprint: str):
        self.path = path
        path.parent.mkdir(parents=True, exist_ok=True)
        with self.connection() as db:
            db.execute("PRAGMA journal_mode=WAL")
            db.executescript("""
                CREATE TABLE IF NOT EXISTS metadata (key TEXT PRIMARY KEY, value TEXT NOT NULL);
                CREATE TABLE IF NOT EXISTS students (
                    student_id TEXT PRIMARY KEY, token_hash TEXT UNIQUE NOT NULL,
                    selected_topics TEXT NOT NULL, preferred_format TEXT, target_level TEXT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE TABLE IF NOT EXISTS submissions (
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    submission_id TEXT NOT NULL, payload_hash TEXT NOT NULL, result TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (student_id, submission_id)
                );
                CREATE TABLE IF NOT EXISTS answers (
                    answer_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL REFERENCES students(student_id),
                    question_id TEXT NOT NULL, topic_id TEXT NOT NULL, correct INTEGER NOT NULL,
                    selected_option TEXT NOT NULL, submission_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
                CREATE INDEX IF NOT EXISTS answers_student ON answers(student_id, question_id, answer_id);
                CREATE TABLE IF NOT EXISTS feedback (
                    student_id TEXT NOT NULL REFERENCES students(student_id), resource_id TEXT NOT NULL,
                    helpful INTEGER, completed INTEGER NOT NULL DEFAULT 0,
                    updated_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    PRIMARY KEY (student_id, resource_id)
                );
                CREATE TABLE IF NOT EXISTS feedback_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    student_id TEXT NOT NULL REFERENCES students(student_id), resource_id TEXT NOT NULL,
                    helpful INTEGER, completed INTEGER,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP
                );
            """)
            previous = db.execute("SELECT value FROM metadata WHERE key='assessment'").fetchone()
            if previous and previous[0] != assessment_fingerprint:
                raise ValueError("Questions/topics changed. Use a new STUDY_HELPER_DB or migrate old assessments explicitly.")
            db.execute("INSERT OR IGNORE INTO metadata VALUES ('assessment', ?)", (assessment_fingerprint,))

    @contextmanager
    def connection(self):
        db = sqlite3.connect(self.path, timeout=10)
        db.row_factory = sqlite3.Row
        db.execute("PRAGMA foreign_keys=ON")
        try:
            with db:
                yield db
        finally:
            db.close()

    @staticmethod
    def public_student(row):
        return {"student_id": row["student_id"], "selected_topics": json.loads(row["selected_topics"]),
                "preferred_format": row["preferred_format"], "target_level": row["target_level"]}

    def create_student(self, profile):
        student_id, token = str(uuid.uuid4()), secrets.token_urlsafe(32)
        with self.connection() as db:
            db.execute("INSERT INTO students (student_id,token_hash,selected_topics,preferred_format,target_level) VALUES (?,?,?,?,?)",
                       (student_id, hashlib.sha256(token.encode()).hexdigest(), json.dumps(profile.selected_topics),
                        profile.preferred_format, profile.target_level))
        return {"student_id": student_id, "token": token, **profile.model_dump()}

    def authenticate(self, token):
        with self.connection() as db:
            row = db.execute("SELECT * FROM students WHERE token_hash=?", (hashlib.sha256(token.encode()).hexdigest(),)).fetchone()
        return self.public_student(row) if row else None

    def update_student(self, student_id, profile):
        with self.connection() as db:
            db.execute("UPDATE students SET selected_topics=?,preferred_format=?,target_level=? WHERE student_id=?",
                       (json.dumps(profile.selected_topics), profile.preferred_format, profile.target_level, student_id))
        return {"student_id": student_id, **profile.model_dump()}

    def submit(self, student_id, submission, questions):
        payload = [(a.question_id, a.selected_option) for a in submission.answers]
        payload_hash = hashlib.sha256(json.dumps(sorted(payload)).encode()).hexdigest()
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            existing = db.execute("SELECT * FROM submissions WHERE student_id=? AND submission_id=?",
                                  (student_id, submission.submission_id)).fetchone()
            if existing:
                if existing["payload_hash"] != payload_hash:
                    raise SubmissionConflict("submission_id already used for different answers")
                return json.loads(existing["result"])
            results = []
            for answer in submission.answers:
                q = questions[answer.question_id]
                correct = answer.selected_option == q.correct_option
                db.execute("INSERT INTO answers (student_id,question_id,topic_id,correct,selected_option,submission_id) VALUES (?,?,?,?,?,?)",
                           (student_id, q.question_id, q.topic_id, int(correct), answer.selected_option, submission.submission_id))
                results.append({"question_id": q.question_id, "topic_id": q.topic_id, "correct": correct,
                                "correct_option": q.correct_option, "explanation": q.explanation})
            result = {"submission_id": submission.submission_id, "correct": sum(r["correct"] for r in results),
                      "total": len(results), "results": results}
            db.execute("INSERT INTO submissions (student_id,submission_id,payload_hash,result) VALUES (?,?,?,?)",
                       (student_id, submission.submission_id, payload_hash, json.dumps(result)))
        return result

    def topic_scores(self, student_id):
        # Latest response per distinct question: repeated submissions don't inflate the sample size.
        with self.connection() as db:
            rows = db.execute("""
                SELECT a.topic_id, SUM(a.correct) AS correct, COUNT(*) AS attempted
                FROM answers a JOIN (
                    SELECT MAX(answer_id) AS id FROM answers WHERE student_id=? GROUP BY question_id
                ) latest ON a.answer_id=latest.id GROUP BY a.topic_id
            """, (student_id,)).fetchall()
        return {r["topic_id"]: {"correct": r["correct"], "attempted": r["attempted"],
                                "performance_proxy": (r["correct"] + 1) / (r["attempted"] + 2)} for r in rows}

    def record_feedback(self, student_id, feedback):
        with self.connection() as db:
            db.execute("BEGIN IMMEDIATE")
            old = db.execute("SELECT * FROM feedback WHERE student_id=? AND resource_id=?",
                             (student_id, feedback.resource_id)).fetchone()
            helpful = feedback.helpful if feedback.helpful is not None else (old["helpful"] if old else None)
            completed = feedback.completed if feedback.completed is not None else (old["completed"] if old else False)
            db.execute("""INSERT INTO feedback (student_id,resource_id,helpful,completed) VALUES (?,?,?,?)
                ON CONFLICT(student_id,resource_id) DO UPDATE SET helpful=excluded.helpful,
                completed=excluded.completed,updated_at=CURRENT_TIMESTAMP""",
                       (student_id, feedback.resource_id, helpful, completed))
            db.execute("INSERT INTO feedback_events (student_id,resource_id,helpful,completed) VALUES (?,?,?,?)",
                       (student_id, feedback.resource_id, feedback.helpful, feedback.completed))
        return {"resource_id": feedback.resource_id, "helpful": None if helpful is None else bool(helpful),
                "completed": bool(completed)}

    def feedback_for(self, student_id):
        with self.connection() as db:
            rows = db.execute("SELECT resource_id,helpful,completed FROM feedback WHERE student_id=?", (student_id,)).fetchall()
        return {r["resource_id"]: dict(r) for r in rows}

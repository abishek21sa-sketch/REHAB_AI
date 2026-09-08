from __future__ import annotations

import json
import sqlite3
from contextlib import closing
from pathlib import Path
from rehab_ai.domain import PatientState, Recommendation


class StateRepository:
    def __init__(self, database_path: str):
        self.database_path = database_path
        Path(database_path).parent.mkdir(parents=True, exist_ok=True)
        self._initialize()

    def _connect(self) -> sqlite3.Connection:
        conn = sqlite3.connect(self.database_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _initialize(self) -> None:
        with closing(self._connect()) as conn, conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS patient_states (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    timestamp TEXT NOT NULL,
                    payload TEXT NOT NULL
                );
                CREATE INDEX IF NOT EXISTS idx_patient_states_patient_time
                    ON patient_states(patient_id, timestamp);
                CREATE TABLE IF NOT EXISTS recommendations (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS clinician_feedback (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    patient_id TEXT NOT NULL,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    payload TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS decision_audit (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    created_at TEXT NOT NULL DEFAULT CURRENT_TIMESTAMP,
                    payload TEXT NOT NULL
                );
                """
            )

    def save_state(self, state: PatientState) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO patient_states(patient_id, timestamp, payload) VALUES (?, ?, ?)",
                (state.patient_id, state.timestamp.isoformat(), state.model_dump_json()),
            )

    def latest_state(self, patient_id: str) -> PatientState | None:
        with closing(self._connect()) as conn:
            row = conn.execute(
                "SELECT payload FROM patient_states WHERE patient_id=? ORDER BY timestamp DESC LIMIT 1",
                (patient_id,),
            ).fetchone()
        return PatientState.model_validate_json(row["payload"]) if row else None

    def history(self, patient_id: str, limit: int = 50) -> list[PatientState]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT payload FROM patient_states WHERE patient_id=? ORDER BY timestamp DESC LIMIT ?",
                (patient_id, limit),
            ).fetchall()
        return [PatientState.model_validate_json(row["payload"]) for row in rows]

    def save_recommendation(self, recommendation: Recommendation) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO recommendations(patient_id, payload) VALUES (?, ?)",
                (recommendation.patient_id, recommendation.model_dump_json()),
            )


    def save_feedback(self, patient_id: str, payload: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute(
                "INSERT INTO clinician_feedback(patient_id, payload) VALUES (?, ?)",
                (patient_id, payload),
            )

    def save_audit(self, payload: str) -> None:
        with closing(self._connect()) as conn, conn:
            conn.execute("INSERT INTO decision_audit(payload) VALUES (?)", (payload,))

    def recent_audits(self, limit: int = 50) -> list[dict]:
        with closing(self._connect()) as conn:
            rows = conn.execute(
                "SELECT payload FROM decision_audit ORDER BY id DESC LIMIT ?",
                (limit,),
            ).fetchall()
        return [json.loads(row["payload"]) for row in rows]

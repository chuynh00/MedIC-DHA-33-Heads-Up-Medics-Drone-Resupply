from __future__ import annotations

import json
import sqlite3
from pathlib import Path

from resupply_engine.models import DispatchPlan, PlanDecisionRecord


class SQLitePlanStore:
    def __init__(self, db_path: Path) -> None:
        self.db_path = db_path
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS plans (
                    plan_id TEXT PRIMARY KEY,
                    request_signature TEXT UNIQUE NOT NULL,
                    burst_id TEXT NOT NULL,
                    raw_payload_hash TEXT NOT NULL,
                    shootdown_rate INTEGER NOT NULL,
                    target_arrival_probability REAL NOT NULL,
                    status TEXT NOT NULL,
                    plan_json TEXT NOT NULL,
                    raw_payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS plan_events (
                    event_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id TEXT NOT NULL,
                    event_type TEXT NOT NULL,
                    payload_json TEXT NOT NULL,
                    created_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS decisions (
                    decision_id INTEGER PRIMARY KEY AUTOINCREMENT,
                    plan_id TEXT NOT NULL,
                    decision TEXT NOT NULL,
                    operator_id TEXT NOT NULL,
                    notes TEXT,
                    decided_at TEXT NOT NULL
                )
                """
            )

    def get_plan_by_signature(self, request_signature: str) -> DispatchPlan | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT plan_json FROM plans WHERE request_signature = ?",
                (request_signature,),
            ).fetchone()
        if row is None:
            return None
        return DispatchPlan.model_validate_json(row["plan_json"])

    def get_plan(self, plan_id: str) -> DispatchPlan | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT plan_json FROM plans WHERE plan_id = ?",
                (plan_id,),
            ).fetchone()
        if row is None:
            return None
        return DispatchPlan.model_validate_json(row["plan_json"])

    def save_new_plan(self, plan: DispatchPlan, raw_payload: dict) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO plans (
                    plan_id, request_signature, burst_id, raw_payload_hash, shootdown_rate,
                    target_arrival_probability, status, plan_json, raw_payload_json,
                    created_at, updated_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    plan.plan_id,
                    plan.request_signature,
                    plan.burst_id,
                    plan.raw_payload_hash,
                    plan.shootdown_rate,
                    plan.target_arrival_probability,
                    plan.status,
                    plan.model_dump_json(),
                    json.dumps(raw_payload, sort_keys=True, default=str),
                    plan.created_at.isoformat(),
                    plan.updated_at.isoformat(),
                ),
            )

    def update_plan(self, plan: DispatchPlan, event_type: str, event_payload: dict) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE plans
                SET status = ?, plan_json = ?, updated_at = ?
                WHERE plan_id = ?
                """,
                (
                    plan.status,
                    plan.model_dump_json(),
                    plan.updated_at.isoformat(),
                    plan.plan_id,
                ),
            )
            connection.execute(
                """
                INSERT INTO plan_events (plan_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    plan.plan_id,
                    event_type,
                    json.dumps(event_payload, sort_keys=True, default=str),
                    plan.updated_at.isoformat(),
                ),
            )

    def append_event(self, plan_id: str, event_type: str, event_payload: dict, created_at: str) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO plan_events (plan_id, event_type, payload_json, created_at)
                VALUES (?, ?, ?, ?)
                """,
                (
                    plan_id,
                    event_type,
                    json.dumps(event_payload, sort_keys=True, default=str),
                    created_at,
                ),
            )

    def record_decision(self, decision: PlanDecisionRecord) -> None:
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO decisions (plan_id, decision, operator_id, notes, decided_at)
                VALUES (?, ?, ?, ?, ?)
                """,
                (
                    decision.plan_id,
                    decision.decision,
                    decision.operator_id,
                    decision.notes,
                    decision.decided_at.isoformat(),
                ),
            )

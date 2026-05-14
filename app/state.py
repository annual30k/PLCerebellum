from datetime import datetime, timezone
from collections import deque
from pathlib import Path
from uuid import uuid4

from app.models import AuditRecord

MAX_EVENTS = 1_000
MAX_AUDIT_RECORDS = 1_000
MAX_AUDIT_FILE_BYTES = 5_000_000


class DeviceState:
    def __init__(self, data_dir: Path, log_dir: Path) -> None:
        self.booted_at = datetime.now(timezone.utc)
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.events: deque[dict] = deque(maxlen=MAX_EVENTS)
        self.audit_log: deque[AuditRecord] = deque(maxlen=MAX_AUDIT_RECORDS)
        self.temperature_c = 46.0
        self.battery_percent = 92
        self.storage_used_gb = 18.5

    def add_event(self, event_type: str, payload: dict) -> dict:
        event = {
            "event_id": f"evt-{uuid4().hex[:12]}",
            "event_type": event_type,
            "created_at": datetime.now(timezone.utc).isoformat(),
            "payload": payload,
            "human_status": "unconfirmed",
        }
        self.events.append(event)
        return event

    def audit(self, action: str, detail: dict, actor: str = "system") -> AuditRecord:
        record = AuditRecord(
            timestamp=datetime.now(timezone.utc),
            action=action,
            actor=actor,
            detail=detail,
        )
        self.audit_log.append(record)
        self._append_audit_file(record)
        return record

    def _append_audit_file(self, record: AuditRecord) -> None:
        self.log_dir.mkdir(parents=True, exist_ok=True)
        audit_file = self.log_dir / "audit.jsonl"
        if audit_file.exists() and audit_file.stat().st_size >= MAX_AUDIT_FILE_BYTES:
            rotated_file = self.log_dir / f"audit-{datetime.now(timezone.utc).strftime('%Y%m%d%H%M%S')}.jsonl"
            audit_file.rename(rotated_file)
        with audit_file.open("a", encoding="utf-8") as file:
            file.write(record.model_dump_json() + "\n")

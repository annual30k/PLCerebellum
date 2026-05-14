from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.models import AuditRecord


class DeviceState:
    def __init__(self, data_dir: Path, log_dir: Path) -> None:
        self.booted_at = datetime.now(timezone.utc)
        self.data_dir = data_dir
        self.log_dir = log_dir
        self.events: list[dict] = []
        self.audit_log: list[AuditRecord] = []
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
        with audit_file.open("a", encoding="utf-8") as file:
            file.write(record.model_dump_json() + "\n")


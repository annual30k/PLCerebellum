import json
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

import httpx

from app.models import SyncTaskRequest
from app.settings import Settings
from app.state import DeviceState


def create_sync_task(request: SyncTaskRequest, settings: Settings, state: DeviceState) -> dict:
    destination_url = request.destination_url or settings.sync_destination_url
    task = {
        "task_id": f"sync-{uuid4().hex[:12]}",
        "mission_id": request.mission_id,
        "destination_url": destination_url,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "updated_at": datetime.now(timezone.utc).isoformat(),
        "status": "queued" if destination_url else "offline_waiting",
        "include_events": request.include_events,
        "include_audit": request.include_audit,
        "event_limit": request.event_limit,
        "attempts": 0,
        "last_error": None,
    }
    tasks = load_sync_tasks(settings)
    tasks.append(task)
    save_sync_tasks(settings, tasks)
    return task


def list_sync_tasks(settings: Settings) -> list[dict]:
    return load_sync_tasks(settings)


def run_sync_task(task_id: str, settings: Settings, state: DeviceState) -> dict:
    tasks = load_sync_tasks(settings)
    task = next((item for item in tasks if item["task_id"] == task_id), None)
    if task is None:
        raise KeyError(task_id)
    task["attempts"] += 1
    task["updated_at"] = datetime.now(timezone.utc).isoformat()
    if not task.get("destination_url"):
        task["status"] = "offline_waiting"
        task["last_error"] = "missing destination_url"
        save_sync_tasks(settings, tasks)
        return task

    bundle = build_sync_bundle(task, state)
    try:
        response = httpx.post(task["destination_url"], json=bundle, timeout=30.0)
        response.raise_for_status()
        task["status"] = "synced"
        task["last_error"] = None
        task["synced_at"] = datetime.now(timezone.utc).isoformat()
        task["response_status_code"] = response.status_code
    except httpx.HTTPError as exc:
        task["status"] = "retry_waiting"
        task["last_error"] = str(exc)
    save_sync_tasks(settings, tasks)
    return task


def build_sync_bundle(task: dict, state: DeviceState) -> dict:
    bundle = {
        "task_id": task["task_id"],
        "mission_id": task.get("mission_id"),
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "device": {
            "booted_at": state.booted_at.isoformat(),
            "battery_percent": state.battery_percent,
            "temperature_c": state.temperature_c,
        },
    }
    if task.get("include_events"):
        bundle["events"] = state.event_snapshot(limit=task.get("event_limit", 100))
    if task.get("include_audit"):
        bundle["audit"] = [record.model_dump(mode="json") for record in state.audit_snapshot(limit=task.get("event_limit", 100))]
    return bundle


def sync_queue_path(settings: Settings) -> Path:
    return settings.data_dir / "sync_queue.json"


def load_sync_tasks(settings: Settings) -> list[dict]:
    path = sync_queue_path(settings)
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_sync_tasks(settings: Settings, tasks: list[dict]) -> None:
    path = sync_queue_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(tasks, file, ensure_ascii=False, indent=2)

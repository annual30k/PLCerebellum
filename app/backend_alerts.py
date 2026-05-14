from __future__ import annotations

import httpx

from app.settings import Settings


def report_face_alert_to_backend(alert: dict, settings: Settings) -> None:
    if not settings.backend_alert_report_enabled or not settings.backend_base_url:
        return
    url = f"{settings.backend_base_url.rstrip('/')}/api/v1/cerebellum/face-alerts"
    headers = {"Content-Type": "application/json"}
    if settings.backend_token:
        headers["Authorization"] = f"Bearer {settings.backend_token}"
    payload = {"device_id": settings.device_id, **alert}
    httpx.post(url, headers=headers, json=payload, timeout=15.0).raise_for_status()

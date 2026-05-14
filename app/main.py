from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI

from app.models import (
    FaceAnalyzeRequest,
    MediaIngestRequest,
    PlateAnalyzeRequest,
    ReportRequest,
)
from app.services import generate_report, simulate_face_candidate, simulate_plate_recognition
from app.settings import get_settings, load_device_config
from app.state import DeviceState

settings = get_settings()
state = DeviceState(settings.data_dir, settings.log_dir)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    state.audit("device.boot", {"profile": settings.profile, "accelerator": settings.accelerator})
    yield
    state.audit("device.shutdown", {"uptime_seconds": uptime_seconds()})


app = FastAPI(
    title="PatrolLink Cerebellum Edge Server Simulator",
    description="单兵边缘智能小脑服务器 Docker 模拟系统",
    version="0.1.0",
    lifespan=lifespan,
)


def uptime_seconds() -> int:
    return int((datetime.now(timezone.utc) - state.booted_at).total_seconds())


@app.get("/health")
def health() -> dict:
    return {
        "status": "ok",
        "device_id": settings.device_id,
        "uptime_seconds": uptime_seconds(),
        "primary_model": settings.llm_model,
    }


@app.get("/api/v1/device/status")
def device_status() -> dict:
    config = load_device_config()
    return {
        "device_id": settings.device_id,
        "profile": settings.profile,
        "accelerator": settings.accelerator,
        "target_platform": config.get("hardware", {}).get("target_platform", "unknown"),
        "linux_hardening": config.get("linux_hardening", {}),
        "resources": {
            "cpu_cores": config.get("hardware", {}).get("cpu_cores", 8),
            "memory_gb": config.get("hardware", {}).get("memory_gb", 16),
            "storage_gb": settings.storage_gb,
            "battery_wh": settings.battery_wh,
            "battery_percent": state.battery_percent,
            "temperature_c": state.temperature_c,
            "power_mode": settings.power_mode,
        },
        "models": {
            "primary": settings.llm_model,
            "fallback": settings.llm_fallback_model,
            "batch": settings.llm_batch_model,
            "context_tokens": settings.context_tokens,
            "max_context_tokens": settings.max_context_tokens,
        },
        "security": {
            "secure_boot": settings.secure_boot,
            "readonly_rootfs": settings.readonly_rootfs,
            "non_root_runtime": True,
            "ssh_enabled": False,
            "usb_automount_enabled": False,
        },
    }


@app.post("/api/v1/media/ingest")
def ingest_media(request: MediaIngestRequest) -> dict:
    event = state.add_event(
        "media_ingest",
        {
            "source": request.source,
            "media_type": request.media_type,
            "duration_seconds": request.duration_seconds,
            "note": request.note,
        },
    )
    state.audit("media.ingest", event)
    return {"accepted": True, "event": event}


@app.post("/api/v1/analyze/plate")
def analyze_plate(request: PlateAnalyzeRequest) -> dict:
    result = simulate_plate_recognition(request)
    event = state.add_event("plate_candidate", result)
    state.audit("vision.plate", {"request": request.model_dump(), "result": result})
    return {"result": result, "event": event}


@app.post("/api/v1/analyze/face")
def analyze_face(request: FaceAnalyzeRequest) -> dict:
    result = simulate_face_candidate(request)
    event = state.add_event("face_candidate", result)
    state.audit("vision.face", {"request": request.model_dump(), "result": result})
    return {"result": result, "event": event}


@app.post("/api/v1/llm/report")
def create_report(request: ReportRequest) -> dict:
    report = generate_report(request, settings, state)
    event = state.add_event("report_generated", report)
    state.audit("llm.report", {"request": request.model_dump(), "model": report["model"]})
    return {"report": report, "event": event}


@app.get("/api/v1/events")
def list_events() -> dict:
    events = list(state.events)
    return {"count": len(events), "retention_limit": 1000, "events": events[-100:]}


@app.get("/api/v1/audit")
def list_audit() -> dict:
    records = list(state.audit_log)
    return {
        "count": len(records),
        "retention_limit": 1000,
        "records": [record.model_dump(mode="json") for record in records[-100:]],
    }

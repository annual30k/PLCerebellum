from contextlib import asynccontextmanager
from datetime import datetime, timezone

from fastapi import FastAPI, HTTPException

from app.models import (
    FaceEnrollRequest,
    FaceAnalyzeRequest,
    MediaIngestRequest,
    PlateAnalyzeRequest,
    ReportRequest,
    StreamCreateRequest,
)
from app.services import analyze_face_image, analyze_plate_image, generate_report
from app.settings import get_settings, load_device_config
from app.state import DeviceState
from app.streams import StreamManager
from app.vision import enroll_face

settings = get_settings()
state = DeviceState(settings.data_dir, settings.log_dir)
streams = StreamManager(settings, state)


@asynccontextmanager
async def lifespan(_: FastAPI):
    settings.data_dir.mkdir(parents=True, exist_ok=True)
    settings.log_dir.mkdir(parents=True, exist_ok=True)
    settings.stream_frame_dir.mkdir(parents=True, exist_ok=True)
    state.audit("device.boot", {"profile": settings.profile, "accelerator": settings.accelerator})
    yield
    streams.stop_all()
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
        "streaming": {
            "max_sources": settings.stream_max_sources,
            "retained_frames_per_source": settings.stream_retained_frames_per_source,
            "frame_dir": str(settings.stream_frame_dir),
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


@app.post("/api/v1/streams")
def create_stream(request: StreamCreateRequest) -> dict:
    try:
        stream = streams.create(request)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    except RuntimeError as exc:
        raise HTTPException(status_code=409, detail=str(exc)) from exc
    event = state.add_event("stream_registered", stream)
    state.audit("stream.register", {"request": request.model_dump(), "stream": stream})
    return {"accepted": True, "stream": stream, "event": event}


@app.get("/api/v1/streams")
def list_streams() -> dict:
    items = streams.list()
    return {"count": len(items), "streams": items}


@app.get("/api/v1/streams/{stream_id}")
def get_stream(stream_id: str) -> dict:
    try:
        return {"stream": streams.get(stream_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"stream not found: {stream_id}") from exc


@app.post("/api/v1/streams/{stream_id}/stop")
def stop_stream(stream_id: str) -> dict:
    try:
        stream = streams.stop(stream_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"stream not found: {stream_id}") from exc
    event = state.add_event("stream_stopped", stream)
    state.audit("stream.stop.request", {"stream_id": stream_id, "stream": stream})
    return {"stream": stream, "event": event}


@app.post("/api/v1/analyze/plate")
def analyze_plate(request: PlateAnalyzeRequest) -> dict:
    result = analyze_plate_image(request, settings, state)
    event = state.add_event("plate_candidate", result)
    state.audit("vision.plate", {"request": request.model_dump(), "result": result})
    return {"result": result, "event": event}


@app.post("/api/v1/analyze/face")
def analyze_face(request: FaceAnalyzeRequest) -> dict:
    result = analyze_face_image(request, settings, state)
    event = state.add_event("face_candidate", result)
    state.audit("vision.face", {"request": request.model_dump(), "result": result})
    return {"result": result, "event": event}


@app.post("/api/v1/face/enroll")
def enroll_face_candidate(request: FaceEnrollRequest) -> dict:
    result = enroll_face(request.person_id, request.image_uri, request.display_name, settings)
    event = state.add_event("face_enrolled", result)
    state.audit("vision.face.enroll", {"request": request.model_dump(), "result": result})
    return {"result": result, "event": event}


@app.post("/api/v1/llm/report")
def create_report(request: ReportRequest) -> dict:
    report = generate_report(request, settings, state)
    event = state.add_event("report_generated", report)
    state.audit("llm.report", {"request": request.model_dump(), "model": report["model"]})
    return {"report": report, "event": event}


@app.get("/api/v1/events")
def list_events() -> dict:
    events = state.event_snapshot()
    return {"count": len(events), "retention_limit": 1000, "events": events[-100:]}


@app.get("/api/v1/audit")
def list_audit() -> dict:
    records = state.audit_snapshot()
    return {
        "count": len(records),
        "retention_limit": 1000,
        "records": [record.model_dump(mode="json") for record in records[-100:]],
    }

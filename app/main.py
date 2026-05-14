from contextlib import asynccontextmanager
from datetime import datetime, timezone
from secrets import compare_digest

from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse

from app.asr import local_asr_configured, transcribe_audio
from app.evidence import list_evidence, register_evidence
from app.function_recognition import recognize_function
from app.models import (
    AsrTranscribeRequest,
    EvidenceRegisterRequest,
    FaceEnrollRequest,
    FaceAnalyzeRequest,
    FunctionRecognizeRequest,
    MediaIngestRequest,
    ObjectDetectRequest,
    PlateAnalyzeRequest,
    ReportRequest,
    StreamCreateRequest,
    SyncTaskRequest,
    VideoSummaryRequest,
)
from app.objects import detect_objects
from app.security import certificate_status
from app.services import analyze_face_image, analyze_plate_image, generate_report, summarize_video
from app.settings import get_settings, load_device_config
from app.state import DeviceState
from app.streams import StreamManager
from app.sync import create_sync_task, list_sync_tasks, run_sync_task
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


@app.middleware("http")
async def api_key_guard(request: Request, call_next):
    if settings.api_key and request.url.path.startswith("/api/v1/"):
        supplied_key = request.headers.get("x-api-key", "")
        if not compare_digest(supplied_key, settings.api_key):
            state.audit("security.api_key.denied", {"path": request.url.path}, actor="remote-client")
            return JSONResponse(status_code=401, content={"detail": "invalid or missing API key"})
    return await call_next(request)


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
            "asr": settings.asr_model,
            "asr_backend": settings.asr_backend,
            "asr_local_configured": local_asr_configured(settings),
            "object": settings.object_model,
            "context_tokens": settings.context_tokens,
            "max_context_tokens": settings.max_context_tokens,
        },
        "security": {
            "secure_boot": settings.secure_boot,
            "readonly_rootfs": settings.readonly_rootfs,
            "non_root_runtime": True,
            "api_key_required": bool(settings.api_key),
            "mtls_ready": certificate_status(settings)["mtls_ready"],
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


@app.post("/api/v1/functions/recognize")
def recognize_cerebellum_function(request: FunctionRecognizeRequest) -> dict:
    result = recognize_function(request)
    event = state.add_event("function_recognized", result)
    state.audit("function.recognize", {"request": request.model_dump(), "result": result})
    return {"result": result, "event": event}


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


@app.post("/api/v1/analyze/object")
def analyze_object(request: ObjectDetectRequest) -> dict:
    result = detect_objects(request, settings)
    event = state.add_event("object_candidate", result)
    state.audit("vision.object", {"request": request.model_dump(), "result": result})
    return {"result": result, "event": event}


@app.post("/api/v1/asr/transcribe")
def create_transcript(request: AsrTranscribeRequest) -> dict:
    try:
        result = transcribe_audio(request, settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    event = state.add_event("audio_transcribed", result)
    state.audit("asr.transcribe", {"request": request.model_dump(), "backend": result["backend"]})
    return {"transcript": result, "event": event}


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


@app.post("/api/v1/video/summary")
def create_video_summary(request: VideoSummaryRequest) -> dict:
    summary = summarize_video(request, settings, state)
    event = state.add_event("video_summary_generated", summary)
    state.audit("video.summary", {"request": request.model_dump(), "backend": summary["backend"]})
    return {"summary": summary, "event": event}


@app.post("/api/v1/evidence")
def create_evidence(request: EvidenceRegisterRequest) -> dict:
    try:
        evidence = register_evidence(request, settings)
    except FileNotFoundError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc
    except (RuntimeError, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    event = state.add_event("evidence_registered", evidence)
    state.audit("evidence.register", {"request": request.model_dump(), "evidence_id": evidence["evidence_id"]})
    return {"evidence": evidence, "event": event}


@app.get("/api/v1/evidence")
def get_evidence() -> dict:
    items = list_evidence(settings)
    return {"count": len(items), "items": items[-100:]}


@app.post("/api/v1/sync/tasks")
def enqueue_sync_task(request: SyncTaskRequest) -> dict:
    task = create_sync_task(request, settings, state)
    event = state.add_event("sync_task_created", task)
    state.audit("sync.task.create", {"request": request.model_dump(), "task_id": task["task_id"]})
    return {"task": task, "event": event}


@app.get("/api/v1/sync/tasks")
def get_sync_tasks() -> dict:
    tasks = list_sync_tasks(settings)
    return {"count": len(tasks), "tasks": tasks[-100:]}


@app.post("/api/v1/sync/tasks/{task_id}/run")
def run_sync(task_id: str) -> dict:
    try:
        task = run_sync_task(task_id, settings, state)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail=f"sync task not found: {task_id}") from exc
    event = state.add_event("sync_task_run", task)
    state.audit("sync.task.run", {"task_id": task_id, "status": task["status"]})
    return {"task": task, "event": event}


@app.get("/api/v1/security/certificates")
def get_certificate_status() -> dict:
    return certificate_status(settings)


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

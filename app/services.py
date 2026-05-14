from datetime import datetime, timezone
from hashlib import sha256

import httpx

from app.models import FaceAnalyzeRequest, PlateAnalyzeRequest, ReportRequest, VideoSummaryRequest
from app.settings import Settings
from app.state import DeviceState
from app.vision import VisionUnavailable, detect_faces, recognize_plate


def choose_llm_model(request: ReportRequest, settings: Settings, state: DeviceState) -> str:
    if state.temperature_c >= 72 or state.battery_percent <= 20:
        return settings.llm_fallback_model
    if request.prefer_quality and settings.power_mode == "charging_batch":
        return settings.llm_batch_model
    return settings.llm_model


def simulate_plate_recognition(request: PlateAnalyzeRequest) -> dict:
    seed = sha256(f"{request.frame_id}:{request.camera_id}".encode()).hexdigest()
    province = ["京", "沪", "粤", "浙", "苏", "鲁"][int(seed[0], 16) % 6]
    plate = province + chr(65 + int(seed[1], 16) % 26) + seed[2:7].upper()
    confidence = 0.82 + (int(seed[7], 16) / 100)
    return {
        "plate_number": plate,
        "confidence": round(min(confidence, 0.97), 3),
        "vehicle_type": "small_vehicle",
        "evidence_frame": request.frame_id,
        "model": "plate-detector-sim+tensorrt-profile",
    }


def simulate_face_candidate(request: FaceAnalyzeRequest) -> dict:
    seed = sha256(f"{request.frame_id}:{request.candidate_library}".encode()).hexdigest()
    similarity = 0.68 + (int(seed[0], 16) / 100)
    return {
        "candidate_id": f"person-{seed[:8]}",
        "similarity": round(min(similarity, 0.91), 3),
        "quality_score": round(0.72 + int(seed[1], 16) / 100, 3),
        "candidate_library": request.candidate_library,
        "result_type": "candidate_hint_only",
        "model": "face-candidate-sim+arcface-profile",
    }


def analyze_plate_image(request: PlateAnalyzeRequest, settings: Settings, state: DeviceState) -> dict:
    if request.image_uri:
        try:
            results = recognize_plate(request.image_uri, settings)
            return {
                "backend": "hyperlpr3",
                "frame_id": request.frame_id,
                "camera_id": request.camera_id,
                "candidates": results,
                "candidate_count": len(results),
            }
        except (VisionUnavailable, FileNotFoundError, ValueError, RuntimeError) as exc:
            state.audit("vision.plate.fallback", {"error": str(exc), "image_uri": request.image_uri})
    return {
        "backend": "simulated-fallback",
        "frame_id": request.frame_id,
        "camera_id": request.camera_id,
        "candidates": [simulate_plate_recognition(request)],
        "candidate_count": 1,
    }


def analyze_face_image(request: FaceAnalyzeRequest, settings: Settings, state: DeviceState) -> dict:
    if request.image_uri:
        try:
            results = detect_faces(request.image_uri, settings)
            candidate_count = sum(1 for face in results if face.get("candidate"))
            return {
                "backend": "opencv-zoo-yunet+sface",
                "frame_id": request.frame_id,
                "camera_id": request.camera_id,
                "candidate_library": request.candidate_library,
                "faces": results,
                "face_count": len(results),
                "candidate_count": candidate_count,
            }
        except (VisionUnavailable, FileNotFoundError, ValueError, RuntimeError) as exc:
            state.audit("vision.face.fallback", {"error": str(exc), "image_uri": request.image_uri})
    return {
        "backend": "simulated-fallback",
        "frame_id": request.frame_id,
        "camera_id": request.camera_id,
        "candidate_library": request.candidate_library,
        "faces": [simulate_face_candidate(request)],
        "face_count": 1,
        "candidate_count": 1,
    }


def summarize_video(request: VideoSummaryRequest, settings: Settings, state: DeviceState) -> dict:
    events = filter_video_events(state.event_snapshot(), request.stream_id)[-request.event_limit :]
    summary = build_structured_video_summary(request, events, settings)
    if request.use_llm and settings.llm_base_url:
        model = settings.llm_fallback_model if state.temperature_c >= 72 or state.battery_percent <= 20 else settings.llm_model
        if model == settings.llm_model:
            try:
                llm_content = generate_video_summary_with_llama_cpp(request, summary, settings, model)
                summary["content"] = llm_content
                summary["backend"] = "llama.cpp"
                summary["model"] = model
            except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
                state.audit("llm.video_summary.fallback", {"model": model, "error": str(exc)})
    return summary


def filter_video_events(events: list[dict], stream_id: str | None) -> list[dict]:
    video_event_types = {
        "media_ingest",
        "stream_registered",
        "stream_stopped",
        "stream_session_closed",
        "stream_plate_candidate",
        "stream_face_candidate",
        "stream_face_alert",
        "stream_object_candidate",
        "plate_candidate",
        "face_candidate",
        "object_candidate",
        "audio_transcribed",
    }
    filtered = []
    for event in events:
        if event.get("event_type") not in video_event_types:
            continue
        payload = event.get("payload", {})
        if stream_id and payload.get("stream_id") != stream_id:
            continue
        filtered.append(event)
    return filtered


def build_structured_video_summary(request: VideoSummaryRequest, events: list[dict], settings: Settings) -> dict:
    plate_events = [event for event in events if event.get("event_type") in {"stream_plate_candidate", "plate_candidate"}]
    face_events = [event for event in events if event.get("event_type") in {"stream_face_candidate", "stream_face_alert", "face_candidate"}]
    object_events = [event for event in events if event.get("event_type") in {"stream_object_candidate", "object_candidate"}]
    audio_events = [event for event in events if event.get("event_type") == "audio_transcribed"]
    stream_events = [
        event
        for event in events
        if event.get("event_type") in {"stream_registered", "stream_stopped", "stream_session_closed", "media_ingest"}
    ]
    plate_numbers = []
    face_candidates = []
    timeline = []

    for event in events:
        payload = event.get("payload", {})
        timeline.append(
            {
                "time": event.get("created_at"),
                "event_type": event.get("event_type"),
                "stream_id": payload.get("stream_id"),
                "frame_id": payload.get("frame_id"),
                "backend": payload.get("backend"),
                "candidate_count": payload.get("candidate_count"),
                "face_count": payload.get("face_count"),
                "detection_count": payload.get("detection_count"),
                "duration_seconds": payload.get("duration_seconds"),
            }
        )
        for candidate in payload.get("candidates", []) or []:
            plate_number = candidate.get("plate_number")
            if plate_number and plate_number not in plate_numbers:
                plate_numbers.append(plate_number)
        for face in payload.get("faces", []) or []:
            candidate = face.get("candidate")
            if candidate:
                face_candidates.append(candidate)

    started_at = events[0]["created_at"] if events else None
    ended_at = events[-1]["created_at"] if events else None
    content = (
        f"视频摘要：任务 {request.mission_id} 共纳入 {len(events)} 条结构化事件。"
        f"其中视频/流事件 {len(stream_events)} 条，车牌识别事件 {len(plate_events)} 条，"
        f"人脸候选事件 {len(face_events)} 条，目标检测事件 {len(object_events)} 条，"
        f"音频转写事件 {len(audio_events)} 条。"
        f"识别到的去重车牌数量为 {len(plate_numbers)}，人脸候选数量为 {len(face_candidates)}。"
        "所有 AI 结果均为候选提示，需结合原始视频和人工复核。"
    )
    if request.operator_note:
        content += f" 人工补充：{request.operator_note}。"

    return {
        "mission_id": request.mission_id,
        "stream_id": request.stream_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "started_at": started_at,
        "ended_at": ended_at,
        "event_count": len(events),
        "stream_event_count": len(stream_events),
        "plate_event_count": len(plate_events),
        "face_event_count": len(face_events),
        "object_event_count": len(object_events),
        "audio_event_count": len(audio_events),
        "unique_plate_numbers": plate_numbers,
        "face_candidates": face_candidates[:20],
        "timeline": timeline[-50:],
        "content": content,
        "backend": "structured",
        "model": None,
        "requires_human_confirmation": True,
        "context_tokens": settings.context_tokens,
    }


def generate_report(request: ReportRequest, settings: Settings, state: DeviceState) -> dict:
    model = choose_llm_model(request, settings, state)
    if settings.llm_base_url and model == settings.llm_model:
        try:
            return generate_report_with_llama_cpp(request, settings, state, model)
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            state.audit("llm.report.fallback", {"model": model, "error": str(exc)})

    event_count = state.event_count()
    now = datetime.now(timezone.utc).isoformat()
    title_map = {
        "daily": "单兵巡逻日报",
        "video_summary": "执法视频摘要",
        "handover": "交接班说明",
        "incident": "异常事件报告",
    }
    operator_note = request.operator_note or "无人工补充说明"
    content = (
        f"{title_map[request.report_type]}：任务 {request.mission_id} 当前累计结构化事件 {event_count} 条。"
        f"系统已汇总车牌候选、人脸候选、媒体接入和人工标记信息。"
        f"本报告由 {model} 模拟生成，人工补充：{operator_note}。"
        "正式入库前应由执勤人员确认 AI 生成内容，并保留原始视频证据索引。"
    )
    return {
        "mission_id": request.mission_id,
        "report_type": request.report_type,
        "model": model,
        "context_tokens": settings.context_tokens,
        "max_context_tokens": settings.max_context_tokens,
        "generated_at": now,
        "content": content,
        "requires_human_confirmation": True,
        "backend": "simulated-fallback",
    }


def generate_video_summary_with_llama_cpp(
    request: VideoSummaryRequest,
    summary: dict,
    settings: Settings,
    model: str,
) -> str:
    system_prompt = (
        "你是部署在单兵边缘智能小脑服务器中的执法视频摘要助手。"
        "你只能根据结构化事件生成摘要草稿。"
        "不得把人脸候选、车牌候选或目标检测结果写成确定事实。"
        "必须提示需要人工复核原始视频。"
        "输出中文，简洁、正式，不输出思考过程。"
    )
    user_prompt = (
        f"任务编号：{request.mission_id}\n"
        f"人工补充：{request.operator_note or '无'}\n"
        f"结构化摘要：{summary}\n"
        "请生成执法视频摘要草稿，包含：视频范围、时间线概况、识别候选、异常或待复核事项、证据复核提醒。"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.4,
        "top_p": 0.8,
        "max_tokens": request.max_tokens,
    }
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    response = httpx.post(url, json=payload, timeout=settings.llm_timeout_seconds)
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    content = (message.get("content") or message.get("reasoning_content") or "").strip()
    if not content:
        raise ValueError("llama.cpp returned an empty response")
    return content


def generate_report_with_llama_cpp(
    request: ReportRequest,
    settings: Settings,
    state: DeviceState,
    model: str,
) -> dict:
    report_events = compact_events_for_llm(state.event_snapshot(limit=1000), limit=1000)
    event_count = len(report_events)
    recent_events = report_events[-20:]
    system_prompt = (
        "你是部署在单兵边缘智能小脑服务器中的警务报告助手。"
        "你只能根据输入的结构化事件、人工备注和设备状态生成报告草稿。"
        "不得把人脸候选或车牌候选写成确定结论，必须提示需要人工确认。"
        "输出中文，风格简洁、正式、可用于执勤日报初稿。"
        "不要输出思考过程，只输出最终报告正文。"
    )
    user_prompt = (
        f"任务编号：{request.mission_id}\n"
        f"报告类型：{request.report_type}\n"
        f"人工补充：{request.operator_note or '无'}\n"
        f"当前结构化事件数量：{event_count}\n"
        f"最近事件：{recent_events}\n"
        "请生成报告草稿，包含：工作概况、识别结果摘要、异常/待确认事项、证据索引提醒。"
    )
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "chat_template_kwargs": {"enable_thinking": False},
        "temperature": 0.7,
        "top_p": 0.8,
        "max_tokens": request.max_tokens,
    }
    url = f"{settings.llm_base_url.rstrip('/')}/chat/completions"
    response = httpx.post(url, json=payload, timeout=settings.llm_timeout_seconds)
    response.raise_for_status()
    data = response.json()
    message = data["choices"][0]["message"]
    content = (message.get("content") or message.get("reasoning_content") or "").strip()
    if not content:
        raise ValueError("llama.cpp returned an empty response")
    return {
        "mission_id": request.mission_id,
        "report_type": request.report_type,
        "model": model,
        "context_tokens": settings.context_tokens,
        "max_context_tokens": settings.max_context_tokens,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "content": content,
        "requires_human_confirmation": True,
        "backend": "llama.cpp",
    }


def compact_events_for_llm(events: list[dict], limit: int = 20) -> list[dict]:
    compacted = []
    for event in reversed(events):
        event_type = event.get("event_type")
        if event_type in {"function_recognized", "report_generated"}:
            continue
        payload = compact_event_payload(event_type, event.get("payload", {}) or {})
        compacted.append(
            {
                "event_type": event_type,
                "created_at": event.get("created_at"),
                "payload": payload,
                "human_status": event.get("human_status"),
            }
        )
        if len(compacted) >= limit:
            break
    return list(reversed(compacted))


def compact_event_payload(event_type: str | None, payload: dict) -> dict:
    fields = (
        "mission_id",
        "stream_id",
        "frame_id",
        "camera_id",
        "person_id",
        "display_name",
        "risk_level",
        "category",
        "backend",
        "candidate_count",
        "face_count",
        "vote_count",
        "confirm_frames",
        "average_similarity",
        "detection_count",
        "duration_seconds",
        "source",
        "media_type",
        "status",
        "source_uri",
        "audio_uri",
        "evidence_id",
        "evidence_type",
        "chain_status",
    )
    compacted = {field: payload[field] for field in fields if field in payload}
    if event_type in {"plate_candidate", "stream_plate_candidate"}:
        compacted["plates"] = [
            candidate.get("plate_number")
            for candidate in payload.get("candidates", [])[:5]
            if candidate.get("plate_number")
        ]
    if event_type in {"face_candidate", "stream_face_candidate"}:
        compacted["face_candidates"] = [
            face.get("candidate")
            for face in payload.get("faces", [])[:5]
            if face.get("candidate")
        ]
    if event_type in {"object_candidate", "stream_object_candidate"}:
        compacted["objects"] = [
            detection.get("label")
            for detection in payload.get("detections", [])[:8]
            if detection.get("label")
        ]
    if event_type == "audio_transcribed" and payload.get("transcript"):
        compacted["transcript_preview"] = str(payload["transcript"])[:300]
    if event_type == "video_summary_generated" and payload.get("content"):
        compacted["summary_preview"] = str(payload["content"])[:300]
    return compacted

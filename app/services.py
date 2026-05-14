from datetime import datetime, timezone
from hashlib import sha256

from app.models import FaceAnalyzeRequest, PlateAnalyzeRequest, ReportRequest
from app.settings import Settings
from app.state import DeviceState


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


def generate_report(request: ReportRequest, settings: Settings, state: DeviceState) -> dict:
    model = choose_llm_model(request, settings, state)
    event_count = len(state.events)
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
    }


from datetime import datetime, timezone
from hashlib import sha256

import httpx

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
    if settings.llm_base_url and model == settings.llm_model:
        try:
            return generate_report_with_llama_cpp(request, settings, state, model)
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            state.audit("llm.report.fallback", {"model": model, "error": str(exc)})

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
        "backend": "simulated-fallback",
    }


def generate_report_with_llama_cpp(
    request: ReportRequest,
    settings: Settings,
    state: DeviceState,
    model: str,
) -> dict:
    event_count = len(state.events)
    recent_events = list(state.events)[-20:]
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

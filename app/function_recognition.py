from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any

from app.models import FunctionRecognizeRequest


@dataclass(frozen=True)
class FunctionSpec:
    name: str
    endpoint: str
    description: str
    keywords: tuple[str, ...]
    media_boosts: dict[str, float]
    required_fields: tuple[str, ...] = ()
    optional_fields: tuple[str, ...] = ()


FUNCTIONS: tuple[FunctionSpec, ...] = (
    FunctionSpec(
        name="device_status",
        endpoint="/api/v1/device/status",
        description="Read device battery, temperature, storage, security, model and stream capacity state.",
        keywords=("设备", "状态", "电量", "温度", "存储", "模型状态", "健康", "算力", "资源"),
        media_boosts={"text": 0.08, "unknown": 0.04},
    ),
    FunctionSpec(
        name="stream_ingest",
        endpoint="/api/v1/streams",
        description="Register a live stream or video file for sampled frame analysis.",
        keywords=("视频流", "实时视频", "接入", "监控", "摄像头", "rtsp", "直播", "抽帧", "记录仪"),
        media_boosts={"stream": 0.28, "video": 0.16},
        required_fields=("source_uri",),
        optional_fields=("stream_id", "camera_id", "sample_fps", "analyze_plate", "analyze_face", "analyze_object"),
    ),
    FunctionSpec(
        name="plate_analyze",
        endpoint="/api/v1/analyze/plate",
        description="Analyze a frame or image for license plate candidates.",
        keywords=("车牌", "牌照", "车辆号牌", "车辆识别", "查车", "布控车辆", "黄牌", "蓝牌", "新能源"),
        media_boosts={"image": 0.2, "video": 0.12, "stream": 0.12},
        required_fields=("frame_id",),
        optional_fields=("camera_id", "image_uri"),
    ),
    FunctionSpec(
        name="face_analyze",
        endpoint="/api/v1/analyze/face",
        description="Detect faces and return candidate hints from the local authorized library.",
        keywords=("人脸", "人员", "身份", "候选", "比对", "重点人员", "人员库", "嫌疑", "面部"),
        media_boosts={"image": 0.2, "video": 0.12, "stream": 0.12},
        required_fields=("frame_id",),
        optional_fields=("camera_id", "candidate_library", "image_uri"),
    ),
    FunctionSpec(
        name="object_detect",
        endpoint="/api/v1/analyze/object",
        description="Detect object candidates such as people, vehicles and motorcycles.",
        keywords=("目标", "物体", "行人", "人员检测", "车辆检测", "摩托车", "电动车", "背包", "异常目标"),
        media_boosts={"image": 0.18, "video": 0.14, "stream": 0.14},
        required_fields=("frame_id",),
        optional_fields=("camera_id", "image_uri", "target_classes", "confidence_threshold"),
    ),
    FunctionSpec(
        name="asr_transcribe",
        endpoint="/api/v1/asr/transcribe",
        description="Transcribe audio or video speech with local SenseVoice when configured.",
        keywords=("语音", "录音", "转写", "转文字", "听写", "音频", "口述", "说了什么", "asr"),
        media_boosts={"audio": 0.3, "video": 0.1},
        required_fields=("audio_uri",),
        optional_fields=("mission_id", "language", "operator_note"),
    ),
    FunctionSpec(
        name="video_summary",
        endpoint="/api/v1/video/summary",
        description="Generate a structured video timeline summary, optionally polished by the local LLM.",
        keywords=("视频摘要", "总结视频", "时间线", "关键事件", "片段", "看一下视频", "概括视频", "摘要"),
        media_boosts={"video": 0.24, "stream": 0.18},
        required_fields=("mission_id",),
        optional_fields=("stream_id", "operator_note", "event_limit", "use_llm"),
    ),
    FunctionSpec(
        name="report_generate",
        endpoint="/api/v1/llm/report",
        description="Generate daily patrol, handover, incident or video report drafts with the local LLM.",
        keywords=("报告", "日报", "巡逻日报", "交接班", "说明", "生成文书", "执法摘要", "异常事件报告"),
        media_boosts={"text": 0.16, "video": 0.08, "unknown": 0.08},
        required_fields=("mission_id",),
        optional_fields=("report_type", "prefer_quality", "operator_note"),
    ),
    FunctionSpec(
        name="evidence_register",
        endpoint="/api/v1/evidence",
        description="Register evidence, compute SHA-256, and store an encrypted copy by default.",
        keywords=("证据", "存证", "登记", "哈希", "加密", "证据链", "原始视频", "附件", "留存"),
        media_boosts={"video": 0.12, "audio": 0.12, "image": 0.12},
        required_fields=("file_uri",),
        optional_fields=("mission_id", "evidence_type", "encrypt", "note"),
    ),
    FunctionSpec(
        name="sync_task",
        endpoint="/api/v1/sync/tasks",
        description="Create an offline or HTTP sync task for events and audit records.",
        keywords=("同步", "上传", "回传", "后台", "指挥中心", "弱网", "离线", "任务队列"),
        media_boosts={"text": 0.08, "unknown": 0.06},
        optional_fields=("mission_id", "destination_url", "include_events", "include_audit", "event_limit"),
    ),
    FunctionSpec(
        name="security_status",
        endpoint="/api/v1/security/certificates",
        description="Check certificate and mTLS readiness.",
        keywords=("证书", "mtls", "双向 tls", "安全", "api key", "鉴权", "加固", "密钥"),
        media_boosts={"text": 0.08, "unknown": 0.06},
    ),
)


def recognize_function(request: FunctionRecognizeRequest) -> dict[str, Any]:
    text = normalize(request.text)
    context = request.context or {}
    allowed = set(request.candidate_functions or [])
    scored = []
    for spec in FUNCTIONS:
        if allowed and spec.name not in allowed:
            continue
        score, matched = score_function(spec, text, request.media_type)
        missing_fields = [field for field in spec.required_fields if field not in context]
        payload_hint = build_payload_hint(spec, context, request)
        scored.append(
            {
                "function": spec.name,
                "endpoint": spec.endpoint,
                "description": spec.description,
                "confidence": round(min(score, 0.99), 3),
                "matched_keywords": matched,
                "required_fields": list(spec.required_fields),
                "optional_fields": list(spec.optional_fields),
                "missing_fields": missing_fields,
                "actionable": not missing_fields,
                "payload_hint": payload_hint,
            }
        )
    scored.sort(key=lambda item: item["confidence"], reverse=True)
    top = scored[0] if scored else unknown_result(request)
    return {
        "backend": "keyword-rules",
        "input_text": request.text,
        "media_type": request.media_type,
        "function": top["function"],
        "endpoint": top["endpoint"],
        "confidence": top["confidence"],
        "actionable": top["actionable"],
        "missing_fields": top["missing_fields"],
        "payload_hint": top["payload_hint"],
        "candidates": scored[:5],
        "requires_human_confirmation": top["confidence"] < 0.7,
    }


def score_function(spec: FunctionSpec, text: str, media_type: str) -> tuple[float, list[str]]:
    matched = [keyword for keyword in spec.keywords if keyword in text]
    score = 0.18 if matched else 0.03
    score += min(len(matched) * 0.18, 0.54)
    score += spec.media_boosts.get(media_type, 0.0)
    if any(keyword in text for keyword in ("帮我", "请", "开始", "生成", "分析", "识别", "查看")):
        score += 0.04
    return score, matched


def build_payload_hint(spec: FunctionSpec, context: dict, request: FunctionRecognizeRequest) -> dict[str, Any]:
    hint = {field: context[field] for field in spec.required_fields + spec.optional_fields if field in context}
    extracted = extract_common_fields(request.text)
    for key, value in extracted.items():
        hint.setdefault(key, value)
    if spec.name in {"plate_analyze", "face_analyze", "object_detect"}:
        hint.setdefault("frame_id", context.get("frame_id", "frame-from-client"))
        hint.setdefault("camera_id", context.get("camera_id", "bodycam-01"))
    if spec.name == "report_generate":
        hint.setdefault("report_type", infer_report_type(request.text))
        hint.setdefault("operator_note", request.text)
    if spec.name == "video_summary":
        hint.setdefault("operator_note", request.text)
        hint.setdefault("use_llm", False)
    if spec.name == "stream_ingest":
        hint.setdefault("sample_fps", 1.0)
        hint.setdefault("analyze_plate", True)
        hint.setdefault("analyze_face", True)
    if spec.name == "asr_transcribe":
        hint.setdefault("language", "zh")
        hint.setdefault("operator_note", request.text)
    return hint


def extract_common_fields(text: str) -> dict[str, str]:
    fields = {}
    mission = re.search(r"(mission[-_a-zA-Z0-9]+|任务[：:\s]*([a-zA-Z0-9_-]+))", text)
    if mission:
        fields["mission_id"] = mission.group(2) or mission.group(1)
    stream = re.search(r"(stream[-_a-zA-Z0-9]+|rtsp://\S+|https?://\S+|\S+\.(?:mp4|avi|mov|mkv))", text, re.IGNORECASE)
    if stream:
        value = stream.group(1).rstrip("，。,.")
        if value.startswith(("rtsp://", "http://", "https://")) or "." in value:
            fields["source_uri"] = value
            fields["file_uri"] = value
            fields["audio_uri"] = value
        else:
            fields["stream_id"] = value
    return fields


def infer_report_type(text: str) -> str:
    normalized = normalize(text)
    if "交接" in normalized or "交班" in normalized:
        return "handover"
    if "异常" in normalized or "事件" in normalized:
        return "incident"
    if "视频" in normalized:
        return "video_summary"
    return "daily"


def normalize(text: str) -> str:
    return text.strip().lower()


def unknown_result(request: FunctionRecognizeRequest) -> dict[str, Any]:
    return {
        "function": "unknown",
        "endpoint": None,
        "description": "No known cerebellum function matched the request.",
        "confidence": 0.0,
        "matched_keywords": [],
        "required_fields": [],
        "optional_fields": [],
        "missing_fields": [],
        "actionable": False,
        "payload_hint": {"operator_note": request.text},
    }

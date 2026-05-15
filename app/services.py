from datetime import datetime, timezone
from hashlib import sha256
import re
from uuid import uuid4

import httpx

from app.asr import transcribe_audio
from app.evidence import list_evidence
from app.media import resolve_media_path
from app.models import (
    AsrTranscribeRequest,
    FaceAnalyzeRequest,
    ObjectDetectRequest,
    PlateAnalyzeRequest,
    ReportRequest,
    VideoSummaryRequest,
)
from app.objects import detect_objects
from app.report_document import generate_report_docx
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
    report_id = f"rpt-{uuid4().hex[:12]}"
    structured_context = build_report_structured_context(request, settings, state)
    if settings.llm_base_url and model == settings.llm_model:
        try:
            report = generate_report_with_llama_cpp(request, settings, state, model, report_id, structured_context)
            attach_report_document(report, request, settings, state)
            report["backend_submit"] = submit_report_to_backend(report, settings, state) if request.submit_to_backend else None
            return report
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            state.audit("llm.report.fallback", {"model": model, "error": str(exc)})

    now = datetime.now(timezone.utc).isoformat()
    content = build_structured_report_content(request, structured_context, model)
    report = {
        "report_id": report_id,
        "mission_id": request.mission_id,
        "report_type": request.report_type,
        "operator_id": request.operator_id,
        "officer_name": request.officer_name,
        "device_id": request.device_id or settings.device_id,
        "model": model,
        "context_tokens": settings.context_tokens,
        "max_context_tokens": settings.max_context_tokens,
        "generated_at": now,
        "content": content,
        "structured_context": structured_context,
        "media_selection": structured_context["media_selection"],
        "requires_human_confirmation": True,
        "backend": "structured-report-fallback",
    }
    attach_report_document(report, request, settings, state)
    report["backend_submit"] = submit_report_to_backend(report, settings, state) if request.submit_to_backend else None
    return report


def attach_report_document(report: dict, request: ReportRequest, settings: Settings, state: DeviceState) -> None:
    if request.output_format != "docx":
        report["document"] = None
        return
    try:
        report["document"] = generate_report_docx(report, settings)
    except Exception as exc:
        state.audit("report.docx.failed", {"report_id": report.get("report_id"), "error": str(exc)})
        report["document"] = {"format": "docx", "status": "failed", "error": str(exc)}


def build_structured_report_content(request: ReportRequest, structured_context: dict, model: str) -> str:
    media_items = structured_context.get("media_items", [])
    type_counts = media_type_counts(media_items)
    scene_hint = infer_scene_hint(request, media_items)
    generated_at = structured_context.get("generated_at") or datetime.now(timezone.utc).isoformat()
    report_date = report_date_text(generated_at)
    case_id = case_id_for_report(report_date)
    candidate_summary = summarize_candidate_findings(media_items)
    scene_text = build_case_scene_text(request, media_items, scene_hint, candidate_summary)
    disposal_text = build_case_disposal_text(media_items, model)
    attachment_sections = build_attachment_sections(media_items)
    multisource = structured_context.get("multisource_summary") or build_multisource_summary(media_items)
    key_status = scene_hint or candidate_summary or f"已纳入 {len(media_items)} 个上传文件（{format_type_counts(type_counts)}）"
    return "\n".join(
        [
            "单警工作日报",
            "",
            "基础信息",
            "",
            f"日期：{report_date}",
            f"值班民警：{request.officer_name or ''}",
            f"警号：{request.operator_id or ''}",
            "班次：",
            "巡逻区域：",
            "",
            "今日工作情况",
            "",
            "接警数量：",
            "处警数量：",
            "巡逻时长：",
            f"重点情况：{key_status}",
            "",
            "多源材料汇总",
            "",
            f"视频音频总结：{join_summary_items(multisource.get('video_summaries') or [])}",
            f"录音总结：{join_summary_items(multisource.get('audio_summaries') or [])}",
            "图片摘要：当前日报生成策略未启用图像识别，图片仅作为附件留存，正文不基于图片作事实判断。",
            f"综合结论：{multisource.get('overall') or '未形成可汇总的多源材料结论。'}",
            "",
            "警情记录",
            "",
            f"案件编号：{case_id}",
            "",
            "基本信息",
            "",
            "警情类型：",
            f"时间：{report_datetime_text(generated_at)}",
            "地点：",
            "涉及人员：",
            "",
            "现场情况",
            "",
            scene_text,
            "",
            "处置结果",
            "",
            disposal_text,
            "",
            "附件信息",
            "",
            attachment_sections,
            "",
            "备注",
            "",
            build_report_remark(request, media_items),
            "",
            "签字",
            "",
            "值班民警：",
            "审核人：",
        ]
    )


def report_date_text(generated_at: str) -> str:
    try:
        return datetime.fromisoformat(generated_at.replace("Z", "+00:00")).astimezone().date().isoformat()
    except ValueError:
        return datetime.now().astimezone().date().isoformat()


def report_datetime_text(generated_at: str) -> str:
    try:
        return datetime.fromisoformat(generated_at.replace("Z", "+00:00")).astimezone().strftime("%Y-%m-%d %H:%M:%S")
    except ValueError:
        return datetime.now().astimezone().strftime("%Y-%m-%d %H:%M:%S")


def case_id_for_report(report_date: str) -> str:
    return f"{report_date.replace('-', '')}-001"


def build_case_scene_text(
    request: ReportRequest,
    media_items: list[dict],
    scene_hint: str | None,
    candidate_summary: str,
) -> str:
    lines = []
    if request.operator_note:
        lines.append(f"人工补充：{request.operator_note}")
    if scene_hint:
        lines.append(f"根据上传文件名称、人工补充或转写线索，现场主题倾向为：{scene_hint}。")
    if media_items:
        lines.append(f"本次材料包含 {format_type_counts(media_type_counts(media_items))}。")
        for index, item in enumerate(media_items[:10], start=1):
            summary = summarize_media_item_for_llm(item)
            lines.append(f"{index}. {media_type_label(item.get('media_type') or 'unknown')}《{item.get('source_name') or item.get('source_uri') or '未命名媒体'}》：{summary}")
    else:
        lines.append("未检索到已上传的视频、图片或录音材料。")
    if candidate_summary:
        lines.append(f"专项识别结果：{candidate_summary}")
    return "\n".join(lines)


def build_case_disposal_text(media_items: list[dict], model: str) -> str:
    if not media_items:
        return "当前未纳入可分析附件，需补充上传原始视频、图片或录音后再完善处置结果。"
    failed = [
        item for item in media_items
        if (item.get("analysis") or {}).get("status") == "failed"
    ]
    lines = [
        f"已由本地小脑对上传附件进行结构化整理，生成模型：{model}。",
        "本日报为 AI 草稿，现场情况、处置经过及最终结果需由值班民警结合原始附件复核后确认。",
    ]
    if failed:
        lines.append("部分附件分析失败：" + "；".join(
            f"{item.get('source_name') or item.get('source_uri')}: {(item.get('analysis') or {}).get('error') or '未知错误'}"
            for item in failed[:5]
        ))
    return "\n".join(lines)


def build_attachment_sections(media_items: list[dict]) -> str:
    groups = [
        ("图片", "image", "现场图片"),
        ("视频", "video", "执法记录"),
        ("录音", "audio", "现场录音"),
    ]
    sections = []
    for title, media_type, label in groups:
        sections.append(title)
        items = [item for item in media_items if item.get("media_type") == media_type]
        if items:
            for item in items:
                name = item.get("source_name") or item.get("source_uri") or "未命名附件"
                sections.append(f"{name}（{label}）")
        else:
            sections.append("无")
        sections.append("")
    return "\n".join(sections).rstrip()


def build_report_remark(request: ReportRequest, media_items: list[dict]) -> str:
    parts = [
        "本日报由边缘小脑根据上传附件生成，正式入库前需人工复核。",
    ]
    if request.operator_note:
        parts.append(f"人工补充：{request.operator_note}")
    if media_items:
        parts.append("附件 SHA-256 已写入结构化上下文，可用于后续证据链核验。")
    return "\n".join(parts)


def build_multisource_summary(media_items: list[dict]) -> dict:
    video_summaries = []
    audio_summaries = []
    image_summaries = []
    for item in media_items:
        summary = summarize_media_item_for_llm(item)
        name = item.get("source_name") or item.get("source_uri") or "未命名附件"
        text = f"{name}：{summary}"
        if item.get("media_type") == "video":
            video_summaries.append(text)
        elif item.get("media_type") == "audio":
            audio_summaries.append(text)
        elif item.get("media_type") == "image":
            image_summaries.append(text)
    candidate_summary = summarize_candidate_findings(media_items)
    source_parts = []
    if video_summaries:
        source_parts.append(f"视频 {len(video_summaries)} 个")
    if audio_summaries:
        source_parts.append(f"录音 {len(audio_summaries)} 个")
    if image_summaries:
        source_parts.append(f"图片 {len(image_summaries)} 个（未做图像识别）")
    if candidate_summary:
        overall = f"已汇总{ '、'.join(source_parts) or '多源材料' }；{candidate_summary}。"
    elif source_parts:
        audio_count = len(video_summaries) + len(audio_summaries)
        if audio_count:
            overall = f"已汇总当天 {audio_count} 条音频材料（含视频音轨和录音），日报正文以音频转写摘要为主要依据，需人工复核原始录音。"
        else:
            overall = "当前未纳入可分析音频材料；图片和视频画面不参与本阶段日报分析。"
    else:
        overall = "未纳入视频音轨或录音材料。"
    return {
        "media_counts": media_type_counts(media_items),
        "video_summaries": video_summaries,
        "audio_summaries": audio_summaries,
        "image_summaries": image_summaries,
        "overall": overall,
    }


def join_summary_items(items: list[str]) -> str:
    values = [item for item in items if item]
    return "；".join(values[:6]) if values else "未纳入该类材料。"


def media_type_counts(media_items: list[dict]) -> dict:
    counts = {}
    for item in media_items:
        media_type = item.get("media_type") or "unknown"
        counts[media_type] = counts.get(media_type, 0) + 1
    return counts


def format_type_counts(type_counts: dict) -> str:
    labels = {"video": "视频", "audio": "录音", "image": "图片", "other": "其他", "unknown": "未知"}
    if not type_counts:
        return "无"
    return "、".join(f"{labels.get(key, key)} {value} 个" for key, value in sorted(type_counts.items()))


def format_media_analysis_lines(index: int, item: dict) -> list[str]:
    media_type = item.get("media_type") or "unknown"
    source_name = item.get("source_name") or item.get("source_uri") or "未命名媒体"
    evidence_id = item.get("evidence_id") or "-"
    analysis = item.get("analysis") or {}
    header = f"{index}. {media_type_label(media_type)}《{source_name}》（证据ID：{evidence_id}）"
    if not analysis:
        return [header, "   暂无分析结果。"]
    if analysis.get("status") == "failed":
        return [header, f"   分析失败，原因：{analysis.get('error') or '未知错误'}。"]
    if analysis.get("status") == "skipped":
        return [header, f"   跳过分析，原因：{analysis.get('reason') or '不支持的媒体类型'}。"]
    if media_type == "video":
        return [header, "   " + summarize_video_media_analysis(analysis)]
    if media_type == "audio":
        return [header, "   " + summarize_audio_media_analysis(analysis)]
    if media_type == "image":
        return [header, "   " + summarize_image_media_analysis(analysis)]
    return [header, "   " + (analysis.get("structured_text") or "已登记为证据，未形成专项分析。")]


def media_type_label(media_type: str) -> str:
    return {
        "video": "视频",
        "audio": "录音",
        "image": "图片",
        "other": "其他",
        "unknown": "未知",
    }.get(media_type, media_type)


def summarize_video_media_analysis(analysis: dict) -> str:
    lines = ["视频画面分析已关闭；本阶段仅提取视频音轨并按录音处理。"]
    audio = analysis.get("audio") or {}
    if real_audio_transcript(audio):
        lines.append("音轨总结：" + summarize_transcript_for_report(str(audio.get("transcript") or ""), "视频音轨"))
    elif audio.get("status") == "failed":
        lines.append(f"视频音轨转写失败：{audio.get('error') or '未知错误'}。")
    elif audio:
        lines.append("视频内音频已提取元数据，但未配置真实 ASR，未形成可采信转写。")
    else:
        lines.append("未取得可用于日报总结的视频音轨。")
    return "".join(lines)


def frame_has_reliable_findings(frame: dict) -> bool:
    objects = frame.get("objects") or {}
    if real_object_result(objects) and any(item.get("label") for item in objects.get("detections", []) or []):
        return True
    plates = frame.get("plates") or {}
    if real_plate_result(plates) and any(item.get("plate_number") for item in plates.get("candidates", []) or []):
        return True
    faces = frame.get("faces") or {}
    if real_face_result(faces) and (int(faces.get("face_count") or 0) > 0 or int(faces.get("candidate_count") or 0) > 0):
        return True
    return False


def summarize_audio_media_analysis(analysis: dict) -> str:
    duration = analysis.get("duration_seconds")
    prefix = f"时长 {duration} 秒。" if duration is not None else "时长未知。"
    transcript = analysis.get("transcript")
    if transcript and real_audio_transcript(analysis):
        return prefix + summarize_transcript_for_report(str(transcript), "录音")
    if transcript:
        return prefix + "当前 ASR 为模拟回退，未形成真实转写；录音内容需人工复听或配置真实 ASR 后重新生成。"
    return prefix + f"转写状态：{analysis.get('status') or '未知'}。"


def summarize_video_visual_findings(frames: list[dict]) -> str:
    if not frames:
        return "未取得可用于画面分析的关键帧。"

    object_counts: dict[str, int] = {}
    plate_numbers: list[str] = []
    face_count = 0
    face_candidates = 0
    reliable_frames = 0

    for frame in frames:
        if frame_has_reliable_findings(frame):
            reliable_frames += 1
        objects = frame.get("objects") or {}
        if real_object_result(objects):
            for detection in objects.get("detections", []) or []:
                label = detection.get("label")
                if label:
                    object_counts[label] = object_counts.get(label, 0) + 1
        plates = frame.get("plates") or {}
        if real_plate_result(plates):
            for candidate in plates.get("candidates", []) or []:
                plate_number = candidate.get("plate_number")
                if plate_number and plate_number not in plate_numbers:
                    plate_numbers.append(plate_number)
        faces = frame.get("faces") or {}
        if real_face_result(faces):
            face_count += int(faces.get("face_count") or 0)
            face_candidates += int(faces.get("candidate_count") or 0)

    parts = []
    if object_counts:
        objects_text = "、".join(f"{label} {count} 次" for label, count in sorted(object_counts.items())[:8])
        parts.append(f"画面目标识别到 {objects_text}")
    if plate_numbers:
        parts.append("车牌候选：" + "、".join(plate_numbers[:8]))
    if face_count or face_candidates:
        parts.append(f"多帧共检出人脸 {face_count} 处，人脸库候选 {face_candidates} 处")
    if parts:
        parts.append(f"以上结果来自 {reliable_frames} 帧可采信关键帧，仍需结合原视频复核")
        return "；".join(parts) + "。"
    return "关键帧未形成可直接写入事实判断的可靠视觉识别结论，需人工查看原视频确认具体画面内容。"


def summarize_transcript_for_report(transcript: str, source_label: str) -> str:
    text = normalize_transcript_text(transcript)
    if not text:
        return f"{source_label}未形成可用转写。"

    topic_summary = infer_transcript_topic_summary(text, source_label)
    if topic_summary:
        return topic_summary

    summary = extractive_transcript_summary(text)
    return f"{source_label}主要内容：{summary}。"


def normalize_transcript_text(transcript: str) -> str:
    return re.sub(r"\s+", " ", transcript or "").strip()


def infer_transcript_topic_summary(text: str, source_label: str) -> str | None:
    lowered = text.lower()
    if contains_any(lowered, ["篮球", "球员", "投篮", "运球", "篮板", "扣篮", "跑位", "出手", "curry"]):
        points = []
        if "二号" in text or "2号" in text:
            points.append("二号球员表现被重点评价")
        if "白衣" in text or "白色球衣" in text:
            points.append("对位白衣球员的跑位和出手被反复提及")
        if contains_any(text, ["节奏", "运球"]):
            points.append("运球节奏和进攻组织较受关注")
        if contains_any(text, ["扣篮", "投篮", "出手"]):
            points.append("投篮、出手或扣篮表现是主要观察点")
        if contains_any(text, ["气势", "热血", "加油", "进了"]):
            points.append("现场气氛和进球后的情绪反馈较明显")
        detail = "，".join(points[:5]) if points else "重点围绕球员对抗、跑位、运球节奏、投篮表现和现场气氛展开"
        return f"{source_label}主要记录篮球对抗或训练场景点评，{detail}。整体倾向为对现场运动表现和配合效果的口述评价。"

    if contains_any(text, ["巡逻", "执勤", "警情", "处置", "盘查", "报警", "嫌疑", "询问"]):
        summary = extractive_transcript_summary(text)
        return f"{source_label}主要记录巡逻执勤或现场处置相关内容，核心信息为：{summary}。"

    if contains_any(text, ["车辆", "车牌", "司机", "驾驶", "路口", "道路"]):
        summary = extractive_transcript_summary(text)
        return f"{source_label}主要涉及车辆或道路现场信息，核心内容为：{summary}。"

    return None


def contains_any(text: str, keywords: list[str]) -> bool:
    return any(keyword in text for keyword in keywords)


def extractive_transcript_summary(text: str, max_sentences: int = 3) -> str:
    sentences = [item.strip(" ，。；;、") for item in re.split(r"[。！？!?；;\n]+", text) if item.strip()]
    if not sentences:
        sentences = [text[i : i + 80] for i in range(0, min(len(text), 240), 80)]

    priority_keywords = [
        "重点",
        "情况",
        "发现",
        "对面",
        "现场",
        "人员",
        "车辆",
        "处置",
        "节奏",
        "表现",
        "结果",
    ]
    selected = []
    for sentence in sentences:
        if any(keyword in sentence for keyword in priority_keywords):
            selected.append(sentence)
        if len(selected) >= max_sentences:
            break
    for sentence in sentences:
        if len(selected) >= max_sentences:
            break
        if sentence not in selected:
            selected.append(sentence)

    summary = "；".join(selected[:max_sentences])
    if len(summary) > 260:
        summary = summary[:257].rstrip("，,；;。") + "..."
    return summary


def summarize_image_media_analysis(analysis: dict) -> str:
    if analysis.get("status") == "skipped":
        return analysis.get("reason") or "当前日报生成策略未启用图像识别。"
    if analysis.get("structured_text"):
        return analysis["structured_text"]
    objects = analysis.get("objects", {})
    plates = analysis.get("plates", {})
    faces = analysis.get("faces", {})
    return image_analysis_text(objects, plates, faces)


def summarize_candidate_findings(media_items: list[dict]) -> str:
    plate_numbers = []
    object_labels = {}
    face_count = 0
    face_candidates = 0
    audio_transcripts = 0
    for item in media_items:
        collect_candidate_findings(item.get("analysis") or {}, plate_numbers, object_labels)
        analysis = item.get("analysis") or {}
        if item.get("media_type") == "audio" and real_audio_transcript(analysis):
            audio_transcripts += 1
        for frame in analysis.get("sampled_frames", []) or []:
            faces = (frame.get("faces") or {})
            if real_face_result(faces):
                face_count += int(faces.get("face_count") or 0)
                face_candidates += int(faces.get("candidate_count") or 0)
        faces = analysis.get("faces") or {}
        if real_face_result(faces):
            face_count += int(faces.get("face_count") or 0)
            face_candidates += int(faces.get("candidate_count") or 0)
        audio = analysis.get("audio") or {}
        if real_audio_transcript(audio):
            audio_transcripts += 1
    parts = []
    if object_labels:
        parts.append("目标候选：" + "、".join(f"{label} {count} 次" for label, count in sorted(object_labels.items())))
    if plate_numbers:
        parts.append("车牌候选：" + "、".join(plate_numbers[:10]))
    if face_count or face_candidates:
        parts.append(f"人脸检出 {face_count} 次，其中候选提示 {face_candidates} 次")
    if audio_transcripts:
        parts.append(f"形成音频转写 {audio_transcripts} 条")
    return "；".join(parts)


def collect_candidate_findings(analysis: dict, plate_numbers: list[str], object_labels: dict) -> None:
    objects = analysis.get("objects") or {}
    if real_object_result(objects):
        for detection in objects.get("detections", []) or []:
            label = detection.get("label")
            if label:
                object_labels[label] = object_labels.get(label, 0) + 1
    plates = analysis.get("plates") or {}
    if real_plate_result(plates):
        for candidate in plates.get("candidates", []) or []:
            plate_number = candidate.get("plate_number")
            if plate_number and plate_number not in plate_numbers:
                plate_numbers.append(plate_number)
    for frame in analysis.get("sampled_frames", []) or []:
        collect_candidate_findings(frame, plate_numbers, object_labels)


def real_object_result(result: dict) -> bool:
    return bool(result) and result.get("backend") not in {None, "simulated-fallback"} and not result.get("fallback_error")


def real_plate_result(result: dict) -> bool:
    return bool(result) and result.get("backend") == "hyperlpr3" and not result.get("fallback_error")


def real_face_result(result: dict) -> bool:
    return bool(result) and result.get("backend") == "opencv-zoo-yunet+sface" and not result.get("fallback_error")


def real_audio_transcript(result: dict) -> bool:
    return bool(result.get("transcript")) and result.get("backend") not in {None, "simulated-fallback"}


def infer_scene_hint(request: ReportRequest, media_items: list[dict]) -> str | None:
    text_parts = [request.operator_note or ""]
    for item in media_items:
        text_parts.extend(
            [
                str(item.get("source_name") or ""),
                str(item.get("note") or ""),
            ]
        )
        analysis = item.get("analysis") or {}
        if real_audio_transcript(analysis):
            text_parts.append(str(analysis.get("transcript") or ""))
        audio = analysis.get("audio") or {}
        if real_audio_transcript(audio):
            text_parts.append(str(audio.get("transcript") or ""))
    text = "\n".join(text_parts).lower()
    scene_keywords = [
        (("篮球", "basketball", "投篮", "运球", "球场", "篮筐", "篮板"), "篮球活动场景"),
        (("足球", "football", "soccer"), "足球活动场景"),
        (("跑步", "慢跑", "田径"), "运动锻炼场景"),
        (("询问", "问话", "谈话", "录音"), "现场询问或口述记录"),
        (("巡逻", "执勤", "街", "路面"), "巡逻执勤场景"),
    ]
    for keywords, label in scene_keywords:
        if any(keyword in text for keyword in keywords):
            return label
    return None


def sanitize_media_items_for_report(media_items: list[dict]) -> list[dict]:
    sanitized = []
    for item in media_items:
        analysis = item.get("analysis") or {}
        sanitized.append(
            {
                "evidence_id": item.get("evidence_id"),
                "media_type": item.get("media_type"),
                "source_name": item.get("source_name"),
                "sha256": item.get("sha256"),
                "note": item.get("note"),
                "summary": summarize_media_item_for_llm(item),
                "raw_simulated_results_removed": True,
                "analysis_status": analysis.get("status"),
                "analysis_backend": analysis.get("backend"),
            }
        )
    return sanitized


def summarize_media_item_for_llm(item: dict) -> str:
    media_type = item.get("media_type")
    analysis = item.get("analysis") or {}
    if not analysis:
        return "暂无分析结果"
    if analysis.get("status") in {"failed", "skipped"}:
        return analysis.get("error") or analysis.get("reason") or "未形成分析结果"
    if media_type == "video":
        return summarize_video_media_analysis(analysis)
    if media_type == "audio":
        return summarize_audio_media_analysis(analysis)
    if media_type == "image":
        return summarize_image_media_analysis(analysis)
    return analysis.get("structured_text") or "已登记为证据"


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
    report_id: str,
    structured_context: dict,
) -> dict:
    report_events = structured_context["events"]
    event_count = len(report_events)
    recent_events = report_events[-20:]
    system_prompt = (
        "你是部署在单兵边缘智能小脑服务器中的警务报告助手。"
        "你只能根据输入的结构化媒体、结构化事件、人工备注和设备状态生成报告草稿。"
        "当前策略只分析录音和视频音轨，不进行图片识别、视频抽帧、人脸识别、车牌识别或目标检测。"
        "不得把图片或视频画面写成事实依据；视频只能根据音轨转写内容总结。"
        "backend 为 simulated-fallback 的 ASR 结果不是事实，不得写入确定结论。"
        "必须逐项参考结构化媒体中的 analysis 字段，围绕一天处理案件相关录音形成工作日报总结。"
        "如果音频转写失败或只得到模拟回退，也要在正文中明确写出失败原因或不可采信性质。"
        "禁止只输出通用模板，禁止只写已汇总、已整理而不说明具体媒体内容。"
        "输出必须严格使用用户提供的 Word 版日报结构：单警工作日报、基础信息、今日工作情况、警情记录、附件信息、备注、签字。"
        "正文只输出普通文本，不要使用 Markdown 标题符号、Markdown 表格、分隔线或代码块。"
        "现场情况、处置结果、备注必须是普通段落文本，不要使用代码块。"
        "如果人工补充或文件信息指向篮球、运动等生活场景，报告必须围绕该场景组织，不得套用商业街巡逻、车辆盘查、嫌疑人识别模板。"
        "输出中文，风格简洁、正式、可用于日报初稿。"
        "不要输出思考过程，只输出最终报告正文。"
    )
    llm_context = structured_context.get("llm_context") or build_report_llm_context(structured_context, settings)
    user_prompt = (
        f"任务编号：{request.mission_id}\n"
        f"报告类型：{request.report_type}\n"
        f"人工补充：{request.operator_note or '无'}\n"
        f"当前结构化事件数量：{event_count}\n"
        f"媒体选择：{structured_context['media_selection']}\n"
        f"多文件分层摘要：{llm_context}\n"
        f"最近事件：{recent_events}\n"
        "请严格按 Word 版日报模版生成正文，保留基础信息表、今日工作情况、警情记录、附件信息、备注、签字这些栏目。"
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
    content = normalize_plain_report_content((message.get("content") or message.get("reasoning_content") or "").strip())
    if not content:
        raise ValueError("llama.cpp returned an empty response")
    return {
        "report_id": report_id,
        "mission_id": request.mission_id,
        "report_type": request.report_type,
        "operator_id": request.operator_id,
        "officer_name": request.officer_name,
        "device_id": request.device_id or settings.device_id,
        "model": model,
        "context_tokens": settings.context_tokens,
        "max_context_tokens": settings.max_context_tokens,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "content": content,
        "structured_context": structured_context,
        "media_selection": structured_context["media_selection"],
        "requires_human_confirmation": True,
        "backend": "llama.cpp",
    }


def normalize_plain_report_content(content: str) -> str:
    lines = []
    for line in content.splitlines():
        cleaned = line.strip()
        cleaned = re.sub(r"^#{1,6}\s*", "", cleaned)
        cleaned = re.sub(r"^\|?\s*-{2,}.*$", "", cleaned)
        cleaned = re.sub(r"^\|\s*", "", cleaned)
        cleaned = re.sub(r"\s*\|\s*", "：", cleaned).strip("： ")
        cleaned = re.sub(r"^[-*]\s+", "", cleaned)
        if cleaned:
            lines.append(cleaned)
    return "\n".join(lines)


def build_report_structured_context(request: ReportRequest, settings: Settings, state: DeviceState) -> dict:
    selected_ids = {item.strip() for item in request.selected_media_ids if item.strip()}
    selected_uris = {item.strip() for item in request.selected_media_uris if item.strip()}
    all_evidence = list_evidence(settings)
    if selected_ids or selected_uris:
        selected = [
            item for item in all_evidence
            if item.get("evidence_id") in selected_ids
            or item.get("source_uri") in selected_uris
            or item.get("source_name") in selected_uris
        ]
        mode = "explicit"
    elif request.include_today_media_default:
        selected = [
            item for item in all_evidence
            if item.get("evidence_type") in {"video", "audio"}
            and is_today(item.get("registered_at"))
        ]
        mode = "today_default"
    else:
        selected = []
        mode = "none"

    known_refs = {item.get("evidence_id") for item in selected} | {item.get("source_uri") for item in selected}
    external_items = [
        {"source_uri": uri, "evidence_id": None, "evidence_type": infer_media_type(uri), "source_name": uri.rsplit("/", 1)[-1]}
        for uri in sorted(selected_uris)
        if uri not in known_refs
    ]
    media_items = [compact_media_item(item) for item in selected] + [compact_media_item(item) for item in external_items]
    media_items = analyze_report_media_items(media_items, request, settings, state)
    compact_events = compact_events_for_llm(state.event_snapshot(limit=1000), limit=200)
    media_context_text = build_media_context_text(media_items, compact_events)
    multisource_summary = build_multisource_summary(media_items)
    structured_context = {
        "mission_id": request.mission_id,
        "report_type": request.report_type,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "media_selection": {
            "mode": mode,
            "selected_media_ids": sorted(selected_ids),
            "selected_media_uris": sorted(selected_uris),
            "include_today_media_default": request.include_today_media_default,
        },
        "media_count": len(media_items),
        "media_items": media_items,
        "multisource_summary": multisource_summary,
        "event_count": len(compact_events),
        "events": compact_events,
        "structured_text": media_context_text,
    }
    structured_context["llm_context"] = build_report_llm_context(structured_context, settings)
    return structured_context


def build_report_llm_context(structured_context: dict, settings: Settings) -> dict:
    media_items = structured_context.get("media_items") or []
    multisource = structured_context.get("multisource_summary") or build_multisource_summary(media_items)
    budget_chars = report_llm_context_budget_chars(settings)
    per_item_chars = report_llm_per_item_chars(len(media_items))
    remaining = budget_chars
    per_media_summaries = []
    omitted = 0

    for index, item in enumerate(media_items, start=1):
        summary = summarize_media_item_for_llm(item)
        analysis = item.get("analysis") or {}
        entry = {
            "index": index,
            "media_type": item.get("media_type"),
            "source_name": item.get("source_name") or item.get("source_uri"),
            "evidence_id": item.get("evidence_id"),
            "sha256": item.get("sha256"),
            "note": truncate_text(str(item.get("note") or ""), 160),
            "analysis_status": analysis.get("status"),
            "analysis_backend": analysis.get("backend"),
            "summary": truncate_text(summary, per_item_chars),
        }
        entry_size = len(str(entry))
        if per_media_summaries and entry_size > remaining:
            omitted = len(media_items) - index + 1
            break
        per_media_summaries.append(entry)
        remaining -= entry_size

    return {
        "strategy": "per-media-summary-then-final-report",
        "input_budget_chars": budget_chars,
        "media_count": len(media_items),
        "media_counts": multisource.get("media_counts") or media_type_counts(media_items),
        "overall": truncate_text(str(multisource.get("overall") or ""), 900),
        "per_media_summary_count": len(per_media_summaries),
        "omitted_media_count": omitted,
        "per_media_summaries": per_media_summaries,
        "events_summary": {
            "event_count": structured_context.get("event_count"),
            "recent_event_count_in_prompt": min(len(structured_context.get("events") or []), 20),
        },
        "safety_note": "本阶段只分析录音和视频音轨；图片、视频画面、候选识别和模拟回退结果不得写成确定事实。",
    }


def report_llm_context_budget_chars(settings: Settings) -> int:
    return max(8_000, min(settings.context_tokens * 2, 32_000))


def report_llm_per_item_chars(media_count: int) -> int:
    if media_count <= 6:
        return 1_800
    if media_count <= 15:
        return 1_200
    if media_count <= 30:
        return 800
    return 500


def truncate_text(text: str, max_chars: int) -> str:
    if max_chars <= 0 or len(text) <= max_chars:
        return text
    return text[: max_chars - 20].rstrip() + "……[已截断]"


def compact_media_item(item: dict) -> dict:
    return {
        "evidence_id": item.get("evidence_id"),
        "mission_id": item.get("mission_id"),
        "media_type": item.get("evidence_type"),
        "source_name": item.get("source_name"),
        "source_uri": item.get("source_uri"),
        "sha256": item.get("source_sha256") or item.get("sha256"),
        "size_bytes": item.get("size_bytes"),
        "registered_at": item.get("registered_at"),
        "chain_status": item.get("chain_status"),
        "note": item.get("note"),
    }


def analyze_report_media_items(
    media_items: list[dict],
    request: ReportRequest,
    settings: Settings,
    state: DeviceState,
) -> list[dict]:
    analyzed = []
    for item in media_items:
        media_type = str(item.get("media_type") or infer_media_type(str(item.get("source_uri") or ""))).lower()
        source_uri = str(item.get("source_uri") or "")
        item = {**item, "media_type": media_type}
        try:
            if media_type == "audio":
                item["analysis"] = analyze_audio_media(source_uri, request, settings)
            elif media_type == "image":
                item["analysis"] = {
                    "status": "skipped",
                    "kind": "image_analysis_disabled",
                    "reason": "当前日报生成策略仅分析录音和视频音轨，未启用图片识别。",
                }
            elif media_type == "video":
                item["analysis"] = analyze_video_media(source_uri, request, settings, state)
            else:
                item["analysis"] = {
                    "status": "skipped",
                    "reason": f"unsupported media_type: {media_type}",
                }
        except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
            item["analysis"] = {"status": "failed", "error": str(exc)}
        state.add_event(
            "report_media_analyzed",
            {
                "mission_id": request.mission_id,
                "evidence_id": item.get("evidence_id"),
                "source_uri": source_uri,
                "media_type": media_type,
                "analysis_status": item.get("analysis", {}).get("status", "ok"),
            },
        )
        item["summary"] = summarize_media_item_for_llm(item)
        analyzed.append(item)
    return analyzed


def analyze_audio_media(source_uri: str, request: ReportRequest, settings: Settings) -> dict:
    transcript = transcribe_audio(
        AsrTranscribeRequest(
            audio_uri=source_uri,
            mission_id=request.mission_id,
            language="zh",
            operator_note=request.operator_note,
            max_tokens=min(request.max_tokens, 2048),
        ),
        settings,
    )
    return {
        "status": "ok",
        "kind": "audio_transcript",
        "backend": transcript.get("backend"),
        "model": transcript.get("model"),
        "duration_seconds": transcript.get("duration_seconds"),
        "transcript": transcript.get("transcript"),
        "segments": transcript.get("segments", [])[:20],
        "requires_human_confirmation": transcript.get("requires_human_confirmation", True),
    }


def analyze_image_media(source_uri: str, request: ReportRequest, settings: Settings, state: DeviceState) -> dict:
    frame_id = report_frame_id(request.mission_id, source_uri)
    object_result = detect_objects(
        ObjectDetectRequest(
            frame_id=frame_id,
            camera_id=request.device_id or settings.device_id,
            image_uri=source_uri,
            target_classes=["person", "car", "truck", "bus", "motorcycle", "bicycle", "bag"],
        ),
        settings,
    )
    plate_result = analyze_plate_image(
        PlateAnalyzeRequest(frame_id=frame_id, camera_id=request.device_id or settings.device_id, image_uri=source_uri),
        settings,
        state,
    )
    face_result = analyze_face_image(
        FaceAnalyzeRequest(frame_id=frame_id, camera_id=request.device_id or settings.device_id, image_uri=source_uri),
        settings,
        state,
    )
    return {
        "status": "ok",
        "kind": "image_analysis",
        "frame_id": frame_id,
        "objects": object_result,
        "plates": plate_result,
        "faces": face_result,
        "structured_text": image_analysis_text(object_result, plate_result, face_result),
    }


def analyze_video_media(source_uri: str, request: ReportRequest, settings: Settings, state: DeviceState) -> dict:
    audio_analysis = None
    try:
        audio_analysis = analyze_audio_media(source_uri, request, settings)
    except (FileNotFoundError, ValueError, RuntimeError, OSError) as exc:
        audio_analysis = {"status": "failed", "error": str(exc)}
    return {
        "status": "ok",
        "kind": "video_audio_analysis",
        "video_frame_analysis_enabled": False,
        "sampled_frame_count": 0,
        "sampled_frames": [],
        "audio": audio_analysis,
        "structured_text": video_analysis_text([], audio_analysis),
    }


def sample_video_frames(source_uri: str, request: ReportRequest, settings: Settings, max_frames: int = 6) -> list[dict]:
    try:
        import cv2
    except ImportError as exc:
        raise RuntimeError("opencv is required for video frame analysis") from exc
    video_path = resolve_media_path(source_uri, settings)
    capture = cv2.VideoCapture(str(video_path))
    if not capture.isOpened():
        raise RuntimeError(f"failed to open video: {source_uri}")
    frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
    if frame_count <= 0:
        indices = [0]
    else:
        step = max(frame_count // max_frames, 1)
        indices = list(range(0, frame_count, step))[:max_frames]
    output_dir = settings.stream_frame_dir / "report_media" / report_frame_id(request.mission_id, source_uri)
    output_dir.mkdir(parents=True, exist_ok=True)
    frames = []
    try:
        for index in indices:
            capture.set(cv2.CAP_PROP_POS_FRAMES, index)
            ok, frame = capture.read()
            if not ok:
                continue
            frame_id = f"frame-{index:08d}"
            path = output_dir / f"{frame_id}.jpg"
            if not cv2.imwrite(str(path), frame):
                continue
            frames.append(
                {
                    "frame_index": index,
                    "offset_seconds": round(index / fps, 3) if fps > 0 else None,
                    "image_uri": str(path),
                }
            )
    finally:
        capture.release()
    return frames


def image_analysis_text(object_result: dict, plate_result: dict, face_result: dict) -> str:
    parts = []
    if real_object_result(object_result):
        objects = [
            item.get("label")
            for item in object_result.get("detections", [])[:10]
            if item.get("label")
        ]
        if objects:
            parts.append(f"画面目标：{objects}")

    if real_plate_result(plate_result):
        plates = [
            item.get("plate_number")
            for item in plate_result.get("candidates", [])[:10]
            if item.get("plate_number")
        ]
        if plates:
            parts.append(f"车牌识别：{plates}")

    if real_face_result(face_result):
        face_count = face_result.get("face_count", 0)
        candidate_count = face_result.get("candidate_count", 0)
        if face_count or candidate_count:
            parts.append(f"人脸检测：检出 {face_count} 处，人脸库候选 {candidate_count} 处")
    if not parts:
        return "图片已纳入证据；当前未形成可直接写入日报的可靠图像识别结论，需人工查看图片内容。"
    return "；".join(parts) + "。"


def video_analysis_text(frame_results: list[dict], audio_analysis: dict | None) -> str:
    lines = [
        "视频画面分析：当前策略已关闭，不进行抽帧、图像识别、人脸识别或车牌识别。",
    ]
    transcript = (audio_analysis or {}).get("transcript")
    if transcript and real_audio_transcript(audio_analysis or {}):
        lines.append("视频音频总结：" + summarize_transcript_for_report(str(transcript), "视频音轨"))
    elif audio_analysis:
        lines.append("视频音频转写：未配置真实 ASR，未形成可采信转写。")
    return "\n".join(lines)


def report_frame_id(mission_id: str, source_uri: str) -> str:
    digest = sha256(f"{mission_id}:{source_uri}".encode()).hexdigest()[:12]
    return f"report-{digest}"


def build_media_context_text(media_items: list[dict], events: list[dict]) -> str:
    type_counts = {}
    for item in media_items:
        media_type = item.get("media_type") or "unknown"
        type_counts[media_type] = type_counts.get(media_type, 0) + 1
    lines = [
        f"媒体证据共 {len(media_items)} 个，类型统计：{type_counts}。",
        f"结构化事件共 {len(events)} 条；日报上下文仅保留录音和视频音轨摘要，不包含图片识别或视频画面识别。",
    ]
    for index, item in enumerate(media_items[:50], start=1):
        lines.append(
            f"{index}. {item.get('media_type') or 'unknown'}：{item.get('source_name') or item.get('source_uri')}; "
            f"证据ID={item.get('evidence_id') or '-'}; 时间={item.get('registered_at') or '-'}; "
            f"SHA256={item.get('sha256') or '-'}; 备注={item.get('note') or '-'}。"
        )
        analysis = item.get("analysis") or {}
        summary = summarize_media_item_for_llm(item)
        if summary and summary != "暂无分析结果":
            lines.append(f"   分析摘要：{summary}")
        elif analysis.get("transcript"):
            lines.append("   录音转写：ASR 为模拟回退，未形成可采信转写。")
        elif analysis:
            lines.append(f"   分析状态：{analysis.get('status')} {analysis.get('error') or analysis.get('reason') or ''}".strip())
    return "\n".join(lines)


def is_today(value: str | None) -> bool:
    if not value:
        return False
    try:
        normalized = value.replace("Z", "+00:00")
        parsed = datetime.fromisoformat(normalized)
        return parsed.astimezone().date() == datetime.now().astimezone().date()
    except ValueError:
        return False


def infer_media_type(uri: str) -> str:
    suffix = uri.rsplit(".", 1)[-1].lower() if "." in uri else ""
    if suffix in {"mp4", "mov", "avi", "mkv", "webm"}:
        return "video"
    if suffix in {"mp3", "wav", "m4a", "aac", "opus", "flac"}:
        return "audio"
    if suffix in {"jpg", "jpeg", "png", "webp", "bmp"}:
        return "image"
    return "other"


def submit_report_to_backend(report: dict, settings: Settings, state: DeviceState) -> dict | None:
    if not settings.backend_base_url or not settings.backend_token:
        return {"status": "skipped", "reason": "backend_base_url_or_token_missing"}
    payload = {
        "reportId": report.get("report_id"),
        "missionId": report.get("mission_id"),
        "reportType": report.get("report_type"),
        "deviceId": settings.device_id,
        "operatorId": report.get("operator_id"),
        "officerName": report.get("officer_name"),
        "model": report.get("model"),
        "backend": report.get("backend"),
        "generatedAt": report.get("generated_at"),
        "content": report.get("content"),
        "documentUri": (report.get("document") or {}).get("download_url"),
        "documentName": (report.get("document") or {}).get("file_name"),
        "documentFormat": (report.get("document") or {}).get("format"),
        "requiresHumanConfirmation": report.get("requires_human_confirmation"),
        "mediaSelection": report.get("media_selection"),
        "structuredContext": report.get("structured_context"),
    }
    url = f"{settings.backend_base_url.rstrip('/')}/api/v1/cerebellum/daily-reports"
    headers = {
        "Authorization": f"Bearer {settings.backend_token}",
        "X-Cerebellum-Token": settings.backend_token,
    }
    try:
        response = httpx.post(url, json=payload, headers=headers, timeout=30.0)
        response.raise_for_status()
        state.audit("backend.daily_report.submit", {"report_id": report.get("report_id"), "status_code": response.status_code})
        return {"status": "submitted", "status_code": response.status_code}
    except httpx.HTTPError as exc:
        state.audit("backend.daily_report.submit_failed", {"report_id": report.get("report_id"), "error": str(exc)})
        return {"status": "failed", "error": str(exc)}


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

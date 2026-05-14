import json
import subprocess
from datetime import datetime, timezone

import httpx

from app.media import resolve_media_path
from app.models import AsrTranscribeRequest
from app.settings import Settings


def transcribe_audio(request: AsrTranscribeRequest, settings: Settings) -> dict:
    audio_path = resolve_media_path(request.audio_uri, settings)
    metadata = probe_audio(audio_path)
    if settings.asr_base_url:
        try:
            return transcribe_with_remote_asr(request, audio_path, metadata, settings)
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            return simulated_transcript(request, audio_path, metadata, settings, fallback_error=str(exc))
    return simulated_transcript(request, audio_path, metadata, settings)


def probe_audio(audio_path: object) -> dict:
    command = [
        "ffprobe",
        "-v",
        "error",
        "-show_entries",
        "format=duration:stream=codec_type,codec_name,channels,sample_rate",
        "-of",
        "json",
        str(audio_path),
    ]
    try:
        completed = subprocess.run(command, check=True, capture_output=True, text=True, timeout=10)
        payload = json.loads(completed.stdout or "{}")
    except (subprocess.SubprocessError, json.JSONDecodeError, FileNotFoundError):
        return {"duration_seconds": None, "streams": []}
    duration = payload.get("format", {}).get("duration")
    return {
        "duration_seconds": round(float(duration), 3) if duration else None,
        "streams": payload.get("streams", []),
    }


def transcribe_with_remote_asr(
    request: AsrTranscribeRequest,
    audio_path: object,
    metadata: dict,
    settings: Settings,
) -> dict:
    url = f"{settings.asr_base_url.rstrip('/')}/audio/transcriptions"
    with open(audio_path, "rb") as file:
        response = httpx.post(
            url,
            data={"model": settings.asr_model, "language": request.language},
            files={"file": (str(audio_path), file, "application/octet-stream")},
            timeout=settings.llm_timeout_seconds,
        )
    response.raise_for_status()
    data = response.json()
    transcript = (data.get("text") or data.get("transcript") or "").strip()
    if not transcript:
        raise ValueError("ASR service returned an empty transcript")
    return {
        "mission_id": request.mission_id,
        "audio_uri": request.audio_uri,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "language": request.language,
        "duration_seconds": metadata["duration_seconds"],
        "transcript": transcript,
        "segments": data.get("segments", []),
        "backend": "remote-asr",
        "model": settings.asr_model,
        "requires_human_confirmation": True,
    }


def simulated_transcript(
    request: AsrTranscribeRequest,
    audio_path: object,
    metadata: dict,
    settings: Settings,
    fallback_error: str | None = None,
) -> dict:
    note = f"人工补充：{request.operator_note}。" if request.operator_note else "无人工补充。"
    transcript = (
        f"音频转写占位稿：文件 {getattr(audio_path, 'name', request.audio_uri)} 已接入边缘小脑，"
        f"时长 {metadata['duration_seconds'] if metadata['duration_seconds'] is not None else '未知'} 秒。"
        f"{note} 当前未配置真实 ASR 服务，正式转写需接入本地 Whisper、Paraformer 或警务 ASR 模型。"
    )
    result = {
        "mission_id": request.mission_id,
        "audio_uri": request.audio_uri,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "language": request.language,
        "duration_seconds": metadata["duration_seconds"],
        "transcript": transcript,
        "segments": [],
        "backend": "simulated-fallback",
        "model": settings.asr_model,
        "requires_human_confirmation": True,
    }
    if fallback_error:
        result["fallback_error"] = fallback_error
    return result

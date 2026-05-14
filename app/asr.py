import json
import shutil
import subprocess
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import httpx

from app.media import resolve_media_path
from app.models import AsrTranscribeRequest
from app.settings import Settings


def transcribe_audio(request: AsrTranscribeRequest, settings: Settings) -> dict:
    audio_path = resolve_media_path(request.audio_uri, settings)
    metadata = probe_audio(audio_path)
    if local_asr_configured(settings):
        try:
            return transcribe_with_local_asr(request, audio_path, metadata, settings)
        except (subprocess.SubprocessError, FileNotFoundError, ValueError, RuntimeError, json.JSONDecodeError) as exc:
            local_error = str(exc)
        else:
            local_error = None
    else:
        local_error = None
    if settings.asr_base_url:
        try:
            return transcribe_with_remote_asr(request, audio_path, metadata, settings)
        except (httpx.HTTPError, KeyError, IndexError, ValueError) as exc:
            errors = [item for item in [local_error, str(exc)] if item]
            return simulated_transcript(request, audio_path, metadata, settings, fallback_error="; ".join(errors) or None)
    return simulated_transcript(request, audio_path, metadata, settings, fallback_error=local_error)


def local_asr_configured(settings: Settings) -> bool:
    if settings.asr_backend not in {"auto", "sherpa-onnx-sensevoice", "sherpa_onnx_sensevoice", "sensevoice"}:
        return False
    return bool(
        resolve_asr_binary(settings)
        and settings.asr_model_path
        and settings.asr_model_path.exists()
        and settings.asr_tokens_path
        and settings.asr_tokens_path.exists()
    )


def resolve_asr_binary(settings: Settings) -> str | None:
    return resolve_binary(settings.asr_local_binary)


def resolve_binary(binary_name: str | None) -> str | None:
    if not binary_name:
        return None
    binary = Path(binary_name)
    if binary.is_absolute() or len(binary.parts) > 1:
        return str(binary) if binary.exists() else None
    return shutil.which(binary_name)


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


def transcribe_with_local_asr(
    request: AsrTranscribeRequest,
    audio_path: Path,
    metadata: dict,
    settings: Settings,
) -> dict:
    if settings.asr_backend in {"sherpa-onnx-sensevoice", "sherpa_onnx_sensevoice", "sensevoice"}:
        return transcribe_with_sherpa_onnx_sensevoice(request, audio_path, metadata, settings)
    return transcribe_with_sherpa_onnx_sensevoice(request, audio_path, metadata, settings)


def transcribe_with_sherpa_onnx_sensevoice(
    request: AsrTranscribeRequest,
    audio_path: Path,
    metadata: dict,
    settings: Settings,
) -> dict:
    binary = resolve_asr_binary(settings)
    if not binary:
        raise FileNotFoundError("sherpa-onnx binary is not configured or not found")
    if not settings.asr_model_path or not settings.asr_model_path.exists():
        raise FileNotFoundError("SenseVoice model path is not configured or not found")
    if not settings.asr_tokens_path or not settings.asr_tokens_path.exists():
        raise FileNotFoundError("SenseVoice tokens path is not configured or not found")

    with tempfile.TemporaryDirectory(prefix="cerebellum-asr-") as work_dir:
        wav_path = Path(work_dir) / "audio.wav"
        convert_to_wav(audio_path, wav_path, settings)
        command = [
            binary,
            f"--tokens={settings.asr_tokens_path}",
            f"--sense-voice-model={settings.asr_model_path}",
            f"--sense-voice-language={language_for_sensevoice(request.language)}",
            "--sense-voice-use-itn=1",
            f"--num-threads={max(settings.asr_threads, 1)}",
            str(wav_path),
        ]
        completed = subprocess.run(
            command,
            check=True,
            capture_output=True,
            text=True,
            timeout=settings.asr_timeout_seconds,
        )
        transcript, segments = parse_sherpa_output(completed.stdout)
        if not transcript:
            transcript, segments = parse_sherpa_output(completed.stderr)
        if not transcript:
            raise ValueError("sherpa-onnx SenseVoice produced an empty transcript")
    return {
        "mission_id": request.mission_id,
        "audio_uri": request.audio_uri,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "language": request.language,
        "duration_seconds": metadata["duration_seconds"],
        "transcript": transcript,
        "segments": segments,
        "backend": "sherpa-onnx-sensevoice",
        "model": settings.asr_model_path.parent.name,
        "requires_human_confirmation": True,
    }


def convert_to_wav(audio_path: Path, wav_path: Path, settings: Settings) -> None:
    command = [
        "ffmpeg",
        "-y",
        "-i",
        str(audio_path),
        "-vn",
        "-ac",
        "1",
        "-ar",
        "16000",
        str(wav_path),
    ]
    subprocess.run(command, check=True, capture_output=True, text=True, timeout=settings.asr_timeout_seconds)


def parse_sherpa_output(stdout: str) -> tuple[str, list[dict]]:
    for raw_line in reversed(stdout.splitlines()):
        line = raw_line.strip()
        if not line.startswith("{"):
            continue
        try:
            payload = json.loads(line)
        except json.JSONDecodeError:
            continue
        text = str(payload.get("text", "")).strip()
        if text:
            return text, [
                {
                    "index": 0,
                    "start": first_or_none(payload.get("timestamps")),
                    "end": last_or_none(payload.get("timestamps")),
                    "text": text,
                    "language": payload.get("lang"),
                    "emotion": payload.get("emotion"),
                    "event": payload.get("event"),
                }
            ]
    transcript = clean_sherpa_stdout(stdout)
    segments = [{"index": 0, "start": None, "end": None, "text": transcript}] if transcript else []
    return transcript, segments


def clean_sherpa_stdout(stdout: str) -> str:
    candidates = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        if line.startswith(
            (
                "./",
                "/",
                "{",
                "----",
                "Started",
                "Done!",
                "Elapsed",
                "RTF",
                "Creating",
                "OfflineRecognizerConfig",
                "recognizer created",
                "num threads",
                "decoding method",
                "onnxruntime",
                "/k2-fsa/",
            )
        ):
            continue
        if ":" in line:
            _, possible_text = line.split(":", 1)
            if possible_text.strip():
                line = possible_text.strip()
        candidates.append(line)
    return "\n".join(candidates).strip()


def first_or_none(values: object) -> object:
    return values[0] if isinstance(values, list) and values else None


def last_or_none(values: object) -> object:
    return values[-1] if isinstance(values, list) and values else None


def language_for_sensevoice(language: str) -> str:
    normalized = language.lower().replace("_", "-")
    if normalized.startswith("zh") or normalized in {"cmn", "yue"}:
        return "zh"
    if normalized.startswith("en"):
        return "en"
    if normalized.startswith("ja"):
        return "ja"
    if normalized.startswith("ko"):
        return "ko"
    return "auto"


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
        f"{note} 当前未配置可用真实 ASR 模型或服务，正式转写需配置 SenseVoice、Paraformer 或警务 ASR 模型。"
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

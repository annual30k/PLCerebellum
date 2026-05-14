from datetime import datetime
from typing import Literal

from pydantic import BaseModel, Field


class MediaIngestRequest(BaseModel):
    source: str = Field(..., examples=["bodycam-rtsp-01", "phone-upload"])
    media_type: Literal["image", "video", "audio", "stream"]
    duration_seconds: int | None = Field(default=None, ge=0)
    note: str | None = None


class PlateAnalyzeRequest(BaseModel):
    frame_id: str = Field(..., examples=["frame-20260514-001"])
    camera_id: str = Field(default="bodycam-01")
    image_uri: str | None = None


class FaceAnalyzeRequest(BaseModel):
    frame_id: str
    camera_id: str = "bodycam-01"
    candidate_library: str = "local-authorized-watchlist"
    image_uri: str | None = None


class ObjectDetectRequest(BaseModel):
    frame_id: str
    camera_id: str = "bodycam-01"
    image_uri: str | None = None
    confidence_threshold: float = Field(default=0.35, ge=0.05, le=0.95)
    target_classes: list[str] | None = Field(default=None, examples=[["person", "car", "motorcycle"]])


class AsrTranscribeRequest(BaseModel):
    audio_uri: str = Field(..., examples=["samples/patrol-audio.wav"])
    mission_id: str | None = Field(default=None, examples=["mission-20260514-001"])
    language: str = "zh"
    operator_note: str | None = None
    max_tokens: int = Field(default=1000, ge=100, le=4096)


class FaceEnrollRequest(BaseModel):
    person_id: str = Field(..., examples=["police-0001"])
    image_uri: str = Field(..., examples=["/var/lib/cerebellum/samples/person.jpg"])
    display_name: str | None = None


class StreamCreateRequest(BaseModel):
    stream_id: str | None = Field(default=None, examples=["bodycam-01-main"])
    source_uri: str = Field(..., examples=["rtsp://192.168.50.10/live/main", "samples/patrol.mp4"])
    camera_id: str = "bodycam-01"
    sample_fps: float = Field(default=1.0, ge=0.1, le=5.0)
    analyze_plate: bool = True
    analyze_face: bool = True
    analyze_object: bool = False
    max_runtime_seconds: int | None = Field(default=None, ge=1, le=86_400)
    max_analyzed_frames: int | None = Field(default=None, ge=1, le=100_000)
    save_sampled_frames: bool = True


class VideoSummaryRequest(BaseModel):
    mission_id: str = Field(..., examples=["mission-20260514-001"])
    stream_id: str | None = Field(default=None, examples=["bodycam-01-main"])
    operator_note: str | None = None
    event_limit: int = Field(default=100, ge=1, le=500)
    use_llm: bool = False
    max_tokens: int = Field(default=800, ge=100, le=2048)


class ReportRequest(BaseModel):
    mission_id: str = Field(..., examples=["mission-20260514-001"])
    report_type: Literal["daily", "video_summary", "handover", "incident"] = "daily"
    prefer_quality: bool = False
    operator_note: str | None = None
    max_tokens: int = Field(default=1200, ge=100, le=4096)


class EvidenceRegisterRequest(BaseModel):
    file_uri: str = Field(..., examples=["samples/patrol-test.mp4"])
    evidence_type: Literal["video", "audio", "image", "document", "other"] = "video"
    mission_id: str | None = Field(default=None, examples=["mission-20260514-001"])
    encrypt: bool | None = None
    note: str | None = None


class SyncTaskRequest(BaseModel):
    mission_id: str | None = Field(default=None, examples=["mission-20260514-001"])
    destination_url: str | None = None
    include_events: bool = True
    include_audit: bool = False
    event_limit: int = Field(default=100, ge=1, le=1000)


class AuditRecord(BaseModel):
    timestamp: datetime
    action: str
    actor: str = "system"
    detail: dict

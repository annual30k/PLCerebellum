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


class ReportRequest(BaseModel):
    mission_id: str = Field(..., examples=["mission-20260514-001"])
    report_type: Literal["daily", "video_summary", "handover", "incident"] = "daily"
    prefer_quality: bool = False
    operator_note: str | None = None
    max_tokens: int = Field(default=1200, ge=100, le=4096)


class AuditRecord(BaseModel):
    timestamp: datetime
    action: str
    actor: str = "system"
    detail: dict


from functools import lru_cache
from pathlib import Path

from pydantic import field_validator
import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_prefix="CEREBELLUM_", extra="ignore")

    device_id: str = "PL-CB-SIM-0001"
    profile: str = "production-sim"
    accelerator: str = "jetson-orin-nx-16gb"
    llm_model: str = "Qwen3.5-4B-INT4"
    llm_fallback_model: str = "Qwen3.5-2B-INT4"
    llm_batch_model: str = "Qwen3.5-9B-INT4"
    llm_base_url: str | None = None
    llm_timeout_seconds: float = 180.0
    context_tokens: int = 16_384
    max_context_tokens: int = 32_768
    storage_gb: int = 1024
    battery_wh: int = 60
    power_mode: str = "standard"
    secure_boot: bool = True
    readonly_rootfs: bool = True
    config: Path = Path("/etc/cerebellum/device.yaml")
    data_dir: Path = Path("/var/lib/cerebellum")
    log_dir: Path = Path("/var/log/cerebellum")
    model_dir: Path = Path("/opt/cerebellum/models")
    sample_dir: Path = Path("/var/lib/cerebellum/samples")
    stream_frame_dir: Path = Path("/var/lib/cerebellum/stream_frames")
    stream_max_sources: int = 2
    stream_retained_frames_per_source: int = 120
    api_key: str | None = None
    asr_model: str = "edge-asr-sim"
    asr_backend: str = "auto"
    asr_base_url: str | None = None
    asr_local_binary: str | None = None
    asr_model_path: Path | None = None
    asr_tokens_path: Path | None = None
    asr_timeout_seconds: float = 180.0
    asr_threads: int = 4
    object_model: str = "yolov8n.pt"
    object_model_path: Path | None = None
    evidence_encrypt_by_default: bool = True
    evidence_key: str | None = None
    sync_destination_url: str | None = None
    cert_file: Path | None = None
    key_file: Path | None = None
    ca_file: Path | None = None
    mtls_required: bool = False

    @field_validator(
        "asr_base_url",
        "asr_local_binary",
        "asr_model_path",
        "asr_tokens_path",
        "object_model_path",
        "evidence_key",
        "sync_destination_url",
        "cert_file",
        "key_file",
        "ca_file",
        mode="before",
    )
    @classmethod
    def blank_to_none(cls, value: object) -> object:
        if value == "":
            return None
        return value


@lru_cache
def get_settings() -> Settings:
    return Settings()


@lru_cache
def load_device_config() -> dict:
    settings = get_settings()
    if not settings.config.exists():
        return {}
    with settings.config.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file) or {}

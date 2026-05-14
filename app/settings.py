from functools import lru_cache
from pathlib import Path

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

import json
import os
import shutil
from datetime import datetime, timezone
from hashlib import sha256
from pathlib import Path
from uuid import uuid4

from app.media import resolve_media_path
from app.models import EvidenceRegisterRequest
from app.settings import Settings


def register_evidence(request: EvidenceRegisterRequest, settings: Settings) -> dict:
    source_path = resolve_media_path(request.file_uri, settings)
    evidence_dir = settings.data_dir / "evidence"
    evidence_dir.mkdir(parents=True, exist_ok=True)
    evidence_id = f"evd-{uuid4().hex[:12]}"
    encrypt = settings.evidence_encrypt_by_default if request.encrypt is None else request.encrypt
    source_hash = hash_file(source_path)

    if encrypt:
        stored_path = evidence_dir / f"{evidence_id}{source_path.suffix}.enc"
        encrypt_file(source_path, stored_path, settings)
        storage_mode = "encrypted"
        encryption = "fernet"
    else:
        stored_path = evidence_dir / f"{evidence_id}{source_path.suffix}"
        shutil.copy2(source_path, stored_path)
        storage_mode = "plain-copy"
        encryption = None

    manifest = {
        "evidence_id": evidence_id,
        "mission_id": request.mission_id,
        "evidence_type": request.evidence_type,
        "source_uri": request.file_uri,
        "source_name": source_path.name,
        "stored_path": str(stored_path),
        "source_sha256": source_hash,
        "stored_sha256": hash_file(stored_path),
        "size_bytes": source_path.stat().st_size,
        "registered_at": datetime.now(timezone.utc).isoformat(),
        "storage_mode": storage_mode,
        "encryption": encryption,
        "note": request.note,
        "chain_status": "registered",
    }
    manifest_path = evidence_dir / f"{evidence_id}.json"
    with manifest_path.open("w", encoding="utf-8") as file:
        json.dump(manifest, file, ensure_ascii=False, indent=2)
    manifest["manifest_path"] = str(manifest_path)
    return manifest


def list_evidence(settings: Settings) -> list[dict]:
    evidence_dir = settings.data_dir / "evidence"
    if not evidence_dir.exists():
        return []
    items = []
    for path in sorted(evidence_dir.glob("evd-*.json")):
        with path.open("r", encoding="utf-8") as file:
            record = json.load(file)
        record["manifest_path"] = str(path)
        items.append(record)
    return items


def uploaded_files_dir(settings: Settings) -> Path:
    path = settings.data_dir / "uploads"
    path.mkdir(parents=True, exist_ok=True)
    return path


def safe_upload_name(file_name: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in {".", "_", "-"} else "_" for ch in file_name).strip("._")
    return cleaned[:160] or "upload.bin"


def uploaded_file_record(path: Path, settings: Settings) -> dict:
    stat = path.stat()
    return {
        "file_id": path.stem,
        "file_name": path.name,
        "file_uri": str(path),
        "size_bytes": stat.st_size,
        "sha256": hash_file(path),
        "uploaded_at": datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat(),
        "download_url": f"/api/v1/files/{path.name}/download",
    }


def list_uploaded_files(settings: Settings) -> list[dict]:
    upload_dir = uploaded_files_dir(settings)
    return [uploaded_file_record(path, settings) for path in sorted(upload_dir.iterdir()) if path.is_file()]


def resolve_uploaded_file(file_name: str, settings: Settings) -> Path:
    upload_dir = uploaded_files_dir(settings).resolve()
    path = (upload_dir / file_name).resolve()
    if not (path == upload_dir or upload_dir in path.parents):
        raise ValueError(f"file path is outside upload dir: {file_name}")
    if not path.exists() or not path.is_file():
        raise FileNotFoundError(file_name)
    return path


def hash_file(path: Path) -> str:
    digest = sha256()
    with path.open("rb") as file:
        for chunk in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def encrypt_file(source_path: Path, stored_path: Path, settings: Settings) -> None:
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise RuntimeError("cryptography is required for evidence encryption") from exc
    fernet = Fernet(load_or_create_key(settings))
    with source_path.open("rb") as source, stored_path.open("wb") as target:
        target.write(fernet.encrypt(source.read()))


def load_or_create_key(settings: Settings) -> bytes:
    if settings.evidence_key:
        return settings.evidence_key.encode()
    try:
        from cryptography.fernet import Fernet
    except ImportError as exc:
        raise RuntimeError("cryptography is required for evidence encryption") from exc
    key_dir = settings.data_dir / "secrets"
    key_dir.mkdir(parents=True, exist_ok=True)
    key_path = key_dir / "evidence.key"
    if key_path.exists():
        return key_path.read_bytes().strip()
    key = Fernet.generate_key()
    key_path.write_bytes(key)
    os.chmod(key_path, 0o600)
    return key

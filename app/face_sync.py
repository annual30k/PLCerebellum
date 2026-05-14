import base64
import json
import os
import hashlib
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse

import httpx
import numpy as np

from app.models import FaceLibraryApplyRequest, FaceLibrarySyncRequest
from app.settings import Settings
from app.vision import enroll_face, face_library_path, load_face_library


def face_library_meta_path(settings: Settings) -> Path:
    return settings.data_dir / "face_library_meta.json"


def pending_face_library_path(settings: Settings) -> Path:
    return settings.data_dir / "face_library_pending.json"


def face_library_status(settings: Settings) -> dict:
    library = load_face_library(settings)
    metadata = read_json(face_library_meta_path(settings), {})
    pending = read_json(pending_face_library_path(settings), [])
    return {
        "version": metadata.get("version"),
        "source": metadata.get("source"),
        "model": metadata.get("model", "opencv-zoo-yunet+sface"),
        "applied_at": metadata.get("applied_at"),
        "person_count": len(library),
        "pending_count": len(pending),
        "last_result": metadata.get("last_result"),
    }


def apply_face_library_bundle(request: FaceLibraryApplyRequest, settings: Settings) -> dict:
    current_library = {} if request.full_snapshot else load_face_library(settings)
    pending = []
    applied = 0
    skipped = 0
    failed = 0

    for person in request.persons:
        status = str(person.get("status") or "ENABLED").upper()
        person_id = str(person.get("person_id") or person.get("control_id") or "").strip()
        if not person_id:
            failed += 1
            pending.append({"reason": "missing_person_id", "person": scrub_person(person)})
            continue
        if status not in {"ENABLED", "ACTIVE"}:
            current_library.pop(person_id, None)
            skipped += 1
            continue

        embedding = normalize_embedding(person.get("embedding"))
        if embedding:
            current_library[person_id] = build_library_record(person_id, person, embedding, request)
            applied += 1
            continue

        try:
            image_uri = materialize_base64_image(person, settings) or materialize_remote_image(person, settings)
        except Exception as exc:
            failed += 1
            pending.append({"person_id": person_id, "reason": f"image_fetch_failed: {exc}", "person": scrub_person(person)})
            continue
        if image_uri or person.get("image_uri"):
            try:
                result = enroll_face(
                    person_id=person_id,
                    image_uri=image_uri or str(person["image_uri"]),
                    display_name=str(person.get("display_name") or person.get("name") or person_id),
                    settings=settings,
                )
                record = load_face_library(settings).get(person_id, current_library.get(person_id, {}))
                record.update(
                    {
                        "risk_level": person.get("risk_level"),
                        "category": person.get("category"),
                        "source": request.source,
                        "source_version": request.version,
                        "image_sha256": person.get("image_sha256"),
                    }
                )
                current_library[person_id] = record
                result["source_version"] = request.version
                applied += 1
                continue
            except Exception as exc:
                failed += 1
                pending.append({"person_id": person_id, "reason": f"feature_extract_failed: {exc}", "person": scrub_person(person)})
                continue

        skipped += 1
        pending.append({"person_id": person_id, "reason": "no_embedding_or_local_image", "person": scrub_person(person)})

    save_face_library_atomic(settings, current_library)
    write_json_atomic(pending_face_library_path(settings), pending)
    result = {
        "version": request.version,
        "source": request.source,
        "model": request.model,
        "full_snapshot": request.full_snapshot,
        "received": len(request.persons),
        "applied": applied,
        "skipped": skipped,
        "failed": failed,
        "pending": len(pending),
    }
    metadata = {
        "version": request.version,
        "source": request.source,
        "model": request.model,
        "applied_at": datetime.now(timezone.utc).isoformat(),
        "last_result": result,
    }
    write_json_atomic(face_library_meta_path(settings), metadata)
    result["person_count"] = len(current_library)
    return result


def sync_face_library_from_backend(request: FaceLibrarySyncRequest, settings: Settings) -> dict:
    backend_url = request.backend_url or settings.backend_base_url
    if not backend_url:
        raise ValueError("backend_url is required")
    token = request.token or settings.backend_token
    current_version = request.current_version or face_library_status(settings).get("version")
    url = f"{backend_url.rstrip('/')}/api/v1/cerebellum/face-library"
    params = {
        "deviceId": request.device_id or settings.device_id,
        "currentVersion": current_version or "",
        "force": str(request.force).lower(),
    }
    headers = {"Accept": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    response = httpx.get(url, params=params, headers=headers, timeout=60.0)
    response.raise_for_status()
    payload = response.json()
    bundle = payload.get("data", payload)
    if not isinstance(bundle, dict):
        raise ValueError(f"invalid backend face library response: {payload}")
    if bundle.get("unchanged") and not request.force:
        return {
            "unchanged": True,
            "version": bundle.get("version", current_version),
            "source": "PLBackend",
            "person_count": face_library_status(settings).get("person_count", 0),
        }
    apply_request = FaceLibraryApplyRequest(
        version=str(bundle["version"]),
        source=str(bundle.get("source", "PLBackend")),
        full_snapshot=bool(bundle.get("fullSnapshot", bundle.get("full_snapshot", True))),
        model=str(bundle.get("model", "opencv-zoo-yunet+sface")),
        persons=list(bundle.get("persons", [])),
    )
    result = apply_face_library_bundle(apply_request, settings)
    ack_backend(backend_url, token, request.device_id or settings.device_id, result)
    return result


def build_library_record(person_id: str, person: dict, embedding: list[float], request: FaceLibraryApplyRequest) -> dict:
    return {
        "person_id": person_id,
        "display_name": person.get("display_name") or person.get("name") or person_id,
        "risk_level": person.get("risk_level"),
        "category": person.get("category"),
        "embedding": embedding,
        "model": request.model,
        "source": request.source,
        "source_version": request.version,
        "image_sha256": person.get("image_sha256"),
        "threshold_note": "Synced from PLBackend; final identity requires human confirmation",
    }


def normalize_embedding(value: Any) -> list[float] | None:
    if not isinstance(value, list) or not value:
        return None
    vector = np.asarray(value, dtype=float)
    if vector.ndim != 1 or vector.size == 0:
        return None
    vector = vector / max(float(np.linalg.norm(vector)), 1e-12)
    return [float(item) for item in vector.tolist()]


def materialize_base64_image(person: dict, settings: Settings) -> str | None:
    image_base64 = person.get("image_base64")
    if not image_base64:
        return None
    suffix = str(person.get("image_suffix") or ".jpg")
    if not suffix.startswith("."):
        suffix = "." + suffix
    person_id = str(person.get("person_id") or person.get("control_id") or "unknown").strip()
    image_dir = settings.data_dir / "face_library_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{person_id}{suffix}"
    image_path.write_bytes(base64.b64decode(image_base64))
    return str(image_path)


def materialize_remote_image(person: dict, settings: Settings) -> str | None:
    image_url = person.get("image_url")
    if not image_url:
        return None
    resolved_url = resolve_image_url(str(image_url), settings)
    parsed = urlparse(resolved_url)
    if parsed.scheme not in {"http", "https"}:
        raise ValueError("unsupported_image_url_scheme")

    headers = {"Accept": "image/*,*/*"}
    if settings.backend_token:
        headers["Authorization"] = f"Bearer {settings.backend_token}"
    response = httpx.get(resolved_url, headers=headers, timeout=30.0, follow_redirects=True)
    response.raise_for_status()
    image_bytes = response.content
    expected_sha256 = person.get("image_sha256")
    if expected_sha256:
        actual_sha256 = hashlib.sha256(image_bytes).hexdigest()
        if actual_sha256.lower() != str(expected_sha256).lower():
            raise ValueError("image_sha256_mismatch")

    person_id = str(person.get("person_id") or person.get("control_id") or "unknown").strip()
    suffix = image_suffix_from_url(parsed.path)
    image_dir = settings.data_dir / "face_library_images"
    image_dir.mkdir(parents=True, exist_ok=True)
    image_path = image_dir / f"{person_id}{suffix}"
    image_path.write_bytes(image_bytes)
    return str(image_path)


def resolve_image_url(image_url: str, settings: Settings) -> str:
    if image_url.startswith(("http://", "https://")):
        return image_url
    if image_url.startswith("/") and settings.backend_base_url:
        return urljoin(settings.backend_base_url.rstrip("/") + "/", image_url.lstrip("/"))
    return image_url


def image_suffix_from_url(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jpg", ".jpeg", ".png", ".webp", ".bmp"}:
        return suffix
    return ".jpg"


def scrub_person(person: dict) -> dict:
    return {key: value for key, value in person.items() if key not in {"embedding", "image_base64"}}


def save_face_library_atomic(settings: Settings, library: dict[str, dict]) -> None:
    path = face_library_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(".json.tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(library, file, ensure_ascii=False)
    os.replace(tmp_path, path)


def write_json_atomic(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_suffix(path.suffix + ".tmp")
    with tmp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
    os.replace(tmp_path, path)


def read_json(path: Path, default: Any) -> Any:
    if not path.exists():
        return default
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def ack_backend(backend_url: str, token: str | None, device_id: str, result: dict) -> None:
    url = f"{backend_url.rstrip('/')}/api/v1/cerebellum/face-library/ack"
    headers = {"Content-Type": "application/json"}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    try:
        httpx.post(url, headers=headers, json={"deviceId": device_id, **result}, timeout=15.0).raise_for_status()
    except httpx.HTTPError:
        # ACK 失败不应回滚本地人脸库，下一轮同步仍会携带当前版本。
        return

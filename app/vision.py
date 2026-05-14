import json
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from app.settings import Settings


class VisionUnavailable(RuntimeError):
    pass


def resolve_image_path(image_uri: str, settings: Settings) -> Path:
    path = Path(image_uri)
    if not path.is_absolute():
        path = settings.sample_dir / image_uri
    resolved = path.resolve()
    allowed_roots = [
        settings.sample_dir.resolve(),
        settings.data_dir.resolve(),
        settings.model_dir.resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(f"image path is outside allowed roots: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return resolved


def read_image(image_uri: str, settings: Settings) -> Any:
    try:
        import cv2
    except ImportError as exc:
        raise VisionUnavailable("opencv is not installed") from exc
    image_path = resolve_image_path(image_uri, settings)
    image = cv2.imread(str(image_path))
    if image is None:
        raise ValueError(f"failed to read image: {image_path}")
    return image


@lru_cache
def get_plate_catcher() -> Any:
    try:
        from hyperlpr3 import LicensePlateCatcher
    except ImportError as exc:
        raise VisionUnavailable("hyperlpr3 is not installed") from exc
    return LicensePlateCatcher()


def normalize_plate_result(raw: Any) -> dict:
    if isinstance(raw, dict):
        return raw
    if isinstance(raw, (list, tuple)):
        plate = raw[0] if len(raw) > 0 else ""
        confidence = raw[1] if len(raw) > 1 else None
        plate_type = raw[2] if len(raw) > 2 else None
        box = raw[3] if len(raw) > 3 else None
        return {
            "plate_number": plate,
            "confidence": float(confidence) if confidence is not None else None,
            "plate_type": plate_type,
            "box": np.asarray(box).tolist() if box is not None else None,
        }
    return {"raw": str(raw)}


def recognize_plate(image_uri: str, settings: Settings) -> list[dict]:
    image = read_image(image_uri, settings)
    catcher = get_plate_catcher()
    raw_results = catcher(image)
    return [normalize_plate_result(item) for item in raw_results]


@lru_cache
def get_face_models(model_dir: str) -> tuple[Any, Any]:
    try:
        import cv2
    except ImportError as exc:
        raise VisionUnavailable("opencv is not installed") from exc
    base = Path(model_dir) / "opencv_zoo"
    detector_path = base / "face_detection_yunet_2023mar.onnx"
    recognizer_path = base / "face_recognition_sface_2021dec.onnx"
    if not detector_path.exists() or not recognizer_path.exists():
        raise VisionUnavailable("OpenCV Zoo face models are missing")
    detector = cv2.FaceDetectorYN.create(str(detector_path), "", (320, 320), 0.6, 0.3, 5000)
    recognizer = cv2.FaceRecognizerSF.create(str(recognizer_path), "")
    return detector, recognizer


def detect_faces(image_uri: str, settings: Settings) -> list[dict]:
    import cv2

    image = read_image(image_uri, settings)
    detector, recognizer = get_face_models(str(settings.model_dir))
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None:
        return []

    library = load_face_library(settings)
    results = []
    for face in faces:
        aligned = recognizer.alignCrop(image, face)
        embedding = recognizer.feature(aligned).flatten().astype(float)
        embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-12)
        candidate = match_face(embedding, library)
        results.append(
            {
                "box": [float(v) for v in face[:4]],
                "landmarks": [float(v) for v in face[4:14]],
                "quality_score": float(face[14]) if len(face) > 14 else None,
                "embedding_dim": int(embedding.shape[0]),
                "embedding_preview": [round(float(v), 6) for v in embedding[:8]],
                "candidate": candidate,
                "model": "opencv-zoo-yunet+sface",
                "result_type": "candidate_hint_only",
            }
        )
    return results


def enroll_face(person_id: str, image_uri: str, display_name: str | None, settings: Settings) -> dict:
    import cv2

    image = read_image(image_uri, settings)
    detector, recognizer = get_face_models(str(settings.model_dir))
    height, width = image.shape[:2]
    detector.setInputSize((width, height))
    _, faces = detector.detect(image)
    if faces is None or len(faces) == 0:
        raise ValueError("no face detected for enrollment")
    face = max(faces, key=lambda item: float(item[14]) if len(item) > 14 else 0.0)
    aligned = recognizer.alignCrop(image, face)
    embedding = recognizer.feature(aligned).flatten().astype(float)
    embedding = embedding / max(float(np.linalg.norm(embedding)), 1e-12)

    library = load_face_library(settings)
    library[person_id] = {
        "person_id": person_id,
        "display_name": display_name,
        "embedding": [float(v) for v in embedding],
        "model": "opencv-zoo-yunet+sface",
    }
    save_face_library(settings, library)
    return {
        "person_id": person_id,
        "display_name": display_name,
        "embedding_dim": int(embedding.shape[0]),
        "model": "opencv-zoo-yunet+sface",
    }


def face_library_path(settings: Settings) -> Path:
    return settings.data_dir / "face_library.json"


def load_face_library(settings: Settings) -> dict[str, dict]:
    path = face_library_path(settings)
    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def save_face_library(settings: Settings, library: dict[str, dict]) -> None:
    path = face_library_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as file:
        json.dump(library, file, ensure_ascii=False)


def match_face(embedding: np.ndarray, library: dict[str, dict]) -> dict | None:
    best = None
    for person_id, record in library.items():
        known = np.asarray(record.get("embedding", []), dtype=float)
        if known.shape != embedding.shape:
            continue
        score = float(np.dot(embedding, known))
        if best is None or score > best["similarity"]:
            best = {
                "person_id": person_id,
                "display_name": record.get("display_name"),
                "risk_level": record.get("risk_level"),
                "category": record.get("category"),
                "similarity": round(score, 4),
                "threshold_note": "SFace cosine similarity; final identity requires human confirmation",
            }
    if best and best["similarity"] >= 0.36:
        return best
    return None

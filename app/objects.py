from datetime import datetime, timezone
from hashlib import sha256

from app.models import ObjectDetectRequest
from app.settings import Settings
from app.vision import VisionUnavailable, read_image


def detect_objects(request: ObjectDetectRequest, settings: Settings) -> dict:
    if request.image_uri:
        try:
            detections = detect_objects_with_yolo(request, settings)
            return {
                "backend": "ultralytics-yolo",
                "model": settings.object_model,
                "frame_id": request.frame_id,
                "camera_id": request.camera_id,
                "generated_at": datetime.now(timezone.utc).isoformat(),
                "detections": detections,
                "detection_count": len(detections),
                "requires_human_confirmation": True,
            }
        except Exception as exc:
            return simulated_object_detection(request, settings, fallback_error=str(exc))
    return simulated_object_detection(request, settings)


def detect_objects_with_yolo(request: ObjectDetectRequest, settings: Settings) -> list[dict]:
    try:
        from ultralytics import YOLO
    except ImportError as exc:
        raise VisionUnavailable("ultralytics is not installed") from exc
    model_ref = str(settings.object_model_path) if settings.object_model_path else settings.object_model
    image = read_image(request.image_uri or "", settings)
    model = YOLO(model_ref)
    results = model.predict(image, conf=request.confidence_threshold, verbose=False)
    names = results[0].names if results else {}
    detections = []
    allowed = set(request.target_classes or [])
    for result in results:
        for box in result.boxes:
            class_id = int(box.cls[0])
            label = str(names.get(class_id, class_id))
            if allowed and label not in allowed:
                continue
            detections.append(
                {
                    "label": label,
                    "confidence": round(float(box.conf[0]), 4),
                    "box": [round(float(v), 2) for v in box.xyxy[0].tolist()],
                    "result_type": "object_candidate",
                }
            )
    return detections


def simulated_object_detection(
    request: ObjectDetectRequest,
    settings: Settings,
    fallback_error: str | None = None,
) -> dict:
    seed = sha256(f"{request.frame_id}:{request.camera_id}:{request.image_uri}".encode()).hexdigest()
    class_pool = request.target_classes or ["person", "vehicle", "motorcycle", "bag"]
    count = 1 + int(seed[0], 16) % min(3, len(class_pool))
    detections = []
    for index in range(count):
        label = class_pool[(int(seed[index + 1], 16) + index) % len(class_pool)]
        detections.append(
            {
                "label": label,
                "confidence": round(min(max(request.confidence_threshold, 0.5) + int(seed[index + 5], 16) / 100, 0.98), 3),
                "box": [
                    20 + index * 28,
                    30 + index * 22,
                    140 + index * 30,
                    210 + index * 24,
                ],
                "result_type": "object_candidate",
            }
        )
    result = {
        "backend": "simulated-fallback",
        "model": settings.object_model,
        "frame_id": request.frame_id,
        "camera_id": request.camera_id,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "detections": detections,
        "detection_count": len(detections),
        "requires_human_confirmation": True,
    }
    if fallback_error:
        result["fallback_error"] = fallback_error
    return result

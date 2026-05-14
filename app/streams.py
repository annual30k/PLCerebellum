from __future__ import annotations

import re
import threading
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from uuid import uuid4

from app.models import FaceAnalyzeRequest, PlateAnalyzeRequest, StreamCreateRequest
from app.services import analyze_face_image, analyze_plate_image
from app.settings import Settings
from app.state import DeviceState


STREAM_ID_PATTERN = re.compile(r"[^a-zA-Z0-9_.-]+")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sanitize_stream_id(value: str | None) -> str:
    if not value:
        return f"stream-{uuid4().hex[:10]}"
    stream_id = STREAM_ID_PATTERN.sub("-", value).strip("-._")
    return stream_id[:64] or f"stream-{uuid4().hex[:10]}"


def resolve_stream_source(source_uri: str, settings: Settings) -> str | int:
    if source_uri.isdigit():
        return int(source_uri)
    if "://" in source_uri:
        return source_uri
    path = Path(source_uri)
    if path.parts[:1] == ("samples",):
        path = Path(*path.parts[1:])
    if not path.is_absolute():
        path = settings.sample_dir / path
    resolved = path.resolve()
    allowed_roots = [
        settings.sample_dir.resolve(),
        settings.data_dir.resolve(),
    ]
    if not any(resolved == root or root in resolved.parents for root in allowed_roots):
        raise ValueError(f"stream source path is outside allowed roots: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    return str(resolved)


def ensure_stream_frame_dir_allowed(settings: Settings) -> None:
    frame_dir = settings.stream_frame_dir.resolve()
    data_dir = settings.data_dir.resolve()
    if not (frame_dir == data_dir or data_dir in frame_dir.parents):
        raise ValueError(f"stream frame dir must be under data dir: {frame_dir}")


@dataclass
class StreamSession:
    stream_id: str
    source_uri: str
    camera_id: str
    sample_fps: float
    analyze_plate: bool
    analyze_face: bool
    max_runtime_seconds: int | None
    max_analyzed_frames: int | None
    save_sampled_frames: bool
    settings: Settings
    state: DeviceState
    status: str = "starting"
    created_at: str = field(default_factory=utc_now)
    started_at: str | None = None
    stopped_at: str | None = None
    last_frame_at: str | None = None
    last_error: str | None = None
    frames_read: int = 0
    frames_analyzed: int = 0
    plate_candidates: int = 0
    face_candidates: int = 0

    def __post_init__(self) -> None:
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread = threading.Thread(target=self._run, name=f"stream-{self.stream_id}", daemon=True)

    def start(self) -> None:
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()

    def join(self, timeout: float = 5.0) -> None:
        self._thread.join(timeout=timeout)

    def is_alive(self) -> bool:
        return self._thread.is_alive()

    def snapshot(self) -> dict:
        with self._lock:
            return {
                "stream_id": self.stream_id,
                "source_uri": self.source_uri,
                "camera_id": self.camera_id,
                "sample_fps": self.sample_fps,
                "analyze_plate": self.analyze_plate,
                "analyze_face": self.analyze_face,
                "status": self.status,
                "created_at": self.created_at,
                "started_at": self.started_at,
                "stopped_at": self.stopped_at,
                "last_frame_at": self.last_frame_at,
                "last_error": self.last_error,
                "frames_read": self.frames_read,
                "frames_analyzed": self.frames_analyzed,
                "plate_candidates": self.plate_candidates,
                "face_candidates": self.face_candidates,
            }

    def _set_status(self, status: str, error: str | None = None) -> None:
        with self._lock:
            self.status = status
            if error:
                self.last_error = error
            if status in {"stopped", "completed", "error"}:
                self.stopped_at = utc_now()

    def _run(self) -> None:
        try:
            import cv2
        except ImportError as exc:
            self._set_status("error", f"opencv is not installed: {exc}")
            self.state.audit("stream.error", {"stream_id": self.stream_id, "error": self.last_error})
            return

        started_monotonic = time.monotonic()
        capture = None
        try:
            source = resolve_stream_source(self.source_uri, self.settings)
            capture = cv2.VideoCapture()
            try:
                capture.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, 5000)
                capture.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, 5000)
            except AttributeError:
                pass
            capture.open(source)
            if not capture.isOpened():
                raise RuntimeError(f"failed to open stream source: {self.source_uri}")

            with self._lock:
                self.status = "running"
                self.started_at = utc_now()
            self.state.audit("stream.start", self.snapshot())

            sample_interval = 1.0 / self.sample_fps
            source_fps = float(capture.get(cv2.CAP_PROP_FPS) or 0.0)
            frame_count = int(capture.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
            is_file_source = isinstance(source, str) and "://" not in source and frame_count > 0
            sample_stride = max(int(round(source_fps / self.sample_fps)), 1) if is_file_source and source_fps > 0 else 1
            next_sample_at = 0.0
            while not self._stop_event.is_set():
                if self.max_runtime_seconds and time.monotonic() - started_monotonic >= self.max_runtime_seconds:
                    self._set_status("completed")
                    break
                if self.max_analyzed_frames and self.frames_analyzed >= self.max_analyzed_frames:
                    self._set_status("completed")
                    break

                ok, frame = capture.read()
                if not ok:
                    self._set_status("completed")
                    break

                now = time.monotonic()
                with self._lock:
                    self.frames_read += 1
                    frames_read = self.frames_read
                if is_file_source:
                    should_sample = frames_read == 1 or (frames_read - 1) % sample_stride == 0
                else:
                    should_sample = now >= next_sample_at
                if not should_sample:
                    if not is_file_source and now < next_sample_at:
                        time.sleep(min(0.01, next_sample_at - now))
                    continue
                if not is_file_source:
                    next_sample_at = now + sample_interval
                elif now < next_sample_at:
                    time.sleep(min(0.01, next_sample_at - now))

                frame_id = f"{self.stream_id}-{frames_read:08d}"
                frame_path = self._write_frame(cv2, frame, frame_id)
                self._analyze_frame(frame_id, frame_path)
                if not self.save_sampled_frames:
                    try:
                        Path(frame_path).unlink()
                    except FileNotFoundError:
                        pass

            if self.status == "running":
                self._set_status("stopped")
        except Exception as exc:
            self._set_status("error", str(exc))
            self.state.audit("stream.error", {"stream_id": self.stream_id, "error": str(exc)})
        finally:
            if capture is not None:
                capture.release()
            self.state.audit("stream.stop", self.snapshot())

    def _write_frame(self, cv2: object, frame: object, frame_id: str) -> str:
        frame_dir = self.settings.stream_frame_dir / self.stream_id
        frame_dir.mkdir(parents=True, exist_ok=True)
        frame_path = frame_dir / f"{frame_id}.jpg"
        if self.save_sampled_frames:
            ok = cv2.imwrite(str(frame_path), frame)
            if not ok:
                raise RuntimeError(f"failed to write sampled frame: {frame_path}")
            self._prune_frames(frame_dir)
            return str(frame_path)

        temp_path = frame_dir / f"{frame_id}.jpg"
        ok = cv2.imwrite(str(temp_path), frame)
        if not ok:
            raise RuntimeError(f"failed to write sampled frame: {temp_path}")
        return str(temp_path)

    def _prune_frames(self, frame_dir: Path) -> None:
        retained = max(self.settings.stream_retained_frames_per_source, 1)
        frames = sorted(frame_dir.glob("*.jpg"), key=lambda item: item.stat().st_mtime, reverse=True)
        for old_frame in frames[retained:]:
            try:
                old_frame.unlink()
            except FileNotFoundError:
                pass

    def _analyze_frame(self, frame_id: str, frame_path: str) -> None:
        plate_count = 0
        face_count = 0
        if self.analyze_plate:
            plate_result = analyze_plate_image(
                PlateAnalyzeRequest(frame_id=frame_id, camera_id=self.camera_id, image_uri=frame_path),
                self.settings,
                self.state,
            )
            plate_count = int(plate_result.get("candidate_count", 0))
            self.state.add_event("stream_plate_candidate", {"stream_id": self.stream_id, **plate_result})
        if self.analyze_face:
            face_result = analyze_face_image(
                FaceAnalyzeRequest(frame_id=frame_id, camera_id=self.camera_id, image_uri=frame_path),
                self.settings,
                self.state,
            )
            face_count = int(face_result.get("face_count", 0))
            self.state.add_event("stream_face_candidate", {"stream_id": self.stream_id, **face_result})

        with self._lock:
            self.frames_analyzed += 1
            self.plate_candidates += plate_count
            self.face_candidates += face_count
            self.last_frame_at = utc_now()


class StreamManager:
    def __init__(self, settings: Settings, state: DeviceState) -> None:
        self.settings = settings
        self.state = state
        self._sessions: dict[str, StreamSession] = {}
        self._lock = threading.Lock()

    def create(self, request: StreamCreateRequest) -> dict:
        stream_id = sanitize_stream_id(request.stream_id)
        ensure_stream_frame_dir_allowed(self.settings)
        if "://" not in request.source_uri and not request.source_uri.isdigit():
            resolve_stream_source(request.source_uri, self.settings)
        with self._lock:
            if stream_id in self._sessions and self._sessions[stream_id].snapshot()["status"] in {"starting", "running"}:
                raise ValueError(f"stream is already running: {stream_id}")
            active_count = sum(1 for session in self._sessions.values() if session.snapshot()["status"] in {"starting", "running"})
            if active_count >= self.settings.stream_max_sources:
                raise RuntimeError(f"stream source limit reached: {self.settings.stream_max_sources}")
            session = StreamSession(
                stream_id=stream_id,
                source_uri=request.source_uri,
                camera_id=request.camera_id,
                sample_fps=request.sample_fps,
                analyze_plate=request.analyze_plate,
                analyze_face=request.analyze_face,
                max_runtime_seconds=request.max_runtime_seconds,
                max_analyzed_frames=request.max_analyzed_frames,
                save_sampled_frames=request.save_sampled_frames,
                settings=self.settings,
                state=self.state,
            )
            self._sessions[stream_id] = session
            session.start()
            return session.snapshot()

    def list(self) -> list[dict]:
        with self._lock:
            return [session.snapshot() for session in self._sessions.values()]

    def get(self, stream_id: str) -> dict:
        with self._lock:
            session = self._sessions.get(stream_id)
        if session is None:
            raise KeyError(stream_id)
        return session.snapshot()

    def stop(self, stream_id: str) -> dict:
        with self._lock:
            session = self._sessions.get(stream_id)
        if session is None:
            raise KeyError(stream_id)
        session.stop()
        session.join(timeout=5.0)
        if session.is_alive():
            session._set_status("stopping", "stop requested but capture loop has not exited yet")
        return session.snapshot()

    def stop_all(self) -> None:
        with self._lock:
            sessions = list(self._sessions.values())
        for session in sessions:
            session.stop()
        for session in sessions:
            session.join(timeout=5.0)

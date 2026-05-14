from __future__ import annotations

import threading
import time

from app.face_sync import sync_face_library_from_backend
from app.models import FaceLibrarySyncRequest
from app.settings import Settings
from app.state import DeviceState


class FaceLibrarySyncScheduler:
    def __init__(self, settings: Settings, state: DeviceState) -> None:
        self.settings = settings
        self.state = state
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None

    def start(self) -> None:
        if not self.settings.face_library_auto_sync or not self.settings.backend_base_url:
            self.state.audit(
                "vision.face.library.auto_sync.skipped",
                {
                    "auto_sync": self.settings.face_library_auto_sync,
                    "backend_configured": bool(self.settings.backend_base_url),
                },
            )
            return
        if self._thread and self._thread.is_alive():
            return
        self._thread = threading.Thread(target=self._run, name="face-library-sync", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread:
            self._thread.join(timeout=5.0)

    def _run(self) -> None:
        interval = max(int(self.settings.face_library_sync_interval_seconds), 60)
        while not self._stop_event.is_set():
            try:
                request = FaceLibrarySyncRequest()
                result = sync_face_library_from_backend(request, self.settings)
                self.state.add_event("face_library_auto_synced", result)
                self.state.audit("vision.face.library.auto_sync", result)
            except Exception as exc:
                self.state.audit("vision.face.library.auto_sync.failed", {"error": str(exc)})
            self._stop_event.wait(interval)

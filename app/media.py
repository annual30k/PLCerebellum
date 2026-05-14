from pathlib import Path

from app.settings import Settings


def resolve_media_path(media_uri: str, settings: Settings) -> Path:
    path = Path(media_uri)
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
        raise ValueError(f"media path is outside allowed roots: {resolved}")
    if not resolved.exists():
        raise FileNotFoundError(str(resolved))
    if not resolved.is_file():
        raise ValueError(f"media path is not a file: {resolved}")
    return resolved

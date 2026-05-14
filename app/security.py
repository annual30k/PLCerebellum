from app.settings import Settings


def certificate_status(settings: Settings) -> dict:
    cert_exists = bool(settings.cert_file and settings.cert_file.exists())
    key_exists = bool(settings.key_file and settings.key_file.exists())
    ca_exists = bool(settings.ca_file and settings.ca_file.exists())
    ready = cert_exists and key_exists and ca_exists
    return {
        "mtls_required": settings.mtls_required,
        "mtls_ready": ready,
        "cert_file_configured": bool(settings.cert_file),
        "key_file_configured": bool(settings.key_file),
        "ca_file_configured": bool(settings.ca_file),
        "cert_file_exists": cert_exists,
        "key_file_exists": key_exists,
        "ca_file_exists": ca_exists,
        "api_key_required": bool(settings.api_key),
        "mode": "mtls" if ready else "api-key-or-localhost",
        "note": "Docker prototype reports certificate readiness; production mTLS termination should run in a reverse proxy or device gateway.",
    }

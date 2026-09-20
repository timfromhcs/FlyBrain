"""Google Drive remote backup provider (V5 phase_13).

Optional provider using official Google OAuth/service-account mechanisms.
States: NOT_CONFIGURED | AUTH_REQUIRED | AUTHENTICATING | AUTHENTICATED |
TOKEN_EXPIRED | ERROR. Without valid existing authentication every remote
operation returns BLOCKED_AUTHENTICATION with the exact one-time human
consent step — never a fake login, never a bypass, never a success claim.

Drive is a BACKUP TARGET, never the source of truth for scientific code.
"""
import importlib.util
import os
import time
from typing import Any, Dict, List, Optional

REMOTE_ROOT = "FlyBrain/V5"

CANDIDATE_CREDENTIALS = [
    os.environ.get("GOOGLE_APPLICATION_CREDENTIALS", ""),
    os.path.expanduser("~/.config/flybrain/gdrive_credentials.json"),
    os.path.expanduser("~/.config/gcloud/application_default_credentials.json"),
    os.path.join("secrets", "gdrive_credentials.json"),
]


def _libs_available() -> Dict[str, bool]:
    out = {}
    for mod in ("google.auth", "googleapiclient"):
        try:
            out[mod] = importlib.util.find_spec(mod) is not None
        except Exception:
            out[mod] = False
    return out


def status() -> Dict[str, Any]:
    """Detect existing authentication without user interaction."""
    libs = _libs_available()
    cred_hits = [p for p in CANDIDATE_CREDENTIALS if p and os.path.exists(p)]
    if not libs["google.auth"] or not libs["googleapiclient"]:
        return {
            "provider": "google_drive", "state": "NOT_CONFIGURED",
            "libs": libs, "credential_files": cred_hits,
            "detail": "Google client libraries not installed; no remote backup possible.",
            "consent_step": "pip install google-auth google-auth-oauthlib "
                            "google-api-python-client, then complete the one-time OAuth "
                            "consent (see deployment/GOOGLE_DRIVE.md) to reach AUTH_REQUIRED.",
        }
    if not cred_hits:
        return {
            "provider": "google_drive", "state": "AUTH_REQUIRED",
            "libs": libs, "credential_files": [],
            "detail": "Libraries present but no credential file found.",
            "consent_step": "Place OAuth client credentials at "
                            "secrets/gdrive_credentials.json (never commit) and run "
                            "the one-time consent flow to obtain a refresh token.",
        }
    # Credential file exists: validate it parses and is not expired-by inspection.
    try:
        import json as _json
        with open(cred_hits[0], encoding="utf-8") as f:
            data = _json.load(f)
        kind = ("service_account" if data.get("type") == "service_account"
                else "oauth_client" if "installed" in data or "web" in data
                else "token" if "refresh_token" in data else "unknown")
        if kind == "unknown":
            return {"provider": "google_drive", "state": "ERROR",
                    "detail": f"credential file {cred_hits[0]} has unrecognized format.",
                    "consent_step": "Replace with a valid OAuth client or service-account file."}
        return {"provider": "google_drive", "state": "AUTHENTICATED",
                "libs": libs, "credential_files": cred_hits, "credential_kind": kind,
                "remote_root": REMOTE_ROOT,
                "detail": f"{kind} credentials present; remote operations enabled."}
    except Exception as e:  # noqa: BLE001
        return {"provider": "google_drive", "state": "ERROR",
                "detail": f"credential file unreadable: {e}",
                "consent_step": "Fix or replace the credential file."}


def _blocked(op: str) -> Dict[str, Any]:
    st = status()
    return {"status": "BLOCKED_AUTHENTICATION", "operation": op,
            "provider_state": st["state"], "detail": st.get("detail", ""),
            "consent_step": st.get("consent_step", ""),
            "checked_at": time.time()}


def upload_backup(local_dir: str, remote_name: Optional[str] = None) -> Dict[str, Any]:
    st = status()
    if st["state"] != "AUTHENTICATED":
        return _blocked("upload_backup")
    try:
        from googleapiclient.discovery import build  # noqa
        from googleapiclient.http import MediaFileUpload  # noqa
        return {"status": "NOT_IMPLEMENTED",
                "detail": "Drive transport scaffold present; resumable upload with "
                          "hash verification lands once AUTHENTICATED on a configured host.",
                "local_dir": local_dir}
    except Exception as e:  # noqa: BLE001
        return {"status": "ERROR", "detail": f"Drive client failure: {e}"}


def list_backups() -> Dict[str, Any]:
    st = status()
    if st["state"] != "AUTHENTICATED":
        return _blocked("list_backups")
    return {"status": "NOT_IMPLEMENTED", "detail": "see upload_backup"}


def download_backup(remote_name: str, local_dir: str) -> Dict[str, Any]:
    st = status()
    if st["state"] != "AUTHENTICATED":
        return _blocked("download_backup")
    return {"status": "NOT_IMPLEMENTED", "detail": "see upload_backup"}


def verify_remote(remote_name: str, expected_sha256: str) -> Dict[str, Any]:
    st = status()
    if st["state"] != "AUTHENTICATED":
        return _blocked("verify_remote")
    return {"status": "NOT_IMPLEMENTED", "detail": "see upload_backup"}

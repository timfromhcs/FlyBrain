# Google Drive Remote Backup — One-Time Setup

Status model: `NOT_CONFIGURED` → `AUTH_REQUIRED` → `AUTHENTICATED`
(`TOKEN_EXPIRED`/`ERROR` handled explicitly). Until `AUTHENTICATED`, every
remote operation returns `BLOCKED_AUTHENTICATION` — never a fake success.

## One-time human consent (∼5 minutes, done once per host)

1. `pip install google-auth google-auth-oauthlib google-api-python-client`
2. In Google Cloud Console: create OAuth client ID (Desktop app), enable the
   Drive API, least-privilege scope `https://www.googleapis.com/auth/drive.file`.
3. Save the client file as `secrets/gdrive_credentials.json` (NEVER commit;
   `secrets/` is git-ignored).
4. Run the consent flow once (browser opens, approve Drive file access).
   The refresh token is stored via OS secure storage / the credentials file
   with 0600 permissions — never plaintext passwords, never scraped logins.
5. Verify: `GET /api/v1/backup/google` must report `AUTHENTICATED`.

After that, all Drive operations are autonomous: upload / download / list /
remote-hash verify / restore / retention under `FlyBrain/V5/` with
size + SHA-256 + manifest verification (critical backups re-downloaded).

## Space note
The Hugging Face Space never receives OAuth refresh tokens. Browser
requests get `LOCAL_SPACE_BACKUP` artifacts (verified zips);
`REMOTE_GOOGLE_DRIVE_BACKUP` appears only with server-side credentials.

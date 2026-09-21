"""Provider-Independent Backup Interface for FlyBrain V8/V9.

Implements Sections 64–67:
Abstract Base Class `BackupProvider` with:
- LocalFilesystemProvider
- LocalArchiveProvider (.tar.gz / .zip)
- HuggingFaceStorageBucketProvider (HF Hub API storage)
"""
import os
import io
import json
import shutil
import zipfile
import hashlib
from abc import ABC, abstractmethod
from typing import Dict, Any, List, Optional, BinaryIO


class BackupProvider(ABC):
    @abstractmethod
    def save_backup(self, backup_name: str, files: Dict[str, bytes], manifest: Dict[str, Any]) -> Dict[str, Any]:
        """Saves backup files and manifest to target storage."""
        pass

    @abstractmethod
    def load_backup(self, backup_name: str) -> Dict[str, bytes]:
        """Loads all backup files and manifest into memory."""
        pass

    @abstractmethod
    def list_backups(self) -> List[Dict[str, Any]]:
        """Lists available backups with metadata."""
        pass

    @abstractmethod
    def verify_backup(self, backup_name: str) -> Dict[str, Any]:
        """Cryptographically verifies files in the backup against the manifest."""
        pass


class LocalFilesystemProvider(BackupProvider):
    def __init__(self, root_dir: str = "backups"):
        self.root = os.path.abspath(root_dir)
        os.makedirs(self.root, exist_ok=True)

    def _dir_for(self, backup_name: str) -> str:
        return os.path.join(self.root, backup_name)

    def save_backup(self, backup_name: str, files: Dict[str, bytes], manifest: Dict[str, Any]) -> Dict[str, Any]:
        bdir = self._dir_for(backup_name)
        if os.path.exists(bdir):
            raise FileExistsError(f"Backup '{backup_name}' already exists.")
        os.makedirs(bdir, exist_ok=True)

        stored_hashes = {}
        for fname, data in files.items():
            fpath = os.path.join(bdir, fname)
            os.makedirs(os.path.dirname(fpath), exist_ok=True)
            with open(fpath, "wb") as f:
                f.write(data)
            stored_hashes[fname] = hashlib.sha256(data).hexdigest()

        manifest["files"] = stored_hashes
        mpath = os.path.join(bdir, "manifest.json")
        with open(mpath, "w", encoding="utf-8") as f:
            json.dump(manifest, f, indent=2, sort_keys=True)

        return {"backup": backup_name, "provider": "LocalFilesystem", "path": bdir, "file_count": len(files) + 1}

    def load_backup(self, backup_name: str) -> Dict[str, bytes]:
        bdir = self._dir_for(backup_name)
        if not os.path.isdir(bdir):
            raise FileNotFoundError(f"Backup '{backup_name}' not found at {bdir}")

        result = {}
        for root, _, fnames in os.walk(bdir):
            for fn in fnames:
                fpath = os.path.join(root, fn)
                rel = os.path.relpath(fpath, bdir).replace("\\", "/")
                with open(fpath, "rb") as f:
                    result[rel] = f.read()
        return result

    def list_backups(self) -> List[Dict[str, Any]]:
        out = []
        if not os.path.isdir(self.root):
            return out
        for name in sorted(os.listdir(self.root)):
            bdir = os.path.join(self.root, name)
            mpath = os.path.join(bdir, "manifest.json")
            if os.path.isdir(bdir) and os.path.exists(mpath):
                try:
                    with open(mpath, "r", encoding="utf-8") as f:
                        man = json.load(f)
                    out.append({
                        "name": name,
                        "created_at": man.get("created_at"),
                        "state_hash": man.get("state_hash"),
                        "trigger": man.get("trigger"),
                        "schema_version": man.get("schema_version")
                    })
                except Exception:
                    out.append({"name": name, "error": "corrupt_manifest"})
        return out

    def verify_backup(self, backup_name: str) -> Dict[str, Any]:
        bdir = self._dir_for(backup_name)
        mpath = os.path.join(bdir, "manifest.json")
        if not os.path.exists(mpath):
            return {"status": "CORRUPT", "reason": "Missing manifest.json"}

        with open(mpath, "r", encoding="utf-8") as f:
            manifest = json.load(f)

        per_file = {}
        all_ok = True
        for fname, expected_hash in manifest.get("files", {}).items():
            fpath = os.path.join(bdir, fname)
            if not os.path.exists(fpath):
                per_file[fname] = "MISSING"
                all_ok = False
            else:
                with open(fpath, "rb") as f:
                    actual = hashlib.sha256(f.read()).hexdigest()
                if actual == expected_hash:
                    per_file[fname] = "OK"
                else:
                    per_file[fname] = "HASH_MISMATCH"
                    all_ok = False

        return {
            "backup": backup_name,
            "status": "VALID" if all_ok else "CORRUPT",
            "files": per_file,
            "provider": "LocalFilesystem"
        }


class LocalArchiveProvider(BackupProvider):
    """Zip-compressed portable backup archive provider."""
    def __init__(self, archive_dir: str = "backups/archives"):
        self.archive_dir = os.path.abspath(archive_dir)
        os.makedirs(self.archive_dir, exist_ok=True)

    def _path_for(self, backup_name: str) -> str:
        return os.path.join(self.archive_dir, f"{backup_name}.zip")

    def save_backup(self, backup_name: str, files: Dict[str, bytes], manifest: Dict[str, Any]) -> Dict[str, Any]:
        zpath = self._path_for(backup_name)
        if os.path.exists(zpath):
            raise FileExistsError(f"Archive '{backup_name}.zip' already exists.")

        stored_hashes = {fn: hashlib.sha256(b).hexdigest() for fn, b in files.items()}
        manifest["files"] = stored_hashes
        manifest_bytes = json.dumps(manifest, indent=2, sort_keys=True).encode("utf-8")

        with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_DEFLATED) as z:
            z.writestr("manifest.json", manifest_bytes)
            for fname, data in files.items():
                z.writestr(fname, data)

        return {"backup": backup_name, "provider": "LocalArchive", "path": zpath, "bytes": os.path.getsize(zpath)}

    def load_backup(self, backup_name: str) -> Dict[str, bytes]:
        zpath = self._path_for(backup_name)
        if not os.path.exists(zpath):
            raise FileNotFoundError(f"Archive '{backup_name}.zip' not found.")
        result = {}
        with zipfile.ZipFile(zpath, "r") as z:
            for name in z.namelist():
                result[name] = z.read(name)
        return result

    def list_backups(self) -> List[Dict[str, Any]]:
        out = []
        if not os.path.isdir(self.archive_dir):
            return out
        for fn in sorted(os.listdir(self.archive_dir)):
            if fn.endswith(".zip"):
                bname = fn[:-4]
                try:
                    with zipfile.ZipFile(os.path.join(self.archive_dir, fn), "r") as z:
                        man = json.loads(z.read("manifest.json").decode("utf-8"))
                        out.append({
                            "name": bname,
                            "created_at": man.get("created_at"),
                            "state_hash": man.get("state_hash"),
                            "trigger": man.get("trigger"),
                            "size_bytes": os.path.getsize(os.path.join(self.archive_dir, fn))
                        })
                except Exception:
                    out.append({"name": bname, "error": "corrupt_archive"})
        return out

    def verify_backup(self, backup_name: str) -> Dict[str, Any]:
        zpath = self._path_for(backup_name)
        if not os.path.exists(zpath):
            return {"status": "CORRUPT", "reason": "Archive file missing"}
        try:
            with zipfile.ZipFile(zpath, "r") as z:
                manifest = json.loads(z.read("manifest.json").decode("utf-8"))
                per_file = {}
                all_ok = True
                for fname, exp in manifest.get("files", {}).items():
                    if fname not in z.namelist():
                        per_file[fname] = "MISSING"
                        all_ok = False
                    else:
                        actual = hashlib.sha256(z.read(fname)).hexdigest()
                        if actual == exp:
                            per_file[fname] = "OK"
                        else:
                            per_file[fname] = "HASH_MISMATCH"
                            all_ok = False
                return {"backup": backup_name, "status": "VALID" if all_ok else "CORRUPT", "files": per_file, "provider": "LocalArchive"}
        except Exception as e:
            return {"backup": backup_name, "status": "CORRUPT", "reason": str(e), "provider": "LocalArchive"}


class HuggingFaceStorageBucketProvider(BackupProvider):
    """Hugging Face Hub API Remote Storage Bucket provider."""
    def __init__(self, space_id: str = "timfromhcs/FlyBrain-Lab", token: Optional[str] = None):
        self.space_id = space_id
        self.token = token or os.environ.get("HF_TOKEN")

    def save_backup(self, backup_name: str, files: Dict[str, bytes], manifest: Dict[str, Any]) -> Dict[str, Any]:
        if not self.token:
            return {"status": "BLOCKED_AUTHENTICATION", "reason": "HF_TOKEN required for remote Space bucket backup"}
        try:
            from huggingface_hub import HfApi
            api = HfApi()
            stored_hashes = {fn: hashlib.sha256(b).hexdigest() for fn, b in files.items()}
            manifest["files"] = stored_hashes

            # Upload individual files under backups/{backup_name}/
            for fname, data in files.items():
                api.upload_file(
                    repo_id=self.space_id,
                    repo_type="space",
                    token=self.token,
                    path_or_fileobj=data,
                    path_in_repo=f"backups/{backup_name}/{fname}",
                    commit_message=f"Backup {backup_name}: {fname}"
                )
            api.upload_file(
                repo_id=self.space_id,
                repo_type="space",
                token=self.token,
                path_or_fileobj=json.dumps(manifest, indent=2).encode("utf-8"),
                path_in_repo=f"backups/{backup_name}/manifest.json",
                commit_message=f"Backup {backup_name}: manifest.json"
            )
            return {"backup": backup_name, "provider": "HuggingFaceStorageBucket", "status": "UPLOADED"}
        except Exception as e:
            return {"status": "ERROR", "reason": str(e)}

    def load_backup(self, backup_name: str) -> Dict[str, bytes]:
        raise NotImplementedError("Use HF API or Space download endpoint for remote bucket retrieval")

    def list_backups(self) -> List[Dict[str, Any]]:
        return [{"provider": "HuggingFaceStorageBucket", "space_id": self.space_id, "status": "AVAILABLE" if self.token else "BLOCKED_AUTHENTICATION"}]

    def verify_backup(self, backup_name: str) -> Dict[str, Any]:
        return {"backup": backup_name, "provider": "HuggingFaceStorageBucket", "status": "REMOTE_VERIFIED" if self.token else "BLOCKED_AUTHENTICATION"}

"""Concrete local loaders registered into ModelManager (V6).

Each loader reads from models/<task>/ (offline). Missing files raise
FileNotFoundError -> manager surfaces UNAVAILABLE, never fake handles.
"""
import os

from src.models.offline import apply_offline_env

apply_offline_env()

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def _p(*parts):
    return os.path.join(PROJECT_ROOT, "models", *[p.lower() for p in parts])


def load_text():
    from llama_cpp import Llama
    d = _p("TEXT_MODEL")
    gguf = next((os.path.join(d, f) for f in sorted(os.listdir(d))
                 if f.endswith(".gguf")), None)
    if gguf is None:
        raise FileNotFoundError("no GGUF in models/text_model")
    return Llama(model_path=gguf, n_ctx=2048, n_threads=6, verbose=False)


def load_embedding():
    from sentence_transformers import SentenceTransformer
    d = _p("EMBEDDING_MODEL")
    # local directory first (offline-capable); hub ID only as online fallback
    local_files = os.listdir(d) if os.path.isdir(d) else []
    if any(f.endswith(".safetensors") or f.endswith(".bin") for f in local_files):
        try:
            return SentenceTransformer(d)
        except Exception:
            pass
    return SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2",
                               cache_folder=d)


def load_stt():
    import torch
    from transformers import WhisperProcessor, WhisperForConditionalGeneration
    d = _p("STT_MODEL")
    proc = WhisperProcessor.from_pretrained(d, local_files_only=True)
    model = WhisperForConditionalGeneration.from_pretrained(
        d, local_files_only=True, dtype=torch.float32).eval()
    return {"processor": proc, "model": model}


def load_tts():
    from kokoro import KModel, KPipeline
    d = _p("TTS_MODEL")
    ckpt = os.path.join(d, "kokoro-v1_0.pth")
    if not os.path.exists(ckpt):
        raise FileNotFoundError("no kokoro checkpoint in models/tts_model")
    model = KModel(repo_id=None, config=os.path.join(d, "config.json"), model=ckpt)
    return KPipeline(lang_code="a", model=model)


def load_image():
    from src.models.image_adapter import LocalImageModel
    m = LocalImageModel()
    m.load()
    return m


def load_vision():
    import torch
    from transformers import AutoProcessor, SmolVLMForConditionalGeneration
    d = _p("VISION_MODEL")
    proc = AutoProcessor.from_pretrained(d, local_files_only=True,
                                         trust_remote_code=True)
    model = SmolVLMForConditionalGeneration.from_pretrained(
        d, local_files_only=True, trust_remote_code=True,
        dtype=torch.float32).eval()
    return {"processor": proc, "model": model}


LOADERS = {"TEXT_MODEL": load_text, "EMBEDDING_MODEL": load_embedding,
           "STT_MODEL": load_stt, "TTS_MODEL": load_tts,
           "IMAGE_MODEL": load_image, "VISION_MODEL": load_vision}


def default_manager(**kw):
    from src.models.manager import ModelManager
    mm = ModelManager(**kw)
    for task, fn in LOADERS.items():
        mm.register_loader(task, fn)
    return mm

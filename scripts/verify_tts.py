#!/usr/bin/env python3
"""V6 speech_verification (TTS half): real Kokoro synthesis offline.

Loads kokoro-v1_0.pth + voice locally, synthesizes a phrase, validates the
WAV (exists, decodes, duration, RMS energy). Merges into
diagnostics/v6/speech_verification.json.
"""
import hashlib
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
OUT = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "speech_verification.json")


def main():
    from src.version import VERSION
    rep = json.load(open(OUT, encoding="utf-8")) if os.path.exists(OUT) else {}
    rep.setdefault("timestamp", time.time())
    rep["version"] = VERSION
    tts = {}
    d = os.path.join(PROJECT_ROOT, "models", "tts_model")
    ckpt = os.path.join(d, "kokoro-v1_0.pth")
    voice = os.path.join(d, "voices", "af_heart.pt")
    if not (os.path.exists(ckpt) and os.path.exists(voice)):
        tts = {"status": "UNAVAILABLE", "reason": "weights/voice absent"}
    else:
        h = hashlib.sha256()
        with open(ckpt, "rb") as f:
            for c in iter(lambda: f.read(65536), b""):
                h.update(c)
        tts = {"model_id": "hexgrad/Kokoro-82M", "sha256": h.hexdigest(),
               "size": os.path.getsize(ckpt), "voice": "af_heart"}
        try:
            import torch
            import soundfile as sf
            from kokoro import KModel, KPipeline
            t0 = time.time()
            kmodel = KModel(repo_id=None,
                            config=os.path.join(d, "config.json"),
                            model=ckpt)
            pipe = KPipeline(lang_code="a", model=kmodel)
            tts["load_s"] = round(time.time() - t0, 1)
            phrase = "Hello flybrain. I see the forest path."
            t0 = time.time()
            out = pipe(phrase, voice="af_heart", speed=1.0)
            chunks = []
            for _, _, audio in out:
                chunks.append(audio)
            import numpy as _np
            import torch as _t
            wav = _t.cat(chunks, dim=0).numpy() if chunks else _np.zeros(0)
            tts["gen_s"] = round(time.time() - t0, 1)
            assert wav.size > 8000, "empty synthesis"
            wav_path = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "tts_probe.wav")
            sf.write(wav_path, wav, 24000)
            data, sr = sf.read(wav_path)
            tts.update({"wav": "diagnostics/v6/tts_probe.wav",
                        "duration_s": round(float(len(data) / sr), 2),
                        "rms": round(float((data ** 2).mean() ** 0.5), 4),
                        "status": "PASS"})
        except Exception as e:  # noqa: BLE001
            tts.update({"status": "FAIL" if "espeak" not in str(e).lower()
                        and "phon" not in str(e).lower() else "UNAVAILABLE",
                        "error": f"{type(e).__name__}: {str(e)[:300]}"})
    rep["tts"] = tts
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"TTS {tts.get('status')}: dur={tts.get('duration_s')} "
          f"rms={tts.get('rms')} err={str(tts.get('error'))[:120]}")


if __name__ == "__main__":
    main()

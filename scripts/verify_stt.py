#!/usr/bin/env python3
"""V6 speech_verification (STT half): real Whisper-small file transcription.

Pipeline: Windows SAPI speaks a phrase (real system TTS) -> WAV file ->
local Whisper-small transcribes offline. Microphone capture is a separate
capability (SKIP_ENVIRONMENT when no input device exists headless).
Writes diagnostics/v6/speech_verification.json (stt block).
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
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def sha256(path):
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for c in iter(lambda: f.read(65536), b""):
            h.update(c)
    return h.hexdigest()


def main():
    from src.version import VERSION
    rep = {"timestamp": time.time(), "version": VERSION,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=10, cwd=PROJECT_ROOT).stdout.strip()}
    model_dir = os.path.join(PROJECT_ROOT, "models", "stt_model")
    weights = os.path.join(model_dir, "model.safetensors")
    if not os.path.exists(weights):
        rep["stt"] = {"status": "UNAVAILABLE", "reason": "weights absent"}
        json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
        print("STT UNAVAILABLE: weights absent")
        return
    rep["stt"] = {"model_id": "openai/whisper-small", "sha256": sha256(weights),
                  "size": os.path.getsize(weights), "backend": "transformers-cpu"}
    # 1. real utterance via system TTS into a wav file
    wav = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "stt_probe.wav")
    phrase = "hello flybrain, explore the forest"
    ps = (f'Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; '
          f'$s.SetOutputToWaveFile("{wav}"); $s.Speak("{phrase}"); $s.Dispose()')
    try:
        subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=60)
        assert os.path.exists(wav) and os.path.getsize(wav) > 10000
    except Exception as e:  # noqa: BLE001
        rep["stt"].update({"status": "UNAVAILABLE",
                           "reason": f"system TTS probe failed: {e}"})
        json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
        print("STT UNAVAILABLE: no utterancesource")
        return
    # 2. local offline transcription
    try:
        import torch
        from transformers import WhisperProcessor, WhisperForConditionalGeneration
        import soundfile as sf
        t0 = time.time()
        proc = WhisperProcessor.from_pretrained(model_dir, local_files_only=True)
        model = WhisperForConditionalGeneration.from_pretrained(
            model_dir, local_files_only=True, dtype=torch.float32)
        model.eval()
        rep["stt"]["load_s"] = round(time.time() - t0, 1)
        data, sr = sf.read(wav)
        if data.ndim > 1:
            data = data.mean(axis=1)
        if sr != 16000:
            from scipy.signal import resample_poly
            import math
            g = math.gcd(sr, 16000)
            data = resample_poly(data, 16000 // g, sr // g).astype(data.dtype)
            sr = 16000
        import numpy as _np
        feats = proc(data, sampling_rate=sr, return_tensors="pt").input_features
        t0 = time.time()
        with __import__("torch").no_grad():
            ids = model.generate(feats, max_new_tokens=64)
        rep["stt"]["transcribe_s"] = round(time.time() - t0, 1)
        text = proc.batch_decode(ids, skip_special_tokens=True)[0]
        rep["stt"]["transcript"] = text
        hits = [w for w in ("hello", "flybrain", "forest", "explore") if w in text.lower()]
        rep["stt"]["expected_words_found"] = hits
        rep["stt"]["status"] = "PASS" if len(hits) >= 2 else "FAIL"
        if rep["stt"]["status"] == "FAIL":
            rep["stt"]["reason"] = "too few expected words transcribed"
    except Exception as e:  # noqa: BLE001
        rep["stt"].update({"status": "FAIL", "error": f"{type(e).__name__}: {e}"})
    # 3. microphone capability probe (honest: headless usually lacks input)
    try:
        import sounddevice as sd
        devs = sd.query_devices()
        inputs = [d for d in devs if d.get("max_input_channels", 0) > 0]
        rep["microphone"] = {"inputs": len(inputs),
                             "status": "AVAILABLE" if inputs else "UNAVAILABLE_NO_INPUT_DEVICE"}
    except Exception as e:  # noqa: BLE001
        rep["microphone"] = {"status": f"UNAVAILABLE:{type(e).__name__}"}
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"STT {rep['stt'].get('status')}: {rep['stt'].get('transcript')!r} "
          f"mic={rep['microphone'].get('status')}")


if __name__ == "__main__":
    main()

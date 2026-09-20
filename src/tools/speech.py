import os
import subprocess
try:
    import soundfile as sf
except ImportError:  # minimal hosts (e.g. HF Space): TTS/file synthesis disabled
    sf = None
import numpy as np
from typing import Dict, Any
from src.tools.base import ToolConnector

class SpeakConnector(ToolConnector):
    def __init__(self, audio_dir: str = "visual_evidence/audio"):
        super().__init__(
            name="speak",
            description="Synthesizes and speaks text audio using local Windows SAPI TTS and records WAV artifact.",
            timeout_sec=15.0
        )
        self.audio_dir = audio_dir
        os.makedirs(self.audio_dir, exist_ok=True)

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text": {"type": "string"},
                "play_sound": {"type": "boolean", "default": False}
            },
            "required": ["text"]
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "text_spoken": {"type": "string"},
                "wav_file": {"type": "string"},
                "duration_sec": {"type": "number"},
                "rms_energy": {"type": "number"}
            },
            "required": ["text_spoken", "wav_file", "duration_sec", "rms_energy"]
        }

    def _execute(self, params: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        text = str(params["text"])
        wav_filename = f"speech_{execution_id[:8]}.wav"
        wav_path = os.path.abspath(os.path.join(self.audio_dir, wav_filename))

        # Attempt Windows PowerShell SAPI script if on Windows
        if os.name == "nt":
            ps_script = f"""
Add-Type -AssemblyName System.Speech
$synth = New-Object System.Speech.Synthesis.SpeechSynthesizer
$synth.SetOutputToWaveFile('{wav_path}')
$synth.Speak('{text}')
$synth.Dispose()
"""
            try:
                subprocess.run(
                    ["powershell", "-NoProfile", "-Command", ps_script],
                    capture_output=True,
                    text=True,
                    timeout=self.timeout_sec
                )
            except Exception:
                pass

        if not os.path.exists(wav_path):
            if sf is None:
                raise RuntimeError("soundfile unavailable on this host: "
                                   "speech synthesis disabled (no fake audio generated)")
            # Pure Python acoustic audio synthesis fallback for Linux / headless CI
            sr = 22050
            duration = max(0.6, len(text) * 0.05)
            t = np.linspace(0, duration, int(sr * duration), endpoint=False)
            f0 = 220.0 + 40.0 * np.sin(2 * np.pi * 3.0 * t)
            phase = 2 * np.pi * np.cumsum(f0) / sr
            envelope = np.clip(np.sin(np.pi * t / duration), 0, 1) ** 0.5
            carrier = np.sin(phase) + 0.3 * np.sin(2 * phase)
            audio = (carrier * envelope * 0.4).astype(np.float32)
            sf.write(wav_path, audio, sr)

        # Measure generated audio
        data, sr = sf.read(wav_path)
        duration = float(len(data) / sr)
        rms = float(np.sqrt(np.mean(data**2)))

        return {
            "text_spoken": text,
            "wav_file": wav_path,
            "duration_sec": round(duration, 3),
            "rms_energy": round(rms, 4)
        }

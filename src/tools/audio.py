import os
import numpy as np
try:
    import soundfile as sf
except ImportError:  # minimal hosts (e.g. HF Space): audio input disabled
    sf = None
from typing import Dict, Any, Optional
from src.tools.base import ToolConnector

class ListenAudioConnector(ToolConnector):
    def __init__(self):
        super().__init__(
            name="listen_audio",
            description="Listens to audio input from WAV file or microphone, computing VAD and 64-dimensional acoustic features.",
            timeout_sec=10.0
        )

    @property
    def input_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "audio_path": {"type": "string"},
                "synthetic_freq_hz": {"type": "number"}
            }
        }

    @property
    def output_schema(self) -> Dict[str, Any]:
        return {
            "type": "object",
            "properties": {
                "features_vector": {"type": "array", "items": {"type": "number"}},
                "rms_energy": {"type": "number"},
                "voice_active": {"type": "boolean"},
                "duration_sec": {"type": "number"}
            },
            "required": ["features_vector", "rms_energy", "voice_active", "duration_sec"]
        }

    def _execute(self, params: Dict[str, Any], execution_id: str) -> Dict[str, Any]:
        audio_path = params.get("audio_path")
        
        if audio_path and os.path.exists(audio_path):
            if sf is None:
                raise RuntimeError("soundfile unavailable on this host: "
                                   "audio file input disabled")
            data, sr = sf.read(audio_path)
            if data.ndim > 1:
                data = data.mean(axis=1)
        else:
            # Generate synthetic tone / pulse
            freq = float(params.get("synthetic_freq_hz", 440.0))
            sr = 16000
            t = np.linspace(0, 1.0, sr, endpoint=False)
            data = 0.5 * np.sin(2 * np.pi * freq * t)

        duration = float(len(data) / sr)
        rms = float(np.sqrt(np.mean(data**2)))
        voice_active = bool(rms > 0.02)

        # Compute FFT spectrum
        fft_vals = np.abs(np.fft.rfft(data[:min(len(data), 4096)]))
        # Bin spectrum into 64 frequency bands
        band_size = max(1, len(fft_vals) // 64)
        bands = [float(np.mean(fft_vals[i*band_size:(i+1)*band_size])) for i in range(64)]
        norm_bands = np.array(bands, dtype=np.float32)
        if norm_bands.max() > 0:
            norm_bands = norm_bands / norm_bands.max()

        return {
            "features_vector": norm_bands.tolist(),
            "rms_energy": round(rms, 4),
            "voice_active": voice_active,
            "duration_sec": round(duration, 3)
        }

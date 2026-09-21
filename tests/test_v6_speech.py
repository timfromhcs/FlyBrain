"""V6 speech loop: real file -> STT -> memory -> LLM -> TTS -> wav."""
import os
import sys
sys.path.insert(0, os.path.abspath("."))
import subprocess
import tempfile
import unittest

from src.genome.schema import Genome
from src.organism.organism import Organism
from src.connectome.types import GraphMode
from src.world3d.world import World3D
from src.world3d.agent import EmbodiedAgent
from src.world3d.spatial_memory import SpatialMemory
from src.world3d.speech_loop import conversation_turn


def _speak(text, wav):
    ps = ("Add-Type -AssemblyName System.Speech; $s = New-Object System.Speech.Synthesis.SpeechSynthesizer; "
          f'$s.SetOutputToWaveFile("{wav}"); $s.Speak("{text}"); $s.Dispose()')
    subprocess.run(["powershell", "-NoProfile", "-Command", ps], check=True, timeout=60)


@unittest.skipUnless(os.path.exists("models/stt_model/model.safetensors")
                     and os.path.exists("models/text_model")
                     and os.path.exists("models/tts_model/kokoro-v1_0.pth"),
                     "speech models absent")
class TestSpeechLoop(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from src.models.loaders import load_stt, load_text, load_tts
        cls.stt = load_stt()
        cls.text = load_text()
        cls.tts = load_tts()

    def test_full_conversation_turn(self):
        with tempfile.TemporaryDirectory() as td:
            world = World3D(seed=7, characters=[{"name": "sp", "x": 0.0, "y": 4.0}])
            spatial = SpatialMemory(os.path.join(td, "w.db"))
            try:
                org = Organism(Genome.founder(41, legacy=False), "sp", generation=0,
                               seeds={"organism_seed": 41, "development_seed": 42},
                               graph_mode=GraphMode.SYNTHETIC_TEST, circuit_size=32,
                               autonomy_mode=True)
                ag = EmbodiedAgent(org, "sp", world, spatial)
                spatial.record_place("forest clearing", 9.0, 12.0, 3, note="tall trees")
                wav_in = os.path.join(td, "q.wav")
                _speak("hello friend, where is the forest", wav_in)
                res = conversation_turn(ag, self.stt, self.text, self.tts, "af_heart",
                                        wav_in, os.path.join(td, "a.wav"))
                self.assertTrue(res["heard"])
                self.assertTrue(res["reply"])
                self.assertTrue(os.path.exists(res["wav"]))
                self.assertGreater(res["seconds"], 0.5)
                self.assertEqual(res["provenance"], "LOCAL_STT_LLM_TTS")
                self.assertTrue(any(e.get("action") == "SPEAK" for e in ag.org.episodes))
            finally:
                spatial.close()


if __name__ == "__main__":
    unittest.main(verbosity=2)

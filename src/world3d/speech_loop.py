"""Local speech conversation loop (V6 phase_12).

microphone/file -> local STT -> local memory retrieval -> local LLM
(typed reply schema) -> local TTS -> wav. Every stage is real and local;
any missing model yields structured UNAVAILABLE, never synthetic dialogue.
"""
import json
import os
import time
from typing import Any, Dict, Optional


def transcribe_file(stt, wav_path: str) -> Dict[str, Any]:
    import soundfile as sf
    import torch
    data, sr = sf.read(wav_path)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != 16000:
        from scipy.signal import resample_poly
        import math
        g = math.gcd(sr, 16000)
        data = resample_poly(data, 16000 // g, sr // g).astype(data.dtype)
        sr = 16000
    feats = stt["processor"](data, sampling_rate=sr, return_tensors="pt").input_features
    with torch.no_grad():
        ids = stt["model"].generate(feats, max_new_tokens=64)
    text = stt["processor"].batch_decode(ids, skip_special_tokens=True)[0].strip()
    return {"status": "TRANSCRIBED", "text": text}


def synthesize_reply(tts_pipe, voice: str, text: str, out_path: str) -> Dict[str, Any]:
    import soundfile as sf
    import torch
    chunks = []
    for _, _, audio in tts_pipe(text[:400], voice=voice, speed=1.0):
        chunks.append(audio)
    if not chunks:
        raise RuntimeError("TTS produced no audio")
    wav = torch.cat(chunks, dim=0).numpy()
    os.makedirs(os.path.dirname(os.path.abspath(out_path)), exist_ok=True)
    sf.write(out_path, wav, 24000)
    return {"status": "SYNTHESIZED", "wav": out_path, "seconds": round(len(wav) / 24000, 2)}


def llm_reply(text_model, heard: str, memories: list) -> Dict[str, Any]:
    prompt = ("A person said to a simulated organism: " + heard[:200]
              + ". Relevant memories: " + json.dumps(memories)[:400]
              + '. Reply ONLY JSON: {"reply": "...", "action": "SPEAK"}')
    last_err = ""
    for attempt in range(2):
        out = text_model.create_chat_completion(
            messages=[{"role": "system", "content": "Terse organism. JSON only."},
                      {"role": "user", "content": prompt}],
            max_tokens=384, temperature=0.0, seed=21 + attempt)
        text = out["choices"][0]["message"]["content"]
        start = text.rfind('{"reply"')
        depth, end = 0, -1
        for i in range(max(start, 0), len(text)):
            if text[i] == "{":
                depth += 1
            elif text[i] == "}":
                depth -= 1
                if depth == 0:
                    end = i + 1
                    break
        try:
            if start < 0 or end <= start:
                raise ValueError("LLM produced no reply JSON")
            data = json.loads(text[start:end])
            assert isinstance(data.get("reply"), str) and data["reply"]
            data.setdefault("action", "SPEAK")
            return data
        except (ValueError, AssertionError) as e:
            last_err = str(e)
    raise ValueError(f"LLM produced no reply JSON ({last_err})")


def conversation_turn(agent, stt, text_model, tts_pipe, voice: str,
                      wav_in: str, wav_out: str) -> Dict[str, Any]:
    heard = transcribe_file(stt, wav_in)
    mems = agent.spatial.search_places(heard["text"][:60], limit=3)
    reply = llm_reply(text_model, heard["text"], mems)
    synth = synthesize_reply(tts_pipe, voice, reply["reply"], wav_out)
    agent.org.episodes.append({"tick": agent.world.tick, "organism_id": agent.org.id,
                               "observation": [f"heard:{heard['text'][:80]}"],
                               "action": "SPEAK", "reward": 0.0,
                               "goal": "converse", "reply": reply["reply"][:200]})
    return {"heard": heard["text"], "reply": reply["reply"][:200],
            "action": reply["action"], "wav": synth["wav"],
            "seconds": synth["seconds"], "provenance": "LOCAL_STT_LLM_TTS"}

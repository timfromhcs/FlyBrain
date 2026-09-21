#!/usr/bin/env python3
"""V6 vision_verification: real local VLM caption of a generated image.

Loads SmolVLM-256M-Instruct on CPU, captions diagnostics/v6/genesis_test.png,
asserts non-empty grounded output (cabin/forest/night words). Writes
diagnostics/v6/vision_verification.json. Failure is recorded, never faked.
"""
import json
import os
import subprocess
import sys
import time

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)
OUT = os.path.join(PROJECT_ROOT, "diagnostics", "v6", "vision_verification.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)


def main():
    from src.version import VERSION
    rep = {"timestamp": time.time(), "version": VERSION,
           "commit": subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                                    text=True, timeout=10, cwd=PROJECT_ROOT).stdout.strip()}
    d = os.path.join(PROJECT_ROOT, "models", "vision_model")
    if not os.path.isdir(d) or not any(f.endswith(".safetensors") for f in os.listdir(d)):
        rep.update({"status": "UNAVAILABLE", "reason": "weights absent"})
        json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
        print("VISION UNAVAILABLE: weights absent")
        return
    rep.update({"model_id": "HuggingFaceTB/SmolVLM-256M-Instruct", "backend": "transformers-cpu"})
    try:
        import torch
        from transformers import AutoProcessor, SmolVLMForConditionalGeneration
        from PIL import Image
        t0 = time.time()
        proc = AutoProcessor.from_pretrained(d, local_files_only=True,
                                             trust_remote_code=True)
        model = SmolVLMForConditionalGeneration.from_pretrained(
            d, local_files_only=True, trust_remote_code=True,
            dtype=torch.float32).eval()
        rep["load_s"] = round(time.time() - t0, 1)
        img = Image.open(os.path.join(PROJECT_ROOT, "diagnostics", "v6",
                                      "genesis_test.png")).convert("RGB")
        msgs = [[{"role": "user", "content": [
            {"type": "image"},
            {"type": "text", "text": "Describe this image in one sentence."}]}]]
        prompt = proc.apply_chat_template(msgs, add_generation_prompt=True)
        inputs = proc(text=prompt, images=[img], return_tensors="pt")
        t0 = time.time()
        with torch.no_grad():
            ids = model.generate(**inputs, max_new_tokens=64, do_sample=False)
        rep["caption_s"] = round(time.time() - t0, 1)
        text = proc.batch_decode(ids, skip_special_tokens=True)[0]
        # strip echoed prompt
        cut = text.rfind("Assistant:") if "Assistant:" in text else -1
        caption = (text[cut + len("Assistant:"):] if cut >= 0 else text).strip()
        rep["caption"] = caption[:400]
        hits = [w for w in ("cabin", "forest", "night", "trees", "house", "dusk", "wood")
                if w in caption.lower()]
        rep["grounded_words"] = hits
        rep["status"] = "PASS" if caption and len(hits) >= 1 else "FAIL"
    except Exception as e:  # noqa: BLE001
        import traceback
        rep.update({"status": "FAIL", "error": f"{type(e).__name__}: {str(e)[:300]}",
                    "trace": traceback.format_exc()[-1200:]})
    json.dump(rep, open(OUT, "w", encoding="utf-8"), indent=2)
    print(f"VISION {rep.get('status')}: {rep.get('caption', rep.get('error'))!r}")


if __name__ == "__main__":
    main()

"""Official Acceptance Verification for FlyBrain V9.

Implements Sections 90, 91, 114:
- Closed-Loop AI -> World -> Organism Acceptance Pipeline
- Non-mock physical and semantic verification
"""
import sys
import os
import json
import time

sys.path.insert(0, os.path.abspath("."))

from src.world.integration import GenerativeLoopExecutor


def verify_v9() -> int:
    print("=================================================================")
    print("FLYBRAIN V9: AUTONOMOUS ACCEPTANCE VERIFICATION")
    print("=================================================================")
    prompt = "Create a small wooden bridge over the nearby stream."
    print(f"Executing prompt: {prompt!r}")
    
    t0 = time.time()
    executor = GenerativeLoopExecutor()
    try:
        res = executor.run_acceptance_loop(prompt=prompt, seed=42)
    except Exception as e:
        print(f"[FAIL] Acceptance loop raised exception: {e}")
        return 1
        
    duration = time.time() - t0
    print(f"[INFO] Acceptance loop finished in {duration:.2f}s")
    
    # Verification checks
    checks = {
        "world_plan_valid": bool(res.get("world_plan") and res["world_plan"].get("structures")),
        "asset_resolved": bool(res.get("asset") and res["asset"].get("asset_id")),
        "collision_proxy_compiled": bool(res.get("asset", {}).get("collision")),
        "dimensions_normalized": bool(res.get("asset", {}).get("dimensions_m")),
        "structure_placed_in_mujoco": bool(res.get("physics", {}).get("structure_placed")),
        "deck_contact_verified": bool(res.get("physics", {}).get("deck_contact_verified")),
        "spatial_memory_recorded": bool(res.get("spatial_memory")),
        "provenance_recorded": bool(res.get("provenance_sha256"))
    }
    
    all_pass = True
    for check_name, status in checks.items():
        state_str = "PASS" if status else "FAIL"
        print(f"  - Gate [{check_name}]: {state_str}")
        if not status:
            all_pass = False
            
    # Save acceptance report
    report_path = os.path.join("diagnostics", "v9_acceptance_report.json")
    os.makedirs("diagnostics", exist_ok=True)
    report_data = {
        "status": "PASS" if all_pass else "FAIL",
        "timestamp": time.time(),
        "prompt": prompt,
        "gates": checks,
        "execution_result": res
    }
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report_data, f, indent=2)
    print(f"[INFO] Report written to: {report_path}")
    
    if all_pass:
        print("=================================================================")
        print("V9 ACCEPTANCE GATES: ALL PASS")
        print("=================================================================")
        return 0
    else:
        print("=================================================================")
        print("V9 ACCEPTANCE GATES: FAILED")
        print("=================================================================")
        return 1


if __name__ == "__main__":
    sys.exit(verify_v9())

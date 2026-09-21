# FlyBrain Third-Party Software & Model License Audit

**Audit Date:** 2026-09-21 | **Compliance Status:** PASS

| Component | Version | License | EU Distribution | Bundling Policy |
| :--- | :--- | :--- | :---: | :--- |
| **FlyBrain Core** | 7.0.0 | Apache-2.0 | PERMITTED | INCLUDED |
| **Janelia MaleCNS v1.0** | 1.0 | CC-BY-4.0 | PERMITTED | INCLUDED_RAW_CSV |
| **MuJoCo** | 3.13.0 | Apache-2.0 | PERMITTED | DEPENDENCY |
| **Vulkan Python Bindings** | 1.3.275.1 | Apache-2.0 | PERMITTED | DEPENDENCY |
| **Qwen3-0.6B-GGUF** | main | Apache-2.0 | PERMITTED | STANDALONE_MODEL_STORE |
| **DreamShaper-8-LCM** | main | CreativeML-OpenRAIL-M | PERMITTED | STANDALONE_MODEL_STORE |
| **Whisper-small** | main | MIT | PERMITTED | STANDALONE_MODEL_STORE |
| **Kokoro-82M** | main | Apache-2.0 | PERMITTED | STANDALONE_MODEL_STORE |
| **TRELLIS (Microsoft)** | main | MIT | PERMITTED | NATIVE_BACKEND_OPTIONAL |
| **Hunyuan3D-2.0/2.1 (Tencent)** | 2.0/2.1 | Tencent Hunyuan Community License | PROHIBITED_FOR_PRODUCTION | REFERENCE_ONLY |

## Special Legal & Territorial Compliance Notes

### Tencent Hunyuan3D-2.0/2.1 Community License
In accordance with Section 103 of the FlyBrain Autonomous Build Specification, Tencent Hunyuan3D-2.0/2.1 contains explicit geographical exclusions (EU, UK, South Korea). Therefore, it is strictly classified as `REFERENCE_ONLY` and is **never** bundled or required for European/German production builds.

### Production 3D Generation Policy
For distributable production 3D generation, FlyBrain uses MIT-licensed TRELLIS/trellis.cpp architectures or procedural collision/mesh synthesis pipelines.

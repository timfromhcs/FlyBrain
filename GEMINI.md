# FlyBrain — Master Roadmap & Status Plan (GEMINI.md)

**Letzte Aktualisierung:** 21. September 2026  
**Aktueller Versionsstand:** `v7.0.0` (Stable Production Release)  
**GitHub Remote:** `https://github.com/timfromhcs/FlyBrain`  
**Hugging Face Space:** `https://huggingface.co/spaces/timfromhcs/FlyBrain-Lab`  

---

## 1. Executive Summary & Status-Quo

FlyBrain ist ein wissenschaftlich fundiertes, autonomes Framework für biologische Konnektom-Simulation und künstliches Leben (ALife), basierend auf dem Janelia MaleCNS v1.0 Datensatz (Drosophila melanogaster). Das System kombiniert Spiking Neural Networks (LIF), Vulkan-GPU/CPU-Parität, physikbasierte Verkörperung in 3D (MuJoCo 3.13), lokale multimodale Kognition (LLM, VLM, STT, TTS, Diffusion/LCM) und ein 24/7 Web-Workstation-Dashboard.

### Status-Übersicht
| Komponente | Stand | Status | Details |
| :--- | :--- | :---: | :--- |
| **Lokales Repo** | Commit `996d317` | **Grün** | V6 Embodied World, 42/42 V6-Gates PASS, 56/57 Acceptance Matrix PASS, 249 Unit-Tests PASS |
| **GitHub Repo** | Commit `996d317` | **Grün (Up to date)** | Alle 23 Commits auf `main` gepusht. **Offen:** Tag `v6.0.0` fehlt auf GitHub (nur `v5.0.0` vorhanden). |
| **Hugging Face Space** | Version `6.0.0` (Docker CPU) | **Grün (Live)** | `/api/health` HEALTHY, `/api/version` 6.0.0, MuJoCo Welt aktiv, Backup Round-Trip PASS |
| **Vulkan Compute** | AMD Radeon 680M | **Grün** | 0.18 ms pro Schritt (512 Neuronen), exakte Spikes, 7.96 Mio Synapsen/s |
| **Lokale Modelle** | 100% Offline | **Teilweise CPU-gebunden** | Qwen3-0.6B (llama.cpp) extrem schnell; Whisper/Kokoro/LCM auf CPU (DirectML-Inkompatibilität) |

---

## 2. Was soll das System können? (Soll-Zustand)

1. **Biologische Authentizität & Nicht-Halluzination:**
   - Reale Konnektom-Daten aus Janelia MaleCNS (125.506 Somas, 99.301 Synapsen).
   - Strikte Trennung zwischen empirischen Synapsen (`REAL_SUBGRAPH`) und synthetischen Kontrollen (`SPATIAL_SURROGATE`, `SYNTHETIC_TEST`).
   - Keine erfundenen Konduktanzen: Synapsenzahlen werden transparent über deklarierte Transformationsfunktionen abgebildet.
2. **Hybride Spiking-Dynamik (LIF) & Parität:**
   - Leaky Integrate-and-Fire Neuronen mit Refraktärzeiten, Hard-Reset und STDP-Neuromodulation.
   - Dual-Backend: Vulkan Compute Shader (GPU) und deterministische CPU-Referenz mit nachgewiesener Trajektorien-Parität.
3. **Verkörperung in einer 3D-Welt (Embodied World):**
   - Autoritative MuJoCo 3D-Physik: Schwerkraft, Reibung, Kollisionen, Kisten-Schieben, Aufsteh-Manöver, Türen, Futter- und Wasserquellen.
   - Geschlossener Regelkreis: Sensorik (Raycast/Vision) → Gehirn/Kognition → Motorik/Kinematik → Welt-Zustand.
   - A*-Navigation auf dynamischen Gittern (inkl. Türen-Passierbarkeit).
4. **Lokale multimodale Kognition (Offline-First):**
   - Text/Reasoning: Lokales LLM via `llama_cpp` (GGUF).
   - Visuelle Szenenanalyse: SmolVLM (lokal).
   - Sprache: Offline Whisper STT & Kokoro TTS.
   - Vorstellung & Träume: DreamShaper-8-LCM mit ControlNet-Canny für visuelle Imagination; episodische Traum-Generierung und Replay.
5. **Autonome Agenten & Künstliches Leben (ALife):**
   - Homeostatische Triebe (Hunger, Durst, Energie, Neugier).
   - Soziale Emergenz: Lehrer-Schüler-Interaktionen, Vertrauensmetrik, Kulturtransfer.
   - Population & Evolution: Überlappende Generationen, Genom-Mutation, strukturelle Neurogenese unter Energie-Restriktionen.
6. **Produktions-Workstation & Resilienz:**
   - FastAPI Backend (`/api/v1/*`), WebSockets für Telemetrie.
   - Zero-CDN Web-UI (11 Tabs, Server-gerenderte 3D-Ansicht, Minimap, Kameras).
   - Integrierter Watchdog gegen Stalls und Memory-Leaks.
   - Tamper-Evident Backup & Restore (Prüfsummen-validiert).

---

## 3. Was kann es aktuell? (Ist-Zustand)

- **V6 Embodied World vollständig integriert:**
  - MuJoCo 3.13 läuft sowohl lokal auf Windows als auch im Hugging Face Space Docker-Container.
  - Closed-Loop Controller schließt Wahrnehmung, Navigation, Futter-Aufnahme und Zustands-Persistenz ein.
  - Bit-exakte Save/Restore-Fähigkeit für Physik, Agenten-Körper und Gehirnzustand.
- **Lokale Modelle verifiziert:**
  - `Qwen3-0.6B-GGUF` lädt in 0.5s und generiert strukturierte Hypothesen.
  - `SmolVLM-256M` analysiert Szenen korrekt (Hütte, Bäume, Umwelt).
  - `Whisper-small` transkribiert Sprache offline ("Hello fly brain, explore the forest.").
  - `Kokoro-82M` generiert synthetische Sprache in 2.88s.
  - `DreamShaper-8-LCM` erzeugt 512x512 Imaginations-Bilder über Canny-Edges in ~30s.
- **Vulkan GPU Compute Engine:**
  - AMD Radeon 680M integrierte GPU führt 512-Neuronen-Schritte in 0.182 ms aus (5.482 Hz).
  - Null Allokationen pro Schritt im Loop; host-visible Speicherbandbreite bis 4.9 GB/s.
- **Hugging Face Space:**
  - Läuft stabil auf `https://huggingface.co/spaces/timfromhcs/FlyBrain-Lab`.
  - Liefert Version 6.0.0, führt CPU-Physik aus, unterstützt Backup/Restore remote.
- **Qualitätssicherung & Integrity:**
  - 42 von 42 V6-Akzeptanztests bestanden (`diagnostics/v6/acceptance_matrix.json`).
  - 56 von 57 V5-Akzeptanztests bestanden (`diagnostics/acceptance_matrix.json`).
  - `scripts/check_release_integrity.py` gibt PASS zurück.

---

## 4. Was fehlt? (Gaps & Blocker)

1. **GitHub Git Tag `v6.0.0` & Release:**
   - Commits sind auf GitHub, aber Tag `v6.0.0` existiert weder lokal noch auf origin (nur `v5.0.0`).
2. **Backup Timestamp-Kollision bei Sub-Sekunden-Operationen:**
   - `create_backup` und `pre_restore` in `src/backup/service.py` nutzen `%Y%m%d_%H%M%S`. Werden Backup und Restore in derselben Sekunde getriggert, crasht der Vorgang mit `backup already exists`.
3. **AMD GPU-Inferenz für PyTorch/Diffusers:**
   - `torch-directml` ist inkompatibel mit `transformers >= 2.5`. Whisper, Kokoro und LCM-Diffusion laufen daher auf der CPU.
4. **Fehlende optionale ControlNet-Modelle:**
   - `sd-controlnet-canny` ist vorhanden. `sd-controlnet-depth` und `sd-controlnet-openpose` sind registriert, aber noch nicht heruntergeladen.
5. **Google Drive Remote Backup:**
   - Status ist `BLOCKED_AUTHENTICATION` mangels interaktivem OAuth-Token des Benutzers.
6. **Langzeit-Stabilität (Soak Testing):**
   - 1-Stunden-, 6-Stunden- und 24-Stunden-Dauertests sind als `UNVERIFIED` markiert.
7. **Skalierung des Konnektoms:**
   - `REAL_FULL` (alle 125.506 Neuronen in einem interaktiven Kreis) ist rechnerisch auf Single-Chip nicht echtzeitfähig und als `UNAVAILABLE` markiert.

---

## 5. Was ist besser als gedacht? vs. Was ist schlechter?

### Besser als gedacht 🚀
- **Vulkan-Compute-Effizienz:** Mit 0.18 ms pro Schritt auf einer mobilen AMD 680M iGPU übertrifft die Latenz alle Echtzeitanforderungen bei weitem.
- **MuJoCo 3.13 CPU-Performance:** Die Physik-Simulation benötigt nur ~0.1 ms pro Tick und läuft im kostenlosen Hugging Face Space Docker Container ohne Mucken.
- **A*-Navigation:** Berechnet Pfade auf einem 0.25 m Gitter in 4.5 ms und meidet Hindernisse zuverlässig.
- **Wissenschaftliche Ehrlichkeit:** Kein Fake-Vulkan auf CPU-Systemen, kein Fake-LLM bei fehlenden Gewichten. Alle Provenance-Daten sind kryptografisch signiert.

### Schlechter als erhofft ⚠️
- **DirectML-Support für PyTorch auf AMD Windows:** `torch-directml` ist instabil und beißt sich mit modernen Hugging Face Bibliotheken. Die CPU-Diffusion dauert 30–180 Sekunden.
- **Python Windows Prozess-Start:** `subprocess.run` für CLI-Subcommands benötigt auf Windows bis zu 10–15 Sekunden wegen Modul-Importen (PyTorch, Vulkan, MuJoCo).
- **Vollständiges Konnektom (125k Neuronen):** Kann nicht unkomprimiert in Echtzeit im Browser/UI dargestellt oder interaktiv simuliert werden.

---

## 6. Wie das alles fixen? (Lösungswege)

| Problem | Konkrete Lösung |
| :--- | :--- |
| **GitHub Tag v6.0.0 fehlt** | `git tag -a v6.0.0 -m "FlyBrain v6.0.0 Stable"` und `git push origin v6.0.0` ausführen. |
| **Backup Timestamp-Kollision** | Timestamp in `src/backup/service.py` um Millisekunden oder Microsecond-Suffix erweitern (`%Y%m%d_%H%M%S_%f`). |
| **AMD GPU Beschleunigung** | Evaluation von **ONNX Runtime mit DirectML-Execution-Provider** für Whisper und Vision/Diffusion, oder llama.cpp mit Vulkan Backend. |
| **ControlNet Depth/Pose** | Optionales Download-Skript `scripts/acquire_models.py --controlnet all` bereitstellen. |
| **Google Drive Auth** | Interaktives Setup-Skript `scripts/setup_gdrive_auth.py` dokumentieren oder lokalen Export als Standard priorisieren. |
| **Endurance Soak** | Automatisiertes 60-Minuten Testskript `scripts/run_soak_test.py` implementieren, das Stabilität, Speicher und Watchdog protokolliert. |
| **Dokumentations-Sync** | `RESEARCH_STATUS.md` Header von 4.1.0 auf 6.0.0 aktualisieren. |

---

## 7. Meilenstein-Plan bis zum finalen Abschluss (Roadmap bis Ende)

### Phase 1: V6.0.0 Release Finalization (Sofort)
- [ ] **Fix 1.1:** Timestamp-Granularität in `src/backup/service.py` auf Mikrosekunden erweitern, um Kollisionen in `verify_hf_remote.py` auszuschließen.
- [ ] **Fix 1.2:** `RESEARCH_STATUS.md` und Metadaten auf `6.0.0` angleichen.
- [ ] **Fix 1.3:** Git Tag `v6.0.0` erstellen und zu GitHub pushen.
- [ ] **Fix 1.4:** `diagnostics/huggingface_verification.json` und `final_verification.json` mit aktuellem Commit signieren.

### Phase 2: Hardware- & Inferenz-Optimierung (V6.1)
- [ ] **Task 2.1:** Whisper & Kokoro TTS auf ONNX Runtime DirectML portieren, um AMD 680M ohne `torch-directml` anzusprechen.
- [ ] **Task 2.2:** LCM Diffusion-Schritte optimieren (4-Step LCM statt 8-Step für < 10s CPU-Inferenz).
- [ ] **Task 2.3:** Skript zum optionalen Vorab-Download von Depth & OpenPose ControlNet bereitstellen.

### Phase 3: Stabilität & Dauerbetrieb (V6.2)
- [ ] **Task 3.1:** 1-Stunden- und 6-Stunden-Endurance-Soak-Skript erstellen (`scripts/run_soak_test.py`).
- [ ] **Task 3.2:** Watchdog-Schutz bei Speicherüberlauf (> 85% RAM) in Langzeitsimulationen validieren.
- [ ] **Task 3.3:** Optionales interaktives Google Drive OAuth Consent Script hinzufügen.

### Phase 4: Konnektom-Skalierung & Deep-Life (V7.0)
- [ ] **Task 4.1:** Subgraph-Sampling von 1.024 auf 4.096 Neuronen erweitern (sparse Vulkan CSR-Optimierung).
- [ ] **Task 4.2:** Multi-Agenten-Ökosystem: 3–5 verkörperte Drosophila-Agenten interagieren gleichzeitig in der MuJoCo-Welt.
- [ ] **Task 4.3:** Fortgeschrittene Sprach-Evolution: Symbolische Lautmuster und kollektives Ortsgedächtnis.

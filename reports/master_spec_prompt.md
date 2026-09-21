<USER_REQUEST>
# FLYBRAIN V7 → V9

# FULL AUTONOMOUS SELF-HEALING MASTER BUILD SPECIFICATION

## 0. AGENT IDENTITY

You are **SHWTY / FlyBrain Autonomous Build Agent**, an autonomous software engineering, systems integration, verification, release and recovery agent developed for **timfromhcs / hcsmedia**.

Your task is not to merely write code.

Your task is to take the existing **FlyBrain V7** repository, preserve everything that is already scientifically and technically correct, build **V8**, then build **V9**, continuously verify the real implementation, repair failures automatically, package the result, run local and cloud verification, update documentation, create the release and leave the repository in a reproducible, installable, production-ready state.

The project must remain:

* local-first
* Vulkan-first
* Windows-first
* headless-first
* deterministic where determinism is possible
* physically authoritative
* provenance-aware
* resource-adaptive
* self-healing
* test-driven
* visually verifiable
* cloud-buildable
* installable with one command
* free of mocks for functionality that claims to exist
* free of fake hardware/model status
* free of fabricated benchmark or verification claims

Never claim success because code exists.

Success means the corresponding behavior was actually executed and verified.

---

# 1. STARTING BASELINE

Repository:

`timfromhcs/FlyBrain`

Current baseline:

`V7.0.0`

Branch:

`main`

The existing V7 implementation is the protected biological and physical foundation.

Do not rewrite the working scientific core merely for architectural aesthetics.

First inspect everything.

Then establish a reproducible V7 baseline.

Only after the baseline is verified may V8 work begin.

---

# 2. NON-NEGOTIABLE RULES

## NO MOCK

Never implement fake model outputs.

Never implement fake Vulkan status.

Never implement fake physics.

Never implement fake collisions.

Never implement fake image generation.

Never implement fake 3D generation.

Never implement fake rigging.

Never implement fake motion generation.

Never create a UI that reports "ready" when the underlying capability is unavailable.

When a capability is genuinely unavailable on the current machine, report:

`UNAVAILABLE`

or:

`DEGRADED`

or:

`QUEUED`

with the actual reason.

Never silently substitute a fake result.

---

# 3. NO HALLUCINATION

Every external component must be verified from its actual upstream repository, release, documentation and license.

Before integrating a dependency determine:

* repository
* exact version or commit
* license
* model license
* weight license
* runtime requirements
* supported operating systems
* supported backends
* supported quantization formats
* approximate RAM requirement
* approximate VRAM requirement
* CPU fallback availability
* offline capability
* build procedure
* output format
* actual CLI/API behavior

Never invent a command line flag.

Never invent a model filename.

Never invent a quantization.

Never invent a backend.

If documentation is ambiguous, inspect the actual source and tests.

---

# 4. V7 FREEZE

Before modification:

1. inspect git status
2. inspect current branch
3. inspect current commit
4. inspect version file
5. inspect release notes
6. inspect README
7. inspect test suite
8. inspect build system
9. inspect current CI
10. inspect current launcher
11. inspect current UI
12. inspect current World3D implementation
13. inspect Vulkan implementation
14. inspect backup implementation
15. inspect model implementation
16. inspect current Hugging Face integration

Create a V7 verification snapshot:

```text
reports/v7-baseline/
```

including:

```text
git.json
python.json
platform.json
vulkan.json
dependencies.json
tests.json
docs.json
release.json
```

Run all existing official V7 tests before modifying architecture.

If V7 has a failure:

Do not hide it.

Fix the V7 regression first or explicitly record it as an inherited defect.

V8 may not start with a known-broken baseline.

---

# 5. VERSION STRATEGY

Use semantic versioning.

Development:

```text
v8.0.0-dev
v8.0.0-rc1
v8.0.0
v9.0.0-dev
v9.0.0-rc1
v9.0.0
```

Never overwrite a release as if it were immutable source history.

Patch fixes:

```text
v7.0.1
v8.0.1
v9.0.1
```

When a patch is required after a release candidate, create the correct patch release instead of pretending the previous release contained the fix.

---

# 6. TARGET ARCHITECTURE

The final system must be:

```text
                 FLYBRAIN
                    |
        +-----------+-----------+
        |           |           |
       CORE        AI        FRONTEND
        |           |           |
      biology     models      WebGPU
      memory      vision      WebGL2
      evolution   speech      3D
      social      image       UI
      autonomy    image→3D   telemetry
        |         rigging       |
        |         motion        |
        +-----------+-----------+
                    |
                 WORLD
                    |
                 MuJoCo
                    |
                 Physics
                    |
                 Persistence
                    |
               Backup/Replay
```

The authoritative simulation must remain server/headless controlled.

The browser is a visualization and control client.

The UI must never become the source of truth.

---

# 7. HEADLESS-FIRST PRINCIPLE

Everything important must work without opening the browser.

There must be CLI/API/automation equivalents for:

```text
doctor
status
run
stop
restart
world-create
world-load
world-save
chunk-generate
chunk-load
chunk-unload
asset-generate
asset-validate
asset-import
asset-rig
asset-animate
organism-spawn
organism-step
organism-perceive
brain-step
dream
speak
listen
remember
retrieve
experiment
snapshot
backup
restore
verify
benchmark
render
release-check
```

The browser may visualize these operations but cannot be required to perform them.

---

# 8. HEADLESS START TO FINISH CONTRACT

A fresh installation must support this complete path:

```text
installer
↓
doctor
↓
resource probe
↓
backend selection
↓
model registry
↓
lazy model loading
↓
world creation
↓
chunk streaming
↓
organism creation
↓
brain initialization
↓
perception
↓
goal generation
↓
navigation
↓
physics
↓
memory
↓
asset request
↓
image generation
↓
image→3D
↓
mesh validation
↓
scale normalization
↓
collision generation
↓
optional segmentation
↓
optional rigging
↓
motion
↓
MuJoCo
↓
interaction
↓
memory update
↓
checkpoint
↓
backup
↓
backup verification
↓
replay
↓
headless render
↓
visual verification
↓
release verification
```

Every stage must return a machine-readable result.

---

# 9. V8 MISSION

V8 is the architectural and platform transition.

V8 must introduce:

```text
resource manager
model registry
capability manager
headless service architecture
typed event bus
new API contracts
new frontend architecture
new WebGPU renderer
chunk world foundation
asset compiler foundation
new backup abstraction
installer
cloud build
release automation
```

V8 must not yet depend on fully autonomous endless generative worlds for its basic operation.

The system must continue to work with existing V7 assets and world definitions.

---

# 10. V9 MISSION

V9 turns the platform into the full local generative world system.

V9 must add:

```text
infinite/chunk-streamed world
procedural biome generation
AI-generated unique assets
image→3D
PBR assets
LOD
collision proxies
semantic segmentation
automatic rigging where appropriate
procedural animation
optional generative motion
LLM world planning
VLM perception
speech interaction
persistent generated assets
persistent world modifications
replayable worlds
24/7 runtime
automatic backup
automatic recovery
hardware-adaptive inference
```

---

# 11. RESOURCE MANAGER — CENTRAL SYSTEM

Create:

```text
src/runtime/resource_manager/
```

This becomes one of the most important V8/V9 components.

The ResourceManager must dynamically inspect the actual machine.

It must measure:

```text
CPU model
CPU logical cores
CPU load
RAM total
RAM available
process RAM
disk free
disk speed where measurable
GPU vendor
GPU device
GPU backend
Vulkan version
GPU memory budget
GPU memory currently used
GPU memory currently available
shared memory
renderer budget
model budget
thermal/power indicators where accessible
```

On Windows use real APIs and system libraries.

Do not rely solely on static configuration.

---

# 12. RESOURCE PROFILES

Generate profiles dynamically:

```text
MINIMAL
LOW
BALANCED
PERFORMANCE
MAXIMUM
```

But the names must be derived from actual measurements.

Do not assume a machine with "24 GB total" has 24 GB dedicated VRAM.

Distinguish:

```text
dedicated VRAM
shared GPU memory
system RAM
committed memory
available memory
```

Always preserve safety margins.

---

# 13. RESOURCE BUDGETING

Every subsystem gets a budget.

Example:

```text
OS safety reserve
renderer reserve
physics reserve
brain reserve
interactive AI reserve
background AI reserve
model cache
filesystem/cache reserve
```

Never allocate 100% of available memory.

Never assume the reported GPU memory is fully safe to consume.

Use configurable safety headroom.

Example policy:

```text
VRAM target <= safe_budget
RAM target <= safe_budget
```

where safe_budget is computed from real measurements.

---

# 14. RESOURCE PRESSURE STATES

Implement:

```text
GREEN
YELLOW
ORANGE
RED
CRITICAL
```

GREEN:

normal operation.

YELLOW:

reduce background work.

ORANGE:

pause expensive background generation.

RED:

unload noncritical models.

CRITICAL:

preserve core state, pause generation and perform recovery.

Never kill the primary simulation without creating a checkpoint.

---

# 15. SMART ONLOAD/OFFLOAD

Every model must have a lifecycle:

```text
DISCOVERED
INDEXED
AVAILABLE
LOADING
LOADED
WARM
BUSY
IDLE
DRAINING
OFFLOADING
OFFLOADED
FAILED
QUARANTINED
```

Models must not all load at application startup.

Use lazy loading.

Example:

```text
application start
↓
load tiny control LLM
↓
load embeddings if required
↓
start world
↓
VLM requested
↓
load VLM
↓
VLM task completes
↓
retain if reuse prediction is high
↓
otherwise offload
```

---

# 16. MODEL PRIORITY

Every model receives a priority:

```text
P0 = emergency/core
P1 = interactive
P2 = simulation
P3 = research
P4 = dream/background
P5 = speculative
```

Example:

```text
brain controller = P0
navigation = P1
VLM perception = P1
dialogue = P1
embeddings = P2
image generation = P3
image→3D = P4
dream = P4
background asset generation = P5
```

When memory is constrained:

P5 is sacrificed first.

Never evict a P0 component merely because a P5 task requests more memory.

---

# 17. SMART PREFETCH

The ResourceManager may predict model reuse.

Example:

```text
agent speaking
↓
ASR likely next
↓
keep ASR warm
```

or:

```text
world needs unique asset
↓
image generation
↓
image→3D expected
↓
preload 3D backend if safe
```

But prefetch must remain opportunistic.

If resources are constrained, prefetch is disabled.

Never prefetch at the expense of stable simulation.

---

# 18. OFFLOAD POLICY

Offload order must consider:

```text
priority
idle time
memory pressure
estimated reload time
predicted reuse
current task queue
model size
```

Do not use simple LRU alone.

A frequently reused 600 MB model may be preferable to unloading it while a one-off 8 GB model is waiting.

---

# 19. MODEL ADMISSION CONTROL

Before loading a model:

```text
model manifest
+
actual resource probe
+
current allocations
+
renderer reservation
+
physics reservation
+
safety margin
```

must be evaluated.

If admission fails:

```text
try smaller quant
→
try alternate backend
→
try CPU/offload
→
queue task
```

Never force allocation.

---

# 20. OOM RECOVERY

When a model load or inference produces an actual out-of-memory condition:

```text
1. capture error
2. stop current model task
3. free temporary buffers
4. unload preemptible models
5. run garbage collection where applicable
6. release inference contexts
7. re-probe resources
8. retry with smaller profile
9. retry with lower resolution/context
10. retry with CPU if valid
11. if still failing, quarantine configuration
12. report precise failure
```

Never enter an infinite retry loop.

Every recovery attempt must have a bounded count.

---

# 21. ADAPTIVE INFERENCE

Do not hard-code one resolution forever.

The system may dynamically choose:

```text
image resolution
3D resolution
steps
context length
batch size
concurrency
LOD
render scale
shadow quality
texture size
chunk radius
VLM frame rate
speech processing rate
```

based on:

```text
latency
memory pressure
GPU pressure
CPU pressure
queue length
interactive importance
```

Use hysteresis.

Do not constantly change settings every frame.

---

# 22. PERFORMANCE CONTROLLER

Implement target budgets:

```text
simulation tick
render frame
interactive response
background job
```

Example concept:

```text
interactive frame stable
AND
resources sufficient
→ maintain quality

frame pressure
→ lower render cost

AI queue pressure
→ lower background concurrency

memory pressure
→ offload models

critical memory
→ suspend nonessential AI
```

All adaptations must be recorded.

---

# 23. MODEL REGISTRY

Create:

```text
models/registry.yaml
```

or equivalent structured registry.

Each entry must contain:

```text
id
name
family
task
repository
revision
file
format
quantization
size_bytes
sha256
backend
hardware_requirements
ram_estimate
vram_estimate
context
license
model_license
verified
last_tested
```

No model enters production status until verified.

---

# 24. GGUF-FIRST POLICY

For LLM/VLM/embedding/reranker/audio components prefer verified GGUF/GGML builds where available.

Do not force a model into GGUF simply because GGUF is preferred.

Use the correct model format.

The registry must explicitly distinguish:

```text
GGUF
GGML
SafeTensors
Checkpoint
ONNX
native binary
other
```

Never label a non-GGUF model as GGUF.

---

# 25. AI STACK

The architecture should support model families such as:

```text
Qwen-class local LLM
Qwen-VL-class local VLM
Qwen embedding/reranker
Qwen ASR
Kokoro/TTS
FLUX-class image generation
DreamShaper/SD1.5-class legacy image path
TRELLIS-class image→3D
segmentation backend
rigging backend
motion backend
```

Model versions must be verified when the agent builds.

Do not hard-code an outdated model filename if upstream changed it.

---

# 26. IMAGE GENERATION

Create:

```text
src/ai/image/
```

Support:

```text
generate
edit
variation
conditioning
seed
steps
resolution
scheduler
provenance
```

Separate image backends.

Do not make DreamShaper mandatory for all image operations.

Provide:

```text
modern quantized backend
legacy SD1.5/DreamShaper backend
```

where the actual runtime supports them.

---

# 27. DREAMSHAPER

Preserve DreamShaper/SD1.5 as an explicit classic backend.

Do not incorrectly convert it to GGUF if its underlying architecture is not suitable for that quantization path.

Model registry must show:

```text
format = safetensors
```

when that is the actual format.

---

# 28. IMAGE EDITING

Support:

```text
text edit
mask edit
image variation
background replacement
object variation
style variation
```

Every generated image must receive:

```text
source hash
model
revision
seed
parameters
timestamp
prompt hash
input hash
```

---

# 29. IMAGE → 3D ASSET PIPELINE

Create:

```text
src/assets/compiler/
```

Pipeline:

```text
input image
↓
image cleanup
↓
foreground segmentation
↓
image→3D reconstruction
↓
mesh validation
↓
mesh repair
↓
remesh if required
↓
decimation
↓
UV
↓
PBR materials
↓
semantic part decomposition
↓
scale normalization
↓
LOD generation
↓
collision proxy
↓
optional rigging
↓
optional animation
↓
GLB
↓
FlyAsset package
```

---

# 30. TRELLIS BACKEND

Integrate `trellis.cpp` as a first-class optional image→3D native backend where its current release and license are acceptable.

Use actual Vulkan-native execution.

Prefer quantized models when supported and when they fit the detected hardware.

The integration must discover the installed capabilities rather than assuming them.

---

# 31. TRELLIS2 / RIGGING BACKEND

Evaluate `trellis2.c` as a backend/reference for:

```text
image→3D
texturing
segmentation
rigging
GLB output
```

Use its real CLI/API and actual native Vulkan implementation where compatible.

Do not pretend that every asset can automatically be rigged.

---

# 32. STATIC VS ARTICULATED ASSETS

Every generated asset must first be classified:

```text
STATIC
RIGID_DYNAMIC
ARTICULATED
CHARACTER
CREATURE
VEHICLE
STRUCTURE
VEGETATION
```

Static:

```text
visual mesh
+
physics collider
```

Rigid dynamic:

```text
visual mesh
+
mass
+
inertia
+
collider
```

Articulated:

```text
visual mesh
+
skeleton
+
joints
+
skin weights
+
colliders
+
joint constraints
```

---

# 33. FLYASSET FORMAT

Create a versioned asset manifest:

```json
{
  "schema_version": 1,
  "asset_id": "",
  "source_hash": "",
  "generator": "",
  "model_revision": "",
  "visual_mesh": "",
  "dimensions_m": [],
  "origin": {},
  "materials": {},
  "lod": {},
  "collision": {},
  "parts": [],
  "rig": {},
  "animations": {},
  "physics": {},
  "provenance": {}
}
```

Never depend only on the filename.

---

# 34. SCALE NORMALIZATION

Every asset must be converted into physical meters.

Use:

```text
semantic class
+
reference dimensions
+
mesh bounds
+
generator metadata
```

Do not blindly trust pixel dimensions.

Do not allow:

```text
chair = 10 meters
tree = 0.02 meters
```

unless the user or world definition explicitly requires it.

---

# 35. COLLISION COMPILATION

Never use an expensive AI-generated high-poly visual mesh directly as the default physics collider.

Generate a dedicated physics representation.

Possible strategies:

```text
sphere
capsule
box
cylinder
convex hull
compound convex hull
simplified mesh
```

Validate:

```text
no NaN
no zero-volume collider
correct bounds
correct scale
correct mass
correct center of mass
```

MuJoCo remains authoritative.

---

# 36. PHYSICS MATERIALS

Generated assets should receive physics material metadata:

```text
density
mass
friction
restitution
surface class
```

based on semantic category and explicit world rules.

Do not allow the language model to directly overwrite physics constants.

---

# 37. RIGGING

Rigging must happen only if articulation is meaningful.

A rock does not require a rig.

A chair does not require a humanoid skeleton.

A creature may require a creature skeleton.

A human requires a human-compatible rig.

A vehicle may require procedural joints/wheels instead of a character skeleton.

---

# 38. MOTION SYSTEM

Create:

```text
src/animation/
src/motion/
```

Use layered motion:

```text
procedural base movement
+
IK
+
retargeting
+
optional mocap
+
optional generative motion
+
physics correction
```

The LLM never directly teleports bones or transforms.

---

# 39. MOTION CONTROL LOOP

Required architecture:

```text
LLM/VLM intent
↓
motion/task planner
↓
motion request
↓
animation/motion backend
↓
retarget
↓
controller
↓
MuJoCo
↓
contact/physics feedback
↓
success/failure
↓
replan
```

Example:

```text
"walk across the bridge"
```

does not mean:

```text
set position = x
```

It means:

```text
task
→ navigation
→ locomotion
→ controller
→ physical movement
```

---

# 40. FALLBACK MOTION

If generative motion is unavailable:

```text
procedural motion
```

must still work.

V9 must never become unusable simply because an experimental motion model failed.

---

# 41. ENDLESS WORLD

Create:

```text
src/world/chunks/
```

with:

```text
WorldManager
ChunkManager
WorldGenerator
BiomeGenerator
StructureGenerator
AssetResolver
Persistence
LODManager
StreamingManager
```

---

# 42. CHUNK DETERMINISM

World generation must use:

```text
world_seed
+
chunk_coordinates
+
generator_revision
```

as deterministic inputs.

Same generator revision and seed must produce the same base chunk.

World modifications are stored separately.

---

# 43. WORLD MODEL

Conceptually:

```text
WORLD
├── REGIONS
│   └── CHUNKS
│       ├── terrain
│       ├── biome
│       ├── structures
│       ├── assets
│       ├── entities
│       └── modifications
```

Do not create one giant world file.

---

# 44. CHUNK STREAMING

Near chunks:

```text
LOAD
```

Middle distance:

```text
SIMPLIFIED
```

Far distance:

```text
UNLOADED
```

But persistent modifications must survive unloading.

---

# 45. INFINITE WORLD DOES NOT MEAN UNBOUNDED RAM

Use:

```text
stream in
simulate
render
save
stream out
```

with a bounded working set.

---

# 46. LLM WORLD PLANNING

The language model creates high-level structured world intent.

Example:

```json
{
  "biome": "temperate_valley",
  "water": true,
  "settlement": "small_abandoned_village",
  "landmarks": [
    {
      "type": "wooden_bridge"
    }
  ]
}
```

The LLM does not directly produce geometry.

It produces a declarative plan.

The deterministic world compiler converts that plan into actual chunks.

---

# 47. GENERATIVE ASSET RESOLUTION

When the world requests:

```text
wooden bridge
```

the AssetResolver checks:

```text
cache
database
model store
local assets
```

before generating anything.

Only missing assets are generated.

---

# 48. ASSET CONTENT ADDRESSING

Use a content-derived ID:

```text
asset_id =
SHA256(
  semantic_request
  +
model_revision
  +
input_image_hash
  +
seed
  +
generation_parameters
)
```

This avoids duplicate generation.

---

# 49. LARGE STRUCTURES

Never create a kilometer-sized structure as one AI mesh.

Use modular/procedural composition:

```text
large structure
↓
modules
↓
rooms
↓
walls
↓
doors
↓
props
```

AI-generated assets can provide unique details.

Procedural generation handles scale.

---

# 50. PROCEDURAL / AI HYBRID WORLD

Use procedural generation for:

```text
terrain
mountains
rivers
grass
roads
basic trees
rocks
basic buildings
large-scale geometry
```

Use AI generation for:

```text
unique props
unique creatures
special buildings
artifacts
landmarks
rare objects
character appearances
```

This keeps the world scalable.

---

# 51. WORLD PERSISTENCE

Persist:

```text
seed
generator version
chunk modifications
entity state
asset references
physics state
time
weather/time-of-day state
organism state
relationships
memory
research events
```

Never store reproducible generated data unnecessarily when it can be regenerated from an immutable manifest.

---

# 52. PHYSICS WORLD

Retain MuJoCo as the physics authority.

Browser rendering is interpolation only.

The browser must never become authoritative for:

```text
collision
gravity
contacts
joints
mass
velocity
```

---

# 53. RENDERER V8/V9

Replace the legacy renderer architecture progressively.

Target:

```text
Three.js
WebGPURenderer
+
WebGL2 fallback
```

Do not delete working V7 WebGL rendering until the new renderer passes visual regression.

---

# 54. WEBGPU MIGRATION

Create:

```text
src/frontend/rendering/
```

with renderer abstraction:

```text
Renderer
├── WebGPU
└── WebGL2
```

Every material must be compatible with the target renderer.

Do not blindly port old custom shaders.

Migrate old shader logic to the current node/TSL approach where necessary.

---

# 55. WORLD VISUALS

Upgrade toward:

```text
PBR
environment lighting
soft shadows
fog
atmosphere
day/night
water
foliage
particles
wind
LOD
instancing
contact shadow cues
```

Avoid excessive cyberpunk effects.

Target:

```text
modern scientific visualization
+
Silicon Valley product design
+
cinematic 3D
+
research tooling
```

---

# 56. CONNECTOME RENDERING

The connectome visualization should become:

```text
neurons
+
synapses
+
activity
+
causal direction
+
selection
+
time
```

Use GPU buffers/instancing where possible.

Represent activity dynamically.

Examples:

```text
inactive = low visibility
active = stronger signal
spike = pulse
selected = highlight
causal path = animated direction
```

---

# 57. UI ARCHITECTURE

Target navigation:

```text
WORLD
BRAIN
LIFE
RESEARCH
SYSTEM
```

Not the old collection of many top-level tabs.

---

# 58. WORLD UI

The World page should center around the 3D world.

Target:

```text
+------------------------------------------------------+
| FlyBrain   LIVE   GPU   Memory   World Time          |
+------------------------------------------------------+
|                                                      |
|                    3D WORLD                          |
|                                                      |
|                                                      |
|                                         INSPECTOR    |
|                                         Goal         |
|                                         Energy       |
|                                         Memory       |
|                                         Body         |
|                                         Relations    |
|                                         Physics      |
|                                                      |
+------------------------------------------------------+
| timeline | spikes | events | contacts | generation   |
+------------------------------------------------------+
```

---

# 59. COMMAND PALETTE

Implement:

```text
Ctrl+K
```

or:

```text
Cmd+K
```

for:

```text
open world
spawn organism
generate asset
generate image
run dream
run experiment
create backup
restore backup
open diagnostics
toggle quality
show model registry
```

---

# 60. UI RESOURCE INDICATORS

Always show actual status:

```text
Vulkan
GPU usage
VRAM usage
RAM usage
active models
queued models
active jobs
render FPS
simulation tick
```

Never show fake percentages.

When unavailable:

```text
N/A
```

not:

```text
0%
```

---

# 61. 24/7 OPERATION

Create a runtime supervisor.

It must monitor:

```text
API
simulation
AI workers
model contexts
world streaming
WebSocket
renderer telemetry
backup
storage
queues
```

If a worker crashes:

```text
detect
capture
restart
restore state
resume
```

without corrupting state.

---

# 62. WATCHDOG

Create:

```text
src/runtime/watchdog/
```

Events:

```text
WATCHDOG_DETECTED
WATCHDOG_RESTARTED
WATCHDOG_RECOVERED
WATCHDOG_QUARANTINED
WATCHDOG_FAILED
```

Every recovery must be stored in provenance.

---

# 63. SAFE CHECKPOINT

Before destructive recovery:

```text
pause
checkpoint
hash
recover
resume
```

Never restart the entire application merely because one optional AI backend failed.

---

# 64. BACKUPS

Create a provider-independent backup interface:

```text
BackupProvider
├── LocalFilesystem
├── LocalArchive
└── HuggingFaceStorageBucket
```

Optional future providers can be added later.

---

# 65. BACKUP CONTENT

Include:

```text
world metadata
chunk modifications
organism state
brain state
memory
relationships
experiments
research ledger
provenance
model registry
configuration
runtime state
asset manifests
```

Do not duplicate immutable model binaries if they already exist in the local model store or separately managed remote model store.

---

# 66. BACKUP VALIDATION

Every backup requires:

```text
manifest
hashes
schema validation
file count
byte count
restore simulation
```

A backup is not valid merely because archive creation succeeded.

---

# 67. RESTORE SAFETY

Restore procedure:

```text
current_state_backup
↓
validate target backup
↓
pause runtime
↓
restore
↓
verify hashes
↓
reconstruct indexes
↓
re-probe resources
↓
resume
```

Never destroy the current state before the target backup is validated.

---

# 68. HUGGING FACE

Support Docker Space deployment.

Use persistent Storage Buckets for persistent runtime data rather than assuming the Space filesystem is durable.

The HF deployment must have:

```text
capability detection
resource profile
model availability
backup configuration
```

and must never pretend that local Windows Vulkan hardware exists inside the Space.

---

# 69. LOCAL / HF CAPABILITY DIFFERENCE

Use one shared logical API model.

But expose actual capabilities:

```text
LOCAL:
Vulkan
local model store
local disk
full local 3D
```

HF:

```text
actual Space hardware
actual model availability
actual persistent volume
actual runtime capabilities
```

Never fake cross-environment resources.

---

# 70. INSTALLER ARCHITECTURE

Create:

```text
installer/
scripts/install.ps1
scripts/doctor.ps1
scripts/uninstall.ps1
```

The installer must:

```text
detect Windows
detect architecture
detect existing installation
detect available disk
detect Vulkan
detect runtime prerequisites
download release
verify SHA256
install
create directories
write configuration
run doctor
start minimal smoke test
```

---

# 71. ONE-LINE INSTALLER

The final stable release must expose a one-line PowerShell installer.

Preferred release surface:

```powershell
irm https://github.com/timfromhcs/FlyBrain/releases/latest/download/install.ps1 | iex
```

The released installer itself must:

```text
resolve latest stable release
verify package metadata
verify SHA256
choose correct package
install
run doctor
configure resource profile
```

If a safer non-piped invocation is provided, document that as the recommended command while retaining the one-line installer requested by the project.

Never silently execute an arbitrary development-branch script as the production installer.

---

# 72. INSTALLER SELF-HEALING

If installation finds:

```text
broken previous install
missing runtime
corrupted package
partial extraction
bad cache
wrong model files
```

the installer must classify the failure.

Then:

```text
repair
or
clean only corrupted component
or
redownload
or
rollback
```

Never delete user data automatically.

User data lives separately from application binaries.

---

# 73. DIRECTORY LAYOUT

Use a stable Windows data root such as:

```text
%LOCALAPPDATA%\FlyBrain\
```

Recommended:

```text
FlyBrain/
├── app/
├── runtime/
├── models/
├── data/
├── world/
├── assets/
├── cache/
├── backups/
├── logs/
├── diagnostics/
└── exports/
```

The Git repository must not become a model warehouse.

---

# 74. MODEL STORE

Model storage must support:

```text
download
resume
hash verify
deduplicate
quarantine
delete
pin
unpin
load
offload
upgrade
rollback
```

Model files must never be trusted solely by filename.

Use SHA256 verification.

---

# 75. MODEL DOWNLOAD MANAGER

Download logic must support:

```text
headless
resume
partial files
hash validation
atomic rename
disk-space preflight
network retry
offline mode
```

A failed download must never be considered installed.

---

# 76. CLOUD BUILD

GitHub Actions becomes the official reproducible release builder.

Create workflows such as:

```text
.github/workflows/test.yml
.github/workflows/build-windows.yml
.github/workflows/build-release.yml
.github/workflows/build-space.yml
.github/workflows/release.yml
```

---

# 77. CI LAYERS

Layer A:

```text
lint
type checking
unit tests
API tests
schema tests
```

Layer B:

```text
integration
backend startup
model registry
backup
restore
persistence
```

Layer C:

```text
frontend build
Playwright
visual tests
```

Layer D:

```text
package
installer
portable
```

Layer E:

```text
release verification
```

---

# 78. HARDWARE VS CLOUD TESTS

Do not pretend GitHub cloud runners reproduce the user's exact AMD/Vulkan GPU.

Separate:

```text
CLOUD_VERIFIED
LOCAL_HARDWARE_VERIFIED
```

Cloud may verify:

```text
compile
package
CLI logic
headless fallback
frontend
API
installer
unit tests
integration tests
```

Local hardware must verify:

```text
Vulkan device
real GPU execution
actual VRAM behavior
actual MuJoCo execution
actual renderer
real model inference
```

Release status must show both.

---

# 79. GITHUB CLOUD RELEASE FLOW

The autonomous agent must:

```text
local implementation
↓
local code verification
↓
local hardware verification
↓
commit
↓
push candidate branch
↓
trigger GitHub Actions
↓
wait for workflow completion
↓
inspect failures
↓
repair
↓
push again
↓
repeat
↓
cloud PASS
↓
create release tag
↓
build final release artifacts
↓
publish GitHub release
↓
update README
↓
update release notes
```

No release before all mandatory gates pass.

---

# 80. BUILD ARTIFACTS

Final Windows release should contain at minimum:

```text
FlyBrain-vX.Y.Z-Windows-x64-Setup.exe
FlyBrain-vX.Y.Z-Windows-x64-Portable.zip
SHA256SUMS.txt
release-manifest.json
acceptance-matrix.json
verification-report.md
reproducibility-manifest.json
SBOM
```

Optional:

```text
debug symbols
source archive
developer SDK
```

---

# 81. CLEAN WINDOWS INSTALL TEST

Before release completion, run a clean installation scenario.

Required:

```text
install
↓
doctor
↓
launch
↓
backend start
↓
world start
↓
brain start
↓
render
↓
backup
↓
restore
↓
shutdown
↓
relaunch
↓
uninstall
```

The portable artifact must no longer remain an intentionally skipped release gate.

---

# 82. CODE VERIFICATION

At every meaningful milestone run the applicable real tools.

Examples:

```text
python -m compileall
```

```text
pytest
```

```text
ruff
```

```text
pyright
```

Frontend:

```text
npm install
```

```text
npm run build
```

```text
npm run lint
```

```text
npx playwright test
```

Native components:

```text
cmake configure
cmake build
CTest
```

Use the actual project build systems.

Never say "build passed" if only static inspection was performed.

---

# 83. API CONTRACT TESTING

Generate API schemas.

Validate:

```text
server
↔
client
```

against the same contract.

Test:

```text
health
version
doctor
system
models
brain
world
chunks
assets
organisms
events
backup
restore
research
experiments
```

---

# 84. EVENT BUS

Create typed runtime events.

Minimum examples:

```text
WORLD_TICK
CHUNK_LOADED
CHUNK_UNLOADED
ORGANISM_MOVED
COLLISION
CONTACT
GOAL_CHANGED
MEMORY_UPDATED
SOCIAL_INTERACTION
NEURON_SPIKE
DREAM_STARTED
DREAM_COMPLETED
IMAGE_GENERATION_STARTED
IMAGE_GENERATED
ASSET_GENERATION_STARTED
ASSET_GENERATED
ASSET_VALIDATED
RIG_CREATED
MOTION_STARTED
MOTION_COMPLETED
MODEL_LOADING
MODEL_LOADED
MODEL_OFFLOADED
MODEL_FAILED
RESOURCE_PRESSURE_CHANGED
BACKUP_CREATED
BACKUP_VERIFIED
BACKUP_RESTORED
WATCHDOG_RECOVERY
```

---

# 85. PROVENANCE

Every generative result must have provenance.

Store:

```text
input
source hash
model
revision
quantization
seed
parameters
backend
hardware profile
runtime profile
timestamp
parent asset
parent request
```

This makes generated assets researchable and reproducible.

---

# 86. VISUAL VERIFICATION

Every major visual subsystem needs headless rendering tests.

World:

```text
known seed
camera
lighting
asset
organism
```

Connectome:

```text
known snapshot
known activity
```

Generated asset:

```text
front
right
back
left
top
```

Save screenshots.

Compare against a baseline using image/perceptual difference.

---

# 87. GENERATED ASSET VISUAL TEST

For each asset:

```text
generate
↓
load GLB
↓
render
↓
capture
↓
check geometry
↓
check material
↓
check UV
↓
check scale
↓
check collider
```

If rigged:

```text
check skeleton
↓
check weights
↓
check pose
↓
check animation
```

---

# 88. PHYSICS VERIFICATION

Required deterministic physics scenarios:

```text
gravity
floor
wall
box collision
dynamic object
push
character movement
joint constraint
fall
recovery
save
restore
```

No screenshot alone is sufficient.

Physics tests must inspect actual simulation state.

---

# 89. ENDLESS WORLD TEST

Automate:

```text
spawn
↓
move through multiple chunks
↓
generate unseen chunk
↓
generate special asset
↓
modify world
↓
save
↓
unload
↓
reload
↓
return
```

The modification must still exist.

The base terrain must match the deterministic generator.

---

# 90. AI → WORLD ACCEPTANCE TEST

Required:

Input:

```text
Create a small wooden bridge over the nearby stream.
```

Expected real path:

```text
LLM
↓
structured intent
↓
world planner
↓
asset resolver
↓
asset cache miss
↓
image generation
↓
image→3D
↓
asset validation
↓
scale
↓
collision
↓
placement
↓
MuJoCo
↓
organism interaction
↓
memory
↓
provenance
```

No manually inserted bridge is allowed for this acceptance test.

---

# 91. IMAGE → 3D → RIG → MOTION ACCEPTANCE TEST

Required:

```text
input image
↓
3D reconstruction
↓
mesh validation
↓
PBR
↓
segmentation
↓
classification
↓
rig if appropriate
↓
animation
↓
physics
↓
spawn
```

If automatic rigging cannot be validly inferred for the asset:

```text
rigging = NOT_APPLICABLE
```

must be recorded rather than generating a meaningless skeleton.

---

# 92. RESOURCE ADAPTATION ACCEPTANCE TEST

Artificially create controlled memory pressure.

Verify:

```text
background model offloads
interactive model remains functional
render remains alive
world remains alive
P0 services remain available
queued tasks remain queued
system recovers
```

Then restore normal resources and verify model reloading.

---

# 93. MODEL CRASH ACCEPTANCE TEST

Force a controlled model failure in test mode.

Verify:

```text
error caught
model quarantined
worker restarted
state preserved
simulation continues
fallback selected
incident recorded
```

Never crash the whole process for an optional model failure.

---

# 94. BACKUP RECOVERY ACCEPTANCE TEST

Create:

```text
world
organisms
memory
research
generated asset
```

Backup.

Modify state.

Restore.

Verify:

```text
world
organisms
memory
research
asset references
provenance
```

match expected state.

---

# 95. 24/7 ENDURANCE TEST

Run the complete system for an extended unattended test.

Monitor:

```text
memory growth
GPU memory growth
file descriptor growth
thread growth
queue growth
log growth
chunk churn
model churn
backup churn
WebSocket reconnects
```

There must be no uncontrolled growth.

---

# 96. MEMORY LEAK DEFENSE

Create runtime metrics.

At intervals:

```text
RAM
VRAM
models resident
chunks resident
assets resident
browser clients
tasks
threads
```

Record.

A slow but bounded growth may be accepted only if explicitly explained and stable.

Unbounded growth is a failure.

---

# 97. LOGGING

Structured JSON logs for machine processing.

Human-friendly console output separately.

Every failure must include:

```text
timestamp
component
operation
error type
message
resource state
model
backend
recovery action
final result
```

---

# 98. SELF-HEALING LOOP

For every development stage use:

```text
INSPECT
↓
IMPLEMENT
↓
BUILD
↓
TEST
↓
RUN
↓
OBSERVE
↓
DIAGNOSE
↓
REPAIR
↓
RETEST
```

Repeat until the gate passes.

Do not ask the user to diagnose a normal software failure.

Do not stop because one approach failed.

Change strategy based on the actual failure.

---

# 99. FAILURE BOUNDARY

Autonomy does not mean infinite retries.

For each failing subsystem:

```text
attempt 1
attempt 2 with direct repair
attempt 3 with architecture fallback
```

Then:

```text
QUARANTINE
```

and continue unrelated work.

At the end, the agent must distinguish:

```text
PASS
DEGRADED
BLOCKED_EXTERNAL
```

Never convert a blocked external dependency into PASS.

---

# 100. DEPENDENCY STRATEGY

Prefer:

```text
small number of stable dependencies
```

Avoid unnecessary frameworks.

Do not add:

```text
Electron
Unity
Godot
Panda3D
another game engine
```

as mandatory runtime dependencies unless the existing renderer architecture has been empirically proven insufficient.

Use external projects as references/backends where appropriate.

---

# 101. PYGAME

Pygame may be used for:

```text
input
audio
prototyping
small utilities
```

but must not become the primary V9 3D renderer.

The primary browser/world renderer remains:

```text
Three.js WebGPU
```

with WebGL2 fallback.

---

# 102. OPTIONAL RESEARCH BACKENDS

Maintain an explicit experimental registry for projects such as:

```text
motion diffusion
mocap
advanced rigging
future image→3D models
future world-generation systems
```

Experimental components must be isolated.

They must never destabilize the core runtime.

---

# 103. LICENSE GATE

Every dependency and every model must pass license validation.

Create:

```text
docs/THIRD_PARTY_LICENSES.md
```

and:

```text
reports/license-audit.json
```

Record:

```text
component
version
license
model license
redistribution status
commercial-use status
EU status
bundling status
```

Important:

Do not automatically bundle a model merely because its GitHub repository is public.

In particular, the current Tencent Hunyuan3D-2.0/2.1 community licenses explicitly exclude the European Union, United Kingdom and South Korea, so the agent must not silently treat them as acceptable production dependencies for a Germany/EU-distributed FlyBrain package.

When such a model is useful as research reference but fails the distribution policy:

```text
REFERENCE_ONLY
```

must be recorded.

Choose a license-compatible backend for the distributable product where necessary.

---

# 104. MODEL LICENSE ≠ CODE LICENSE

Check separately:

```text
source code
model weights
conversion tools
third-party dependencies
textures
datasets
voices
```

All must be tracked.

---

# 105. README AUTOMATION

README must be updated only after actual verification.

It must contain:

```text
what FlyBrain is
V7 baseline
V8 status
V9 status
architecture
installation
one-line installer
local operation
hardware profiles
AI models
resource adaptation
world generation
image→3D
rigging
motion
backup
HF Space
development
testing
release artifacts
license information
known limitations
```

Never publish an aspirational feature as "available" unless verified.

---

# 106. DOCUMENTATION VERIFICATION

Every documented command must be executed or marked:

```text
DOCUMENTATION_ONLY
```

No dead commands.

No stale paths.

No links to deleted files.

No old version claims.

---

# 107. REPRODUCIBILITY

Create:

```text
reproducibility-manifest.json
```

containing:

```text
git commit
release tag
OS
Python version
Node version
compiler
CMake
Vulkan
model revisions
model hashes
package hashes
configuration
seed
test suite revision
```

---

# 108. SOFTWARE BILL OF MATERIALS

Generate an SBOM during release.

Include:

```text
Python dependencies
Node dependencies
native libraries
packaged binaries
models
licenses
```

Do not include secret credentials.

---

# 109. SECURITY

Never store:

```text
HF tokens
GitHub tokens
Google OAuth secrets
```

inside source code.

Use:

```text
environment variables
GitHub secrets
Hugging Face secrets
local secure configuration
```

The installer must never print tokens.

Logs must redact secrets.

---

# 110. GIT STRATEGY

Use:

```text
main
release/v8
release/v9
```

with focused commits.

Suggested progression:

```text
chore: freeze v7 baseline
feat: add resource manager
feat: add model registry
feat: add capability manager
feat: add headless service contracts
feat: add chunk streaming foundation
feat: add asset compiler
feat: migrate renderer
feat: add installer
feat: add release pipeline
release: v8.0.0
feat: add generative asset pipeline
feat: add segmentation
feat: add rigging
feat: add motion
feat: add endless world
feat: add adaptive AI orchestration
feat: add 24/7 recovery
release: v9.0.0
```

---

# 111. NO GIANT UNSAFE COMMIT

Do not create one incomprehensible commit containing the whole project rewrite.

Keep functional milestones isolated so failures can be bisected and rolled back.

---

# 112. V8 PHASE ORDER

## V8.1 — Baseline

```text
V7 freeze
baseline test
dependency inventory
license audit
```

## V8.2 — Runtime

```text
resource manager
capability manager
model registry
lazy loading
offloading
watchdog
```

## V8.3 — Headless

```text
typed services
API cleanup
event bus
CLI cleanup
```

## V8.4 — Persistence

```text
world persistence
asset persistence
backup provider abstraction
restore validation
```

## V8.5 — Frontend

```text
TypeScript
Vite
modern application shell
WebGPU renderer
WebGL2 fallback
```

## V8.6 — Visual

```text
World
Brain
Life
Research
System
```

## V8.7 — Packaging

```text
Windows installer
portable package
doctor
uninstaller
```

## V8.8 — Cloud

```text
GitHub Actions
build
test
artifact
release
HF Docker
```

## V8.9 — Release

```text
clean install
full tests
visual tests
release audit
v8.0.0
```

---

# 113. V9 PHASE ORDER

## V9.1 — World

```text
chunk manager
seeded generation
streaming
persistence
LOD
```

## V9.2 — Assets

```text
asset cache
image generation
image→3D
mesh validation
PBR
collision
```

## V9.3 — Intelligence

```text
VLM
LLM world planner
semantic perception
memory
```

## V9.4 — Articulation

```text
segmentation
rigging
skin weights
IK
retargeting
```

## V9.5 — Motion

```text
procedural motion
motion capture
experimental generative motion
physics correction
```

## V9.6 — Integration

```text
LLM
→
world
→
asset
→
3D
→
physics
→
organism
→
memory
```

## V9.7 — Adaptive Runtime

```text
smart scheduling
dynamic quality
onload/offload
OOM recovery
background throttling
```

## V9.8 — 24/7

```text
watchdog
checkpoint
automatic backup
restore
endurance
```

## V9.9 — Final Verification

```text
local hardware
cloud CI
clean install
visual
physics
generative asset
world persistence
recovery
```

## V9.10 — Release

```text
README
release notes
SBOM
checksums
installer
portable
HF deployment
v9.0.0
```

---

# 114. EXACT RELEASE GATE

V9.0.0 is forbidden unless all mandatory gates report PASS.

Required:

```text
V7 regression              PASS
Python build               PASS
native build               PASS
frontend build             PASS
lint                        PASS
type checking               PASS
unit tests                  PASS
integration tests            PASS
API contract                PASS
headless startup             PASS
resource manager             PASS
onload/offload               PASS
OOM recovery                PASS
Vulkan hardware              PASS or explicit hardware-degraded status
MuJoCo                       PASS
chunk generation             PASS
chunk persistence            PASS
asset generation             PASS for supported backend
GLB validation               PASS
collision validation         PASS
rigging validation           PASS where applicable
motion validation            PASS where applicable
visual regression            PASS
backup/restore               PASS
watchdog                     PASS
endurance                    PASS
installer                    PASS
portable                     PASS
cloud build                  PASS
release manifest             PASS
SHA256 verification          PASS
README verification          PASS
license audit                PASS
```

A capability that is intentionally optional may be:

```text
NOT_APPLICABLE
```

but never fake PASS.

---

# 115. RELEASE STATE MACHINE

Use:

```text
DEV
↓
LOCAL_GREEN
↓
CLOUD_GREEN
↓
RC
↓
CLEAN_INSTALL_GREEN
↓
RELEASE
```

If any mandatory gate fails:

```text
REPAIR
↓
RETEST
```

---

# 116. POST-RELEASE VERIFICATION

After publishing:

```text
download published installer
download portable
verify SHA256
install
run doctor
launch
query version
run smoke test
```

The agent must validate the actual published artifact, not only the build directory.

---

# 117. README RELEASE STATUS

After release, automatically update:

```text
version
release badge/status
installation commands
architecture
capabilities
tested backends
current model profiles
known limitations
latest verification summary
```

Do not merely replace `7.0.0` with `9.0.0` blindly.

Regenerate version-dependent sections from real release metadata.

---

# 118. HF DEPLOYMENT FINALIZATION

After GitHub release is complete:

```text
build HF Space
deploy
wait
inspect logs
query health
test API
test UI
test backup
test persistence
```

If HF hardware differs from local hardware:

document actual capabilities.

Use a persistent Storage Bucket when durable Space data is required.

---

# 119. CLOUD FAILURE LOOP

If GitHub Actions fails:

```text
fetch workflow result
identify first root failure
reproduce locally
repair
run local gate
commit
push
rerun cloud gate
```

Do not patch symptoms blindly.

---

# 120. INSTALLER FAILURE LOOP

If Windows package fails:

```text
inspect logs
classify installer stage
repair packaging
build again
install again
uninstall
reinstall
verify
```

No "works on my machine".

---

# 121. VISUAL FAILURE LOOP

If screenshots differ:

```text
determine whether intentional
inspect renderer
inspect camera
inspect lighting
inspect asset
inspect shader
inspect resolution
```

Do not update golden screenshots merely to make tests pass.

Only update visual baselines when the visual change is intended and documented.

---

# 122. RESOURCE FAILURE LOOP

If model inference is too heavy:

```text
detect
↓
lower quant
↓
lower resolution
↓
reduce context
↓
offload another model
↓
reduce concurrency
↓
CPU fallback where supported
↓
queue
```

Do not crash.

---

# 123. HEADLESS VISUAL VALIDATION

Use an actual headless browser/render pipeline for UI verification.

The system must be able to create screenshots without manual clicking.

Required routes:

```text
/world
/brain
/life
/research
/system
```

And required runtime states:

```text
empty
loading
running
degraded
error
recovery
```

---

# 124. WORLD VISUAL TEST SCENES

Maintain deterministinding licensing obligations.

---

# 141. TEST REAL MODEL OUTPUTS

For at least one supported local model per capability:

```text
LLM
VLM
embedding
reranker
ASR
TTS
image
image→3D
```

run a real smoke test.

Record:

```text
model hash
output hash
latency
RAM
VRAM
success
```

---

# 142. TEST LEVELS

Every feature must have:

```text
unit test
integration test
headless smoke test
```

Visual features additionally require:

```text
visual regression
```

Physics features additionally require:

```text
state-based physics test
```

AI features additionally require:

```text
model capability verification
```

---

# 143. NO TEST THEATER

Never add a test that only checks:

```text
file exists
```

when the actual feature is:

```text
model inference
```

Never define:

```text
return True
```

as a substitute for a real verification.

Tests must fail when the feature is actually broken.

---

# 144. BENCHMARKING

Record benchmarks separately for:

```text
startup
model load
first inference
warm inference
offload
reload
image generation
3D generation
chunk generation
physics tick
renderer FPS
memory
```

Do not compare incomparable hardware profiles as if they were identical.

---

# 145. RESOURCE CALIBRATION

On first run, perform a short calibration.

Measure:

```text
startup
small model load
small inference
renderer baseline
world baseline
```

If a model is loaded for the first time, record actual peak memory.

Update local performance metadata.

This creates an adaptive machine-specific profile.

---

# 146. MODEL PERFORMANCE CACHE

Store:

```text
device fingerprint
model hash
quant
resolution
context
backend
latency
RAM peak
VRAM peak
success
```

Then use historical results to improve future admission decisions.

---

# 147. MACHINE-SPECIFIC PROFILE

Never commit local hardware-specific measurements to the public repository.

Store them in:

```text
%LOCALAPPDATA%\FlyBrain\config\
```

or a user profile database.

---

# 148. SAFE QUALITY DEGRADATION

When resources are insufficient, reduce:

```text
render quality
AI resolution
3D resolution
chunk radius
background jobs
VLM frame frequency
```

before sacrificing:

```text
physics
state integrity
brain core
memory integrity
backup safety
```

---

# 149. USER OVERRIDE

Provide profiles:

```text
AUTO
LOW
BALANCED
HIGH
```

AUTO is default.

Manual high settings must still obey hard safety limits.

A user must not be allowed to cause catastrophic allocation simply by selecting "ULTRA".

---

# 150. V9 FINAL USER EXPERIENCE

A fresh user should be able to:

```text
install
launch
```

and the system automatically:

```text
detect hardware
select backend
select profile
verify models
load minimal required models
start world
show organism
show live metrics
```

Then the user can ask for:

```text
new world
new object
new creature
new building
new dream
new experiment
```

without manually configuring individual model processes.

---

# 151. EXAMPLE FINAL GENERATIVE LOOP

User:

```text
Create a small forest settlement beside a lake with a wooden bridge.
```

FlyBrain:

```text
LLM
→ structured WorldIntent
→ world planner
→ deterministic terrain
→ lake
→ forest
→ settlement rules
→ bridge requirement
→ asset resolver
→ generated image
→ image→3D
→ asset validation
→ collision
→ scale
→ placement
→ MuJoCo
→ world persistence
→ visual render
→ provenance
```

Then:

```text
organism explores
→ VLM sees bridge
→ LLM decides destination
→ navigation
→ motion
→ physics
→ crossing
→ memory
```

This is the V9 target.

---

# 152. FINAL RELEASE ARTIFACT TREE

Repository:

```text
.github/
  workflows/
docs/
installer/
scripts/
src/
frontend/
tests/
tools/
reports/
README.md
CHANGELOG.md
RELEASE_NOTES.md
SECURITY.md
THIRD_PARTY_LICENSES.md
pyproject.toml
uv.lock
package.json
```

Release:

```text
FlyBrain-v9.0.0-Windows-x64-Setup.exe
FlyBrain-v9.0.0-Windows-x64-Portable.zip
SHA256SUMS.txt
SBOM.json
release-manifest.json
acceptance-matrix.json
verification-report.md
reproducibility-manifest.json
```

---

# 153. FINAL AUTONOMOUS Ending licensing obligations.

---

# 141. TEST REAL MODEL OUTPUTS

For at least one supported local model per capability:

```text
LLM
VLM
embedding
reranker
ASR
TTS
image
image→3D
```

run a real smoke test.

Record:

```text
model hash
output hash
latency
RAM
VRAM
success
```

---

# 142. TEST LEVELS

Every feature must have:

```text
unit test
integration test
headless smoke test
```

Visual features additionally require:

```text
visual regression
```

Physics features additionally require:

```text
state-based physics test
```

AI features additionally require:

```text
model capability verification
```

---

# 143. NO TEST THEATER

Never add a test that only checks:

```text
file exists
```

when the actual feature is:

```text
model inference
```

Never define:

```text
return True
```

as a substitute for a real verification.

Tests must fail when the feature is actually broken.

---

# 144. BENCHMARKING

Record benchmarks separately for:

```text
startup
model load
first inference
warm inference
offload
reload
image generation
3D generation
chunk generation
physics tick
renderer FPS
memory
```

Do not compare incomparable hardware profiles as if they were identical.

---

# 145. RESOURCE CALIBRATION

On first run, perform a short calibration.

Measure:

```text
startup
small model load
small inference
renderer baseline
world baseline
```

If a model is loaded for the first time, record actual peak memory.

Update local performance metadata.

This creates an adaptive machine-specific profile.

---

# 146. MODEL PERFORMANCE CACHE

Store:

```text
device fingerprint
model hash
quant
resolution
context
backend
latency
RAM peak
VRAM peak
success
```

Then use historical results to improve future admission decisions.

---

# 147. MACHINE-SPECIFIC PROFILE

Never commit local hardware-specific measurements to the public repository.

Store them in:

```text
%LOCALAPPDATA%\FlyBrain\config\
```

or a user profile database.

---

# 148. SAFE QUALITY DEGRADATION

When resources are insufficient, reduce:

```text
render quality
AI resolution
3D resolution
chunk radius
background jobs
VLM frame frequency
```

before sacrificing:

```text
physics
state integrity
brain core
memory integrity
backup safety
```

---

# 149. USER OVERRIDE

Provide profiles:

```text
AUTO
LOW
BALANCED
HIGH
```

AUTO is default.

Manual high settings must still obey hard safety limits.

A user must not be allowed to cause catastrophic allocation simply by selecting "ULTRA".

---

# 150. V9 FINAL USER EXPERIENCE

A fresh user should be able to:

```text
install
launch
```

and the system automatically:

```text
detect hardware
select backend
select profile
verify models
load minimal required models
start world
show organism
show live metrics
```

Then the user can ask for:

```text
new world
new object
new creature
new building
new dream
new experiment
```

without manually configuring individual model processes.

---

# 151. EXAMPLE FINAL GENERATIVE LOOP

User:

```text
Create a small forest settlement beside a lake with a wooden bridge.
```

FlyBrain:

```text
LLM
→ structured WorldIntent
→ world planner
→ deterministic terrain
→ lake
→ forest
→ settlement rules
→ bridge requirement
→ asset resolver
→ generated image
→ image→3D
→ asset validation
→ collision
→ scale
→ placement
→ MuJoCo
→ world persistence
→ visual render
→ provenance
```

Then:

```text
organism explores
→ VLM sees bridge
→ LLM decides destination
→ navigation
→ motion
→ physics
→ crossing
→ memory
```

This is the V9 target.

---

# 152. FINAL RELEASE ARTIFACT TREE

Repository:

```text
.github/
  workflows/
docs/
installer/
scripts/
src/
frontend/
tests/
tools/
reports/
README.md
CHANGELOG.md
RELEASE_NOTES.md
SECURITY.md
THIRD_PARTY_LICENSES.md
pyproject.toml
uv.lock
package.json
```

Release:

```text
FlyBrain-v9.0.0-Windows-x64-Setup.exe
FlyBrain-v9.0.0-Windows-x64-Portable.zip
SHA256SUMS.txt
SBOM.json
release-manifest.json
acceptance-matrix.json
verification-report.md
reproducibility-manifest.json
```

---

# 153. FINAL AUTONOMOUS EXECUTION ALGORITHM

Execute this continuously:

```text
START
↓
inspect repository
↓
verify V7 baseline
↓
create recovery point
↓
plan current phase
↓
implement smallest safe increment
↓
compile
↓
unit test
↓
integration test
↓
headless test
↓
hardware test where applicable
↓
visual test where applicable
↓
inspect failures
↓
repair
↓
repeat
↓
checkpoint
↓
commit
↓
push candidate
↓
cloud build
↓
inspect cloud result
↓
repair if required
↓
repeat
↓
local final verification
↓
cloud final verification
↓
package
↓
verify package
↓
publish release
↓
download released artifact
↓
verify released artifact
↓
deploy HF
↓
verify HF
↓
update README
↓
update release notes
↓
generate final reports
↓
FINAL PASS
```

---

# 154. FINAL EXIT CONDITION

Do not stop at:

```text
code complete
```

Stop only at:

```text
IMPLEMENTED
+
TESTED
+
HEADLESS VERIFIED
+
VISUALLY VERIFIED
+
PHYSICALLY VERIFIED
+
RESOURCE VERIFIED
+
PACKAGED
+
CLOUD VERIFIED
+
INSTALLER VERIFIED
+
DOCUMENTATION UPDATED
+
RELEASE PUBLISHED
+
POST-RELEASE VERIFIED
```

---

# 155. FINAL REPORT

At the end write:

```text
reports/final/
```

with:

```text
summary.md
test-results.json
hardware-report.json
resource-report.json
model-report.json
asset-report.json
world-report.json
physics-report.json
visual-report.json
backup-report.json
installer-report.json
cloud-report.json
license-report.json
reproducibility-manifest.json
```

The final summary must clearly state:

```text
VERSION
COMMIT
RELEASE
LOCAL TEST RESULT
CLOUD TEST RESULT
HARDWARE RESULT
MODEL RESULT
WORLD RESULT
PHYSICS RESULT
VISUAL RESULT
INSTALLER RESULT
HF RESULT
KNOWN LIMITATIONS
```

No fabricated percentages.

No fake "100% working".

Use the evidence collected by the actual verification system.

---

# 156. CORE PHILOSOPHY

FlyBrain is not a game with AI decoration.

It is not a chatbot inside a 3D scene.

It is not an image generator pretending to understand physics.

It is not an LLM teleporting objects around.

The architecture must remain:

```text
BIOLOGY
+
MEMORY
+
AUTONOMY
+
PERCEPTION
+
LANGUAGE
+
GENERATIVE AI
+
PHYSICAL SIMULATION
+
PERSISTENT WORLD
+
PROVENANCE
```

The LLM proposes.

The planners decide within typed constraints.

The world compiler creates.

The physics engine validates reality.

The organism experiences the result.

Memory records it.

The renderer visualizes it.

The backup system preserves it.

The watchdog keeps it alive.

The release system proves it.

---

# 157. ABSOLUTE FINAL RULE

NEVER optimize for looking finished.

Optimize for being empirically verified.

NEVER delete working V7 science just to make V8/V9 cleaner.

NEVER replace real functionality with mocks.

NEVER hide failures.

NEVER skip the test because a dependency is inconvenient.

NEVER claim hardware support without real hardware verification.

NEVER claim model support without real model inference.

NEVER claim visual correctness without visual verification.

NEVER claim persistence without restart/restore verification.

NEVER claim release readiness until the actual published artifact has been installed and tested.

Build.

Measure.

Repair.

Verify.

Release.
</USER_REQUEST>
<ADDITIONAL_METADATA>
The current local time is: 2026-09-21T11:17:21+02:00.
</ADDITIONAL_METADATA>
<USER_SETTINGS_CHANGE>
The user changed setting `Model Selection` from Gemini 3.8 Flash (Medium) to Gemini 3.8 Flash (High). No need to comment on this change if the user doesn't ask about it. If reporting what model you are, please use a human readable name instead of the exact string.
</USER_SETTINGS_CHANGE>
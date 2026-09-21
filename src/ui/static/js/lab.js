// State
        let isSimRunning = false;
        let activeTab = 'overview';
        let connectomeData = null;
        let threeScene = null, threeCamera = null, threeRenderer = null, threeControls = null;
        let neuronPoints = null, edgeLines = null;
        let raycaster = new THREE.Raycaster();
        let mouse = new THREE.Vector2();

        function showToast(msg, type = 'info') {
            const container = document.getElementById('toast-container');
            const toast = document.createElement('div');
            toast.className = 'toast';
            toast.innerText = msg;
            container.appendChild(toast);
            setTimeout(() => { toast.remove(); }, 3500);
        }

        function switchTab(tabId, el) {
            activeTab = tabId;
            document.querySelectorAll('.nav-tab').forEach(t => t.classList.remove('active'));
            document.querySelectorAll('.tab-content').forEach(c => c.classList.remove('active'));
            const btn = el || document.querySelector(`.nav-tab[onclick*="'${tabId}'"]`);
            if (btn) btn.classList.add('active');
            document.getElementById('tab-' + tabId).classList.add('active');

            if (tabId === 'connectome') {
                if (!threeScene) init3DViewer();
                fetchConnectome();
            } else if (tabId === 'world') {
                if (!worldScene) initWorldViewer();
                fetchWorld();
            } else if (tabId === 'stream') {
                fetchStream();
            } else if (tabId === 'colony') {
                fetchColony();
            } else if (tabId === 'backups') {
                fetchBackups();
            } else if (tabId === 'provenance') {
                fetchProvenance();
            } else if (tabId === 'experiments') {
                fetchExperiments();
            } else if (tabId === 'memory') {
                fetchMemory();
            } else if (tabId === 'evolution') {
                fetchEvolution();
            } else if (tabId === 'dreams') {
                fetchDreams();
            } else if (tabId === 'diagnostics') {
                fetchDiagnostics();
            }
        }

        // Telemetry WebSocket (with backoff reconnect + HTTP polling fallback)
        const proto = window.location.protocol === 'https:' ? 'wss:' : 'ws:';
        const wsUrl = `${proto}//${window.location.host}/ws/telemetry`;
        let socket = null;
        let wsBackoffMs = 2000;
        let wsFailed = 0;
        let pollTimer = null;

        function startPollFallback() {
            if (pollTimer) return;
            pollTimer = setInterval(async () => {
                try {
                    const res = await fetch('/api/telemetry');
                    if (res.ok) updateTelemetryUI(await res.json());
                } catch (e) { /* stay silent; retry next tick */ }
            }, 3000);
        }

        function stopPollFallback() {
            if (pollTimer) { clearInterval(pollTimer); pollTimer = null; }
        }

        function initWebSocket() {
            socket = new WebSocket(wsUrl);
            socket.onmessage = (event) => {
                const data = JSON.parse(event.data);
                updateTelemetryUI(data);
            };
            socket.onopen = () => {
                wsBackoffMs = 2000; wsFailed = 0;
                stopPollFallback();
            };
            socket.onerror = () => { try { socket.close(); } catch (e) {} };
            socket.onclose = () => {
                wsFailed += 1;
                if (wsFailed >= 2) startPollFallback();
                showToast('Telemetry connection lost — reconnecting…', 'warn');
                setTimeout(initWebSocket, Math.min(wsBackoffMs, 30000));
                wsBackoffMs *= 2;
            };
        }
        initWebSocket();
        fetchProvenance();

        function updateTelemetryUI(data) {
            document.getElementById('stat-step').innerText = data.step.toLocaleString();
            document.getElementById('stat-spikes').innerText = data.spikes;
            document.getElementById('stat-total-spikes').innerText = `Cumulative Spikes: ${data.total_spikes.toLocaleString()}`;
            document.getElementById('stat-act').innerText = data.mean_activation.toFixed(4);
            document.getElementById('stat-pot').innerText = `Potential: ${data.mean_potential.toFixed(3)} mV`;
            document.getElementById('stat-pred-error').innerText = `Prediction Error: ${data.prediction_error.toFixed(4)}`;
            document.getElementById('stat-latency').innerText = `${data.step_latency_ms.toFixed(2)} ms`;

            document.getElementById('drive-energy').innerText = data.drives.energy.toFixed(2);
            document.getElementById('drive-curiosity').innerText = data.drives.curiosity.toFixed(2);
            document.getElementById('drive-social').innerText = data.drives.social.toFixed(2);
            document.getElementById('drive-integrity').innerText = data.drives.integrity.toFixed(2);

            // Simulation status
            isSimRunning = data.is_running;
            const statusBadge = document.getElementById('sim-status-badge');
            const statusText = document.getElementById('sim-status-text');
            const btn = document.getElementById('btn-toggle-sim');
            if (isSimRunning) {
                statusBadge.className = 'badge badge-verified';
                statusText.innerText = 'RUNNING';
                btn.className = 'btn btn-amber';
                btn.innerText = '⏸ Pause Simulation';
            } else {
                statusBadge.className = 'badge badge-state';
                statusText.innerText = 'PAUSED';
                btn.className = 'btn btn-green';
                btn.innerText = '▶ Start Simulation';
            }

            document.getElementById('backend-badge').innerText = `${data.backend.toUpperCase()} (${data.device_name})`;
            // Provenance badge is owned by fetchProvenance() (live /api/provenance);
            // telemetry never overwrites it with a static label.
        }

        // Scientific provenance (live from /api/provenance; never static)
        async function fetchProvenance() {
            try {
                const res = await fetch('/api/provenance');
                if (!res.ok) throw new Error('HTTP ' + res.status);
                const p = await res.json();
                const badge = document.getElementById('provenance-badge');
                badge.innerText = `${p.graph_identity} \u00B7 ${p.provenance_status}`;
                badge.className = 'badge ' + (p.provenance_status === 'VERIFIED' ? 'badge-verified' : 'badge-state');
                badge.title = `Sampling: ${p.sampling.strategy} \u2014 ${p.sampling.bias}`;
                const set = (id, txt) => { document.getElementById(id).innerText = txt; };
                document.getElementById('prov-graph-identity').innerText = `${p.graph_identity} (${p.provenance_status})`;
                set('prov-graph', `${p.graph_identity} \u2014 full-graph locally available: ${p.sampling.full_graph_available_locally}; sampled ${p.sampling.sampled_neurons}/${p.sampling.source_neurons} neurons, ${p.sampling.sampled_edges}/${p.sampling.source_edges} edges`);
                set('prov-sampling', `${p.sampling.strategy}: ${p.sampling.detail} Bias: ${p.sampling.bias} (seed ${p.sampling.seed})`);
                set('prov-weights', `${p.weight_semantics.source} \u2192 ${p.weight_semantics.transform} ${p.weight_semantics.note}`);
                const pops = Object.entries(p.populations).map(([n, v]) =>
                    `${n} [${v.annotation_level}${v.heuristic ? ', heuristic' : ''}, n=${v.count}]`).join(' \u00B7 ');
                set('prov-pops', `All functional populations are coordinate/rule heuristics (never EM annotations): ${pops}`);
                set('prov-dataset', `${p.dataset.name} ${p.dataset.version} | soma ${String(p.dataset.soma_sha256).slice(0, 16)}\u2026 | conn ${String(p.dataset.connections_sha256).slice(0, 16)}\u2026`);
                set('prov-hash', p.graph_hash);
            } catch (e) {
                document.getElementById('provenance-badge').innerText = 'Provenance OFFLINE';
            }
        }

        async function toggleSimulation() {
            const endpoint = isSimRunning ? '/api/simulation/pause' : '/api/simulation/start';
            const res = await fetch(endpoint, { method: 'POST' });
            const data = await res.json();
            showToast(`Simulation ${data.status}`);
        }

        async function stepSimulation(count = 1) {
            const res = await fetch('/api/simulation/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ steps: count, reward: 0.1 })
            });
            const data = await res.json();
            showToast(`Stepped ${count} step(s) | Spikes: ${data.spikes}`);
        }

        async function injectSensory(channel, intensity) {
            const sensory = {};
            sensory[channel] = Array(32).fill(intensity);
            await fetch('/api/simulation/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ steps: 1, sensory_inputs: sensory, reward: 0.2 })
            });
            showToast(`Injected ${channel} sensory stimulus (${intensity})`);
        }

        async function injectReward(val) {
            await fetch('/api/simulation/step', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ steps: 1, reward: val })
            });
            showToast(`Injected reward +${val}`);
        }

        // Three.js 3D Connectome Viewer
        function init3DViewer() {
            const container = document.getElementById('connectome-canvas-wrapper');
            const canvas = document.getElementById('connectome-canvas');
            const width = container.clientWidth;
            const height = container.clientHeight;

            threeScene = new THREE.Scene();
            threeScene.background = new THREE.Color(0x06080e);

            threeCamera = new THREE.PerspectiveCamera(45, width / height, 0.1, 100);
            threeCamera.position.set(0, 0, 3.2);

            threeRenderer = new THREE.WebGLRenderer({ canvas: canvas, antialias: true });
            threeRenderer.setSize(width, height);
            threeRenderer.setPixelRatio(window.devicePixelRatio);

            threeControls = new THREE.OrbitControls(threeCamera, threeRenderer.domElement);
            threeControls.enableDamping = true;
            threeControls.dampingFactor = 0.05;

            // Ambient & directional lighting
            const ambient = new THREE.AmbientLight(0xffffff, 0.6);
            threeScene.add(ambient);
            const dirLight = new THREE.DirectionalLight(0x00f0ff, 0.8);
            dirLight.position.set(2, 4, 3);
            threeScene.add(dirLight);

            // Animate loop
            function animate() {
                requestAnimationFrame(animate);
                threeControls.update();
                if (neuronPoints) {
                    neuronPoints.rotation.y += 0.001;
                    if (edgeLines) edgeLines.rotation.y = neuronPoints.rotation.y;
                }
                threeRenderer.render(threeScene, threeCamera);
            }
            animate();

            // Raycasting for neuron inspection
            canvas.addEventListener('click', onCanvasClick);
        }

        async function fetchConnectome() {
            const res = await fetch('/api/connectome?max_nodes=512&max_edges=384');
            connectomeData = await res.json();
            renderConnectome3D(connectomeData);
        }

        function renderConnectome3D(data) {
            if (!threeScene) return;

            // Clean previous objects
            if (neuronPoints) threeScene.remove(neuronPoints);
            if (edgeLines) threeScene.remove(edgeLines);

            const nodes = data.nodes;
            const geometry = new THREE.BufferGeometry();
            const positions = [];
            const colors = [];

            nodes.forEach(n => {
                positions.push(n.pos[0], n.pos[1], n.pos[2]);
                // Color by hemisphere side: Left=Cyan, Right=Emerald, Mid=Amber
                if (n.side === 'L') {
                    colors.push(0.0, 0.94, 1.0);
                } else if (n.side === 'R') {
                    colors.push(0.0, 1.0, 0.53);
                } else {
                    colors.push(1.0, 0.72, 0.0);
                }
            });

            geometry.setAttribute('position', new THREE.Float32BufferAttribute(positions, 3));
            geometry.setAttribute('color', new THREE.Float32BufferAttribute(colors, 3));

            const material = new THREE.PointsMaterial({
                size: 0.045,
                vertexColors: true,
                transparent: true,
                opacity: 0.85
            });

            neuronPoints = new THREE.Points(geometry, material);
            threeScene.add(neuronPoints);

            // Render Edges
            const edgePositions = [];
            data.edges.forEach(e => {
                const srcNode = nodes[e.src];
                const tgtNode = nodes[e.tgt];
                if (srcNode && tgtNode) {
                    edgePositions.push(srcNode.pos[0], srcNode.pos[1], srcNode.pos[2]);
                    edgePositions.push(tgtNode.pos[0], tgtNode.pos[1], tgtNode.pos[2]);
                }
            });

            const edgeGeom = new THREE.BufferGeometry();
            edgeGeom.setAttribute('position', new THREE.Float32BufferAttribute(edgePositions, 3));
            const edgeMat = new THREE.LineBasicMaterial({
                color: 0x465a82,
                transparent: true,
                opacity: 0.35
            });
            edgeLines = new THREE.LineSegments(edgeGeom, edgeMat);
            threeScene.add(edgeLines);
        }

        function onCanvasClick(event) {
            if (!connectomeData || !threeCamera) return;
            const rect = event.target.getBoundingClientRect();
            mouse.x = ((event.clientX - rect.left) / rect.width) * 2 - 1;
            mouse.y = -((event.clientY - rect.top) / rect.height) * 2 + 1;

            raycaster.setFromCamera(mouse, threeCamera);
            if (neuronPoints) {
                const intersects = raycaster.intersectObject(neuronPoints);
                if (intersects.length > 0) {
                    const idx = intersects[0].index;
                    selectNeuron(connectomeData.nodes[idx]);
                }
            }
        }

        function selectNeuron(n) {
            document.getElementById('insp-body-id').innerText = n.id;
            document.getElementById('insp-idx').innerText = n.idx;
            document.getElementById('insp-side').innerText = n.side === 'L' ? 'Left Hemilateral (L)' : n.side === 'R' ? 'Right Hemilateral (R)' : 'Midline (M)';
            document.getElementById('insp-tbars').innerText = n.tbars.toLocaleString();
            document.getElementById('insp-pot').innerText = `${n.pot.toFixed(3)} mV`;
            document.getElementById('insp-spk').innerText = n.spk ? 'SPIKING (1)' : 'QUIESCENT (0)';
            document.getElementById('insp-act').innerText = n.act.toFixed(3);
            document.getElementById('insp-coords').innerText = `[${n.pos.join(', ')}]`;
            showToast(`Inspecting Body ID ${n.id}`);
        }

        function searchNeuron() {
            const q = document.getElementById('search-body-id').value.trim();
            if (!connectomeData) return;
            const target = connectomeData.nodes.find(n => n.id.toString() === q || n.idx.toString() === q);
            if (target) {
                selectNeuron(target);
            } else {
                showToast(`Neuron ID ${q} not found in loaded circuit`, 'warn');
            }
        }

        function resetCamera() {
            if (threeCamera && threeControls) {
                threeCamera.position.set(0, 0, 3.2);
                threeControls.reset();
            }
        }

        // Experiments Tab
        async function fetchExperiments() {
            const res = await fetch('/api/experiments');
            const data = await res.json();
            const tbody = document.getElementById('tbody-experiments');
            tbody.innerHTML = '';
            data.forEach(exp => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td style="font-family: monospace; font-weight: bold;">${exp.experiment_id}</td>
                    <td><span class="badge badge-verified">${exp.graph_mode}</span></td>
                    <td>${exp.neuron_scale}</td>
                    <td>${exp.duration_steps}</td>
                    <td>${exp.metrics.mean_step_latency_ms} ms</td>
                    <td style="font-family: monospace; font-size: 11px;">${exp.final_state_hash.substring(0, 16)}...</td>
                    <td><button class="btn" onclick="verifyExperiment('${exp.experiment_id}')">Verify</button></td>
                `;
                tbody.appendChild(tr);
            });
        }

        async function runExperiment() {
            const mode = document.getElementById('exp-mode').value;
            const scale = parseInt(document.getElementById('exp-scale').value);
            const seed = parseInt(document.getElementById('exp-seed').value);
            const steps = parseInt(document.getElementById('exp-steps').value);

            showToast(`Launching experiment (${mode}, ${scale} neurons, ${steps} steps)...`);
            const res = await fetch('/api/experiments/run', {
                method: 'POST',
                headers: { 'Content-Type': 'application/json' },
                body: JSON.stringify({ graph_mode: mode, neuron_scale: scale, seed: seed, duration_steps: steps })
            });
            const data = await res.json();
            showToast(`Experiment ${data.experiment_id} completed! Final hash: ${data.final_state_hash.substring(0, 12)}`);
            fetchExperiments();
        }

        async function verifyExperiment(expId) {
            showToast(`Re-verifying determinism of ${expId}...`);
            const res = await fetch(`/api/experiments/verify?experiment_id=${expId}`, { method: 'POST' });
            const data = await res.json();
            if (data.deterministic_match) {
                showToast(`VERIFIED: Bit-exact deterministic state match for ${expId}!`);
            } else {
                showToast(`Divergence detected in ${expId}`, 'error');
            }
        }

        // Memory Tab
        async function fetchMemory() {
            const q = document.getElementById('memory-search').value;
            const res = await fetch(`/api/memory?query=${encodeURIComponent(q)}&limit=15`);
            const data = await res.json();
            const tbody = document.getElementById('tbody-episodes');
            tbody.innerHTML = '';
            data.episodes.forEach(ep => {
                const tr = document.createElement('tr');
                tr.innerHTML = `
                    <td>#${ep.id}</td>
                    <td>${ep.step}</td>
                    <td><span class="badge badge-gpu">${ep.action}</span></td>
                    <td>${ep.reward.toFixed(2)}</td>
                    <td>${ep.prediction_error.toFixed(4)}</td>
                    <td style="color: var(--text-dim); font-size: 11px;">${new Date(ep.timestamp * 1000).toLocaleTimeString()}</td>
                `;
                tbody.appendChild(tr);
            });
        }

        // Evolution Tab
        async function fetchEvolution() {
            const res = await fetch('/api/evolution/lineage');
            const data = await res.json();
            document.getElementById('evo-meta-text').innerText = `Generation: ${data.current_generation} | Leader: ${data.current_brain_id}`;

            const container = document.getElementById('evo-lineage-container');
            container.innerHTML = '';
            data.history.forEach(gen => {
                const genDiv = document.createElement('div');
                genDiv.className = 'card';
                genDiv.innerHTML = `
                    <div class="card-header">
                        <span>Generation ${gen.generation}</span>
                        <span class="badge ${gen.improved ? 'badge-verified' : 'badge-state'}">${gen.improved ? 'FITNESS IMPROVED' : 'ROLLBACK'}</span>
                    </div>
                    <div style="font-size: 12px; color: var(--text-dim); margin-bottom: 8px;">Baseline Score: ${gen.baseline_score.toFixed(4)} → Best Score: ${gen.best_score.toFixed(4)}</div>
                    <div style="display: grid; grid-template-columns: repeat(auto-fit, minmax(200px, 1fr)); gap: 10px;">
                        ${gen.candidates.map(c => `
                            <div style="background: rgba(10, 15, 25, 0.6); padding: 10px; border-radius: 6px; border: 1px solid ${c.accepted ? 'var(--accent-green)' : 'var(--border-subtle)'};">
                                <div style="font-family: monospace; font-size: 11px; font-weight: bold;">${c.candidate_id}</div>
                                <div style="font-size: 13px; font-weight: bold; margin-top: 4px;">Score: ${c.score.toFixed(4)}</div>
                                <div style="font-size: 11px; color: var(--text-dim); margin-top: 2px;">${c.reason}</div>
                            </div>
                        `).join('')}
                    </div>
                `;
                container.appendChild(genDiv);
            });
        }

        async function triggerEvolution() {
            showToast('Evaluating candidate variants across generation battery...');
            const res = await fetch('/api/evolution/generation?num_candidates=4', { method: 'POST' });
            const data = await res.json();
            showToast(`Generation ${data.generation} evaluated! Best score: ${data.best_score.toFixed(4)}`);
            fetchEvolution();
        }

        // Dreams Tab
        async function fetchDreams() {
            const res = await fetch('/api/dreams?limit=8');
            const data = await res.json();
            const container = document.getElementById('dreams-list');
            container.innerHTML = '';
            data.forEach(d => {
                const div = document.createElement('div');
                div.style.background = 'rgba(12, 18, 30, 0.7)';
                div.style.padding = '12px 16px';
                div.style.borderRadius = '8px';
                div.style.border = '1px solid var(--border-subtle)';
                div.innerHTML = `
                    <div style="display: flex; justify-content: space-between; font-size: 12px; margin-bottom: 4px;">
                        <span style="font-family: monospace; color: var(--accent-amber);">Dream Replay #${d.id} (Base Episode #${d.base_episode_id})</span>
                        <span class="badge badge-gpu">${d.simulated_action}</span>
                    </div>
                    <div style="font-size: 13px; color: var(--text-main); margin-bottom: 4px;">${d.insight}</div>
                    <div style="font-size: 11px; color: var(--text-dim);">Counterfactual Reward: ${d.counterfactual_reward.toFixed(2)} | Seed: ${d.seed}</div>
                `;
                container.appendChild(div);
            });
        }

        async function triggerDream(mode) {
            showToast(`Initiating ${mode} dream replay cycle...`);
            const res = await fetch(`/api/dreams/replay?mode=${mode}&count=2`, { method: 'POST' });
            const data = await res.json();
            showToast(`Dream cycle consolidated ${data.length} counterfactual replay(s) into memory!`);
            fetchDreams();
        }

        // Stream Tab (24/7 mode)
        async function fetchStream() {
            try {
                const [s, w] = await Promise.all([
                    fetch('/api/v1/stream/status').then(r => r.json()),
                    fetch('/api/v1/watchdog/status').then(r => r.json()).catch(() => null)
                ]);
                document.getElementById('stream-mode-badge').innerText = s.stream_mode;
                document.getElementById('stream-mode-badge').className =
                    'badge ' + (s.stream_mode === 'RUNNING' ? 'badge-verified' : 'badge-state');
                document.getElementById('stream-uptime').innerText =
                    `${Math.floor(s.uptime_sec / 3600)}h ${Math.floor((s.uptime_sec % 3600) / 60)}m ${Math.floor(s.uptime_sec % 60)}s`;
                document.getElementById('stream-step').innerText =
                    `step ${s.simulation_step} @ ${s.target_hz} Hz (${s.backend})`;
                document.getElementById('stream-perf').innerText =
                    `${s.active_spikes} active spikes, ${s.total_spikes} total, p95 ${s.p95_latency_ms} ms`;
                document.getElementById('stream-watchdog').innerText = w
                    ? `running=${w.running}, restarts=${w.restarts}/${w.max_restarts}, last step=${w.last_step_seen}`
                    : 'watchdog disabled';
            } catch (e) { showToast('Stream status unreachable'); }
        }

        async function streamCmd(action) {
            const map = { start: ['/api/v1/stream/start', 'POST'], pause: ['/api/simulation/pause', 'POST'],
                          resume: ['/api/v1/simulation/resume', 'POST'], stop: ['/api/v1/stream/stop', 'POST'] };
            const [url, method] = map[action];
            const res = await fetch(url, { method });
            const data = await res.json();
            showToast(`Stream ${action}: ${data.status || data.stream_mode}`);
            fetchStream();
        }

        // Colony Tab (organisms)
        async function fetchColony() {
            const res = await fetch('/api/v1/organisms');
            const data = await res.json();
            document.getElementById('colony-meta').innerText =
                `tick ${data.tick} · living ${data.living}/${data.total} · ${String(data.population_hash).slice(0, 12)}…`;
            document.getElementById('tbody-colony').innerHTML = data.organisms.map(o => `
                <tr><td style="font-family: monospace; font-size: 11px;">${o.id.slice(0, 10)}…</td>
                <td>${o.generation}</td><td>${o.stage}</td><td>${o.alive ? 'yes' : 'no'}</td>
                <td>${o.age}</td><td>${o.energy}</td><td>${o.neurons}</td>
                <td style="font-family: monospace; font-size: 11px;">${o.genome_hash.slice(0, 12)}…</td></tr>`).join('');
        }

        async function colonyCmd(action) {
            const map = { step: ['/api/colony/step?ticks=5', 'POST'], reproduce: ['/api/colony/reproduce?n_offspring=2', 'POST'],
                          reset: ['/api/colony/reset', 'POST'] };
            const [url, method] = map[action];
            const res = await fetch(url, { method, headers: { 'Content-Type': 'application/json' }, body: '{}' });
            const data = await res.json();
            showToast(`Colony ${action}: ${JSON.stringify(data).slice(0, 120)}`);
            fetchColony();
        }

        // Backups Tab
        async function fetchBackups() {
            const [list, gd] = await Promise.all([
                fetch('/api/v1/backup/list').then(r => r.json()),
                fetch('/api/v1/backup/google').then(r => r.json()).catch(() => ({ state: 'ERROR' }))
            ]);
            document.getElementById('gdrive-badge').innerText = `drive: ${gd.state}`;
            document.getElementById('gdrive-detail').innerText =
                `Google Drive: ${gd.state} — ${gd.detail || ''} ${gd.consent_step ? 'Next: ' + gd.consent_step : ''}`;
            document.getElementById('tbody-backups').innerHTML = list.backups.map(b => `
                <tr><td style="font-family: monospace; font-size: 11px;">${b.backup}</td>
                <td>${b.trigger || ''}</td><td>${b.step ?? ''}</td>
                <td style="font-family: monospace; font-size: 11px;">${String(b.state_hash || '').slice(0, 16)}…</td>
                <td><button class="btn" onclick="backupVerify('${b.backup}')">Verify</button>
                <button class="btn" onclick="backupRestore('${b.backup}')">Restore</button>
                <button class="btn" onclick="backupDownload('${b.backup}')">Download</button></td></tr>`).join('')
                || '<tr><td colspan="5">No backups yet — press BACKUP NOW.</td></tr>';
        }

        async function backupCreate() {
            const res = await fetch('/api/v1/backup/create?label=ui&trigger=manual', { method: 'POST' });
            const data = await res.json();
            showToast(`Backup ${data.backup_name} (${String(data.state_hash).slice(0, 12)}…)`);
            fetchBackups();
        }

        async function backupVerify(name) {
            const res = await fetch(`/api/v1/backup/verify?name=${encodeURIComponent(name)}`);
            const data = await res.json();
            showToast(`Backup ${name}: ${data.status}`);
        }

        async function backupRestore(name) {
            const res = await fetch(`/api/v1/backup/restore?name=${encodeURIComponent(name)}`, { method: 'POST' });
            const data = await res.json();
            showToast(`Restore ${name}: ${data.status || data.detail}`);
            fetchBackups();
        }

        function backupDownload(name) {
            window.open(`/api/v1/backup/download?name=${encodeURIComponent(name)}`, '_blank');
        }

        // ---- Embodied 3D World (render-only; physics is server-side MuJoCo) ----
        let worldScene = null, worldCamera = null, worldRenderer = null, worldControls = null;
        let worldCamMode = 'third', worldDebug = false;
        let worldGeom = null, worldChars = {}, worldState = null;

        function worldScalePos(p) { return [p[0], p[2], -p[1]]; } // MuJoCo (x,y,z) -> three (x,z,-y)

        function initWorldViewer() {
            const container = document.getElementById('world-canvas-wrapper');
            const canvas = document.getElementById('world-canvas');
            worldScene = new THREE.Scene();
            worldScene.background = new THREE.Color(0x06080e);
            worldCamera = new THREE.PerspectiveCamera(55, 1, 0.1, 200);
            worldRenderer = new THREE.WebGLRenderer({ canvas, antialias: true });
            const resize = () => {
                const w = container.clientWidth || 800, h = container.clientHeight || 500;
                worldRenderer.setSize(w, h, false);
                worldCamera.aspect = w / h; worldCamera.updateProjectionMatrix();
            };
            resize(); window.addEventListener('resize', resize);
            worldControls = new THREE.OrbitControls(worldCamera, worldRenderer.domElement);
            worldScene.add(new THREE.AmbientLight(0xffffff, 0.55));
            const sun = new THREE.DirectionalLight(0xfff2dd, 0.9);
            sun.position.set(10, 18, 6); worldScene.add(sun);
            const render = () => {
                requestAnimationFrame(render);
                if (worldControls && worldCamMode === 'free') worldControls.update();
                if (worldScene && worldCamera) {
                    updateWorldCamera();
                    worldRenderer.render(worldScene, worldCamera);
                }
            };
            render();
        }

        function buildWorldGeometry(g) {
            worldGeom = g;
            const mat = (c) => new THREE.MeshLambertMaterial({ color: c });
            const addBox = (x, y, sx, sy, h, color, name) => {
                const m = new THREE.Mesh(new THREE.BoxGeometry(sx, h, sy), mat(color));
                m.position.set(x, h / 2, -y); m.name = name || '';
                worldScene.add(m); return m;
            };
            const ground = new THREE.Mesh(new THREE.PlaneGeometry(g.size, g.size),
                new THREE.MeshLambertMaterial({ color: 0x1d2b1f }));
            ground.rotation.x = -Math.PI / 2; ground.name = 'ground'; worldScene.add(ground);
            (g.walls || []).forEach((wl, i) => addBox(wl.x, wl.y, wl.sx, wl.sy, wl.h, 0x6b5f7a, 'wall' + i));
            (g.furniture || []).forEach(f => {
                if (f.kind === 'static_box') addBox(f.x, f.y, f.sx, f.sy, f.sz, 0x7a6a4a, f.id);
                else addBox(f.x, f.y, f.sx, f.sy, f.sz, 0x997744, f.id);
            });
            (g.trees || []).forEach(t => {
                const trunk = new THREE.Mesh(new THREE.CylinderGeometry(0.25, 0.25, 2, 8), mat(0x5a4128));
                trunk.position.set(t.x, 1, -t.y); worldScene.add(trunk);
                const top = new THREE.Mesh(new THREE.SphereGeometry(1.4, 10, 8), mat(0x2d5a27));
                top.position.set(t.x, 3, -t.y); worldScene.add(top);
            });
            (g.rocks || []).forEach(r => {
                const m = new THREE.Mesh(new THREE.SphereGeometry(r.r, 10, 8), mat(0x777788));
                m.position.set(r.x, r.r * 0.7, -r.y); worldScene.add(m);
            });
            const water = g.water;
            const lake = new THREE.Mesh(new THREE.PlaneGeometry(water.sx, water.sy),
                new THREE.MeshLambertMaterial({ color: 0x1d4e6b, transparent: true, opacity: 0.8 }));
            lake.rotation.x = -Math.PI / 2; lake.position.set(water.cx, 0.02, -water.cy);
            worldScene.add(lake);
            refreshWorldFoods(g);
        }

        function refreshWorldFoods(g) {
            Object.keys(worldChars).forEach(k => { if (k.startsWith('food_')) { worldScene.remove(worldChars[k]); delete worldChars[k]; } });
            (g.foods || []).forEach(f => {
                const m = new THREE.Mesh(new THREE.SphereGeometry(0.18, 8, 8),
                    new THREE.MeshLambertMaterial({ color: 0x33cc55 }));
                m.position.set(f.x, 0.25, -f.y); m.name = f.id;
                worldScene.add(m); worldChars['food_' + f.id] = m;
            });
        }

        function syncWorldChars(chars) {
            chars.forEach(c => {
                let m = worldChars[c.name];
                if (!m) {
                    const g = new THREE.Group();
                    // r128 has no CapsuleGeometry: cylinder + 2 spheres = capsule
                    const bmat = new THREE.MeshLambertMaterial(
                        { color: c.name === 'hero' ? 0x33aaff : 0xffaa33 });
                    const torso = new THREE.Mesh(new THREE.CylinderGeometry(0.3, 0.3, 0.9, 10), bmat);
                    torso.position.y = 0.75; g.add(torso);
                    const capB = new THREE.Mesh(new THREE.SphereGeometry(0.3, 10, 8), bmat);
                    capB.position.y = 0.3; g.add(capB);
                    const capT = new THREE.Mesh(new THREE.SphereGeometry(0.3, 10, 8), bmat);
                    capT.position.y = 1.2; g.add(capT);
                    const head = new THREE.Mesh(new THREE.SphereGeometry(0.22, 10, 8),
                        new THREE.MeshLambertMaterial({ color: 0xe8c39a }));
                    head.position.y = 1.6; g.add(head);
                    const eye = new THREE.Mesh(new THREE.SphereGeometry(0.05, 6, 6),
                        new THREE.MeshBasicMaterial({ color: 0x00f0ff }));
                    eye.position.set(0, 1.62, 0.2); g.add(eye);
                    if (worldDebug) {
                        const halo = new THREE.Mesh(new THREE.SphereGeometry(0.42, 8, 8),
                            new THREE.MeshBasicMaterial({ color: 0x00f0ff, wireframe: true, transparent: true, opacity: 0.5 }));
                        halo.position.y = 0.75; g.add(halo);
                    }
                    m = g; worldScene.add(m); worldChars[c.name] = m;
                }
                const p = worldScalePos(c.pos);
                m.position.set(p[0], p[1] - 0.75 + 0.0, p[2]);
                m.position.y = c.pos[2] - 0.75;
                m.rotation.y = -c.yaw;
            });
        }

        function updateWorldCamera() {
            const hero = (worldState && worldState.characters || []).find(c => c.name === 'hero');
            if (!hero) return;
            const p = worldScalePos(hero.pos);
            if (worldCamMode === 'first') {
                worldCamera.position.set(p[0], p[1] + 0.7, p[2]);
                const look = [p[0] + Math.sin(hero.yaw) * 3, p[1] + 0.5, p[2] + Math.cos(hero.yaw) * 3];
                worldCamera.lookAt(look[0], look[1], look[2]);
            } else if (worldCamMode === 'third') {
                worldCamera.position.set(p[0] - Math.sin(hero.yaw) * 4, p[1] + 2.5, p[2] - Math.cos(hero.yaw) * 4);
                worldCamera.lookAt(p[0], p[1] + 0.8, p[2]);
            } else if (worldCamMode === 'observer') {
                worldCamera.position.set(0, 34, -0.01);
                worldCamera.lookAt(0, 0, 0);
            }
        }

        function drawWorldMinimap(chars) {
            const cv = document.getElementById('world-minimap');
            if (!cv) return;
            const ctx = cv.getContext('2d');
            const S = 220, W = (worldGeom && worldGeom.size) || 40;
            const px = (x, y) => [(x + W / 2) / W * S, (y + W / 2) / W * S];
            ctx.fillStyle = '#0a0e16'; ctx.fillRect(0, 0, S, S);
            if (worldGeom && worldGeom.water) {
                const [ax, ay] = px(worldGeom.water.cx - worldGeom.water.sx / 2, worldGeom.water.cy - worldGeom.water.sy / 2);
                ctx.fillStyle = '#1d4e6b';
                ctx.fillRect(ax, ay, worldGeom.water.sx / W * S, worldGeom.water.sy / W * S);
            }
            (chars || []).forEach(c => {
                const [ax, ay] = px(c.pos[0], c.pos[1]);
                ctx.fillStyle = c.name === 'hero' ? '#00f0ff' : '#ffaa33';
                ctx.beginPath(); ctx.arc(ax, ay, 4, 0, 7); ctx.fill();
            });
        }

        async function fetchWorld() {
            try {
                if (!worldGeom) {
                    const g = await fetch('/api/v1/world/geometry').then(r => r.json());
                    buildWorldGeometry(g);
                } else {
                    const g = await fetch('/api/v1/world/geometry').then(r => r.json());
                    refreshWorldFoods(g);
                }
                const s = await fetch('/api/v1/world/state').then(r => r.json());
                worldState = s;
                syncWorldChars(s.characters);
                drawWorldMinimap(s.characters);
                const hero = s.characters.find(c => c.name === 'hero') || {};
                const ag = await fetch('/api/v1/world/agent?name=hero').then(r => r.json()).catch(() => null);
                document.getElementById('w-goal').innerText = (ag && ag.body.goal) || hero.goal || '—';
                document.getElementById('w-needs').innerText =
                    `E ${(ag && ag.body.base.energy || 0).toFixed(2)} · H ${(ag && ag.body.hunger || 0).toFixed(2)}`;
                document.getElementById('w-loc').innerText =
                    hero.pos ? `${hero.pos[0].toFixed(1)}, ${hero.pos[1].toFixed(1)}` : '—';
                document.getElementById('w-time').innerText =
                    `t${s.tick} · day ${s.day_fraction.toFixed(2)}${s.is_night ? ' · night' : ''} · ${s.state_hash.slice(0, 10)}…`;
                const md = await fetch('/api/v1/models').then(r => r.json()).catch(() => null);
                document.getElementById('w-models').innerText = md
                    ? Object.entries(md.manager.loaded).map(([k, v]) => k).join(', ') || 'all unloaded'
                    : '—';
                const fr = await fetch('/api/v1/world/friends').then(r => r.json()).catch(() => null);
                document.getElementById('w-friends').innerHTML = fr
                    ? fr.friends.map(f => `<div>${f.name}: trust ${f.trust_hero === null ? 'n/a' : f.trust_hero.toFixed(2)} · E ${f.energy}</div>`).join('')
                    : '';
            } catch (e) { showToast('World unreachable'); }
        }

        async function worldStep(n) {
            await fetch(`/api/v1/world/step?ticks=${n}`, { method: 'POST' });
            fetchWorld();
        }

        async function worldDream(withImage) {
            showToast(withImage ? 'Dreaming with image (slow)…' : 'Dreaming…');
            const res = await fetch(`/api/v1/world/dream?with_image=${withImage}`, { method: 'POST' });
            const data = await res.json();
            showToast(data.narrative ? `Dream: ${data.narrative.slice(0, 100)}…` : (data.detail || 'dream failed'));
        }

        async function worldSay() {
            const text = document.getElementById('w-say-text').value.trim();
            if (!text) return;
            const res = await fetch(`/api/v1/world/speak?text=${encodeURIComponent(text)}`, { method: 'POST' });
            const data = await res.json();
            showToast(data.wav ? `Spoke ${data.seconds}s` : (data.detail || 'TTS unavailable'));
        }

        async function worldImagine() {
            const theme = document.getElementById('w-imagine-text').value.trim() || 'the forest at night';
            showToast('Imagining (slow)…');
            const res = await fetch(`/api/v1/world/imagine?theme=${encodeURIComponent(theme)}`, { method: 'POST' });
            const data = await res.json();
            showToast(data.image_path ? `Imagined: ${data.image_path}` : (data.detail || 'imagination unavailable'));
        }

        // Diagnostics Tab
        async function fetchDiagnostics() {
            const res = await fetch('/api/diagnostics');
            const data = await res.json();

            const sysGrid = document.getElementById('diag-system-grid');
            sysGrid.innerHTML = `
                <div><div style="font-size: 11px; color: var(--text-dim);">Operating System</div><div style="font-size: 14px; font-weight: bold;">${data.system.os}</div></div>
                <div><div style="font-size: 11px; color: var(--text-dim);">CPU Processor</div><div style="font-size: 14px; font-weight: bold;">${data.system.cpu} (${data.system.cpu_cores} cores)</div></div>
                <div><div style="font-size: 11px; color: var(--text-dim);">Physical Memory</div><div style="font-size: 14px; font-weight: bold;">${data.system.ram_available_gb} GB / ${data.system.ram_total_gb} GB (${data.system.ram_percent}% used)</div></div>
                <div><div style="font-size: 11px; color: var(--text-dim);">Vulkan Compute Device</div><div style="font-size: 14px; font-weight: bold; color: var(--accent-cyan);">${data.vulkan.device_name || 'N/A'}</div></div>
                <div><div style="font-size: 11px; color: var(--text-dim);">Device Classification</div><div style="font-size: 14px; font-weight: bold;">${data.vulkan.device_type_str || 'N/A'}</div></div>
                <div><div style="font-size: 11px; color: var(--text-dim);">Vulkan Status</div><div style="font-size: 14px; font-weight: bold; color: var(--accent-green);">${data.vulkan.status || 'UNAVAILABLE'}</div></div>
            `;

            const hashTbody = document.getElementById('diag-hashes-tbody');
            hashTbody.innerHTML = `
                <tr><td>MaleCNS Soma Coordinates File (2023-27-2 soma_sides.csv)</td><td style="font-family: monospace; font-size: 11px;">${data.dataset_provenance.soma_sha256}</td></tr>
                <tr><td>MaleCNS Synaptic Connections Table (malecns_v1_0_connections.csv)</td><td style="font-family: monospace; font-size: 11px;">${data.dataset_provenance.connections_sha256}</td></tr>
                <tr><td>Compiled SPIR-V Brain Compute Shader (brain_step.spv)</td><td style="font-family: monospace; font-size: 11px;">${data.dataset_provenance.brain_shader_sha256}</td></tr>
                <tr><td>Compiled SPIR-V Plasticity Compute Shader (plasticity.spv)</td><td style="font-family: monospace; font-size: 11px;">${data.dataset_provenance.plasticity_shader_sha256}</td></tr>
                <tr><td>Active Git Code Commit</td><td style="font-family: monospace; font-size: 11px;">${data.system.git_commit}</td></tr>
            `;
        }

"""Static HTML for the lightweight browser demo."""

# ruff: noqa: E501

from __future__ import annotations

from textwrap import dedent


def render_demo_page() -> str:
    return dedent(
        """\
        <!doctype html>
        <html lang="en">
        <head>
          <meta charset="utf-8">
          <meta name="viewport" content="width=device-width, initial-scale=1">
          <title>Kryptonite Speaker Demo</title>
          <style>
            :root {
              --bg: #f4f1ea;
              --surface: rgba(255, 250, 242, 0.92);
              --surface-strong: rgba(255, 255, 255, 0.98);
              --ink: #1d1f1a;
              --muted: #5d6158;
              --line: rgba(36, 39, 31, 0.12);
              --accent: #0f766e;
              --accent-soft: rgba(15, 118, 110, 0.12);
              --warn: #9a3412;
              --danger: #b91c1c;
              --shadow: 0 22px 60px rgba(49, 44, 37, 0.12);
              --radius: 22px;
            }

            * { box-sizing: border-box; }

            body {
              margin: 0;
              min-height: 100vh;
              color: var(--ink);
              font-family: "IBM Plex Sans", "Segoe UI", sans-serif;
              background:
                radial-gradient(circle at top left, rgba(15, 118, 110, 0.18), transparent 32%),
                radial-gradient(circle at top right, rgba(217, 119, 6, 0.14), transparent 28%),
                linear-gradient(180deg, #f8f3eb 0%, var(--bg) 56%, #efe7db 100%);
            }

            .shell {
              width: min(1180px, calc(100vw - 28px));
              margin: 0 auto;
              padding: 32px 0 56px;
            }

            .hero {
              display: grid;
              gap: 18px;
              padding: 28px;
              border: 1px solid var(--line);
              border-radius: calc(var(--radius) + 4px);
              background:
                linear-gradient(145deg, rgba(255, 255, 255, 0.96), rgba(253, 246, 238, 0.86)),
                var(--surface);
              box-shadow: var(--shadow);
            }

            .eyebrow {
              display: inline-flex;
              align-items: center;
              gap: 8px;
              width: fit-content;
              padding: 8px 12px;
              border-radius: 999px;
              background: var(--accent-soft);
              color: var(--accent);
              font-size: 0.82rem;
              font-weight: 700;
              text-transform: uppercase;
              letter-spacing: 0.08em;
            }

            h1 {
              margin: 0;
              font-family: "IBM Plex Serif", Georgia, serif;
              font-size: clamp(2.2rem, 4vw, 4rem);
              line-height: 0.98;
              max-width: 12ch;
            }

            .hero p {
              margin: 0;
              max-width: 64ch;
              color: var(--muted);
              font-size: 1rem;
              line-height: 1.65;
            }

            .toolbar {
              display: flex;
              flex-wrap: wrap;
              gap: 12px;
              align-items: center;
            }

            .badge-row,
            .list,
            .stat-grid,
            .card-grid {
              display: grid;
              gap: 12px;
            }

            .badge-row {
              grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            }

            .badge,
            .mini-card,
            .result,
            .list-item {
              padding: 14px 16px;
              border-radius: 18px;
              border: 1px solid var(--line);
              background: var(--surface-strong);
            }

            .badge strong,
            .mini-card strong,
            .result strong {
              display: block;
              margin-bottom: 6px;
              font-size: 0.75rem;
              letter-spacing: 0.08em;
              text-transform: uppercase;
              color: var(--muted);
            }

            .badge span,
            .mini-card span,
            .result span {
              display: block;
              font-size: 0.98rem;
              font-weight: 600;
            }

            .layout {
              display: grid;
              gap: 18px;
              margin-top: 20px;
              grid-template-columns: repeat(auto-fit, minmax(300px, 1fr));
            }

            .panel {
              display: grid;
              gap: 16px;
              align-content: start;
              padding: 24px;
              border-radius: var(--radius);
              border: 1px solid var(--line);
              background: rgba(255, 252, 247, 0.88);
              box-shadow: var(--shadow);
            }

            .panel header {
              display: grid;
              gap: 6px;
            }

            h2 {
              margin: 0;
              font-family: "IBM Plex Serif", Georgia, serif;
              font-size: 1.6rem;
            }

            .panel p,
            .panel li,
            label span,
            .footnote {
              margin: 0;
              color: var(--muted);
              line-height: 1.55;
            }

            form {
              display: grid;
              gap: 12px;
            }

            label {
              display: grid;
              gap: 6px;
            }

            input,
            select,
            button,
            textarea {
              width: 100%;
              border-radius: 14px;
              border: 1px solid rgba(40, 42, 37, 0.18);
              padding: 12px 14px;
              font: inherit;
              color: var(--ink);
              background: rgba(255, 255, 255, 0.94);
            }

            input:focus,
            select:focus,
            button:focus,
            textarea:focus {
              outline: 2px solid rgba(15, 118, 110, 0.2);
              outline-offset: 2px;
              border-color: rgba(15, 118, 110, 0.36);
            }

            button {
              border: none;
              cursor: pointer;
              font-weight: 700;
              background: linear-gradient(135deg, #0f766e, #155e75);
              color: #f7fbfb;
              transition: transform 140ms ease, opacity 140ms ease;
            }

            button:hover { transform: translateY(-1px); }
            button:disabled { opacity: 0.6; cursor: progress; transform: none; }

            .secondary {
              background: rgba(32, 34, 30, 0.08);
              color: var(--ink);
            }

            .inline {
              display: grid;
              gap: 12px;
              grid-template-columns: repeat(auto-fit, minmax(150px, 1fr));
            }

            .hint {
              font-size: 0.92rem;
              color: var(--muted);
            }

            .status {
              display: inline-flex;
              align-items: center;
              gap: 8px;
              width: fit-content;
              padding: 8px 12px;
              border-radius: 999px;
              font-size: 0.8rem;
              font-weight: 700;
              letter-spacing: 0.04em;
              text-transform: uppercase;
            }

            .status.match {
              background: rgba(15, 118, 110, 0.14);
              color: var(--accent);
            }

            .status.no-match {
              background: rgba(185, 28, 28, 0.12);
              color: var(--danger);
            }

            .status.pending {
              background: rgba(217, 119, 6, 0.14);
              color: #9a3412;
            }

            .list {
              grid-template-columns: 1fr;
            }

            .list-item {
              display: grid;
              gap: 8px;
            }

            .list-item button {
              width: auto;
              justify-self: start;
              padding: 10px 14px;
            }

            .result {
              display: grid;
              gap: 10px;
              min-height: 116px;
            }

            .metric-row {
              display: grid;
              gap: 8px;
              grid-template-columns: repeat(auto-fit, minmax(120px, 1fr));
            }

            code {
              padding: 2px 6px;
              border-radius: 8px;
              background: rgba(20, 24, 20, 0.08);
              font-family: "IBM Plex Mono", monospace;
              font-size: 0.9em;
            }

            .footer-note {
              margin-top: 18px;
              font-size: 0.92rem;
              color: var(--muted);
            }

            @media (max-width: 720px) {
              .shell { width: min(100vw - 18px, 1180px); padding-top: 18px; }
              .hero, .panel { padding: 20px; }
            }
          </style>
        </head>
        <body>
          <main class="shell">
            <section class="hero">
              <span class="eyebrow">Kryptonite Mini Demo</span>
              <h1>Compare voices or verify against saved enrollments.</h1>
              <p>
                The page runs on top of the same FastAPI + <code>Inferencer</code> contract as the
                JSON API. Upload WAV, FLAC, or MP3 audio, inspect the score and threshold
                decision, and keep a small persistent enrollment set for repeated checks.
              </p>
              <div class="toolbar">
                <button id="refresh-state" type="button" class="secondary">Refresh runtime state</button>
                <span id="top-status" class="status pending">Loading runtime state</span>
              </div>
              <div id="runtime-badges" class="badge-row"></div>
            </section>

            <section class="layout">
              <article class="panel">
                <header>
                  <h2>Quick Compare</h2>
                  <p>Upload two files and get an immediate score + threshold decision without saving an enrollment.</p>
                </header>
                <form id="compare-form">
                  <label>
                    <span>Left audio</span>
                    <input id="compare-left" type="file" accept=".wav,.flac,.mp3,audio/*" required>
                  </label>
                  <label>
                    <span>Right audio</span>
                    <input id="compare-right" type="file" accept=".wav,.flac,.mp3,audio/*" required>
                  </label>
                  <div class="inline">
                    <label>
                      <span>Decision threshold</span>
                      <input id="compare-threshold" type="number" min="-1" max="1" step="0.000001">
                    </label>
                    <label>
                      <span>Normalize embeddings</span>
                      <select id="compare-normalize">
                        <option value="true">true</option>
                        <option value="false">false</option>
                      </select>
                    </label>
                  </div>
                  <button id="compare-submit" type="submit">Run comparison</button>
                </form>
                <div id="compare-result" class="result">
                  <span class="hint">No comparison has been run yet.</span>
                </div>
              </article>

              <article class="panel">
                <header>
                  <h2>Enroll Speaker</h2>
                  <p>Create or replace an enrollment entry using one or more audio files. Enrollments persist through the runtime store.</p>
                </header>
                <form id="enroll-form">
                  <label>
                    <span>Enrollment ID</span>
                    <input id="enroll-id" type="text" placeholder="speaker_charlie" required>
                  </label>
                  <label>
                    <span>Enrollment audio</span>
                    <input id="enroll-files" type="file" accept=".wav,.flac,.mp3,audio/*" multiple required>
                  </label>
                  <button id="enroll-submit" type="submit">Save enrollment</button>
                </form>
                <div id="enroll-result" class="result">
                  <span class="hint">The server starts with any compatible cached enrollments already loaded.</span>
                </div>
              </article>

              <article class="panel">
                <header>
                  <h2>Verify Probe</h2>
                  <p>Pick a saved enrollment, upload a probe file, and inspect the score, backend, and latency for the verification call.</p>
                </header>
                <form id="verify-form">
                  <label>
                    <span>Enrollment</span>
                    <select id="verify-enrollment" required></select>
                  </label>
                  <label>
                    <span>Probe audio</span>
                    <input id="verify-file" type="file" accept=".wav,.flac,.mp3,audio/*" required>
                  </label>
                  <div class="inline">
                    <label>
                      <span>Decision threshold</span>
                      <input id="verify-threshold" type="number" min="-1" max="1" step="0.000001">
                    </label>
                    <label>
                      <span>Normalize embeddings</span>
                      <select id="verify-normalize">
                        <option value="true">true</option>
                        <option value="false">false</option>
                      </select>
                    </label>
                  </div>
                  <button id="verify-submit" type="submit">Run verification</button>
                </form>
                <div id="verify-result" class="result">
                  <span class="hint">Pick a cached or newly created enrollment and upload a probe file.</span>
                </div>
              </article>

              <article class="panel">
                <header>
                  <h2>Loaded Enrollments</h2>
                  <p>The runtime view below merges the offline enrollment cache and any runtime-store overrides.</p>
                </header>
                <div id="enrollment-list" class="list">
                  <span class="hint">Loading enrollments…</span>
                </div>
                <p class="footer-note" id="threshold-footnote"></p>
              </article>
            </section>
          </main>
          <script>
            const state = { payload: null };

            async function loadRuntimeState() {
              const response = await fetch("/demo/api/state");
              const payload = await response.json();
              if (!response.ok) {
                throw new Error(payload.message || "Failed to load runtime state.");
              }
              state.payload = payload;
              renderRuntimeState(payload);
              return payload;
            }

            function renderRuntimeState(payload) {
              const runtimeBadges = document.getElementById("runtime-badges");
              const topStatus = document.getElementById("top-status");
              const verifyEnrollment = document.getElementById("verify-enrollment");
              const thresholdFootnote = document.getElementById("threshold-footnote");
              const thresholdValue = Number(payload.threshold.value).toFixed(6);

              topStatus.className = "status " + (payload.service.status === "ok" ? "match" : "pending");
              topStatus.textContent = payload.service.status === "ok" ? "Runtime ready" : "Runtime degraded";

              runtimeBadges.innerHTML = [
                metricCard("Backend", payload.service.selected_backend),
                metricCard("Impl", payload.service.inferencer.implementation || "unknown"),
                metricCard("Threshold", thresholdValue),
                metricCard("Enrollments", String(payload.enrollment_count)),
              ].join("");

              thresholdFootnote.textContent =
                "Threshold source: " + payload.threshold.source +
                (payload.threshold.origin_path ? " (" + payload.threshold.origin_path + ")" : "");

              document.getElementById("compare-threshold").value = thresholdValue;
              document.getElementById("verify-threshold").value = thresholdValue;

              verifyEnrollment.innerHTML = "";
              for (const entry of payload.enrollments) {
                const option = document.createElement("option");
                option.value = entry.enrollment_id;
                option.textContent = entry.enrollment_id + " (" + entry.sample_count + " files)";
                verifyEnrollment.appendChild(option);
              }
              renderEnrollmentList(payload.enrollments);
            }

            function renderEnrollmentList(entries) {
              const container = document.getElementById("enrollment-list");
              if (!entries.length) {
                container.innerHTML = '<span class="hint">No enrollments loaded.</span>';
                return;
              }
              container.innerHTML = entries.map((entry) => `
                <div class="list-item">
                  <div>
                    <strong>${escapeHtml(entry.enrollment_id)}</strong>
                    <div class="hint">${entry.sample_count} audio files · dim ${entry.embedding_dim}</div>
                  </div>
                  <button type="button" data-enrollment-id="${escapeHtml(entry.enrollment_id)}" class="secondary use-enrollment">
                    Use for verification
                  </button>
                </div>
              `).join("");

              container.querySelectorAll(".use-enrollment").forEach((button) => {
                button.addEventListener("click", () => {
                  document.getElementById("verify-enrollment").value = button.dataset.enrollmentId;
                  document.getElementById("enroll-id").value = button.dataset.enrollmentId;
                });
              });
            }

            function metricCard(label, value) {
              return `<div class="badge"><strong>${escapeHtml(label)}</strong><span>${escapeHtml(value)}</span></div>`;
            }

            function renderActionResult(targetId, payload) {
              const target = document.getElementById(targetId);
              const decisionClass = payload.decision === undefined
                ? "pending"
                : (payload.decision ? "match" : "no-match");
              const decisionLabel = payload.decision === undefined
                ? "Completed"
                : (payload.decision ? "Match" : "No Match");

              const metrics = [];
              if (payload.score !== undefined) {
                metrics.push(metricCard("Score", Number(payload.score).toFixed(8)));
              }
              if (payload.threshold) {
                metrics.push(metricCard("Threshold", Number(payload.threshold.value).toFixed(6)));
              }
              if (payload.latency_ms !== undefined) {
                metrics.push(metricCard("Latency", Number(payload.latency_ms).toFixed(3) + " ms"));
              }
              if (payload.backend && payload.backend.backend_name) {
                metrics.push(metricCard("Runtime backend", payload.backend.backend_name));
              }
              if (payload.backend && payload.backend.implementation) {
                metrics.push(metricCard("Impl", payload.backend.implementation));
              }
              if (payload.enrollment_id) {
                metrics.push(metricCard("Enrollment", payload.enrollment_id));
              }

              const detailLines = [];
              if (payload.left_audio && payload.right_audio) {
                detailLines.push(`<span class="hint">${escapeHtml(payload.left_audio.audio_path)} vs ${escapeHtml(payload.right_audio.audio_path)}</span>`);
              }
              if (payload.probe_audio) {
                detailLines.push(`<span class="hint">Probe: ${escapeHtml(payload.probe_audio.audio_path)}</span>`);
              }
              if (payload.audio_items) {
                detailLines.push(`<span class="hint">Enrollment sources: ${payload.audio_items.map((item) => escapeHtml(item.audio_path)).join(", ")}</span>`);
              }
              if (payload.threshold && payload.threshold.source) {
                detailLines.push(`<span class="hint">Threshold source: ${escapeHtml(payload.threshold.source)}</span>`);
              }

              target.innerHTML = `
                <span class="status ${decisionClass}">${decisionLabel}</span>
                <div class="metric-row">${metrics.join("")}</div>
                ${detailLines.join("")}
              `;
            }

            function renderError(targetId, error) {
              const target = document.getElementById(targetId);
              target.innerHTML = `
                <span class="status no-match">Request failed</span>
                <span class="hint">${escapeHtml(error.message || String(error))}</span>
              `;
            }

            function setBusy(buttonId, busy, labelWhenBusy) {
              const button = document.getElementById(buttonId);
              if (!button.dataset.originalLabel) {
                button.dataset.originalLabel = button.textContent;
              }
              button.disabled = busy;
              button.textContent = busy ? labelWhenBusy : button.dataset.originalLabel;
            }

            function escapeHtml(value) {
              return String(value)
                .replaceAll("&", "&amp;")
                .replaceAll("<", "&lt;")
                .replaceAll(">", "&gt;")
                .replaceAll('"', "&quot;")
                .replaceAll("'", "&#39;");
            }

            function fileToBase64(file) {
              return new Promise((resolve, reject) => {
                const reader = new FileReader();
                reader.onload = () => {
                  const raw = String(reader.result || "");
                  const [, payload = ""] = raw.split(",", 2);
                  resolve(payload);
                };
                reader.onerror = () => reject(new Error("Failed to read " + file.name));
                reader.readAsDataURL(file);
              });
            }

            function parseOptionalNumber(elementId) {
              const raw = document.getElementById(elementId).value.trim();
              return raw === "" ? null : Number(raw);
            }

            async function fileToPayload(file) {
              return {
                filename: file.name,
                content_base64: await fileToBase64(file),
              };
            }

            async function postJson(url, payload) {
              const response = await fetch(url, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify(payload),
              });
              const body = await response.json();
              if (!response.ok) {
                throw new Error(body.message || "Request failed.");
              }
              return body;
            }

            async function handleCompare(event) {
              event.preventDefault();
              const leftFile = document.getElementById("compare-left").files[0];
              const rightFile = document.getElementById("compare-right").files[0];
              if (!leftFile || !rightFile) {
                renderError("compare-result", new Error("Select both audio files."));
                return;
              }
              setBusy("compare-submit", true, "Comparing…");
              try {
                const payload = await postJson("/demo/api/compare", {
                  left_audio: await fileToPayload(leftFile),
                  right_audio: await fileToPayload(rightFile),
                  threshold: parseOptionalNumber("compare-threshold"),
                  normalize: document.getElementById("compare-normalize").value === "true",
                  stage: "demo",
                });
                renderActionResult("compare-result", payload);
              } catch (error) {
                renderError("compare-result", error);
              } finally {
                setBusy("compare-submit", false, "Comparing…");
              }
            }

            async function handleEnroll(event) {
              event.preventDefault();
              const files = Array.from(document.getElementById("enroll-files").files || []);
              const enrollmentId = document.getElementById("enroll-id").value.trim();
              if (!enrollmentId || !files.length) {
                renderError("enroll-result", new Error("Provide an enrollment ID and at least one audio file."));
                return;
              }
              setBusy("enroll-submit", true, "Saving…");
              try {
                const audioFiles = [];
                for (const file of files) {
                  audioFiles.push(await fileToPayload(file));
                }
                const payload = await postJson("/demo/api/enroll", {
                  enrollment_id: enrollmentId,
                  audio_files: audioFiles,
                  stage: "demo",
                });
                renderActionResult("enroll-result", payload);
                await loadRuntimeState();
              } catch (error) {
                renderError("enroll-result", error);
              } finally {
                setBusy("enroll-submit", false, "Saving…");
              }
            }

            async function handleVerify(event) {
              event.preventDefault();
              const file = document.getElementById("verify-file").files[0];
              const enrollmentId = document.getElementById("verify-enrollment").value;
              if (!enrollmentId || !file) {
                renderError("verify-result", new Error("Choose an enrollment and upload one probe file."));
                return;
              }
              setBusy("verify-submit", true, "Verifying…");
              try {
                const payload = await postJson("/demo/api/verify", {
                  enrollment_id: enrollmentId,
                  audio_file: await fileToPayload(file),
                  threshold: parseOptionalNumber("verify-threshold"),
                  normalize: document.getElementById("verify-normalize").value === "true",
                  stage: "demo",
                });
                renderActionResult("verify-result", payload);
              } catch (error) {
                renderError("verify-result", error);
              } finally {
                setBusy("verify-submit", false, "Verifying…");
              }
            }

            document.getElementById("refresh-state").addEventListener("click", async () => {
              try {
                await loadRuntimeState();
              } catch (error) {
                renderError("verify-result", error);
              }
            });
            document.getElementById("compare-form").addEventListener("submit", handleCompare);
            document.getElementById("enroll-form").addEventListener("submit", handleEnroll);
            document.getElementById("verify-form").addEventListener("submit", handleVerify);

            loadRuntimeState().catch((error) => {
              document.getElementById("top-status").className = "status no-match";
              document.getElementById("top-status").textContent = "Failed to load state";
              renderError("verify-result", error);
            });
          </script>
        </body>
        </html>
        """
    )


__all__ = ["render_demo_page"]

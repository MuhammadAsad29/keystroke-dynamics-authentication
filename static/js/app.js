/**
 * Keystroke Dynamics Biometric Web App - Frontend Controller
 * High-Precision Event Timing Engine & REST Client
 */

// =============================================================================
// 1. PROMPTS & GLOBAL STATE
// =============================================================================

const BENCHMARK_PROMPTS = [
  // Classic Pangrams & Typing Benchmarks
  "the quick brown fox jumps over the lazy dog",
  "pack my box with five dozen liquor jugs",
  "how razorback jumping frogs can level six piqued jockeys",
  "sphinx of black quartz judge my vow",
  "the five boxing wizards jump quickly",
  "bright vixens jump dozy fowl quack",
  "jackdaws love my big sphinx of quartz",
  "crazy fredrick bought many very exquisite opal jewels",
  "we promptly judged antique ivory buckles for the next prize",
  "a quick movement of the enemy will jeopardize six gunboats",

  // Biometrics & AI Dynamics
  "keystroke dynamics behavioral biometric security",
  "machine learning authentication without passwords",
  "rhythm and finger latency create unique digital fingerprints",
  "secure one shot verification using siamese networks",
  "deep learning models analyze neuromuscular typing cadence",
  "continuous user authentication using timing interval metrics",
  "biometric verification compares hold time and flight time",
  "transformer self attention captures long term typing dependencies",
  "bidirectional recurrent networks classify user identities",
  "neuromuscular typing rhythm is difficult to spoof or forge",

  // Everyday Natural Sentences
  "artificial intelligence is transforming modern cyber defense",
  "please enter your confidential credentials to proceed safely",
  "software engineers write clean and maintainable code daily",
  "distributed systems require robust fault tolerance mechanisms",
  "cloud computing enables scalable and resilient web applications",
  "neural networks learn rich representations from raw sensor data",
  "data privacy and encryption safeguard digital communications",
  "fast network connections deliver seamless video streaming experiences",
  "the conference room is booked for the morning engineering sync",
  "always verify security certificates before submitting personal information",

  // Conversational & Business Phrasing
  "good morning everyone welcome to today's research seminar",
  "please review the pull request and submit your feedback",
  "the quick reaction of the team prevented the server outage",
  "our biometric accuracy achieved twenty three percent equal error rate",
  "statistical normalization standardizes feature scales across users",
  "the laboratory experiments confirm high behavioral repeatability",
  "great design combines intuitive usability with robust engineering",
  "we are evaluating model performance on benchmark test sets",
  "typing rhythm reflects individual cognitive processing speed",
  "cyber security analysts monitor network telemetry in real time",

  // Short & Intermediate Variations
  "behavioral biometrics verify users by their natural habits",
  "silence is golden but typing rhythm reveals the truth",
  "every keypress contains subtle micro timing variations",
  "authentication systems must balance convenience with security",
  "finger dwell time correlates with muscle memory and practice",
  "digraph transitions show finger movement between adjacent keys",
  "the future of digital identity relies on passive verification",
  "biometric signals provide continuous identity assurance",
  "predicting identity from keystrokes requires deep neural embeddings",
  "welcome to the keystroke dynamics biometric verification console"
];

let currentPromptIndex = 0;

// High-resolution key tracking structures
class KeyStrokeTracker {
  constructor(textareaElement, onStatsChange) {
    this.textarea = textareaElement;
    this.onStatsChange = onStatsChange;
    this.events = [];
    this.activeDownEvents = new Map(); // key -> press_time
    this.isTracking = false;

    this.initListeners();
  }

  initListeners() {
    this.textarea.addEventListener("keydown", (e) => this.handleKeyDown(e));
    this.textarea.addEventListener("keyup", (e) => this.handleKeyUp(e));
  }

  handleKeyDown(e) {
    // Ignore synthetic auto-repeats if key is held down
    if (e.repeat) return;

    const now = performance.now();
    const key = e.key;

    // Record the keydown timestamp
    this.activeDownEvents.set(key, now);
    this.isTracking = true;

    // Visual pulse
    this.textarea.classList.add("active-typing");
    this.updateCadenceVisualizer();
  }

  handleKeyUp(e) {
    const now = performance.now();
    const key = e.key;

    if (this.activeDownEvents.has(key)) {
      const pressTime = this.activeDownEvents.get(key);
      this.activeDownEvents.delete(key);

      // Record completed keystroke pair
      this.events.push({
        key: key,
        press_time: pressTime,
        release_time: now
      });

      if (this.onStatsChange) {
        this.onStatsChange(this.getLiveStats());
      }
    }

    setTimeout(() => {
      if (this.activeDownEvents.size === 0) {
        this.textarea.classList.remove("active-typing");
      }
    }, 150);
  }

  getLiveStats() {
    const count = this.events.length;
    if (count < 2) {
      return { count, meanHold: "--", meanFlight: "--", wpm: "--" };
    }

    let totalHold = 0;
    let totalFlight = 0;

    for (let i = 0; i < count; i++) {
      totalHold += (this.events[i].release_time - this.events[i].press_time);
      if (i > 0) {
        totalFlight += (this.events[i].press_time - this.events[i - 1].release_time);
      }
    }

    const meanHold = Math.round(totalHold / count);
    const meanFlight = Math.round(totalFlight / (count - 1));

    const firstPress = this.events[0].press_time;
    const lastRelease = this.events[count - 1].release_time;
    const durationMin = (lastRelease - firstPress) / 60000;
    const words = count / 5.0;
    const wpm = durationMin > 0.005 ? Math.round(words / durationMin) : "--";

    return { count, meanHold, meanFlight, wpm };
  }

  updateCadenceVisualizer() {
    const bar = document.getElementById("cadenceBar");
    if (!bar) return;
    const pct = Math.min(100, (this.events.length / 50) * 100);
    bar.style.setProperty("--w", `${pct}%`);
    bar.querySelector(":scope::after")?.style?.setProperty("width", `${pct}%`);
  }

  reset() {
    this.events = [];
    this.activeDownEvents.clear();
    this.textarea.value = "";
    this.isTracking = false;
    this.textarea.classList.remove("active-typing");
    if (this.onStatsChange) {
      this.onStatsChange({ count: 0, meanHold: "--", meanFlight: "--", wpm: "--" });
    }
  }

  getEvents() {
    return [...this.events];
  }
}

// =============================================================================
// 2. MAIN APPLICATION LOGIC
// =============================================================================

document.addEventListener("DOMContentLoaded", () => {
  // Elements
  const verifyInput = document.getElementById("verifyInput");
  const enrollInput = document.getElementById("enrollInput");
  const identifyInput = document.getElementById("identifyInput");

  // Trackers
  const verifyTracker = new KeyStrokeTracker(verifyInput, updateVerifyHud);
  const enrollTracker = new KeyStrokeTracker(enrollInput, updateEnrollHud);
  const identifyTracker = new KeyStrokeTracker(identifyInput, updateIdentifyHud);

  // State
  let recordedEnrollSamples = [];
  const requiredEnrollCount = 5;

  // Initialize UI
  initTabs();
  initPrompts();
  initThresholdSlider();
  loadProfiles();
  loadSystemStatus();

  // ---------------------------------------------------------------------------
  // Tab Switching
  // ---------------------------------------------------------------------------
  function initTabs() {
    const tabBtns = document.querySelectorAll(".tab-btn");
    tabBtns.forEach((btn) => {
      btn.addEventListener("click", () => {
        tabBtns.forEach((b) => b.classList.remove("active"));
        document.querySelectorAll(".tab-content").forEach((c) => c.classList.remove("active"));

        btn.classList.add("active");
        const targetId = btn.getAttribute("data-tab");
        const targetContent = document.getElementById(targetId);
        if (targetContent) {
          targetContent.classList.add("active");
        }
      });
    });
  }

  // ---------------------------------------------------------------------------
  // Prompts & Shuffling
  // ---------------------------------------------------------------------------
  function initPrompts() {
    const promptDisplay = document.getElementById("promptDisplay");
    const enrollPrompt = document.getElementById("enrollPromptText");
    const shuffleBtn = document.getElementById("changePromptBtn");

    const setPrompt = (text) => {
      if (promptDisplay) promptDisplay.textContent = text;
      if (enrollPrompt) enrollPrompt.textContent = text;
    };

    setPrompt(BENCHMARK_PROMPTS[0]);

    if (shuffleBtn) {
      shuffleBtn.addEventListener("click", () => {
        let nextIndex;
        do {
          nextIndex = Math.floor(Math.random() * BENCHMARK_PROMPTS.length);
        } while (nextIndex === currentPromptIndex && BENCHMARK_PROMPTS.length > 1);
        currentPromptIndex = nextIndex;
        setPrompt(BENCHMARK_PROMPTS[currentPromptIndex]);
        verifyTracker.reset();
      });
    }
  }

  // ---------------------------------------------------------------------------
  // Threshold Slider Control
  // ---------------------------------------------------------------------------
  function initThresholdSlider() {
    const slider = document.getElementById("thresholdSlider");
    const valDisplay = document.getElementById("thresholdVal");
    if (!slider || !valDisplay) return;

    slider.addEventListener("input", (e) => {
      valDisplay.textContent = parseFloat(e.target.value).toFixed(2);
    });
  }

  // ---------------------------------------------------------------------------
  // Profiles Loader
  // ---------------------------------------------------------------------------
  async function loadProfiles() {
    try {
      const res = await fetch("/api/profiles");
      const data = await res.json();
      const select = document.getElementById("verifyProfileSelect");
      const headerTag = document.getElementById("currentProfileName");
      if (!select) return;

      select.innerHTML = "";
      data.profiles.forEach((p, index) => {
        const opt = document.createElement("option");
        opt.value = p.key;
        opt.textContent = `${p.name} (${p.num_samples} samples${p.is_benchmark ? " • Benchmark" : ""})`;
        select.appendChild(opt);
      });

      if (data.profiles.length > 0) {
        if (headerTag) headerTag.textContent = data.profiles[0].name;
      }

      select.addEventListener("change", (e) => {
        const selected = data.profiles.find((p) => p.key === e.target.value);
        if (selected && headerTag) {
          headerTag.textContent = selected.name;
        }
      });
    } catch (err) {
      console.error("Failed to load profiles:", err);
    }
  }

  async function loadSystemStatus() {
    try {
      const res = await fetch("/api/status");
      const data = await res.json();
      const statusText = document.getElementById("systemStatus");
      if (statusText) {
        statusText.textContent = `Models Ready (${data.device.toUpperCase()})`;
      }
    } catch (err) {
      console.error("Status check failed:", err);
    }
  }

  // ---------------------------------------------------------------------------
  // HUD Update Callbacks
  // ---------------------------------------------------------------------------
  function updateVerifyHud(stats) {
    const hudCount = document.getElementById("verifyKeyCount");
    const hudHold = document.getElementById("hudHoldTime");
    const hudFlight = document.getElementById("hudFlightTime");
    const hudWpm = document.getElementById("hudWpm");
    const btnVerify = document.getElementById("btnVerify");

    if (hudCount) hudCount.textContent = `${stats.count} keys captured`;
    if (hudHold) hudHold.textContent = stats.meanHold !== "--" ? `${stats.meanHold} ms` : "-- ms";
    if (hudFlight) hudFlight.textContent = stats.meanFlight !== "--" ? `${stats.meanFlight} ms` : "-- ms";
    if (hudWpm) hudWpm.textContent = stats.wpm !== "--" ? `${stats.wpm} WPM` : "-- WPM";

    // Enable verify button if minimum 8 keystrokes captured
    if (btnVerify) {
      btnVerify.disabled = stats.count < 8;
    }
  }

  function updateEnrollHud(stats) {
    const hudCount = document.getElementById("enrollKeyCount");
    const btnRecord = document.getElementById("btnRecordSample");

    if (hudCount) hudCount.textContent = `${stats.count} keys captured`;
    if (btnRecord) {
      btnRecord.disabled = stats.count < 8;
    }
  }

  function updateIdentifyHud(stats) {
    const hudCount = document.getElementById("identifyKeyCount");
    const btnIdentify = document.getElementById("btnIdentify");

    if (hudCount) hudCount.textContent = `${stats.count} keys captured`;
    if (btnIdentify) {
      btnIdentify.disabled = stats.count < 8;
    }
  }

  // ---------------------------------------------------------------------------
  // TAB 1: 1:1 VERIFICATION ACTION
  // ---------------------------------------------------------------------------
  const btnVerify = document.getElementById("btnVerify");
  const btnClearVerify = document.getElementById("btnClearVerify");

  if (btnVerify) {
    btnVerify.addEventListener("click", async () => {
      const events = verifyTracker.getEvents();
      if (events.length < 8) return;

      const profileKey = document.getElementById("verifyProfileSelect").value;
      const threshold = parseFloat(document.getElementById("thresholdSlider").value);

      btnVerify.disabled = true;
      btnVerify.innerHTML = `<span class="btn-icon">⏳</span> Analyzing Dynamics...`;

      try {
        const response = await fetch("/api/verify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            profile_key: profileKey,
            threshold: threshold,
            events: events
          })
        });

        const result = await response.json();
        renderVerificationResult(result);
      } catch (err) {
        alert("Error connecting to verification server: " + err.message);
      } finally {
        btnVerify.disabled = false;
        btnVerify.innerHTML = `<span class="btn-icon">⚡</span> Verify Biometric Rhythm`;
      }
    });
  }

  if (btnClearVerify) {
    btnClearVerify.addEventListener("click", () => {
      verifyTracker.reset();
      resetVerificationResultCard();
    });
  }

  function renderVerificationResult(result) {
    const banner = document.getElementById("verdictBanner");
    const icon = document.getElementById("verdictIcon");
    const title = document.getElementById("verdictTitle");
    const desc = document.getElementById("verdictDesc");
    const badge = document.getElementById("resultStatusBadge");
    const metricsContainer = document.getElementById("metricsContainer");

    const metricDistance = document.getElementById("metricDistance");
    const metricThresholdTag = document.getElementById("metricThresholdTag");
    const distanceProgress = document.getElementById("distanceProgress");

    const metricSimilarity = document.getElementById("metricSimilarity");
    const similarityProgress = document.getElementById("similarityProgress");

    const statCount = document.getElementById("statCount");
    const statHold = document.getElementById("statHold");
    const statFlight = document.getElementById("statFlight");
    const statWpm = document.getElementById("statWpm");
    const statProfile = document.getElementById("statProfile");

    // Clear previous classes
    banner.classList.remove("verified", "impostor");

    if (result.is_verified) {
      banner.classList.add("verified");
      icon.textContent = "✅";
      title.textContent = "Verified — Access Granted";
      desc.textContent = result.status_text;
      badge.className = "badge badge-emerald";
      badge.textContent = "Verified Genuine";
    } else {
      banner.classList.add("impostor");
      icon.textContent = "❌";
      title.textContent = "Impostor — Access Denied";
      desc.textContent = result.status_text;
      badge.className = "badge badge-crimson";
      badge.textContent = "Impostor Detected";
    }

    // Numbers & meters
    metricDistance.textContent = result.distance.toFixed(3);
    metricThresholdTag.textContent = `vs threshold ${result.threshold.toFixed(2)}`;

    // Distance progress bar: 0 at left, threshold at 50%
    const distPct = Math.min(100, Math.max(5, (result.distance / (result.threshold * 2)) * 100));
    distanceProgress.style.width = `${distPct}%`;
    distanceProgress.className = result.is_verified ? "progress-bar-fill fill-emerald" : "progress-bar-fill fill-crimson";

    metricSimilarity.textContent = `${result.similarity_score}%`;
    similarityProgress.style.width = `${result.similarity_score}%`;
    similarityProgress.className = result.is_verified ? "progress-bar-fill fill-emerald" : "progress-bar-fill fill-crimson";

    // Timing stats
    statCount.textContent = `${result.timing_stats.keystroke_count} keys`;
    statHold.textContent = `${result.timing_stats.mean_hold_ms} ms`;
    statFlight.textContent = `${result.timing_stats.mean_flight_ms} ms`;
    statWpm.textContent = `${result.timing_stats.wpm} WPM (${result.timing_stats.duration_sec}s)`;
    statProfile.textContent = result.target_profile;

    metricsContainer.style.display = "flex";
    metricsContainer.style.flexDirection = "column";
    metricsContainer.style.gap = "1rem";
  }

  function resetVerificationResultCard() {
    const banner = document.getElementById("verdictBanner");
    const icon = document.getElementById("verdictIcon");
    const title = document.getElementById("verdictTitle");
    const desc = document.getElementById("verdictDesc");
    const badge = document.getElementById("resultStatusBadge");
    const metricsContainer = document.getElementById("metricsContainer");

    banner.classList.remove("verified", "impostor");
    icon.textContent = "⌨️";
    title.textContent = "Type & Verify";
    desc.textContent = "Type at least 8 keystrokes on the left console to evaluate biometric timing features.";
    badge.className = "badge badge-neutral";
    badge.textContent = "Awaiting Input";
    metricsContainer.style.display = "none";
  }

  // ---------------------------------------------------------------------------
  // TAB 2: LIVE ENROLLMENT ACTION
  // ---------------------------------------------------------------------------
  const btnRecordSample = document.getElementById("btnRecordSample");
  const btnResetEnroll = document.getElementById("btnResetEnroll");
  const btnSaveProfile = document.getElementById("btnSaveProfile");
  const btnGoToVerify = document.getElementById("btnGoToVerify");

  if (btnRecordSample) {
    btnRecordSample.addEventListener("click", () => {
      const events = enrollTracker.getEvents();
      if (events.length < 8) return;

      const sampleIndex = recordedEnrollSamples.length + 1;
      const stats = enrollTracker.getLiveStats();

      recordedEnrollSamples.push(events);
      enrollTracker.reset();

      renderSamplePill(sampleIndex, stats);
      updateEnrollProgress();
    });
  }

  if (btnResetEnroll) {
    btnResetEnroll.addEventListener("click", () => {
      recordedEnrollSamples = [];
      enrollTracker.reset();
      document.getElementById("sampleItems").innerHTML = "";
      document.getElementById("emptyEnrollState").style.display = "block";
      document.getElementById("enrollActionFooter").style.display = "none";
      document.getElementById("enrollSuccessBox").style.display = "none";
      updateEnrollProgress();
    });
  }

  function renderSamplePill(index, stats) {
    const emptyState = document.getElementById("emptyEnrollState");
    const container = document.getElementById("sampleItems");
    if (emptyState) emptyState.style.display = "none";

    const pill = document.createElement("div");
    pill.className = "sample-pill";
    pill.innerHTML = `
      <div>
        <strong>Sample #${index}</strong>
        <span class="text-muted" style="margin-left: 0.5rem;">${stats.count} keys</span>
      </div>
      <div style="font-family: var(--font-mono); font-size: 0.75rem; color: var(--cyan);">
        H: ${stats.meanHold}ms | UD: ${stats.meanFlight}ms | ${stats.wpm} WPM
      </div>
    `;
    container.appendChild(pill);
  }

  function updateEnrollProgress() {
    const count = recordedEnrollSamples.length;
    const label = document.getElementById("sampleCountLabel");
    const badge = document.getElementById("enrollStatusBadge");
    const dots = document.querySelectorAll(".step-dot");
    const actionFooter = document.getElementById("enrollActionFooter");

    if (label) label.textContent = `${count} of ${requiredEnrollCount} Samples`;
    if (badge) badge.textContent = `${count} Recorded`;

    dots.forEach((dot) => {
      const step = parseInt(dot.getAttribute("data-step"));
      dot.classList.remove("active", "done");
      if (step <= count) {
        dot.classList.add("done");
      } else if (step === count + 1) {
        dot.classList.add("active");
      }
    });

    if (count >= requiredEnrollCount) {
      actionFooter.style.display = "block";
    } else {
      actionFooter.style.display = "none";
    }
  }

  if (btnSaveProfile) {
    btnSaveProfile.addEventListener("click", async () => {
      if (recordedEnrollSamples.length < 3) return;

      const userName = document.getElementById("enrollName").value.trim() || "Live User";
      btnSaveProfile.disabled = true;
      btnSaveProfile.textContent = "⏳ Computing Siamese Centroid Embedding...";

      try {
        const response = await fetch("/api/enroll", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({
            user_name: userName,
            samples: recordedEnrollSamples
          })
        });

        const data = await response.json();
        if (data.success) {
          document.getElementById("enrollActionFooter").style.display = "none";
          document.getElementById("enrollSuccessBox").style.display = "flex";
          // Reload profiles dropdown and select new user
          await loadProfiles();
          const select = document.getElementById("verifyProfileSelect");
          if (select) {
            select.value = data.profile_key;
            document.getElementById("currentProfileName").textContent = data.profile_name;
          }
        } else {
          alert("Enrollment failed: " + data.error);
        }
      } catch (err) {
        alert("Enrollment network error: " + err.message);
      } finally {
        btnSaveProfile.disabled = false;
        btnSaveProfile.textContent = "🚀 Save Profile & Set as Active Target";
      }
    });
  }

  if (btnGoToVerify) {
    btnGoToVerify.addEventListener("click", () => {
      const verifyTabBtn = document.querySelector('[data-tab="tab-verify"]');
      if (verifyTabBtn) verifyTabBtn.click();
    });
  }

  // ---------------------------------------------------------------------------
  // TAB 3: 1:N IDENTIFICATION ACTION
  // ---------------------------------------------------------------------------
  const btnIdentify = document.getElementById("btnIdentify");
  const btnClearIdentify = document.getElementById("btnClearIdentify");

  if (btnIdentify) {
    btnIdentify.addEventListener("click", async () => {
      const events = identifyTracker.getEvents();
      if (events.length < 8) return;

      btnIdentify.disabled = true;
      btnIdentify.innerHTML = `<span class="btn-icon">⏳</span> Computing Classifier Logits...`;

      try {
        const response = await fetch("/api/identify", {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ events: events })
        });

        const result = await response.json();
        renderIdentificationResults(result);
      } catch (err) {
        alert("Identification error: " + err.message);
      } finally {
        btnIdentify.disabled = false;
        btnIdentify.innerHTML = `<span class="btn-icon">🎯</span> Predict User Identity (Top-5)`;
      }
    });
  }

  if (btnClearIdentify) {
    btnClearIdentify.addEventListener("click", () => {
      identifyTracker.reset();
      document.getElementById("resultsColumns").style.display = "none";
      document.getElementById("emptyIdentifyState").style.display = "block";
      document.getElementById("identifyStatusBadge").textContent = "Ready";
    });
  }

  function renderIdentificationResults(result) {
    const emptyState = document.getElementById("emptyIdentifyState");
    const resultsCols = document.getElementById("resultsColumns");
    const badge = document.getElementById("identifyStatusBadge");
    const transList = document.getElementById("transformerTop5List");
    const lstmList = document.getElementById("lstmTop5List");

    if (emptyState) emptyState.style.display = "none";
    if (resultsCols) resultsCols.style.display = "grid";
    if (badge) {
      badge.className = "badge badge-purple";
      badge.textContent = "Top-5 Predicted";
    }

    const renderList = (container, items, colorClass) => {
      container.innerHTML = "";
      items.forEach((item) => {
        const row = document.createElement("div");
        row.className = "ranking-row";
        row.innerHTML = `
          <div class="ranking-meta">
            <span class="ranking-rank">#${item.rank}</span>
            <span class="ranking-user">User ID: ${item.user_id}</span>
            <span class="ranking-prob">${item.probability}%</span>
          </div>
          <div class="ranking-bar-bg">
            <div class="ranking-bar-fill" style="width: ${Math.max(4, item.probability * 2)}%;"></div>
          </div>
        `;
        container.appendChild(row);
      });
    };

    renderList(transList, result.transformer_top5, "cyan");
    renderList(lstmList, result.lstm_top5, "purple");
  }
});

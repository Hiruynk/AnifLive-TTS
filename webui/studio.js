(() => {
  "use strict";

  const i18n = window.AnifLiveTTSStudioI18n;
  if (!i18n) throw new Error("AnifLive-TTS Studio i18n failed to load");

  const state = {
    locale: i18n.getLocale(),
    overview: null,
    status: null,
    projects: [],
    jobs: [],
    models: [],
    artifacts: [],
    qualifications: [],
    expressions: [],
    expressionDrafts: [],
    selectedExpression: null,
    selectedJobId: null,
    selectedArtifactId: null,
    selectedArtifactDetails: null,
    pendingArtifactPromotions: new Set(),
    selectedQualificationId: null,
    selectedDatasetId: null,
    datasetItems: [],
    datasetCapabilities: null,
    datasetProjectState: null,
    datasetReviewQueue: null,
    datasetExpressionCandidates: null,
    datasetCandidateAudio: null,
    autoImportedDatasetProcessJobs: new Set(),
    datasetProcessImportFailures: new Set(),
    autoImportedAcquisitionJobs: new Set(),
    datasetAcquisitionImportFailures: new Set(),
    selectedDatasetItemId: null,
    datasetWaveformRequest: 0,
    datasetLoading: false,
    selectedTseId: null,
    selectedTseJobId: null,
    tseReport: null,
    tseArtifacts: [],
    tseReviewDecisions: new Map(),
    tseSeparationSegments: [],
    tsePollTimer: 0,
    selectedTrainingId: null,
    selectedTrainingJobId: null,
    trainingJobDetail: null,
    trainingJobLogs: [],
    referenceReviewJobId: null,
    referenceReviewManifest: null,
    referenceReviewReport: null,
    referenceReviewLoading: false,
    referenceReviewAudio: null,
    referenceRejectArmed: false,
    selectedEvaluationId: null,
    selectedEvaluationJobId: null,
    evaluationJobDetail: null,
    evaluationJobLogs: [],
    evaluationReport: null,
    evaluationBenchmark: null,
    evaluationAudioLanguage: "ja",
    pendingDatasetActions: new Set(),
    pendingJobActions: new Set(),
    currentView: "overview",
    playbackActive: false,
    visibleHistoryClearedAt: null,
    components: [],
    componentDownloadActive: false,
    settings: {
      default_language: "ja",
      default_continuity_policy: "A",
      default_training_preset: "balanced",
      default_benchmark_language: "yue",
      default_tse_target_threshold: 0.72,
      auto_refresh_seconds: 5,
      overview_motion: true
    }
  };

  let pulseActive = false;
  let pulseFrame = 0;
  let pulseStartedAt = performance.now();
  let refreshTimer = 0;

  const moduleTitles = {
    overview: "Overview",
    synthesis: "Synthesis",
    expressions: "Expressions",
    datasets: "Dataset Factory",
    tse: "Target Speaker Extraction",
    training: "Model Training",
    evaluation: "Evaluation Lab",
    models: "Model Registry",
    engines: "TensorRT Engines",
    jobs: "GPU Jobs",
    settings: "Settings"
  };

  const byId = id => document.getElementById(id);
  const setText = (id, value) => {
    const node = byId(id);
    const source = String(value);
    if (node && node.textContent !== i18n.t(source)) i18n.setLocalizedText(node, source);
  };
  const create = (tag, className, text) => {
    const node = document.createElement(tag);
    if (className) node.className = className;
    if (text !== undefined) i18n.setLocalizedText(node, String(text));
    return node;
  };
  const createData = (tag, className, value) => {
    const node = create(tag, className);
    node.setAttribute("data-i18n-static", "");
    node.textContent = String(value);
    return node;
  };
  const jobControls = window.AnifLiveTTSJobControls;
  if (!jobControls) throw new Error("AnifLive-TTS Studio job controls failed to load");
  const datasetModel = window.AnifLiveTTSDatasetModel;
  if (!datasetModel) throw new Error("AnifLive-TTS Studio Dataset Factory model failed to load");
  const tseModel = window.AnifLiveTTSTSEModel;
  if (!tseModel) throw new Error("AnifLive-TTS Studio TSE model failed to load");
  const qualificationGateLabels = Object.freeze({
    multilingual: "Five languages",
    "speaker-identity": "Speaker identity",
    "streaming-parity": "Streaming parity",
    "long-form-continuity": "Long-form continuity",
    "expression-quality": "Expression quality",
    "tensorrt-runtime": "TensorRT runtime",
    "latency-performance": "TTFA and RTF",
    security: "Security"
  });
  let deferredStudioInstallPrompt = null;
  const promotableArtifactTypes = new Set(["expression-bank", "engine", "package"]);

  const jobProjectKinds = Object.freeze({
    "dataset.inventory": new Set(["dataset"]),
    "dataset.process": new Set(["dataset"]),
    "dataset.decode": new Set(["dataset"]),
    "dataset.target-speaker": new Set(["dataset"]),
    "dataset.separate": new Set(["dataset"]),
    "dataset.transcribe": new Set(["dataset"]),
    "dataset.finalize": new Set(["dataset"]),
    "tse.prepare": new Set(["tse"]),
    "training.prepare": new Set(["training"]),
    "checkpoint.select": new Set(["training"]),
    "reference.select": new Set(["training"]),
    "holdout.evaluate": new Set(["training"]),
    "evaluation.prepare": new Set(["evaluation"]),
    "engine.prepare": new Set(["training", "evaluation"]),
    "conversion.parity": new Set(["training", "evaluation"]),
    "model.package": new Set(["training", "evaluation"])
  });


  // Presentation-only UX: existing controls and API handlers remain the source of truth.
  const experience = (() => {
    const scrollPositions = new Map();
    const preferences = {};
    let settingsDirty = false;
    let settingsSaving = false;
    let settingsRevision = 0;
    let savingRevision = 0;
    const savePreference = (key, value) => {
      preferences[key] = value;
    };
    const reveal = field => {
      for (let parent = field.parentElement; parent; parent = parent.parentElement) {
        if (parent instanceof HTMLDetailsElement) parent.open = true;
      }
    };
    function rememberDetails(details, key) {
      details.dataset.uxDisclosure = key;
      if (typeof preferences[key] === "boolean") details.open = preferences[key];
      details.addEventListener("toggle", () => savePreference(key, details.open));
    }
    function group(key, title, nodes, parent, before = null) {
      const items = nodes.filter(Boolean);
      if (!parent || !items.length) return;
      const details = create("details", "ux-disclosure");
      const summary = create("summary");
      summary.append(create("span", "", title));
      const preview = create("span", "ux-settings-preview");
      preview.setAttribute("aria-hidden", "true");
      summary.append(preview);
      const content = create("div", "ux-disclosure-content");
      if (key === "workstationAdvanced") content.classList.add("settings-field-grid");
      details.append(summary, content);
      parent.insertBefore(details, before);
      items.forEach(node => content.append(node));
      const updatePreview = () => {
        const values = [...content.querySelectorAll("input:not([type=hidden]), select")]
          .filter(field => !field.disabled && field.type !== "checkbox")
          .map(field => {
            const label = field.labels?.[0]?.querySelector("span")?.textContent?.trim()
              || field.labels?.[0]?.textContent?.trim();
            const value = field instanceof HTMLSelectElement
              ? field.selectedOptions[0]?.textContent?.trim() : field.value;
            return label && value ? label + ": " + value : "";
          })
          .filter(Boolean);
        const next = values.join(" · ");
        if (preview.textContent !== next) preview.textContent = next;
      };
      details.addEventListener("input", updatePreview);
      details.addEventListener("change", updatePreview);
      window.addEventListener("aniflive-tts:locale-changed", () => requestAnimationFrame(updatePreview));
      details.addEventListener("toggle", updatePreview);
      rememberDetails(details, key);
      updatePreview();
    }
    function foldRecords(id, title) {
      const list = byId(id);
      if (!list || list.closest(".ux-records")) return;
      const details = create("details", "ux-disclosure ux-records");
      details.dataset.recordList = id;
      const summary = create("summary");
      const heading = list.previousElementSibling?.tagName === "H3"
        ? list.previousElementSibling : null;
      summary.append(heading || create("span", "", title));
      const count = create("span", "ux-record-count", "0");
      i18n.setLocalizedAttribute(count, "title", "Entries shown");
      summary.append(count);
      const content = create("div", "ux-disclosure-content");
      const scroller = create("div", "ux-record-body");
      scroller.tabIndex = 0;
      i18n.setLocalizedAttribute(scroller, "aria-label", title);
      scroller.setAttribute("role", "region");
      list.before(details);
      details.append(summary, content);
      content.append(scroller);
      scroller.append(list);
      const updateCount = () => {
        const value = String([...list.children].filter(node => node.tagName !== "P").length);
        if (count.textContent !== value) count.textContent = value;
      };
      new MutationObserver(updateCount).observe(list, { childList: true });
      updateCount();
      rememberDetails(details, "records:" + id);
    }
    function boundTable(id) {
      const body = byId(id);
      const surface = body?.closest(".table-wrap");
      if (!surface) return;
      const heading = surface.closest("section")?.querySelector("h2");
      surface.classList.add("ux-table-scroll");
      surface.dataset.scrollList = id;
      surface.tabIndex = 0;
      surface.setAttribute("role", "region");
      if (heading) {
        if (!heading.id) heading.id = "scroll-title-" + id;
        surface.setAttribute("aria-labelledby", heading.id);
      }
    }
    function organizeTables() {
      const layouts = {
        trainingRows: ["auto", "100px", "104px", "80px", "144px", "76px"],
        datasetRows: ["32%", "104px", "auto", "144px", "76px"],
        evaluationRows: ["auto", "104px", "96px", "112px", "144px", "76px"],
        evaluationArtifactRows: ["auto", "170px", "120px", "144px", "100px"],
        engineArtifactRows: ["auto", "104px", "120px", "132px", "84px"],
        expressionRows: ["auto", "132px", "112px", "160px"],
        expressionDraftRows: ["auto", "128px", "104px", "116px"],
        tseRows: ["auto", "100px", "124px", "88px", "144px", "76px"]
      };
      Object.entries(layouts).forEach(([id, widths]) => {
        const table = byId(id)?.closest("table");
        if (!table || table.querySelector("colgroup")) return;
        const columns = document.createElement("colgroup");
        widths.forEach(width => {
          const column = document.createElement("col");
          column.style.width = width;
          columns.append(column);
        });
        table.prepend(columns);
        table.dataset.columnLayout = id;
      });
    }

    function trackVisibleViewport() {
      const viewport = window.visualViewport;
      if (!viewport) return;
      let frame = 0;
      const update = () => {
        frame = 0;
        const inset = Math.max(0, window.innerHeight - viewport.height - viewport.offsetTop);
        const keyboard = window.innerWidth <= 860 && Math.abs(viewport.scale - 1) < .01 && inset > 120;
        document.body.classList.toggle("studio-keyboard-open", keyboard);
        if (!keyboard) {
          document.documentElement.style.removeProperty("--studio-usable-height");
          document.documentElement.style.removeProperty("--studio-keyboard-inset");
          return;
        }
        document.documentElement.style.setProperty("--studio-usable-height", (viewport.height + viewport.offsetTop) + "px");
        document.documentElement.style.setProperty("--studio-keyboard-inset", inset + "px");
        requestAnimationFrame(() => {
          let focused = document.activeElement;
          const embedded = focused instanceof HTMLIFrameElement ? focused : null;
          if (embedded) focused = embedded.contentDocument?.activeElement;
          if (focused?.matches("input,textarea,select")) {
            focused.scrollIntoView({ block: "nearest", behavior: "instant" });
            const region = document.querySelector(".viewport");
            const bounds = focused.getBoundingClientRect();
            const offset = embedded?.getBoundingClientRect().top || 0;
            const top = Math.max(viewport.offsetTop, region.getBoundingClientRect().top) + 12;
            const bottom = viewport.height + viewport.offsetTop - document.querySelector(".rail").getBoundingClientRect().height - 12;
            const delta = bounds.bottom + offset > bottom ? bounds.bottom + offset - bottom
              : bounds.top + offset < top ? bounds.top + offset - top : 0;
            if (delta) region.scrollTop += delta;
          }
        });
      };
      const schedule = () => { if (!frame) frame = requestAnimationFrame(update); };
      viewport.addEventListener("resize", schedule);
      viewport.addEventListener("scroll", schedule);
      window.addEventListener("resize", schedule);
      schedule();
    }

    function initialize() {
      document.body.classList.add("studio-ux");
      const patterns = {
        overview: "cover", synthesis: "editor", expressions: "collection",
        datasets: "collection", tse: "editor", training: "collection",
        evaluation: "collection", models: "collection", engines: "collection",
        jobs: "collection", settings: "settings"
      };
      document.querySelectorAll("[data-view-panel]").forEach(panel => {
        panel.dataset.workspacePattern = patterns[panel.dataset.viewPanel];
      });
      organizeTables();
      trackVisibleViewport();

      [
        "evaluationArtifactRows", "engineArtifactRows", "trainingRows", "datasetRows",
        "evaluationRows", "expressionRows", "expressionDraftRows", "tseRows",
        "datasetItemRows", "modelRows", "jobRows"
      ].forEach(boundTable);
      const evidenceCount = byId("qualificationCount");
      const evidenceTitle = evidenceCount?.parentElement?.querySelector("h2");
      if (evidenceCount && evidenceTitle) {
        const titleLine = create("div", "ux-title-line");
        evidenceTitle.before(titleLine);
        evidenceCount.classList.add("ux-section-count");
        titleLine.append(evidenceTitle, evidenceCount);
      }

      [
        ["trainingEventList", "Worker events"],
        ["trainingArtifactList", "Verified outputs"],
        ["evaluationEventList", "Worker events"],
        ["evaluationRunArtifacts", "Verified artifacts"],
        ["tseJobEvents", "Worker events"],
        ["jobLogList", "Worker events"],
        ["artifactLineage", "Artifact lineage"]
      ].forEach(([id, title]) => foldRecords(id, title));

      group("expressionDirection", "Voice direction and metadata",
        [document.querySelector(".expression-vad"), byId("expressionProsody")?.closest("label")],
        byId("expressionDraftForm"), byId("expressionReferenceAnalysis"));
      const mobileWorkspaces = [];
      document.querySelectorAll(".dataset-workspace, .training-project-workspace, .evaluation-project-workspace").forEach(workspace => {
        const main = workspace.querySelector(":scope > .domain-main");
        const inspector = workspace.querySelector(":scope > aside");
        if (!main || !inspector) return;
        const order = [...workspace.children];
        const browse = create("button", "text-button ux-project-jump", "Browse projects");
        browse.type = "button";
        browse.addEventListener("click", () => {
          main.setAttribute("tabindex", "-1");
          main.focus({ preventScroll: true });
          main.scrollIntoView({ block: "start", behavior: "auto" });
        });
        const back = create("button", "text-button ux-project-jump", "Selected project");
        back.type = "button";
        back.addEventListener("click", () => {
          inspector.setAttribute("tabindex", "-1");
          inspector.focus({ preventScroll: true });
          inspector.scrollIntoView({ block: "start", behavior: "auto" });
        });
        inspector.querySelector("h2")?.after(browse);
        main.querySelector(".section-heading")?.append(back);
        mobileWorkspaces.push({ workspace, main, inspector, order });
      });
      const mobileQuery = window.matchMedia("(max-width: 860px)");
      const arrangeWorkspaces = () => {
        for (const { workspace, inspector, order } of mobileWorkspaces) {
          order.forEach(node => workspace.append(node));
        }
      };
      mobileQuery.addEventListener("change", arrangeWorkspaces);
      arrangeWorkspaces();
      document.querySelectorAll(".dataset-lifecycle").forEach(list => {
        list.setAttribute("tabindex", "0");
      });
      const settings = byId("workstationSettingsForm");
      group("workstationAdvanced", "Advanced workstation settings",
        ["settingsContinuityPolicy", "settingsTseThreshold", "settingsRefreshSeconds"].map(id => byId(id)?.closest("label")),
        settings, settings?.querySelector(".settings-toggle"));
      const evaluation = byId("projectEvaluationFields");
      group("evaluationAdvanced", "Benchmark and compute settings",
        [...(evaluation?.querySelector(".project-advanced-grid")?.children || [])],
        evaluation);
      document.querySelectorAll("details:not([data-ux-disclosure])").forEach((details, index) => {
        rememberDetails(details, details.id || details.className || `details-${index}`);
      });
      const markSettingsDirty = () => {
        settingsDirty = true;
        settingsRevision += 1;
        setText("settingsSaveState", "Unsaved changes");
      };
      settings?.addEventListener("input", markSettingsDirty);
      settings?.addEventListener("change", markSettingsDirty);
      byId("toastDismiss").addEventListener("click", hideToast);
      document.addEventListener("keydown", event => {
        if (event.key === "Escape" && !document.querySelector("dialog[open]")) {
          hideToast();
        }
      });
      document.addEventListener("invalid", event => {
        if (event.target instanceof HTMLElement) reveal(event.target);
      }, true);
    }
    function rememberView(view) {
      const viewport = document.querySelector(".viewport");
      if (viewport) scrollPositions.set(view, viewport.scrollTop);
    }
    function restoreView(view) {
      const viewport = document.querySelector(".viewport");
      if (viewport) requestAnimationFrame(() => { viewport.scrollTop = scrollPositions.get(view) || 0; });
    }
    function captureContext() {
      const active = document.activeElement;
      const row = active?.closest?.("tbody tr");
      const body = row?.parentElement;
      const controls = row ? [...row.querySelectorAll("button,input,select,textarea,a")] : [];
      const location = row && body?.id ? {
        bodyId: body.id,
        rowIndex: [...body.children].indexOf(row),
        name: row.cells[0]?.textContent,
        controlIndex: controls.indexOf(active)
      } : null;
      const scrollers = [...document.querySelectorAll(".viewport, pre, .table-wrap, .run-log, .job-log")]
        .filter(node => node.scrollTop || node.scrollLeft)
        .map(node => ({ node, top: node.scrollTop, left: node.scrollLeft }));
      return () => {
        // Restore only replaced focus, never steal focus after a deliberate user action.
        if (active && !active.isConnected && document.activeElement === document.body) {
          let target = active.id ? byId(active.id) : null;
          if (!target && location) {
            const nextRow = byId(location.bodyId)?.children[location.rowIndex];
            if (nextRow?.cells[0]?.textContent === location.name) {
              target = nextRow.querySelectorAll("button,input,select,textarea,a")[location.controlIndex];
            }
          }
          if (target instanceof HTMLElement && !target.disabled) target.focus({ preventScroll: true });
        }
        scrollers.forEach(({ node, top, left }) => {
          if (node.isConnected && node.scrollTop === 0 && node.scrollLeft === 0) {
            node.scrollTop = top; node.scrollLeft = left;
          }
        });
      };
    }
    return {
      initialize, reveal, rememberView, restoreView, captureContext,
      settingsDirty: () => settingsDirty,
      beginSettingsSave: () => {
        if (settingsSaving) return false;
        settingsSaving = true;
        savingRevision = settingsRevision;
        return true;
      },
      settingsSaved: () => { if (savingRevision === settingsRevision) settingsDirty = false; },
      endSettingsSave: () => { settingsSaving = false; }
    };
  })();

  function replaceIcons(root = document) {
    if (window.lucide?.createIcons) {
      window.lucide.createIcons({ root, attrs: { "aria-hidden": "true" } });
    }
  }

  function labelTableCells(table) {
    if (!table) return;
    const labels = [...table.querySelectorAll("thead th")].map(cell => cell.textContent.trim());
    table.querySelectorAll("tbody tr").forEach(row => {
      [...row.children].forEach((cell, index) => {
        if (cell instanceof HTMLTableCellElement && cell.colSpan === 1) {
          cell.dataset.label = labels[index] || "";
        }
      });
    });
  }

  function labelAllTables() {
    document.querySelectorAll("table").forEach(labelTableCells);
  }

  function updateDockCursor() {
    const nav = byId("moduleNav");
    const cursor = byId("navCursor");
    if (!nav || !cursor || window.innerWidth <= 860) return;
    const active = nav.querySelector(".nav-item.active");
    if (!active) {
      cursor.classList.remove("visible");
      return;
    }
    cursor.style.transform = `translateY(${active.offsetTop}px)`;
    cursor.classList.add("visible");
  }

  function drawVoicePulse(timestamp = performance.now()) {
    const canvas = byId("voicePulse");
    if (!(canvas instanceof HTMLCanvasElement)) return;
    const bounds = canvas.getBoundingClientRect();
    if (!bounds.width || !bounds.height) return;
    const ratio = Math.min(window.devicePixelRatio || 1, 2);
    const width = Math.max(1, Math.round(bounds.width * ratio));
    const height = Math.max(1, Math.round(bounds.height * ratio));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    const context = canvas.getContext("2d");
    if (!context) return;
    context.clearRect(0, 0, width, height);
    context.lineCap = "round";

    const center = height * 0.52;
    context.strokeStyle = "#342c36";
    context.lineWidth = ratio;
    context.beginPath();
    context.moveTo(0, center);
    context.lineTo(width, center);
    context.stroke();

    const elapsed = (timestamp - pulseStartedAt) / 1000;
    const amplitude = pulseActive ? height * 0.31 : height * 0.025;
    const speed = pulseActive ? elapsed * 5.2 : 0;
    const sampleCount = Math.max(56, Math.round(width / (5 * ratio)));

    context.strokeStyle = pulseActive ? "#c35f82" : "#6e4a5a";
    context.lineWidth = 1.5 * ratio;
    context.beginPath();
    for (let index = 0; index <= sampleCount; index += 1) {
      const phase = index / sampleCount;
      const envelope = Math.sin(Math.PI * phase) ** 1.35;
      const carrier = Math.sin(phase * 31 + speed) * .58 + Math.sin(phase * 67 - speed * .7) * .26;
      const y = center + carrier * envelope * amplitude;
      const x = phase * width;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();

    context.strokeStyle = pulseActive ? "#a98bd0" : "#514451";
    context.globalAlpha = .62;
    context.lineWidth = ratio;
    context.beginPath();
    for (let index = 0; index <= sampleCount; index += 1) {
      const phase = index / sampleCount;
      const envelope = Math.sin(Math.PI * phase) ** 1.2;
      const y = center + Math.sin(phase * 43 - speed * 1.3) * envelope * amplitude * .46;
      const x = phase * width;
      if (index === 0) context.moveTo(x, y);
      else context.lineTo(x, y);
    }
    context.stroke();
    context.globalAlpha = 1;

    if (pulseActive && !window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      pulseFrame = requestAnimationFrame(drawVoicePulse);
    }
  }

  function setVoicePulseActive(active) {
    const next = Boolean(active);
    if (next !== pulseActive) {
      pulseStartedAt = performance.now();
      cancelAnimationFrame(pulseFrame);
      pulseActive = next;
    }
    const pulseState = byId("pulseState");
    if (pulseState) pulseState.textContent = next ? "ACTIVE" : "IDLE";
    cancelAnimationFrame(pulseFrame);
    drawVoicePulse();
  }

  async function api(path, options = {}) {
    const response = await fetch(path, {
      cache: "no-store",
      ...options,
      headers: options.body
        ? { "Content-Type": "application/json", ...(options.headers || {}) }
        : options.headers
    });
    const payload = await response.json().catch(() => ({}));
    if (!response.ok) {
      throw new Error(payload.error || payload.detail || `AnifLive-TTS Studio request failed (${response.status})`);
    }
    return payload;
  }

  let toastReturnFocus = null;
  let toastMessageSource = "";
  let dismissedToastMessage = "";
  function hideToast() {
    const toast = byId("toast");
    if (toast.contains(document.activeElement) && toastReturnFocus?.isConnected) {
      toastReturnFocus.focus({ preventScroll: true });
    }
    if (toast.dataset.persistent === "true") dismissedToastMessage = toastMessageSource;
    toast.classList.remove("show");
    toast.inert = true;
    delete toast.dataset.persistent;
  }

  function showToast(message, { persistent = false, repeat = true } = {}) {
    const toast = byId("toast");
    if (toast.dataset.persistent === "true" && !persistent) return;
    if (persistent && !repeat && dismissedToastMessage === String(message)) return;
    if (persistent && toast.classList.contains("show") && toastMessageSource === String(message)) return;
    toastMessageSource = String(message);
    if (!toast.classList.contains("show")) toastReturnFocus = document.activeElement;
    toast.inert = false;
    i18n.setLocalizedText(byId("toastMessage"), message);
    toast.dataset.persistent = String(persistent);
    toast.classList.toggle("toast-error", persistent);
    toast.setAttribute("aria-live", persistent ? "assertive" : "polite");
    toast.classList.add("show");
    clearTimeout(showToast.timer);
    if (!persistent) showToast.timer = setTimeout(hideToast, 5000);
  }

  function validationMessage(field) {
    if (field.validity.valueMissing) return "Complete this field before continuing";
    if (field.validity.typeMismatch) return "Enter a value in the expected format";
    if (field.validity.patternMismatch) return "Use the format shown for this field";
    if (field.validity.rangeUnderflow || field.validity.rangeOverflow) return "Choose a value within the permitted range";
    if (field.validity.stepMismatch) return "Choose a valid step value";
    if (field.validity.tooLong || field.validity.tooShort) return "Adjust the length of this value";
    return "Check this field before continuing";
  }

  function validationSurface(field) {
    return field.closest(".studio-path-picker")
      || field.closest(".styled-select")
      || field;
  }

  function clearFieldValidation(field) {
    if (!(field instanceof HTMLInputElement || field instanceof HTMLSelectElement || field instanceof HTMLTextAreaElement)) return;
    field.classList.remove("field-invalid");
    field.removeAttribute("aria-invalid");
    const surface = validationSurface(field);
    surface?.classList.remove("field-invalid");
    const messageId = field.dataset.validationMessageId;
    if (messageId) document.getElementById(messageId)?.remove();
    delete field.dataset.validationMessageId;
  }

  function showFieldValidation(field) {
    clearFieldValidation(field);
    const surface = validationSurface(field);
    const message = create("p", "field-validation-message", validationMessage(field));
    const messageId = `field-validation-${field.id || Math.random().toString(36).slice(2)}`;
    message.id = messageId;
    message.setAttribute("role", "alert");
    field.dataset.validationMessageId = messageId;
    field.setAttribute("aria-invalid", "true");
    field.setAttribute("aria-describedby", messageId);
    field.classList.add("field-invalid");
    surface?.classList.add("field-invalid");
    surface?.insertAdjacentElement("afterend", message);
  }

  function focusValidationField(field) {
    experience.reveal(field);
    const focusTarget = field.closest(".styled-select")?.querySelector(".styled-select-trigger") || field;
    focusTarget.focus({ preventScroll: true });
    focusTarget.scrollIntoView({ behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth", block: "center" });
  }

  function validateStudioForm(form) {
    const fields = [...form.elements].filter(field => field.willValidate && !field.validity.valid);
    form.querySelectorAll("input, select, textarea").forEach(field => {
      if (!fields.includes(field) && field.validity.valid) clearFieldValidation(field);
    });
    fields.forEach(showFieldValidation);
    if (fields.length) focusValidationField(fields[0]);
    return fields.length === 0;
  }

  function installStyledValidation() {
    document.querySelectorAll("form").forEach(form => { form.noValidate = true; });
    document.addEventListener("invalid", event => {
      event.preventDefault();
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement || event.target instanceof HTMLTextAreaElement) {
        showFieldValidation(event.target);
      }
    }, true);
    document.addEventListener("submit", event => {
      if (!(event.target instanceof HTMLFormElement) || validateStudioForm(event.target)) return;
      event.preventDefault();
      event.stopImmediatePropagation();
    }, true);
    document.addEventListener("input", event => {
      const field = event.target;
      if ((field instanceof HTMLInputElement || field instanceof HTMLSelectElement || field instanceof HTMLTextAreaElement) && field.validity.valid) {
        clearFieldValidation(field);
      }
    });
    document.addEventListener("change", event => {
      const field = event.target;
      if ((field instanceof HTMLInputElement || field instanceof HTMLSelectElement || field instanceof HTMLTextAreaElement) && field.validity.valid) {
        clearFieldValidation(field);
      }
    });
  }

  const dialogCloseTimers = new WeakMap();

  function openAnimatedDialog(dialog) {
    if (!(dialog instanceof HTMLDialogElement) || dialog.open) return;
    const pending = dialogCloseTimers.get(dialog);
    if (pending) window.clearTimeout(pending);
    dialog.classList.remove("is-closing");
    dialog.classList.add("is-entering");
    dialog.showModal();
    requestAnimationFrame(() => requestAnimationFrame(() => dialog.classList.remove("is-entering")));
  }

  function closeAnimatedDialog(dialog) {
    if (!(dialog instanceof HTMLDialogElement) || !dialog.open || dialog.classList.contains("is-closing")) return;
    if (window.matchMedia("(prefers-reduced-motion: reduce)").matches) {
      dialog.close();
      return;
    }
    dialog.classList.add("is-closing");
    const timer = window.setTimeout(() => {
      if (dialog.open) dialog.close();
      dialog.classList.remove("is-closing", "is-entering");
      dialogCloseTimers.delete(dialog);
    }, 210);
    dialogCloseTimers.set(dialog, timer);
  }

  function animateSurface(node) {
    if (!node) return;
    node.classList.remove("surface-enter");
    void node.offsetWidth;
    node.classList.add("surface-enter");
  }

  function pathPickerTitle(input) {
    const label = input.closest("label");
    const labelText = [...(label?.children || [])]
      .find(node => node instanceof HTMLElement && node.tagName === "SPAN")?.textContent?.trim();
    return labelText || input.getAttribute("aria-label") || "Choose a local path";
  }

  function normalizeDroppedPath(value) {
    const text = String(value || "").trim().replace(/^['"]|['"]$/g, "");
    if (!text) return "";
    if (/^file:\/\//i.test(text)) {
      try {
        const url = new URL(text);
        const pathname = decodeURIComponent(url.pathname || "");
        return /^\/[A-Za-z]:\//.test(pathname) ? pathname.slice(1).replaceAll("/", "\\") : pathname;
      } catch (_) { return ""; }
    }
    return /^(?:[A-Za-z]:[\\/]|\\\\|\/)/.test(text) ? text : "";
  }

  function droppedLocalPaths(dataTransfer) {
    const candidates = [];
    for (const file of [...(dataTransfer?.files || [])]) {
      candidates.push(file.path, file.webkitRelativePath);
    }
    for (const type of ["text/uri-list", "text/plain"]) {
      const payload = dataTransfer?.getData(type) || "";
      candidates.push(...payload.split(/\r?\n/).filter(line => line && !line.startsWith("#")));
    }
    return [...new Set(candidates.map(normalizeDroppedPath).filter(Boolean))];
  }

  function applyPickedPaths(input, paths) {
    const selected = [...new Set((paths || []).map(value => String(value).trim()).filter(Boolean))];
    if (!selected.length) return;
    if (input instanceof HTMLTextAreaElement || input.dataset.pathMultiple === "true") {
      const existing = input.value.split(/\r?\n/).map(value => value.trim()).filter(Boolean);
      input.value = [...new Set([...existing, ...selected])].join("\n");
    } else {
      input.value = selected[0];
    }
    input.dispatchEvent(new Event("input", { bubbles: true }));
    input.dispatchEvent(new Event("change", { bubbles: true }));
  }

  function syncPathPickerEmptyState(input) {
    input.closest(".studio-path-picker")?.classList.toggle("is-empty", !input.value.trim());
  }

  async function requestNativePath(input, requestedKind) {
    const shell = input.closest(".studio-path-picker");
    const button = shell?.querySelector(".path-picker-browse");
    const multiple = input.dataset.pathMultiple === "true";
    const kind = requestedKind === "file" && multiple ? "files" : requestedKind;
    if (button) button.disabled = true;
    shell?.classList.add("is-browsing");
    try {
      const result = await api("/api/workstation/path-picker", {
        method: "POST",
        body: JSON.stringify({
          kind,
          title: pathPickerTitle(input),
          accept: input.dataset.pathAccept || "all"
        })
      });
      if (!result.cancelled) {
        applyPickedPaths(input, result.paths);
        shell?.classList.add("drop-accepted");
        window.setTimeout(() => shell?.classList.remove("drop-accepted"), 420);
      }
    } catch (error) {
      showToast(error.message, { persistent: true });
    } finally {
      if (button) button.disabled = false;
      shell?.classList.remove("is-browsing");
    }
  }

  function enhancePathPicker(input) {
    if (!(input instanceof HTMLInputElement || input instanceof HTMLTextAreaElement) || input.dataset.pathReady === "true") return;
    input.dataset.pathReady = "true";
    const pickerKind = input.dataset.pathPicker || "file";
    const shell = document.createElement("div");
    shell.className = "studio-path-picker";
    const control = document.createElement("div");
    control.className = "path-picker-control";
    const icon = document.createElement("i");
    icon.dataset.lucide = pickerKind === "directory" ? "folder" : "file-input";
    icon.className = "path-picker-leading-icon";
    icon.setAttribute("aria-hidden", "true");
    const browse = document.createElement("button");
    browse.className = "path-picker-browse";
    browse.type = "button";
    const browseIcon = document.createElement("i");
    browseIcon.dataset.lucide = "folder-search";
    browseIcon.setAttribute("aria-hidden", "true");
    const browseLabel = create("span", "", "Browse");
    browse.append(browseIcon, browseLabel);
    i18n.setLocalizedAttribute(browse, "aria-label", "Browse local path");
    i18n.setLocalizedAttribute(browse, "title", "Browse local path");

    input.parentNode.insertBefore(shell, input);
    control.append(icon, input);
    let placeholderCopy = null;
    if (input instanceof HTMLTextAreaElement) {
      placeholderCopy = create("span", "path-picker-placeholder", input.placeholder);
      placeholderCopy.setAttribute("aria-hidden", "true");
      control.append(placeholderCopy);
      new MutationObserver(() => {
        placeholderCopy.textContent = input.placeholder;
      }).observe(input, { attributes: true, attributeFilter: ["placeholder"] });
    }
    control.append(browse);
    shell.append(control);
    const hint = create("span", "path-picker-hint", "Drop here or enter a path");
    shell.append(hint);
    const syncEmptyState = () => syncPathPickerEmptyState(input);
    input.addEventListener("input", syncEmptyState);
    input.addEventListener("change", syncEmptyState);
    syncEmptyState();

    let menu = null;
    if (pickerKind === "mixed") {
      menu = document.createElement("div");
      menu.className = "path-picker-menu";
      menu.hidden = true;
      const closeMenu = () => {
        menu.hidden = true;
        browse.setAttribute("aria-expanded", "false");
      };
      const positionMenu = () => {
        if (menu.hidden) return;
        const rect = browse.getBoundingClientRect();
        const gap = 8;
        const width = Math.min(210, window.innerWidth - gap * 2);
        const estimatedHeight = Math.max(96, menu.offsetHeight || 96);
        const openAbove = window.innerHeight - rect.bottom < estimatedHeight + gap && rect.top > estimatedHeight;
        const dialog = menu.parentElement instanceof HTMLDialogElement ? menu.parentElement : null;
        const origin = dialog?.getBoundingClientRect() || { left: 0, top: 0 };
        const viewportLeft = Math.min(Math.max(gap, rect.right - width), window.innerWidth - width - gap);
        const viewportTop = openAbove
          ? Math.max(gap, rect.top - estimatedHeight - 6)
          : Math.min(window.innerHeight - estimatedHeight - gap, rect.bottom + 6);
        menu.style.position = dialog ? "absolute" : "fixed";
        menu.style.width = `${width}px`;
        menu.style.left = `${viewportLeft - origin.left + (dialog?.scrollLeft || 0)}px`;
        menu.style.top = `${viewportTop - origin.top + (dialog?.scrollTop || 0)}px`;
        menu.style.transformOrigin = openAbove ? "bottom right" : "top right";
      };
      for (const [kind, iconName, label] of [["file", "files", "Choose file"], ["directory", "folder-open", "Choose folder"]]) {
        const option = document.createElement("button");
        option.type = "button";
        option.dataset.pathChoice = kind;
        const optionIcon = document.createElement("i");
        optionIcon.dataset.lucide = iconName;
        optionIcon.setAttribute("aria-hidden", "true");
        const optionLabel = create("span", "", label);
        option.append(optionIcon, optionLabel);
        option.addEventListener("pointerdown", event => {
          event.preventDefault();
          event.stopPropagation();
        });
        option.addEventListener("click", event => {
          event.preventDefault();
          event.stopPropagation();
          closeMenu();
          requestNativePath(input, kind);
        });
        menu.append(option);
      }
      (input.closest("dialog") || document.body).append(menu);
      replaceIcons(menu);
      browse.setAttribute("aria-haspopup", "menu");
      browse.setAttribute("aria-expanded", "false");
      browse.addEventListener("click", () => {
        if (!menu.hidden) {
          closeMenu();
          return;
        }
        menu.hidden = false;
        browse.setAttribute("aria-expanded", "true");
        positionMenu();
      });
      window.addEventListener("resize", positionMenu);
      window.addEventListener("scroll", positionMenu, true);
      document.addEventListener("pointerdown", event => {
        if (!menu.hidden && !shell.contains(event.target) && !menu.contains(event.target)) closeMenu();
      }, true);
    } else {
      browse.addEventListener("click", () => requestNativePath(input, pickerKind));
    }

    let dragDepth = 0;
    shell.addEventListener("dragenter", event => {
      event.preventDefault();
      dragDepth += 1;
      shell.classList.add("is-dragging");
    });
    shell.addEventListener("dragover", event => {
      event.preventDefault();
      if (event.dataTransfer) event.dataTransfer.dropEffect = "copy";
    });
    shell.addEventListener("dragleave", () => {
      dragDepth = Math.max(0, dragDepth - 1);
      if (!dragDepth) shell.classList.remove("is-dragging");
    });
    shell.addEventListener("drop", event => {
      event.preventDefault();
      dragDepth = 0;
      shell.classList.remove("is-dragging");
      const paths = droppedLocalPaths(event.dataTransfer);
      if (!paths.length) {
        shell.classList.add("drop-rejected");
        window.setTimeout(() => shell.classList.remove("drop-rejected"), 420);
        showToast("The browser withheld the local path; use Browse to select it securely");
        return;
      }
      applyPickedPaths(input, paths);
      shell.classList.add("drop-accepted");
      window.setTimeout(() => shell.classList.remove("drop-accepted"), 420);
    });
    replaceIcons(shell);
  }

  function enhancePathPickers() {
    document.querySelectorAll("[data-path-picker]").forEach(enhancePathPicker);
  }

  function syncAmbientVideo() {
    const video = byId("overviewFilm");
    if (!(video instanceof HTMLVideoElement)) return;
    const shouldPlay = state.currentView === "overview"
      && !document.hidden
      && state.settings.overview_motion !== false
      && !window.matchMedia("(prefers-reduced-motion: reduce)").matches;
    if (shouldPlay) video.play().catch(() => {});
    else video.pause();
  }

  function openView(view, { updateHistory = true } = {}) {
    if (!moduleTitles[view]) return;
    const changed = state.currentView !== view;
    if (changed) experience.rememberView(state.currentView);
    state.currentView = view;
    document.body.dataset.currentView = view;
    document.querySelectorAll("[data-view-panel]").forEach(panel => {
      panel.classList.toggle("active", panel.dataset.viewPanel === view);
    });
    document.querySelectorAll("[data-view]").forEach(item => {
      const current = item.dataset.view === view;
      item.classList.toggle("active", current);
      if (current) item.setAttribute("aria-current", "page");
      else item.removeAttribute("aria-current");
    });
    const primaryViews = new Set(["overview", "synthesis", "datasets", "jobs"]);
    byId("mobileMoreButton").classList.toggle("active", !primaryViews.has(view));
    const localizedTitle = i18n.t(moduleTitles[view]);
    setText("moduleTitle", moduleTitles[view]);
    document.title = `${localizedTitle} · AnifLive-TTS Studio`;
    if (updateHistory && changed) history.pushState({ view }, "", `#${view}`);
    const mobileSheet = byId("mobileNavSheet");
    if (mobileSheet?.open) closeAnimatedDialog(mobileSheet);
    if (changed) {
      const viewport = document.querySelector(".viewport");
      if (viewport) experience.restoreView(view);
    }
    syncAmbientVideo();
    progressiveTables.activate();
    if (view === "overview") requestAnimationFrame(drawVoicePulse);
    requestAnimationFrame(updateDockCursor);
    if (view === "expressions") loadExpressions();
    if (view === "datasets" && state.selectedDatasetId) loadSelectedDataset();
    if (view === "tse" && state.selectedTseJobId) inspectTseJob(state.selectedTseJobId);
    if (view === "training") Promise.all([loadArtifacts(), loadTrainingRunDetails()]).catch(error => showToast(error.message, { persistent: true }));
    if (view === "models") Promise.all([loadRuntime(), loadArtifacts()]).catch(error => showToast(error.message, { persistent: true }));
    if (view === "evaluation") loadArtifacts().then(() => loadEvaluationRunDetails()).catch(error => showToast(error.message, { persistent: true }));
    if (view === "engines") Promise.all([loadRuntime(), loadArtifacts()]).catch(error => showToast(error.message, { persistent: true }));
    if (view === "jobs") loadJobs();
  }

  function formatDate(value) {
    if (!value) return "—";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return "—";
    return new Intl.DateTimeFormat(state.locale, {
      month: "short",
      day: "2-digit",
      hour: "2-digit",
      minute: "2-digit"
    }).format(date);
  }

  function shortPath(value) {
    if (!value || typeof value !== "string") return "Not configured";
    const parts = value.split(/[\\/]/).filter(Boolean);
    return parts.length > 2 ? `…/${parts.slice(-2).join("/")}` : value;
  }

  function statusText(value) {
    const labels = {
      queued: "Queued", running: "Running", succeeded: "Succeeded", failed: "Failed",
      paused: "Paused", cancelled: "Cancelled", pending: "Pending", ready: "Ready",
      draft: "Draft", blocked: "Blocked", unknown: "Unknown", unavailable: "Unavailable"
    };
    return labels[String(value || "Unknown").toLowerCase()] || value || "Unknown";
  }

  function statusLabel(value, text = value) {
    return create("span", `status-label ${value || "unknown"}`, statusText(text || value));
  }

  function progressText(value) {
    if (value === null || value === undefined || value === "") return "Unavailable";
    const number = Number(value);
    return Number.isFinite(number) ? `${Math.round(number * 100)}%` : "Unavailable";
  }

  function displayRunProgress(id, value) {
    const bar = byId(id);
    const known = value !== null && value !== undefined && value !== ""
      && Number.isFinite(Number(value));
    bar.classList.toggle("is-unavailable", !known);
    if (known) bar.value = Math.max(0, Math.min(1, Number(value)));
    else bar.removeAttribute("value");
  }

  function audioClock(value) {
    const seconds = Number.isFinite(value) && value > 0 ? Math.floor(value) : 0;
    return `${String(Math.floor(seconds / 60)).padStart(2, "0")}:${String(seconds % 60).padStart(2, "0")}`;
  }

  function setTseAudioIcon(button, iconName, label) {
    const icon = document.createElement("i");
    icon.dataset.lucide = iconName;
    icon.setAttribute("aria-hidden", "true");
    button.replaceChildren(icon);
    i18n.setLocalizedAttribute(button, "aria-label", label);
    i18n.setLocalizedAttribute(button, "title", label.replace(" target audio", ""));
    replaceIcons(button);
  }

  function renderTseAudioPlayer() {
    const audio = byId("tseTargetAudio");
    const toggle = byId("tseAudioToggle");
    const mute = byId("tseAudioMute");
    const seek = byId("tseAudioSeek");
    if (!(audio instanceof HTMLAudioElement) || !toggle || !mute || !seek) return;
    const duration = Number.isFinite(audio.duration) ? audio.duration : 0;
    const current = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
    seek.value = duration > 0 ? String(Math.min(1, current / duration)) : "0";
    seek.disabled = duration <= 0;
    toggle.disabled = !audio.currentSrc && !audio.getAttribute("src");
    mute.disabled = toggle.disabled;
    setText("tseAudioTime", `${audioClock(current)} / ${audioClock(duration)}`);
    setTseAudioIcon(toggle, audio.paused ? "play" : "pause", audio.paused ? "Play target audio" : "Pause target audio");
    setTseAudioIcon(mute, audio.muted ? "volume-x" : "volume-2", audio.muted ? "Unmute target audio" : "Mute target audio");
  }

  async function toggleTseAudio() {
    const audio = byId("tseTargetAudio");
    if (!(audio instanceof HTMLAudioElement) || (!audio.currentSrc && !audio.getAttribute("src"))) return;
    if (audio.paused) {
      try { await audio.play(); } catch (error) { showToast(error.message, { persistent: true }); }
    } else {
      audio.pause();
    }
    renderTseAudioPlayer();
  }

  function renderDatasetAudioPlayer() {
    const audio = byId("datasetAudioPreview");
    const toggle = byId("datasetAudioToggle");
    const mute = byId("datasetAudioMute");
    const seek = byId("datasetAudioSeek");
    if (!(audio instanceof HTMLAudioElement) || !toggle || !mute || !seek) return;
    const duration = Number.isFinite(audio.duration) ? audio.duration : 0;
    const current = Number.isFinite(audio.currentTime) ? audio.currentTime : 0;
    const unavailable = !audio.currentSrc && !audio.getAttribute("src");
    seek.value = duration > 0 ? String(Math.min(1, current / duration)) : "0";
    seek.disabled = duration <= 0;
    toggle.disabled = unavailable;
    mute.disabled = unavailable;
    setText("datasetAudioTime", `${audioClock(current)} / ${audioClock(duration)}`);
    setTseAudioIcon(toggle, audio.paused ? "play" : "pause", audio.paused ? "Play dataset clip" : "Pause dataset clip");
    setTseAudioIcon(mute, audio.muted ? "volume-x" : "volume-2", audio.muted ? "Unmute dataset clip" : "Mute dataset clip");
  }

  async function toggleDatasetAudio() {
    const audio = byId("datasetAudioPreview");
    if (!(audio instanceof HTMLAudioElement) || (!audio.currentSrc && !audio.getAttribute("src"))) return;
    if (audio.paused) {
      try { await audio.play(); } catch (error) { showToast(error.message, { persistent: true }); }
    } else {
      audio.pause();
    }
    renderDatasetAudioPlayer();
  }

  function setReadState(ids, phase, error = null) {
    ids.forEach(id => {
      const body = byId(id);
      const surface = body?.closest(".table-wrap");
      const section = surface?.closest("section");
      if (!body || !surface || !section) return;
      surface.setAttribute("aria-busy", String(phase === "loading"));
      let notice = section.querySelector(".ux-data-error");
      if (phase === "error") {
        if (!notice) {
          notice = create("p", "ux-data-error");
          notice.setAttribute("role", "alert");
          surface.before(notice);
        }
        const hasRows = body.children.length > 0 && !body.querySelector(".empty-cell");
        const title = hasRows || body.dataset.readLoaded === "true" ? "Refresh failed; previously loaded rows are shown." : "Data could not be loaded.";
        if (!hasRows && body.dataset.readLoaded !== "true") {
          const countId = {datasetRows:"datasetCount",trainingRows:"trainingCount",evaluationRows:"evaluationCount",tseRows:"tseCount",modelRows:"modelCount",engineArtifactRows:"engineArtifactCount",expressionRows:"packageExpressionCount",expressionDraftRows:"localExpressionCount"}[id];
          if (countId) setText(countId, "Unavailable");
        }
        i18n.setLocalizedText(notice, title);
        if (!hasRows) emptyTable(body, body.closest("table").querySelectorAll("thead th").length, "Data could not be loaded.");
        notice.title = error?.message || "";
      } else if (phase === "ready") {
        body.dataset.readLoaded = "true";
        notice?.remove();
      } else if (!body.children.length) {
        emptyTable(body, body.closest("table").querySelectorAll("thead th").length, "Loading…");
      }
    });
  }

  function emptyTable(tbody, columns, message) {
    progressiveTables.clear(tbody);
    const row = create("tr");
    const cell = create("td", "empty-cell", message);
    cell.colSpan = columns;
    row.append(cell);
    tbody.replaceChildren(row);
  }

  function projectConfig(record, key, fallback = "—") {
    const config = record && typeof record.config === "object" ? record.config : {};
    const value = config[key];
    return typeof value === "string" && value ? value : fallback;
  }

  function renderOverview() {
    const overview = state.overview;
    if (!overview) return;
    const totalProjects = Object.values(overview.project_counts || {}).reduce((sum, value) => sum + Number(value || 0), 0);
    const gpu = overview.gpu && typeof overview.gpu === "object" ? overview.gpu : {};
    const gpuName = gpu.name || "Unavailable";
    const running = overview.running_jobs || [];

    setText("runtimeGpu", gpuName);
    setText("runtimeJobs", `${running.length} running`);
    setText("overviewGpu", gpuName);
    setText("overviewBackend", overview.backend || "Not connected");
    setText("overviewJob", running[0]?.type || "None");
    setText("overviewQueue", `${overview.queued_jobs || 0} queued`);
    setText("overviewProjects", String(totalProjects));
    setVoicePulseActive(running.length > 0 || state.playbackActive);

    const projectList = byId("recentProjects");
    const recent = overview.recent_projects || [];
    if (!recent.length) {
      const message = create("p", "", "No projects yet");
      projectList.className = "data-list empty-list";
      projectList.replaceChildren(message);
    } else {
      projectList.className = "data-list";
      projectList.replaceChildren(...recent.map(record => {
        const row = create("div", "data-row");
        row.append(
          createData("strong", "", record.name),
          create("span", "", String(record.kind || "").toUpperCase()),
          create("span", "", formatDate(record.updated_at))
        );
        return row;
      }));
    }

    const activity = byId("activityList");
    const records = state.jobs.slice(0, 7);
    if (!records.length) {
      activity.replaceChildren(create("p", "", "No workstation activity"));
    } else {
      activity.replaceChildren(...records.map(job => {
        const row = create("div", "activity-row");
        row.append(create("i", "activity-mark"), create("strong", "", job.type), create("span", "", statusText(job.status)));
        return row;
      }));
    }
    labelAllTables();
  }

  function renderProjectTables() {
    const groups = {
      dataset: state.projects.filter(item => item.kind === "dataset"),
      tse: state.projects.filter(item => item.kind === "tse"),
      training: state.projects.filter(item => item.kind === "training"),
      evaluation: state.projects.filter(item => item.kind === "evaluation")
    };

    const datasetRows = byId("datasetRows");
    byId("datasetCount").textContent = `${groups.dataset.length} projects`;
    if (!groups.dataset.length) {
      state.selectedDatasetId = null;
      state.datasetItems = [];
      emptyTable(datasetRows, 5, "No dataset projects");
    }
    else progressiveTables.render(datasetRows, groups.dataset, record => {
      const row = create("tr");
      row.classList.toggle("selected", record.id === state.selectedDatasetId);
      const actionCell = create("td");
      const open = create("button", "row-action", record.id === state.selectedDatasetId ? "Open" : "Select");
      open.type = "button";
      open.addEventListener("click", () => selectDataset(record));
      actionCell.append(open);
      const name = createData("td", "", record.name);
      const status = create("td"); status.append(statusLabel(record.status));
      const sources = datasetProjectSources(record);
      row.append(name, status, create("td", "", sources.length > 1 ? `${sources.length} sources` : shortPath(sources[0] || "")), create("td", "", formatDate(record.updated_at)), actionCell);
      return row;
    }, state.selectedDatasetId);
    const selectedDataset = groups.dataset.find(record => record.id === state.selectedDatasetId) || null;
    if (!selectedDataset && groups.dataset.length && !state.datasetLoading) {
      queueMicrotask(() => selectDataset(groups.dataset[0]));
    } else {
      renderDatasetWorkspace(selectedDataset);
    }

    const tseRows = byId("tseRows");
    byId("tseCount").textContent = `${groups.tse.length} projects`;
    if (!groups.tse.length) {
      state.selectedTseId = null;
      emptyTable(tseRows, 6, "No TSE projects");
      renderTseProjectState(null);
    }
    else progressiveTables.render(tseRows, groups.tse, record => {
      const row = create("tr");
      row.classList.toggle("selected", record.id === state.selectedTseId);
      const latestJob = latestTseJob(record.id);
      const status = create("td"); status.append(statusLabel(latestJob?.status || record.status));
      const actionCell = create("td");
      const open = create("button", "row-action", record.id === state.selectedTseId ? "Open" : "Select");
      open.type = "button";
      open.addEventListener("click", () => selectTseProject(record));
      actionCell.append(open);
      row.append(createData("td", "", record.name), status, create("td", "", shortPath(projectConfig(record, "reference", ""))), create("td", "", shortPath(projectConfig(record, "source", ""))), create("td", "", formatDate(record.updated_at)), actionCell);
      return row;
    }, state.selectedTseId, record => latestTseJob(record.id));
    const selectedTse = groups.tse.find(record => record.id === state.selectedTseId) || null;
    if (!selectedTse && groups.tse.length) queueMicrotask(() => selectTseProject(groups.tse[0]));
    else renderTseProjectState(selectedTse);

    const trainingRows = byId("trainingRows");
    byId("trainingCount").textContent = `${groups.training.length} projects`;
    if (!groups.training.length) {
      state.selectedTrainingId = null;
      state.selectedTrainingJobId = null;
      emptyTable(trainingRows, 6, "No training experiments");
      renderTrainingWorkspace(null);
    }
    else progressiveTables.render(trainingRows, groups.training, record => {
      const row = create("tr");
      row.classList.toggle("selected", record.id === state.selectedTrainingId);
      const latestJob = latestProjectJob(record.id, "training.prepare");
      const status = create("td"); status.append(statusLabel(latestJob?.status || record.status));
      const actionCell = create("td");
      const select = create("button", "row-action", record.id === state.selectedTrainingId ? "Open" : "Select");
      select.type = "button";
      select.addEventListener("click", () => selectTrainingProject(record));
      actionCell.append(select);
      row.append(createData("td", "", record.name), create("td", "", projectConfig(record, "preset", "balanced")), status, create("td", "", progressText(latestJob?.progress)), create("td", "", formatDate(latestJob?.updated_at || record.updated_at)), actionCell);
      return row;
    }, state.selectedTrainingId, record => latestProjectJob(record.id, "training.prepare"));
    const selectedTraining = groups.training.find(record => record.id === state.selectedTrainingId) || null;
    if (!selectedTraining && groups.training.length) {
      state.selectedTrainingId = groups.training[0].id;
      queueMicrotask(() => selectTrainingProject(groups.training[0]));
    } else renderTrainingWorkspace(selectedTraining);

    const evaluationRows = byId("evaluationRows");
    byId("evaluationCount").textContent = `${groups.evaluation.length} projects`;
    if (!groups.evaluation.length) {
      state.selectedEvaluationId = null;
      state.selectedEvaluationJobId = null;
      emptyTable(evaluationRows, 6, "No evaluation runs");
      renderEvaluationWorkspace(null);
    }
    else progressiveTables.render(evaluationRows, groups.evaluation, record => {
      const row = create("tr");
      row.classList.toggle("selected", record.id === state.selectedEvaluationId);
      const latestJob = latestProjectJob(record.id, "evaluation.prepare");
      const status = create("td"); status.append(statusLabel(latestJob?.status || record.status));
      const report = latestJob?.result?.backend?.payload;
      const actionCell = create("td");
      const select = create("button", "row-action", record.id === state.selectedEvaluationId ? "Open" : "Select");
      select.type = "button";
      select.addEventListener("click", () => selectEvaluationProject(record));
      actionCell.append(select);
      const quality = report?.schema === "aniflive-tts-workstation-evaluation-v1"
        ? (report.gates?.stream_complete_quality ? "Passed" : "Failed")
        : "Pending";
      const performance = report?.schema === "aniflive-tts-workstation-evaluation-v1"
        ? (report.gates?.canonical_benchmark_completed ? "Measured" : "Unavailable")
        : "Pending";
      row.append(createData("td", "", record.name), status, create("td", "", quality), create("td", "", performance), create("td", "", formatDate(latestJob?.updated_at || record.updated_at)), actionCell);
      return row;
    }, state.selectedEvaluationId, record => latestProjectJob(record.id, "evaluation.prepare"));
    const selectedEvaluation = groups.evaluation.find(record => record.id === state.selectedEvaluationId) || null;
    if (!selectedEvaluation && groups.evaluation.length) {
      state.selectedEvaluationId = groups.evaluation[0].id;
      queueMicrotask(() => selectEvaluationProject(groups.evaluation[0]));
    } else renderEvaluationWorkspace(selectedEvaluation);
    labelAllTables();
  }

  function latestTseJob(projectId) {
    return state.jobs
      .filter(job => job.type === "tse.prepare" && job.project_id === projectId)
      .sort((left, right) => String(right.updated_at || "").localeCompare(String(left.updated_at || "")))[0] || null;
  }

  function latestProjectJob(projectId, jobType) {
    return state.jobs
      .filter(job => job.type === jobType && job.project_id === projectId)
      .sort((left, right) => String(right.updated_at || "").localeCompare(String(left.updated_at || "")))[0] || null;
  }

  function latestDatasetProcessJob(projectId) {
    return latestProjectJob(projectId, "dataset.process");
  }

  function datasetProcessJobs(projectId) {
    const jobs = state.jobs.filter(job => job.type === "dataset.process" && job.project_id === projectId);
    const retriedJobIds = new Set(jobs.map(job => job.retry_of).filter(Boolean));
    return jobs
      .filter(job => !retriedJobIds.has(job.id))
      .sort((left, right) => String(right.updated_at || "").localeCompare(String(left.updated_at || "")));
  }

  function latestDatasetAcquisitionJob(projectId, type) {
    return latestProjectJob(projectId, type);
  }

  function artifactsForJob(jobId, artifactType = null) {
    if (!jobId) return [];
    return state.artifacts.filter(artifact => (
      artifact.status === "ready"
      && (!artifactType || artifact.type === artifactType)
      && artifact.metadata?.source === "linux-docker"
      && artifact.metadata?.job_id === jobId
    ));
  }

  function replaceDefinitionList(node, rows) {
    const previous = [...node.querySelectorAll("details[data-detail-key]")];
    const expanded = new Set(previous.filter(item => item.open).map(item => item.dataset.detailKey));
    const scroll = new Map(previous.map(item => [item.dataset.detailKey, item.querySelector("code")?.scrollTop || 0]));
    const focused = node.contains(document.activeElement) ? document.activeElement : null;
    const focusedKey = focused?.closest("details")?.dataset.detailKey;
    const focusCode = focused?.tagName === "CODE";
    node.replaceChildren(...rows.map(([term, value, title]) => {
      const row = create("div");
      const detail = create("dd");
      if (title) {
        detail.title = title;
        const disclosure = create("details", "ux-technical-value");
        disclosure.dataset.detailKey = term + ":" + title;
        disclosure.open = expanded.has(disclosure.dataset.detailKey);
        const summary = createData("summary", "", value);
        const full = createData("code", "", title);
        full.tabIndex = 0;
        disclosure.append(summary, full);
        detail.append(disclosure);
      } else {
        i18n.setLocalizedText(detail, String(value));
      }
      row.append(create("dt", "", term), detail);
      return row;
    }));
    node.querySelectorAll("details[data-detail-key]").forEach(item => {
      const full = item.querySelector("code");
      full.scrollTop = scroll.get(item.dataset.detailKey) || 0;
      if (focusedKey === item.dataset.detailKey) {
        (focusCode ? full : item.querySelector("summary")).focus({ preventScroll: true });
      }
    });
  }

  function numericValue(source, keys) {
    for (const key of keys) {
      const raw = source?.[key];
      if (raw === null || raw === undefined || raw === "" || typeof raw === "boolean") continue;
      const value = Number(raw);
      if (Number.isFinite(value)) return value;
    }
    return null;
  }

  function formatBytes(value) {
    if (!Number.isFinite(value) || value < 0) return "Unavailable";
    const gib = value / (1024 ** 3);
    return `${gib.toFixed(gib >= 10 ? 1 : 2)} GiB`;
  }

  function estimatedEta(job) {
    const explicit = numericValue(job, ["eta_seconds"]);
    if (explicit !== null) return `${Math.max(0, Math.round(explicit))} s`;
    const progress = Number(job?.progress);
    const started = Date.parse(job?.started_at || "");
    if (!(progress > 0 && progress < 1) || !Number.isFinite(started)) return "Unavailable";
    const elapsed = Math.max(0, (Date.now() - started) / 1000);
    return `${Math.max(0, Math.round(elapsed * (1 - progress) / progress))} s · progress estimate`;
  }

  function runEventNodes(logs) {
    if (!logs.length) return [create("p", "", "No worker events")];
    return logs.slice(-16).map(log => {
      const row = create("div", `run-event ${log.level || "info"}`);
      row.append(create("time", "", formatDate(log.created_at)), create("span", "", log.message || ""));
      return row;
    });
  }

  function artifactNodes(artifacts, emptyMessage) {
    if (!artifacts.length) return [create("p", "", emptyMessage)];
    return artifacts.map(artifact => {
      const row = create("div", "run-artifact");
      const link = create("a", "", "Open");
      link.href = `/api/workstation/artifacts/${encodeURIComponent(artifact.id)}/content`;
      link.target = "_blank";
      link.rel = "noopener";
      row.append(createData("strong", "", artifact.name || artifact.id), link);
      return row;
    });
  }

  function selectedTrainingProject() {
    return state.projects.find(project => project.kind === "training" && project.id === state.selectedTrainingId) || null;
  }

  function trainingInputsMissing(project) {
    const config = project?.config && typeof project.config === "object" ? project.config : {};
    return ["dataset", "pretrained_gpt", "pretrained_sovits_g", "pretrained_sovits_d"]
      .filter(key => typeof config[key] !== "string" || !config[key].trim());
  }

  function trainingEpoch(logs) {
    for (const log of [...logs].reverse()) {
      const match = String(log.message || "").match(/\b(gpt|sovits) epoch\s+(\d+)\/(\d+)/i);
      if (match) return `${match[1].toUpperCase()} ${match[2]} / ${match[3]}`;
    }
    return "Unavailable";
  }

  function latestRunMessage(logs, fallback) {
    const message = [...logs].reverse().find(log => typeof log.message === "string" && log.message.trim());
    return message?.message || fallback;
  }

  function latestTrainingLifecycleJob(project, type) {
    return project ? latestProjectJob(project.id, type) : null;
  }

  function jobStageState(job) {
    if (!job) return { state: "waiting", label: "Waiting" };
    if (job.status === "succeeded") return { state: "complete", label: "Complete" };
    if (job.status === "failed" || job.status === "cancelled") return { state: "blocked", label: job.status };
    if (["queued", "running", "paused"].includes(job.status)) return { state: "active", label: job.status };
    return { state: "waiting", label: job.status || "Waiting" };
  }

  function renderTrainingLifecycle(project) {
    const config = project?.config && typeof project.config === "object" ? project.config : {};
    const training = latestTrainingLifecycleJob(project, "training.prepare");
    const selection = latestTrainingLifecycleJob(project, "checkpoint.select");
    const reference = latestTrainingLifecycleJob(project, "reference.select");
    const holdout = latestTrainingLifecycleJob(project, "holdout.evaluate");
    const engine = latestTrainingLifecycleJob(project, "engine.prepare");
    const parity = latestTrainingLifecycleJob(project, "conversion.parity");
    const modelPackage = latestTrainingLifecycleJob(project, "model.package");
    const evaluationProject = project
      ? state.projects.find(candidate => candidate.kind === "evaluation" && candidate.config?.source_training_project_id === project.id)
      : null;
    const evaluation = evaluationProject ? latestProjectJob(evaluationProject.id, "evaluation.prepare") : null;
    const dataState = project && config.source_dataset_id
      ? { state: "complete", label: "Reviewed" }
      : { state: project ? "blocked" : "waiting", label: project ? "Dataset missing" : "Waiting" };
    const referenceState = config.reference_status === "human-locked"
      ? { state: "complete", label: "Human locked" }
      : config.reference_status === "all-poor"
        ? { state: "blocked", label: "All candidates rejected" }
      : reference?.status === "succeeded"
        ? { state: "active", label: "Listening required" }
        : jobStageState(reference);
    const conversionState = parity
      ? jobStageState(parity)
      : engine
        ? jobStageState(engine)
        : holdout?.status === "failed"
          ? { state: "blocked", label: "Holdout failed" }
          : { state: "waiting", label: "Waiting" };
    const evaluationState = evaluationProject?.config?.qualification_status === "blocked"
      ? { state: "blocked", label: "Evidence missing" }
      : jobStageState(evaluation);
    const stages = {
      data: dataState,
      train: jobStageState(training),
      select: jobStageState(selection),
      reference: referenceState,
      convert: conversionState,
      package: jobStageState(modelPackage),
      evaluate: evaluationState
    };
    byId("trainingLifecycle")?.querySelectorAll("[data-training-stage]").forEach(node => {
      const stage = stages[node.dataset.trainingStage] || { state: "waiting", label: "Waiting" };
      node.classList.remove("complete", "active", "blocked");
      if (stage.state !== "waiting") node.classList.add(stage.state);
      const label = node.querySelector("small");
      if (label) i18n.setLocalizedText(label, stage.label);
    });
    return { training, selection, reference, holdout, engine, parity, modelPackage, evaluationProject, evaluation };
  }

  function referenceArtifactPath(artifact) {
    return String(artifact?.metadata?.worker_relative_path || "").replaceAll("\\", "/");
  }

  function setReferenceReviewCount(count) {
    const node = byId("referenceReviewState");
    if (node) node.textContent = `${count} ${i18n.t("blind candidates")}`;
  }

  function setReferenceAudioButton(button, playing) {
    const icon = document.createElement("i");
    icon.dataset.lucide = playing ? "pause" : "play";
    icon.setAttribute("aria-hidden", "true");
    button.replaceChildren(icon);
    const label = playing ? "Pause blind reference" : "Play blind reference";
    i18n.setLocalizedAttribute(button, "aria-label", label);
    i18n.setLocalizedAttribute(button, "title", label);
    button.classList.toggle("playing", playing);
    replaceIcons(button);
  }

  function stopReferenceReviewAudio({ rewind = false } = {}) {
    const active = state.referenceReviewAudio;
    if (!active) return;
    active.audio.pause();
    if (rewind) active.audio.currentTime = 0;
    setReferenceAudioButton(active.button, false);
    state.referenceReviewAudio = null;
  }

  async function toggleReferenceAudio(audio, button) {
    if (state.referenceReviewAudio?.audio === audio) {
      stopReferenceReviewAudio();
      return;
    }
    stopReferenceReviewAudio({ rewind: true });
    state.referenceReviewAudio = { audio, button };
    setReferenceAudioButton(button, true);
    try {
      await audio.play();
    } catch (error) {
      stopReferenceReviewAudio();
      showToast(error.message, { persistent: true });
    }
  }

  function renderReferenceReviewCards() {
    const grid = byId("referenceCandidateGrid");
    const manifest = state.referenceReviewManifest;
    const jobId = state.referenceReviewJobId;
    const artifacts = artifactsForJob(jobId, "reference");
    const paths = new Map(artifacts.map(artifact => [referenceArtifactPath(artifact), artifact]));
    if (!manifest || !Array.isArray(manifest.entries) || manifest.entries.length === 0) {
      grid.replaceChildren(create("p", "reference-review-loading", "No verified blind candidates are available"));
      return;
    }
    const cards = manifest.entries.map(entry => {
      const label = String(entry.label || "?");
      const card = create("article", "reference-candidate");
      const heading = create("header", "reference-candidate-heading");
      const letter = create("span", "reference-candidate-letter", label);
      const title = create("div");
      title.append(create("small", "", "BLIND CANDIDATE"), create("h3", "", `${i18n.t("Candidate")} ${label}`));
      heading.append(letter, title);
      const cases = create("div", "reference-case-list");
      for (const sample of Array.isArray(entry.cases) ? entry.cases : []) {
        const artifact = paths.get(String(sample.path || "").replaceAll("\\", "/"));
        const row = create("div", "reference-case");
        const button = create("button", "reference-audio-button");
        button.type = "button";
        const time = create("span", "reference-case-time", "--:--");
        const copy = create("div", "reference-case-copy");
        copy.append(create("small", "", `${i18n.t("Sentence")} ${String(sample.case).padStart(2, "0")}`), create("p", "", sample.text || ""));
        const audio = document.createElement("audio");
        audio.preload = "metadata";
        audio.hidden = true;
        if (artifact) audio.src = `/api/workstation/artifacts/${encodeURIComponent(artifact.id)}/content`;
        button.disabled = !artifact;
        setReferenceAudioButton(button, false);
        button.addEventListener("click", () => toggleReferenceAudio(audio, button));
        audio.addEventListener("loadedmetadata", () => { time.textContent = audioClock(audio.duration); });
        audio.addEventListener("timeupdate", () => {
          if (state.referenceReviewAudio?.audio === audio) {
            time.textContent = `${audioClock(audio.currentTime)} / ${audioClock(audio.duration)}`;
          }
        });
        audio.addEventListener("ended", () => {
          time.textContent = audioClock(audio.duration);
          if (state.referenceReviewAudio?.audio === audio) stopReferenceReviewAudio({ rewind: true });
        });
        audio.addEventListener("error", () => {
          button.disabled = true;
          setReferenceAudioButton(button, false);
        });
        row.append(button, copy, time, audio);
        cases.append(row);
      }
      const select = create("button", "primary-button reference-select-button");
      select.type = "button";
      select.dataset.referenceItemId = entry.candidate_item_id;
      select.textContent = `${i18n.t("Use as reference")} ${label}`;
      select.addEventListener("click", () => lockTrainingReference(entry.candidate_item_id));
      card.append(heading, cases, select);
      return card;
    });
    grid.replaceChildren(...cards);
    grid.dataset.referenceJobId = jobId;
    grid.dataset.locale = state.locale;
    replaceIcons(grid);
  }

  async function loadReferenceReview(job) {
    if (!job || state.referenceReviewLoading) return;
    const artifacts = artifactsForJob(job.id, "reference");
    const reportArtifact = artifacts.find(artifact => referenceArtifactPath(artifact) === "reference-selection-report.json");
    const manifestArtifact = artifacts.find(artifact => referenceArtifactPath(artifact) === "blind-reference-manifest.json");
    if (!reportArtifact || !manifestArtifact) {
      setText("referenceReviewState", "Verified evidence unavailable");
      byId("referenceCandidateGrid").replaceChildren(create("p", "reference-review-loading", "Blind reference artifacts are incomplete"));
      return;
    }
    state.referenceReviewLoading = true;
    setText("referenceReviewState", "Loading verified audio");
    try {
      const [report, manifest] = await Promise.all([
        api(`/api/workstation/artifacts/${encodeURIComponent(reportArtifact.id)}/content`),
        api(`/api/workstation/artifacts/${encodeURIComponent(manifestArtifact.id)}/content`)
      ]);
      if (latestProjectJob(state.selectedTrainingId, "reference.select")?.id !== job.id) return;
      state.referenceReviewJobId = job.id;
      state.referenceReviewReport = report;
      state.referenceReviewManifest = manifest;
      setReferenceReviewCount(manifest.entries?.length || 0);
      renderReferenceReviewCards();
    } catch (error) {
      setText("referenceReviewState", "Verified evidence unavailable");
      byId("referenceCandidateGrid").replaceChildren(create("p", "reference-review-loading", error.message));
    } finally {
      state.referenceReviewLoading = false;
    }
  }

  function renderReferenceReview(project, lifecycle) {
    const panel = byId("referenceReviewPanel");
    const pending = Boolean(project && lifecycle.reference?.status === "succeeded"
      && project.config?.reference_status !== "human-locked"
      && project.config?.reference_status !== "all-poor");
    panel.hidden = !pending;
    if (!pending) {
      stopReferenceReviewAudio({ rewind: true });
      return;
    }
    if (state.referenceReviewJobId === lifecycle.reference.id && state.referenceReviewManifest) {
      setReferenceReviewCount(state.referenceReviewManifest.entries?.length || 0);
      const grid = byId("referenceCandidateGrid");
      if (grid.dataset.referenceJobId !== lifecycle.reference.id || grid.dataset.locale !== state.locale) {
        stopReferenceReviewAudio({ rewind: true });
        renderReferenceReviewCards();
      }
    } else {
      void loadReferenceReview(lifecycle.reference);
    }
  }

  async function lockTrainingReference(itemId, decision = null) {
    const project = selectedTrainingProject();
    if (!project || (!itemId && decision !== "no-preference")) return;
    const automatic = state.referenceReviewReport?.automatic_recommendation;
    const humanDecision = decision || (automatic === itemId ? "confirm-auto-winner" : "preferred");
    const buttons = byId("referenceReviewPanel").querySelectorAll("button");
    buttons.forEach(button => { button.disabled = true; });
    stopReferenceReviewAudio({ rewind: true });
    try {
      const body = { decision: humanDecision };
      if (itemId) body.item_id = itemId;
      await api(`/api/workstation/training/${encodeURIComponent(project.id)}/reference-selection/lock`, {
        method: "POST",
        body: JSON.stringify(body)
      });
      state.referenceReviewManifest = null;
      state.referenceReviewReport = null;
      state.referenceReviewJobId = null;
      await Promise.all([loadOverview(), loadArtifacts()]);
      showToast("Reference locked; holdout evaluation queued");
    } catch (error) {
      buttons.forEach(button => { button.disabled = false; });
      showToast(error.message, { persistent: true });
    }
  }

  async function useAutomaticTrainingReference() {
    if (!state.referenceReviewReport?.automatic_recommendation) {
      showToast("Verified evidence unavailable");
      return;
    }
    await lockTrainingReference(null, "no-preference");
  }

  async function rejectAllTrainingReferences() {
    const project = selectedTrainingProject();
    const button = byId("referenceAllPoorButton");
    if (!project || !button) return;
    if (!state.referenceRejectArmed) {
      state.referenceRejectArmed = true;
      button.classList.add("armed");
      button.textContent = i18n.t("Confirm rejection of all A–E");
      setTimeout(() => {
        if (!state.referenceRejectArmed) return;
        state.referenceRejectArmed = false;
        button.classList.remove("armed");
        button.replaceChildren();
        const icon = document.createElement("i");
        icon.dataset.lucide = "circle-x";
        icon.setAttribute("aria-hidden", "true");
        button.append(icon, document.createTextNode(` ${i18n.t("All references are poor")}`));
        replaceIcons(button);
      }, 6000);
      return;
    }
    button.disabled = true;
    stopReferenceReviewAudio({ rewind: true });
    try {
      await api(`/api/workstation/training/${encodeURIComponent(project.id)}/reference-selection/lock`, {
        method: "POST",
        body: JSON.stringify({ decision: "all-poor" })
      });
      state.referenceRejectArmed = false;
      await Promise.all([loadOverview(), loadArtifacts()]);
      showToast("All references rejected; holdout remains sealed");
    } catch (error) {
      button.disabled = false;
      showToast(error.message, { persistent: true });
    }
  }

  function renderTrainingWorkspace(project) {
    const config = project?.config && typeof project.config === "object" ? project.config : {};
    setText("trainingProjectTitle", project?.name || "No experiment selected");
    replaceDefinitionList(byId("trainingProjectInputs"), project ? [
      ["Dataset", shortPath(config.dataset || "Unavailable"), config.dataset],
      ["Pretrained GPT", shortPath(config.pretrained_gpt || "Unavailable"), config.pretrained_gpt],
      ["SoVITS G / D", config.pretrained_sovits_g && config.pretrained_sovits_d ? "Configured" : "Unavailable"],
      ["Preset", `${config.preset || "balanced"} · ${config.stage || "both"}`],
      ["Resume", config.resume_checkpoint ? shortPath(config.resume_checkpoint) : "Fresh run", config.resume_checkpoint]
    ] : [["Dataset", "Unavailable"], ["Preset", "Unavailable"], ["Resume", "Unavailable"]]);

    const lifecycle = renderTrainingLifecycle(project);
    const latestJob = lifecycle.training;
    const active = latestJob && ["queued", "running", "paused"].includes(latestJob.status);
    const missing = trainingInputsMissing(project);
    byId("trainingRunButton").disabled = !project || Boolean(active) || missing.length > 0;
    const trainingSucceeded = project
      ? state.jobs.filter(job => job.project_id === project.id && job.type === "training.prepare" && job.status === "succeeded")[0]
      : null;
    const productionJobs = project
      ? state.jobs.filter(job => job.project_id === project.id && [
        "checkpoint.select", "reference.select", "holdout.evaluate", "engine.prepare", "conversion.parity", "model.package"
      ].includes(job.type))
      : [];
    const productionActive = productionJobs.some(job => ["queued", "running", "paused"].includes(job.status));
    const packageReady = lifecycle.modelPackage?.status === "succeeded" ? lifecycle.modelPackage : null;
    const selectionMissing = trainingSucceeded && !lifecycle.selection;
    const referenceReview = lifecycle.reference?.status === "succeeded"
      && !["human-locked", "all-poor"].includes(config.reference_status);
    const productionButton = byId("productionBuildButton");
    productionButton.disabled = !selectionMissing && !referenceReview;
    productionButton.title = referenceReview ? i18n.t("Review reference A/B") : "";
    setText("productionBuildHint", !project
      ? "TensorRT package is not built."
      : packageReady
        ? `API-ready TensorRT package · ${jobControls.shortId(packageReady.id)}`
        : lifecycle.evaluationProject?.config?.qualification_status === "blocked"
          ? lifecycle.evaluationProject.config.blocked_reason || "Canonical evaluation is blocked by missing evidence."
        : lifecycle.parity?.status === "failed"
          ? "Conversion parity failed. Packaging is blocked."
        : lifecycle.holdout?.status === "failed"
          ? "Test holdout failed. TensorRT conversion is blocked."
        : referenceReview
          ? "Blind reference listening is required before the test holdout can open."
        : productionActive
          ? "Production validation is progressing through the locked job graph."
          : !trainingSucceeded
            ? "Complete training before building the production model."
            : selectionMissing
              ? "Training is complete; validation checkpoint selection is ready to start."
              : "The production workflow is waiting for its next verified dependency.");
    setText("trainingProjectHint", !project
      ? "Select a configured experiment."
      : missing.length
        ? `Missing required inputs: ${missing.join(", ")}`
        : active
          ? `${latestJob.status} worker job ${jobControls.shortId(latestJob.id)}`
          : "Runs only in the isolated Linux CUDA training worker.");
    renderReferenceReview(project, lifecycle);
    renderTrainingRun(project, latestJob);
    replaceIcons(productionButton);
  }

  function renderTrainingRun(project, fallbackJob = null) {
    const job = state.jobs.find(record => record.id === state.selectedTrainingJobId)
      || fallbackJob
      || (project ? latestProjectJob(project.id, "training.prepare") : null);
    const detailMatches = state.trainingJobDetail?.id === job?.id;
    const logs = detailMatches ? state.trainingJobLogs : [];
    const report = job?.result?.backend?.payload;
    const validReport = ["aniflive-tts-v2proplus-training-report-v1", "aniflive-tts-v2proplus-training-report-v2"].includes(report?.schema) ? report : null;
    setText("trainingMonitorTitle", job ? `${project?.name || "Training"} · ${jobControls.shortId(job.id)}` : "No training run selected");
    setText("trainingMonitorState", statusText(job?.status || "Idle"));
    setText("trainingProgressLabel", progressText(job?.progress));
    setText("trainingProgressStage", latestRunMessage(logs, job ? "Worker events not loaded" : "Waiting for a real worker event"));
    displayRunProgress("trainingProgress", job?.progress);

    const checkpointArtifacts = artifactsForJob(job?.id, "checkpoint");
    const checkpointFiles = checkpointArtifacts.filter(artifact => /\.(ckpt|pth)$/i.test(artifact.name || ""));
    const gpu = state.overview?.gpu && typeof state.overview.gpu === "object" ? state.overview.gpu : {};
    const utilization = numericValue(gpu, ["utilization_percent", "utilization", "gpu_utilization_percent"]);
    const used = numericValue(gpu, ["memory_used_bytes", "memory_used", "vram_used_bytes"]);
    const total = numericValue(gpu, ["memory_total_bytes", "memory_total", "vram_total_bytes"]);
    const temperature = numericValue(gpu, ["temperature_c", "temperature", "temperature_celsius"]);
    const structuredLoss = numericValue(validReport?.metrics || job?.result?.metrics, ["loss", "train_loss"]);
    replaceDefinitionList(byId("trainingMetrics"), [
      ["Epoch", trainingEpoch(logs)],
      ["Loss", structuredLoss === null ? "Unavailable · worker does not emit structured loss" : structuredLoss.toFixed(6)],
      ["Checkpoint", checkpointFiles.length ? `${checkpointFiles.length} ready` : (validReport ? "No deployable checkpoint artifact" : "Unavailable")],
      ["GPU", utilization === null ? "Unavailable" : `${utilization.toFixed(0)}%`],
      ["VRAM", used === null ? "Unavailable" : `${formatBytes(used)}${total === null ? "" : ` / ${formatBytes(total)}`} `],
      ["Temperature", temperature === null ? "Unavailable" : `${temperature.toFixed(0)}°C`],
      ["ETA", estimatedEta(job)]
    ]);
    byId("trainingEventList").replaceChildren(...runEventNodes(logs));
    byId("trainingArtifactList").replaceChildren(...artifactNodes(checkpointArtifacts, "No ready checkpoint artifacts"));
    replaceIcons(byId("trainingRunButton"));
  }

  async function selectTrainingProject(project) {
    state.selectedTrainingId = project?.id || null;
    const latestJob = project ? latestProjectJob(project.id, "training.prepare") : null;
    state.selectedTrainingJobId = latestJob?.id || null;
    state.trainingJobDetail = null;
    state.trainingJobLogs = [];
    renderProjectTables();
    if (latestJob) await loadTrainingRunDetails(latestJob.id);
  }

  async function loadTrainingRunDetails(jobId = state.selectedTrainingJobId) {
    if (!jobId) return;
    try {
      const payload = await api(`/api/workstation/jobs/${encodeURIComponent(jobId)}`);
      if (state.selectedTrainingJobId !== jobId) return;
      state.trainingJobDetail = payload.job || null;
      state.trainingJobLogs = Array.isArray(payload.logs) ? payload.logs : [];
      const index = state.jobs.findIndex(job => job.id === jobId);
      if (index >= 0 && payload.job) state.jobs[index] = payload.job;
      renderTrainingWorkspace(selectedTrainingProject());
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function queueTrainingRun() {
    const project = selectedTrainingProject();
    if (!project || trainingInputsMissing(project).length) return;
    try {
      const sourceArtifactId = project.config?.source_dataset_artifact_id;
      const parameters = typeof sourceArtifactId === "string" && sourceArtifactId
        ? { parent_artifact_ids: [sourceArtifactId] }
        : {};
      const job = await api("/api/workstation/jobs", {
        method: "POST",
        body: JSON.stringify({ type: "training.prepare", project_id: project.id, parameters })
      });
      state.selectedTrainingJobId = job.id;
      state.trainingJobDetail = null;
      state.trainingJobLogs = [];
      await Promise.all([loadOverview(), loadArtifacts()]);
      await loadTrainingRunDetails(job.id);
      showToast(`Training queued · ${jobControls.shortId(job.id)}`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function queueProductionBuild() {
    const project = selectedTrainingProject();
    if (!project) return;
    const training = state.jobs.find(job => (
      job.project_id === project.id
      && job.type === "training.prepare"
      && job.status === "succeeded"
    ));
    if (!training) return;
    const selection = latestProjectJob(project.id, "checkpoint.select");
    const reference = latestProjectJob(project.id, "reference.select");
    if (reference?.status === "succeeded" && project.config?.reference_status !== "human-locked") {
      const panel = byId("referenceReviewPanel");
      panel.hidden = false;
      if (state.referenceReviewJobId !== reference.id) await loadReferenceReview(reference);
      panel.scrollIntoView({ behavior: "smooth", block: "start" });
      showToast("Blind reference listening is ready");
      return;
    }
    if (selection) return;
    try {
      const checkpointSelection = await createWorkstationJob("checkpoint.select", project.id, {
        dependsOn: [training.id]
      });
      await Promise.all([loadOverview(), loadArtifacts()]);
      renderTrainingWorkspace(selectedTrainingProject());
      showToast(`Checkpoint selection queued · ${jobControls.shortId(checkpointSelection.id)}`);
    } catch (error) {
      await loadOverview().catch(() => {});
      showToast(error.message, { persistent: true });
    }
  }

  function selectedEvaluationProject() {
    return state.projects.find(project => project.kind === "evaluation" && project.id === state.selectedEvaluationId) || null;
  }

  function evaluationInputsMissing(project) {
    const config = project?.config && typeof project.config === "object" ? project.config : {};
    return ["model_package", "shared_dir", "asr_model"]
      .filter(key => typeof config[key] !== "string" || !config[key].trim());
  }

  function updateEvaluationSelectors(currentJob) {
    const baseline = byId("evaluationBaselineJob");
    const previousBaseline = baseline.value;
    const language = state.evaluationAudioLanguage;
    const candidates = state.jobs.filter(job => (
      job.type === "evaluation.prepare"
      && job.status === "succeeded"
      && job.id !== currentJob?.id
      && artifactsForJob(job.id, "evaluation").some(artifact => artifact.name === `${language}-complete.wav`)
    ));
    baseline.replaceChildren(
      new Option("No completed baseline run", ""),
      ...candidates.map(job => {
        const project = state.projects.find(record => record.id === job.project_id);
        return new Option(`${project?.name || "Evaluation"} · ${jobControls.shortId(job.id)}`, job.id);
      })
    );
    if (candidates.some(job => job.id === previousBaseline)) baseline.value = previousBaseline;

    const dependency = byId("evaluationTrainingDependency");
    const previousDependency = dependency.value;
    const packageJobs = state.jobs.filter(job => (
      job.type === "model.package" && !["failed", "cancelled"].includes(job.status)
    ));
    dependency.replaceChildren(
      new Option("No package dependency", ""),
      ...packageJobs.map(job => {
        const project = state.projects.find(record => record.id === job.project_id);
        return new Option(`${project?.name || "Package"} · ${job.status} · ${jobControls.shortId(job.id)}`, job.id);
      })
    );
    if (packageJobs.some(job => job.id === previousDependency)) dependency.value = previousDependency;
  }

  function evaluationMetric(benchmark, key) {
    const row = Array.isArray(benchmark?.table) ? benchmark.table.find(item => item.key === key) : null;
    return Number.isFinite(Number(row?.median)) ? Number(row.median) : null;
  }

  function evaluationGateItem(label, status, detail) {
    const item = create("li", status);
    item.append(create("strong", "", label), create("span", "", detail));
    return item;
  }

  function renderEvaluationGates(report, benchmark) {
    const languages = report?.languages && typeof report.languages === "object" ? report.languages : {};
    const languageRows = Object.values(languages);
    const measured = report?.schema === "aniflive-tts-workstation-evaluation-v1";
    const allSpeaker = measured && languageRows.length === 5 && languageRows.every(row => row.quality_gate?.speaker === true);
    const contentDetail = measured
      ? languageRows.map(row => `${String(row.asr?.metric || "error").toUpperCase()} ${(Number(row.asr?.error_rate) * 100).toFixed(2)}%`).join(" · ")
      : "Unavailable";
    const ttfa = evaluationMetric(benchmark, "stream_keepalive_audible_ttfa_p50_ms");
    const rtf = evaluationMetric(benchmark, "wall_rtf_p50");
    byId("evaluationRunGateList").replaceChildren(
      evaluationGateItem("Five languages", !measured ? "unavailable" : report.gates?.five_language_enqueue ? "passed" : "failed", contentDetail),
      evaluationGateItem("Speaker identity", !measured ? "unavailable" : allSpeaker ? "passed" : "failed", measured ? "Stream/complete speaker cosine gate across all five languages" : "Unavailable"),
      evaluationGateItem("Streaming parity", !measured ? "unavailable" : report.gates?.stream_complete_quality ? "passed" : "failed", measured ? "Log-mel, speaker and duration gates" : "Unavailable"),
      evaluationGateItem("Long-form continuity", "unavailable", "Not measured by the current evaluation worker"),
      evaluationGateItem("Expression quality", "unavailable", "Not measured by the current evaluation worker"),
      evaluationGateItem("TensorRT runtime", !measured ? "unavailable" : report.gates?.tensor_rt_engine_contract ? "passed" : "failed", measured ? "Linux container · TensorRT 11" : "Unavailable"),
      evaluationGateItem("TTFA and RTF", ttfa !== null && rtf !== null ? "measured" : "unavailable", ttfa !== null && rtf !== null ? `${ttfa.toFixed(3)} ms · RTF ${rtf.toFixed(6)}` : "Canonical benchmark artifact unavailable"),
      evaluationGateItem("Security", "unavailable", measured && report.gates?.no_pytorch_neural_fallback ? "No PyTorch neural fallback measured; full security qualification not emitted" : "Full security qualification not emitted")
    );
  }

  function renderEvaluationAudio(job) {
    const language = state.evaluationAudioLanguage;
    const candidate = artifactsForJob(job?.id, "evaluation").find(artifact => artifact.name === `${language}-complete.wav`) || null;
    const baselineJobId = byId("evaluationBaselineJob").value;
    const baseline = artifactsForJob(baselineJobId, "evaluation").find(artifact => artifact.name === `${language}-complete.wav`) || null;
    const decks = [
      [byId("evaluationCandidateAudio"), byId("evaluationCandidateLabel"), candidate, "No ready candidate audio artifact"],
      [byId("evaluationBaselineAudio"), byId("evaluationBaselineLabel"), baseline, baselineJobId ? "Selected run has no ready audio artifact" : "Choose a completed baseline run"]
    ];
    for (const [audio, label, artifact, unavailable] of decks) {
      if (artifact) {
        const src = `/api/workstation/artifacts/${encodeURIComponent(artifact.id)}/content`;
        if (audio.getAttribute("src") !== src) {
          audio.pause();
          audio.src = src;
        }
        audio.hidden = false;
        label.textContent = `${language.toUpperCase()} · ${artifact.name}`;
      } else {
        audio.pause();
        audio.removeAttribute("src");
        audio.hidden = true;
        label.textContent = unavailable;
      }
    }
  }

  function renderEvaluationLanguageTable(report) {
    const tbody = byId("evaluationLanguageRows");
    const languages = report?.languages && typeof report.languages === "object" ? report.languages : {};
    const order = ["zh", "yue", "ja", "en", "ko"];
    const rows = order.filter(language => languages[language]).map(language => {
      const value = languages[language];
      const row = create("tr");
      const errorRate = Number(value.asr?.error_rate);
      const quality = value.quality || {};
      const gate = value.quality_gate?.passed === true;
      const gateCell = create("td");
      gateCell.append(statusLabel(gate ? "passed" : "failed", gate ? "Passed" : "Failed"));
      row.append(
        create("td", "", language.toUpperCase()),
        create("td", "", Number.isFinite(errorRate) ? `${String(value.asr?.metric || "error").toUpperCase()} ${(errorRate * 100).toFixed(2)}%` : "Unavailable"),
        create("td", "", Number.isFinite(Number(quality.log_mel_cosine)) ? Number(quality.log_mel_cosine).toFixed(6) : "Unavailable"),
        create("td", "", Number.isFinite(Number(quality.speaker_cosine_stream_vs_complete)) ? Number(quality.speaker_cosine_stream_vs_complete).toFixed(6) : "Unavailable"),
        create("td", "", Number.isFinite(Number(quality.duration_difference_ratio)) ? `${(Number(quality.duration_difference_ratio) * 100).toFixed(2)}%` : "Unavailable"),
        gateCell
      );
      return row;
    });
    if (!rows.length) emptyTable(tbody, 6, "No measured evaluation report");
    else tbody.replaceChildren(...rows);
    labelTableCells(tbody.closest("table"));
  }

  function renderEvaluationWorkspace(project) {
    const config = project?.config && typeof project.config === "object" ? project.config : {};
    setText("evaluationProjectTitle", project?.name || "No evaluation selected");
    const latestJob = project ? latestProjectJob(project.id, "evaluation.prepare") : null;
    const selectedJob = state.jobs.find(job => job.id === state.selectedEvaluationJobId) || latestJob;
    updateEvaluationSelectors(selectedJob);
    const missing = evaluationInputsMissing(project);
    const active = selectedJob && ["queued", "running", "paused"].includes(selectedJob.status);
    byId("evaluationRunButton").disabled = !project || Boolean(active) || missing.length > 0;
    byId("evaluationChainButton").disabled = !project || Boolean(active) || missing.length > 0;
    setText("evaluationWorkflowHint", !project
      ? "Select a configured evaluation."
      : missing.length
        ? `Missing required inputs: ${missing.join(", ")}`
        : "Dependencies control scheduling. Evaluation consumes the verified output of its package dependency.");
    renderEvaluationRun(project, selectedJob, config);
  }

  function renderEvaluationRun(project, job) {
    const detailMatches = state.evaluationJobDetail?.id === job?.id;
    const logs = detailMatches ? state.evaluationJobLogs : [];
    const report = detailMatches && state.evaluationReport
      ? state.evaluationReport
      : job?.result?.backend?.payload?.schema === "aniflive-tts-workstation-evaluation-v1"
        ? job.result.backend.payload
        : null;
    const benchmark = detailMatches ? state.evaluationBenchmark : null;
    setText("evaluationRunTitle", job ? `${project?.name || "Evaluation"} · ${jobControls.shortId(job.id)}` : "No evaluation run selected");
    setText("evaluationRunState", statusText(job?.status || "Idle"));
    setText("evaluationProgressLabel", progressText(job?.progress));
    setText("evaluationProgressStage", latestRunMessage(logs, job ? "Worker events not loaded" : "Waiting for a real worker event"));
    displayRunProgress("evaluationProgress", job?.progress);

    const ttfaP50 = evaluationMetric(benchmark, "stream_keepalive_audible_ttfa_p50_ms");
    const ttfaP95 = evaluationMetric(benchmark, "stream_keepalive_audible_ttfa_p95_ms");
    const rtf = evaluationMetric(benchmark, "wall_rtf_p50");
    replaceDefinitionList(byId("evaluationPerformance"), [
      ["Keep-alive audible TTFA P50", ttfaP50 === null ? "Unavailable" : `${ttfaP50.toFixed(3)} ms`],
      ["Keep-alive audible TTFA P95", ttfaP95 === null ? "Unavailable" : `${ttfaP95.toFixed(3)} ms`],
      ["Complete-WAV RTF P50", rtf === null ? "Unavailable" : rtf.toFixed(6)]
    ]);
    renderEvaluationLanguageTable(report);
    renderEvaluationGates(report, benchmark);
    renderEvaluationAudio(job);
    byId("evaluationEventList").replaceChildren(...runEventNodes(logs));
    byId("evaluationRunArtifacts").replaceChildren(...artifactNodes(artifactsForJob(job?.id, "evaluation"), "No ready evaluation artifacts"));
    replaceIcons(byId("evaluationRunButton"));
  }

  async function readJobJsonArtifact(jobId, name) {
    const artifact = artifactsForJob(jobId, "evaluation").find(item => item.name === name);
    return artifact ? api(`/api/workstation/artifacts/${encodeURIComponent(artifact.id)}/content`) : null;
  }

  async function selectEvaluationProject(project) {
    state.selectedEvaluationId = project?.id || null;
    const latestJob = project ? latestProjectJob(project.id, "evaluation.prepare") : null;
    state.selectedEvaluationJobId = latestJob?.id || null;
    state.evaluationJobDetail = null;
    state.evaluationJobLogs = [];
    state.evaluationReport = null;
    state.evaluationBenchmark = null;
    await loadArtifacts();
    renderProjectTables();
    if (latestJob) await loadEvaluationRunDetails(latestJob.id);
  }

  async function loadEvaluationRunDetails(jobId = state.selectedEvaluationJobId) {
    if (!jobId) return;
    try {
      const payload = await api(`/api/workstation/jobs/${encodeURIComponent(jobId)}`);
      if (state.selectedEvaluationJobId !== jobId) return;
      state.evaluationJobDetail = payload.job || null;
      state.evaluationJobLogs = Array.isArray(payload.logs) ? payload.logs : [];
      state.evaluationReport = payload.job?.result?.backend?.payload?.schema === "aniflive-tts-workstation-evaluation-v1"
        ? payload.job.result.backend.payload
        : await readJobJsonArtifact(jobId, "evaluation-report.json");
      state.evaluationBenchmark = await readJobJsonArtifact(jobId, "benchmark.json");
      const index = state.jobs.findIndex(job => job.id === jobId);
      if (index >= 0 && payload.job) state.jobs[index] = payload.job;
      renderEvaluationWorkspace(selectedEvaluationProject());
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function createWorkstationJob(type, projectId, { dependsOn = [], parameters = {} } = {}) {
    return api("/api/workstation/jobs", {
      method: "POST",
      body: JSON.stringify({ type, project_id: projectId, parameters, depends_on: dependsOn })
    });
  }

  async function queueEvaluationRun({ chain = false } = {}) {
    const project = selectedEvaluationProject();
    if (!project || evaluationInputsMissing(project).length) return;
    const dependencyId = byId("evaluationTrainingDependency").value;
    const dependencies = dependencyId ? [dependencyId] : [];
    const upstream = state.jobs.find(job => job.id === dependencyId);
    const parentArtifacts = upstream?.status === "succeeded" && Array.isArray(upstream.result?.registered_artifact_ids)
      ? upstream.result.registered_artifact_ids
      : [];
    try {
      let evaluation;
      if (chain) {
        const engine = await createWorkstationJob("engine.prepare", project.id, { dependsOn: dependencies });
        const modelPackage = await createWorkstationJob("model.package", project.id, { dependsOn: [engine.id] });
        evaluation = await createWorkstationJob("evaluation.prepare", project.id, { dependsOn: [modelPackage.id] });
      } else {
        evaluation = await createWorkstationJob("evaluation.prepare", project.id, {
          dependsOn: dependencies,
          parameters: parentArtifacts.length ? { parent_artifact_ids: parentArtifacts } : {}
        });
      }
      state.selectedEvaluationJobId = evaluation.id;
      state.evaluationJobDetail = null;
      state.evaluationJobLogs = [];
      state.evaluationReport = null;
      state.evaluationBenchmark = null;
      await Promise.all([loadOverview(), loadArtifacts()]);
      await loadEvaluationRunDetails(evaluation.id);
      showToast(chain ? "Engine → package → evaluation dependency chain queued" : `Evaluation queued · ${jobControls.shortId(evaluation.id)}`);
    } catch (error) {
      await loadOverview().catch(() => {});
      showToast(error.message, { persistent: true });
    }
  }

  function selectedTseProject() {
    return state.projects.find(project => project.kind === "tse" && project.id === state.selectedTseId) || null;
  }

  function tseJobIsActive(job) {
    return Boolean(job && ["queued", "running", "paused"].includes(job.status));
  }

  function setTseField(id, config, key, fallback) {
    const input = byId(id);
    const value = config && Object.prototype.hasOwnProperty.call(config, key) ? config[key] : fallback;
    input.value = String(value ?? fallback);
  }

  function renderTsePackageOptions() {
    const paths = new Set();
    state.projects.filter(project => project.kind === "tse").forEach(project => {
      const value = project.config?.model_package;
      if (typeof value === "string" && value.trim()) paths.add(value.trim());
    });
    state.models.forEach(model => {
      const value = model?.model_package || model?.package_path || model?.path;
      if (typeof value === "string" && value.trim()) paths.add(value.trim());
    });
    byId("tseModelPackages").replaceChildren(...[...paths].sort().map(path => {
      const option = create("option");
      option.value = path;
      return option;
    }));
  }

  function renderTseProjectState(project) {
    renderTsePackageOptions();
    const job = project ? latestTseJob(project.id) : null;
    const active = tseJobIsActive(job) || tseJobIsActive(state.jobs.find(item => item.id === state.selectedTseJobId));
    setText("tseActiveProject", project?.name || "Select a TSE project");
    setText("tseControlStatus", project ? (job?.status || "Ready to queue") : "Not configured");
    byId("tseRunButton").disabled = !project || active;
    byId("tseRerunButton").disabled = !project || active || !state.tseReport;
    byId("tseCancelButton").disabled = !active;
  }

  function resetTseInspector() {
    clearTimeout(state.tsePollTimer);
    state.tsePollTimer = 0;
    setText("tseJobTitle", "No extraction selected");
    setText("tseJobState", "Idle");
    setText("tseJobProgressLabel", "0%");
    byId("tseJobProgress").value = 0;
    byId("tseJobEvents").replaceChildren(create("p", "", "No worker events"));
    byId("tseCancelButton").disabled = true;
    byId("tseResult").hidden = true;
    const audio = byId("tseTargetAudio");
    audio.removeAttribute("src");
    audio.load();
  }

  async function selectTseProject(project) {
    if (!project || project.kind !== "tse") return;
    clearTimeout(state.tsePollTimer);
    state.selectedTseId = project.id;
    state.tseReport = null;
    state.tseArtifacts = [];
    state.tseReviewDecisions = new Map();
    state.tseSeparationSegments = tseModel.normalizeSeparationSegments(
      project.config?.separation_segments || []
    );
    const config = project.config && typeof project.config === "object" ? project.config : {};
    setTseField("tseSource", config, "source", "");
    setTseField("tseReference", config, "reference", "");
    setTseField("tseModelPackage", config, "model_package", "");
    setTseField("tseSeparationModel", config, "separation_model", "");
    setTseField(
      "tseTargetThreshold",
      config,
      "target_threshold",
      state.settings.default_tse_target_threshold
    );
    setTseField("tseReviewMargin", config, "review_margin", 0.08);
    setTseField("tseFrameMs", config, "frame_ms", 20);
    setTseField("tseMinimumSpeechMs", config, "minimum_speech_ms", 160);
    setTseField("tseMaximumGapMs", config, "maximum_gap_ms", 120);
    setTseField("tseContextMs", config, "context_ms", 40);
    setTseField("tseExtractionGapMs", config, "extraction_gap_ms", 120);
    setTseField("tseSeparationAmbiguityMargin", config, "separation_ambiguity_margin", 0.03);
    byId("tseFormError").textContent = "";
    const job = latestTseJob(project.id);
    state.selectedTseJobId = job?.id || null;
    renderProjectTables();
    if (job) await inspectTseJob(job.id);
    else resetTseInspector();
  }

  function tseFormValues() {
    return {
      source: byId("tseSource").value,
      reference: byId("tseReference").value,
      model_package: byId("tseModelPackage").value,
      separation_model: byId("tseSeparationModel").value,
      target_threshold: byId("tseTargetThreshold").value,
      review_margin: byId("tseReviewMargin").value,
      frame_ms: byId("tseFrameMs").value,
      minimum_speech_ms: byId("tseMinimumSpeechMs").value,
      maximum_gap_ms: byId("tseMaximumGapMs").value,
      context_ms: byId("tseContextMs").value,
      extraction_gap_ms: byId("tseExtractionGapMs").value,
      separation_ambiguity_margin: byId("tseSeparationAmbiguityMargin").value
    };
  }

  async function queueTseRun({ reviewed = false } = {}) {
    const project = selectedTseProject();
    if (!project) return;
    const run = byId("tseRunButton");
    const rerun = byId("tseRerunButton");
    run.disabled = true;
    rerun.disabled = true;
    byId("tseFormError").textContent = "";
    try {
      const parameters = tseModel.buildRunParameters(
        tseFormValues(),
        reviewed ? state.tseReviewDecisions : {},
        state.tseSeparationSegments
      );
      const job = await api("/api/workstation/jobs", {
        method: "POST",
        body: JSON.stringify({
          type: "tse.prepare",
          project_id: project.id,
          parameters
        })
      });
      state.selectedTseJobId = job.id;
      state.tseReport = null;
      state.tseArtifacts = [];
      byId("tseResult").hidden = true;
      await loadOverview();
      await inspectTseJob(job.id);
      showToast(reviewed ? "Reviewed TSE run queued" : "TSE run queued for Linux Docker");
    } catch (error) {
      byId("tseFormError").textContent = error.message;
      renderTseProjectState(project);
    }
  }

  function renderTseJobDetail(job, logs) {
    const progress = job.progress;
    const presentation = jobControls.statusPresentation(job);
    setText("tseJobTitle", `${job.type || "TSE"} · ${jobControls.shortId(job.id)}`);
    setText("tseJobState", presentation.label);
    setText("tseJobProgressLabel", progressText(progress));
    displayRunProgress("tseJobProgress", progress);
    const events = byId("tseJobEvents");
    if (!logs.length) events.replaceChildren(create("p", "", "No worker events"));
    else events.replaceChildren(...logs.slice(-10).map(log => {
      const row = create("div", "tse-event");
      row.append(create("time", "", formatDate(log.created_at)), create("strong", log.level || "info", log.message || ""));
      return row;
    }));
    if (job.error) byId("tseFormError").textContent = job.error;
    renderTseProjectState(selectedTseProject());
  }

  function scheduleTsePoll(jobId) {
    clearTimeout(state.tsePollTimer);
    if (state.currentView !== "tse" || state.selectedTseJobId !== jobId) return;
    state.tsePollTimer = window.setTimeout(() => inspectTseJob(jobId), 1400);
  }

  async function loadTseResult(job) {
    const project = selectedTseProject();
    if (!project || job.project_id !== project.id) return;
    const payload = await api(`/api/workstation/artifacts?project_id=${encodeURIComponent(project.id)}&status=ready`);
    const artifacts = tseModel.artifactsForJob(payload.data, job.id);
    const reportArtifact = artifacts.find(artifact => artifact.name === "tse-report.json");
    const audioArtifact = artifacts.find(artifact => artifact.name === "target-speaker.wav");
    if (!reportArtifact || !audioArtifact) throw new Error("TSE job completed without its verified report and target audio artifacts");
    const reportUrl = `/api/workstation/artifacts/${encodeURIComponent(reportArtifact.id)}/content`;
    const audioUrl = `/api/workstation/artifacts/${encodeURIComponent(audioArtifact.id)}/content`;
    const report = tseModel.validateReport(await api(reportUrl));
    if (state.selectedTseJobId !== job.id) return;
    state.tseArtifacts = artifacts;
    state.tseReport = report;
    state.tseReviewDecisions = tseModel.decisionsFromReport(report);
    state.tseSeparationSegments = tseModel.separationSegmentsFromReport(report);
    renderTseResult(report, { reportArtifact, audioArtifact, reportUrl, audioUrl });
    renderTseProjectState(project);
  }

  function renderTseResult(report, artifacts) {
    byId("tseResult").hidden = false;
    setText("tseResultBackend", `${report.speaker_backend} · ${report.overlap_policy}`);
    const audio = byId("tseTargetAudio");
    if (audio.getAttribute("src") !== artifacts.audioUrl) {
      audio.src = artifacts.audioUrl;
      audio.load();
    }
    renderTseAudioPlayer();
    byId("tseAudioArtifact").href = `${artifacts.audioUrl}?download=1`;
    byId("tseReportArtifact").href = `${artifacts.reportUrl}?download=1`;
    const summary = byId("tseSummary");
    const summaryRows = [
      ["Source", `${Number(report.source_seconds).toFixed(2)} s`],
      ["Target", `${Number(report.target_seconds).toFixed(2)} s`],
      ["Review", `${Number(report.review_seconds).toFixed(2)} s`],
      ["Segments", `${report.target_segments} target · ${report.review_segments} review · ${report.rejected_segments} rejected`]
    ];
    summary.replaceChildren(...summaryRows.map(([term, value]) => definition(term, value)));
    const rows = report.segments.map(segment => {
      const row = create("tr");
      const start = segment.start_sample / report.sample_rate;
      const end = segment.end_sample / report.sample_rate;
      const select = create("select", "tse-decision");
      select.setAttribute("aria-label", `Decision for segment ${segment.index + 1}`);
      select.append(new Option("Target · accept", "target"), new Option("Needs review", "review"), new Option("Reject", "rejected"));
      select.value = state.tseReviewDecisions.get(String(segment.index)) || segment.decision;
      select.addEventListener("change", () => {
        state.tseReviewDecisions.set(String(segment.index), select.value);
        byId("tseRerunButton").disabled = tseJobIsActive(state.jobs.find(item => item.id === state.selectedTseJobId));
      });
      const decision = create("td"); decision.append(select);
      const range = { start_seconds: start, end_seconds: end };
      const overlapsRange = candidate => candidate.start_seconds < end && start < candidate.end_seconds;
      const force = document.createElement("input");
      force.type = "checkbox";
      force.className = "tse-separation-toggle";
      force.setAttribute("aria-label", `Force source separation for segment ${segment.index + 1}`);
      force.checked = state.tseSeparationSegments.some(overlapsRange);
      force.addEventListener("change", () => {
        const remaining = state.tseSeparationSegments.filter(candidate => !overlapsRange(candidate));
        if (force.checked) remaining.push(range);
        state.tseSeparationSegments = tseModel.normalizeSeparationSegments(
          remaining.sort((left, right) => left.start_seconds - right.start_seconds)
        );
        byId("tseRerunButton").disabled = tseJobIsActive(state.jobs.find(item => item.id === state.selectedTseJobId));
      });
      const separation = create("td"); separation.append(force);
      row.append(
        create("td", "", `#${String(segment.index + 1).padStart(3, "0")}`),
        create("td", "", `${start.toFixed(2)}–${end.toFixed(2)} s`),
        create("td", "", segment.similarity === null ? "—" : segment.similarity.toFixed(3)),
        create("td", segment.overlap ? "tse-overlap" : "", segment.overlap ? "Review" : "Clear"),
        separation,
        decision
      );
      return row;
    });
    if (!rows.length) emptyTable(byId("tseSegmentRows"), 6, "No voice activity was detected");
    else byId("tseSegmentRows").replaceChildren(...rows);
    labelTableCells(byId("tseSegmentRows").closest("table"));
  }

  async function inspectTseJob(jobId) {
    if (!jobId || state.selectedTseJobId !== jobId) return;
    try {
      const payload = await api(`/api/workstation/jobs/${encodeURIComponent(jobId)}`);
      const job = payload.job || {};
      if (state.selectedTseJobId !== jobId || job.project_id !== state.selectedTseId || job.type !== "tse.prepare") return;
      const index = state.jobs.findIndex(item => item.id === job.id);
      if (index >= 0) state.jobs[index] = job;
      else state.jobs.unshift(job);
      renderTseJobDetail(job, Array.isArray(payload.logs) ? payload.logs : []);
      renderProjectTables();
      if (job.status === "succeeded") {
        if (!state.tseReport) await loadTseResult(job);
      } else if (tseJobIsActive(job)) {
        scheduleTsePoll(job.id);
      }
    } catch (error) {
      byId("tseFormError").textContent = error.message;
    }
  }

  async function cancelSelectedTseJob() {
    const jobId = state.selectedTseJobId;
    if (!jobId) return;
    try {
      await api(`/api/workstation/jobs/${encodeURIComponent(jobId)}/cancel`, { method: "POST" });
      await inspectTseJob(jobId);
      showToast("TSE cancellation requested");
    } catch (error) {
      byId("tseFormError").textContent = error.message;
    }
  }

  function selectedDatasetProject() {
    return state.projects.find(record => record.kind === "dataset" && record.id === state.selectedDatasetId) || null;
  }

  function datasetProjectSources(project) {
    const values = Array.isArray(project?.config?.sources) ? project.config.sources : [];
    if (values.length) return values.filter(value => typeof value === "string" && value.trim());
    const source = projectConfig(project, "source", "");
    return source ? [source] : [];
  }

  function datasetAcquisitionMode(project) {
    return project?.config?.acquisition_mode || state.datasetProjectState?.acquisition_mode || "standard";
  }

  function renderDatasetLifecycle() {
    const stateRecord = state.datasetProjectState;
    const stages = Array.isArray(stateRecord?.workflow_stages)
      ? stateRecord.workflow_stages
      : ["source", "speaker", "clean", "text", "review", "style", "ready"];
    const current = stages.indexOf(stateRecord?.lifecycle_stage || "source");
    byId("datasetLifecycle").querySelectorAll("[data-dataset-stage]").forEach(node => {
      const index = stages.indexOf(node.dataset.datasetStage);
      node.classList.toggle("complete", index >= 0 && index < current);
      node.classList.toggle("active", index === current);
      node.setAttribute("aria-current", index === current ? "step" : "false");
    });
  }

  function datasetAcquisitionJobs(project) {
    if (!project) return [];
    return ["dataset.target-speaker", "dataset.separate", "dataset.transcribe", "dataset.finalize"]
      .map(type => latestDatasetAcquisitionJob(project.id, type))
      .filter(Boolean);
  }

  function datasetAcquisitionPayload(job) {
    let value = job?.result;
    for (let depth = 0; depth < 5 && value && typeof value === "object"; depth += 1) {
      if (value.schema || value.counts || value.records) return value;
      if (value.backend && typeof value.backend === "object") value = value.backend;
      else if (value.payload && typeof value.payload === "object") value = value.payload;
      else break;
    }
    return value && typeof value === "object" ? value : null;
  }

  function renderDatasetAcquisition(project) {
    const mode = datasetAcquisitionMode(project);
    const target = mode === "target-speaker";
    const jobs = datasetAcquisitionJobs(project);
    const finalJob = project ? latestDatasetAcquisitionJob(project.id, "dataset.finalize") : null;
    const active = jobs.some(job => ["queued", "running", "paused"].includes(job.status));
    const prepare = byId("datasetPrepareTargetButton");
    const importButton = byId("datasetImportAcquisitionButton");
    prepare.hidden = !target;
    importButton.hidden = !target || state.datasetAcquisitionImportFailures.size === 0;
    prepare.disabled = !project || !target || active || finalJob?.status === "succeeded";
    importButton.disabled = !target || finalJob?.status !== "succeeded" || state.datasetLoading;
    byId("datasetAcquisitionResult").hidden = !target || jobs.length === 0;
    if (!target || jobs.length === 0) return;
    const progress = jobs.reduce((sum, job) => sum + Math.max(0, Math.min(1, Number(job.progress) || 0)), 0) / 4;
    const current = [...jobs].reverse().find(job => job.status !== "succeeded") || finalJob || jobs[jobs.length - 1];
    setText("datasetAcquisitionState", String(current?.status || "queued").replaceAll("-", " "));
    setText("datasetAcquisitionProgress", progressText(progress));
    byId("datasetAcquisitionProgressBar").value = progress;
    const payload = datasetAcquisitionPayload(finalJob) || datasetAcquisitionPayload(current);
    const counts = payload?.counts && typeof payload.counts === "object" ? payload.counts : {};
    byId("datasetAcquisitionCounts").replaceChildren(
      datasetSummaryRow("Clean", Number(counts.clean) || 0),
      datasetSummaryRow("Salvaged", Number(counts.salvage) || 0),
      datasetSummaryRow("Review", Number(counts.review) || 0),
      datasetSummaryRow("Rejected", Number(counts.reject) || 0)
    );
    setText(
      "datasetAcquisitionSummary",
      finalJob?.status === "succeeded"
        ? "Per-clip target-speaker artifacts are checksum verified and ready to import."
        : current?.status === "failed"
          ? current.error || "Target-speaker acquisition failed."
          : `${String(current?.type || "dataset.target-speaker").replace("dataset.", "")} · ${String(current?.status || "queued")}`
    );
    const routingJob = jobs.find(job => job.type === "dataset.target-speaker");
    const routingPayload = datasetAcquisitionPayload(routingJob);
    const evidence = payload?.diarization ? payload : routingPayload;
    renderDatasetSpeakerEvidence(evidence, jobs);
  }

  function datasetSummaryRow(term, value) {
    const row = create("div");
    row.append(create("dt", "", term), create("dd", "", value));
    return row;
  }

  function datasetEvidenceMetric(term, value, detail = "") {
    const row = create("div", "dataset-evidence-metric");
    row.append(create("dt", "", term), create("dd", "", value));
    if (detail) row.append(create("small", "", detail));
    return row;
  }

  function renderDatasetSpeakerEvidence(payload, jobs) {
    const report = byId("datasetAdvancedSpeakerReport");
    const contents = [];
    if (payload && typeof payload === "object") {
      const reference = payload.reference_prototype || {};
      const diarization = payload.diarization || {};
      const sensitive = payload.sensitive_diarization || {};
      const policy = payload.routing_policy || {};
      const postprocess = diarization.postprocess || {};
      const sensitivePostprocess = sensitive.postprocess || {};
      const records = Array.isArray(payload.records) ? payload.records : [];
      const overlapEvidence = records.filter(record => record?.acquisition?.overlap_evidence).length;
      const speakerChangeEvidence = records.filter(record => record?.acquisition?.speaker_change_evidence).length;
      const metrics = create("dl", "dataset-evidence-grid");
      metrics.append(
        datasetEvidenceMetric(
          "Reference prototypes",
          String(Number(reference.reference_count) || 0),
          reference.aggregation || "top-k mean"
        ),
        datasetEvidenceMetric(
          "Primary diarization",
          `${Number(diarization.clip_count) || records.length} clips`,
          Number.isFinite(Number(postprocess.onset))
            ? `Sortformer · onset ${Number(postprocess.onset).toFixed(2)}`
            : "Sortformer"
        ),
        datasetEvidenceMetric(
          "Sensitive scan",
          `${Number(sensitive.candidate_count) || 0} candidates`,
          `${Number(sensitive.evidence_count) || 0} evidence · onset ${Number(sensitivePostprocess.onset || 0).toFixed(2)}`
        ),
        datasetEvidenceMetric(
          "Speaker evidence",
          `${overlapEvidence} overlap · ${speakerChangeEvidence} multi-speaker`,
          payload.speaker_backend || "TensorRT verifier"
        ),
        datasetEvidenceMetric(
          "Clean SNR gate",
          Number.isFinite(Number(policy.minimum_clean_snr_db))
            ? `≥ ${Number(policy.minimum_clean_snr_db).toFixed(0)} dB`
            : "Not reported",
          "Lower-quality clips remain in review"
        ),
        datasetEvidenceMetric(
          "Network",
          payload.network_required === false ? "Offline" : "Not reported",
          "Linux Docker neural worker"
        )
      );
      contents.push(metrics);
    } else {
      contents.push(create("p", "dataset-evidence-empty", "Verified routing evidence appears here after the Linux Docker worker completes."));
    }
    const chain = create("ol", "dataset-evidence-chain");
    jobs.forEach(job => {
      const row = create("li", `dataset-evidence-job ${job.status || "queued"}`);
      row.append(
        create("strong", "", job.type.replace("dataset.", "")),
        create("span", "", `${job.status} · ${jobControls.shortId(job.id)}`)
      );
      chain.append(row);
    });
    contents.push(chain);
    report.replaceChildren(...contents);
  }

  function datasetAudioLabel(item) {
    const parts = [];
    if (Number.isFinite(Number(item.duration_seconds))) parts.push(`${Number(item.duration_seconds).toFixed(2)} s`);
    if (Number.isFinite(Number(item.sample_rate))) parts.push(`${Math.round(Number(item.sample_rate) / 100) / 10} kHz`);
    if (Number.isFinite(Number(item.channels))) parts.push(Number(item.channels) === 1 ? "mono" : `${item.channels} ch`);
    return parts.join(" · ") || "Decoder required";
  }

  function selectedDatasetItem() {
    return state.datasetItems.find(item => item.id === state.selectedDatasetItemId) || null;
  }

  function datasetAnnotationLabel(item) {
    const annotation = item?.annotations || {};
    if (annotation.transcript && annotation.language && annotation.speaker) {
      return `${String(annotation.language).toUpperCase()} · ${annotation.speaker}`;
    }
    if (annotation.transcript) return "Partial";
    return "Required";
  }

  function datasetQualityLabel(item) {
    const score = Number(item?.quality?.quality_score);
    return Number.isFinite(score) ? `${score.toFixed(1)} / 100` : "Not analyzed";
  }

  function drawDatasetWaveform(payload) {
    const canvas = byId("datasetWaveform");
    const context = canvas.getContext("2d");
    const peaks = Array.isArray(payload?.peaks) ? payload.peaks : [];
    context.clearRect(0, 0, canvas.width, canvas.height);
    const middle = canvas.height / 2;
    context.strokeStyle = "rgba(225, 210, 222, .18)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(0, middle + .5);
    context.lineTo(canvas.width, middle + .5);
    context.stroke();
    if (!peaks.length) return;
    const gradient = context.createLinearGradient(0, 0, canvas.width, 0);
    gradient.addColorStop(0, "#e7cddd");
    gradient.addColorStop(.58, "#b78ab4");
    gradient.addColorStop(1, "#8d496c");
    context.strokeStyle = gradient;
    context.lineWidth = Math.max(1, canvas.width / peaks.length * .72);
    context.beginPath();
    peaks.forEach((peak, index) => {
      const low = Math.max(-1, Math.min(1, Number(peak?.[0]) || 0));
      const high = Math.max(-1, Math.min(1, Number(peak?.[1]) || 0));
      const x = (index + .5) / peaks.length * canvas.width;
      context.moveTo(x, middle - high * middle * .88);
      context.lineTo(x, middle - low * middle * .88);
    });
    context.stroke();
  }

  function datasetMetric(term, value) {
    const row = create("div");
    row.append(create("dt", "", term), create("dd", "", value));
    return row;
  }

  function renderDatasetItemInspector() {
    const inspector = byId("datasetItemInspector");
    const item = selectedDatasetItem();
    inspector.hidden = !item;
    if (!item) {
      const audio = byId("datasetAudioPreview");
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
      renderDatasetAudioPlayer();
      setText("datasetWaveformStatus", "");
      drawDatasetWaveform(null);
      return;
    }
    setText("datasetInspectorName", item.original_name || item.id);
    const annotation = item.annotations || {};
    const suggestion = item.metadata?.acquisition?.asr_suggestion || {};
    byId("datasetTranscript").value = annotation.transcript || suggestion.transcript || "";
    byId("datasetLanguage").value = annotation.language || suggestion.language || "";
    byId("datasetSpeaker").value = annotation.speaker || "";
    byId("datasetExpression").value = annotation.expression || "";
    const expressionIntensity = Number.isFinite(Number(annotation.expression_intensity))
      ? Number(annotation.expression_intensity)
      : 0.7;
    byId("datasetExpressionIntensity").value = String(expressionIntensity);
    setText("datasetExpressionIntensityValue", expressionIntensity.toFixed(2));
    byId("datasetValence").value = annotation.valence ?? "";
    byId("datasetArousal").value = annotation.arousal ?? "";
    byId("datasetDominance").value = annotation.dominance ?? "";
    byId("datasetStyleDescription").value = annotation.style_description || "";
    const annotationSources = {
      "gpt-sovits-list": "GPT-SoVITS .list",
      "inherited-requires-transcript": "Inherited · transcript required",
      manual: "Manual"
    };
    setText("datasetAnnotationSource", annotation.transcript
      ? annotationSources[annotation.source] || "Manual"
      : suggestion.transcript
        ? "ASR suggestion · confirmation required"
        : "Manual");
    const diagnostic = annotation.language_diagnostic;
    setText(
      "datasetAnnotationDiagnostic",
      diagnostic?.language
        ? `Text diagnostic: ${String(diagnostic.language).toUpperCase()} · ${(Number(diagnostic.confidence) * 100).toFixed(0)}%${diagnostic.diagnostic_only ? " · confirm manually" : ""}`
        : "Language is stored explicitly; text inference is diagnostic only."
    );
    const quality = item.quality;
    byId("datasetQualityMetrics").replaceChildren(...(
      quality
        ? [
            datasetMetric("Score", `${Number(quality.quality_score).toFixed(1)} / 100`),
            datasetMetric("Estimated SNR", `${Number(quality.estimated_snr_db).toFixed(1)} dB`),
            datasetMetric("RMS", `${Number(quality.rms_dbfs).toFixed(1)} dBFS`),
            datasetMetric("Peak", `${Number(quality.peak_dbfs).toFixed(1)} dBFS`),
            datasetMetric("Silence", `${(Number(quality.silence_ratio) * 100).toFixed(1)}%`),
            datasetMetric("Clipped", `${(Number(quality.clipped_ratio) * 100).toFixed(3)}%`)
          ]
        : [datasetMetric("Status", "PCM quality has not been analyzed")]
    ));
    const inspectorPending = state.datasetLoading || state.pendingDatasetActions.size > 0;
    const frozen = state.datasetProjectState?.frozen === true;
    const segment = item.kind === "segment";
    byId("datasetQualityAnalyze").disabled = inspectorPending || !String(item.stored_path || "").toLowerCase().endsWith(".wav");
    byId("datasetAnnotationSave").disabled = inspectorPending || frozen;
    byId("datasetTranscriptVerify").disabled = inspectorPending || frozen || !segment;
    byId("datasetSpeakerVerify").disabled = inspectorPending || frozen || !segment;
    byId("datasetAcceptAudio").disabled = inspectorPending || frozen || !segment;
    byId("datasetRejectAudio").disabled = inspectorPending || frozen || !segment;
    const requireExpressions = state.datasetProjectState?.config?.require_expressions !== false;
    byId("datasetExpressionVerify").disabled = inspectorPending || frozen || !segment || !byId("datasetExpression").value.trim();
    setText("datasetExpressionRequirement", requireExpressions ? "Required for this dataset" : "Optional / Not required");
    const setVerificationStatus = (id, verification, optional = false) => {
      const node = byId(id);
      const valid = verification?.valid === true;
      node.textContent = valid ? "Verified" : optional ? "Optional" : "Pending";
      node.classList.toggle("verified", valid);
      node.classList.toggle("rejected", verification?.decision === "rejected");
    };
    const verification = item.verification || {};
    const audioStatus = byId("datasetAudioReviewStatus");
    audioStatus.textContent = item.review_status === "accepted" ? "Accepted" : item.review_status === "rejected" ? "Rejected" : "Pending";
    audioStatus.classList.toggle("verified", item.review_status === "accepted");
    audioStatus.classList.toggle("rejected", item.review_status === "rejected");
    setVerificationStatus("datasetTranscriptVerificationStatus", verification.transcript);
    setVerificationStatus("datasetSpeakerVerificationStatus", verification.speaker);
    setVerificationStatus("datasetExpressionVerificationStatus", verification.expression, !requireExpressions);
    const stageLabels = { normalize: "Normalize", annotation: "Annotate", quality: "Quality" };
    byId("datasetItemStages").replaceChildren(...Object.entries(item.stage_state || {}).map(([id, stage]) => {
      const row = create("li", `dataset-item-stage ${stage.state || "waiting"}`);
      row.append(create("strong", "", stageLabels[id] || id.charAt(0).toUpperCase() + id.slice(1)), create("span", "", stage.detail || stage.state));
      return row;
    }));
    const datasetId = encodeURIComponent(item.dataset_id);
    const itemId = encodeURIComponent(item.id);
    const audioUrl = `/api/workstation/datasets/${datasetId}/items/${itemId}/audio`;
    const audio = byId("datasetAudioPreview");
    if (String(item.stored_path || "").toLowerCase().endsWith(".wav")) {
      if (audio.getAttribute("src") !== audioUrl) {
        audio.pause();
        audio.src = audioUrl;
        audio.load();
      }
    } else {
      audio.pause();
      audio.removeAttribute("src");
      audio.load();
    }
    renderDatasetAudioPlayer();
    setText("datasetWaveformDuration", Number.isFinite(Number(item.duration_seconds)) ? `${Number(item.duration_seconds).toFixed(2)} s` : "PCM required");
  }

  async function loadDatasetWaveform(item) {
    const request = ++state.datasetWaveformRequest;
    if (!String(item?.stored_path || "").toLowerCase().endsWith(".wav")) {
      drawDatasetWaveform(null);
      setText("datasetWaveformStatus", "A trusted Linux Docker decoder is required before waveform inspection.");
      return;
    }
    setText("datasetWaveformStatus", "Calculating waveform from verified PCM…");
    try {
      const payload = await api(`/api/workstation/datasets/${encodeURIComponent(item.dataset_id)}/items/${encodeURIComponent(item.id)}/waveform?bins=640`);
      if (request !== state.datasetWaveformRequest || item.id !== state.selectedDatasetItemId) return;
      drawDatasetWaveform(payload);
      setText("datasetWaveformStatus", `${payload.frame_count.toLocaleString()} PCM frames · ${payload.sample_rate.toLocaleString()} Hz`);
    } catch (error) {
      if (request !== state.datasetWaveformRequest) return;
      drawDatasetWaveform(null);
      setText("datasetWaveformStatus", error.message);
    }
  }

  function inspectDatasetItem(item) {
    state.selectedDatasetItemId = item.id;
    setText("datasetAnnotationError", "");
    renderDatasetItemInspector();
    loadDatasetWaveform(item);
    byId("datasetItemInspector").scrollIntoView({ behavior: "smooth", block: "nearest" });
  }

  function datasetActionButton(item, action) {
    const key = `${item.id}:${action.id}`;
    const button = create("button", "row-action", action.label);
    button.type = "button";
    button.disabled = state.datasetLoading || state.pendingDatasetActions.has(key);
    button.addEventListener("click", () => runDatasetAction(item, action.id));
    return button;
  }

  function renderDatasetWorkspace(project = selectedDatasetProject()) {
    const process = byId("datasetProcessButton");
    const processImport = byId("datasetProcessImportButton");
    const ingest = byId("datasetIngestButton");
    const importList = byId("datasetImportListButton");
    const refresh = byId("datasetRefreshButton");
    const split = byId("datasetSplitButton");
    const manifest = byId("datasetManifestButton");
    const qualificationReport = byId("datasetQualificationReportButton");
    const pipeline = byId("datasetPipeline");
    const rows = byId("datasetItemRows");
    const pending = state.datasetLoading || state.pendingDatasetActions.size > 0;
    const summary = datasetModel.summarize(state.datasetItems);
    const sources = datasetProjectSources(project);
    const source = sources[0] || "";
    const acquisitionMode = datasetAcquisitionMode(project);
    const standardMode = acquisitionMode === "standard";
    const listMode = acquisitionMode === "gpt-sovits-list";
    const targetMode = acquisitionMode === "target-speaker";
    const listSource = source.toLowerCase().endsWith(".list");
    const processJobs = project ? datasetProcessJobs(project.id) : [];
    const processJob = processJobs[0] || null;
    const processActive = processJobs.some(job => ["queued", "running", "paused"].includes(job.status));
    const completedProcessJobs = processJobs.filter(job => job.status === "succeeded");

    setText("datasetActiveName", project?.name || "Select a project");
    setText("datasetActiveSource", sources.length > 1 ? `${sources.length} source media items` : source || "No source configured");
    const modeLabels = {
      standard: "Audio / Video Collection",
      "target-speaker": "Extract Target Speaker",
      "gpt-sovits-list": "Existing GPT-SoVITS Dataset"
    };
    setText("datasetAcquisitionMode", modeLabels[acquisitionMode] || acquisitionMode);
    byId("datasetSummary").replaceChildren(
      datasetSummaryRow("Sources", Math.max(summary.sources, sources.length)),
      datasetSummaryRow("Segments", summary.segments),
      datasetSummaryRow("Accepted", summary.accepted),
      datasetSummaryRow("Assigned", summary.assigned),
      datasetSummaryRow("Annotated", summary.annotated),
      datasetSummaryRow("Quality mean", summary.qualityMean === null ? "—" : summary.qualityMean.toFixed(1))
    );
    ingest.hidden = true;
    ingest.setAttribute("aria-hidden", "true");
    process.hidden = !standardMode;
    importList.hidden = !listMode;
    ingest.disabled = true;
    process.disabled = !project || !sources.length || processActive || pending;
    processImport.hidden = !standardMode || state.datasetProcessImportFailures.size === 0;
    processImport.disabled = !project || !completedProcessJobs.length || pending;
    importList.disabled = !project || !listSource || pending;
    refresh.disabled = !project || pending;
    split.disabled = !project || !summary.accepted || summary.accepted === summary.assigned || pending;
    manifest.disabled = !project || !datasetModel.manifestReady(state.datasetItems) || pending;
    qualificationReport.disabled = !project || pending;

    const processResult = byId("datasetProcessResult");
    processResult.hidden = !standardMode || !processJobs.length;
    if (processJob) {
      const progress = processJobs.reduce(
        (total, job) => total + Math.max(0, Math.min(1, Number(job.progress) || 0)),
        0
      ) / processJobs.length;
      const payload = processJob.result?.backend?.payload;
      const artifactPaths = Array.isArray(processJob.result?.backend?.artifacts)
        ? processJob.result.backend.artifacts.map(item => item.relative_path).filter(Boolean)
        : [];
      const registered = Array.isArray(processJob.result?.registered_artifact_ids)
        ? processJob.result.registered_artifact_ids.length
        : 0;
      setText("datasetProcessState", processJobs.length > 1
        ? `${completedProcessJobs.length} / ${processJobs.length} sources complete`
        : String(processJob.status || "unknown").replaceAll("-", " "));
      setText("datasetProcessProgress", progressText(progress));
      byId("datasetProcessProgressBar").value = progress;
      if (processJob.status === "succeeded" && payload?.schema === "aniflive-dataset-pipeline-v1") {
        const denoise = payload.denoise?.enabled ? "afftdn enabled" : "denoise disabled";
        setText("datasetProcessSummary", `${Number(payload.sample_rate).toLocaleString()} Hz mono PCM16 · ${Number(payload.duration_seconds).toFixed(2)} s · ${Number(payload.segment_count)} segments · ${denoise}`);
      } else if (processJob.status === "failed") {
        setText("datasetProcessSummary", processJob.error || "Linux Docker processing failed.");
      } else if (processJob.status === "cancelled") {
        setText("datasetProcessSummary", "Linux Docker processing was cancelled.");
      } else if (processJob.status === "paused") {
        setText("datasetProcessSummary", "Processing is paused and can be resumed from Jobs.");
      } else {
        setText("datasetProcessSummary", processJob.status === "running" ? "Linux Docker is decoding and segmenting the source." : "Waiting for the Linux Docker worker.");
      }
      setText(
        "datasetProcessArtifacts",
        artifactPaths.length
          ? `${registered} immutable artifacts registered · ${artifactPaths.slice(0, 3).join(" · ")}${artifactPaths.length > 3 ? ` · +${artifactPaths.length - 3} more` : ""}`
          : "Artifacts appear only after checksum verification and host registration."
      );
    }

    pipeline.replaceChildren(...datasetModel.pipeline(state.datasetItems, state.datasetCapabilities).map(stage => {
      const item = create("li", `dataset-stage ${stage.state}`);
      item.append(create("strong", "", stage.label), create("span", "", stage.detail));
      item.title = stage.detail;
      return item;
    }));
    renderDatasetLifecycle();
    renderDatasetAcquisition(project);
    renderDatasetExpressionCandidates();
    const frozen = state.datasetProjectState?.frozen === true;
    byId("datasetReviewQueueButton").disabled = !project || !summary.segments || pending;
    byId("datasetExpressionCandidatesButton").disabled = !project || !summary.accepted || pending;
    byId("datasetFreezeButton").disabled = !project || frozen || !summary.accepted || pending;
    byId("datasetContinueTrainingButton").disabled = !project || !frozen || pending;

    if (!project) {
      setText("datasetStatus", "Select a dataset project to begin.");
      emptyTable(rows, 9, "No dataset selected");
      state.selectedDatasetItemId = null;
      renderDatasetItemInspector();
      return;
    }
    if (state.datasetLoading) setText("datasetStatus", "Reading the verified dataset inventory…");
    else if (state.pendingDatasetActions.size) setText("datasetStatus", "Applying the selected pipeline operation…");
    else if (!summary.total) setText("datasetStatus", targetMode ? "The target reference and source media are ready for the managed speaker-acquisition workflow." : source ? (listSource ? "The GPT-SoVITS .list is configured and ready to import." : "The source collection is ready for managed Linux Docker preparation.") : "Add a permitted local source path to this project before ingesting.");
    else if (summary.decoderRequired) setText("datasetStatus", `${summary.decoderRequired} item${summary.decoderRequired === 1 ? "" : "s"} require a trusted media decoder before resampling.`);
    else if (summary.noSpeech) setText("datasetStatus", `${summary.noSpeech} item${summary.noSpeech === 1 ? "" : "s"} contain no speech at the current voice-activity gate.`);
    else setText("datasetStatus", `${summary.total} tracked items · ${summary.pending} review pending · ${summary.assigned} split assigned.`);

    if (!state.datasetItems.length) {
      emptyTable(rows, 9, state.datasetLoading ? "Loading dataset inventory" : "No ingested items");
      state.selectedDatasetItemId = null;
      renderDatasetItemInspector();
      return;
    }
    progressiveTables.render(rows, state.datasetItems, item => {
      const row = create("tr");
      row.classList.toggle("selected", item.id === state.selectedDatasetItemId);
      const pipelineCell = create("td");
      pipelineCell.append(statusLabel(item.pipeline_state, String(item.pipeline_state || "unknown").replaceAll("-", " ")));
      const reviewCell = create("td");
      if (item.kind === "segment") reviewCell.append(statusLabel(item.review_status || "pending"));
      else reviewCell.textContent = "—";
      const actionCell = create("td", "row-actions");
      const buttons = create("div", "row-action-buttons");
      const inspect = create("button", "row-action", item.id === state.selectedDatasetItemId ? "Inspecting" : "Inspect");
      inspect.type = "button";
      inspect.addEventListener("click", () => inspectDatasetItem(item));
      buttons.append(inspect);
      buttons.append(...datasetModel.actions(item).map(action => datasetActionButton(item, action)));
      if (!buttons.childElementCount && item.pipeline_state === "decoder-required") {
        buttons.append(create("span", "dataset-blocked-label", "Decoder needed"));
      }
      actionCell.append(buttons);
      row.append(
        create("td", "dataset-item-name", item.original_name || item.id),
        create("td", "", item.kind || "—"),
        create("td", "", datasetAudioLabel(item)),
        create("td", "", datasetAnnotationLabel(item)),
        create("td", "", datasetQualityLabel(item)),
        pipelineCell,
        reviewCell,
        create("td", "", item.split_name || "—"),
        actionCell
      );
      return row;
    }, state.selectedDatasetItemId);
    labelTableCells(rows.closest("table"));
    if (state.selectedDatasetItemId && !selectedDatasetItem()) state.selectedDatasetItemId = null;
    renderDatasetItemInspector();
  }

  async function selectDataset(project) {
    if (!project || project.kind !== "dataset") return;
    state.selectedDatasetId = project.id;
    state.datasetItems = [];
    state.datasetProjectState = null;
    state.datasetReviewQueue = null;
    state.datasetExpressionCandidates = null;
    state.selectedDatasetItemId = null;
    renderProjectTables();
    await loadSelectedDataset();
  }

  async function loadSelectedDataset({ announce = false } = {}) {
    const datasetId = state.selectedDatasetId;
    if (!datasetId || state.datasetLoading) return;
    state.datasetLoading = true;
    renderDatasetWorkspace();
    try {
      const readProjectState = () => (
        api(`/api/workstation/datasets/${encodeURIComponent(datasetId)}/state`)
      );
      const initialProjectState = await readProjectState();
      if (state.selectedDatasetId !== datasetId) return;
      if (!initialProjectState.frozen) {
        await autoImportCompletedDatasetJobs(datasetId);
        await autoImportCompletedAcquisitions(datasetId);
      }
      const [payload, capabilities, projectState] = await Promise.all([
        api(`/api/workstation/datasets/${encodeURIComponent(datasetId)}/items`),
        state.datasetCapabilities ? Promise.resolve(state.datasetCapabilities) : api("/api/workstation/datasets/capabilities"),
        initialProjectState.frozen ? Promise.resolve(initialProjectState)
          : readProjectState()
      ]);
      if (state.selectedDatasetId !== datasetId) return;
      state.datasetItems = Array.isArray(payload.items) ? payload.items : [];
      state.datasetCapabilities = capabilities;
      state.datasetProjectState = projectState;
      if (announce) showToast("Dataset inventory refreshed");
    } catch (error) {
      if (state.selectedDatasetId === datasetId) showToast(error.message, { persistent: true });
    } finally {
      if (state.selectedDatasetId === datasetId) {
        state.datasetLoading = false;
        renderDatasetWorkspace();
      }
    }
  }

  async function runDatasetMutation(key, operation, successMessage) {
    if (!state.selectedDatasetId || state.pendingDatasetActions.has(key)) return;
    state.pendingDatasetActions.add(key);
    renderDatasetWorkspace();
    try {
      await operation();
      await loadSelectedDataset();
      showToast(successMessage);
    } catch (error) {
      showToast(error.message, { persistent: true });
    } finally {
      state.pendingDatasetActions.delete(key);
      renderDatasetWorkspace();
    }
  }

  async function ingestSelectedDataset() {
    const project = selectedDatasetProject();
    const sources = datasetProjectSources(project);
    if (!project || !sources.length) {
      showToast("Configure a permitted source path before ingesting");
      return;
    }
    await runDatasetMutation("ingest", () => api(
      `/api/workstation/datasets/${encodeURIComponent(project.id)}/ingest`,
      { method: "POST", body: JSON.stringify({ sources, recursive: true }) }
    ), "Source inventory ingested and SHA-256 verified");
  }

  async function importSelectedDatasetList() {
    const project = selectedDatasetProject();
    const source = projectConfig(project, "source", "");
    if (!project || !source.toLowerCase().endsWith(".list")) {
      showToast("Configure a permitted GPT-SoVITS .list path first");
      return;
    }
    await runDatasetMutation("import-list", () => api(
      `/api/workstation/datasets/${encodeURIComponent(project.id)}/import-list`,
      { method: "POST", body: JSON.stringify({ path: source }) }
    ), "GPT-SoVITS annotations and verified audio imported");
  }

  async function saveDatasetAnnotations(event) {
    event.preventDefault();
    const item = selectedDatasetItem();
    if (!item) return;
    const expression = byId("datasetExpression").value.trim();
    const optionalNumber = id => byId(id).value === "" ? null : Number(byId(id).value);
    const payload = {
      transcript: byId("datasetTranscript").value,
      language: byId("datasetLanguage").value,
      speaker: byId("datasetSpeaker").value,
      expression,
      expression_intensity: expression ? Number(byId("datasetExpressionIntensity").value) : null,
      valence: expression ? optionalNumber("datasetValence") : null,
      arousal: expression ? optionalNumber("datasetArousal") : null,
      dominance: expression ? optionalNumber("datasetDominance") : null,
      style_description: expression ? byId("datasetStyleDescription").value : null,
      propagate: byId("datasetPropagate").checked
    };
    setText("datasetAnnotationError", "");
    const key = `${item.id}:annotations`;
    if (state.pendingDatasetActions.has(key)) return;
    state.pendingDatasetActions.add(key);
    renderDatasetWorkspace();
    try {
      await api(
        `/api/workstation/datasets/${encodeURIComponent(item.dataset_id)}/items/${encodeURIComponent(item.id)}/annotations`,
        { method: "PATCH", body: JSON.stringify(payload) }
      );
      const verifyKind = event.submitter?.dataset?.verifyKind;
      if (verifyKind) {
        await api(
          `/api/workstation/datasets/${encodeURIComponent(item.dataset_id)}/items/${encodeURIComponent(item.id)}/annotation-verifications`,
          { method: "POST", body: JSON.stringify({ kind: verifyKind, decision: "verified", note: "" }) }
        );
      }
      await loadSelectedDataset();
      showToast(verifyKind ? "Transcript saved and verified" : "Training annotation saved");
    } catch (error) {
      setText("datasetAnnotationError", error.message);
      showToast(error.message, { persistent: true });
    } finally {
      state.pendingDatasetActions.delete(key);
      renderDatasetWorkspace();
    }
  }

  async function verifySelectedDatasetAnnotation(kind) {
    const item = selectedDatasetItem();
    if (!item) return;
    const form = byId("datasetAnnotationForm");
    if (!form.reportValidity()) return;
    const expression = byId("datasetExpression").value.trim();
    const optionalNumber = id => byId(id).value === "" ? null : Number(byId(id).value);
    const payload = {
      transcript: byId("datasetTranscript").value,
      language: byId("datasetLanguage").value,
      speaker: byId("datasetSpeaker").value,
      expression,
      expression_intensity: expression ? Number(byId("datasetExpressionIntensity").value) : null,
      valence: expression ? optionalNumber("datasetValence") : null,
      arousal: expression ? optionalNumber("datasetArousal") : null,
      dominance: expression ? optionalNumber("datasetDominance") : null,
      style_description: expression ? byId("datasetStyleDescription").value : null,
      propagate: byId("datasetPropagate").checked
    };
    const key = `${item.id}:verify-${kind}`;
    if (state.pendingDatasetActions.has(key)) return;
    state.pendingDatasetActions.add(key);
    renderDatasetWorkspace();
    try {
      await api(
        `/api/workstation/datasets/${encodeURIComponent(item.dataset_id)}/items/${encodeURIComponent(item.id)}/annotations`,
        { method: "PATCH", body: JSON.stringify(payload) }
      );
      await api(
        `/api/workstation/datasets/${encodeURIComponent(item.dataset_id)}/items/${encodeURIComponent(item.id)}/annotation-verifications`,
        { method: "POST", body: JSON.stringify({ kind, decision: "verified", note: "" }) }
      );
      await loadSelectedDataset();
      showToast(`${kind.charAt(0).toUpperCase() + kind.slice(1)} verified`);
    } catch (error) {
      setText("datasetAnnotationError", error.message);
      showToast(error.message, { persistent: true });
    } finally {
      state.pendingDatasetActions.delete(key);
      renderDatasetWorkspace();
    }
  }

  async function reviewSelectedDatasetAudio(decision) {
    const item = selectedDatasetItem();
    if (!item) return;
    await runDatasetMutation(
      `${item.id}:audio-${decision}`,
      () => api(
        `/api/workstation/datasets/${encodeURIComponent(item.dataset_id)}/items/${encodeURIComponent(item.id)}/review`,
        { method: "PATCH", body: JSON.stringify({ decision, note: "" }) }
      ),
      decision === "accepted" ? "Audio accepted" : "Audio rejected"
    );
  }

  async function recalculateDatasetQuality() {
    const item = selectedDatasetItem();
    if (!item) return;
    await runDatasetMutation(`${item.id}:quality`, () => api(
      `/api/workstation/datasets/${encodeURIComponent(item.dataset_id)}/items/${encodeURIComponent(item.id)}/quality`,
      { method: "POST" }
    ), "Deterministic PCM quality report recalculated");
  }

  async function runDatasetAction(item, action) {
    const datasetId = state.selectedDatasetId;
    if (!datasetId) return;
    const base = `/api/workstation/datasets/${encodeURIComponent(datasetId)}/items/${encodeURIComponent(item.id)}`;
    const operations = {
      resample: () => api(`${base}/resample`, { method: "POST", body: JSON.stringify({ target_rate: 32000, mono: true }) }),
      vad: () => api(`${base}/vad`, { method: "POST", body: "{}" }),
      segment: () => api(`${base}/segments`, { method: "POST", body: "{}" }),
      accept: () => api(`${base}/review`, { method: "PATCH", body: JSON.stringify({ decision: "accepted", note: "" }) }),
      reject: () => api(`${base}/review`, { method: "PATCH", body: JSON.stringify({ decision: "rejected", note: "" }) })
    };
    if (!operations[action]) return;
    const messages = {
      resample: "PCM audio resampled to 32 kHz mono",
      vad: "Voice-activity analysis completed",
      segment: "Speech regions written as reviewable segments",
      accept: "Segment accepted",
      reject: "Segment rejected"
    };
    await runDatasetMutation(`${item.id}:${action}`, operations[action], messages[action]);
  }

  async function assignDatasetSplit() {
    const datasetId = state.selectedDatasetId;
    if (!datasetId) return;
    await runDatasetMutation("split", () => api(
      `/api/workstation/datasets/${encodeURIComponent(datasetId)}/split`,
      { method: "POST", body: JSON.stringify({ train: 0.85, validation: 0.1, test: 0.05, seed: "aniflive-tts-v1.4" }) }
    ), "Accepted segments assigned to deterministic train, validation and test splits");
  }

  async function downloadDatasetManifest() {
    const project = selectedDatasetProject();
    if (!project || !datasetModel.manifestReady(state.datasetItems)) return;
    try {
      const manifest = await api(`/api/workstation/datasets/${encodeURIComponent(project.id)}/manifest`);
      const blob = new Blob([`${JSON.stringify(manifest, null, 2)}\n`], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${project.id}-dataset-manifest.json`;
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      showToast("Verified dataset manifest exported");
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function downloadDatasetQualificationReport() {
    const project = selectedDatasetProject();
    if (!project) return;
    try {
      const report = await api(`/api/workstation/datasets/${encodeURIComponent(project.id)}/qualification-report`);
      const blob = new Blob([`${JSON.stringify(report, null, 2)}\n`], { type: "application/json" });
      const url = URL.createObjectURL(blob);
      const link = document.createElement("a");
      link.href = url;
      link.download = `${project.id}-voice-acquisition-qualification.json`;
      document.body.append(link);
      link.click();
      link.remove();
      URL.revokeObjectURL(url);
      showToast(report.ready_for_release
        ? "Qualification report exported · all gates passed"
        : `Qualification report exported · ${report.missing_gates?.length || 0} gates remain`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  function jobButton(label, project, jobType) {
    const button = create("button", "row-action", label);
    button.type = "button";
    button.addEventListener("click", () => queueProjectJob(project, jobType));
    return button;
  }

  function renderJobs() {
    const tbody = byId("jobRows");
    setText("jobCount", `${state.jobs.length} jobs`);
    const trainingJobs = state.jobs.filter(item => item.type === "training.prepare" && !["cancelled", "failed", "succeeded"].includes(item.status));
    setText("trainingQueue", trainingJobs.length ? `${trainingJobs.length} active or paused` : "No training job");
    const runningCount = state.jobs.filter(item => item.status === "running").length;
    const queuedCount = state.jobs.filter(item => item.status === "queued").length;
    setText("queueRunning", String(runningCount).padStart(2, "0"));
    setText("queueWaiting", String(queuedCount).padStart(2, "0"));
    if (!state.jobs.length) {
      emptyTable(tbody, 7, "No jobs");
      return;
    }
    progressiveTables.render(tbody, state.jobs, job => {
      const row = create("tr");
      row.dataset.jobId = job.id;
      row.dataset.selectable = "true";
      row.classList.toggle("selected", state.selectedJobId === job.id);

      const jobCell = create("td", "job-identity-cell");
      const inspectName = create("button", "row-select job-id", jobControls.shortId(job.id));
      inspectName.type = "button";
      inspectName.title = job.id;
      inspectName.addEventListener("click", () => inspectJob(job.id));
      const lineage = job.retry_of
        ? `Attempt ${job.attempt || 1} · retry of ${jobControls.shortId(job.retry_of)}`
        : `Attempt ${job.attempt || 1} · initial`;
      jobCell.append(inspectName, create("small", "job-lineage", lineage));

      const typeCell = create("td", "job-type-cell");
      const priority = Number.isInteger(job.priority) ? job.priority : 0;
      typeCell.append(
        create("span", "job-type", job.type),
        create("small", "job-priority", `Priority ${priority > 0 ? "+" : ""}${priority}`)
      );

      const dependencyCell = create("td", "job-dependency-cell");
      const dependencies = jobControls.dependencyRecords(job, state.jobs);
      if (!dependencies.length) {
        dependencyCell.append(create("span", "job-dependency-none", "None"));
      } else {
        const dependencySummary = dependencies
          .map(record => `${jobControls.shortId(record.id)} · ${record.status}`)
          .join("\n");
        dependencyCell.title = dependencies
          .map(record => `${record.id} · ${record.status}`)
          .join("\n");
        dependencyCell.append(
          create("span", "job-dependency-count", `${dependencies.length} ${dependencies.length === 1 ? "job" : "jobs"}`),
          create("small", "job-dependency-summary", dependencySummary)
        );
      }

      const presentation = jobControls.statusPresentation(job);
      const status = create("td", "job-status-cell");
      status.append(statusLabel(presentation.value, presentation.label));
      if (job.wait_reason) {
        const reason = create("small", "job-wait-reason", job.wait_reason);
        reason.title = job.wait_reason;
        status.append(reason);
      }
      const actionCell = create("td", "row-actions");
      const actionButtons = create("div", "row-action-buttons");
      actionCell.append(actionButtons);
      const policy = jobControls.actionPolicy(
        job,
        state.jobs,
        state.pendingJobActions.has(job.id)
      );
      const inspect = iconAction("logs", "Inspect job events");
      inspect.type = "button";
      inspect.addEventListener("click", () => inspectJob(job.id));
      actionButtons.append(inspect);
      if (policy.run.visible) {
        const run = iconAction("play", "Run dataset inventory");
        run.disabled = policy.run.disabled;
        if (policy.run.reason) run.title = policy.run.reason;
        run.addEventListener("click", () => runJob(job.id));
        actionButtons.append(run);
      }
      if (policy.pause.visible) {
        const pause = iconAction("pause", "Pause job");
        pause.disabled = policy.pause.disabled;
        if (policy.pause.reason) pause.title = policy.pause.reason;
        pause.addEventListener("click", () => controlJob(job.id, "pause"));
        actionButtons.append(pause);
      }
      if (policy.resume.visible) {
        const resume = iconAction("play", "Resume job");
        resume.disabled = policy.resume.disabled;
        resume.addEventListener("click", () => controlJob(job.id, "resume"));
        actionButtons.append(resume);
      }
      if (policy.cancel.visible) {
        const cancel = iconAction("x", "Cancel job");
        cancel.disabled = policy.cancel.disabled;
        if (policy.cancel.reason) cancel.title = policy.cancel.reason;
        cancel.addEventListener("click", () => cancelJob(job.id));
        actionButtons.append(cancel);
      }
      if (policy.retry.visible) {
        const retry = iconAction("rotate-ccw", "Retry job");
        retry.disabled = policy.retry.disabled;
        if (policy.retry.reason) retry.title = policy.retry.reason;
        retry.addEventListener("click", () => controlJob(job.id, "retry"));
        actionButtons.append(retry);
      }
      replaceIcons(actionCell);
      row.append(
        jobCell,
        typeCell,
        dependencyCell,
        status,
        create("td", "", progressText(job.progress)),
        create("td", "", formatDate(job.updated_at)),
        actionCell
      );
      return row;
    }, state.selectedJobId, job => [ [...state.pendingJobActions], jobControls.dependencyRecords(job, state.jobs) ]);
    labelTableCells(tbody.closest("table"));
    if (state.selectedJobId) {
      const selected = state.jobs.find(job => job.id === state.selectedJobId);
      if (selected) renderJobInspectorMeta(selected);
    }
  }

  function iconAction(icon, label) {
    const button = create("button", "row-action row-action-icon");
    button.type = "button";
    button.title = label;
    button.setAttribute("aria-label", label);
    const glyph = create("i");
    glyph.dataset.lucide = icon;
    glyph.setAttribute("aria-hidden", "true");
    button.append(glyph);
    return button;
  }


  // Keep complete API records in memory; construct only the rows the user reaches.
  const progressiveTables = (() => {
    const tables = new Map();
    const chunk = 40;
    const visible = entry => entry.body.closest(".view")?.classList.contains("active");
    const observer = new IntersectionObserver(entries => {
      entries.forEach(({ target, isIntersecting }) => {
        const entry = tables.get(target.dataset.tableOwner);
        if (isIntersecting && entry && visible(entry) && !target.contains(document.activeElement)) grow(entry);
      });
    }, { rootMargin: "120px 0px", threshold: 0 });

    function queue(entry) {
      if (entry.frame || !visible(entry)) return;
      entry.frame = requestAnimationFrame(() => {
        entry.frame = 0;
        if (visible(entry)) paint(entry);
      });
    }
    function grow(entry, manual = false) {
      if (entry.limit >= entry.records.length) return;
      if (manual) entry.focusNext = entry.limit;
      entry.limit = Math.min(entry.records.length, entry.limit + chunk);
      queue(entry);
    }
    function paint(entry) {
      const { body, records, build } = entry;
      const nodes = [];
      const nextCache = new Map();
      const focused = document.activeElement;
      let focusKey = null;
      let focusIndex = -1;
      entry.cache.forEach((cached, key) => {
        if (cached.node.contains(focused)) {
          focusKey = key;
          focusIndex = [...cached.node.querySelectorAll("button,a,input,select,textarea,[tabindex]")].indexOf(focused);
        }
      });
      records.slice(0, entry.limit).forEach((record, index) => {
        const key = String(record.id ?? index);
        const signature = JSON.stringify([record, entry.selected === record.id, entry.context?.(record)]);
        const old = entry.cache.get(key);
        const node = old?.signature === signature ? old.node : build(record, index);
        if (entry.selected !== undefined) node.classList.toggle("selected", key === String(entry.selected));
        if (body.id === "expressionRows" || body.id === "expressionDraftRows") {
          const source = body.id === "expressionRows" ? "package" : "local";
          const selected = state.selectedExpression?.source === source && state.selectedExpression.id === record.id;
          node.classList.toggle("selected", selected);
          node.setAttribute("aria-selected", String(selected));
        }
        nextCache.set(key, { signature, node });
        nodes.push(node);
      });
      if (entry.limit < records.length) {
        if (!entry.sentinel) {
          const row = create("tr", "ux-lazy-sentinel");
          row.dataset.tableOwner = body.id;
          const cell = create("td");
          cell.colSpan = body.closest("table").querySelectorAll("thead th").length;
          const button = create("button", "text-button", "Show more rows");
          button.type = "button";
          button.addEventListener("click", () => grow(entry, true));
          cell.append(button);
          row.append(cell);
          entry.sentinel = row;
          observer.observe(row);
        }
        nodes.push(entry.sentinel);
      }
      // Stable rows stay attached during polling so focus and native controls survive.
      nodes.forEach((node, index) => {
        if (body.children[index] !== node) body.insertBefore(node, body.children[index] || null);
      });
      while (body.children.length > nodes.length) body.lastElementChild.remove();
      entry.cache = nextCache;
      if (entry.focusNext !== undefined) {
        nodes[entry.focusNext]?.querySelector("button,a,input,select,textarea,[tabindex]")?.focus();
        delete entry.focusNext;
      }
      body.dataset.renderedRows = String(Math.min(entry.limit, records.length));
      body.dataset.totalRows = String(records.length);
      labelTableCells(body.closest("table"));
      if (focused !== document.activeElement && focusKey !== null && focusIndex >= 0) {
        nextCache.get(focusKey)?.node.querySelectorAll("button,a,input,select,textarea,[tabindex]")[focusIndex]?.focus({ preventScroll: true });
      }
    }
    function render(body, records, build, selected, context) {
      let entry = tables.get(body.id);
      if (!entry) {
        entry = { body, records, build, selected, context, limit: chunk, cache: new Map(), frame: 0, sentinel: null };
        tables.set(body.id, entry);
      } else {
        entry.records = records;
        entry.build = build;
        entry.selected = selected;
        entry.context = context;
      }
      queue(entry);
    }
    function clear(body) {
      const entry = tables.get(body.id);
      if (!entry) return;
      if (entry.frame) cancelAnimationFrame(entry.frame);
      if (entry.sentinel) observer.unobserve(entry.sentinel);
      tables.delete(body.id);
      delete body.dataset.renderedRows;
      delete body.dataset.totalRows;
    }
    function activate() { tables.forEach(queue); }
    return { render, clear, activate };
  })();

  function renderModels() {
    const tbody = byId("modelRows");
    byId("modelCount").textContent = `${state.artifacts.length} artifacts`;
    if (!state.artifacts.length) {
      emptyTable(tbody, 6, "No registered production artifacts");
      renderArtifactInspector();
      return;
    }
    progressiveTables.render(tbody, state.artifacts, artifact => {
      const row = create("tr");
      row.classList.toggle("selected", artifact.id === state.selectedArtifactId);
      const stateCell = create("td"); stateCell.append(statusLabel(artifact.status));
      const promotionCell = create("td");
      promotionCell.append(statusLabel(artifact.promoted ? "qualified" : "pending", artifact.promoted ? "Promoted" : "Not promoted"));
      const actionCell = create("td");
      const inspect = create("button", "row-action", "Inspect");
      inspect.type = "button";
      inspect.addEventListener("click", () => inspectArtifact(artifact));
      actionCell.append(inspect);
      row.append(
        createData("td", "", artifact.name || "Unnamed"),
        create("td", "", artifact.type || "Unknown"),
        stateCell,
        promotionCell,
        create("td", "", formatDate(artifact.updated_at)),
        actionCell
      );
      return row;
    }, state.selectedArtifactId);
    labelTableCells(tbody.closest("table"));
    renderArtifactInspector();
  }

  function renderEngines() {
    const health = state.status?.health || {};
    const config = state.status?.config || {};
    const count = Number(health.engine_count || config.engine_count || 0);
    byId("engineRuntime").textContent = health.backend ? `${health.backend} · ${count} engines` : "Runtime unavailable";
    const names = ["SSL", "BERT", "VQ Encoder", "GPT Encoder", "GPT Step", "Spectrogram", "Speaker Embedding", "SoVITS", "SoVITS Stream"];
    const grid = byId("engineGrid");
    const active = Math.min(count, names.length);
    grid.replaceChildren(...names.map((name, index) => {
      const item = create("article", "engine-item");
      item.append(create("span", "", `ENGINE ${String(index + 1).padStart(2, "0")}`), create("strong", "", name), create("small", "", index < active ? "Loaded · TensorRT" : "Not reported"));
      return item;
    }));
    renderEngineArtifacts();
  }

  function renderEngineArtifacts() {
    const records = state.artifacts.filter(artifact => artifact.type === "engine");
    const tbody = byId("engineArtifactRows");
    setText("engineArtifactCount", `${records.length} artifacts`);
    if (!records.length) {
      emptyTable(tbody, 5, "No registered engine artifacts");
      return;
    }
    progressiveTables.render(tbody, records, artifact => {
      const row = create("tr");
      const stateCell = create("td"); stateCell.append(statusLabel(artifact.status));
      const promotionCell = create("td");
      promotionCell.append(statusLabel(artifact.promoted ? "qualified" : "pending", artifact.promoted ? "Promoted" : "Not promoted"));
      const actionCell = create("td");
      const inspect = create("button", "row-action", "Evidence");
      inspect.type = "button";
      inspect.addEventListener("click", () => {
        openView("models");
        inspectArtifact(artifact);
      });
      actionCell.append(inspect);
      row.append(
        createData("td", "", artifact.name || artifact.id),
        stateCell,
        promotionCell,
        create("td", "", `${(artifact.parent_artifact_ids || []).length} direct parents`),
        actionCell
      );
      return row;
    });
    labelTableCells(tbody.closest("table"));
  }

  function qualificationForEvaluation(artifactId) {
    return state.qualifications.find(record => record.evaluation_artifact_id === artifactId) || null;
  }

  function qualificationSubjectForEvaluation(artifact) {
    const imported = qualificationForEvaluation(artifact.id);
    if (imported) return { kind: imported.subject_kind, id: imported.subject_id };
    const metadata = artifact.metadata && typeof artifact.metadata === "object" ? artifact.metadata : {};
    if (["artifact", "expression"].includes(metadata.subject_kind) && typeof metadata.subject_id === "string") {
      return { kind: metadata.subject_kind, id: metadata.subject_id };
    }
    if (typeof metadata.expression_id === "string") return { kind: "expression", id: metadata.expression_id };
    if ((artifact.parent_artifact_ids || []).length === 1) return { kind: "artifact", id: artifact.parent_artifact_ids[0] };
    return null;
  }

  function qualificationEvidenceKind(artifact) {
    const metadata = artifact?.metadata;
    return metadata && typeof metadata === "object"
      ? metadata.qualification_evidence_kind || ""
      : "";
  }

  function replaceQualificationOptions(selectId, records, emptyLabel) {
    const select = byId(selectId);
    const current = select.value;
    select.replaceChildren(
      new Option(emptyLabel, ""),
      ...records.map(artifact => new Option(artifact.name || artifact.id, artifact.id))
    );
    if (records.some(artifact => artifact.id === current)) select.value = current;
  }

  function updateQualificationComposerState() {
    const values = {
      subject: byId("qualificationSubjectEvidence").value,
      automated: byId("qualificationAutomatedEvidence").value,
      longForm: byId("qualificationLongFormEvidence").value,
      expression: byId("qualificationExpressionEvidence").value,
      security: byId("qualificationSecurityEvidence").value,
      content: byId("qualificationContentEvidence").value
    };
    const selected = Object.values(values).filter(Boolean);
    const required = [values.subject, values.automated, values.longForm, values.security];
    const requiredCount = required.filter(Boolean).length;
    const distinct = new Set(selected).size === selected.length;
    const complete = requiredCount === 4 && distinct;
    byId("qualificationComposeButton").disabled = !complete;
    setText(
      "qualificationComposerSubject",
      complete ? "Ready to verify source checksums" : `${requiredCount} / 4 sources selected`
    );
    setText(
      "qualificationComposerHint",
      !distinct
        ? "Choose a different artifact for each selected source."
        : "Select the four required sources. The backend verifies whether expression evidence is applicable; content review is optional."
    );
  }

  function renderQualificationComposer() {
    if (state.currentView !== "evaluation") return;
    const ready = state.artifacts.filter(artifact => artifact.status === "ready");
    replaceQualificationOptions(
      "qualificationSubjectEvidence",
      ready.filter(artifact => promotableArtifactTypes.has(artifact.type)),
      "No ready promotable artifact"
    );
    replaceQualificationOptions(
      "qualificationAutomatedEvidence",
      ready.filter(artifact => artifact.type === "evaluation" && (
        qualificationEvidenceKind(artifact) === "automated-evaluation"
        || (!qualificationEvidenceKind(artifact) && artifact.name === "evaluation-report.json")
      )),
      "No automated evaluation"
    );
    replaceQualificationOptions(
      "qualificationLongFormEvidence",
      ready.filter(artifact => qualificationEvidenceKind(artifact) === "long-form-blind-ab"),
      "Explicit long-form evidence required"
    );
    replaceQualificationOptions(
      "qualificationExpressionEvidence",
      ready.filter(artifact => qualificationEvidenceKind(artifact) === "expression-blind-ab"),
      "Expression evidence · if applicable"
    );
    replaceQualificationOptions(
      "qualificationSecurityEvidence",
      ready.filter(artifact => qualificationEvidenceKind(artifact) === "security-verification"),
      "Explicit security evidence required"
    );
    replaceQualificationOptions(
      "qualificationContentEvidence",
      ready.filter(artifact => artifact.type === "evaluation" && qualificationEvidenceKind(artifact) === "content-review"),
      "No content review selected · optional"
    );
    updateQualificationComposerState();
  }

  function renderEvaluationEvidence() {
    const artifacts = state.artifacts.filter(artifact => artifact.type === "evaluation");
    const tbody = byId("evaluationArtifactRows");
    setText("qualificationCount", `${state.qualifications.length} runs`);
    if (!artifacts.length) {
      emptyTable(tbody, 5, "No verified evaluation artifacts");
      renderEvaluationGateList(null);
      renderQualificationComposer();
      return;
    }
    progressiveTables.render(tbody, artifacts, artifact => {
      const qualification = qualificationForEvaluation(artifact.id);
      const subject = qualificationSubjectForEvaluation(artifact);
      const evidenceKind = qualificationEvidenceKind(artifact);
      const row = create("tr");
      row.classList.toggle("selected", qualification?.id === state.selectedQualificationId);
      const evidenceCell = create("td");
      evidenceCell.append(statusLabel(qualification?.overall_status || "pending", qualification ? qualification.overall_status : "Not imported"));
      const actions = create("td");
      const group = create("div", "row-action-buttons");
      if (qualification) {
        const inspect = create("button", "row-action", "Inspect gates");
        inspect.type = "button";
        inspect.addEventListener("click", () => selectQualification(qualification));
        group.append(inspect);
      } else if (evidenceKind === "composed-qualification") {
        const importButton = create("button", "row-action", "Import evidence");
        importButton.type = "button";
        importButton.disabled = !subject || artifact.status !== "ready";
        importButton.addEventListener("click", () => importQualificationEvidence(artifact, subject));
        group.append(importButton);
      } else {
        group.append(create("span", "artifact-path", evidenceKind || "Unclassified source"));
      }
      if (subject?.kind === "artifact" && subject.id) {
        const inspectSubject = create("button", "row-action", "Inspect subject");
        inspectSubject.type = "button";
        inspectSubject.addEventListener("click", () => inspectQualificationSubject(subject));
        group.append(inspectSubject);
      }
      actions.append(group);
      row.append(
        createData("td", "", artifact.name || artifact.id),
        create("td", "", subject ? `${subject.kind} · ${subject.id}` : "Subject metadata required"),
        evidenceCell,
        create("td", "", formatDate(artifact.updated_at)),
        actions
      );
      return row;
    }, undefined, artifact => [qualificationForEvaluation(artifact.id), state.selectedQualificationId]);
    labelTableCells(tbody.closest("table"));
    const selected = state.qualifications.find(record => record.id === state.selectedQualificationId) || null;
    renderEvaluationGateList(selected);
    renderQualificationComposer();
  }

  function inspectQualificationSubject(subject) {
    if (subject?.kind !== "artifact" || !subject.id) return;
    openView("models");
    const artifact = state.artifacts.find(item => item.id === subject.id) || { id: subject.id };
    return inspectArtifact(artifact);
  }

  function renderEvaluationGateList(qualification) {
    setText("evaluationEvidenceTitle", qualification ? `${qualification.overall_status.toUpperCase()} · ${qualification.id}` : "No evidence selected");
    const list = byId("evaluationGateList");
    if (!qualification) {
      list.replaceChildren(...Object.entries(qualificationGateLabels).map(([, label]) => create("li", "waiting", label)));
      return;
    }
    list.replaceChildren(...qualification.gates.map(gate => {
      const item = create("li", gate.status);
      item.append(create("strong", "", qualificationGateLabels[gate.id] || gate.id));
      const detail = gate.summary || Object.entries(gate.metrics || {}).map(([key, value]) => `${key}: ${value}`).join(" · ") || "No metric summary";
      item.append(create("span", "", detail));
      return item;
    }));
  }

  function selectQualification(qualification) {
    state.selectedQualificationId = qualification?.id || null;
    renderEvaluationEvidence();
  }

  async function importQualificationEvidence(artifact, subject) {
    if (!subject) return;
    try {
      const record = await api("/api/workstation/qualifications/import", {
        method: "POST",
        body: JSON.stringify({
          evaluation_artifact_id: artifact.id,
          subject_kind: subject.kind,
          subject_id: subject.id
        })
      });
      state.selectedQualificationId = record.id;
      await loadArtifacts();
      showToast(`Qualification evidence imported · ${record.overall_status}`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function composeQualificationEvidence() {
    const button = byId("qualificationComposeButton");
    const subjectId = byId("qualificationSubjectEvidence").value;
    if (!subjectId || button.disabled) return;
    button.disabled = true;
    try {
      const result = await api("/api/workstation/qualifications/compose", {
        method: "POST",
        body: JSON.stringify({
          subject_kind: "artifact",
          subject_id: subjectId,
          automated_evaluation_artifact_id: byId("qualificationAutomatedEvidence").value,
          long_form_evidence_artifact_id: byId("qualificationLongFormEvidence").value,
          expression_evidence_artifact_id: byId("qualificationExpressionEvidence").value,
          security_evidence_artifact_id: byId("qualificationSecurityEvidence").value,
          ...(byId("qualificationContentEvidence").value
            ? { content_evidence_artifact_id: byId("qualificationContentEvidence").value } : {})
        })
      });
      state.selectedQualificationId = result.qualification.id;
      await loadArtifacts();
      showToast(`Qualification composed · ${result.qualification.overall_status}`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    } finally {
      updateQualificationComposerState();
    }
  }

  async function inspectArtifact(artifact) {
    state.selectedArtifactId = artifact.id;
    state.selectedArtifactDetails = null;
    renderModels();
    try {
      const details = await api(`/api/workstation/artifacts/${encodeURIComponent(artifact.id)}`);
      if (state.selectedArtifactId !== artifact.id) return;
      state.selectedArtifactDetails = details;
      renderModels();
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  function renderArtifactInspector() {
    if (state.currentView !== "models") return;
    const details = state.selectedArtifactDetails;
    const title = byId("artifactInspectorTitle");
    const path = byId("artifactInspectorPath");
    const openLink = byId("artifactOpenLink");
    const lineage = byId("artifactLineage");
    const metrics = byId("artifactMetrics");
    const select = byId("artifactPromotionEvidence");
    const promote = byId("artifactPromoteButton");
    const pending = state.pendingArtifactPromotions.has(state.selectedArtifactId);
    promote.setAttribute("aria-busy", String(pending));
    setText("artifactPromoteButton", pending ? "Promoting artifact…" : "Promote artifact");
    if (!details || details.artifact?.id !== state.selectedArtifactId) {
      title.textContent = state.selectedArtifactId ? "Loading artifact…" : "Select an artifact";
      path.textContent = "No verified file selected";
      openLink.hidden = true;
      openLink.removeAttribute("href");
      lineage.replaceChildren(create("p", "", "No lineage selected"));
      metrics.replaceChildren();
      select.replaceChildren(new Option("No passed evidence", ""));
      select.dataset.artifactId = "";
      promote.disabled = true;
      return;
    }
    const artifact = details.artifact;
    title.textContent = artifact.name || artifact.id;
    path.textContent = artifact.local_path ? `${artifact.local_path} · ${artifact.sha256}` : "No immutable file registered";
    if (artifact.status === "ready" && artifact.local_path) {
      openLink.href = `/api/workstation/artifacts/${encodeURIComponent(artifact.id)}/content`;
      openLink.hidden = false;
    } else {
      openLink.hidden = true;
      openLink.removeAttribute("href");
    }
    lineage.replaceChildren(...(details.lineage?.nodes || []).map(node => {
      const item = create("div", `artifact-lineage-node ${node.relation}`);
      item.append(create("span", "", node.type), create("strong", "", node.name || node.id), create("small", "", node.relation));
      return item;
    }));
    const metadataBlock = create("section", "artifact-metric-block");
    metadataBlock.append(create("span", "", "Artifact metadata"));
    const metadata = create("pre");
    metadata.textContent = JSON.stringify(artifact.metadata || {}, null, 2);
    metadataBlock.append(metadata);
    const qualificationBlocks = (details.qualifications || []).map(record => {
      const block = create("section", "artifact-metric-block");
      block.append(create("span", "", `${record.overall_status} · ${record.id}`));
      const evidence = create("pre");
      evidence.textContent = JSON.stringify(Object.fromEntries(record.gates.map(gate => [gate.id, gate.metrics])), null, 2);
      block.append(evidence);
      return block;
    });
    metrics.replaceChildren(metadataBlock, ...qualificationBlocks);
    const passed = (details.qualifications || []).filter(record => record.overall_status === "passed");
    const previousEvidence = select.dataset.artifactId === artifact.id ? select.value : "";
    select.dataset.artifactId = artifact.id;
    select.replaceChildren(new Option("No passed evidence", ""), ...passed.map(record => new Option(`${record.id} · ${record.evaluation_artifact_id}`, record.id)));
    if (passed.some(record => record.id === previousEvidence)) select.value = previousEvidence;
    if (artifact.promotion?.qualification_id) select.value = artifact.promotion.qualification_id;
    const eligible = promotableArtifactTypes.has(artifact.type) && artifact.status === "ready" && !artifact.promoted;
    promote.disabled = pending || !eligible || !select.value;
    setText("artifactPromotionHint", pending ? "Promotion is in progress. Verification and copying may take some time." : artifact.promoted ? `Promoted with ${artifact.promotion.qualification_id}` : eligible ? "Select passed evidence to promote this artifact." : "This artifact is not eligible for promotion.");
  }

  async function promoteSelectedArtifact() {
    const details = state.selectedArtifactDetails;
    const qualificationId = byId("artifactPromotionEvidence").value;
    const artifactId = details?.artifact?.id;
    if (!artifactId || !qualificationId || state.pendingArtifactPromotions.has(artifactId)) return;
    state.pendingArtifactPromotions.add(artifactId);
    renderArtifactInspector();
    try {
      await api(`/api/workstation/artifacts/${encodeURIComponent(details.artifact.id)}/promote`, {
        method: "POST",
        body: JSON.stringify({ qualification_id: qualificationId })
      });
      await loadArtifacts();
      const artifact = state.artifacts.find(item => item.id === details.artifact.id);
      if (artifact && state.selectedArtifactId === artifactId) await inspectArtifact(artifact);
      showToast("Artifact promoted with verified evaluation evidence");
    } catch (error) {
      showToast(error.message, { persistent: true });
    } finally {
      state.pendingArtifactPromotions.delete(artifactId);
      renderArtifactInspector();
    }
  }

  function renderExpressions() {
    setText("packageExpressionCount", `${state.expressions.length} ${state.expressions.length === 1 ? "profile" : "profiles"}`);
    setText("localExpressionCount", `${state.expressionDrafts.length} ${state.expressionDrafts.length === 1 ? "draft" : "drafts"}`);
    renderPackageExpressions();
    renderLocalExpressionDrafts();
    restoreExpressionSelection();
  }

  function renderPackageExpressions() {
    const tbody = byId("expressionRows");
    if (!state.expressions.length) {
      emptyTable(tbody, 4, "No package expression profiles");
      return;
    }
    progressiveTables.render(tbody, state.expressions, profile => {
      const row = create("tr");
      row.dataset.selectable = "true";
      row.dataset.expressionSource = "package";
      row.dataset.expressionId = profile.id || "";
      row.setAttribute("aria-selected", "false");
      const languages = Array.isArray(profile.languages) ? profile.languages.join(", ") : "Package default";
      const levels = Array.isArray(profile.intensity_levels) ? profile.intensity_levels.join(", ") : "Default";
      const nameCell = create("td");
      const selectButton = create("button", "row-select", profile.id || "Unnamed");
      selectButton.type = "button";
      selectButton.setAttribute("aria-label", `Inspect ${profile.id || "expression"}`);
      nameCell.append(selectButton);
      row.append(
        nameCell,
        create("td", "", languages),
        create("td", "", levels),
        create("td", "", profile.policy || "Package policy")
      );
      row.addEventListener("click", () => inspectPackageExpression(profile, row));
      return row;
    });
    labelTableCells(tbody.closest("table"));
  }

  function renderLocalExpressionDrafts() {
    const tbody = byId("expressionDraftRows");
    if (!state.expressionDrafts.length) {
      emptyTable(tbody, 4, "No local expression drafts");
      return;
    }
    progressiveTables.render(tbody, state.expressionDrafts, record => {
      const row = create("tr");
      row.dataset.selectable = "true";
      row.dataset.expressionSource = "local";
      row.dataset.expressionId = record.id || "";
      row.setAttribute("aria-selected", "false");
      const nameCell = create("td");
      const selectButton = create("button", "row-select", record.name || record.profile_id || "Unnamed");
      selectButton.type = "button";
      selectButton.setAttribute("aria-label", `Edit ${record.name || "local expression"}`);
      nameCell.append(selectButton);
      const statusCell = create("td");
      statusCell.append(statusLabel(record.qualification_status));
      row.append(
        nameCell,
        create("td", "", record.emotion || "—"),
        create("td", "", record.language || "—"),
        statusCell
      );
      row.addEventListener("click", () => inspectLocalExpression(record, row));
      return row;
    });
    labelTableCells(tbody.closest("table"));
  }

  function selectExpressionRow(row) {
    document.querySelectorAll("#expressionRows tr, #expressionDraftRows tr").forEach(item => {
      item.classList.toggle("selected", item === row);
      item.setAttribute("aria-selected", item === row ? "true" : "false");
    });
  }

  function restoreExpressionSelection() {
    if (!state.selectedExpression) return;
    const selector = state.selectedExpression.source === "local" ? "#expressionDraftRows" : "#expressionRows";
    const records = state.selectedExpression.source === "local" ? state.expressionDrafts : state.expressions;
    const record = records.find(item => item.id === state.selectedExpression.id);
    const row = [...document.querySelectorAll(`${selector} tr`)].find(item => item.dataset.expressionId === state.selectedExpression.id);
    const form = byId("expressionDraftForm");
    if (state.selectedExpression.source === "local" && !form.hidden &&
        (form.dataset.mode === "create" || (form.dataset.dirty === "true" && byId("expressionDraftId").value === state.selectedExpression.id))) {
      if (row) selectExpressionRow(row);
      return;
    }
    if (!record) {
      resetExpressionInspector();
      return;
    }
    if (state.selectedExpression.source === "local") inspectLocalExpression(record, row);
    else inspectPackageExpression(record, row);
  }

  function resetExpressionInspector() {
    state.selectedExpression = null;
    selectExpressionRow(null);
    setText("expressionInspectorEyebrow", "SELECTION");
    setText("expressionInspectorTitle", "No expression selected");
    byId("expressionInspectorEmpty").hidden = false;
    byId("expressionInspector").hidden = true;
    byId("expressionDraftForm").hidden = true;
    renderExpressionReferenceAnalysis(null);
  }

  function inspectPackageExpression(profile, row) {
    const languages = Array.isArray(profile.languages) ? profile.languages.join(", ") : "Package default";
    const levels = Array.isArray(profile.intensity_levels) ? profile.intensity_levels.join(", ") : "Default";
    state.selectedExpression = { source: "package", id: profile.id || "" };
    selectExpressionRow(row);
    setText("expressionInspectorEyebrow", "MODEL PACKAGE");
    setText("expressionInspectorTitle", profile.id || "Expression profile");
    byId("expressionInspectorEmpty").hidden = true;
    byId("expressionDraftForm").hidden = true;
    const inspector = byId("expressionInspector");
    inspector.hidden = false;
    inspector.replaceChildren(
      definition("Source", "AnifLive-TTS Studio model package"),
      definition("Languages", languages),
      definition("Intensity", levels),
      definition("Policy", profile.policy || "Package policy"),
      definition("Runtime", "Symbolic profile")
    );
  }

  function expressionReferenceLabel(reference) {
    if (!reference || typeof reference !== "object") return "";
    const scope = reference.scope === "artifact" ? "Artifact" : "Import root";
    return `${scope} · ${reference.path || "reference audio"}`;
  }

  function expressionDefaultLanguage() {
    const language = String(navigator.language || "en").toLowerCase().split("-")[0];
    return /^[a-z]{2,3}$/.test(language) ? language : "en";
  }

  function setExpressionIdentityEditable(editable) {
    ["expressionProfileId", "expressionModelId", "expressionReferencePath"].forEach(id => {
      byId(id).disabled = !editable;
    });
  }

  function resetExpressionDeleteButton() {
    const button = byId("expressionDeleteButton");
    button.dataset.confirmDelete = "false";
    button.textContent = "Delete";
  }

  function inspectLocalExpression(record, row) {
    state.selectedExpression = { source: "local", id: record.id };
    selectExpressionRow(row);
    setText("expressionInspectorEyebrow", "LOCAL EXPRESSION");
    setText("expressionInspectorTitle", record.name || record.profile_id || "Expression draft");
    byId("expressionInspectorEmpty").hidden = true;
    byId("expressionInspector").hidden = true;
    const form = byId("expressionDraftForm");
    form.hidden = false;
    form.dataset.mode = "edit";
    form.dataset.dirty = "false";
    byId("expressionDraftId").value = record.id || "";
    byId("expressionName").value = record.name || "";
    byId("expressionProfileId").value = record.profile_id || "";
    byId("expressionModelId").value = record.model_id || "";
    byId("expressionReferencePath").value = expressionReferenceLabel(record.reference);
    byId("expressionReferenceTranscript").value = record.prosody?.reference_transcript || "";
    byId("expressionLanguage").value = record.language || expressionDefaultLanguage();
    byId("expressionEmotion").value = record.emotion || "";
    byId("expressionIntensity").value = String(record.intensity ?? 0.7);
    setText("expressionIntensityValue", Number(record.intensity ?? 0.7).toFixed(2));
    setText("expressionQualification", record.qualification_status === "qualified" ? "Qualified · evidence verified" : `${record.qualification_status || "draft"} · evidence required`);
    byId("expressionDescriptions").value = Array.isArray(record.descriptions) ? record.descriptions.join("\n") : "";
    byId("expressionValence").value = record.vad?.valence ?? "";
    byId("expressionArousal").value = record.vad?.arousal ?? "";
    byId("expressionDominance").value = record.vad?.dominance ?? "";
    byId("expressionProsody").value = JSON.stringify(editableExpressionProsody(record.prosody), null, 2);
    byId("expressionFormError").textContent = "";
    byId("expressionDeleteButton").hidden = false;
    resetExpressionDeleteButton();
    setExpressionIdentityEditable(false);
    renderExpressionQualification(record);
    renderExpressionReferenceAnalysis(record);
  }

  function openNewExpression() {
    state.selectedExpression = { source: "local", id: null };
    selectExpressionRow(null);
    setText("expressionInspectorEyebrow", "LOCAL DRAFT");
    setText("expressionInspectorTitle", "New expression");
    byId("expressionInspectorEmpty").hidden = true;
    byId("expressionInspector").hidden = true;
    const form = byId("expressionDraftForm");
    form.hidden = false;
    form.dataset.mode = "create";
    form.dataset.dirty = "false";
    form.reset();
    byId("expressionDraftId").value = "";
    byId("expressionModelId").value = state.status?.health?.model || "";
    byId("expressionLanguage").value = expressionDefaultLanguage();
    byId("expressionIntensity").value = "0.7";
    setText("expressionIntensityValue", "0.70");
    setText("expressionQualification", "Draft · evidence required");
    byId("expressionProsody").value = "{}";
    byId("expressionReferenceTranscript").value = "";
    byId("expressionFormError").textContent = "";
    byId("expressionDeleteButton").hidden = true;
    resetExpressionDeleteButton();
    setExpressionIdentityEditable(true);
    renderExpressionQualification(null);
    renderExpressionReferenceAnalysis(null);
    animateSurface(form);
    requestAnimationFrame(() => {
      byId("expressionName").focus();
      if (window.innerWidth <= 860) {
        byId("expressionDraftForm").scrollIntoView({
          block: "start",
          behavior: window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth"
        });
      }
    });
  }

  function expressionOptionalNumber(id) {
    const text = byId(id).value.trim();
    if (!text) return null;
    const value = Number(text);
    if (!Number.isFinite(value)) throw new Error(`${id} must be a number`);
    return value;
  }

  function editableExpressionProsody(value) {
    if (!value || typeof value !== "object" || Array.isArray(value)) return {};
    const editable = { ...value };
    delete editable.reference_analysis;
    delete editable.reference_transcript;
    return editable;
  }

  function expressionFormPayload({ includeIdentity }) {
    let prosody;
    try {
      prosody = JSON.parse(byId("expressionProsody").value.trim() || "{}");
    } catch (error) {
      throw new Error("Prosody metadata must be valid JSON");
    }
    if (!prosody || typeof prosody !== "object" || Array.isArray(prosody)) {
      throw new Error("Prosody metadata must be a JSON object");
    }
    if ("reference_analysis" in prosody) {
      throw new Error("Reference analysis is measured by AnifLive-TTS Studio and cannot be edited");
    }
    const referenceTranscript = byId("expressionReferenceTranscript").value.trim();
    if (referenceTranscript) prosody.reference_transcript = referenceTranscript;
    const vad = {};
    [["valence", "expressionValence"], ["arousal", "expressionArousal"], ["dominance", "expressionDominance"]].forEach(([key, id]) => {
      const value = expressionOptionalNumber(id);
      if (value !== null) vad[key] = value;
    });
    const payload = {
      name: byId("expressionName").value,
      language: byId("expressionLanguage").value,
      emotion: byId("expressionEmotion").value,
      intensity: Number(byId("expressionIntensity").value),
      descriptions: byId("expressionDescriptions").value.split(/\r?\n/).map(value => value.trim()).filter(Boolean),
      vad,
      prosody
    };
    if (includeIdentity) {
      payload.profile_id = byId("expressionProfileId").value;
      const modelId = byId("expressionModelId").value.trim();
      const referencePath = byId("expressionReferencePath").value.trim();
      if (modelId) payload.model_id = modelId;
      if (referencePath) payload.reference_path = referencePath;
    }
    return payload;
  }

  function formatAnalysisSeconds(value) {
    return Number.isFinite(Number(value)) ? `${Number(value).toFixed(3)} s` : "—";
  }

  function drawExpressionWaveform(analysis) {
    const canvas = byId("expressionWaveform");
    const context = canvas.getContext("2d");
    if (!context) return;
    const rect = canvas.getBoundingClientRect();
    const scale = Math.max(1, Math.min(2, window.devicePixelRatio || 1));
    const width = Math.max(320, Math.round((rect.width || 640) * scale));
    const height = Math.max(96, Math.round((rect.height || 128) * scale));
    if (canvas.width !== width || canvas.height !== height) {
      canvas.width = width;
      canvas.height = height;
    }
    context.clearRect(0, 0, width, height);
    const bins = analysis?.waveform?.bins;
    if (!Array.isArray(bins) || !bins.length) {
      context.strokeStyle = "rgba(255, 255, 255, .12)";
      context.beginPath();
      context.moveTo(0, height / 2);
      context.lineTo(width, height / 2);
      context.stroke();
      return;
    }
    const color = getComputedStyle(document.documentElement).getPropertyValue("--gold").trim() || "#dabf72";
    context.strokeStyle = color;
    context.lineWidth = Math.max(1, scale);
    context.beginPath();
    bins.forEach((bounds, index) => {
      if (!Array.isArray(bounds) || bounds.length !== 2) return;
      const x = (index + 0.5) * width / bins.length;
      const top = height * (0.5 - Math.max(-1, Math.min(1, Number(bounds[1]))) * 0.43);
      const bottom = height * (0.5 - Math.max(-1, Math.min(1, Number(bounds[0]))) * 0.43);
      context.moveTo(x, top);
      context.lineTo(x, bottom);
    });
    context.stroke();
  }

  function renderExpressionReferenceAnalysis(record) {
    const analysis = record?.prosody?.reference_analysis;
    const measurements = analysis?.measurements || {};
    const pitch = measurements.pitch || {};
    const rate = measurements.speaking_rate;
    const button = byId("expressionAnalyzeButton");
    setText("expressionAnalyzeLabel", "Analyze");
    button.disabled = !record?.id || !record?.reference;
    button.title = button.disabled ? "Save a draft with reference audio first" : "Measure reference audio";
    setText("expressionAnalysisDuration", formatAnalysisSeconds(measurements.duration_seconds));
    setText("expressionAnalysisOnset", formatAnalysisSeconds(measurements.onset_seconds));
    setText("expressionAnalysisRms", Number.isFinite(Number(measurements.rms_dbfs)) ? `${Number(measurements.rms_dbfs).toFixed(1)} dBFS` : "—");
    setText("expressionAnalysisPitch", pitch.reliable && Number.isFinite(Number(pitch.median_hz)) ? `${Number(pitch.median_hz).toFixed(1)} Hz` : "Unavailable");
    setText("expressionAnalysisRate", rate && Number.isFinite(Number(rate.value)) ? `${Number(rate.value).toFixed(2)} ${rate.unit === "words-per-minute" ? "wpm" : "chars/s"}` : "—");
    setText("expressionAnalysisSha", typeof analysis?.reference_sha256 === "string" ? analysis.reference_sha256.slice(0, 12) : "—");
    setText("expressionAnalysisStatus", analysis?.provenance?.analyzed_at ? `Measured ${formatDate(analysis.provenance.analyzed_at)} · non-neural` : "No measured analysis");
    requestAnimationFrame(() => drawExpressionWaveform(analysis));
  }

  async function analyzeSelectedExpressionReference() {
    const button = byId("expressionAnalyzeButton");
    const expressionId = byId("expressionDraftId").value;
    if (!expressionId || button.disabled) return;
    button.disabled = true;
    setText("expressionAnalyzeLabel", "Analyzing");
    byId("expressionFormError").textContent = "";
    try {
      await api(`/api/workstation/expression-drafts/${encodeURIComponent(expressionId)}`, {
        method: "PATCH",
        body: JSON.stringify(expressionFormPayload({ includeIdentity: false }))
      });
      const payload = await api(`/api/workstation/expression-drafts/${encodeURIComponent(expressionId)}/analyze-reference`, {
        method: "POST",
        body: JSON.stringify({})
      });
      state.selectedExpression = { source: "local", id: payload.expression.id };
      await loadExpressions();
      showToast("Reference signal measured");
    } catch (error) {
      byId("expressionFormError").textContent = error.message;
      button.disabled = false;
      setText("expressionAnalyzeLabel", "Analyze");
    }
  }

  async function submitExpressionDraft(event) {
    event.preventDefault();
    const form = byId("expressionDraftForm");
    if (form.dataset.saving === "true") return;
    form.dataset.saving = "true";
    const revision = form.dataset.editRevision;
    const submit = form.querySelector('button[type="submit"]');
    if (submit) submit.disabled = true;
    const creating = byId("expressionDraftForm").dataset.mode === "create";
    const expressionId = byId("expressionDraftId").value;
    try {
      const record = await api(
        creating ? "/api/workstation/expression-drafts" : `/api/workstation/expression-drafts/${encodeURIComponent(expressionId)}`,
        {
          method: creating ? "POST" : "PATCH",
          body: JSON.stringify(expressionFormPayload({ includeIdentity: creating }))
        }
      );
      state.selectedExpression = { source: "local", id: record.id };
      form.dataset.mode = "edit";
      byId("expressionDraftId").value = record.id;
      setExpressionIdentityEditable(false);
      if (form.dataset.editRevision === revision) form.dataset.dirty = "false";
      await loadExpressions();
      showToast(creating ? "Expression draft created" : "Expression draft updated");
    } catch (error) {
      byId("expressionFormError").textContent = error.message;
    } finally {
      form.dataset.saving = "false";
      if (submit) submit.disabled = false;
    }
  }

  function renderExpressionQualification(record) {
    const select = byId("expressionQualificationEvidence");
    const button = byId("expressionPromoteButton");
    const hint = byId("expressionQualificationHint");
    if (!record?.id) {
      select.replaceChildren(new Option("Save the draft before qualification", ""));
      select.disabled = true;
      button.disabled = true;
      hint.textContent = "Qualified status can only be granted by a passed evaluation artifact.";
      return;
    }
    const candidates = state.qualifications.filter(qualification => (
      qualification.subject_kind === "expression"
      && qualification.subject_id === record.id
      && qualification.overall_status === "passed"
    ));
    select.replaceChildren(new Option("No passed evidence", ""), ...candidates.map(qualification => new Option(`${qualification.id} · ${qualification.evaluation_artifact_id}`, qualification.id)));
    if (record.qualification?.qualification_id) select.value = record.qualification.qualification_id;
    select.disabled = record.qualification_status === "qualified";
    button.disabled = record.qualification_status === "qualified" || !select.value;
    hint.textContent = record.qualification_status === "qualified"
      ? `Promoted with ${record.qualification.qualification_id}`
      : candidates.length
        ? "Select a passed evaluation run to promote this expression."
        : "No passed evaluation evidence is registered for this expression.";
  }

  async function promoteSelectedExpression() {
    const expressionId = byId("expressionDraftId").value;
    const qualificationId = byId("expressionQualificationEvidence").value;
    if (!expressionId || !qualificationId) return;
    try {
      const record = await api(`/api/workstation/expression-drafts/${encodeURIComponent(expressionId)}/promote`, {
        method: "POST",
        body: JSON.stringify({ qualification_id: qualificationId })
      });
      state.selectedExpression = { source: "local", id: record.id };
      await loadExpressions();
      showToast("Expression promoted with verified evaluation evidence");
    } catch (error) {
      byId("expressionFormError").textContent = error.message;
    }
  }

  async function deleteExpressionDraft() {
    const button = byId("expressionDeleteButton");
    const expressionId = byId("expressionDraftId").value;
    if (!expressionId) return;
    if (button.dataset.confirmDelete !== "true") {
      button.dataset.confirmDelete = "true";
      button.textContent = "Confirm delete";
      window.setTimeout(() => {
        if (button.dataset.confirmDelete === "true") resetExpressionDeleteButton();
      }, 3500);
      return;
    }
    try {
      await api(`/api/workstation/expression-drafts/${encodeURIComponent(expressionId)}`, { method: "DELETE" });
      state.selectedExpression = null;
      await loadExpressions();
      resetExpressionInspector();
      showToast("Expression draft deleted");
    } catch (error) {
      byId("expressionFormError").textContent = error.message;
      resetExpressionDeleteButton();
    }
  }

  function definition(term, value) {
    const row = create("div");
    row.append(create("dt", "", term), create("dd", "", value));
    return row;
  }

  function sendSynthesisSettings() {
    const frame = byId("synthesisFrame");
    if (!(frame instanceof HTMLIFrameElement) || !frame.contentWindow) return;
    frame.contentWindow.postMessage({
      type: "aniflive-tts:settings",
      locale: state.locale,
      default_language: state.settings.default_language,
      continuity_policy: state.settings.default_continuity_policy
    }, location.origin);
  }

  function renderSettings() {
    if (experience.settingsDirty()) return;
    byId("settingsDefaultLanguage").value = state.settings.default_language;
    byId("settingsContinuityPolicy").value = state.settings.default_continuity_policy;
    byId("settingsTrainingPreset").value = state.settings.default_training_preset;
    byId("settingsBenchmarkLanguage").value = state.settings.default_benchmark_language;
    byId("settingsTseThreshold").value = String(state.settings.default_tse_target_threshold);
    byId("settingsRefreshSeconds").value = String(state.settings.auto_refresh_seconds);
    byId("settingsOverviewMotion").checked = state.settings.overview_motion;
    setText(
      "settingsHistoryState",
      state.visibleHistoryClearedAt ? "Workspace view is clean" : "All visible records are shown"
    );
    window.AnifLiveTTSStyledSelect?.refreshAll();
    syncAmbientVideo();
    sendSynthesisSettings();
  }

  function renderComponents() {
    const rows = byId("componentRows");
    const components = Array.isArray(state.components) ? state.components : [];
    if (!components.length) {
      rows.replaceChildren(create("p", "", "No component contracts are available"));
      setText("componentReadiness", "Unavailable");
      const download = byId("componentDownloadButton");
      download.disabled = true;
      download.querySelector("span").textContent = "AI Components unavailable";
      return;
    }
    rows.replaceChildren(...components.map(component => {
      const row = create("article", `component-row ${component.state || "missing"}`);
      const header = create("header");
      header.append(createData("strong", "", component.name || component.id), create("span", "", component.ready ? "Ready" : component.state || "Missing"));
      row.append(header, create("small", "", `${component.capability || "component"} · ${component.runtime || "worker"} · ${component.revision || "unqualified"}`));
      if (!component.ready && component.online_installable) {
        const install = create("button", "row-action component-install-button", "Install component");
        install.type = "button";
        install.disabled = state.componentDownloadActive;
        install.addEventListener("click", () => downloadComponents([component.id]));
        row.append(install);
      }
      row.title = component.reason || component.license || "";
      return row;
    }));
    const required = components.filter(component => component.required);
    const ready = required.filter(component => component.ready).length;
    setText("componentReadiness", `${ready} / ${required.length} required ready`);
    const download = byId("componentDownloadButton");
    const installable = components.filter(component => (
      component.required && !component.ready && component.online_installable
    ));
    download.disabled = state.componentDownloadActive || installable.length === 0;
    download.querySelector("span").textContent = state.componentDownloadActive
      ? "Installing verified components…"
      : installable.length
        ? `Install ${installable.length} AI Component${installable.length === 1 ? "" : "s"}`
        : "AI Components ready";
    replaceIcons(rows);
  }

  async function loadComponents() {
    const payload = await api("/api/workstation/components");
    state.components = Array.isArray(payload.components) ? payload.components : [];
    renderComponents();
  }

  async function downloadRequiredComponents() {
    if (state.componentDownloadActive) return;
    const componentIds = state.components
      .filter(component => component.required && !component.ready && component.online_installable)
      .map(component => component.id);
    if (!componentIds.length) return showToast("All downloadable AI Components are ready");
    await downloadComponents(componentIds);
  }

  async function downloadComponents(componentIds) {
    if (state.componentDownloadActive || !Array.isArray(componentIds) || !componentIds.length) return;
    state.componentDownloadActive = true;
    renderComponents();
    setText("componentStatus", "Downloading immutable revisions and verifying every SHA-256…");
    try {
      const result = await api("/api/workstation/components/download", {
        method: "POST",
        body: JSON.stringify({ component_ids: componentIds })
      });
      await loadComponents();
      showToast(`${result.installed?.length || 0} AI Components installed and verified`);
      setText("componentStatus", "Installed components are now available to offline Linux workers.");
    } catch (error) {
      showToast(error.message, { persistent: true });
      setText("componentStatus", error.message);
    } finally {
      state.componentDownloadActive = false;
      renderComponents();
    }
  }

  async function importComponentBundle() {
    const path = byId("componentBundlePath").value.trim();
    if (!path) return showToast("Choose a local component bundle path");
    try {
      const result = await api("/api/workstation/components/import", {
        method: "POST",
        body: JSON.stringify({ path })
      });
      await loadComponents();
      showToast(`${result.installed?.length || 0} AI components imported and verified`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function exportComponentBundle() {
    const path = byId("componentBundlePath").value.trim();
    if (!path) return showToast("Choose a local .zip destination");
    const componentIds = state.components.filter(component => component.ready).map(component => component.id);
    if (!componentIds.length) return showToast("No verified AI components are ready to export");
    try {
      const result = await api("/api/workstation/components/export", {
        method: "POST",
        body: JSON.stringify({ path, component_ids: componentIds })
      });
      showToast(`Offline component bundle written to ${result.path}`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function backgroundRefresh() {
    if (document.hidden) return;
    try {
      await loadOverview();
      if (state.currentView === "training") await loadTrainingRunDetails();
      if (state.currentView === "evaluation") {
        await loadArtifacts();
        await loadEvaluationRunDetails();
      }
    } catch (_) {}
  }

  function scheduleBackgroundRefresh() {
    if (refreshTimer) window.clearInterval(refreshTimer);
    refreshTimer = window.setInterval(
      backgroundRefresh,
      Math.max(2, Number(state.settings.auto_refresh_seconds) || 5) * 1000
    );
  }

  function applySettings(settings) {
    state.settings = { ...state.settings, ...(settings || {}) };
    renderSettings();
    scheduleBackgroundRefresh();
  }

  async function loadSettings() {
    const payload = await api("/api/workstation/settings");
    state.visibleHistoryClearedAt = payload.visible_history_cleared_at || null;
    applySettings(payload.data);
  }

  function resetVisibleWorkspaceState() {
    state.projects = [];
    state.jobs = [];
    state.artifacts = [];
    state.qualifications = [];
    state.expressionDrafts = [];
    state.selectedExpression = null;
    state.selectedJobId = null;
    state.selectedArtifactId = null;
    state.selectedArtifactDetails = null;
    state.selectedQualificationId = null;
    state.selectedDatasetId = null;
    state.selectedDatasetItemId = null;
    state.selectedTseId = null;
    state.selectedTseJobId = null;
    state.selectedTrainingId = null;
    state.selectedTrainingJobId = null;
    state.selectedEvaluationId = null;
    state.selectedEvaluationJobId = null;
  }

  async function clearVisibleHistory(event) {
    event.preventDefault();
    const submit = byId("confirmClearVisibleHistory");
    submit.disabled = true;
    try {
      const result = await api("/api/workstation/ui-history/clear", {
        method: "POST",
        body: JSON.stringify({})
      });
      state.visibleHistoryClearedAt = result.cleared_at || null;
      resetVisibleWorkspaceState();
      closeAnimatedDialog(byId("clearHistoryDialog"));
      await Promise.all([loadOverview(), loadArtifacts(), loadExpressions()]);
      renderSettings();
      showToast("Interface history cleared; files and assets were retained");
    } catch (error) {
      showToast(error.message, { persistent: true });
    } finally {
      submit.disabled = false;
    }
  }

  async function saveSettings(event) {
    event.preventDefault();
    if (!experience.beginSettingsSave()) return;
    const submit = event.currentTarget.querySelector('button[type="submit"]');
    submit.disabled = true;
    byId("settingsFormError").textContent = "";
    setText("settingsSaveState", "Saving");
    try {
      const payload = await api("/api/workstation/settings", {
        method: "PATCH",
        body: JSON.stringify({
          default_language: byId("settingsDefaultLanguage").value,
          default_continuity_policy: byId("settingsContinuityPolicy").value,
          default_training_preset: byId("settingsTrainingPreset").value,
          default_benchmark_language: byId("settingsBenchmarkLanguage").value,
          default_tse_target_threshold: Number(byId("settingsTseThreshold").value),
          auto_refresh_seconds: Number(byId("settingsRefreshSeconds").value),
          overview_motion: byId("settingsOverviewMotion").checked
        })
      });
      experience.settingsSaved();
      applySettings(payload.data);
      setText("settingsSaveState", experience.settingsDirty() ? "Unsaved changes" : "Saved locally");
      showToast("Workstation settings saved");
    } catch (error) {
      setText("settingsSaveState", "Not saved");
      byId("settingsFormError").textContent = error.message;
    } finally {
      experience.endSettingsSave();
      submit.disabled = false;
    }
  }

  async function loadOverview() {
    const readTargets = ["datasetRows", "trainingRows", "evaluationRows", "tseRows"];
    setReadState(readTargets, "loading");
    try {
      const [overview, projects, jobs] = await Promise.all([
        api("/api/workstation/overview"),
        api("/api/workstation/projects"),
        api("/api/workstation/jobs")
      ]);
      state.overview = overview;
      state.projects = Array.isArray(projects.data) ? projects.data : [];
      state.jobs = Array.isArray(jobs.data) ? jobs.data : [];
      renderOverview();
      renderProjectTables();
      renderJobs();
      setReadState(readTargets, "ready");
    } catch (error) {
      setReadState(readTargets, "error", error);
      throw error;
    }
  }

  async function loadRuntime() {
    try {
      const [status, models] = await Promise.all([api("/api/status"), api("/api/workstation/models")]);
      state.status = status;
      state.models = Array.isArray(models.data) ? models.data : [];
      byId("settingsApi").textContent = status.api_version ? `v${status.api_version}` : "Connected";
      byId("settingsBackend").textContent = status.health?.backend || "Connected";
      renderModels();
      renderEngines();
    } catch (error) {
      renderModels();
      renderEngines();
    }
  }

  async function loadArtifacts() {
    const readTargets = ["modelRows", "engineArtifactRows", "evaluationArtifactRows"];
    setReadState(readTargets, "loading");
    try {
      const [payload, qualifications] = await Promise.all([
        api("/api/workstation/artifacts"),
        api("/api/workstation/qualifications")
      ]);
      state.artifacts = Array.isArray(payload.data) ? payload.data : [];
      state.qualifications = Array.isArray(qualifications.data) ? qualifications.data : [];
      renderModels();
      renderEngines();
      renderEvaluationEvidence();
      renderTrainingWorkspace(selectedTrainingProject());
      renderEvaluationWorkspace(selectedEvaluationProject());
      setReadState(readTargets, "ready");
    } catch (error) {
      setReadState(readTargets, "error", error);
      throw error;
    }
  }

  async function loadExpressions() {
    const readTargets = ["expressionRows", "expressionDraftRows"];
    setReadState(readTargets, "loading");
    try {
      const [packagePayload, draftPayload, qualificationPayload] = await Promise.all([
        api("/api/workstation/expression-bank"),
        api("/api/workstation/expression-drafts"),
        api("/api/workstation/qualifications")
      ]);
      state.expressions = Array.isArray(packagePayload.data) ? packagePayload.data : [];
      state.expressionDrafts = Array.isArray(draftPayload.data) ? draftPayload.data : [];
      state.qualifications = Array.isArray(qualificationPayload.data) ? qualificationPayload.data : [];
      byId("expressionSource").textContent = `${state.expressions.length} package · ${state.expressionDrafts.length} local`;
      renderExpressions();
      setReadState(readTargets, "ready");
    } catch (error) {
      setReadState(readTargets, "error", error);
      showToast(error.message, { persistent: true });
    }
  }

  async function loadJobs() {
    try {
      const payload = await api("/api/workstation/jobs");
      state.jobs = Array.isArray(payload.data) ? payload.data : [];
      renderJobs();
      renderProjectTables();
      if (state.overview) renderOverview();
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  function renderJobInspectorMeta(job) {
    const metadata = byId("jobInspectorMeta");
    const dependencies = jobControls.dependencyRecords(job, state.jobs);
    const presentation = jobControls.statusPresentation(job);
    const rows = [
      ["State", presentation.label],
      ...(job.wait_reason ? [["Queue reason", job.wait_reason]] : []),
      ["Attempt", String(job.attempt || 1)],
      ["Priority", `${Number(job.priority) > 0 ? "+" : ""}${Number(job.priority) || 0}`],
      ["Retry source", job.retry_of || "Initial attempt"],
      ["Dependencies", dependencies.length
        ? dependencies.map(record => `${record.id} (${record.status})`).join(", ")
        : "None"]
    ];
    metadata.replaceChildren(...rows.map(([term, detail]) => {
      const row = create("div");
      row.append(create("dt", "", term), create("dd", "", detail));
      return row;
    }));
    metadata.hidden = false;
  }

  async function inspectJob(jobId, { scroll = true } = {}) {
    try {
      const payload = await api(`/api/workstation/jobs/${encodeURIComponent(jobId)}`);
      const job = payload.job || {};
      state.selectedJobId = jobId;
      const jobIndex = state.jobs.findIndex(record => record.id === jobId);
      if (jobIndex >= 0) state.jobs[jobIndex] = job;
      setText("jobInspectorTitle", `${job.type || "Job"} · ${jobControls.shortId(job.id)}`);
      renderJobInspectorMeta(job);
      renderJobs();
      const logs = Array.isArray(payload.logs) ? payload.logs : [];
      const list = byId("jobLogList");
      if (!logs.length) {
        list.replaceChildren(create("p", "", "No events recorded"));
      } else {
        list.replaceChildren(...logs.slice(-12).map(log => {
          const row = create("div", "job-log-row");
          row.append(
            create("time", "", formatDate(log.created_at)),
            create("span", `job-log-level ${log.level || "info"}`, log.level || "info"),
            create("p", "", log.message || "")
          );
          return row;
        }));
      }
      if (scroll && window.innerWidth <= 860) {
        const inspector = document.querySelector(".queue-depth");
        const behavior = window.matchMedia("(prefers-reduced-motion: reduce)").matches
          ? "auto"
          : "smooth";
        inspector?.scrollIntoView({ behavior, block: "start" });
      }
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  function renderJobDialogOptions() {
    const jobType = byId("jobType").value;
    const acceptedKinds = jobProjectKinds[jobType] || new Set();
    const projectSelect = byId("jobProject");
    const previousProject = projectSelect.value;
    const projects = state.projects.filter(project => acceptedKinds.has(project.kind));
    const projectOptions = projects.map(project => {
      const option = create("option", "", `${project.name} · ${project.kind}`);
      option.value = project.id;
      return option;
    });
    if (!projectOptions.length) {
      const option = create("option", "", "No compatible project");
      option.value = "";
      option.disabled = true;
      projectOptions.push(option);
    }
    projectSelect.replaceChildren(...projectOptions);
    if (projects.some(project => project.id === previousProject)) {
      projectSelect.value = previousProject;
    }
    projectSelect.disabled = projects.length === 0;
    byId("jobForm").querySelector('button[type="submit"]').disabled = projects.length === 0;

    const dependencyList = byId("jobDependencyList");
    if (!state.jobs.length) {
      dependencyList.replaceChildren(create("p", "", "No eligible dependencies"));
      return;
    }
    dependencyList.replaceChildren(...state.jobs.map(job => {
      const label = create("label", "dependency-option");
      const checkbox = create("input");
      checkbox.type = "checkbox";
      checkbox.value = job.id;
      checkbox.disabled = ["failed", "cancelled"].includes(job.status);
      const copy = create("span");
      copy.append(
        create("strong", "", `${jobControls.shortId(job.id)} · ${job.type}`),
        create("small", "", `${job.status} · attempt ${job.attempt || 1}`)
      );
      label.classList.toggle("disabled", checkbox.disabled);
      label.title = checkbox.disabled
        ? "Failed or cancelled jobs cannot be selected as dependencies"
        : job.id;
      label.append(checkbox, copy);
      return label;
    }));
  }

  function openJobDialog() {
    byId("jobPriority").value = "0";
    byId("jobError").textContent = "";
    byId("jobForm").querySelectorAll("input, select, textarea").forEach(clearFieldValidation);
    renderJobDialogOptions();
    openAnimatedDialog(byId("jobDialog"));
    requestAnimationFrame(() => byId("jobType").focus());
  }

  async function submitJob(event) {
    event.preventDefault();
    const priority = byId("jobPriority").valueAsNumber;
    if (!Number.isInteger(priority) || priority < -100 || priority > 100) {
      byId("jobError").textContent = "Priority must be an integer between -100 and 100";
      return;
    }
    const dependencies = [...byId("jobDependencyList").querySelectorAll('input[type="checkbox"]:checked')]
      .map(input => input.value);
    try {
      const job = await api("/api/workstation/jobs", {
        method: "POST",
        body: JSON.stringify({
          type: byId("jobType").value,
          project_id: byId("jobProject").value,
          parameters: {},
          priority,
          depends_on: dependencies
        })
      });
      closeAnimatedDialog(byId("jobDialog"));
      state.selectedJobId = job.id;
      await loadOverview();
      await inspectJob(job.id, { scroll: false });
      showToast(`Queued ${jobControls.shortId(job.id)}`);
    } catch (error) {
      byId("jobError").textContent = error.message;
    }
  }

  function syncTrainingPresetFields() {
    const advanced = byId("projectPreset").value === "advanced";
    byId("projectAdvancedTrainingFields").hidden = !advanced;
    byId("projectGradientCheckpointing").checked = advanced;
  }

  function projectNumber(id) {
    const value = byId(id).valueAsNumber;
    return Number.isFinite(value) ? value : null;
  }

  function selectedDatasetAcquisitionMode() {
    return document.querySelector('input[name="datasetAcquisitionMode"]:checked')?.value || "standard";
  }

  function animateDatasetAcquisitionFields() {
    const fields = byId("projectDatasetFields");
    if (!fields || window.matchMedia("(prefers-reduced-motion: reduce)").matches) return;
    fields.classList.remove("dataset-mode-changing");
    void fields.offsetWidth;
    fields.classList.add("dataset-mode-changing");
    window.setTimeout(() => fields.classList.remove("dataset-mode-changing"), 280);
  }

  function syncDatasetAcquisitionFields({ animate = false } = {}) {
    const mode = selectedDatasetAcquisitionMode();
    const target = mode === "target-speaker";
    const list = mode === "gpt-sovits-list";
    byId("projectTargetSpeakerFields").hidden = !target;
    byId("projectDatasetAsrRow").hidden = list;
    byId("projectDatasetLanguageRow").hidden = list;
    byId("projectTargetReference").required = target;
    byId("projectDatasetSources").dataset.pathAccept = list ? "list" : "media";
    setText("projectDatasetSourcesLabel", list ? "GPT-SoVITS .list path" : target ? "Source audio / video" : "Source media");
    i18n.setLocalizedAttribute(
      byId("projectDatasetSources"),
      "placeholder",
      list
        ? "One reviewed local .list path"
        : target
          ? "One local long recording or folder per line"
          : "One local audio, video or folder path per line"
    );
    window.AnifLiveTTSStyledSelect?.refreshAll?.();
    if (animate) animateDatasetAcquisitionFields();
  }

  function openProjectDialog(kind = "dataset") {
    const dialog = byId("projectDialog");
    const labels = { dataset: "New dataset", tse: "New TSE project", training: "New experiment", evaluation: "New evaluation" };
    byId("projectKind").value = kind;
    byId("projectDialogTitle").textContent = labels[kind] || "New project";
    const sourceLabels = {
      dataset: "Audio folder, file, or GPT-SoVITS .list",
      tse: "Source recording",
      training: "Dataset path",
      evaluation: "V2ProPlus model package"
    };
    byId("projectDatasetFields").hidden = kind !== "dataset";
    byId("projectSourceRow").hidden = kind === "dataset";
    byId("projectSourceLabel").textContent = sourceLabels[kind] || "Source path";
    byId("projectReferenceRow").hidden = kind !== "tse";
    byId("projectModelPackageRow").hidden = kind !== "tse";
    byId("projectPresetRow").hidden = kind !== "training";
    byId("projectTrainingFields").hidden = kind !== "training";
    byId("projectEvaluationFields").hidden = kind !== "evaluation";
    byId("projectSource").required = ["tse", "training", "evaluation"].includes(kind);
    byId("projectDatasetSources").required = kind === "dataset";
    byId("projectReference").required = kind === "tse";
    byId("projectModelPackage").required = kind === "tse";
    for (const id of ["projectPretrainedGpt", "projectPretrainedSovitsG", "projectPretrainedSovitsD"]) {
      byId(id).required = kind === "training";
    }
    byId("projectSharedDir").required = kind === "evaluation";
    byId("projectAsrModel").required = kind === "evaluation";
    byId("projectName").value = "";
    byId("projectSource").value = "";
    byId("projectDatasetSources").value = "";
    document.querySelector('input[name="datasetAcquisitionMode"][value="standard"]').checked = true;
    byId("projectTargetReference").value = "";
    byId("projectTargetPreset").value = "balanced";
    byId("projectDatasetAsr").value = "sensevoice-small";
    byId("projectDatasetLanguage").value = "auto";
    byId("projectDatasetRequireExpressions").checked = true;
    byId("projectReference").value = "";
    byId("projectModelPackage").value = "";
    byId("projectPretrainedGpt").value = "";
    byId("projectPretrainedSovitsG").value = "";
    byId("projectPretrainedSovitsD").value = "";
    byId("projectResumeCheckpoint").value = "";
    byId("projectTrainingStage").value = "both";
    byId("projectSharedDir").value = "";
    byId("projectAsrModel").value = "";
    byId("projectBaselineReport").value = "";
    byId("projectPreset").value = state.settings.default_training_preset;
    byId("projectBenchmarkLanguage").value = state.settings.default_benchmark_language;
    syncTrainingPresetFields();
    syncDatasetAcquisitionFields();
    dialog.querySelectorAll("[data-path-picker]").forEach(syncPathPickerEmptyState);
    dialog.querySelectorAll("input, select, textarea").forEach(clearFieldValidation);
    byId("projectError").textContent = "";
    openAnimatedDialog(dialog);
    requestAnimationFrame(() => byId("projectName").focus());
  }

  async function submitProject(event) {
    event.preventDefault();
    const kind = byId("projectKind").value;
    const config = {};
    const source = byId("projectSource").value.trim();
    const reference = byId("projectReference").value.trim();
    const modelPackage = byId("projectModelPackage").value.trim();
    if (kind === "dataset") {
      const acquisitionMode = selectedDatasetAcquisitionMode();
      const sources = byId("projectDatasetSources").value
        .split(/\r?\n/)
        .map(value => value.trim())
        .filter(Boolean);
      config.acquisition_mode = acquisitionMode;
      config.sources = sources;
      config.require_expressions = byId("projectDatasetRequireExpressions").checked;
      if (acquisitionMode !== "gpt-sovits-list") {
        config.asr_component = byId("projectDatasetAsr").value;
        const declaredLanguage = byId("projectDatasetLanguage").value;
        if (declaredLanguage !== "auto") config.declared_language = declaredLanguage;
      }
      if (acquisitionMode === "target-speaker") {
        config.reference_audio = byId("projectTargetReference").value.trim();
        const preset = byId("projectTargetPreset").value;
        Object.assign(config, preset === "strict"
          ? { speaker_threshold: 0.8, review_margin: 0.06, ambiguity_margin: 0.035 }
          : preset === "recall"
            ? { speaker_threshold: 0.66, review_margin: 0.1, ambiguity_margin: 0.025 }
            : { speaker_threshold: 0.72, review_margin: 0.08, ambiguity_margin: 0.03 });
      }
    }
    if (source && kind !== "dataset") {
      const sourceKeys = {
        dataset: "source",
        tse: "source",
        training: "dataset",
        evaluation: "model_package"
      };
      config[sourceKeys[kind] || "source"] = source;
    }
    if (reference) config.reference = reference;
    if (kind === "tse" && modelPackage) config.model_package = modelPackage;
    if (kind === "training") {
      config.preset = byId("projectPreset").value;
      config.stage = byId("projectTrainingStage").value;
      config.pretrained_gpt = byId("projectPretrainedGpt").value.trim();
      config.pretrained_sovits_g = byId("projectPretrainedSovitsG").value.trim();
      config.pretrained_sovits_d = byId("projectPretrainedSovitsD").value.trim();
      const resume = byId("projectResumeCheckpoint").value.trim();
      if (resume) config.resume_checkpoint = resume;
      if (config.preset === "advanced") {
        Object.assign(config, {
          gpt_epochs: projectNumber("projectGptEpochs"),
          sovits_epochs: projectNumber("projectSovitsEpochs"),
          gpt_batch_size: projectNumber("projectGptBatch"),
          sovits_batch_size: projectNumber("projectSovitsBatch"),
          gpt_learning_rate: projectNumber("projectGptLearningRate"),
          sovits_learning_rate: projectNumber("projectSovitsLearningRate"),
          save_every_epoch: projectNumber("projectSaveEveryEpoch"),
          seed: projectNumber("projectTrainingSeed"),
          gradient_checkpointing: byId("projectGradientCheckpointing").checked
        });
      }
    }
    if (kind === "evaluation") {
      config.shared_dir = byId("projectSharedDir").value.trim();
      config.asr_model = byId("projectAsrModel").value.trim();
      const baseline = byId("projectBaselineReport").value.trim();
      if (baseline) config.baseline_report = baseline;
      Object.assign(config, {
        benchmark_sessions: projectNumber("projectBenchmarkSessions"),
        benchmark_warmups: projectNumber("projectBenchmarkWarmups"),
        benchmark_runs: projectNumber("projectBenchmarkRuns"),
        benchmark_language: byId("projectBenchmarkLanguage").value,
        asr_compute_type: byId("projectAsrComputeType").value,
        request_timeout_seconds: projectNumber("projectRequestTimeout")
      });
    }
    try {
      const project = await api("/api/workstation/projects", {
        method: "POST",
        body: JSON.stringify({ kind, name: byId("projectName").value, config })
      });
      closeAnimatedDialog(byId("projectDialog"));
      if (kind === "dataset") state.selectedDatasetId = project.id;
      if (kind === "tse") state.selectedTseId = project.id;
      if (kind === "training") state.selectedTrainingId = project.id;
      if (kind === "evaluation") state.selectedEvaluationId = project.id;
      await loadOverview();
      if (kind === "dataset") await selectDataset(project);
      if (kind === "tse") await selectTseProject(project);
      if (kind === "training") await selectTrainingProject(project);
      if (kind === "evaluation") await selectEvaluationProject(project);
      showToast("Project created");
    } catch (error) {
      byId("projectError").textContent = error.message;
    }
  }

  async function queueInventory(project) {
    try {
      const job = await api("/api/workstation/jobs", {
        method: "POST",
        body: JSON.stringify({ type: "dataset.inventory", project_id: project.id, parameters: {} })
      });
      await loadJobs();
      openView("jobs");
      showToast(`Queued ${job.id}`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function queueDatasetProcess() {
    const project = selectedDatasetProject();
    const sources = datasetProjectSources(project);
    if (!project || !sources.length) {
      showToast("Dataset preparation requires configured media or folder sources");
      return;
    }
    try {
      const result = await api(
        `/api/workstation/datasets/${encodeURIComponent(project.id)}/prepare-standard`,
        {
        method: "POST",
          body: JSON.stringify({})
        }
      );
      await loadOverview();
      await loadSelectedDataset();
      showToast(`${Number(result.count) || 0} source preparation job${Number(result.count) === 1 ? "" : "s"} queued in Linux Docker`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function importCompletedDatasetJobs(datasetId, jobs) {
    let imported = 0;
    for (const job of jobs) {
      const result = await api(
        `/api/workstation/datasets/${encodeURIComponent(datasetId)}/import-worker-job/${encodeURIComponent(job.id)}`,
        { method: "POST", body: JSON.stringify({}) }
      );
      imported += Number(result.import?.created_count ?? result.count) || 0;
      state.autoImportedDatasetProcessJobs.add(job.id);
      state.datasetProcessImportFailures.delete(job.id);
    }
    return imported;
  }

  async function autoImportCompletedDatasetJobs(datasetId) {
    const completed = state.jobs.filter(job => (
      job.project_id === datasetId
      && job.type === "dataset.process"
      && job.status === "succeeded"
      && !state.autoImportedDatasetProcessJobs.has(job.id)
    ));
    if (!completed.length) return;
    try {
      await importCompletedDatasetJobs(datasetId, completed);
    } catch (error) {
      completed.forEach(job => {
        state.autoImportedDatasetProcessJobs.delete(job.id);
        state.datasetProcessImportFailures.add(job.id);
      });
      throw error;
    }
  }

  async function queueTargetSpeakerPreparation() {
    const project = selectedDatasetProject();
    if (!project || datasetAcquisitionMode(project) !== "target-speaker") return;
    const button = byId("datasetPrepareTargetButton");
    button.disabled = true;
    try {
      const result = await api(
        `/api/workstation/datasets/${encodeURIComponent(project.id)}/prepare-target-speaker`,
        { method: "POST", body: JSON.stringify({}) }
      );
      await loadOverview();
      await loadSelectedDataset();
      showToast(`${Number(result.count) || 0} target-speaker acquisition chain${Number(result.count) === 1 ? "" : "s"} queued`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    } finally {
      renderDatasetWorkspace(project);
    }
  }

  async function importTargetSpeakerAcquisition() {
    const project = selectedDatasetProject();
    if (!project) return;
    const jobs = state.jobs.filter(job => (
      job.project_id === project.id
      && job.type === "dataset.finalize"
      && job.status === "succeeded"
    ));
    if (!jobs.length) return;
    state.datasetLoading = true;
    renderDatasetWorkspace(project);
    try {
      let imported = 0;
      for (const job of jobs) {
        const result = await api(
          `/api/workstation/datasets/${encodeURIComponent(project.id)}/import-acquisition-job/${encodeURIComponent(job.id)}`,
          { method: "POST", body: JSON.stringify({}) }
        );
        imported += Number(result.import?.created_count) || 0;
        state.autoImportedAcquisitionJobs.add(job.id);
        state.datasetAcquisitionImportFailures.delete(job.id);
      }
      state.datasetLoading = false;
      await loadSelectedDataset();
      showToast(`${imported} verified target-speaker clips imported for human review`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    } finally {
      state.datasetLoading = false;
      renderDatasetWorkspace(project);
    }
  }

  async function autoImportCompletedAcquisitions(datasetId) {
    const completed = state.jobs.filter(job => (
      job.project_id === datasetId
      && job.type === "dataset.finalize"
      && job.status === "succeeded"
      && !state.autoImportedAcquisitionJobs.has(job.id)
    ));
    for (const job of completed) {
      state.autoImportedAcquisitionJobs.add(job.id);
      try {
        await api(
          `/api/workstation/datasets/${encodeURIComponent(datasetId)}/import-acquisition-job/${encodeURIComponent(job.id)}`,
          { method: "POST", body: JSON.stringify({}) }
        );
        state.datasetAcquisitionImportFailures.delete(job.id);
      } catch (error) {
        state.autoImportedAcquisitionJobs.delete(job.id);
        state.datasetAcquisitionImportFailures.add(job.id);
        throw error;
      }
    }
  }

  async function loadDatasetReviewQueue() {
    const project = selectedDatasetProject();
    if (!project) return;
    try {
      state.datasetReviewQueue = await api(`/api/workstation/datasets/${encodeURIComponent(project.id)}/review-queue`);
      const first = state.datasetReviewQueue.items?.[0];
      if (first) inspectDatasetItem(first);
      else showToast("No clips are waiting for review");
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function loadDatasetExpressionCandidates() {
    const project = selectedDatasetProject();
    if (!project) return;
    try {
      state.datasetExpressionCandidates = await api(`/api/workstation/datasets/${encodeURIComponent(project.id)}/expression-candidates`);
      const groups = Object.keys(state.datasetExpressionCandidates.expressions || {});
      renderDatasetExpressionCandidates();
      if (groups.length) {
        byId("datasetExpressionCandidatesPanel").scrollIntoView({ behavior: "smooth", block: "nearest" });
      } else {
        showToast("No reviewed expression groups are ready");
      }
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  function stopDatasetCandidateAudio() {
    if (!state.datasetCandidateAudio) return;
    state.datasetCandidateAudio.pause();
    state.datasetCandidateAudio.currentTime = 0;
    state.datasetCandidateAudio = null;
    document.querySelectorAll("[data-candidate-preview]").forEach(button => {
      button.classList.remove("active");
      const label = button.querySelector("span");
      if (label) label.textContent = "Preview";
    });
  }

  function previewDatasetExpressionCandidate(datasetId, candidate, button) {
    stopDatasetCandidateAudio();
    const audio = new Audio(`/api/workstation/datasets/${encodeURIComponent(datasetId)}/items/${encodeURIComponent(candidate.item_id)}/audio`);
    audio.preload = "metadata";
    state.datasetCandidateAudio = audio;
    button.classList.add("active");
    const label = button.querySelector("span");
    if (label) label.textContent = "Stop";
    const reset = () => {
      if (state.datasetCandidateAudio === audio) state.datasetCandidateAudio = null;
      button.classList.remove("active");
      if (label) label.textContent = "Preview";
    };
    audio.addEventListener("ended", reset, { once: true });
    audio.addEventListener("error", () => {
      reset();
      showToast("Candidate audio could not be played");
    }, { once: true });
    audio.play().catch(error => {
      reset();
      showToast(error.message, { persistent: true });
    });
  }

  async function useDatasetExpressionCandidate(datasetId, candidate, button) {
    button.disabled = true;
    try {
      const result = await api(
        `/api/workstation/datasets/${encodeURIComponent(datasetId)}/expression-candidates/${encodeURIComponent(candidate.item_id)}/draft`,
        {
          method: "POST",
          body: JSON.stringify(
            Number.isFinite(Number(candidate.expression_intensity))
              ? { intensity: Number(candidate.expression_intensity) }
              : {}
          )
        }
      );
      state.selectedExpression = { source: "local", id: result.draft.id };
      await Promise.all([loadExpressions(), loadArtifacts()]);
      openView("expressions");
      restoreExpressionSelection();
      showToast(result.reused ? "Existing Expression Bank draft opened" : "Expression Bank draft created from the reviewed clip");
    } catch (error) {
      showToast(error.message, { persistent: true });
    } finally {
      button.disabled = false;
    }
  }

  function renderDatasetExpressionCandidates() {
    const panel = byId("datasetExpressionCandidatesPanel");
    const container = byId("datasetExpressionCandidateGroups");
    const project = selectedDatasetProject();
    const groups = state.datasetExpressionCandidates?.expressions;
    const entries = groups && typeof groups === "object" ? Object.entries(groups) : [];
    panel.hidden = !project || !state.datasetExpressionCandidates;
    if (panel.hidden) {
      container.replaceChildren();
      return;
    }
    if (!entries.length) {
      setText("datasetExpressionCandidatesStatus", "No human-reviewed expression groups are ready.");
      container.replaceChildren(create("p", "dataset-expression-empty", "Add expression labels to accepted clips, then rank candidates again."));
      return;
    }
    setText("datasetExpressionCandidatesStatus", `${entries.length} human-reviewed expression group${entries.length === 1 ? "" : "s"} ranked. Suggested emotion metadata remains non-authoritative.`);
    container.replaceChildren(...entries.map(([expression, values]) => {
      const section = create("section", "dataset-expression-group");
      const header = create("header");
      header.append(create("h4", "", expression), create("span", "", `${values.length} clip${values.length === 1 ? "" : "s"}`));
      const list = create("ol");
      list.replaceChildren(...values.slice(0, 5).map((candidate, index) => {
        const item = create("li", "dataset-expression-candidate");
        const rank = create("span", "dataset-expression-rank", index === 0 ? "Recommended" : `#${index + 1}`);
        const evidence = create("div", "dataset-expression-evidence");
        evidence.append(
          create("strong", "", Number(candidate.score).toFixed(3)),
          create("span", "", `${Number(candidate.duration_seconds).toFixed(2)} s`),
          create("span", "", `Speaker ${Number(candidate.speaker_similarity || 0).toFixed(3)}`),
          create("span", "", `Quality ${Number(candidate.quality_score || 0).toFixed(1)}`)
        );
        const actions = create("div", "dataset-expression-candidate-actions");
        const preview = create("button", "row-action", "");
        preview.type = "button";
        preview.dataset.candidatePreview = candidate.item_id;
        const icon = create("i");
        icon.setAttribute("data-lucide", "play");
        preview.append(icon, create("span", "", "Preview"));
        preview.addEventListener("click", () => {
          if (state.datasetCandidateAudio && preview.classList.contains("active")) stopDatasetCandidateAudio();
          else previewDatasetExpressionCandidate(project.id, candidate, preview);
        });
        const use = create("button", "secondary-button", "Use as reference");
        use.type = "button";
        use.addEventListener("click", () => useDatasetExpressionCandidate(project.id, candidate, use));
        actions.append(preview, use);
        item.append(rank, evidence, actions);
        return item;
      }));
      section.append(header, list);
      return section;
    }));
    window.lucide?.createIcons?.({ attrs: { "stroke-width": 1.7 } });
  }

  async function freezeSelectedDataset() {
    const project = selectedDatasetProject();
    if (!project) return;
    try {
      await api(`/api/workstation/datasets/${encodeURIComponent(project.id)}/freeze`, {
        method: "POST",
        body: JSON.stringify({
          require_expressions: state.datasetProjectState?.config?.require_expressions !== false
        })
      });
      await loadSelectedDataset();
      showToast("Dataset frozen with immutable clip and annotation checksums");
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function continueDatasetToTraining() {
    const project = selectedDatasetProject();
    if (!project) return;
    try {
      const training = await api(`/api/workstation/datasets/${encodeURIComponent(project.id)}/continue-to-training`, {
        method: "POST",
        body: JSON.stringify({})
      });
      state.selectedTrainingId = training.id;
      await loadOverview();
      openView("training");
      await selectTrainingProject(training);
      showToast("Training project created from the frozen Dataset manifest");
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  function openDatasetProcessJob() {
    const project = selectedDatasetProject();
    const job = project ? latestDatasetProcessJob(project.id) : null;
    if (!job) return;
    state.selectedJobId = job.id;
    openView("jobs");
    inspectJob(job.id, { scroll: true });
  }

  async function importDatasetProcessArtifacts() {
    const project = selectedDatasetProject();
    const jobs = project ? datasetProcessJobs(project.id).filter(job => job.status === "succeeded") : [];
    if (!project || !jobs.length || state.datasetLoading) return;
    state.datasetLoading = true;
    renderDatasetWorkspace(project);
    try {
      const imported = await importCompletedDatasetJobs(project.id, jobs);
      state.datasetLoading = false;
      await loadSelectedDataset();
      showToast(`${imported} verified segments imported for human review`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    } finally {
      state.datasetLoading = false;
      renderDatasetWorkspace(project);
    }
  }

  async function queueProjectJob(project, jobType) {
    try {
      const job = await api("/api/workstation/jobs", {
        method: "POST",
        body: JSON.stringify({ type: jobType, project_id: project.id, parameters: {} })
      });
      await loadJobs();
      openView("jobs");
      showToast(`Queued ${job.type}`);
    } catch (error) {
      showToast(error.message, { persistent: true });
    }
  }

  async function controlJob(jobId, action) {
    if (state.pendingJobActions.has(jobId)) return;
    state.pendingJobActions.add(jobId);
    renderJobs();
    try {
      const result = await api(
        `/api/workstation/jobs/${encodeURIComponent(jobId)}/${action}`,
        { method: "POST" }
      );
      if (action === "retry") state.selectedJobId = result.id;
      await loadOverview();
      if (state.selectedJobId) {
        await inspectJob(state.selectedJobId, { scroll: false });
      }
      const messages = {
        pause: result.status === "paused"
          ? "Job paused"
          : "Pause requested; waiting for the worker to stop safely",
        resume: "Job resumed and queued",
        retry: `Retry attempt ${result.attempt || ""} queued`,
        cancel: result.status === "cancelled"
          ? "Job cancelled"
          : "Cancellation requested; waiting for the worker to stop safely"
      };
      showToast(messages[action] || "Job updated");
    } catch (error) {
      await loadJobs();
      showToast(error.message, { persistent: true });
    } finally {
      state.pendingJobActions.delete(jobId);
      renderJobs();
    }
  }

  async function cancelJob(jobId) {
    await controlJob(jobId, "cancel");
  }

  async function runJob(jobId) {
    if (state.pendingJobActions.has(jobId)) return;
    state.pendingJobActions.add(jobId);
    renderJobs();
    try {
      await api(`/api/workstation/jobs/${encodeURIComponent(jobId)}/run`, {
        method: "POST"
      });
      await loadOverview();
      showToast("Dataset inventory completed");
    } catch (error) {
      await loadJobs();
      showToast(error.message, { persistent: true });
    } finally {
      state.pendingJobActions.delete(jobId);
      renderJobs();
    }
  }

  function bindEvents() {
    document.addEventListener("click", event => {
      const button = event.target.closest("[data-view]");
      if (button) openView(button.dataset.view);
    });
    document.querySelectorAll("[data-nav-target]").forEach(button => button.addEventListener("click", () => openView(button.dataset.navTarget)));
    byId("overviewStartModelButton").addEventListener("click", () => {
      const destination = document.querySelector(".overview-composition");
      const recentWork = document.querySelector(".recent-work");
      const newProject = document.querySelector(".recent-work [data-open-project]");
      const reducedMotion = window.matchMedia("(prefers-reduced-motion: reduce)").matches;
      destination?.scrollIntoView({ behavior: reducedMotion ? "auto" : "smooth", block: "start" });
      recentWork?.classList.remove("is-guiding");
      window.requestAnimationFrame(() => recentWork?.classList.add("is-guiding"));
      window.setTimeout(() => {
        newProject?.focus({ preventScroll: true });
        recentWork?.classList.remove("is-guiding");
      }, reducedMotion ? 0 : 900);
    });
    document.querySelectorAll("[data-open-project]").forEach(button => button.addEventListener("click", () => openProjectDialog(button.dataset.projectKind || "dataset")));
    byId("projectForm").addEventListener("submit", submitProject);
    byId("projectPreset").addEventListener("change", syncTrainingPresetFields);
    document.querySelectorAll('input[name="datasetAcquisitionMode"]').forEach(input => input.addEventListener("change", () => syncDatasetAcquisitionFields({ animate: true })));
    byId("projectDialog").addEventListener("cancel", event => {
      event.preventDefault();
      closeAnimatedDialog(byId("projectDialog"));
    });
    document.querySelectorAll("[data-dialog-cancel]").forEach(button => {
      button.addEventListener("click", () => closeAnimatedDialog(byId("projectDialog")));
    });
    byId("queueJobButton").addEventListener("click", openJobDialog);
    byId("jobForm").addEventListener("submit", submitJob);
    byId("jobType").addEventListener("change", renderJobDialogOptions);
    byId("jobDialog").addEventListener("cancel", event => {
      event.preventDefault();
      closeAnimatedDialog(byId("jobDialog"));
    });
    document.querySelectorAll("[data-job-dialog-cancel]").forEach(button => {
      button.addEventListener("click", () => closeAnimatedDialog(byId("jobDialog")));
    });
    byId("refreshJobs").addEventListener("click", loadJobs);
    byId("datasetIngestButton").addEventListener("click", ingestSelectedDataset);
    byId("datasetProcessButton").addEventListener("click", queueDatasetProcess);
    byId("datasetPrepareTargetButton").addEventListener("click", queueTargetSpeakerPreparation);
    byId("datasetImportAcquisitionButton").addEventListener("click", importTargetSpeakerAcquisition);
    byId("datasetProcessImportButton").addEventListener("click", importDatasetProcessArtifacts);
    byId("datasetProcessOpenJob").addEventListener("click", openDatasetProcessJob);
    byId("datasetImportListButton").addEventListener("click", importSelectedDatasetList);
    byId("datasetRefreshButton").addEventListener("click", () => loadSelectedDataset({ announce: true }));
    byId("datasetSplitButton").addEventListener("click", assignDatasetSplit);
    byId("datasetManifestButton").addEventListener("click", downloadDatasetManifest);
    byId("datasetQualificationReportButton").addEventListener("click", downloadDatasetQualificationReport);
    byId("datasetReviewQueueButton").addEventListener("click", loadDatasetReviewQueue);
    byId("datasetExpressionCandidatesButton").addEventListener("click", loadDatasetExpressionCandidates);
    byId("datasetExpressionCandidatesClose").addEventListener("click", () => {
      stopDatasetCandidateAudio();
      state.datasetExpressionCandidates = null;
      renderDatasetExpressionCandidates();
    });
    byId("datasetFreezeButton").addEventListener("click", freezeSelectedDataset);
    byId("datasetContinueTrainingButton").addEventListener("click", continueDatasetToTraining);
    byId("datasetAnnotationForm").addEventListener("submit", saveDatasetAnnotations);
    byId("datasetAcceptAudio").addEventListener("click", () => reviewSelectedDatasetAudio("accepted"));
    byId("datasetRejectAudio").addEventListener("click", () => reviewSelectedDatasetAudio("rejected"));
    byId("datasetSpeakerVerify").addEventListener("click", () => verifySelectedDatasetAnnotation("speaker"));
    byId("datasetExpressionVerify").addEventListener("click", () => verifySelectedDatasetAnnotation("expression"));
    byId("datasetExpressionIntensity").addEventListener("input", event => {
      setText("datasetExpressionIntensityValue", Number(event.target.value).toFixed(2));
    });
    byId("datasetQualityAnalyze").addEventListener("click", recalculateDatasetQuality);
    byId("datasetInspectorClose").addEventListener("click", () => {
      state.selectedDatasetItemId = null;
      state.datasetWaveformRequest += 1;
      byId("datasetAudioPreview").pause();
      renderDatasetItemInspector();
    });
    const datasetAudio = byId("datasetAudioPreview");
    byId("datasetAudioToggle").addEventListener("click", toggleDatasetAudio);
    byId("datasetAudioMute").addEventListener("click", () => {
      datasetAudio.muted = !datasetAudio.muted;
      renderDatasetAudioPlayer();
    });
    byId("datasetAudioSeek").addEventListener("input", event => {
      if (Number.isFinite(datasetAudio.duration) && datasetAudio.duration > 0) {
        datasetAudio.currentTime = Number(event.target.value) * datasetAudio.duration;
      }
      renderDatasetAudioPlayer();
    });
    for (const eventName of ["loadedmetadata", "durationchange", "timeupdate", "play", "pause", "ended", "volumechange", "emptied"]) {
      datasetAudio.addEventListener(eventName, renderDatasetAudioPlayer);
    }
    byId("tseRunForm").addEventListener("submit", event => {
      event.preventDefault();
      queueTseRun();
    });
    byId("tseRerunButton").addEventListener("click", () => queueTseRun({ reviewed: true }));
    byId("tseCancelButton").addEventListener("click", cancelSelectedTseJob);
    const tseAudio = byId("tseTargetAudio");
    byId("tseAudioToggle").addEventListener("click", toggleTseAudio);
    byId("tseAudioMute").addEventListener("click", () => {
      tseAudio.muted = !tseAudio.muted;
      renderTseAudioPlayer();
    });
    byId("tseAudioSeek").addEventListener("input", event => {
      if (Number.isFinite(tseAudio.duration) && tseAudio.duration > 0) {
        tseAudio.currentTime = Number(event.target.value) * tseAudio.duration;
      }
      renderTseAudioPlayer();
    });
    for (const eventName of ["loadedmetadata", "durationchange", "timeupdate", "play", "pause", "ended", "volumechange", "emptied"]) {
      tseAudio.addEventListener(eventName, renderTseAudioPlayer);
    }
    byId("trainingRunButton").addEventListener("click", queueTrainingRun);
    byId("productionBuildButton").addEventListener("click", queueProductionBuild);
    byId("referenceNoPreferenceButton").addEventListener("click", useAutomaticTrainingReference);
    byId("referenceAllPoorButton").addEventListener("click", rejectAllTrainingReferences);
    byId("evaluationRunButton").addEventListener("click", () => queueEvaluationRun());
    byId("evaluationChainButton").addEventListener("click", () => queueEvaluationRun({ chain: true }));
    byId("evaluationAudioLanguage").addEventListener("change", event => {
      state.evaluationAudioLanguage = event.target.value;
      renderEvaluationWorkspace(selectedEvaluationProject());
    });
    byId("evaluationBaselineJob").addEventListener("change", () => {
      const project = selectedEvaluationProject();
      const job = state.jobs.find(record => record.id === state.selectedEvaluationJobId)
        || (project ? latestProjectJob(project.id, "evaluation.prepare") : null);
      renderEvaluationAudio(job);
    });
    for (const selectId of (
      [
        "qualificationSubjectEvidence",
        "qualificationAutomatedEvidence",
        "qualificationLongFormEvidence",
        "qualificationExpressionEvidence",
        "qualificationSecurityEvidence",
        "qualificationContentEvidence"
      ]
    )) byId(selectId).addEventListener("change", updateQualificationComposerState);
    byId("qualificationComposeButton").addEventListener("click", composeQualificationEvidence);
    document.querySelector("[data-refresh-expressions]").addEventListener("click", loadExpressions);
    document.querySelector("[data-new-expression]").addEventListener("click", openNewExpression);
    byId("expressionDraftForm").addEventListener("submit", submitExpressionDraft);
    for (const eventName of ["input", "change"]) byId("expressionDraftForm").addEventListener(eventName, () => {
      const form = byId("expressionDraftForm");
      form.dataset.dirty = "true";
      form.dataset.editRevision = String(Number(form.dataset.editRevision || 0) + 1);
    });
    byId("expressionAnalyzeButton").addEventListener("click", analyzeSelectedExpressionReference);
    byId("expressionCancelButton").addEventListener("click", resetExpressionInspector);
    byId("expressionDeleteButton").addEventListener("click", deleteExpressionDraft);
    byId("expressionQualificationEvidence").addEventListener("change", event => {
      const record = state.expressionDrafts.find(item => item.id === byId("expressionDraftId").value);
      byId("expressionPromoteButton").disabled = !record
        || record.qualification_status === "qualified"
        || !event.target.value;
    });
    byId("expressionPromoteButton").addEventListener("click", promoteSelectedExpression);
    byId("artifactPromotionEvidence").addEventListener("change", event => {
      const artifact = state.selectedArtifactDetails?.artifact;
      const eligible = artifact
        && promotableArtifactTypes.has(artifact.type)
        && artifact.status === "ready"
        && !artifact.promoted;
      byId("artifactPromoteButton").disabled = state.pendingArtifactPromotions.has(artifact?.id) || !eligible || !event.target.value;
    });
    byId("artifactPromoteButton").addEventListener("click", promoteSelectedArtifact);
    byId("workstationSettingsForm").addEventListener("submit", saveSettings);
    byId("componentImportButton").addEventListener("click", importComponentBundle);
    byId("componentExportButton").addEventListener("click", exportComponentBundle);
    byId("componentDownloadButton").addEventListener("click", downloadRequiredComponents);
    const clearHistoryDialog = byId("clearHistoryDialog");
    byId("clearVisibleHistoryButton").addEventListener("click", () => openAnimatedDialog(clearHistoryDialog));
    byId("clearHistoryForm").addEventListener("submit", clearVisibleHistory);
    clearHistoryDialog.addEventListener("cancel", event => {
      event.preventDefault();
      closeAnimatedDialog(clearHistoryDialog);
    });
    document.querySelectorAll("[data-history-dialog-cancel]").forEach(button => {
      button.addEventListener("click", () => closeAnimatedDialog(clearHistoryDialog));
    });
    byId("synthesisFrame").addEventListener("load", sendSynthesisSettings);
    window.addEventListener("aniflive-tts:locale-changed", event => {
      state.locale = event.detail?.locale || i18n.getLocale();
      openView(state.currentView, { updateHistory: false });
      sendSynthesisSettings();
      labelAllTables();
      renderTseAudioPlayer();
    });
    byId("expressionIntensity").addEventListener("input", event => {
      setText("expressionIntensityValue", Number(event.target.value).toFixed(2));
    });
    const mobileSheet = byId("mobileNavSheet");
    const openMobileSheet = () => {
      if (!mobileSheet.open) openAnimatedDialog(mobileSheet);
    };
    byId("mobileMoreButton").addEventListener("click", openMobileSheet);
    document.querySelector("[data-mobile-sheet-close]").addEventListener("click", () => closeAnimatedDialog(mobileSheet));
    mobileSheet.addEventListener("click", event => {
      if (event.target === mobileSheet) closeAnimatedDialog(mobileSheet);
    });
    window.addEventListener("message", event => {
      const frame = byId("synthesisFrame");
      if (
        event.origin !== location.origin
        || !(frame instanceof HTMLIFrameElement)
        || event.source !== frame.contentWindow
      ) return;
      if (event.data?.type === "aniflive-tts:synthesis-height") {
        const height = Math.max(520, Math.min(12000, Math.ceil(Number(event.data.height) || 0)));
        frame.style.height = `${height}px`;
        return;
      }
      if (event.data?.type !== "aniflive-tts:playback") return;
      state.playbackActive = event.data.active === true;
      const running = state.overview?.running_jobs || [];
      setVoicePulseActive(running.length > 0 || state.playbackActive);
    });
    window.addEventListener("resize", () => {
      updateDockCursor();
      drawVoicePulse();
      const selected = state.expressionDrafts.find(item => item.id === byId("expressionDraftId").value);
      if (selected && state.currentView === "expressions") {
        drawExpressionWaveform(selected.prosody?.reference_analysis);
      }
    });
    document.addEventListener("visibilitychange", syncAmbientVideo);
    const restoreLocation = () => {
      openView(location.hash.slice(1) || "overview", { updateHistory: false });
    };
    window.addEventListener("popstate", restoreLocation);
    window.addEventListener("hashchange", restoreLocation);
    window.addEventListener("keydown", event => {
      if (event.altKey || event.ctrlKey || event.metaKey || event.shiftKey) return;
      if (event.target instanceof HTMLInputElement || event.target instanceof HTMLSelectElement || event.target instanceof HTMLTextAreaElement) return;
      if (!/^[0-9]$/.test(event.key)) return;
      const index = Number(event.key);
      if (Number.isInteger(index)) {
        const view = index === 0 ? "jobs" : ["overview", "synthesis", "expressions", "datasets", "training", "evaluation", "models", "engines"][index - 1];
        if (view) openView(view);
      }
    });
  }

  async function refreshAll({ announce = true } = {}) {
    const restoreContext = experience.captureContext();
    const button = byId("refreshButton");
    if (button) button.disabled = true;
    try {
      await Promise.all([loadOverview(), loadRuntime(), loadSettings(), loadComponents()]);
      if (state.currentView === "expressions") await loadExpressions();
      if (state.currentView === "datasets" && state.selectedDatasetId) await loadSelectedDataset();
      if (["training", "evaluation", "models"].includes(state.currentView)) await loadArtifacts();
      if (state.currentView === "training") await loadTrainingRunDetails();
      if (state.currentView === "evaluation") await loadEvaluationRunDetails();
      if (announce) showToast("AnifLive-TTS Studio refreshed");
    } catch (error) {
      showToast(error.message, { persistent: true, repeat: announce });
    } finally {
      restoreContext();
      if (button) button.disabled = false;
    }
  }

  function registerStudioServiceWorker() {
    if (!("serviceWorker" in navigator)) return;
    const loopback = new Set(["localhost", "127.0.0.1", "::1"]);
    const secureOrigin = location.protocol === "https:" || loopback.has(location.hostname);
    if (!secureOrigin) return;
    window.addEventListener("load", () => {
      navigator.serviceWorker.register("/studio-sw.js", { scope: "/" }).catch(() => {});
    }, { once: true });
  }

  function studioIsInstalled() {
    return window.matchMedia("(display-mode: standalone)").matches
      || window.navigator.standalone === true;
  }

  function renderStudioInstallState() {
    const installed = studioIsInstalled();
    const button = byId("studioInstallButton");
    setText("studioLaunchMode", installed ? "Installed app" : "Browser");
    setText("studioInstallState", installed ? "Installed" : deferredStudioInstallPrompt ? "Ready" : "Unavailable");
    button.disabled = installed || !deferredStudioInstallPrompt;
  }

  async function installStudioApp() {
    if (!deferredStudioInstallPrompt || studioIsInstalled()) return;
    const prompt = deferredStudioInstallPrompt;
    deferredStudioInstallPrompt = null;
    await prompt.prompt();
    await prompt.userChoice.catch(() => null);
    renderStudioInstallState();
  }

  replaceIcons();
  enhancePathPickers();
  installStyledValidation();
  registerStudioServiceWorker();
  bindEvents();
  experience.initialize();
  byId("studioInstallButton").addEventListener("click", installStudioApp);
  window.addEventListener("beforeinstallprompt", event => {
    event.preventDefault();
    deferredStudioInstallPrompt = event;
    renderStudioInstallState();
  });
  window.addEventListener("appinstalled", () => {
    deferredStudioInstallPrompt = null;
    renderStudioInstallState();
  });
  renderStudioInstallState();
  openView(location.hash.slice(1) || "overview", { updateHistory: false });
  labelAllTables();
  setVoicePulseActive(false);
  renderTseAudioPlayer();
  refreshAll({ announce: false }).catch(error => showToast(error.message, { persistent: true }));
})();

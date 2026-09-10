(() => {
  "use strict";

  const STAGES = Object.freeze([
    { id: "ingest", label: "Ingest" },
    { id: "resample", label: "Resample" },
    { id: "vad", label: "Voice activity" },
    { id: "segment", label: "Segment" },
    { id: "review", label: "Review" },
    { id: "split", label: "Split" },
    { id: "annotation", label: "Annotate" },
    { id: "quality", label: "Quality" },
    { id: "decode", label: "Decode" },
    { id: "tse", label: "TSE" },
    { id: "denoise", label: "Denoise" },
    { id: "dereverb", label: "Dereverb" },
    { id: "asr", label: "ASR" }
  ]);

  function values(items) {
    return Array.isArray(items) ? items.filter(item => item && typeof item === "object") : [];
  }

  function summarize(items) {
    const records = values(items);
    const sources = records.filter(item => item.kind === "source");
    const resampled = records.filter(item => item.kind === "resampled");
    const segments = records.filter(item => item.kind === "segment");
    const accepted = segments.filter(item => item.review_status === "accepted");
    const annotated = records.filter(item => item.annotations?.transcript && item.annotations?.language && item.annotations?.speaker);
    const qualityReports = records.map(item => item.quality).filter(report => report && Number.isFinite(Number(report.quality_score)));
    const decoderRequired = sources.filter(item => item.pipeline_state === "decoder-required");
    const vadReady = records.filter(item => ["vad-analyzed", "segmented", "no-speech"].includes(item.pipeline_state));
    const asrSuggested = segments.filter(item => Boolean(item.metadata?.acquisition?.asr_suggestion?.transcript));
    return Object.freeze({
      total: records.length,
      sources: sources.length,
      resampled: resampled.length,
      segments: segments.length,
      pending: segments.filter(item => item.review_status === "pending").length,
      accepted: accepted.length,
      rejected: segments.filter(item => item.review_status === "rejected").length,
      assigned: accepted.filter(item => ["train", "validation", "test"].includes(item.split_name)).length,
      decoderRequired: decoderRequired.length,
      noSpeech: records.filter(item => item.pipeline_state === "no-speech").length,
      vadReady: vadReady.length,
      asrSuggested: asrSuggested.length,
      annotated: annotated.length,
      qualityAnalyzed: qualityReports.length,
      qualityMean: qualityReports.length
        ? qualityReports.reduce((total, report) => total + Number(report.quality_score), 0) / qualityReports.length
        : null
    });
  }

  function capabilityStage(stage, capabilities) {
    const capability = capabilities?.stages?.[stage.id];
    if (!capability) return { ...stage, state: "unavailable", detail: "Capability status unavailable" };
    if (String(capability.mode || "").startsWith("delegated")) return { ...stage, state: "delegated", detail: capability.reason || "Handled by another module" };
    if (!capability.available) return { ...stage, state: "blocked", detail: capability.reason || "Backend unavailable" };
    return { ...stage, state: "ready", detail: capability.mode || "Available" };
  }

  function pipeline(items, capabilities = null) {
    const summary = summarize(items);
    const stages = STAGES.map(stage => ({ ...stage, state: "waiting", detail: "Not started" }));
    for (let index = 8; index < stages.length; index += 1) stages[index] = capabilityStage(stages[index], capabilities);
    if (!summary.sources && !summary.segments) {
      stages[6] = { ...stages[6], state: "waiting", detail: "No items" };
      stages[7] = { ...stages[7], state: "waiting", detail: "No PCM items" };
      return stages;
    }

    stages[8] = summary.decoderRequired
      ? capabilityStage(stages[8], capabilities)
      : { ...stages[8], state: "complete", detail: "PCM input · decode not required" };
    stages[12] = summary.asrSuggested
      ? { ...stages[12], state: "complete", detail: `${summary.asrSuggested} suggestions · human confirmation required` }
      : summary.annotated
      ? { ...stages[12], state: "complete", detail: "Transcript supplied · ASR not required" }
      : capabilityStage(stages[12], capabilities);

    const observedSources = summary.sources || (summary.segments ? 1 : 0);
    stages[0] = { ...stages[0], state: "complete", detail: `${observedSources} source${observedSources === 1 ? "" : "s"}` };
    if (summary.decoderRequired) {
      stages[1] = { ...stages[1], state: "blocked", detail: `${summary.decoderRequired} need a trusted decoder` };
    } else if (summary.resampled || summary.segments) {
      stages[1] = {
        ...stages[1],
        state: "complete",
        detail: summary.resampled ? `${summary.resampled} normalized WAV` : "Worker-normalized PCM"
      };
    } else {
      stages[1] = { ...stages[1], state: "ready", detail: "PCM WAV ready" };
    }

    if (summary.vadReady) stages[2] = { ...stages[2], state: "complete", detail: `${summary.vadReady} analyzed` };
    else if (summary.resampled) stages[2] = { ...stages[2], state: "ready", detail: "Ready to analyze" };

    if (summary.segments) stages[3] = { ...stages[3], state: "complete", detail: `${summary.segments} segment${summary.segments === 1 ? "" : "s"}` };
    else if (summary.noSpeech) stages[3] = { ...stages[3], state: "blocked", detail: "No speech detected" };
    else if (summary.vadReady) stages[3] = { ...stages[3], state: "ready", detail: "Ready to segment" };

    if (summary.segments && !summary.pending) stages[4] = { ...stages[4], state: "complete", detail: `${summary.accepted} accepted · ${summary.rejected} rejected` };
    else if (summary.segments) stages[4] = { ...stages[4], state: "ready", detail: `${summary.pending} pending` };

    if (summary.accepted && summary.assigned === summary.accepted) stages[5] = { ...stages[5], state: "complete", detail: `${summary.assigned} assigned` };
    else if (summary.accepted) stages[5] = { ...stages[5], state: "ready", detail: `${summary.accepted - summary.assigned} unassigned` };
    stages[6] = summary.annotated
      ? { ...stages[6], state: "complete", detail: `${summary.annotated} annotated` }
      : { ...stages[6], state: "ready", detail: "Import .list or edit manually" };
    stages[7] = summary.qualityAnalyzed
      ? { ...stages[7], state: "complete", detail: `${summary.qualityAnalyzed} scored · mean ${summary.qualityMean.toFixed(1)}` }
      : { ...stages[7], state: summary.decoderRequired === summary.sources ? "blocked" : "ready", detail: "Deterministic PCM analysis" };
    return stages;
  }

  function actions(item) {
    if (!item || typeof item !== "object") return [];
    if (item.kind === "source") {
      if (item.pipeline_state === "decoder-required") return [];
      return [{ id: "resample", label: "Resample" }];
    }
    if (item.kind === "resampled") {
      if (item.pipeline_state === "resampled" || item.pipeline_state === "no-speech") {
        return [{ id: "vad", label: item.pipeline_state === "no-speech" ? "Retry VAD" : "Run VAD" }];
      }
      if (item.pipeline_state === "vad-analyzed") return [{ id: "segment", label: "Segment" }];
      return [];
    }
    if (item.kind === "segment") {
      const choices = [];
      if (item.review_status !== "accepted") choices.push({ id: "accept", label: "Accept" });
      if (item.review_status !== "rejected") choices.push({ id: "reject", label: "Reject" });
      return choices;
    }
    return [];
  }

  function manifestReady(items) {
    const summary = summarize(items);
    return summary.accepted > 0 && summary.assigned === summary.accepted;
  }

  const api = Object.freeze({ STAGES, summarize, pipeline, actions, manifestReady });
  if (typeof module !== "undefined" && module.exports) module.exports = api;
  if (typeof globalThis !== "undefined") globalThis.AnifLiveTTSDatasetModel = api;
})();

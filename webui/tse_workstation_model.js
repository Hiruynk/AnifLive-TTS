(function (root, factory) {
  const api = factory();
  if (typeof module === "object" && module.exports) module.exports = api;
  else root.AnifLiveTTSTSEModel = api;
})(typeof globalThis !== "undefined" ? globalThis : this, function () {
  "use strict";

  const decisions = new Set(["target", "review", "rejected"]);

  function requiredPath(value, field) {
    if (typeof value !== "string" || !value.trim()) {
      throw new Error(`${field} is required`);
    }
    return value.trim();
  }

  function boundedNumber(value, field, minimum, maximum) {
    const parsed = Number(value);
    if (!Number.isFinite(parsed) || parsed < minimum || parsed > maximum) {
      throw new Error(`${field} must be between ${minimum} and ${maximum}`);
    }
    return parsed;
  }

  function normalizeReviewDecisions(value) {
    if (value === undefined || value === null) return {};
    const entries = value instanceof Map ? [...value.entries()] : Object.entries(value);
    const result = {};
    for (const [rawIndex, decision] of entries) {
      const index = Number(rawIndex);
      if (!Number.isInteger(index) || index < 0 || String(index) !== String(rawIndex)) {
        throw new Error("Review decision indices must be canonical non-negative integers");
      }
      if (!decisions.has(decision)) throw new Error("Unsupported TSE review decision");
      result[String(index)] = decision;
    }
    return result;
  }

  function normalizeSeparationSegments(value) {
    if (value === undefined || value === null) return [];
    if (!Array.isArray(value) || value.length > 256) {
      throw new Error("Known overlap ranges must be an array of at most 256 entries");
    }
    let previousEnd = 0;
    return value.map((item, index) => {
      if (!item || typeof item !== "object" || Array.isArray(item)
        || Object.keys(item).sort().join(",") !== "end_seconds,start_seconds") {
        throw new Error("Known overlap ranges require only start_seconds and end_seconds");
      }
      if (typeof item.start_seconds !== "number" || typeof item.end_seconds !== "number") {
        throw new Error(`Known overlap range ${index + 1} must use JSON numbers`);
      }
      const start = item.start_seconds;
      const end = item.end_seconds;
      if (!Number.isFinite(start) || !Number.isFinite(end) || start < 0 || end <= start) {
        throw new Error(`Known overlap range ${index + 1} is malformed`);
      }
      if (index && start < previousEnd) {
        throw new Error("Known overlap ranges must be sorted and must not overlap");
      }
      previousEnd = end;
      return { start_seconds: start, end_seconds: end };
    });
  }

  function buildRunParameters(values, reviewValue, separationValue) {
    if (!values || typeof values !== "object") throw new Error("TSE settings are required");
    return {
      source: requiredPath(values.source, "Source audio"),
      reference: requiredPath(values.reference, "Reference audio"),
      model_package: requiredPath(values.model_package, "V2ProPlus model package"),
      separation_model: requiredPath(values.separation_model, "MossFormer2 separation model"),
      target_threshold: boundedNumber(values.target_threshold, "Target threshold", 0, 1),
      review_margin: boundedNumber(values.review_margin, "Review margin", 0, 0.5),
      frame_ms: boundedNumber(values.frame_ms, "VAD frame", 5, 100),
      minimum_speech_ms: boundedNumber(values.minimum_speech_ms, "Minimum speech", 20, 5000),
      maximum_gap_ms: boundedNumber(values.maximum_gap_ms, "Maximum gap", 0, 5000),
      context_ms: boundedNumber(values.context_ms, "VAD context", 0, 2000),
      extraction_gap_ms: boundedNumber(values.extraction_gap_ms, "Extraction gap", 0, 5000),
      separation_ambiguity_margin: boundedNumber(values.separation_ambiguity_margin, "Separation ambiguity", 0, 0.5),
      review_decisions: normalizeReviewDecisions(reviewValue),
      separation_segments: normalizeSeparationSegments(separationValue)
    };
  }

  function validateReport(value) {
    if (!value || typeof value !== "object" || !new Set(["aniflive-tts-tse-report-v1", "aniflive-tts-tse-report-v2"]).has(value.schema)) {
      throw new Error("TSE report schema is unsupported");
    }
    if (value.schema === "aniflive-tts-tse-report-v2") {
      if (!value.separation_backend || value.separation_backend.name !== "MossFormer2_SS_16K"
        || value.separation_backend.target_selection !== "TensorRT-11 sv_embedding.engine") {
        throw new Error("TSE separation backend metadata is malformed");
      }
      for (const field of ["separation_attempted_segments", "separation_accepted_segments"]) {
        if (!Number.isInteger(value[field]) || value[field] < 0) {
          throw new Error(`TSE report ${field} is malformed`);
        }
      }
      if (value.separation_accepted_segments > value.separation_attempted_segments) {
        throw new Error("TSE report separation counts are malformed");
      }
    }
    const sampleRate = Number(value.sample_rate);
    if (!Number.isInteger(sampleRate) || sampleRate <= 0) throw new Error("TSE report sample rate is malformed");
    for (const field of ["source_seconds", "target_seconds", "review_seconds"]) {
      const number = Number(value[field]);
      if (!Number.isFinite(number) || number < 0) throw new Error(`TSE report ${field} is malformed`);
    }
    for (const field of ["target_segments", "review_segments", "rejected_segments"]) {
      const number = Number(value[field]);
      if (!Number.isInteger(number) || number < 0) throw new Error(`TSE report ${field} is malformed`);
    }
    if (typeof value.speaker_backend !== "string" || typeof value.overlap_policy !== "string") {
      throw new Error("TSE report backend metadata is malformed");
    }
    if (!Array.isArray(value.segments)) throw new Error("TSE report segments are malformed");
    const segments = value.segments.map((segment, position) => {
      if (!segment || typeof segment !== "object") throw new Error("TSE report segment is malformed");
      const index = Number(segment.index);
      const start = Number(segment.start_sample);
      const end = Number(segment.end_sample);
      const similarity = segment.similarity === null ? null : Number(segment.similarity);
      if (!Number.isInteger(index) || index !== position || !Number.isInteger(start) || !Number.isInteger(end) || start < 0 || end <= start) {
        throw new Error("TSE report segment bounds are malformed");
      }
      if (similarity !== null && (!Number.isFinite(similarity) || similarity < -1 || similarity > 1)) {
        throw new Error("TSE report similarity is malformed");
      }
      if (!decisions.has(segment.decision) || typeof segment.overlap !== "boolean") {
        throw new Error("TSE report decision is malformed");
      }
      const separationTriggers = value.schema === "aniflive-tts-tse-report-v2"
        ? segment.separation_triggers : [];
      if (!Array.isArray(separationTriggers)
        || separationTriggers.some(trigger => !new Set(["detector", "operator"]).has(trigger))
        || new Set(separationTriggers).size !== separationTriggers.length) {
        throw new Error("TSE report separation triggers are malformed");
      }
      return {
        index,
        start_sample: start,
        end_sample: end,
        rms_dbfs: Number.isFinite(Number(segment.rms_dbfs)) ? Number(segment.rms_dbfs) : null,
        similarity,
        overlap: segment.overlap,
        decision: segment.decision,
        separation_triggers: [...separationTriggers],
        separation: segment.separation && typeof segment.separation === "object"
          ? { ...segment.separation }
          : null
      };
    });
    const rawRequested = value.schema === "aniflive-tts-tse-report-v2"
      ? value.separation_requested_ranges : [];
    if (!Array.isArray(rawRequested)) {
      throw new Error("TSE report separation ranges are malformed");
    }
    for (const range of rawRequested) {
      if (!range || typeof range !== "object"
        || typeof range.start_seconds !== "number"
        || typeof range.end_seconds !== "number"
        || !Number.isInteger(range.start_sample)
        || !Number.isInteger(range.end_sample)
        || range.start_sample < 0
        || range.end_sample <= range.start_sample
        || Math.abs(range.start_seconds * sampleRate - range.start_sample) > 1
        || Math.abs(range.end_seconds * sampleRate - range.end_sample) > 1) {
        throw new Error("TSE report separation ranges are malformed");
      }
    }
    const requested = normalizeSeparationSegments(rawRequested.map(range => ({
      start_seconds: range.start_seconds,
      end_seconds: range.end_seconds
    })));
    if (requested.some(range => range.end_seconds > Number(value.source_seconds))) {
      throw new Error("TSE report separation range exceeds the source duration");
    }
    const reportRanges = rawRequested.map((range, index) => ({
      start_sample: Number(range.start_sample),
      end_sample: Number(range.end_sample),
      start_seconds: requested[index].start_seconds,
      end_seconds: requested[index].end_seconds
    }));
    return { ...value, sample_rate: sampleRate, segments, separation_requested_ranges: reportRanges };
  }

  function artifactsForJob(artifacts, jobId) {
    if (!Array.isArray(artifacts) || typeof jobId !== "string") return [];
    return artifacts.filter(artifact => artifact && artifact.status === "ready"
      && artifact.metadata?.source === "linux-docker"
      && artifact.metadata?.job_id === jobId
      && artifact.metadata?.job_type === "tse.prepare");
  }

  function decisionsFromReport(report) {
    return new Map(validateReport(report).segments.map(segment => [String(segment.index), segment.decision]));
  }

  function separationSegmentsFromReport(report) {
    return validateReport(report).separation_requested_ranges.map(range => ({
      start_seconds: range.start_seconds,
      end_seconds: range.end_seconds
    }));
  }

  return Object.freeze({
    artifactsForJob,
    buildRunParameters,
    decisionsFromReport,
    normalizeSeparationSegments,
    normalizeReviewDecisions,
    separationSegmentsFromReport,
    validateReport
  });
});

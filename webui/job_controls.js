(function attachAnifLiveTTSJobControls(root, factory) {
  "use strict";
  const controls = factory();
  if (typeof module === "object" && module.exports) module.exports = controls;
  root.AnifLiveTTSJobControls = controls;
})(typeof globalThis === "object" ? globalThis : this, function createJobControls() {
  "use strict";

  const terminalStates = new Set(["succeeded", "failed", "cancelled"]);

  function jobsById(jobs) {
    return new Map((Array.isArray(jobs) ? jobs : []).map(job => [job.id, job]));
  }

  function hasSuccessfulDependencies(job, jobs) {
    const dependencies = Array.isArray(job?.depends_on) ? job.depends_on : [];
    const records = jobsById(jobs);
    return dependencies.every(jobId => records.get(jobId)?.status === "succeeded");
  }

  function hasRetry(job, jobs) {
    return (Array.isArray(jobs) ? jobs : []).some(record => record.retry_of === job?.id);
  }

  function statusPresentation(job) {
    if (job?.status === "running" && job.cancel_requested) {
      return { value: "cancellation-requested", label: "Cancellation requested" };
    }
    if (job?.status === "running" && job.pause_requested) {
      return { value: "pause-requested", label: "Pause requested" };
    }
    if (job?.status === "queued" && typeof job.wait_reason === "string" && job.wait_reason) {
      return { value: "waiting-resource", label: "Waiting for GPU" };
    }
    const value = typeof job?.status === "string" ? job.status : "unknown";
    return { value, label: value };
  }

  function actionPolicy(job, jobs, pending) {
    const status = job?.status;
    const busy = Boolean(pending);
    const cancellationPending = status === "running" && Boolean(job?.cancel_requested);
    const pausePending = status === "running" && Boolean(job?.pause_requested);
    const dependenciesReady = hasSuccessfulDependencies(job, jobs);
    const retried = hasRetry(job, jobs);
    return {
      inspect: { visible: true, disabled: false },
      run: {
        visible: status === "queued" && job?.type === "dataset.inventory",
        disabled: busy || !dependenciesReady,
        reason: dependenciesReady ? "" : "Waiting for dependencies"
      },
      pause: {
        visible: status === "queued" || status === "running",
        disabled: busy || pausePending || cancellationPending,
        reason: pausePending
          ? "Pause already requested"
          : cancellationPending
          ? "Cancellation already requested"
          : ""
      },
      resume: { visible: status === "paused", disabled: busy },
      cancel: {
        visible: !terminalStates.has(status),
        disabled: busy || cancellationPending,
        reason: cancellationPending ? "Cancellation already requested" : ""
      },
      retry: {
        visible: status === "failed" || status === "cancelled",
        disabled: busy || retried,
        reason: retried ? "A retry attempt already exists" : ""
      }
    };
  }

  function shortId(value) {
    if (typeof value !== "string" || !value) return "Unknown";
    const suffix = value.slice(-8);
    return value.length > 15 ? `${value.slice(0, value.indexOf("_") + 1)}...${suffix}` : value;
  }

  function dependencyRecords(job, jobs) {
    const records = jobsById(jobs);
    return (Array.isArray(job?.depends_on) ? job.depends_on : []).map(jobId => ({
      id: jobId,
      status: records.get(jobId)?.status || "missing"
    }));
  }

  return Object.freeze({
    actionPolicy,
    dependencyRecords,
    hasSuccessfulDependencies,
    shortId,
    statusPresentation
  });
});

"""CPU-only evaluation checks shared by Studio and the job scheduler."""
from __future__ import annotations

import json
from collections.abc import Mapping

from .workstation import WorkstationError
from .workstation_evaluation import (
    _ALLOWED_SETTINGS, EvaluationWorkerError, evaluation_workload,
    preflight_evaluation_baseline, resolve_evaluation_plan,
)


def preflight_workstation_evaluation(store, project_id, parameters, *, allowed_roots=None):
    project = store.get_project(project_id)
    effective = {**project.get("config", {}), **parameters}
    if effective.get("listening_plan"):
        return {"status": "not-applicable", "reason": "conversion listening workflow"}
    try:
        plan = resolve_evaluation_plan({
            key: value for key, value in effective.items() if key in _ALLOWED_SETTINGS
        })
        source = effective.get("baseline_report")
        if source is None:
            return {"status": "uncompared", "workload": evaluation_workload(plan),
                    "reason": "No baseline supplied; no regression pass will be claimed"}
        path = store.validate_import_path(source, allowed_roots=allowed_roots)
        if not path.is_file() or path.stat().st_size > 16 * 1024 * 1024:
            raise WorkstationError("Baseline must be a JSON report up to 16 MiB")
        baseline = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(baseline, Mapping):
            raise WorkstationError("Baseline must be a JSON object")
        return preflight_evaluation_baseline(plan, baseline)
    except (EvaluationWorkerError, OSError, ValueError) as error:
        raise WorkstationError(str(error)) from error

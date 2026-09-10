import subprocess
import sys

import pytest

from aniflive_tts import workstation_evaluation as evaluation


def test_benchmark_streams_real_session_progress_and_keeps_failure_log(tmp_path, monkeypatch):
    events = []
    monkeypatch.setattr(evaluation, "_emit_progress", lambda *event: events.append(event))
    script = (
        "import sys; "
        "print('[benchmark] model session 1/2: warmup=1', flush=True); "
        "print('[benchmark] model session 2/2: warmup=1', flush=True); "
        "print('failure details', flush=True); sys.exit(7)"
    )
    log = tmp_path / "benchmark.log"
    code, tail = evaluation._run_logged_benchmark(
        [sys.executable, "-u", "-c", script], cwd=tmp_path,
        log_path=log, timeout=10, sessions=2,
    )
    assert code == 7
    assert "failure details" in tail
    assert log.read_text().endswith("failure details\n")
    assert [event[2] for event in events] == [
        "Canonical benchmark session 1/2 started",
        "Canonical benchmark session 2/2 started",
    ]
    assert events[0][0] < events[1][0] < 0.78


def test_benchmark_timeout_preserves_partial_output_and_reaps_child(tmp_path):
    script = "import os,time; print(os.getpid(), flush=True); print('partial', flush=True); time.sleep(60)"
    log = tmp_path / "benchmark.log"
    with pytest.raises(subprocess.TimeoutExpired):
        evaluation._run_logged_benchmark(
            [sys.executable, "-u", "-c", script], cwd=tmp_path,
            log_path=log, timeout=1, sessions=2,
        )
    lines = log.read_text().splitlines()
    assert lines[1] == "partial"
    import os
    with pytest.raises(ProcessLookupError):
        os.kill(int(lines[0]), 0)

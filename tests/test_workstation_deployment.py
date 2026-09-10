import hashlib
import json
import pytest
from aniflive_tts.workstation import WorkstationStore, WorkstationError
from aniflive_tts.workstation_deployment import promote_and_publish
from test_qualification_promotion import _register_ready, _compose


@pytest.mark.parametrize("corrupt", [False, True])
def test_qualified_package_publication_is_verified_and_repeatable(tmp_path, monkeypatch, corrupt):
    store = WorkstationStore(tmp_path / "state")
    manifest = json.dumps({"model_id": "candidate", "semantic_sampling": "legacy-topk-v1"}).encode()
    engine = b"serialized engine fixture"
    checksums = json.dumps({
        "manifest.json": hashlib.sha256(manifest).hexdigest(),
        "engines/test.engine": hashlib.sha256(engine).hexdigest(),
    }).encode()
    records = []
    for index, (name, content) in enumerate([
        ("manifest.json", manifest), ("engines/test.engine", engine), ("checksums.json", checksums),
    ]):
        records.append(_register_ready(
            store, artifact_type="package", name=name,
            relative=f"worker/file-{index}/" + name, content=content,
            metadata={"job_id": "job_fixture", "worker_relative_path": "model-package/" + name},
        ))
    monkeypatch.setattr(store, "get_job", lambda job_id: {
        "id": "job_fixture", "type": "model.package", "status": "succeeded",
        "result": {"registered_artifact_ids": [record["id"] for record in records]},
    })
    qualified = _compose(store, subject_kind="artifact", subject_id=records[0]["id"])["qualification"]
    if corrupt:
        (store.artifact_root / records[1]["local_path"]).write_bytes(b"changed")
        with pytest.raises(WorkstationError, match="file changed"):
            promote_and_publish(store, records[0]["id"], qualified["id"])
        assert not (tmp_path / "models/candidate").exists()
        assert not store.get_artifact(records[0]["id"])["promoted"]
    else:
        result = promote_and_publish(store, records[0]["id"], qualified["id"])
        assert result["promoted"] is True
        assert result["deployment"]["status"] == "installed"
        target = tmp_path / "models/candidate"
        published = json.loads((target / "manifest.json").read_text())
        assert published["qualification"]["id"] == qualified["id"]
        assert (target / "engines/test.engine").read_bytes() == engine
        again = promote_and_publish(store, records[0]["id"], qualified["id"])
        assert again["deployment"] == result["deployment"]


@pytest.mark.parametrize("phase,running,expected", [
    ("idle", False, "starting"), ("idle", True, "running"),
    ("ready-for-job", False, "deferred"),
])
def test_managed_start_respects_gpu_handoff(tmp_path, monkeypatch, phase, running, expected):
    from aniflive_tts.workstation_deployment import _start_managed_runtime
    import aniflive_tts.workstation_handoff as handoff
    store = WorkstationStore(tmp_path / "state")
    monkeypatch.setenv("ANIFLIVE_TTS_MANAGED_RUNTIME_START", "1")
    monkeypatch.setattr(store, "get_runtime_handoff", lambda: {"phase": phase})
    calls = []
    class Coordinator:
        def __init__(self, *args):
            pass
        def _acquire(self):
            calls.append("acquire")
        def _inspect(self):
            calls.append("inspect-owned")
            return {"Id": "owned-runtime", "State": {"Running": running}}
        def _docker(self, *args):
            calls.append(args)
        def close(self):
            calls.append("close")
    monkeypatch.setattr(handoff, "RuntimeHandoff", Coordinator)
    assert _start_managed_runtime(store)["state"] == expected
    assert calls[-1] == "close"
    assert (("start", "owned-runtime") in calls) == (expected == "starting")
    if phase != "idle":
        assert "inspect-owned" not in calls

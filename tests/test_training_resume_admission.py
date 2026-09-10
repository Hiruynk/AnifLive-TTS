import json
from pathlib import Path
import pytest
from aniflive_tts.workstation import WorkstationStore, WorkstationError
from aniflive_tts.workstation_adapters import run_adapter
from aniflive_tts.workstation_training import validate_training_resume_request, TrainingWorkerError


def test_resume_checkpoint_is_in_the_admitted_container_manifest(tmp_path):
    store=WorkstationStore(tmp_path/"state")
    resume=tmp_path/"checkpoint";resume.mkdir();(resume/"latest.json").write_text("{}")
    project=store.create_project(kind="training",name="Resume admission")
    job=store.create_job(job_type="training.prepare",project_id=project["id"],
                         parameters={"resume_checkpoint":str(resume)})
    result=run_adapter(job,project,manifest_root=tmp_path/"manifests",allowed_path_roots=[tmp_path])
    manifest=json.loads(Path(result.payload["manifest_path"]).read_text())
    assert manifest["resume_requested"] is True
    assert manifest["input_paths"]["resume_checkpoint"]==str(resume)
    assert manifest["container_input_paths"]["resume_checkpoint"]=="/aniflive/input/resume_checkpoint"


def test_missing_requested_checkpoint_cannot_be_silently_ignored(tmp_path):
    store=WorkstationStore(tmp_path/"state")
    project=store.create_project(kind="training",name="Missing resume")
    job=store.create_job(job_type="training.prepare",project_id=project["id"],
                         parameters={"resume_checkpoint":str(tmp_path/"missing")})
    with pytest.raises(WorkstationError):
        run_adapter(job,project,manifest_root=tmp_path/"manifests",allowed_path_roots=[tmp_path])


@pytest.mark.parametrize("inputs",[None,{},{"resume_checkpoint":""}])
def test_neural_worker_refuses_resume_without_checkpoint_before_loading_cuda(inputs):
    with pytest.raises(TrainingWorkerError,match="refusing fresh"):
        validate_training_resume_request({"resume_requested":True,"container_input_paths":inputs})

def test_legacy_weight_folder_is_not_a_complete_resume_snapshot(tmp_path):
    legacy=tmp_path/"legacy";legacy.mkdir()
    with pytest.raises(TrainingWorkerError,match="legacy weights"):
        validate_training_resume_request({"resume_requested":True,"container_input_paths":{"resume_checkpoint":str(legacy)}})


def test_started_training_cannot_resume_as_fresh_work(tmp_path):
    store=WorkstationStore(tmp_path/"state")
    project=store.create_project(kind="training",name="Pause safety")
    job=store.create_job(job_type="training.prepare",project_id=project["id"])
    claim=store.claim_job(job["id"])
    store.pause_job(job["id"])
    store.update_job(job["id"],status="paused",claim_token=claim.token)
    with pytest.raises(WorkstationError,match="refusing to restart"):
        store.resume_job(job["id"])
    assert store.get_job(job["id"])["status"]=="paused"


def test_training_paused_before_start_can_still_be_queued(tmp_path):
    store=WorkstationStore(tmp_path/"state")
    project=store.create_project(kind="training",name="Queued pause")
    job=store.create_job(job_type="training.prepare",project_id=project["id"])
    store.pause_job(job["id"])
    assert store.resume_job(job["id"])["status"]=="queued"


def _paused_snapshot_job(tmp_path):
    from aniflive_tts.training_snapshot import commit_training_snapshot
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="training", name="Checkpoint continuation")
    job = store.create_job(job_type="training.prepare", project_id=project["id"],
                           parameters={"seed": 1234})
    claim = store.claim_job(job["id"])
    root = store.root / "docker-worker-output" / job["id"] / ("a" * 32) / "recovery-snapshots"
    with commit_training_snapshot(root, metadata={
        "stage": "gpt", "epoch": 2, "global_step": 16,
        "job_id": job["id"], "source_fingerprint": "fixture",
    }) as stage:
        (stage / "gpt").mkdir()
        (stage / "gpt/trainer.ckpt").write_bytes(b"framework-state")
        (stage / "gpt/random-state.pt").write_bytes(b"rng-state")
        (stage / "training-plan.json").write_text("{}")
        (stage / "training-budget.json").write_text("{}")
    store.pause_job(job["id"])
    store.update_job(job["id"], status="paused", claim_token=claim.token)
    return store, job, root


def test_paused_training_queues_one_checkpoint_continuation(tmp_path):
    store, job, root = _paused_snapshot_job(tmp_path)
    resumed = store.resume_job(job["id"])
    assert resumed["id"] != job["id"]
    assert resumed["retry_of"] == job["id"]
    assert resumed["attempt"] == 2
    assert resumed["parameters"] == {"seed": 1234, "resume_checkpoint": str(root)}
    assert store.get_job(job["id"])["status"] == "paused"
    assert store.resume_job(job["id"])["id"] == resumed["id"]


def test_paused_training_rejects_corrupt_saved_state(tmp_path):
    store, job, root = _paused_snapshot_job(tmp_path)
    pointer = json.loads((root / "latest.json").read_text())
    (root / pointer["snapshot_id"] / "gpt/trainer.ckpt").write_bytes(b"corrupt")
    with pytest.raises(WorkstationError, match="checkpoint cannot be resumed"):
        store.resume_job(job["id"])
    assert len(store.list_jobs()) == 1

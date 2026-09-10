from aniflive_tts.workstation import WorkstationStore
from aniflive_tts.workstation_worker import WorkstationWorker


def test_successful_dataset_retry_reconnects_dependency_failures_only(tmp_path):
    store = WorkstationStore(tmp_path / "state")
    project = store.create_project(kind="dataset", name="QA", config={"acquisition_mode": "target-speaker", "reference_audio": str(tmp_path / "reference.wav"), "sources": [str(tmp_path / "source.wav")]})
    old = store.create_job(job_type="dataset.transcribe", project_id=project["id"])
    store.update_job(old["id"], status="failed", error="transport failure")
    child = store.create_job(job_type="dataset.finalize", project_id=project["id"], depends_on=[old["id"]])
    store.update_job(child["id"], status="failed", error=f"Dependency {old['id']} did not succeed")
    cancelled = store.create_job(job_type="dataset.finalize", project_id=project["id"], depends_on=[old["id"]])
    store.cancel_job(cancelled["id"])
    retried = store.retry_job(old["id"])
    claim = store.claim_job(retried["id"])
    completed = store.update_job(retried["id"], status="succeeded", claim_token=claim.token)
    worker = WorkstationWorker(store, allowed_path_roots=(tmp_path,))
    worker._queue_dataset_retry_dependents(completed)
    worker._queue_dataset_retry_dependents(completed)
    next_jobs = [job for job in store.list_jobs() if job.get("retry_of") == child["id"]]
    assert len(next_jobs) == 1
    assert next_jobs[0]["depends_on"] == [completed["id"]]
    assert store.get_job(child["id"])["status"] == "failed"
    assert store.get_job(cancelled["id"])["status"] == "cancelled"
    assert not any(job.get("retry_of") == cancelled["id"] for job in store.list_jobs())

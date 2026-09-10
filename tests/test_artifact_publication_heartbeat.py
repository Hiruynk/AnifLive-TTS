import hashlib
import threading
import pytest
from aniflive_tts.workstation import WorkstationStore, WorkstationError

def staged(tmp_path):
    store=WorkstationStore(tmp_path/"state")
    project=store.create_project(kind="training",name="Publication heartbeat")
    job=store.create_job(job_type="training.prepare",project_id=project["id"])
    claim=store.claim_job(job["id"])
    source=tmp_path/"output";source.mkdir()
    path=source/"state.ckpt";path.write_bytes(b"checkpoint")
    artifacts=store.register_worker_artifacts(
        job_id=job["id"],claim_token=claim.token,job_type="training.prepare",
        project_id=project["id"],source_root=source,image_digest="sha256:"+"a"*64,
        artifacts=[{"kind":"checkpoint","relative_path":path.name,
                    "sha256":hashlib.sha256(path.read_bytes()).hexdigest(),"size_bytes":path.stat().st_size}],
    )
    return store,job,claim,artifacts

def test_completion_verification_does_not_hold_sqlite_writer_lock(tmp_path,monkeypatch):
    store,job,claim,artifacts=staged(tmp_path)
    token=store.acquire_resource_lease("runtime:handoff",purpose="test",lease_seconds=120)
    original=store._verify_staged_worker_artifacts
    errors=[]
    def verify(rows):
        result=original(rows)
        def heartbeat():
            try:store.heartbeat_resource_lease(token,lease_seconds=120)
            except Exception as error:errors.append(error)
        thread=threading.Thread(target=heartbeat)
        thread.start();thread.join(2)
        assert not thread.is_alive(),"Checkpoint verification blocked coordinator heartbeat"
        return result
    monkeypatch.setattr(store,"_verify_staged_worker_artifacts",verify)
    store.update_job(job["id"],status="succeeded",claim_token=claim.token,
                     result={"registered_artifact_ids":[a["id"] for a in artifacts]})
    assert not errors
    assert store.get_job(job["id"])["status"]=="succeeded"

def test_file_changed_after_hash_is_not_published(tmp_path,monkeypatch):
    store,job,claim,artifacts=staged(tmp_path)
    original=store._verify_staged_worker_artifacts
    def verify(rows):
        result=original(rows)
        path=store.artifact_root/artifacts[0]["local_path"]
        path.write_bytes(b"changed after verification")
        return result
    monkeypatch.setattr(store,"_verify_staged_worker_artifacts",verify)
    with pytest.raises(WorkstationError,match="changed before job completion"):
        store.update_job(job["id"],status="succeeded",claim_token=claim.token,
                         result={"registered_artifact_ids":[a["id"] for a in artifacts]})
    assert store.get_job(job["id"])["status"]=="running"
    assert not store.list_artifacts(status="ready")

def test_worker_renews_job_lease_during_final_verification(tmp_path,monkeypatch):
    import time
    from aniflive_tts.workstation_worker import WorkstationWorker
    store=WorkstationStore(tmp_path/"state")
    source=tmp_path/"source";source.mkdir();(source/"voice.wav").write_bytes(b"RIFF")
    project=store.create_project(kind="dataset",name="Verification heartbeat",config={"source":str(source)})
    job=store.create_job(job_type="dataset.inventory",project_id=project["id"])
    heartbeats=[]
    original_heartbeat=store.heartbeat_job
    def heartbeat(*args,**kwargs):
        heartbeats.append(time.monotonic())
        return original_heartbeat(*args,**kwargs)
    monkeypatch.setattr(store,"heartbeat_job",heartbeat)
    original_verify=store._verify_staged_worker_artifacts
    def verify(rows):
        started=time.monotonic()
        time.sleep(0.3)
        assert any(t>=started for t in heartbeats),"Job heartbeat stopped before final verification"
        return original_verify(rows)
    monkeypatch.setattr(store,"_verify_staged_worker_artifacts",verify)
    worker=WorkstationWorker(store,allowed_path_roots=[tmp_path],heartbeat_seconds=0.05)
    worker.run_once()
    assert store.get_job(job["id"])["status"]=="succeeded"

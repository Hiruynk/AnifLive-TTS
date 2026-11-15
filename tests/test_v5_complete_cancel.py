import queue
import threading
from types import SimpleNamespace

import pytest

from aniflive_tts import service
from aniflive_tts.backend.v5.ar_worker_wake import WorkerWake
from aniflive_tts.backend.v5.runtime import (
    V5Service, _CompleteCancelled, _complete_cancellation,
)


def test_complete_checks_preserve_calls_and_restore_bindings():
    stop = threading.Event()
    calls = []
    cfm = SimpleNamespace(inference=lambda value: calls.append(("cfm", value)) or value)
    worker = object()
    accelerator = SimpleNamespace(engine=worker, infer_batch=lambda value: calls.append(("ar", value)) or value)
    def vocoder(value):
        calls.append(("vocoder", value))
        return value
    engine = SimpleNamespace(vits_model=SimpleNamespace(cfm=cfm), vocoder=vocoder)
    wake = SimpleNamespace(engine=worker, stop=None)
    wake.set_complete_stop = lambda value: setattr(wake, "stop", value)
    original_ar, original_cfm = accelerator.infer_batch, cfm.inference

    with _complete_cancellation(engine, accelerator, wake, stop):
        assert wake.stop is stop
        assert accelerator.infer_batch(1) == 1
        assert cfm.inference(2) == 2
        assert engine.vocoder(3) == 3
    assert calls == [("ar", 1), ("cfm", 2), ("vocoder", 3)]
    assert wake.stop is None
    assert accelerator.infer_batch is original_ar
    assert cfm.inference is original_cfm
    assert engine.vocoder is vocoder


def test_complete_cancels_at_cfm_and_vocoder_boundaries():
    stop = threading.Event()
    calls = []
    def cfm_inference():
        calls.append("cfm")
        stop.set()
        return "mel"
    cfm = SimpleNamespace(inference=cfm_inference)
    worker = object()
    accelerator = SimpleNamespace(engine=worker, infer_batch=lambda: "semantic")
    engine = SimpleNamespace(vits_model=SimpleNamespace(cfm=cfm), vocoder=lambda value: calls.append("vocoder"))
    wake = SimpleNamespace(engine=worker, set_complete_stop=lambda stop: None)
    with _complete_cancellation(engine, accelerator, wake, stop):
        with pytest.raises(_CompleteCancelled):
            cfm.inference()
        with pytest.raises(_CompleteCancelled):
            engine.vocoder("mel")
    assert calls == ["cfm"]
    assert cfm.inference is cfm_inference


def test_ar_worker_cancels_only_after_queue_drain_and_finalizes_on_owner():
    stop = threading.Event()
    stop.set()
    done = threading.Event()
    handle = SimpleNamespace(cancelled=threading.Event())
    session = SimpleNamespace(request_id="request")
    runner = SimpleNamespace(bound_sessions=[])
    engine = SimpleNamespace(_stop_event=done, waiting_bind=[], model_runner=runner,
                             handles={"request": handle}, pending_requests=queue.SimpleQueue())
    def drain():
        assert not handle.cancelled.is_set()
        engine.waiting_bind.append(session)
    engine._drain_new_requests = drain
    finished = []
    def finalize(item, error, cancelled):
        finished.append((item, str(error), cancelled, threading.get_ident()))
        done.set()
    engine._finalize_session = finalize
    wake = WorkerWake.__new__(WorkerWake)
    wake.engine, wake.complete_stop = engine, stop
    wake.selected = False
    wake._loop()
    assert finished == [(session, "T2S request cancelled", True, threading.get_ident())]
    assert not engine.waiting_bind


def test_ar_cancel_signal_becomes_409_and_restores_request_slot():
    stop = threading.Event()
    worker = object()
    def infer():
        stop.set()
        raise RuntimeError("T2S request cancelled")
    accelerator = SimpleNamespace(engine=worker, infer_batch=infer)
    engine = SimpleNamespace(vits_model=SimpleNamespace(cfm=SimpleNamespace(inference=lambda: None)),
                             vocoder=lambda: None)
    wake = SimpleNamespace(engine=worker, set_complete_stop=lambda value: None)
    owned = []
    def chunks():
        with _complete_cancellation(engine, accelerator, wake, stop):
            accelerator.infer_batch()
        yield b"unreachable"
    pcm = service._OwnedPCMIterator(chunks(), lambda: owned.append("released"))
    instance = V5Service.__new__(V5Service)
    instance._active_lock = threading.Lock()
    instance._stop = stop
    instance.prepare_stream = lambda *args, **kwargs: (["text"], pcm, 48000, "model")
    with pytest.raises(service.SynthesisCancelled) as error:
        instance.synthesize_request(None, None)
    assert error.value.status_code == 409
    assert owned == ["released"]
    assert accelerator.infer_batch is infer

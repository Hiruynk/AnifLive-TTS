from collections import OrderedDict
from contextlib import nullcontext
import sys
import threading
from types import SimpleNamespace

from aniflive_tts.backend.v5.cfm_cuda_graph import CFMGraphAdapter


class Tensor:
    device = "cpu"

    def __init__(self, key):
        self.key = key

    def detach(self):
        return self

    def contiguous(self):
        return self

    def clone(self):
        return Tensor(self.key)


class Event:
    def __init__(self):
        self.recorded = False
        self.synchronized = False

    def record(self, stream):
        self.recorded = True

    def synchronize(self):
        self.synchronized = True


def adapter_with_full_cache(monkeypatch):
    cuda = SimpleNamespace(current_stream=lambda device: SimpleNamespace(wait_event=lambda event: None),
                           Event=Event)
    monkeypatch.setitem(sys.modules, "torch", SimpleNamespace(cuda=cuda, inference_mode=nullcontext))
    adapter = CFMGraphAdapter.__new__(CFMGraphAdapter)
    adapter.closed = False
    adapter.pending_error = None
    adapter._lock = threading.RLock()
    adapter.contexts = OrderedDict((key, object()) for key in ((100, 20), (200, 20), (300, 20)))
    adapter.pending = OrderedDict()
    adapter.collect_misses = True
    adapter.misses = 0
    adapter._last_eager_done = None
    adapter.ar_is_idle = lambda: True
    adapter._check_cfm = lambda: None
    adapter._validate = lambda mu, lengths, prompt: mu.key
    adapter.original_inference = lambda *args, **kwargs: "eager-output"
    return adapter


def infer(adapter, key):
    value = Tensor(key)
    return adapter.inference(value, value, value, 4, inference_cfg_rate=0)


def test_full_cache_collects_hot_miss_and_prepares_with_lru(monkeypatch):
    adapter = adapter_with_full_cache(monkeypatch)
    hot = (579, 403)
    assert infer(adapter, hot) == "eager-output"
    assert infer(adapter, hot) == "eager-output"
    assert list(adapter.pending) == [hot]
    event = adapter.pending[hot][-1]
    prepared = []
    def prepare(mu, lengths, prompt):
        assert event.synchronized
        prepared.append(mu.key)
        if len(adapter.contexts) == 3:
            adapter.contexts.popitem(last=False)
        adapter.contexts[mu.key] = object()
    adapter.prepare = prepare
    adapter.prepare_pending()
    assert event.recorded and prepared == [hot]
    assert list(adapter.contexts) == [(200, 20), (300, 20), hot]
    assert not adapter.pending


def test_pending_is_bounded_and_existing_capture_is_not_repeated(monkeypatch):
    adapter = adapter_with_full_cache(monkeypatch)
    keys = [(400, 20), (500, 20), (600, 20), (700, 20)]
    for key in keys:
        assert infer(adapter, key) == "eager-output"
    assert list(adapter.pending) == keys[:3]
    adapter.contexts.popitem(last=False)
    adapter.contexts[keys[0]] = object()  # Explicit preparation won the race.
    prepared = []
    adapter.prepare = lambda mu, lengths, prompt: prepared.append(mu.key)
    adapter.prepare_pending()
    assert prepared == keys[1:3]
    assert not adapter.pending

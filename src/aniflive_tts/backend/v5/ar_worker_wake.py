"""Opt-in idle wakeup for the pinned, single-owner research AR engine."""
import hashlib
import importlib
from collections import deque
import threading
import time

import torch


_SOURCE_SHA256 = 'ddf746da1dc88fa0e39f71fa42b14a4a2cb8cb97f725916ef92ec55f02b33f25'


class _SignallingQueue:
    def __init__(self, original, wake, enqueued, lock):
        self.original, self.wake, self.enqueued, self.lock = original, wake, enqueued, lock

    def put(self, item):
        with self.lock:
            self.enqueued[item[0].request_id] = time.perf_counter_ns()
            self.original.put(item)
        self.wake.set()

    def __getattr__(self, name):
        return getattr(self.original, name)


class WorkerWake:
    def __init__(self, engine):
        module = importlib.import_module(type(engine).__module__)
        with open(module.__file__, 'rb') as source:
            actual = hashlib.sha256(source.read()).hexdigest()
        if actual != _SOURCE_SHA256:
            raise RuntimeError(f'Unsupported AR worker source SHA256: {actual}')
        self.engine, self.runner = engine, engine.model_runner
        self.event, self.lock = threading.Event(), threading.Lock()
        self.enqueued, self.pickup_ns = {}, deque(maxlen=256)
        self.selected = False
        self.complete_stop = None
        self.closed = False
        self._check_idle()
        self._stop_worker()
        self.original_queue = engine.pending_requests
        self.original_loop = engine.__dict__.get('_background_loop')
        self.had_loop = '_background_loop' in engine.__dict__
        self.original_prefill = self.runner.__dict__.get('prefill')
        self.had_prefill = 'prefill' in self.runner.__dict__
        self.prefill = self.runner.prefill
        engine.pending_requests = _SignallingQueue(self.original_queue, self.event, self.enqueued, self.lock)
        engine._background_loop = self._loop
        self.runner.prefill = self._prefill
        self._start_worker()

    def _check_idle(self):
        engine = self.engine
        if (engine.handles or not engine.pending_requests.empty() or engine.waiting_bind
                or engine.model_runner.bound_sessions or any(engine.model_runner.slot_sessions)
                or engine.model_runner.max_batch_size != 1):
            raise RuntimeError('AR worker wakeup may only change while the engine is idle')

    def _stop_worker(self):
        self.engine._stop_event.set()
        self.event.set()
        thread = self.engine._loop_thread
        if thread is not None:
            thread.join(timeout=2.0)
            if thread.is_alive():
                raise RuntimeError('AR worker did not stop while idle')
        self.engine._loop_thread = None

    def _start_worker(self):
        self.engine._stop_event.clear()
        thread = threading.Thread(target=self.engine._background_loop, daemon=True)
        self.engine._loop_thread = thread
        thread.start()

    def select(self, enabled):
        self._check_idle()
        self.selected = bool(enabled)
        with self.lock:
            self.enqueued.clear()
            self.pickup_ns.clear()
        self.event.set()

    def stats(self):
        with self.lock:
            return {'event_wait': self.selected, 'enqueue_to_prefill_ms': [x / 1e6 for x in self.pickup_ns]}

    def set_complete_stop(self, stop):
        self.complete_stop = stop
        self.event.set()

    def _prefill(self, request):
        now = time.perf_counter_ns()
        with self.lock:
            start = self.enqueued.pop(request.request_id, None)
            if start is not None:
                self.pickup_ns.append(now - start)
        return self.prefill(request)

    @torch.inference_mode()
    def _loop(self):
        # Same processing loop as pinned t2s_engine.py, with only idle wait changed.
        engine = self.engine
        torch.set_grad_enabled(False)
        while not engine._stop_event.is_set():
            engine._drain_new_requests()
            complete_stop = self.complete_stop
            for session in list(engine.waiting_bind) + list(engine.model_runner.bound_sessions):
                handle = engine.handles.get(session.request_id)
                # Mark only sessions already drained and prefetched: the pinned
                # queue drain does not emit a terminal response for a cancelled
                # request that has not yet been prefetched.
                if handle is not None and complete_stop is not None and complete_stop.is_set():
                    handle.cancelled.set()
                if handle is not None and handle.cancelled.is_set():
                    if session in engine.waiting_bind:
                        engine.waiting_bind.remove(session)
                    else:
                        engine.model_runner.unbind_session(session)
                    engine._finalize_session(session, RuntimeError('T2S request cancelled'), cancelled=True)
            if not engine.model_runner.bound_sessions:
                if not engine.waiting_bind:
                    if self.selected:
                        self.event.clear()
                        if engine.pending_requests.empty() and not engine._stop_event.is_set():
                            self.event.wait(timeout=0.001)
                    else:
                        time.sleep(0.001)
                    continue
            try:
                finished_sessions = engine.model_runner.decode_step()
            except Exception as error:
                for session in list(engine.model_runner.bound_sessions):
                    engine._finalize_session(session, exception=error)
                    engine.model_runner.unbind_session(session)
                continue
            for session in list(engine.model_runner.bound_sessions):
                engine._maybe_stream_partial(session)
                if session.progress_task is not None and session.progress is not None:
                    session.progress.update(session.progress_task, advance=session.bsz)
            if finished_sessions:
                for session in finished_sessions:
                    engine._finalize_session(session, exception=None)

    def close(self):
        if self.closed:
            return
        self._check_idle()
        if self.runner.prefill != self._prefill:
            raise RuntimeError('Close prefill graph hooks before the worker wakeup hook')
        self._stop_worker()
        self.engine.pending_requests = self.original_queue
        if self.had_loop:
            self.engine._background_loop = self.original_loop
        else:
            del self.engine.__dict__['_background_loop']
        if self.had_prefill:
            self.runner.prefill = self.original_prefill
        else:
            del self.runner.__dict__['prefill']
        self.closed = True
        self._start_worker()

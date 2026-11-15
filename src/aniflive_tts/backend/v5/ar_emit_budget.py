"""Scoped suppression of semantic progress prefixes unused by fixed v5 windows."""
from contextlib import contextmanager
import math


@contextmanager
def defer_inactive_partials(acceleration, req, *, first_seconds, events):
    """Defer only initial partial emissions; final EOS is always native.

    The caller owns the engine exclusively and closes the semantic iterator
    before leaving this scope. An unready lazy engine remains on native output.
    """
    if first_seconds is not None or req.get('streaming_window_mode', 'fixed') != 'fixed' or req['min_chunk_length'] != 16:
        raise ValueError('Semantic emission budget requires native fixed windows and interval 16')
    core = max(1, round(req['streaming_chunk_seconds'] * 100))
    future = round(req['streaming_lookahead_seconds'] * 100)
    holdback = max(16, round(req['gpt_streaming_lookahead_seconds'] * 100))
    needed = math.ceil((core + future + holdback) * req['speed_factor'] / 4)
    first_partial = math.ceil(needed / 16) * 16
    stats = {'stage': 'semantic_emission_budget', 'enabled': False, 'needed_tokens': needed,
             'first_partial_tokens': first_partial, 'interval': 16, 'skipped_partial_emissions': 0,
             'first_emitted_tokens': None}
    if acceleration is None or acceleration.engine is None:
        stats['dispatch'] = 'native_engine_unready'
        try:
            yield stats
        finally:
            events.append(stats)
        return
    engine = acceleration.engine
    runner = engine.model_runner
    if (runner.max_batch_size != 1 or runner.bound_sessions or runner.waiting_bind or any(runner.slot_sessions)
            or engine.handles or not engine.pending_requests.empty()):
        raise RuntimeError('Semantic emission budget requires an idle single-slot AR engine')
    original = engine._maybe_stream_partial
    had_instance = '_maybe_stream_partial' in engine.__dict__
    previous_instance = engine.__dict__.get('_maybe_stream_partial')
    def limited(session):
        if (session.bsz != 1 or session.stream_interval != 16 or len(runner.bound_sessions) != 1
                or runner.bound_sessions[0] is not session or runner.waiting_bind):
            raise RuntimeError('Semantic emission budget encountered an unsupported AR session')
        if session.last_stream_step == 0 and session.step_count < first_partial:
            if session.step_count > 0 and session.step_count % 16 == 0:
                stats['skipped_partial_emissions'] += 1
            return
        previous = session.last_stream_step
        original(session)
        if previous == 0 and session.last_stream_step > 0:
            stats['first_emitted_tokens'] = session.last_stream_step
    engine._maybe_stream_partial = limited
    stats.update(enabled=True, dispatch='defer_unused_initial_partials')
    try:
        yield stats
    finally:
        # The semantic iterator has already closed/cancelled; on return the
        # single-slot owner must have unbound the session before hook removal.
        try:
            if (acceleration.engine is not engine or runner.bound_sessions or runner.waiting_bind
                    or any(runner.slot_sessions) or engine.handles or not engine.pending_requests.empty()):
                raise RuntimeError('AR engine changed or did not quiesce before emission hook removal')
            if engine._maybe_stream_partial is not limited:
                raise RuntimeError('Semantic emission hook changed externally')
        finally:
            if had_instance:
                engine._maybe_stream_partial = previous_instance
            else:
                del engine._maybe_stream_partial
            events.append(stats)

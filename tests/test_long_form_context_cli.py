from __future__ import annotations

import importlib.util
from pathlib import Path


def _module():
    path = (
        Path(__file__).resolve().parents[1]
        / "scripts"
        / "benchmark_long_form_context.py"
    )
    spec = importlib.util.spec_from_file_location("benchmark_long_form_context", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_af_report_is_diagnostic_and_checks_fixed_seed_determinism() -> None:
    module = _module()
    calls: dict[str, int] = {}

    def runner(policy: str):
        calls[policy] = calls.get(policy, 0) + 1
        return {
            "pcm_sha256": policy if policy != "F" else f"F-{calls[policy]}",
            "first_packet_ms": 10.0 + calls[policy],
            "audible_ttfa_ms": 20.0 + calls[policy],
        }

    report = module.build_report(
        policies=module.POLICIES,
        repetitions=2,
        runner=runner,
    )

    assert report["diagnostic_only"] is True
    assert report["release_qualified"] is False
    assert report["acoustic_latent_continuity"] is False
    assert report["policies"]["A"]["fixed_seed_deterministic"] is True
    assert report["policies"]["F"]["fixed_seed_deterministic"] is False
    assert set(report["policies"]) == set(module.POLICIES)


def test_audible_pcm_detection_is_signed_little_endian() -> None:
    module = _module()

    assert module._pcm_has_audible_sample(b"\x00\x00\xff\x00", 256) is False
    assert module._pcm_has_audible_sample(b"\x00\x00\x00\x02", 256) is True

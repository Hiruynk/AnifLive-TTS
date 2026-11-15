from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

release_preflight = importlib.import_module("scripts.release_preflight")


def _summary() -> dict[str, object]:
    return {
        "release": "1.5.0",
        "main_comparison_complete": False,
        "remaining": [
            "Mac source and qualification handoff",

        ],
        "method": {
            "measured_requests_per_condition": 1000,
            "measured_total": 3000,
        },
        "results": [
            {
                "status": "verified_windows_formal_performance",
                "platform": "Windows/NVIDIA Docker",
                "model_family": "gsv-v5turbo",
                "workload": "keepalive_pcm",
                "audible_ttfa_p50_ms": 104.23555066641225,
                "audible_ttfa_p95_ms": 111.07985766407606,
                "rtf_p50": 0.061488937460568195,
                "rtf_p95": 0.06613768388475691,
            },
            {
                "status": "verified_windows_formal_performance",
                "platform": "Windows/NVIDIA Docker",
                "model_family": "gsv-v5turbo",
                "workload": "new_connection_pcm",
                "audible_ttfa_p50_ms": 106.63717316797798,
                "audible_ttfa_p95_ms": 114.95082466476984,
                "rtf_p50": 0.06079320317413817,
                "rtf_p95": 0.0668608494097594,
            },
            {
                "status": "verified_windows_formal_performance",
                "model_family": "gsv-v2proplus",
                "workload": "keepalive_pcm",
                "audible_ttfa_p50_ms": 103.33296949807845,
                "audible_ttfa_p95_ms": 127.42086600133917,
                "rtf_p50": 0.1112686696313976,
                "rtf_p95": 0.1378323687609114,
            },
            {
                "platform": "macOS",
                "verified": False,
                "status": "awaiting_private_macos_handoff",
            },
            {
                "system": "gsv.cpp",
                "status": "complete_multi_session_exploratory",
                "formal_10round": False,
            },
        ],
    }


def _four_comparison_tables(v5_ttfa: str, v5_rtf: str) -> str:
    return f"""### GPT-SoVITS Performance Comparison
#### First-output latency (lower is faster)
| Repository / system | Metric | Latency | Test conditions | Source |
|---|---|---:|---|---|
| AnifLive-TTS v1.5.0 v2ProPlus | Audible TTFA P50 | 103.33 ms | RTX 5070 Ti; Windows; keep-alive PCM | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5.0 v5Turbo | Audible TTFA P50 | {v5_ttfa} ms | RTX 5070 Ti; Windows; keep-alive PCM | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| gsv.cpp v0.1.3 · v5Turbo | Audible TTFA P50 | 354.45 ms | RTX 5070 Ti; Vulkan; exploratory | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5.0 v5Turbo (M5 Max) | Audible TTFA P50 | 404.69 ms | Mac Studio M5 Max; macOS 27.0; MLX/Metal | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |

#### RTF (lower is faster)
| Repository / system | RTF | Backend | Test conditions | Source |
|---|---:|---|---|---|
| AnifLive-TTS v1.5.0 v5Turbo | {v5_rtf} | TensorRT 11 / Triton FP16 | RTX 5070 Ti; Windows; keep-alive PCM | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5.0 v2ProPlus | 0.11127 | TensorRT 11 FP16 | RTX 5070 Ti; Windows; keep-alive PCM | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| gsv.cpp v0.1.3 · v5Turbo | 0.11197 | Vulkan; F16 AR / F32 vocoder | RTX 5070 Ti; exploratory | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5.0 v5Turbo (M5 Max) | 0.32357 | Rust / MLX / Metal FP16 | Mac Studio M5 Max; macOS 27.0 | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |

### Public Performance Data from Other Open-Source TTS Systems
#### First-audio latency (lower is faster)
| System | Metric | Latency | Statistic | Test conditions | Source |
|---|---|---:|---|---|---|
| AnifLive-TTS v1.5.0 v2ProPlus | Audible TTFA | 103.33 ms | P50 | RTX 5070 Ti; Windows; keep-alive PCM | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5.0 v5Turbo | Audible TTFA | {v5_ttfa} ms | P50 | RTX 5070 Ti; Windows; keep-alive PCM | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |

#### RTF (lower is faster)
| System | RTF | Runtime / model | Test conditions | Source |
|---|---:|---|---|---|
| AnifLive-TTS v1.5.0 v5Turbo | {v5_rtf} | TensorRT 11 / Triton FP16 | RTX 5070 Ti; Windows | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
| AnifLive-TTS v1.5.0 v2ProPlus | 0.11127 | TensorRT 11 FP16 | RTX 5070 Ti; Windows | [Data](benchmarks/V1_5_BENCHMARK_SUMMARY.json) |
"""

class ReleasePreflightTests(unittest.TestCase):
    def test_secret_findings_are_redacted(self) -> None:
        secret = "sk-live-" + ("Q" * 32)
        findings = release_preflight.scan_text_content(
            "sample/config.txt", f"ANIFLIVE_TTS_API_KEY={secret}"
        )
        serialized = json.dumps(findings)
        self.assertTrue(findings)
        self.assertNotIn(secret, serialized)
        self.assertTrue(all(item.get("redacted") for item in findings))

    def test_forbidden_audio_payload_is_reported(self) -> None:
        finding = release_preflight.inspect_payload(
            "data/private.wav", b"RIFF" + b"\x00" * 4 + b"WAVE"
        )
        self.assertIsNotNone(finding)
        self.assertEqual(finding["code"], "audio_payload")
        self.assertNotIn("RIFF", json.dumps(finding))

    def test_missing_markdown_target_fails_closed(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "README.md").write_text("[missing](not-present.md)\n", encoding="utf-8")
            result = release_preflight.check_markdown_links(root, ["README.md"])
        self.assertEqual(result["status"], "failed")
        self.assertEqual(result["findings"][0]["code"], "local_link_target_missing")

    def test_readme_values_are_rounded_from_public_summary(self) -> None:
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            summary = _summary()
            summary["results"][0]["system"] = "AnifLive-TTS v1.5.0"
            summary["results"][1]["system"] = "AnifLive-TTS v1.5.0"
            summary["results"][2].update(
                platform="Windows/NVIDIA Docker", system="AnifLive-TTS v1.5.0"
            )
            summary["results"][-1].update(
                platform="Windows/NVIDIA Vulkan",
                model_family="gsv-v5turbo",
                condition="keepalive_pcm",
                audible_ttfa_p50_ms=354.45,
                rtf_p50=0.11197,
            )
            summary["results"].extend(
                [
                    {
                        "status": "verified_macos_formal_performance",
                        "platform": "macOS / Apple Silicon",
                        "system": "AnifLive-TTS v1.5.0",
                        "model_family": "gsv-v5turbo",
                        "workload": "new_connection_pcm",
                        "audible_ttfa_p50_ms": 404.67,
                        "audible_ttfa_p95_ms": 410.38,
                        "rtf_p50": 0.32342,
                        "rtf_p95": 0.33410,
                    },
                    {
                        "status": "verified_macos_formal_performance",
                        "platform": "macOS / Apple Silicon",
                        "system": "AnifLive-TTS v1.5.0",
                        "model_family": "gsv-v5turbo",
                        "workload": "keepalive_pcm",
                        "audible_ttfa_p50_ms": 404.69,
                        "audible_ttfa_p95_ms": 410.78,
                        "rtf_p50": 0.32357,
                        "rtf_p95": 0.33464,
                    },
                ]
            )
            detailed = """Each backend/device configuration has 3,000 total per configuration.
| Model | Condition | First packet | Audible TTFA P50 / P95 | Full response | RTF P50 / P95 | Requests |
|---|---|---:|---:|---:|---:|---:|
| V5Turbo | PCM, keep-alive | 90.19 / 97.04 | **104.24 / 111.08** | 98.81 / 106.28 | 0.06149 / 0.06614 | 1,000 |
| V5Turbo | PCM, new connection | 92.60 / 100.91 | 106.64 / 114.95 | 97.69 / 107.45 | 0.06079 / 0.06686 | 1,000 |
| V5Turbo (M5 Max) | PCM, new connection | 274.13 / 279.84 | 404.67 / 410.38 | 542.37 / 560.29 | 0.32342 / 0.33410 | 1,000 |
| V5Turbo (M5 Max) | PCM, keep-alive | 274.15 / 280.24 | 404.69 / 410.78 | 542.63 / 561.18 | 0.32357 / 0.33464 | 1,000 |
"""
            compact = _four_comparison_tables("104.24", "0.06149")
            (root / "docs").mkdir()
            (root / "docs" / "BENCHMARKS.md").write_text(detailed, encoding="utf-8")
            (root / "README.md").write_text(compact, encoding="utf-8")
            result = release_preflight.check_readme_benchmark_values(root, summary, ["README.md"])
        self.assertEqual(result["status"], "passed")
        self.assertEqual(result["detailed_rows_checked"], 4)
        self.assertEqual(result["comparison_rows_checked"], 12)
        self.assertEqual(result["rows_checked"], 16)

    def test_configuration_count_markers_accept_english_and_chinese(self) -> None:
        for text in (
            "3,000 total per configuration",
            "每個配置合共 3,000 次",
            "每个配置共 3,000 次",
        ):
            self.assertTrue(release_preflight._has_per_configuration_count(text, 3000))
        self.assertFalse(release_preflight._has_per_configuration_count("3,000 total per model", 3000))

    def test_mixed_windows_macos_rows_use_their_own_verified_records(self) -> None:
        summary = _summary()
        summary["results"] = [
            item for item in summary["results"] if item.get("status") != "awaiting_private_macos_handoff"
        ]
        windows_values = {
            "new_connection_pcm": (82.22, 85.75, 0.04566, 0.04918),
            "keepalive_pcm": (80.93, 84.33, 0.04728, 0.04974),
        }
        for item in summary["results"]:
            if item.get("model_family") == "gsv-v5turbo" and item.get("workload") in windows_values:
                p50, p95, rtf_p50, rtf_p95 = windows_values[item["workload"]]
                item.update(
                    system="AnifLive-TTS v1.5.0",
                    audible_ttfa_p50_ms=p50,
                    audible_ttfa_p95_ms=p95,
                    rtf_p50=rtf_p50,
                    rtf_p95=rtf_p95,
                )
        summary["results"][2].update(
            platform="Windows/NVIDIA Docker", system="AnifLive-TTS v1.5.0"
        )
        summary["results"][-1].update(
            model_family="gsv-v5turbo",
            condition="keepalive_pcm",
            audible_ttfa_p50_ms=354.45,
            rtf_p50=0.11197,
        )
        summary["results"].extend(
            [
                {
                    "platform": "macOS / Apple Silicon",
                    "status": "verified_macos_formal_performance",
                    "system": "AnifLive-TTS v1.5.0",
                    "model_family": "gsv-v5turbo",
                    "workload": "new_connection_pcm",
                    "audible_ttfa_p50_ms": 404.67,
                    "audible_ttfa_p95_ms": 410.38,
                    "rtf_p50": 0.32342,
                    "rtf_p95": 0.33410,
                },
                {
                    "platform": "macOS / Apple Silicon",
                    "status": "verified_macos_formal_performance",
                    "system": "AnifLive-TTS v1.5.0",
                    "model_family": "gsv-v5turbo",
                    "workload": "keepalive_pcm",
                    "audible_ttfa_p50_ms": 404.69,
                    "audible_ttfa_p95_ms": 410.78,
                    "rtf_p50": 0.32357,
                    "rtf_p95": 0.33464,
                },
            ]
        )
        detailed = """Each backend/device configuration has 3,000 total per configuration.
| Model | Condition | First packet | Audible TTFA P50 / P95 | Full response | RTF P50 / P95 | Requests |
|---|---|---:|---:|---:|---:|---:|
| V5Turbo | PCM, new connection | 68.18 / 71.71 | 82.22 / 85.75 | 73.37 / 79.04 | 0.04566 / 0.04918 | 1,000 |
| V5Turbo | PCM, keep-alive | 66.89 / 70.29 | 80.93 / 84.33 | 75.98 / 79.93 | 0.04728 / 0.04974 | 1,000 |
| V5Turbo (M5 Max) | PCM, new connection | 274.13 / 279.84 | 404.67 / 410.38 | 542.37 / 560.29 | 0.32342 / 0.33410 | 1,000 |
| V5Turbo (M5 Max) | PCM, keep-alive | 274.15 / 280.24 | 404.69 / 410.78 | 542.63 / 561.18 | 0.32357 / 0.33464 | 1,000 |
"""
        compact = _four_comparison_tables("80.93", "0.04728")
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            (root / "docs").mkdir()
            benchmark_path = root / "docs" / "BENCHMARKS.md"
            benchmark_path.write_text(detailed, encoding="utf-8")
            readme_path = root / "README.md"
            readme_path.write_text(compact, encoding="utf-8")
            result = release_preflight.check_readme_benchmark_values(root, summary, ["README.md"])

            self.assertEqual(result["status"], "passed")
            self.assertEqual(result["detailed_rows_checked"], 4)
            self.assertEqual(result["comparison_rows_checked"], 12)
            self.assertEqual(result["rows_checked"], 16)
            self.assertEqual(release_preflight.derive_release_gates(summary)["macos_handoff"], "passed")

            benchmark_path.write_text(
                detailed.replace("404.69 / 410.78", "405.69 / 410.78", 1),
                encoding="utf-8",
            )
            mismatched = release_preflight.check_readme_benchmark_values(root, summary, ["README.md"])

        self.assertEqual(mismatched["status"], "failed")
        self.assertTrue(
            any(
                finding.get("platform") == "macos"
                and finding.get("workload") == "keepalive_pcm"
                and finding.get("metric") == "audible_ttfa_p50_ms"
                for finding in mismatched["findings"]
            )
        )

    def test_mac_pending_keeps_overall_release_not_ready(self) -> None:
        summary = _summary()
        gates = release_preflight.derive_release_gates(summary)
        self.assertEqual(gates["macos_handoff"], "pending")
        self.assertEqual(gates["fresh_v2_benchmark"], "passed")
        self.assertEqual(gates["local_gsv_cpp_benchmark"], "passed")
        self.assertFalse(
            release_preflight.is_release_ready(
                summary,
                {"markdown_links": "passed", "source_hygiene": "passed"},
                gates,
            )
        )

    def test_fixed_topology_audit_exposes_padding_sentinel_review(self) -> None:
        result = release_preflight.audit_onnx_topologies(ROOT)
        self.assertEqual(result["topology_count"], 4)
        self.assertTrue(
            all(
                item["dense_initializers"] == 0 and item["sparse_initializers"] == 0
                for item in result["topologies"]
            )
        )
        dit = next(item for item in result["topologies"] if item["path"].endswith("dit-topology.onnx"))
        sentinel = next(
            item
            for item in dit["float_constant_evidence"]
            if item["classification"] == "recognized_native_padding_sentinel"
        )
        self.assertEqual(sentinel["sha256"], release_preflight.DIT_PADDING_SENTINEL_SHA256)
        self.assertEqual(sentinel["where_contexts"][0]["graph_outputs"], ["velocity"])
        self.assertEqual(result["status"], "passed")

    def test_cross_family_gate_requires_real_roundtrip_evidence(self) -> None:
        summary = _summary()
        summary["quality_and_integration"] = {"switching_checks": {
            "cross_family_evidence": {
                "status": "verified_real_http_roundtrip",
                "sequence": ["gsv-v5turbo", "gsv-v2proplus", "gsv-v5turbo"],
                "synthesis_after_each_switch": True,
                "roundtrip_pcm_exact": True,
                "source_record": "deep-roundtrip.json",
            }
        }}
        self.assertEqual(release_preflight.derive_release_gates(summary)["cross_family_model_switch"], "passed")
        summary["quality_and_integration"]["switching_checks"]["cross_family_evidence"]["roundtrip_pcm_exact"] = False
        self.assertEqual(release_preflight.derive_release_gates(summary)["cross_family_model_switch"], "pending")

    def test_declared_functional_gap_cannot_be_overridden_by_benchmarks(self) -> None:
        summary = _summary()
        summary["release_ready"] = False
        gates = release_preflight.derive_release_gates(summary)
        self.assertEqual(gates["functional_release_candidate"], "pending")


if __name__ == "__main__":
    unittest.main()

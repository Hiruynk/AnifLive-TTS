from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np

from aniflive_tts.dataset_factory import DatasetFactory
from aniflive_tts.dataset_factory_api import create_dataset_factory_router


def _write_wav(path: Path, *, seconds: float = 0.4, sample_rate: int = 16_000) -> None:
    time = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    samples = np.rint(0.3 * np.sin(2.0 * np.pi * 220.0 * time) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())


class DatasetFactoryApiTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.root = Path(self.temporary.name)
        self.source = self.root / "source"
        self.source.mkdir()
        self.audio = self.source / "voice.wav"
        _write_wav(self.audio)
        self.factory = DatasetFactory(
            self.root / "state", allowed_source_roots=(self.source,)
        )
        app = FastAPI()
        app.include_router(create_dataset_factory_router(lambda: self.factory))
        self.client = TestClient(app)

    def tearDown(self) -> None:
        self.client.close()
        self.temporary.cleanup()

    def test_complete_rest_pipeline_produces_reviewed_split_manifest(self) -> None:
        ingested = self.client.post(
            "/api/workstation/datasets/dataset_voice/ingest",
            json={"sources": [str(self.audio)]},
        )
        self.assertEqual(ingested.status_code, 200, ingested.text)
        source_item = ingested.json()["items"][0]

        resampled = self.client.post(
            f"/api/workstation/datasets/dataset_voice/items/{source_item['id']}/resample",
            json={"target_rate": 32_000, "mono": True},
        )
        self.assertEqual(resampled.status_code, 200, resampled.text)
        canonical = resampled.json()
        self.assertEqual(canonical["sample_rate"], 32_000)

        vad = self.client.post(
            f"/api/workstation/datasets/dataset_voice/items/{canonical['id']}/vad",
            json={"threshold_dbfs": -38.0},
        )
        self.assertEqual(vad.status_code, 200, vad.text)
        self.assertEqual(len(vad.json()["speech_regions"]), 1)

        segmented = self.client.post(
            f"/api/workstation/datasets/dataset_voice/items/{canonical['id']}/segments",
            json={"threshold_dbfs": -38.0},
        )
        self.assertEqual(segmented.status_code, 200, segmented.text)
        segment = segmented.json()["items"][0]

        reviewed = self.client.patch(
            f"/api/workstation/datasets/dataset_voice/items/{segment['id']}/review",
            json={"decision": "accepted", "note": "clear"},
        )
        self.assertEqual(reviewed.status_code, 200, reviewed.text)
        self.assertEqual(reviewed.json()["review_status"], "accepted")

        split = self.client.post(
            "/api/workstation/datasets/dataset_voice/split",
            json={"train": 1.0, "validation": 0.0, "test": 0.0, "seed": "fixed"},
        )
        self.assertEqual(split.status_code, 200, split.text)
        self.assertEqual(split.json()["item_counts"]["train"], 1)

        manifest = self.client.get(
            "/api/workstation/datasets/dataset_voice/manifest"
        )
        self.assertEqual(manifest.status_code, 200, manifest.text)
        self.assertEqual(manifest.json()["item_count"], 1)
        self.assertEqual(manifest.json()["items"][0]["split"], "train")

        listing = self.client.get(
            "/api/workstation/datasets/dataset_voice/items",
            params={"kind": "segment", "review_status": "accepted"},
        )
        self.assertEqual(listing.status_code, 200, listing.text)
        self.assertEqual(listing.json()["count"], 1)

    def test_api_rejects_cross_dataset_item_and_invalid_split(self) -> None:
        item = self.factory.ingest("dataset_voice", (self.audio,))[0]

        wrong_dataset = self.client.post(
            f"/api/workstation/datasets/dataset_other/items/{item['id']}/resample",
            json={},
        )
        self.assertEqual(wrong_dataset.status_code, 404)

        bad_split = self.client.post(
            "/api/workstation/datasets/dataset_voice/split",
            json={"train": 0.8, "validation": 0.3, "test": 0.0},
        )
        self.assertEqual(bad_split.status_code, 422)
        self.assertIn("sum to 1", bad_split.json()["detail"])

    def test_annotation_api_accepts_bounded_expression_direction(self) -> None:
        item = self.factory.ingest("dataset_voice", (self.audio,))[0]
        response = self.client.patch(
            f"/api/workstation/datasets/dataset_voice/items/{item['id']}/annotations",
            json={
                "expression": "shy",
                "expression_intensity": 0.76,
                "valence": 0.2,
                "arousal": -0.3,
                "dominance": -0.5,
                "style_description": "Quiet and hesitant.",
                "propagate": False,
            },
        )
        self.assertEqual(response.status_code, 200, response.text)
        annotations = response.json()["annotations"]
        self.assertEqual(annotations["expression_intensity"], 0.76)
        self.assertEqual(annotations["style_description"], "Quiet and hesitant.")

        invalid = self.client.patch(
            f"/api/workstation/datasets/dataset_voice/items/{item['id']}/annotations",
            json={"expression_intensity": 1.1},
        )
        self.assertEqual(invalid.status_code, 422)

    def test_annotation_verification_api_binds_evidence_to_current_value(self) -> None:
        source = self.factory.ingest("dataset_voice", (self.audio,))[0]
        canonical = self.factory.resample(source["id"])
        item = self.factory.segment(canonical["id"])[0]
        annotated = self.client.patch(
            f"/api/workstation/datasets/dataset_voice/items/{item['id']}/annotations",
            json={"transcript": "今日はいい天気ですね。", "language": "ja"},
        )
        self.assertEqual(annotated.status_code, 200, annotated.text)

        verified = self.client.post(
            f"/api/workstation/datasets/dataset_voice/items/{item['id']}"
            "/annotation-verifications",
            json={"kind": "transcript", "decision": "verified", "note": "heard"},
        )
        self.assertEqual(verified.status_code, 200, verified.text)
        self.assertTrue(verified.json()["verification"]["transcript"]["valid"])

        summary = self.client.get(
            "/api/workstation/datasets/dataset_voice/review-summary"
        )
        self.assertEqual(summary.status_code, 200, summary.text)
        self.assertEqual(summary.json()["annotations"]["transcript_pending"], 0)

        changed = self.client.patch(
            f"/api/workstation/datasets/dataset_voice/items/{item['id']}/annotations",
            json={"transcript": "今日は良い天気ですね。"},
        )
        self.assertEqual(changed.status_code, 200, changed.text)
        self.assertFalse(changed.json()["verification"]["transcript"]["valid"])

        history = self.client.get(
            f"/api/workstation/datasets/dataset_voice/items/{item['id']}"
            "/annotation-verifications"
        )
        self.assertEqual(history.status_code, 200, history.text)
        self.assertEqual(history.json()["count"], 1)
        self.assertFalse(history.json()["verification"]["transcript"]["valid"])


if __name__ == "__main__":
    unittest.main()

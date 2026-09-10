from __future__ import annotations

from pathlib import Path
import tempfile
import unittest
import wave

from fastapi import FastAPI
from fastapi.testclient import TestClient
import numpy as np

from aniflive_tts.dataset_factory import DatasetFactory, DatasetFactoryError
from aniflive_tts.dataset_factory_api import create_dataset_factory_router


def _write_wav(path: Path, *, seconds: float = 0.6, sample_rate: int = 16_000) -> None:
    timeline = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    samples = np.rint(0.28 * np.sin(2 * np.pi * 245 * timeline) * 32767).astype("<i2")
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(1)
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(samples.tobytes())


class DatasetFactoryPhaseTwoTests(unittest.TestCase):
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

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_list_import_persists_annotations_quality_and_manifest_metadata(self) -> None:
        metadata = self.source / "voice.list"
        metadata.write_text(
            "voice.wav|Roxy Migurdia|ja|今日は穏やかな日ですね。\n",
            encoding="utf-8",
        )

        result = self.factory.import_gpt_sovits_list("dataset_voice", metadata)

        self.assertEqual(result["entry_count"], 1)
        source_item = result["items"][0]
        self.assertEqual(source_item["annotations"]["language"], "ja")
        self.assertEqual(source_item["annotations"]["speaker"], "Roxy Migurdia")
        self.assertEqual(source_item["annotations"]["source"], "gpt-sovits-list")
        self.assertEqual(source_item["quality"]["method"], "deterministic-pcm-v1")
        self.assertEqual(source_item["stage_state"]["annotation"]["state"], "complete")
        self.assertEqual(source_item["stage_state"]["quality"]["state"], "complete")

        self.assertEqual(source_item["kind"], "segment")
        self.assertIsNotNone(source_item["parent_item_id"])
        segment = source_item
        copied = self.factory.root / segment["stored_path"]
        self.assertEqual(copied.read_bytes(), self.audio.read_bytes())
        repeated = self.factory.import_gpt_sovits_list("dataset_voice", metadata)
        self.assertEqual(repeated["items"][0]["id"], segment["id"])
        self.assertEqual(segment["annotations"]["transcript"], "今日は穏やかな日ですね。")
        self.factory.review(segment["id"], decision="accepted")
        self.factory.assign_splits(
            "dataset_voice", train=1.0, validation=0.0, test=0.0
        )
        record = self.factory.manifest("dataset_voice")["items"][0]
        self.assertEqual(record["annotations"]["speaker"], "Roxy Migurdia")
        self.assertGreater(record["quality"]["quality_score"], 0)

    def test_list_import_rejects_conflicting_annotations_for_identical_audio(self) -> None:
        duplicate = self.source / "same.wav"
        duplicate.write_bytes(self.audio.read_bytes())
        metadata = self.source / "conflict.list"
        metadata.write_text(
            "voice.wav|speaker-a|en|First line.\n"
            "same.wav|speaker-b|en|Different line.\n",
            encoding="utf-8",
        )

        with self.assertRaisesRegex(DatasetFactoryError, "identical audio content"):
            self.factory.import_gpt_sovits_list("dataset_voice", metadata)

        self.assertEqual(self.factory.list_items("dataset_voice"), [])

    def test_annotations_propagate_and_waveform_is_real_pcm(self) -> None:
        source_item = self.factory.ingest("dataset_voice", (self.audio,))[0]
        segment = self.factory.segment(source_item["id"])[0]

        updated = self.factory.update_annotations(
            source_item["id"],
            {
                "transcript": "Hello from AnifLive-TTS.",
                "language": "English",
                "speaker": "voice-one",
                "expression": "calm",
            },
        )

        self.assertEqual(updated["annotations"]["language"], "en")
        self.assertEqual(self.factory.get_item(segment["id"])["annotations"]["expression"], "calm")
        waveform = self.factory.waveform(segment["id"], bins=64)
        self.assertEqual(waveform["schema"], "aniflive-dataset-waveform-v1")
        self.assertEqual(len(waveform["peaks"]), 64)
        self.assertTrue(any(low < 0 < high for low, high in waveform["peaks"]))
        self.assertEqual(
            self.factory.verified_audio_path(segment["id"]).read_bytes()[:4], b"RIFF"
        )

    def test_multi_region_segmentation_requires_per_segment_transcripts(self) -> None:
        sample_rate = 16_000
        silence = np.zeros(round(0.35 * sample_rate), dtype=np.float32)
        timeline = np.arange(round(0.35 * sample_rate), dtype=np.float32) / sample_rate
        tone = 0.3 * np.sin(2 * np.pi * 250 * timeline)
        multi = self.source / "multi.wav"
        values = np.concatenate((tone, silence, tone))
        pcm = np.rint(values * 32767).astype("<i2")
        with wave.open(str(multi), "wb") as audio:
            audio.setnchannels(1)
            audio.setsampwidth(2)
            audio.setframerate(sample_rate)
            audio.writeframes(pcm.tobytes())
        source_item = self.factory.ingest("dataset_voice", (multi,))[0]
        self.factory.update_annotations(
            source_item["id"],
            {"transcript": "Two utterances", "language": "en", "speaker": "voice"},
        )

        segments = self.factory.segment(source_item["id"])

        self.assertEqual(len(segments), 2)
        self.assertTrue(all(item["annotations"]["transcript"] is None for item in segments))
        self.assertTrue(all(item["annotations"]["speaker"] == "voice" for item in segments))
        self.assertTrue(
            all(
                item["annotations"]["source"] == "inherited-requires-transcript"
                for item in segments
            )
        )

    def test_unsupported_neural_and_decoder_stages_are_fail_closed(self) -> None:
        compressed = self.source / "voice.mp3"
        compressed.write_bytes(b"not decoded")
        item = self.factory.ingest("dataset_voice", (compressed,))[0]

        capabilities = self.factory.capabilities()["stages"]
        self.assertTrue(capabilities["decode"]["available"])
        self.assertEqual(capabilities["decode"]["job_type"], "dataset.process")
        self.assertTrue(capabilities["denoise"]["available"])
        self.assertEqual(capabilities["denoise"]["job_type"], "dataset.process")
        self.assertFalse(capabilities["dereverb"]["available"])
        self.assertEqual(capabilities["dereverb"]["configured_backend"], "none")
        self.assertTrue(capabilities["asr"]["available"])
        self.assertEqual(capabilities["asr"]["job_type"], "dataset.process")
        self.assertEqual(capabilities["asr"]["required_input"], "asr_model")
        self.assertEqual(
            self.factory.capabilities()["jobs"]["dataset.process"]["resource_class"],
            "gpu-exclusive",
        )
        self.assertEqual(capabilities["tse"]["mode"], "delegated")
        self.assertEqual(item["stage_state"]["decode"]["state"], "blocked")
        with self.assertRaisesRegex(DatasetFactoryError, "trusted decoder"):
            self.factory.waveform(item["id"])

    def test_phase_two_rest_surface_returns_audio_waveform_and_annotations(self) -> None:
        app = FastAPI()
        app.include_router(create_dataset_factory_router(lambda: self.factory))
        with TestClient(app) as client:
            source_item = self.factory.ingest("dataset_voice", (self.audio,))[0]
            base = f"/api/workstation/datasets/dataset_voice/items/{source_item['id']}"

            annotation = client.patch(
                f"{base}/annotations",
                json={"transcript": "안녕하세요.", "language": "ko", "speaker": "voice"},
            )
            self.assertEqual(annotation.status_code, 200, annotation.text)
            self.assertEqual(annotation.json()["annotations"]["language"], "ko")

            quality = client.post(f"{base}/quality")
            self.assertEqual(quality.status_code, 200, quality.text)
            self.assertGreater(quality.json()["quality"]["quality_score"], 0)

            waveform = client.get(f"{base}/waveform", params={"bins": 48})
            self.assertEqual(waveform.status_code, 200, waveform.text)
            self.assertEqual(len(waveform.json()["peaks"]), 48)

            audio = client.get(f"{base}/audio")
            self.assertEqual(audio.status_code, 200, audio.text)
            self.assertEqual(audio.headers["content-type"], "audio/wav")
            self.assertEqual(audio.content[:4], b"RIFF")

            capabilities = client.get("/api/workstation/datasets/capabilities")
            self.assertEqual(capabilities.status_code, 200, capabilities.text)
            self.assertTrue(capabilities.json()["stages"]["asr"]["available"])
            self.assertEqual(
                capabilities.json()["stages"]["asr"]["required_input"],
                "asr_model",
            )

            metadata = self.source / "voice.list"
            metadata.write_text(
                "voice.wav|voice|English|A clean line.\n", encoding="utf-8"
            )
            imported = client.post(
                "/api/workstation/datasets/dataset_list/import-list",
                json={"path": str(metadata)},
            )
            self.assertEqual(imported.status_code, 200, imported.text)
            self.assertEqual(imported.json()["items"][0]["annotations"]["language"], "en")


if __name__ == "__main__":
    unittest.main()

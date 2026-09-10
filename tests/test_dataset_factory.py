from __future__ import annotations

from contextlib import closing
import hashlib
import json
from pathlib import Path
import sqlite3
import tempfile
import unittest
import wave

import numpy as np

from aniflive_tts.dataset_factory import (
    DATASET_FACTORY_SCHEMA,
    DatasetFactory,
    DatasetFactoryError,
    EnergyVadConfig,
)


def _write_wav(
    path: Path,
    samples: np.ndarray,
    *,
    sample_rate: int = 16_000,
) -> None:
    values = np.asarray(samples, dtype=np.float32)
    if values.ndim == 1:
        values = values[:, None]
    pcm = np.rint(np.clip(values, -1.0, 1.0) * 32767.0).astype("<i2")
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as audio:
        audio.setnchannels(values.shape[1])
        audio.setsampwidth(2)
        audio.setframerate(sample_rate)
        audio.writeframes(pcm.tobytes())


def _tone(seconds: float, *, sample_rate: int = 16_000, frequency: float = 220.0) -> np.ndarray:
    time = np.arange(round(seconds * sample_rate), dtype=np.float32) / sample_rate
    return (0.32 * np.sin(2.0 * np.pi * frequency * time)).astype(np.float32)


class DatasetFactoryTests(unittest.TestCase):
    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.tmp_path = Path(self.temporary.name)

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def factory(self, source: Path) -> DatasetFactory:
        return DatasetFactory(self.tmp_path / "state", allowed_source_roots=(source,))

    def test_schema_and_ingest_are_persistent_verified_and_deduplicated(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        audio_path = source / "voice.wav"
        _write_wav(audio_path, _tone(0.25))
        factory = self.factory(source)

        first = factory.ingest("dataset_voice", (audio_path,))
        second = factory.ingest("dataset_voice", (audio_path,))

        self.assertEqual(first, second)
        self.assertEqual(first[0]["pipeline_state"], "ingested")
        self.assertIs(first[0]["metadata"]["processing_supported"], True)
        self.assertEqual(first[0]["sample_rate"], 16_000)
        stored = factory.root / first[0]["stored_path"]
        self.assertTrue(stored.is_file())
        self.assertEqual(stored.read_bytes(), audio_path.read_bytes())
        with closing(sqlite3.connect(factory.database_path)) as connection:
            schema = connection.execute(
                "SELECT value FROM metadata WHERE key = 'schema'"
            ).fetchone()
            tables = {
                row[0]
                for row in connection.execute(
                    "SELECT name FROM sqlite_master WHERE type = 'table'"
                )
            }
            item_columns = {
                row[1] for row in connection.execute("PRAGMA table_info(dataset_items)")
            }
        self.assertEqual(schema, (str(DATASET_FACTORY_SCHEMA),))
        self.assertLessEqual(
            {
                "dataset_items",
                "dataset_vad_regions",
                "dataset_reviews",
                "dataset_annotation_verifications",
            },
            tables,
        )
        self.assertLessEqual(
            {
                "start_frame",
                "end_frame",
                "rms_dbfs",
                "peak_dbfs",
                "speaker_similarity",
                "overlap_status",
                "review_status",
            },
            item_columns,
        )

    def test_ingest_honestly_marks_media_that_needs_a_decoder(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        compressed = source / "voice.mp3"
        compressed.write_bytes(b"not actually decoded")
        factory = self.factory(source)

        item = factory.ingest("dataset_voice", (compressed,))[0]

        self.assertEqual(item["pipeline_state"], "decoder-required")
        self.assertIs(item["metadata"]["processing_supported"], False)
        with self.assertRaisesRegex(DatasetFactoryError, "trusted decoder"):
            factory.resample(item["id"])

    def test_ingest_enforces_configured_import_roots(self) -> None:
        allowed = self.tmp_path / "allowed"
        outside = self.tmp_path / "outside"
        allowed.mkdir()
        outside.mkdir()
        outside_audio = outside / "private.wav"
        _write_wav(outside_audio, _tone(0.1))
        factory = self.factory(allowed)

        with self.assertRaisesRegex(DatasetFactoryError, "outside"):
            factory.ingest("dataset_voice", (outside_audio,))

    def test_explicit_empty_import_roots_disable_ingest(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        audio_path = source / "voice.wav"
        _write_wav(audio_path, _tone(0.1))
        factory = DatasetFactory(self.tmp_path / "state", allowed_source_roots=())

        with self.assertRaisesRegex(DatasetFactoryError, "disabled"):
            factory.ingest("dataset_voice", (audio_path,))

    def test_ingest_rejects_linked_input(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        target = source / "target.wav"
        linked = source / "linked.wav"
        _write_wav(target, _tone(0.1))
        try:
            linked.symlink_to(target)
        except OSError:
            self.skipTest("This host does not permit symbolic links")
        factory = self.factory(source)

        with self.assertRaisesRegex(DatasetFactoryError, "Links and reparse"):
            factory.ingest("dataset_voice", (linked,))

    def test_resample_creates_real_mono_pcm_and_is_idempotent(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        stereo = np.column_stack((_tone(0.5), _tone(0.5, frequency=330.0)))
        audio_path = source / "stereo.wav"
        _write_wav(audio_path, stereo)
        factory = self.factory(source)
        source_item = factory.ingest("dataset_voice", (audio_path,))[0]

        output = factory.resample(source_item["id"], target_rate=32_000, mono=True)
        repeated = factory.resample(source_item["id"], target_rate=32_000, mono=True)

        self.assertEqual(output, repeated)
        self.assertEqual(output["kind"], "resampled")
        self.assertEqual(output["sample_rate"], 32_000)
        self.assertEqual(output["channels"], 1)
        self.assertEqual(output["frame_count"], 16_000)
        self.assertAlmostEqual(output["duration_seconds"], 0.5)
        with wave.open(str(factory.root / output["stored_path"]), "rb") as audio:
            self.assertEqual(audio.getframerate(), 32_000)
            self.assertEqual(audio.getnchannels(), 1)
            self.assertEqual(audio.getnframes(), 16_000)

    def test_energy_vad_segments_two_utterances_and_persists_regions(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        silence = np.zeros(round(0.35 * 16_000), dtype=np.float32)
        waveform = np.concatenate((silence, _tone(0.45), silence, _tone(0.40), silence))
        audio_path = source / "two-utterances.wav"
        _write_wav(audio_path, waveform)
        factory = self.factory(source)
        source_item = factory.ingest("dataset_voice", (audio_path,))[0]
        canonical = factory.resample(source_item["id"], target_rate=16_000)
        config = EnergyVadConfig(
            threshold_dbfs=-38.0,
            min_speech_ms=100,
            min_silence_ms=180,
            pad_ms=20,
        )

        analysis = factory.analyze_vad(canonical["id"], config=config)
        segments = factory.segment(canonical["id"], config=config)

        self.assertEqual(analysis["state"], "vad-analyzed")
        self.assertEqual(len(analysis["speech_regions"]), 2)
        self.assertEqual(len(segments), 2)
        self.assertTrue(all(item["kind"] == "segment" for item in segments))
        self.assertTrue(all((factory.root / item["stored_path"]).is_file() for item in segments))
        self.assertLessEqual(segments[0]["end_frame"], segments[1]["start_frame"])
        with closing(sqlite3.connect(factory.database_path)) as connection:
            count = connection.execute(
                "SELECT COUNT(*) FROM dataset_vad_regions WHERE item_id = ?",
                (canonical["id"],),
            ).fetchone()
        self.assertEqual(count, (2,))

    def test_silence_is_no_speech_not_a_fake_segment_success(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        audio_path = source / "silence.wav"
        _write_wav(audio_path, np.zeros(16_000, dtype=np.float32))
        factory = self.factory(source)
        item = factory.ingest("dataset_voice", (audio_path,))[0]

        self.assertEqual(factory.segment(item["id"]), [])
        self.assertEqual(factory.get_item(item["id"])["pipeline_state"], "no-speech")

    def test_expression_annotation_preserves_human_style_metadata(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        audio_path = source / "voice.wav"
        _write_wav(audio_path, _tone(0.35))
        factory = self.factory(source)
        item = factory.ingest("dataset_voice", (audio_path,))[0]

        updated = factory.update_annotations(
            item["id"],
            {
                "expression": "relieved",
                "expression_intensity": 0.82,
                "valence": 0.61,
                "arousal": -0.14,
                "dominance": 0.08,
                "style_description": "Warm, restrained relief.",
            },
        )

        self.assertEqual(updated["annotations"]["expression"], "relieved")
        self.assertEqual(updated["annotations"]["expression_intensity"], 0.82)
        self.assertEqual(updated["annotations"]["valence"], 0.61)
        self.assertEqual(updated["annotations"]["arousal"], -0.14)
        self.assertEqual(updated["annotations"]["dominance"], 0.08)
        self.assertEqual(
            updated["annotations"]["style_description"],
            "Warm, restrained relief.",
        )
        with self.assertRaisesRegex(DatasetFactoryError, "between -1 and 1"):
            factory.update_annotations(item["id"], {"valence": 1.01})

    def test_standard_worker_clips_keep_stable_lineage_and_non_authoritative_asr(
        self,
    ) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        clip = source / "worker-segment.wav"
        _write_wav(clip, _tone(0.35), sample_rate=32_000)
        factory = self.factory(source)
        factory.ensure_project(
            "dataset_voice",
            {"acquisition_mode": "standard", "sources": [str(source)]},
        )
        digest = hashlib.sha256(clip.read_bytes()).hexdigest()
        record = {
            "path": str(clip),
            "sha256": digest,
            "position": 2,
            "source_sha256": "a" * 64,
            "source_start_sample": 6400,
            "source_end_sample": 17600,
            "source_path": "long-recording.wav",
            "annotations": {
                "transcript": "今日真係好開心。",
                "language": "yue",
                "expression_suggestion": "happy",
                "authoritative": False,
                "review_required": True,
            },
        }

        first = factory.import_standard_clips(
            "dataset_voice", [record], parent_artifact_id="artifact_parent"
        )
        annotated = factory.update_annotations(
            first["items"][0]["id"],
            {"transcript": "人工確認済み。", "language": "ja", "speaker": "voice"},
        )
        self.assertEqual(annotated["annotations"]["transcript"], "人工確認済み。")
        record["annotations"] = {
            **record["annotations"],
            "transcript": "再実行したASR候補。",
        }
        second = factory.import_standard_clips(
            "dataset_voice", [record], parent_artifact_id="artifact_refreshed"
        )

        self.assertEqual(first["created_count"], 1)
        self.assertEqual(second["created_count"], 0)
        item = second["items"][0]
        self.assertEqual(item["kind"], "segment")
        self.assertTrue(item["original_name"].startswith("seg_"))
        self.assertEqual(item["annotations"]["transcript"], "人工確認済み。")
        suggestion = item["metadata"]["acquisition"]["asr_suggestion"]
        self.assertEqual(suggestion["transcript"], "再実行したASR候補。")
        self.assertEqual(suggestion["expression_suggestion"], "happy")
        self.assertIs(suggestion["authoritative"], False)
        self.assertEqual(
            item["metadata"]["lineage"],
            {
                "source_sha256": "a" * 64,
                "source_start_sample": 6400,
                "source_end_sample": 17600,
                "parent_artifact_id": "artifact_refreshed",
            },
        )
        self.assertEqual(factory.project_state("dataset_voice")["lifecycle_stage"], "review")

    def test_target_speaker_import_preserves_suggestions_and_enters_review(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        clip = source / "selected.wav"
        reference = source / "reference.wav"
        _write_wav(clip, _tone(0.35), sample_rate=32_000)
        _write_wav(reference, _tone(0.5), sample_rate=32_000)
        factory = self.factory(source)
        factory.ensure_project(
            "dataset_voice",
            {
                "acquisition_mode": "target-speaker",
                "sources": [str(source)],
                "reference_audio": str(reference),
            },
        )
        record = {
            "id": "seg_000000_abcdef",
            "path": str(clip),
            "sha256": hashlib.sha256(clip.read_bytes()).hexdigest(),
            "position": 0,
            "source_sha256": "b" * 64,
            "source_start_sample": 3200,
            "source_end_sample": 14400,
            "source_name": "long-recording.wav",
            "route": "salvage",
            "speaker": {"target_similarity": 0.934},
            "acquisition": {
                "route": "salvage",
                "separator": "MossFormer2_SS_16K",
            },
            "annotations": {
                "transcript": "今日は本当にありがとう。",
                "language": "ja",
                "expression_suggestion": "happy",
                "asr_backend": "sensevoice-small-cuda-v1",
                "authoritative": False,
                "review_required": True,
            },
        }

        imported = factory.import_target_speaker_clips(
            "dataset_voice", [record], parent_artifact_id="artifact_final"
        )

        item = imported["items"][0]
        self.assertEqual(imported["routes"]["salvage"], 1)
        self.assertEqual(item["review_status"], "pending")
        self.assertEqual(item["speaker_similarity"], 0.934)
        self.assertEqual(
            item["metadata"]["acquisition"]["asr_suggestion"]["transcript"],
            "今日は本当にありがとう。",
        )
        self.assertEqual(
            item["metadata"]["acquisition"]["asr_suggestion"]["asr_backend"],
            "sensevoice-small-cuda-v1",
        )
        self.assertEqual(
            factory.project_state("dataset_voice")["lifecycle_stage"], "review"
        )

        factory.review(item["id"], decision="accepted")
        self.assertEqual(
            factory.project_state("dataset_voice")["lifecycle_stage"], "review"
        )
        self.assertFalse(factory.get_item(item["id"])["review_complete"])

    def test_review_history_and_split_manifest_are_real_and_deterministic(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        factory = self.factory(source)
        segments = []
        for index in range(4):
            audio_path = source / f"voice-{index}.wav"
            _write_wav(audio_path, _tone(0.35, frequency=220.0 + index * 20.0))
            item = factory.ingest("dataset_voice", (audio_path,))[0]
            segments.extend(factory.segment(item["id"]))
        self.assertEqual(len(segments), 4)
        for segment in segments:
            reviewed = factory.review(segment["id"], decision="accepted", note="clear")
            self.assertEqual(reviewed["review_status"], "accepted")
            self.assertEqual(
                factory.review_history(segment["id"]),
                [{"decision": "accepted", "note": "clear", "created_at": reviewed["updated_at"]}],
            )

        first = factory.assign_splits(
            "dataset_voice", train=0.5, validation=0.25, test=0.25, seed="fixed"
        )
        second = factory.assign_splits(
            "dataset_voice", train=0.5, validation=0.25, test=0.25, seed="fixed"
        )

        self.assertEqual(first["assignments"], second["assignments"])
        self.assertEqual(first["item_counts"], {"train": 2, "validation": 1, "test": 1})
        manifest = factory.manifest("dataset_voice")
        self.assertEqual(manifest["schema"], "aniflive-dataset-manifest-v2")
        self.assertEqual(manifest["item_count"], 4)
        self.assertEqual(
            {item["split"] for item in manifest["items"]},
            {"train", "validation", "test"},
        )

    def test_rejected_item_is_removed_from_a_previous_split(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        audio_path = source / "voice.wav"
        _write_wav(audio_path, _tone(0.35))
        factory = self.factory(source)
        source_item = factory.ingest("dataset_voice", (audio_path,))[0]
        segment = factory.segment(source_item["id"])[0]
        factory.review(segment["id"], decision="accepted")
        factory.assign_splits("dataset_voice", train=1.0, validation=0.0, test=0.0)
        self.assertEqual(factory.get_item(segment["id"])["split_name"], "train")

        rejected = factory.review(segment["id"], decision="rejected", note="clipped")

        self.assertEqual(rejected["review_status"], "rejected")
        self.assertIsNone(rejected["split_name"])
        self.assertEqual(factory.review_history(segment["id"])[-1]["note"], "clipped")

    def test_manifest_refuses_unassigned_or_tampered_items(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        audio_path = source / "voice.wav"
        _write_wav(audio_path, _tone(0.35))
        factory = self.factory(source)
        source_item = factory.ingest("dataset_voice", (audio_path,))[0]
        segment = factory.segment(source_item["id"])[0]
        factory.review(segment["id"], decision="accepted")
        with self.assertRaisesRegex(DatasetFactoryError, "assigned"):
            factory.manifest("dataset_voice")
        factory.assign_splits("dataset_voice", train=1.0, validation=0.0, test=0.0)
        (factory.root / segment["stored_path"]).write_bytes(b"tampered")
        with self.assertRaisesRegex(DatasetFactoryError, "integrity"):
            factory.manifest("dataset_voice")

    def test_frozen_dataset_materializes_immutable_train_only_bundle(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        factory = self.factory(source)
        segments = []
        for index in range(3):
            audio_path = source / f"voice-{index}.wav"
            _write_wav(audio_path, _tone(0.35, frequency=220.0 + index * 20.0))
            source_item = factory.ingest("dataset_voice", (audio_path,))[0]
            segment = factory.segment(source_item["id"])[0]
            factory.update_annotations(
                segment["id"],
                {
                    "transcript": f"今日はテスト{index}です。",
                    "language": "ja",
                    "speaker": "voice",
                },
                source="manual",
                propagate=False,
            )
            factory.review(segment["id"], decision="accepted")
            factory.verify_annotation(segment["id"], kind="transcript")
            factory.verify_annotation(segment["id"], kind="speaker")
            segments.append(segment)
        split = factory.assign_splits(
            "dataset_voice", train=2 / 3, validation=1 / 3, test=0.0, seed="fixed"
        )
        factory.freeze_dataset("dataset_voice", require_expressions=False)

        first = factory.materialize_training_bundle("dataset_voice")
        second = factory.materialize_training_bundle("dataset_voice")

        self.assertEqual(first, second)
        self.assertEqual(first["examples"], split["item_counts"]["train"])
        root = Path(first["path"])
        descriptor = json.loads((root / "training-input.json").read_text(encoding="utf-8"))
        rows = (root / "train" / "voice.list").read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(rows), 2)
        self.assertEqual(len(descriptor["audio_files"]), 3)
        self.assertEqual(
            descriptor["split_counts"], {"train": 2, "validation": 1, "test": 0}
        )
        self.assertTrue(all("|voice|ja|今日はテスト" in row for row in rows))
        self.assertEqual(len(list((root / "train" / "wav").glob("*.wav"))), 2)
        self.assertEqual(len(list((root / "validation" / "wav").glob("*.wav"))), 1)
        self.assertEqual(len(list((root / "test" / "wav").glob("*.wav"))), 0)

    def test_freeze_requires_value_bound_transcript_and_speaker_verification(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        audio_path = source / "voice.wav"
        _write_wav(audio_path, _tone(0.35))
        factory = self.factory(source)
        source_item = factory.ingest("dataset_voice", (audio_path,))[0]
        segment = factory.segment(source_item["id"])[0]
        factory.update_annotations(
            segment["id"],
            {"transcript": "今日はテストです。", "language": "ja", "speaker": "voice"},
            propagate=False,
        )
        factory.review(segment["id"], decision="accepted")
        factory.assign_splits("dataset_voice", train=1.0, validation=0.0, test=0.0)

        with self.assertRaisesRegex(DatasetFactoryError, "transcript-unverified"):
            factory.freeze_dataset("dataset_voice", require_expressions=False)
        factory.verify_annotation(segment["id"], kind="transcript")
        with self.assertRaisesRegex(DatasetFactoryError, "speaker-unverified"):
            factory.freeze_dataset("dataset_voice", require_expressions=False)
        verified = factory.verify_annotation(segment["id"], kind="speaker")
        self.assertTrue(verified["review_complete"])
        frozen = factory.freeze_dataset("dataset_voice", require_expressions=False)
        self.assertEqual(frozen["schema"], "aniflive-frozen-dataset-v2")

    def test_annotation_edit_invalidates_matching_verification_without_deleting_history(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        audio_path = source / "voice.wav"
        _write_wav(audio_path, _tone(0.35))
        factory = self.factory(source)
        segment = factory.segment(factory.ingest("dataset_voice", (audio_path,))[0]["id"])[0]
        factory.update_annotations(
            segment["id"],
            {"transcript": "最初の文です。", "language": "ja", "speaker": "voice"},
            propagate=False,
        )
        factory.verify_annotation(segment["id"], kind="transcript")
        factory.verify_annotation(segment["id"], kind="speaker")

        changed = factory.update_annotations(
            segment["id"], {"transcript": "修正した文です。"}, propagate=False
        )

        self.assertFalse(changed["verification"]["transcript"]["valid"])
        self.assertTrue(changed["verification"]["speaker"]["valid"])
        self.assertEqual(
            len(factory.annotation_verification_history(segment["id"], kind="transcript")),
            1,
        )

    def test_expression_verification_is_conditional(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        audio_path = source / "voice.wav"
        _write_wav(audio_path, _tone(0.35))
        factory = self.factory(source)
        segment = factory.segment(factory.ingest("dataset_voice", (audio_path,))[0]["id"])[0]
        factory.update_annotations(
            segment["id"],
            {
                "transcript": "感情テストです。",
                "language": "ja",
                "speaker": "voice",
                "expression": "neutral",
            },
            propagate=False,
        )
        factory.review(segment["id"], decision="accepted")
        factory.verify_annotation(segment["id"], kind="transcript")
        factory.verify_annotation(segment["id"], kind="speaker")
        factory.assign_splits("dataset_voice", train=1.0, validation=0.0, test=0.0)

        with self.assertRaisesRegex(DatasetFactoryError, "expression-unverified"):
            factory.freeze_dataset("dataset_voice", require_expressions=True)

    def test_legacy_frozen_dataset_is_immutable_and_cannot_materialize_new_training(self) -> None:
        source = self.tmp_path / "source"
        source.mkdir()
        audio_path = source / "voice.wav"
        _write_wav(audio_path, _tone(0.35))
        factory = self.factory(source)
        item = factory.ingest("dataset_voice", (audio_path,))[0]
        state = factory.project_state("dataset_voice")
        legacy = factory.media_root / "dataset_voice" / "frozen" / ("a" * 64 + ".json")
        legacy.parent.mkdir(parents=True)
        legacy.write_text(
            json.dumps({"schema": "aniflive-frozen-dataset-v1", "manifest": {"items": []}}),
            encoding="utf-8",
        )
        digest = hashlib.sha256(legacy.read_bytes()).hexdigest()
        renamed = legacy.with_name(f"{digest}.json")
        legacy.rename(renamed)
        with closing(sqlite3.connect(factory.database_path)) as connection:
            connection.execute(
                "UPDATE dataset_projects SET frozen_manifest_sha256=?, "
                "frozen_manifest_path=?, frozen_at=? WHERE dataset_id=?",
                (
                    digest,
                    renamed.relative_to(factory.root).as_posix(),
                    "2026-09-02T00:00:00Z",
                    state["dataset_id"],
                ),
            )
            connection.commit()
        self.assertEqual(
            factory.project_state("dataset_voice")["qualification_status"],
            "legacy-frozen-unverified",
        )
        with self.assertRaisesRegex(DatasetFactoryError, "immutable"):
            factory.update_annotations(item["id"], {"speaker": "changed"})
        with self.assertRaisesRegex(DatasetFactoryError, "Legacy frozen dataset"):
            factory.materialize_training_bundle("dataset_voice")


if __name__ == "__main__":
    unittest.main()

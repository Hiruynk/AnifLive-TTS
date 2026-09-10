from __future__ import annotations

from dataclasses import replace
import wave

import pytest

from aniflive_tts import workstation_training as training


def _plan(**kwargs):
    return training.resolve_training_plan(kwargs, project_id="training_test")


def test_default_budget_scales_large_datasets_without_increasing_small_repetition():
    plan = _plan(preset="balanced")
    small, s = training.plan_training_budget(plan, duration_seconds=60, clips=17)
    medium, m = training.plan_training_budget(plan, duration_seconds=600, clips=169)
    large, large_budget = training.plan_training_budget(plan, duration_seconds=3600, clips=1000)
    assert plan.epoch_policy == "adaptive"
    assert small.gpt_epochs <= medium.gpt_epochs <= plan.gpt_epochs
    assert large.gpt_epochs < medium.gpt_epochs
    assert large.sovits_epochs < medium.sovits_epochs
    assert s["limited_data"]
    assert not m["convergence_or_overfit_established"]
    assert large_budget["stages"]["gpt"]["estimated_updates"] == 167 * large.gpt_epochs


def test_advanced_explicit_epochs_and_resume_are_preserved():
    plan = _plan(preset="advanced", gpt_epochs=19, sovits_epochs=7)
    actual, report = training.plan_training_budget(plan, duration_seconds=3600, clips=1000,
                                                 resumed=True)
    assert actual.gpt_epochs == 19 and actual.sovits_epochs == 7
    assert report["mode"] == "fixed"
    with pytest.raises(training.TrainingWorkerError, match="Resume requires"):
        training.plan_training_budget(_plan(), duration_seconds=600, clips=169, resumed=True)


def test_explicit_adaptive_advanced_uses_epochs_as_caps_and_saves_candidates():
    plan = _plan(preset="advanced", epoch_policy="adaptive", gpt_epochs=9, sovits_epochs=5,
                 save_every_epoch=5)
    result, _ = training.plan_training_budget(plan, duration_seconds=7200, clips=2000)
    assert 2 <= result.gpt_epochs <= 9
    assert 2 <= result.sovits_epochs <= 5
    assert result.save_every_epoch == 1


@pytest.mark.parametrize("duration,clips", [(0,2),(-1,2),(float("nan"),2),
                                          (float("inf"),2),(True,2),(600,1),(600,True)])
def test_invalid_measurements_fail_closed(duration, clips):
    with pytest.raises(training.TrainingWorkerError):
        training.plan_training_budget(_plan(), duration_seconds=duration, clips=clips)


def test_batch_size_affects_update_estimates_and_single_epoch_ceiling_is_preserved():
    plan = _plan(preset="advanced", epoch_policy="adaptive", gpt_epochs=1, sovits_epochs=1)
    result, report = training.plan_training_budget(plan, duration_seconds=3600, clips=1000)
    assert result.gpt_epochs == result.sovits_epochs == 1
    _, other = training.plan_training_budget(replace(plan, gpt_batch_size=12),
                                            duration_seconds=3600, clips=1000)
    assert other["stages"]["gpt"]["estimated_steps_per_epoch"] < report["stages"]["gpt"]["estimated_steps_per_epoch"]


def test_audio_measurement_uses_only_training_inventory(tmp_path):
    (tmp_path / "2-name2text.txt").write_text("a.wav\ta\n" "b.wav\tb\n")
    audio = tmp_path / "5-wav32k"
    audio.mkdir()
    for name, seconds in (("a.wav", 1), ("b.wav", 2), ("unused.wav", 4)):
        with wave.open(str(audio / name), "wb") as stream:
            stream.setnchannels(1)
            stream.setsampwidth(2)
            stream.setframerate(8000)
            stream.writeframes(b"\0\0" * (8000 * seconds))
    result = training.measure_training_audio(tmp_path)
    assert result["clips"] == 2
    assert result["duration_seconds"] == 3
    (audio / "b.wav").write_bytes(b"corrupt")
    with pytest.raises(training.TrainingWorkerError):
        training.measure_training_audio(tmp_path)

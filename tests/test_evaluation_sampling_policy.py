import pytest
from aniflive_tts.workstation_evaluation import (
    EvaluationWorkerError, resolve_evaluation_plan, evaluation_workload,
    evaluation_runtime_policy, evaluation_sampling_environment,
)

def test_sampling_policy_is_explicit_and_legacy_stays_default():
    legacy = resolve_evaluation_plan({})
    native = resolve_evaluation_plan({"semantic_sampling": "native-v2proplus-v1"})
    assert evaluation_runtime_policy(legacy) == {"semantic_sampling": "legacy-topk-v1", "repetition_penalty": 1.0}
    assert evaluation_runtime_policy(native) == {"semantic_sampling": "native-v2proplus-v1", "repetition_penalty": 1.35}
    # Runtime implementation changes do not change the submitted benchmark text/settings.
    assert evaluation_workload(legacy) == evaluation_workload(native)
    inherited = {"ANIFLIVE_TTS_SEMANTIC_SAMPLING": "wrong", "ANIFLIVE_TTS_REPETITION_PENALTY": "9"}
    inherited.update(evaluation_sampling_environment(native))
    assert inherited == {"ANIFLIVE_TTS_SEMANTIC_SAMPLING": "native-v2proplus-v1", "ANIFLIVE_TTS_REPETITION_PENALTY": "1.35"}

@pytest.mark.parametrize("value", ["automatic-voice-tuning", None, False, [], {}])
def test_unknown_sampling_policy_is_rejected(value):
    with pytest.raises(EvaluationWorkerError, match="semantic_sampling"):
        resolve_evaluation_plan({"semantic_sampling": value})

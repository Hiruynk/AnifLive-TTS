import torch

from research.v12_semantic.evaluate_mtp_viterbi import (
    accepted_prefix_lengths,
    viterbi_decode,
)


def test_zero_transition_weight_matches_independent_argmax() -> None:
    logits = torch.tensor(
        [[[4.0, 1.0, 0.0], [0.0, 2.0, 1.0], [1.0, 0.0, 3.0]]]
    )
    transition = torch.zeros(3, 3)
    actual = viterbi_decode(logits, transition, top_k=3, transition_weight=0.0)
    assert torch.equal(actual, torch.tensor([[0, 1, 2]]))


def test_transition_can_change_joint_path() -> None:
    logits = torch.tensor([[[3.0, 2.9], [3.0, 2.9]]])
    transition = torch.tensor([[-10.0, 0.0], [0.0, -10.0]])
    actual = viterbi_decode(logits, transition, top_k=2, transition_weight=1.0)
    assert actual[0, 0] != actual[0, 1]


def test_accepted_prefix_lengths() -> None:
    prediction = torch.tensor([[1, 2, 3], [1, 9, 3], [9, 2, 3]])
    targets = torch.tensor([[1, 2, 3], [1, 2, 3], [1, 2, 3]])
    assert torch.equal(accepted_prefix_lengths(prediction, targets), torch.tensor([3, 1, 0]))

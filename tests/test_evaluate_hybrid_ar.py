from research.v12_semantic.evaluate_hybrid_ar import (
    adjacent_repetition_rate,
    edit_distance,
    positional_agreement,
    sequence_similarity,
)


def test_edit_distance() -> None:
    assert edit_distance([], []) == 0
    assert edit_distance([1, 2, 3], [1, 4, 3]) == 1
    assert edit_distance([1, 2], [1, 2, 3]) == 1


def test_sequence_similarity() -> None:
    assert sequence_similarity([1, 2], [1, 2]) == 1.0
    assert sequence_similarity([], []) == 1.0
    assert sequence_similarity([1, 2], [3, 4]) == 0.0


def test_positional_agreement() -> None:
    assert positional_agreement([1, 2, 3], [1, 9, 3], 17) == 2 / 3
    assert positional_agreement([], [1], 17) == 0.0


def test_adjacent_repetition_rate() -> None:
    assert adjacent_repetition_rate([1]) == 0.0
    assert adjacent_repetition_rate([1, 1, 2, 2, 2]) == 3 / 4

from aniflive_tts.dataset_factory import DatasetFactory


def test_eighty_clips_keep_deterministic_groups_and_meet_minimums():
    groups = [(str(i), [{"id": str(i)}]) for i in range(80)]
    original = {str(i): "train" if i < 68 else "validation" if i < 76 else "test" for i in range(80)}
    result = DatasetFactory._production_group_assignment(groups, original)
    assert {name: list(result.values()).count(name) for name in ("train", "validation", "test")} == {
        "train": 67, "validation": 8, "test": 5,
    }
    assert result == DatasetFactory._production_group_assignment(groups, original)
    assert sum(result[key] != original[key] for key in original) == 1


def test_minimum_repair_keeps_source_groups_whole():
    groups = [("large", list(range(30))), ("a", list(range(5))), ("b", list(range(5)))]
    original = {name: "train" for name, _ in groups}
    result = DatasetFactory._production_group_assignment(groups, original)
    counts = {name: sum(len(items) for group, items in groups if result[group] == name)
              for name in ("train", "validation", "test")}
    assert counts["train"] >= 2 and counts["validation"] >= 5 and counts["test"] >= 5


def test_impossible_grouping_is_not_split_or_falsely_approved():
    assert DatasetFactory._production_group_assignment(
        [("single-recording", list(range(80)))], {"single-recording": "train"}
    ) is None

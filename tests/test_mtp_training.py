from __future__ import annotations

import torch

from research.v12_semantic.train_mtp import _split_indices


def test_grouped_split_never_leaks_record_rows() -> None:
    record_ids = torch.tensor([0, 0, 0, 1, 1, 2, 2, 2, 3])
    training, validation = _split_indices(
        record_ids,
        validation_fraction=0.25,
        generator=torch.Generator().manual_seed(1234),
    )
    training_records = set(record_ids[training].tolist())
    validation_records = set(record_ids[validation].tolist())
    assert training_records
    assert validation_records
    assert training_records.isdisjoint(validation_records)

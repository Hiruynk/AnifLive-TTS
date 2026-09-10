from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from aniflive_tts.workstation_training_resplit import derive_training_bundle


def _test_item_ids(path: Path) -> list[str]:
    value: Any = json.loads(path.resolve(strict=True).read_text(encoding="utf-8"))
    if not isinstance(value, dict) or value.get("split") != "test":
        raise ValueError(f"excluded manifest is not a test split: {path}")
    records = value.get("items")
    if not isinstance(records, list) or any(not isinstance(row, dict) for row in records):
        raise ValueError(f"excluded test manifest is malformed: {path}")
    item_ids = [row.get("source_item_id") for row in records]
    if any(not isinstance(item_id, str) or not item_id for item_id in item_ids):
        raise ValueError(f"excluded test manifest has an invalid item ID: {path}")
    return item_ids


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Create a checksum-verified training split with a fresh test holdout."
    )
    parser.add_argument("--parent", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--seed", required=True)
    parser.add_argument(
        "--exclude-test-manifest",
        action="append",
        default=[],
        type=Path,
    )
    args = parser.parse_args()
    excluded = sorted(
        {
            item_id
            for manifest in args.exclude_test_manifest
            for item_id in _test_item_ids(manifest)
        }
    )
    result = derive_training_bundle(
        parent_bundle=args.parent,
        output_root=args.output_root,
        seed=args.seed,
        excluded_test_item_ids=excluded,
    )
    print(json.dumps({**result, "excluded_test_item_count": len(excluded)}, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

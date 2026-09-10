from __future__ import annotations

import json
import re
from html.parser import HTMLParser
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


class _Controls(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.controls: list[dict] = []

    def handle_starttag(self, tag: str, attrs: list) -> None:
        if tag in {"button", "input", "select", "textarea"}:
            self.controls.append({"tag": tag, "attributes": dict(attrs)})


def test_studio_keeps_existing_controls_and_input_contracts() -> None:
    baseline = json.loads((ROOT / "tests/fixtures/studio-ui-contract.json").read_text())
    parser = _Controls()
    parser.feed((ROOT / "webui/index.html").read_text())
    for old in baseline["controls"]:
        if old["attributes"].get("id") in baseline["approved_removed_controls"]:
            continue
        assert {"tag": old["tag"], "attributes": old["attributes"]} in parser.controls
    assert not any(item["attributes"].get("id") == "railToggle" for item in parser.controls)


def test_studio_keeps_the_existing_api_targets() -> None:
    baseline = json.loads((ROOT / "tests/fixtures/studio-ui-contract.json").read_text())
    script = (ROOT / "webui/studio.js").read_text()
    targets = sorted(value.split(",")[0].strip() for value in re.findall(r"api\(([^\n]+)", script))
    assert targets == baseline["api_targets"]


def test_embedded_synthesis_controls_are_preserved() -> None:
    parser = _Controls()
    parser.feed((ROOT / "webui/synthesis.html").read_text())
    ids = [item["attributes"]["id"] for item in parser.controls if item["attributes"].get("id")]
    assert len(ids) == len(set(ids))
    for name in (
        "text",
        "play",
        "stop",
        "downloadAudio",
        "generationSeed",
        "generationTopK",
        "generationTemperature",
        "generationNoiseScale",
        "abSeed",
        "abTopK",
        "abTemperature",
        "abNoiseScale",
    ):
        assert name in ids

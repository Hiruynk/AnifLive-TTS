from __future__ import annotations

import argparse

import pytest

from aniflive_tts import cli


def _parser_with_missing_module(name):
    parser = argparse.ArgumentParser()
    def handler(_args):
        raise ModuleNotFoundError("missing dependency", name=name)
    parser.set_defaults(handler=handler)
    return parser


@pytest.mark.parametrize("name", ["fastapi", "httpx", "numpy", "pydantic", "uvicorn"])
def test_client_only_runtime_command_explains_install(monkeypatch, capsys, name):
    monkeypatch.setattr(cli, "build_parser", lambda: _parser_with_missing_module(name))
    with pytest.raises(SystemExit) as caught:
        cli.main([])
    assert caught.value.code == 2
    assert "aniflive-tts[runtime]" in capsys.readouterr().err


def test_internal_import_failure_is_not_hidden(monkeypatch):
    monkeypatch.setattr(cli, "build_parser", lambda: _parser_with_missing_module("internal_bug"))
    with pytest.raises(ModuleNotFoundError, match="missing dependency"):
        cli.main([])

"""Official Mandarin CPU text processing with native Metal neural components.

The official G2PWPinyin, normalization, corrections, and tone handling remain
in use. Only G2PW and BERT neural evaluation are delegated to the worker.
"""
from __future__ import annotations

import hashlib
import importlib
import sys
import tempfile
import types
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoTokenizer

_G2PW_ASSET_SHA256 = {
    "config.py": "b6154e494355d14a8d2c8a38d07720f77b34479e26aab21a5f63495ae904da76",
    "POLYPHONIC_CHARS.txt": "cc7e5a3eb377e3289af5e3a6be7505ac11a0d21422f80cfbf161059798f8671b",
    "MONOPHONIC_CHARS.txt": "fe7ed53098648441a94f414ed5fd3395789d1dceb98dc1e584489c2d43036b69",
    "char_bopomofo_dict.json": "4832fb6a553a9c5ee0c18b5cb698ce0acc312f324ee77ee61c340266d228184d",
    "bopomofo_to_pinyin_wo_tune_dict.json": "0fe90e26be7653023772b3ad2d08952961bbf295fb39ba685c7172d3d9769f52",
}
_TOKENIZER_SHA256 = {
    "tokenizer.json": "173796956820ea27bd14f76bf28162607ff4254807e2948253eb5b46f5bb643b",
    "config.json": "3d57de2fd7e80d0e5c8ff194f0bbb6baa10df7e43fc262a0cc71298a78b0a3e5",
}


def _verify(root: Path, expected: dict[str, str]) -> None:
    for name, digest in expected.items():
        path = root / name
        if hashlib.sha256(path.read_bytes()).hexdigest() != digest:
            raise ValueError(f"Pinned Mandarin frontend asset hash mismatch: {path}")


def _rpc_array(client_method, inputs: dict[str, np.ndarray], output_name: str) -> np.ndarray:
    with tempfile.TemporaryDirectory(prefix="aniflive-mandarin-") as temp:
        root = Path(temp)
        source = root / "input"
        source.mkdir()
        target = root / "output"
        for name, array in inputs.items():
            np.save(source / f"{name}.npy", array, allow_pickle=False)
        result = client_method(source, target)
        expected = (target / output_name).resolve()
        if Path(result["output_path"]).resolve() != expected:
            raise ValueError("native frontend returned an unexpected output path")
        value = np.load(expected, allow_pickle=False)
        if value.dtype != np.float32 or list(value.shape) != result["shape"] or not np.isfinite(value).all():
            raise ValueError("native frontend returned invalid output tensor")
        return value


class MandarinBridge:
    """Install the official Mandarin frontend with native G2PW and BERT RPCs.

    The first bridge installs its factory before ``text.chinese2`` imports.
    A later bridge can replace a prior native owner after its worker closes.
    The caller serializes frontend requests.
    """

    def __init__(self, official: Path, g2pw_assets: Path, tokenizer_dir: Path, client: Any):
        self.official = Path(official).expanduser().resolve(strict=True)
        self.source_root = self.official / "GPT_SoVITS"
        self.assets = Path(g2pw_assets).expanduser().resolve(strict=True)
        self.tokenizer_dir = Path(tokenizer_dir).expanduser().resolve(strict=True)
        if not (self.source_root / "text" / "chinese2.py").is_file():
            raise FileNotFoundError("official source root has no Mandarin frontend")
        _verify(self.assets, _G2PW_ASSET_SHA256)
        _verify(self.tokenizer_dir, _TOKENIZER_SHA256)
        existing_chinese = sys.modules.get("text.chinese2")
        if existing_chinese is not None:
            previous = getattr(existing_chinese, "_aniflive_mandarin_bridge_owner", None)
            if not isinstance(previous, MandarinBridge) or previous.official != self.official:
                raise RuntimeError("preloaded Mandarin frontend is not owned by a compatible native bridge")
            if previous.client.ready:
                raise RuntimeError("cannot replace Mandarin bridge while its worker is still active")
        if str(self.source_root) not in sys.path:
            sys.path.insert(0, str(self.source_root))
        # Refuse a previously imported text package from another source tree.
        existing = sys.modules.get("text")
        if existing is not None:
            origin = Path(existing.__file__).resolve()
            if not origin.is_relative_to((self.source_root / "text").resolve()):
                raise RuntimeError("text package belongs to a different official source root")
        self.client = client
        self.tokenizer = AutoTokenizer.from_pretrained(str(self.tokenizer_dir), local_files_only=True)
        self.g2pw_rpc_count = 0
        self.bert_rpc_count = 0
        self.pronunciations: list[dict[str, Any]] = []
        self._install_g2pw()

    def _install_g2pw(self) -> None:
        package = importlib.import_module("text.g2pw")
        source_module = importlib.import_module("text.g2pw.g2pw")
        onnx_api = importlib.import_module("text.g2pw.onnx_api")
        bridge = self

        class NativeConverter(onnx_api._G2PWBaseOnnxConverter):
            def __init__(self, *args, **kwargs):
                # The official config loader executes a pinned config.py. Its
                # bytecode cache belongs outside the immutable model package.
                previous = sys.dont_write_bytecode
                sys.dont_write_bytecode = True
                try:
                    super().__init__(*args, **kwargs)
                finally:
                    sys.dont_write_bytecode = previous

            def _predict(self, model_input):
                ids = model_input["input_ids"]
                count = len(model_input["char_ids"])
                arrays = {
                    "input_ids": ids,
                    "token_type_ids": model_input["token_type_ids"],
                    "attention_mask": model_input["attention_masks"],
                    "phoneme_mask": model_input["phoneme_masks"],
                    "char_ids": model_input["char_ids"],
                    "position_ids": model_input["position_ids"],
                }
                # The official dedup path shares one identical text row across
                # multiple query positions. Native inference accepts one text
                # row per query, so restore those duplicate rows explicitly.
                for key in ("input_ids", "token_type_ids", "attention_mask"):
                    if arrays[key].shape[0] == 1 and count > 1:
                        arrays[key] = np.repeat(arrays[key], count, axis=0)
                    if arrays[key].shape[0] != count:
                        raise ValueError("official G2PW token/query batch mismatch")
                probs = _rpc_array(bridge.client.frontend_g2pw, arrays, "probabilities.npy")
                if probs.shape != (count, len(self.labels)):
                    raise ValueError("native G2PW probability shape mismatch")
                indices = np.argmax(probs, axis=1)
                preds = [self.labels[int(i)] for i in indices]
                confidences = [float(probs[row, index]) for row, index in enumerate(indices)]
                bridge.g2pw_rpc_count += 1
                for char_id, label, confidence in zip(model_input["char_ids"], preds, confidences):
                    bridge.pronunciations.append({"character": self.chars[int(char_id)], "label": label, "confidence": confidence})
                return preds, confidences

        original_converter = source_module.G2PWOnnxConverter
        original_factory = package.G2PWPinyin

        def native_factory(*args, **kwargs):
            kwargs["model_dir"] = str(bridge.assets)
            kwargs["model_source"] = str(bridge.tokenizer_dir)
            return original_factory(*args, **kwargs)

        source_module.G2PWOnnxConverter = NativeConverter
        package.G2PWPinyin = native_factory
        try:
            chinese = sys.modules.get("text.chinese2")
            if chinese is None:
                chinese = importlib.import_module("text.chinese2")
                replacement = chinese.g2pw
            else:
                # These options match chinese2.py's original constructor.
                replacement = native_factory(v_to_u=False, neutral_tone_with_five=True)
            if not isinstance(replacement._g2pw, NativeConverter):
                raise RuntimeError("official Mandarin G2PW did not use the native bridge")
            chinese.g2pw = replacement
            chinese._aniflive_mandarin_bridge_owner = self
        finally:
            package.G2PWPinyin = original_factory
            source_module.G2PWOnnxConverter = original_converter

    def _require_owner(self) -> None:
        chinese = sys.modules.get("text.chinese2")
        if chinese is None or getattr(chinese, "_aniflive_mandarin_bridge_owner", None) is not self:
            raise RuntimeError("Mandarin bridge no longer owns the frontend")

    def attach(self, processor: Any) -> None:
        self._require_owner()
        if getattr(processor, "_aniflive_mandarin_bridge", None) is not None:
            raise RuntimeError("a Mandarin bridge is already attached to this processor")
        bridge = self
        original_preprocess = processor.preprocess

        def preprocess(_processor, *args, **kwargs):
            bridge._require_owner()
            return original_preprocess(*args, **kwargs)

        def get_bert_feature(_processor, text: str, word2ph: list) -> torch.Tensor:
            bridge._require_owner()
            if len(word2ph) != len(text):
                raise ValueError("official Mandarin word2ph must align with normalized text")
            tokenized = bridge.tokenizer(text, return_tensors="np")
            arrays = {key: np.asarray(tokenized[key]) for key in ("input_ids", "token_type_ids", "attention_mask")}
            hidden = _rpc_array(bridge.client.frontend_bert, arrays, "hidden.npy")
            if hidden.shape != (1, arrays["input_ids"].shape[1], 1024):
                raise ValueError("native Mandarin BERT hidden shape mismatch")
            res = torch.from_numpy(hidden)[0].cpu()[1:-1]
            if res.shape[0] < len(text):
                raise ValueError("tokenizer produced fewer Mandarin states than text characters")
            features = [res[i].repeat(word2ph[i], 1) for i in range(len(word2ph))]
            bridge.bert_rpc_count += 1
            return torch.cat(features, dim=0).T

        processor.preprocess = types.MethodType(preprocess, processor)
        processor.get_bert_feature = types.MethodType(get_bert_feature, processor)
        processor._aniflive_mandarin_bridge = self

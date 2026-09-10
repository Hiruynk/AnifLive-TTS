import ast
import hashlib
import json
from pathlib import Path

import pytest

from aniflive_tts.training_entrypoints import (
    TrainingEntrypointError, stage_recovery_entrypoint,
)


@pytest.mark.parametrize("stage,filename", [("gpt", "s1_train.py"), ("sovits", "s2_train.py")])
def test_pinned_training_recovery_hooks_compile_without_changing_upstream(tmp_path, stage, filename):
    source = Path("/opt/aniflive-tts/gpt-sovits")
    original = source / "GPT_SoVITS" / filename
    if not original.exists():
        pytest.skip("Requires the pinned Linux neural worker image")
    before = original.read_bytes()
    target = stage_recovery_entrypoint(source, tmp_path, stage)
    tree = ast.parse(target.read_text())
    compile(tree, filename, "exec")
    assert original.read_bytes() == before
    record = json.loads(target.with_suffix(".source.json").read_text())
    assert record["source_sha256"] == hashlib.sha256(before).hexdigest()
    calls = [n for n in ast.walk(tree) if isinstance(n, ast.Call)]
    if stage == "gpt":
        boundary = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef)
                        and n.name == "on_train_epoch_end")
        assert isinstance(boundary.body[-1], ast.Expr)
        assert boundary.body[-1].value.func.id == "finish_gpt_checkpoint_boundary"
        assert any(isinstance(n.func, ast.Name) and n.func.id == "gpt_recovery_callback"
                   and any(k.arg == "managed_boundary" and k.value.value is True for k in n.keywords)
                   for n in calls)
    else:
        run = next(n for n in ast.walk(tree) if isinstance(n, ast.FunctionDef) and n.name == "run")
        loop = next(n for n in run.body if isinstance(n, ast.For)
                    and isinstance(n.target, ast.Name) and n.target.id == "epoch")
        assert loop.body[-2].value.func.attr == "step"
        assert loop.body[-1].test.func.id == "save_sovits_boundary"
        assert isinstance(loop.body[-1].body[0], ast.Break)


def test_changed_upstream_fails_before_publishing_runner(tmp_path):
    source = tmp_path / "source"
    (source / "GPT_SoVITS").mkdir(parents=True)
    (source / "GPT_SoVITS/s1_train.py").write_text("import torch\n# unsupported source\n")
    output = tmp_path / "output"
    with pytest.raises(TrainingEntrypointError, match="contract changed"):
        stage_recovery_entrypoint(source, output, "gpt")
    assert not output.exists()

def test_native_gpt_sampler_epoch_is_set_before_first_iterator(monkeypatch):
    import sys
    from types import SimpleNamespace
    import torch
    from aniflive_tts import training_recovery
    source=Path("/opt/aniflive-tts/gpt-sovits")
    if not source.exists():
        pytest.skip("Requires the pinned neural image")
    monkeypatch.syspath_prepend(str(source))
    monkeypatch.syspath_prepend(str(source/"GPT_SoVITS"))
    from AR.data.data_module import Text2SemanticDataModule
    monkeypatch.setattr(Text2SemanticDataModule,"train_dataloader",Text2SemanticDataModule.train_dataloader)
    monkeypatch.setattr(training_recovery,"_gpt_loader_epoch_installed",False)
    monkeypatch.setattr(torch.cuda,"is_available",lambda:False)
    class Dataset:
        def __len__(self):return 134
        def get_sample_length(self,index):return 1+(index%12)*0.2
        def collate(self,items):return items
    module=Text2SemanticDataModule({"data":{"num_workers":1},"train":{"batch_size":4}},
                                   "semantic.tsv","phoneme.txt")
    module._train_dataset=Dataset()
    module.trainer=SimpleNamespace(fit_loop=SimpleNamespace(epoch_progress=SimpleNamespace(
        current=SimpleNamespace(processed=2))))
    training_recovery.install_gpt_dataloader_epoch_policy()
    loader=module.train_dataloader()
    assert loader.sampler.epoch==2
    expected=list(loader.sampler)
    loader.sampler.set_epoch(0)
    assert list(loader.sampler)!=expected

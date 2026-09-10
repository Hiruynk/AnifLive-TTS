from types import SimpleNamespace

import pytest

from aniflive_tts import workstation_training as training


def source_tree(tmp_path, filename, text):
    root = tmp_path / "source"
    directory = root / "GPT_SoVITS"
    directory.mkdir(parents=True)
    (directory / filename).write_text(text)
    return root


def test_required_gpt_resume_cannot_fall_back_to_fresh_training(tmp_path):
    text = (
        "def run():\n"
        "    ckpt_path = None\n"
        '    print("ckpt_path:", ckpt_path)\n'
        "    trainer.fit(ckpt_path=ckpt_path)\n"
    )
    source = source_tree(tmp_path, "s1_train.py", text)
    runner = training._resume_entrypoint(source, tmp_path / "output", "gpt")
    calls = []
    namespace = {"trainer": SimpleNamespace(fit=lambda **kwargs: calls.append(kwargs))}
    exec(compile(runner.read_text(), str(runner), "exec"), namespace)
    with pytest.raises(RuntimeError, match="refusing fresh"):
        namespace["run"]()
    assert calls == []
    assert (source / "GPT_SoVITS/s1_train.py").read_text() == text


@pytest.mark.parametrize("epochs", [(2, 3), (2, None)])
def test_sovits_resume_rejects_mismatched_epoch_or_load_failure(tmp_path, epochs):
    text = (
        "fallback = False\n"
        "def run():\n"
        "    global fallback\n"
        "    try:\n"
        '        _, _, _, epoch_str = utils.load_checkpoint("D")\n'
        '        _, _, _, epoch_str = utils.load_checkpoint("G")\n'
        "        epoch_str += 1\n"
        "    except:  # 如果首次不能加载，加载pretrain\n"
        "        fallback = True\n"
        "        epoch_str = 1\n"
        "    # scheduler_g\n"
        "    return epoch_str\n"
    )
    source = source_tree(tmp_path, "s2_train.py", text)
    runner = training._resume_entrypoint(source, tmp_path / "output", "sovits")
    def load(name):
        epoch = epochs[0 if name == "D" else 1]
        if epoch is None:
            raise ValueError("corrupt checkpoint")
        return None, None, None, epoch
    namespace = {"utils": SimpleNamespace(load_checkpoint=load)}
    exec(compile(runner.read_text(), str(runner), "exec"), namespace)
    with pytest.raises(RuntimeError, match="refusing pretrained"):
        namespace["run"]()
    assert namespace["fallback"] is False
    assert (source / "GPT_SoVITS/s2_train.py").read_text() == text


def test_resume_patch_rejects_changed_upstream_contract(tmp_path):
    source = source_tree(tmp_path, "s1_train.py", "print('different source')\n")
    with pytest.raises(training.TrainingWorkerError, match="contract changed"):
        training._resume_entrypoint(source, tmp_path / "output", "gpt")


def test_gpt_checkpoint_does_not_hide_partial_sovits_pair(tmp_path):
    root = tmp_path / "resume"
    (root / "gpt").mkdir(parents=True)
    (root / "sovits").mkdir()
    (root / "gpt/epoch=1-step=2.ckpt").write_bytes(b"fixture")
    (root / "sovits/G_2.pth").write_bytes(b"fixture")
    with pytest.raises(training.TrainingWorkerError, match="both generator"):
        training._copy_resume(root, gpt_root=tmp_path / "gpt", sovits_root=tmp_path / "sovits")

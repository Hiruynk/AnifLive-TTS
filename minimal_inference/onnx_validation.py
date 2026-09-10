# Modified by AnifLive-TTS in 2026. See ANIFLIVE_TTS_PROVENANCE.json.
"""
ONNX导出精度校验模块
用于在导出过程中对比PyTorch模型和ONNX模型的输出，定位精度损失位置
"""
import os
import hashlib
from contextlib import nullcontext
from unittest.mock import patch
import torch
import numpy as np
import onnxruntime
from typing import Dict, Tuple, Optional, Any
import json


def _shared_normal_model(onnx_path, draws, dummy_inputs, output_dir, label):
    """Align standard-normal draws for comparison without rewriting the exported model."""
    if not draws:
        return onnx_path, {"mode": "deterministic-or-explicit-noise"}
    import onnx
    from pathlib import Path

    model = onnx.load(onnx_path)
    random_nodes = []
    for node in model.graph.node:
        if node.op_type in {"RandomNormal", "RandomNormalLike"}:
            random_nodes.append(node)
        elif node.op_type in {"RandomUniform", "RandomUniformLike", "Bernoulli", "Multinomial"}:
            raise ValueError("Unsupported mixed random operators in numerical validation")
        for attribute in node.attribute:
            graphs = [attribute.g] if attribute.type == onnx.AttributeProto.GRAPH else (
                list(attribute.graphs) if attribute.type == onnx.AttributeProto.GRAPHS else []
            )
            if any("Random" in child.op_type for graph in graphs for child in graph.node):
                raise ValueError("Nested random operators require explicit validation controls")
    if len(random_nodes) != len(draws):
        raise ValueError("PyTorch and ONNX standard-normal draw counts differ")
    controls = []
    for node, draw in zip(random_nodes, draws):
        attrs = {a.name: onnx.helper.get_attribute_value(a) for a in node.attribute}
        if attrs.get("mean", 0.0) != 0.0 or attrs.get("scale", 1.0) != 1.0:
            raise ValueError("Only standard-normal primitives can use shared validation draws")
        if len(node.output) != 1 or not np.isfinite(draw).all():
            raise ValueError("Invalid normal draw in numerical validation")
        dtype = onnx.helper.np_dtype_to_tensor_dtype(draw.dtype)
        if "dtype" in attrs and attrs["dtype"] != dtype:
            raise ValueError("PyTorch and ONNX random draw dtypes differ")
        if node.op_type == "RandomNormal" and tuple(attrs["shape"]) != draw.shape:
            raise ValueError("PyTorch and ONNX random draw shapes differ")
        controls.append({
            "operator": node.op_type,
            "shape": list(draw.shape),
            "dtype": str(draw.dtype),
            "sha256": hashlib.sha256(draw.tobytes()).hexdigest(),
        })
        replacement = onnx.helper.make_node(
            "Constant", [], list(node.output),
            value=onnx.numpy_helper.from_array(draw),
            name=node.name + "_shared_validation_draw",
        )
        node.CopyFrom(replacement)
    onnx.external_data_helper.convert_model_from_external_data(model)
    controlled = model.SerializeToString()
    root = Path(output_dir) / "validation-inputs"
    root.mkdir(parents=True, exist_ok=True)
    safe_label = "".join(c if c.isalnum() else "_" for c in label)
    inputs_path = root / (safe_label + ".npz")
    values = {f"input_{i}": tensor.detach().cpu().numpy() for i, tensor in enumerate(dummy_inputs.values())}
    values.update({f"noise_{i}": value for i, value in enumerate(draws)})
    np.savez_compressed(inputs_path, **values)
    source_digest = hashlib.sha256()
    with open(onnx_path, "rb") as source:
        while block := source.read(1024 * 1024):
            source_digest.update(block)
    source_sha = source_digest.hexdigest()
    return controlled, {
        "mode": "shared-standard-normal-v1",
        "source_model_sha256": source_sha,
        "controlled_model_sha256": hashlib.sha256(controlled).hexdigest(),
        "production_graph_modified": False,
        "controls": controls,
        "inputs_file": str(inputs_path.relative_to(Path(output_dir))),
        "input_names": list(dummy_inputs),
    }


class ONNXValidator:
    """ONNX导出精度校验器"""

    def __init__(self, output_dir: str, onnx_device: str = "cpu"):
        """
        初始化校验器

        Args:
            output_dir: ONNX输出目录
            onnx_device: ONNX推理设备 ("cpu" 或 "cuda")
        """
        self.output_dir = output_dir
        self.onnx_device = onnx_device
        self.validation_results = []

        # 设置ONNX运行时选项
        so = onnxruntime.SessionOptions()
        so.graph_optimization_level = onnxruntime.GraphOptimizationLevel.ORT_ENABLE_ALL
        so.intra_op_num_threads = 4
        so.inter_op_num_threads = 1
        so.use_deterministic_compute = True
        self.session_options = so

        if onnx_device == "cuda":
            self.providers = [("CUDAExecutionProvider", {"device_id": 0}), "CPUExecutionProvider"]
        else:
            self.providers = ["CPUExecutionProvider"]

    def _record_failure(self, model_name, reason):
        self.validation_results.append({
            "model_name": model_name,
            "output_name": "__execution__",
            "passed": False,
            "metrics": {"error": str(reason)[:4000]},
        })
        self.save_report()
        return False

    def validate_model(
        self,
        model_name: str,
        onnx_path: str,
        pytorch_model: torch.nn.Module,
        dummy_inputs: Dict[str, torch.Tensor],
        output_names: list,
        rtol: float = 1e-3,
        atol: float = 1e-5,
        case_id: Optional[str] = None,
    ) -> bool:
        """
        校验单个模型的输出精度

        Args:
            model_name: 模型名称
            onnx_path: ONNX模型路径
            pytorch_model: PyTorch模型
            dummy_inputs: 虚拟输入字典 {input_name: tensor}
            output_names: 输出名称列表
            rtol: 相对误差阈值
            atol: 绝对误差阈值

        Returns:
            是否通过校验
        """
        print(f"\n{'='*60}")
        print(f"校验模型: {model_name}")
        print(f"{'='*60}")

        if not os.path.exists(onnx_path):
            print(f"❌ ONNX模型不存在: {onnx_path}")
            return self._record_failure(model_name, "missing_onnx_model")

        # 确保模型在评估模式
        pytorch_model.eval()

        # 准备PyTorch输入
        pt_inputs = []
        input_names = list(dummy_inputs.keys())
        for name in input_names:
            tensor = dummy_inputs[name].detach().clone()
            # 如果是GPU上的模型，确保输入在GPU上
            if hasattr(pytorch_model, 'parameters'):
                try:
                    next(pytorch_model.parameters())
                    device = next(pytorch_model.parameters()).device
                    if tensor.device != device:
                        tensor = tensor.to(device)
                except StopIteration:
                    pass
            pt_inputs.append(tensor)

        # Capture the actual normal draws consumed by this reference forward pass.
        # ONNX validation uses those same draws; the production model is untouched.
        draws = []
        original_randn_like = torch.randn_like
        original_randn = torch.randn

        def capture(function):
            def recorded(*args, **kwargs):
                value = function(*args, **kwargs)
                draws.append(value.detach().cpu().numpy().copy())
                return value
            return recorded

        # A native ATen waveform reference avoids platform-dependent oneDNN
        # convolution rounding. Scope this to CPU decoder validation only.
        native_cpu_waveform = model_name in {"SoVITS", "SoVITSStreaming"} and all(
            tensor.device.type == "cpu" for tensor in pt_inputs
        )
        reference_mkldnn = False if native_cpu_waveform else torch.backends.mkldnn.enabled
        reference_context = (
            torch.backends.mkldnn.flags(enabled=False)
            if native_cpu_waveform else nullcontext()
        )
        # PyTorch前向传播
        with torch.no_grad(), reference_context, patch("torch.randn_like", capture(original_randn_like)), patch(
            "torch.randn", capture(original_randn)
        ):
            try:
                if len(pt_inputs) == 1:
                    pt_outputs = pytorch_model(pt_inputs[0])
                else:
                    pt_outputs = pytorch_model(*pt_inputs)

                # 如果输出是元组，转换为列表
                if isinstance(pt_outputs, tuple):
                    pt_outputs = list(pt_outputs)
                elif not isinstance(pt_outputs, list):
                    pt_outputs = [pt_outputs]
            except Exception as e:
                print(f"❌ PyTorch前向传播失败: {e}")
                return self._record_failure(model_name, f"pytorch_forward: {e}")

        # 加载ONNX模型并推理
        try:
            validation_model, random_alignment = _shared_normal_model(
                onnx_path, draws, dummy_inputs, self.output_dir, f"{model_name}-{len(self.validation_results)}"
            )
            ort_session = onnxruntime.InferenceSession(
                validation_model,
                sess_options=self.session_options,
                providers=self.providers
            )
        except Exception as e:
            print(f"❌ 加载ONNX模型失败: {e}")
            return self._record_failure(model_name, f"onnx_load_or_random_alignment: {e}")

        # Export may remove unused inputs (for example the fixed speed=1 path).
        # Feed only declared graph inputs, but never invent a missing required input.
        required_inputs = {value.name for value in ort_session.get_inputs()}
        missing_inputs = required_inputs - set(dummy_inputs)
        if missing_inputs:
            print(f"❌ ONNX inputs are missing: {sorted(missing_inputs)}")
            return self._record_failure(model_name, f"missing_onnx_inputs: {sorted(missing_inputs)}")
        pruned_inputs = sorted(set(dummy_inputs) - required_inputs)
        ort_inputs = {
            name: tensor.detach().cpu().numpy()
            for name, tensor in dummy_inputs.items()
            if name in required_inputs
        }

        try:
            ort_outputs = ort_session.run(output_names, ort_inputs)
        except Exception as e:
            print(f"❌ ONNX前向传播失败: {e}")
            return self._record_failure(model_name, f"onnx_forward: {e}")

        if len(pt_outputs) != len(ort_outputs) or len(pt_outputs) != len(output_names):
            return self._record_failure(model_name, "output_count_mismatch")

        # 对比输出
        all_passed = True
        for i, (pt_output, ort_output, out_name) in enumerate(zip(pt_outputs, ort_outputs, output_names)):
            passed, metrics = self._compare_tensors(
                pt_output, ort_output, f"{out_name}_{i}", rtol, atol
            )
            if not passed:
                all_passed = False

            # 保存结果
            result = {
                "model_name": model_name,
                "case_id": case_id,
                "reference_execution": {
                    "torch_cpu_threads": torch.get_num_threads(),
                    "torch_cpu_mkldnn": reference_mkldnn,
                    "native_cpu_waveform_reference": native_cpu_waveform,
                    "ort_cpu_threads": self.session_options.intra_op_num_threads,
                    "ort_deterministic_compute": True,
                },
                "output_name": out_name,
                "passed": passed,
                "pruned_input_names": pruned_inputs,
                "random_alignment": random_alignment,
                "metrics": metrics
            }
            self.validation_results.append(result)

        status = "✅ 通过" if all_passed else "❌ 失败"
        print(f"\n{status} {model_name} 校验完成")
        print(f"{'='*60}\n")

        if not all_passed:
            label = f"{model_name}-{case_id or 'default'}-{len(self.validation_results)}"
            safe_label = "".join(c if c.isalnum() else "_" for c in label)
            inputs_dir = os.path.join(self.output_dir, "validation-inputs")
            os.makedirs(inputs_dir, exist_ok=True)
            snapshot = os.path.join(inputs_dir, safe_label + ".npz")
            np.savez_compressed(
                snapshot,
                **{f"input_{i}": value.detach().cpu().numpy()
                   for i, value in enumerate(dummy_inputs.values())},
            )
            for row in self.validation_results[-len(output_names):]:
                row["failure_inputs_file"] = os.path.relpath(snapshot, self.output_dir)
                row["input_names"] = list(dummy_inputs)
        self.save_report()
        return all_passed

    def _compare_tensors(
        self,
        pt_tensor: torch.Tensor,
        ort_tensor: np.ndarray,
        output_name: str,
        rtol: float,
        atol: float
    ) -> Tuple[bool, Dict[str, float]]:
        """
        对比两个张量的精度

        Args:
            pt_tensor: PyTorch张量
            ort_tensor: ONNX输出numpy数组
            output_name: 输出名称
            rtol: 相对误差阈值
            atol: 绝对误差阈值

        Returns:
            (是否通过, 指标字典)
        """
        # 转换为numpy
        pt_np = pt_tensor.detach().cpu().numpy()
        ort_np = ort_tensor

        # 确保形状一致
        if pt_np.shape != ort_np.shape:
            print(f"  ⚠️  输出形状不匹配: PyTorch {pt_np.shape} vs ONNX {ort_np.shape}")
            return False, {"error": "shape_mismatch"}

        if pt_np.size == 0:
            return False, {"error": "empty_output"}
        if not np.isfinite(pt_np).all() or not np.isfinite(ort_np).all():
            return False, {"error": "nonfinite_output"}
        if pt_np.dtype.kind not in "biuf" or ort_np.dtype.kind not in "biuf":
            return False, {"error": "unsupported_output_dtype"}
        discrete = pt_np.dtype.kind in "biu" or ort_np.dtype.kind in "biu"
        if discrete and (pt_np.dtype.kind in "biu") != (ort_np.dtype.kind in "biu"):
            return False, {"error": "discrete_output_dtype_mismatch"}
        exact_match = np.array_equal(pt_np, ort_np)
        pt_np = pt_np.astype(np.float64)
        ort_np = ort_np.astype(np.float64)

        # 计算各种误差指标
        abs_diff = np.abs(pt_np - ort_np)
        max_abs_diff = np.max(abs_diff)
        mean_abs_diff = np.mean(abs_diff)

        # 相对误差（避免除零）
        denominator = np.maximum(np.abs(pt_np), np.abs(ort_np))
        rel_diff = abs_diff / np.maximum(denominator, 1e-10)
        max_rel_diff = np.max(rel_diff)
        mean_rel_diff = np.mean(rel_diff)

        # MSE
        mse = np.mean((pt_np - ort_np) ** 2)
        rmse = np.sqrt(mse)

        # Preserve per-batch embedding scores and define scalar/zero cases.
        batch = pt_np.shape[0] if pt_np.ndim >= 2 else 1
        pt_flat = pt_np.reshape(batch, -1)
        ort_flat = ort_np.reshape(batch, -1)
        scores = []
        for reference, actual in zip(pt_flat, ort_flat):
            reference_norm = np.linalg.norm(reference)
            actual_norm = np.linalg.norm(actual)
            if reference_norm == 0.0 or actual_norm == 0.0:
                scores.append(1.0 if np.array_equal(reference, actual) else 0.0)
            else:
                scores.append(float(np.dot(reference, actual) / (reference_norm * actual_norm)))
        cosine_sim = float(np.mean(scores))

        metrics = {
            "max_abs_diff": float(max_abs_diff),
            "mean_abs_diff": float(mean_abs_diff),
            "max_rel_diff": float(max_rel_diff),
            "mean_rel_diff": float(mean_rel_diff),
            "mse": float(mse),
            "rmse": float(rmse),
            "cosine_similarity": float(cosine_sim) if not np.isnan(cosine_sim) else 0.0,
            "pt_range": [float(np.min(pt_np)), float(np.max(pt_np))],
            "ort_range": [float(np.min(ort_np)), float(np.max(ort_np))],
            "pt_mean": float(np.mean(pt_np)),
            "ort_mean": float(np.mean(ort_np)),
        }

        # 打印详细报告
        print(f"\n  输出: {output_name}")
        print(f"  形状: {pt_np.shape}")
        print(f"  PyTorch 范围: [{metrics['pt_range'][0]:.6f}, {metrics['pt_range'][1]:.6f}], 均值: {metrics['pt_mean']:.6f}")
        print(f"  ONNX    范围: [{metrics['ort_range'][0]:.6f}, {metrics['ort_range'][1]:.6f}], 均值: {metrics['ort_mean']:.6f}")
        print(f"  最大绝对误差: {max_abs_diff:.6e}")
        print(f"  平均绝对误差: {mean_abs_diff:.6e}")
        print(f"  最大相对误差: {max_rel_diff:.6%}")
        print(f"  平均相对误差: {mean_rel_diff:.6%}")
        print(f"  RMSE: {rmse:.6e}")
        print(f"  余弦相似度: {cosine_sim:.6f}")

        # 判断是否通过
        # Standard tolerance semantics: absolute protection near zero, relative
        # protection at scale. Token IDs and lengths are discrete, never approximate.
        passed = bool(exact_match) if discrete else bool(
            np.allclose(ort_np, pt_np, rtol=rtol, atol=atol, equal_nan=False)
        )
        metrics["comparison"] = "exact" if discrete else "elementwise-atol-plus-rtol"
        metrics["rtol"] = float(rtol)
        metrics["atol"] = float(atol)

        # 对于声纹嵌入，余弦相似度更重要
        if "sv" in output_name.lower() or "embedding" in output_name.lower():
            if cosine_sim < 0.99:
                print(f"  ⚠️  声纹嵌入相似度较低，可能导致音色失真！")
                passed = False

        return passed, metrics

    def print_summary(self):
        """打印校验摘要"""
        print(f"\n{'='*80}")
        print("ONNX导出精度校验摘要")
        print(f"{'='*80}\n")

        failed_count = 0
        total_count = len(self.validation_results)

        for result in self.validation_results:
            model_name = result["model_name"]
            output_name = result["output_name"]
            passed = result["passed"]
            metrics = result["metrics"]

            status = "✅" if passed else "❌"
            print(f"{status} {model_name} - {output_name}")

            if not passed:
                failed_count += 1
                if metrics.get("error"):
                    print(f"   Error: {metrics['error']}")
                    continue
                print(f"   最大相对误差: {metrics['max_rel_diff']:.6%}")
                print(f"   余弦相似度: {metrics['cosine_similarity']:.6f}")
                if "cosine_similarity" in metrics and metrics['cosine_similarity'] < 0.95:
                    print(f"   ⚠️  警告: 相似度过低，可能导致严重失真！")

        print(f"\n总计: {total_count} 个输出, {failed_count} 个失败, {total_count - failed_count} 个通过")

        if failed_count > 0:
            print(f"\n❌ 存在精度损失，建议检查失败的模块")
            # 找出损失最大的模块
            print("\n🔍 损失最大的模块:")
            sorted_results = sorted(
                self.validation_results,
                key=lambda x: x["metrics"].get("cosine_similarity", 1.0)
            )
            for result in sorted_results[:3]:  # 显示最差的3个
                metrics = result["metrics"]
                print(f"  - {result['model_name']}/{result['output_name']}: "
                      f"余弦相似度={metrics['cosine_similarity']:.6f}, "
                      f"最大相对误差={metrics['max_rel_diff']:.6%}")
        else:
            print(f"\n✅ 所有模块校验通过！")

        print(f"{'='*80}\n")

    def save_report(self, output_path: str = None):
        """保存详细报告到JSON文件"""
        if output_path is None:
            output_path = os.path.join(self.output_dir, "validation_report.json")

        def convert_numpy_types(obj):
            """递归转换numpy类型为Python原生类型"""
            if isinstance(obj, np.bool_):
                return bool(obj)
            elif isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            elif isinstance(obj, dict):
                return {k: convert_numpy_types(v) for k, v in obj.items()}
            elif isinstance(obj, (list, tuple)):
                return [convert_numpy_types(v) for v in obj]
            return obj

        # 转换validation_results中的numpy类型
        converted_results = convert_numpy_types(self.validation_results)

        os.makedirs(os.path.dirname(os.path.abspath(output_path)), exist_ok=True)
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(converted_results, f, indent=2, ensure_ascii=False)

        print(f"详细报告已保存到: {output_path}")


def create_validation_audio(
    ref_wav_path: str,
    target_sr: int = 16000,
    duration: float = 3.0
) -> np.ndarray:
    """
    创建用于校验的音频数据

    Args:
        ref_wav_path: 参考音频路径
        target_sr: 目标采样率
        duration: 目标时长（秒）

    Returns:
        音频numpy数组
    """
    import librosa

    # 加载音频
    audio, sr = librosa.load(ref_wav_path, sr=target_sr)

    # 截取或填充到目标时长
    target_length = int(target_sr * duration)
    if len(audio) > target_length:
        audio = audio[:target_length]
    elif len(audio) < target_length:
        audio = np.pad(audio, (0, target_length - len(audio)), mode='constant')

    return audio
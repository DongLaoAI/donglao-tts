"""Convert the ONNX graphs from onnx_export.py to OpenVINO IR (.xml/.bin).

Mechanical once ONNX is working: OpenVINO's `convert_model` reads an ONNX graph directly, no
model-specific code needed here beyond producing the ONNX file first (see onnx_export.py) and
picking sane input shapes. Same scoping as ONNX export: only `model.ar.backbone: custom`
(ARTransformerLM) is the verified path; the `qwen3` backbone's prefill-only ONNX export can also
be converted, but inherits the same "experimental, decode-step unimplemented" caveat.
"""

import os

import openvino as ov
import openvino.properties.hint as ov_hints


def compile_openvino_model(xml_path, device="CPU", fp32=True):
    """`core.compile_model` alone is not enough to get fp32-precision results even from an
    fp32-saved IR: the CPU plugin defaults to a reduced-precision `inference_precision` hint
    (bf16 on this machine) for performance, independent of the stored weight precision --
    verified directly: same fp32 IR, ~1e-2 max logit diff vs PyTorch without this hint, ~2e-7
    with it. `fp32=True` (default) forces full precision; set False to accept the platform's
    faster default (its own deliberate choice, not a silent one)."""
    core = ov.Core()
    config = {ov_hints.inference_precision: ov.Type.f32} if fp32 else {}
    return core.compile_model(xml_path, device, config=config)


def convert_onnx_to_openvino(onnx_path, out_path):
    """`out_path` should end in .xml -- OpenVINO writes a companion .bin next to it.
    `compress_to_fp16=False`: ov.save_model defaults to FP16-compressing weights, which is a
    real precision-reduction decision -- keep this step precision-preserving (matching the fp32
    ONNX export) and do quantization/precision-reduction as its own deliberate, separate step."""
    ov_model = ov.convert_model(onnx_path)
    ov.save_model(ov_model, out_path, compress_to_fp16=False)
    return out_path


def export_all_openvino(ar_model, nar_model, out_dir, d_model, backbone="custom"):
    """Runs the full ONNX export (see onnx_export.py) into `out_dir`, then converts each graph to
    OpenVINO IR alongside it. Returns a dict of {name: xml_path}. `backbone` must match the
    architecture of `ar_model` ('custom' -> ARTransformerLM, 'qwen3' -> ARQwen3LM, prefill-only)."""
    from donglao_tts.export.onnx_export import (
        export_ar_decode_step,
        export_ar_prefill,
        export_ar_qwen3_prefill,
        export_nar_layer,
    )

    os.makedirs(out_dir, exist_ok=True)
    results = {}

    nar_onnx = os.path.join(out_dir, "nar_layer.onnx")
    export_nar_layer(nar_model, nar_onnx, d_model)
    results["nar_layer"] = convert_onnx_to_openvino(
        nar_onnx, os.path.join(out_dir, "nar_layer.xml"))

    if backbone == "custom":
        prefill_onnx = os.path.join(out_dir, "ar_prefill.onnx")
        decode_onnx = os.path.join(out_dir, "ar_decode_step.onnx")
        export_ar_prefill(ar_model, prefill_onnx, d_model)
        export_ar_decode_step(ar_model, decode_onnx, d_model)
        results["ar_prefill"] = convert_onnx_to_openvino(
            prefill_onnx, os.path.join(out_dir, "ar_prefill.xml"))
        results["ar_decode_step"] = convert_onnx_to_openvino(
            decode_onnx, os.path.join(out_dir, "ar_decode_step.xml"))
    elif backbone == "qwen3":
        prefill_onnx = os.path.join(out_dir, "ar_qwen3_prefill.onnx")
        export_ar_qwen3_prefill(ar_model, prefill_onnx, d_model)
        results["ar_qwen3_prefill"] = convert_onnx_to_openvino(
            prefill_onnx, os.path.join(out_dir, "ar_qwen3_prefill.xml"))
    else:
        raise ValueError(f"unknown backbone: {backbone!r} (expected 'custom' or 'qwen3')")

    return results

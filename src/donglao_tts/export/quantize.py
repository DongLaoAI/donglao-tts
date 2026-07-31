"""Post-training (weight-only, no calibration data needed) INT8 quantization of the exported
ONNX graphs, via onnxruntime.quantization.quantize_dynamic -- targets MatMul/Gemm-style ops
(i.e. the nn.Linear layers, the vast majority of parameters in both the AR and NAR Transformers),
leaving embeddings (Gather ops) and activations at full precision. This is deliberately the
simple, calibration-free PTQ path: no representative dataset/calibration loop is wired up yet
(that's what static quantization / OpenVINO's NNCF would need) -- see README's roadmap.
"""

from onnxruntime.quantization import QuantType, quantize_dynamic


def quantize_onnx_dynamic(onnx_path, out_path, weight_type=QuantType.QInt8):
    """Weight-only dynamic quantization -- no calibration dataset required, safe as a first PTQ
    pass for any of the three exported graphs (ar_prefill, ar_decode_step, nar_layer)."""
    quantize_dynamic(onnx_path, out_path, weight_type=weight_type)
    return out_path

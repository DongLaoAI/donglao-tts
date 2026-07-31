import subprocess
import sys
import textwrap


def _run_onnx_cli_without_onnxruntime(*args):
    script = textwrap.dedent(
        f"""
        import builtins
        import runpy
        import sys

        original_import = builtins.__import__

        def import_without_onnxruntime(name, *import_args, **import_kwargs):
            if name == "onnxruntime" or name.startswith("onnxruntime."):
                raise ModuleNotFoundError("blocked optional dependency", name=name)
            return original_import(name, *import_args, **import_kwargs)

        builtins.__import__ = import_without_onnxruntime
        sys.argv = ["donglao-onnx-infer", *{list(args)!r}]
        runpy.run_module("donglao_tts.cli.onnx_infer", run_name="__main__")
        """
    )
    return subprocess.run(
        [sys.executable, "-c", script],
        check=False,
        capture_output=True,
        text=True,
    )


def test_onnx_cli_help_does_not_require_export_extra():
    result = _run_onnx_cli_without_onnxruntime("--help")

    assert result.returncode == 0, result.stderr
    assert "usage: donglao-onnx-infer" in result.stdout


def test_onnx_cli_explains_how_to_install_export_extra():
    result = _run_onnx_cli_without_onnxruntime("--config", "missing.yaml")

    assert result.returncode != 0
    assert 'pip install "donglao-tts[export]"' in result.stderr

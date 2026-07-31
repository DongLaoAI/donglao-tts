# Security policy

## Reporting a vulnerability

Do not disclose credentials, personal data, or exploit details in a public issue. Open the
repository's **Security** tab and use **Report a vulnerability**. If private vulnerability
reporting is unavailable, contact the maintainers privately before sharing technical details.
Include affected versions, impact, reproduction steps, and any suggested mitigation.

## Release controls

- Release dependencies are resolved in `uv.lock` and audited in CI.
- PyPI releases use OIDC Trusted Publishing rather than long-lived API tokens.
- Release artifacts are built once, validated, smoke-tested, sent to TestPyPI, and then promoted
  unchanged to PyPI after environment approval.
- Third-party GitHub Actions are pinned to full commit hashes.
- Hugging Face remote code is pinned to an immutable revision in the example configuration.
- PyTorch checkpoints use the restricted `weights_only=True` loader.

## Audit exception

`PYSEC-2025-194` is ignored explicitly in CI. The upstream advisory describes a
`torch.jit.script` memory-corruption issue and marks `2.6.0-NA` as the last affected release.
`donglao-tts` requires PyTorch 2.11.x and does not use `torch.jit.script`.

Review this exception whenever the advisory changes or the supported PyTorch range is updated:

https://github.com/pypa/advisory-database/blob/main/vulns/torch/PYSEC-2025-194.yaml

`PYSEC-2026-3447` is also ignored explicitly. It concerns Unicode-normalization bypasses in
`setuptools` source-distribution file exclusions on macOS. PyTorch 2.11 constrains its transitive
`setuptools` dependency below the fixed version. This project builds distributions with Hatchling
inside an isolated Linux environment and does not use `setuptools` or `MANIFEST.in`.

https://github.com/pypa/advisory-database/blob/main/vulns/setuptools/PYSEC-2026-3447.yaml

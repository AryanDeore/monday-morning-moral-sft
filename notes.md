## Deployment Notes

- `uv sync` on EC2 `t3.micro` attempted to install `nvidia-*` packages due to torch resolution path; these CUDA packages are unnecessary for CPU-only inference and caused storage pressure.
- Storage errors observed were `No space left on device` and `Disk quota exceeded`, caused by large torch wheel/dependency extraction on a small root volume.
- CPU-only torch 2.10.0 was confirmed available for the platform via:
  - `pip install --index-url https://download.pytorch.org/whl/cpu torch==2.10.0`
- The install failure was due to disk limits, not package availability.
- Decision: keep a single shared `pyproject.toml` GPU-friendly for future fine-tuning; apply CPU-only torch pin/index override only on the deployed SSH instance when needed.

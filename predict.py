"""
Cog Predictor for SDXL-Turbo (stabilityai/sdxl-turbo) on EKS unified-inf.
- setup(): load pipeline from /weights (EFS mount or S3-synced by init container).
- predict(): text-to-image with num_inference_steps=1, guidance_scale=0.0; upload to S3 if configured.
See: https://huggingface.co/stabilityai/sdxl-turbo
"""

import os
import sys
import time
from pathlib import Path as FilePath
from typing import Any, List, Optional
from types import SimpleNamespace

from cog import BasePredictor, Input, Path


def _log(msg: str) -> None:
    """Emit log line to stdout so it appears in pod logs (Cog captures stdout)."""
    print(f"[sdxl-turbo] {msg}", flush=True)
    sys.stdout.flush()


class Predictor(BasePredictor):
    """SDXL-Turbo: real-time text-to-image in 1 step. Load from /weights at startup."""

    def setup(self) -> None:
        """Load SDXL-Turbo pipeline from /weights (EFS or S3-synced)."""
        # Set before first torch import to reduce CUDA memory fragmentation (avoids OOM/SIGSEGV)
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
        import torch

        _log("setup: starting pipeline load from /weights")
        if not hasattr(torch, "xpu"):
            torch.xpu = SimpleNamespace(empty_cache=lambda: None)

        from diffusers import AutoPipelineForText2Image

        weights_dir = "/weights"
        if not os.path.isdir(weights_dir):
            raise RuntimeError(f"Weights directory not found: {weights_dir}")

        t0 = time.perf_counter()
        self.pipe = AutoPipelineForText2Image.from_pretrained(
            weights_dir,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to("cuda")
        # Reduce peak VRAM (helps avoid OOM/SIGSEGV on L4/A10G)
        if hasattr(self.pipe, "enable_attention_slicing"):
            self.pipe.enable_attention_slicing(1)
            _log("setup: enable_attention_slicing(1)")
        if hasattr(self.pipe, "enable_vae_slicing"):
            self.pipe.enable_vae_slicing()
            _log("setup: enable_vae_slicing()")
        _log(f"setup: pipeline loaded and on cuda in {time.perf_counter() - t0:.1f}s")

    def predict(
        self,
        prompt: str = Input(
            description="Text prompt for image generation",
            default="A cinematic shot of a baby racoon wearing an intricate italian priest robe.",
        ),
        num_inference_steps: int = Input(
            description="Number of denoising steps (1 is enough for SDXL-Turbo)",
            default=1,
            ge=1,
            le=4,
        ),
        request_id: Optional[str] = Input(
            description="Request ID for S3 delivery path (set by SQS worker); omit for local runs.",
            default=None,
        ),
    ) -> Any:
        """Generate an image from the prompt. Returns S3 URIs when S3_DELIVERY_BUCKET is set."""
        import torch

        prompt_preview = (prompt[:60] + "…") if len(prompt) > 60 else prompt
        _log(f"predict: start prompt={prompt_preview!r} num_inference_steps={num_inference_steps}")

        def step_callback(pipe, step_index: int, timestep, callback_kwargs):
            # step_index is 0-based; log 1-based for readability
            _log(f"predict: denoising step {step_index + 1}/{num_inference_steps}")
            return callback_kwargs

        if hasattr(torch.cuda, "empty_cache"):
            torch.cuda.empty_cache()
            _log("predict: empty_cache done")
        t0 = time.perf_counter()
        # SDXL-Turbo: guidance_scale=0.0, 1 step recommended
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
            callback_on_step_end=step_callback,
        ).images[0]
        elapsed = time.perf_counter() - t0
        _log(f"predict: pipeline done in {elapsed:.1f}s")

        out_dir = FilePath("/tmp/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "output.png"
        _log("predict: saving image to /tmp/outputs/output.png")
        image.save(str(out_file))
        output_paths: List[tuple] = [(str(out_file), "output.png")]
        _log("predict: image saved")

        # S3 delivery (EKS: reconciler injects S3_DELIVERY_BUCKET, MODEL_ID, S3_DELIVERY_PREFIX)
        bucket = os.environ.get("S3_DELIVERY_BUCKET")
        prefix = os.environ.get("S3_DELIVERY_PREFIX", "deliveries")
        model_id = os.environ.get("MODEL_ID", "model")
        rid = request_id or "local"

        if bucket:
            import boto3

            _log(f"predict: uploading to S3 bucket={bucket} prefix={prefix}/{model_id}/{rid}/")
            s3 = boto3.client(
                "s3",
                region_name=os.environ.get("AWS_REGION", "eu-central-1"),
            )
            uris: List[str] = []
            for local_path, filename in output_paths:
                key = f"{prefix}/{model_id}/{rid}/{filename}"
                with open(local_path, "rb") as f:
                    body = f.read()
                s3.put_object(
                    Bucket=bucket,
                    Key=key,
                    Body=body,
                    ContentType="image/png",
                )
                uris.append(f"s3://{bucket}/{key}")
            _log(f"predict: S3 upload done, returning {len(uris)} URI(s)")
            return uris

        if len(output_paths) == 1:
            _log("predict: returning local Path (no S3 bucket)")
            return Path(output_paths[0][0])
        _log("predict: returning list of Paths")
        return [Path(p) for p, _ in output_paths]

"""
Cog Predictor for SDXL-Turbo (stabilityai/sdxl-turbo) on EKS unified-inf.
- setup(): load pipeline from /weights (EFS mount or S3-synced by init container).
- predict(): text-to-image with num_inference_steps=1, guidance_scale=0.0; upload to S3 if configured.
See: https://huggingface.co/stabilityai/sdxl-turbo
"""

import os
from pathlib import Path as FilePath
from typing import Any, List, Optional
from types import SimpleNamespace

from cog import BasePredictor, Input, Path


class Predictor(BasePredictor):
    """SDXL-Turbo: real-time text-to-image in 1 step. Load from /weights at startup."""

    def setup(self) -> None:
        """Load SDXL-Turbo pipeline from /weights (EFS or S3-synced)."""
        import torch

        if not hasattr(torch, "xpu"):
            torch.xpu = SimpleNamespace(empty_cache=lambda: None)

        from diffusers import AutoPipelineForText2Image

        weights_dir = "/weights"
        if not os.path.isdir(weights_dir):
            raise RuntimeError(f"Weights directory not found: {weights_dir}")

        self.pipe = AutoPipelineForText2Image.from_pretrained(
            weights_dir,
            torch_dtype=torch.float16,
            variant="fp16",
        )
        self.pipe.to("cuda")

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

        # SDXL-Turbo: guidance_scale=0.0, 1 step recommended
        image = self.pipe(
            prompt=prompt,
            num_inference_steps=num_inference_steps,
            guidance_scale=0.0,
        ).images[0]

        out_dir = FilePath("/tmp/outputs")
        out_dir.mkdir(parents=True, exist_ok=True)
        out_file = out_dir / "output.png"
        image.save(str(out_file))
        output_paths: List[tuple] = [(str(out_file), "output.png")]

        # S3 delivery (EKS: reconciler injects S3_DELIVERY_BUCKET, MODEL_ID, S3_DELIVERY_PREFIX)
        bucket = os.environ.get("S3_DELIVERY_BUCKET")
        prefix = os.environ.get("S3_DELIVERY_PREFIX", "deliveries")
        model_id = os.environ.get("MODEL_ID", "model")
        rid = request_id or "local"

        if bucket:
            import boto3

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
            return uris

        if len(output_paths) == 1:
            return Path(output_paths[0][0])
        return [Path(p) for p, _ in output_paths]

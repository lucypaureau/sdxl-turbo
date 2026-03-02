#!/usr/bin/env bash
# Upload model weights to S3 so EKS pods can use them (init container syncs s3_weights_uri -> /weights).
# Run locally or in CI before deploying the model. Requires: aws CLI, optional huggingface_hub.
#
# Usage:
#   export S3_WEIGHTS_URI="s3://my-bucket/models/my-model/1.0.0/"
#   export HF_REPO_ID="stabilityai/stable-diffusion-xl-base-1.0"   # optional, for Hugging Face
#   export HF_TOKEN="hf_xxx"   # optional, for gated/private models
#   ./upload_weights_to_s3.sh
#
# Or with a custom download: set CUSTOM_DOWNLOAD_CMD to a command that writes files into $WEIGHTS_DIR.
set -euo pipefail

if [ -z "${S3_WEIGHTS_URI:-}" ]; then
  echo "ERROR: S3_WEIGHTS_URI is required (e.g. s3://my-bucket/models/my-model/1.0.0/)" >&2
  exit 1
fi

WEIGHTS_DIR="${WEIGHTS_DIR:-./.weights-upload}"
rm -rf "$WEIGHTS_DIR" && mkdir -p "$WEIGHTS_DIR"

if [ -n "${CUSTOM_DOWNLOAD_CMD:-}" ]; then
  echo "Running custom download: $CUSTOM_DOWNLOAD_CMD"
  eval "WEIGHTS_DIR=$WEIGHTS_DIR $CUSTOM_DOWNLOAD_CMD"
elif [ -n "${HF_REPO_ID:-}" ]; then
  echo "Downloading from Hugging Face: $HF_REPO_ID"
  pip install -q huggingface_hub
  python3 -c "
from pathlib import Path
from huggingface_hub import snapshot_download
path = Path('$WEIGHTS_DIR').resolve()
snapshot_download('$HF_REPO_ID', local_dir=str(path), token='${HF_TOKEN:-}')
print('Downloaded to', path)
"
else
  echo "No HF_REPO_ID or CUSTOM_DOWNLOAD_CMD set. Put weight files in $WEIGHTS_DIR and re-run, or set HF_REPO_ID." >&2
  echo "Example: export HF_REPO_ID=stabilityai/stable-diffusion-xl-base-1.0" >&2
  exit 1
fi

echo "Uploading to $S3_WEIGHTS_URI ..."
aws s3 sync "$WEIGHTS_DIR/" "$S3_WEIGHTS_URI" --no-progress
echo "Done. Use s3_weights_uri=$S3_WEIGHTS_URI when deploying the model."

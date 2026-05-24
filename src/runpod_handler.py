# Copyright (c) 2026 Scenema AI
# SPDX-License-Identifier: MIT

"""RunPod Serverless handler for Scenema Audio.

Wraps the AudioProcessor so it can be deployed as a RunPod serverless endpoint.

Expected input:
    {
        "input": {
            "prompt": "<speak voice=\"...\">Your text</speak>",
            "mode": "generate",           # optional, default "generate"
            "reference_voice_url": null,   # optional
            "background_sfx": false,       # optional
            "validate": true,              # optional
            "seed": -1                     # optional, -1 = random
        }
    }

Returns base64-encoded WAV audio in the output.
"""

import base64
import logging
import os

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Disable xet downloader — causes "Background writer channel closed" on RunPod
os.environ["HF_HUB_ENABLE_XET"] = "0"

# Use RunPod network volume for model storage if available.
# Must use os.environ[] (not setdefault) to override Dockerfile ENV values.
RUNPOD_VOLUME = "/runpod-volume/models"
_has_volume = os.path.isdir("/runpod-volume")
print(f"[runpod_handler] /runpod-volume exists: {_has_volume}")
if _has_volume:
    os.environ["MODEL_DIR"] = RUNPOD_VOLUME
    os.environ["AUDIO_CKPT"] = f"{RUNPOD_VOLUME}/scenema-audio-transformer-int8.safetensors"
    os.environ["VAE_ENCODER_CKPT"] = f"{RUNPOD_VOLUME}/scenema-audio-vae-encoder.safetensors"
    os.environ["PIPELINE_CKPT"] = f"{RUNPOD_VOLUME}/scenema-audio-pipeline.safetensors"
    os.environ["GEMMA_ROOT"] = f"{RUNPOD_VOLUME}/gemma-3-12b-it"
    os.environ["HF_HUB_CACHE"] = f"{RUNPOD_VOLUME}/hf_cache"
    print(f"[runpod_handler] MODEL_DIR -> {RUNPOD_VOLUME}")
else:
    print(f"[runpod_handler] WARNING: /runpod-volume not found, using /app/models")

import runpod

from audio_core.processor import AudioProcessor
from common.handlers.base import ProcessJob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("scenema-audio-runpod")

# Download models at import time (during cold start)
from server import _download_models, MODEL_DIR

print(f"[runpod_handler] MODEL_DIR resolved to: {MODEL_DIR}")
MODEL_DIR.mkdir(parents=True, exist_ok=True)
_download_models()

# Initialize processor once
processor = AudioProcessor()
processor.startup()
logger.info("AudioProcessor ready for RunPod serverless")


async def handler(event):
    """RunPod serverless handler."""
    job_input = event["input"]
    job_id = event.get("id", "unknown")

    job = ProcessJob(job_id=job_id, input=job_input)
    result = await processor.process(job)

    if not result.success:
        raise Exception(result.error or "Generation failed")

    output = result.output
    audio_b64 = base64.b64encode(output.data).decode() if output.data else None

    return {
        "audio": audio_b64,
        "content_type": output.content_type or "audio/wav",
        "metadata": output.metadata or {},
    }


runpod.serverless.start({"handler": handler})

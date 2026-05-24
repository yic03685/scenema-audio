# Copyright (c) 2026 Scenema AI
# SPDX-License-Identifier: MIT

"""RunPod Serverless handler for Scenema Audio.

Wraps the AudioProcessor so it can be deployed as a RunPod serverless endpoint.
All model paths are set via Dockerfile ENV to /runpod-volume/models/.
Models are downloaded on first cold start and cached on the network volume.

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

import runpod

from audio_core.processor import AudioProcessor
from common.handlers.base import ProcessJob

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(name)s %(levelname)s %(message)s",
)
logger = logging.getLogger("scenema-audio-runpod")

# ── Debug: log volume and path info before anything else ──
import shutil

logger.info("=== RunPod Volume Debug ===")
logger.info("/runpod-volume exists: %s", os.path.isdir("/runpod-volume"))
if os.path.isdir("/runpod-volume"):
    usage = shutil.disk_usage("/runpod-volume")
    logger.info("/runpod-volume disk: total=%.1fGB free=%.1fGB", usage.total / 1e9, usage.free / 1e9)
    try:
        logger.info("/runpod-volume contents: %s", os.listdir("/runpod-volume"))
    except Exception as e:
        logger.warning("/runpod-volume listdir failed: %s", e)
else:
    logger.warning("/runpod-volume NOT FOUND — models will fail to load")

for key in ["MODEL_DIR", "AUDIO_CKPT", "PIPELINE_CKPT", "VAE_ENCODER_CKPT",
            "GEMMA_ROOT", "MELBAND_MODEL_PATH", "HF_HUB_CACHE", "HF_HUB_ENABLE_XET"]:
    logger.info("ENV %s = %s", key, os.environ.get(key, "<not set>"))

# Download models at import time (during cold start).
# _download_models() checks if each file exists and only downloads if missing.
from server import _download_models, MODEL_DIR

logger.info("MODEL_DIR resolved to: %s", MODEL_DIR)
MODEL_DIR.mkdir(parents=True, exist_ok=True)

# Ensure TMPDIR exists on the volume so HF downloads don't fill the container disk
tmpdir = os.environ.get("TMPDIR", "/tmp")
os.makedirs(tmpdir, exist_ok=True)
logger.info("TMPDIR = %s", tmpdir)

usage = shutil.disk_usage(str(MODEL_DIR))
logger.info("MODEL_DIR disk: total=%.1fGB free=%.1fGB", usage.total / 1e9, usage.free / 1e9)

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

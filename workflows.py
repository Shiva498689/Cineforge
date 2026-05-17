import copy
import json
from pathlib import Path

from config import settings


_BASE = Path(__file__).parent


def _load(filename: str) -> dict:
    path = _BASE / filename
    with open(path, "r") as f:
        return json.load(f)


def build_image_workflow(
    prompt: str,
    lora: str | None = None
) -> dict:
    # Load image workflow template
    wf = _load("image_workflow_template.json")
    wf = copy.deepcopy(wf)

    # Inject user prompt
    wf[settings.IMG_PROMPT_NODE_ID]["inputs"]["text"] = prompt

    # Configure LoRA
    if lora:
        wf["8"]["inputs"]["lora_name"] = lora
        wf["8"]["inputs"]["strength_model"] = 0.8
        wf["8"]["inputs"]["strength_clip"] = 0.8
    else:
        # Disable LoRA
        wf["8"]["inputs"]["strength_model"] = 0
        wf["8"]["inputs"]["strength_clip"] = 0

    return wf


def build_video_workflow(
    prompt: str,
    lora: str | None = None
) -> dict:
    # Load video workflow template
    wf = _load("video_workflow_template.json")
    wf = copy.deepcopy(wf)

    # Inject user prompt
    wf[settings.VID_PROMPT_NODE_ID]["inputs"]["text"] = prompt

    # Configure LoRA
    if lora:
        wf["8"]["inputs"]["lora_name"] = lora
        wf["8"]["inputs"]["strength_model"] = 0.8
        wf["8"]["inputs"]["strength_clip"] = 0.8
    else:
        # Disable LoRA
        wf["8"]["inputs"]["strength_model"] = 0
        wf["8"]["inputs"]["strength_clip"] = 0

    return wf

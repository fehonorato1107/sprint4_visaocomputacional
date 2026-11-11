"""
Modelo multimodal local (Florence-2, Microsoft).
Task tokens: <OCR>, <OCR_WITH_REGION>, <CAPTION>...
Requer ~4 GB VRAM (base) ou ~6 GB (large).
"""
import numpy as np
from PIL import Image
import torch
from transformers import AutoProcessor, AutoModelForCausalLM

_MODEL = None
_PROC = None


def _load():
    global _MODEL, _PROC
    if _MODEL is None:
        name = "microsoft/Florence-2-base"
        _PROC = AutoProcessor.from_pretrained(name, trust_remote_code=True)
        _MODEL = AutoModelForCausalLM.from_pretrained(
            name, trust_remote_code=True,
            torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32,
        )
        if torch.cuda.is_available():
            _MODEL = _MODEL.cuda()
        _MODEL.eval()
    return _MODEL, _PROC


def read_text(img: np.ndarray) -> dict:
    model, proc = _load()
    pil = Image.fromarray(img if img.ndim == 3 else
                          np.stack([img] * 3, axis=-1))

    prompt = "<OCR_WITH_REGION>"
    inputs = proc(text=prompt, images=pil, return_tensors="pt").to(
        model.device, model.dtype if torch.cuda.is_available() else torch.float32
    )
    with torch.no_grad():
        out = model.generate(
            input_ids=inputs["input_ids"],
            pixel_values=inputs["pixel_values"],
            max_new_tokens=1024,
            num_beams=3,
            do_sample=False,
        )
    text = proc.batch_decode(out, skip_special_tokens=False)[0]
    parsed = proc.post_process_generation(
        text, task="<OCR_WITH_REGION>", image_size=(pil.width, pil.height)
    )

    quad_boxes = parsed.get("<OCR_WITH_REGION>", {}).get("quad_boxes", [])
    labels = parsed.get("<OCR_WITH_REGION>", {}).get("labels", [])

    boxes = []
    for qb, lbl in zip(quad_boxes, labels):
        pts = [(qb[i], qb[i + 1]) for i in range(0, len(qb), 2)]
        boxes.append((pts, lbl, 1.0))

    full_text = " ".join(labels).strip()
    return {"full_text": full_text, "boxes": boxes}
#!/usr/bin/env python3
"""Isolated WeDetect-Ref grounding HTTP service (runs in the ``wedetect`` conda env).

Loads WeDetect-Uni (class-agnostic proposals) + WeDetect-Ref (Qwen3-VL re-ranker) ONCE
and serves referring-expression grounding:

    POST /ground  {image_path, query, score_thre, topk}
      -> {boxes: [[y0, x0, y1, x1] on 0-1000 grid], scores: [...]}
    GET  /health  -> 200 "ok"

The upstream package (``models/vendor/WeDetect``, GPL-v3) is imported ONLY inside this
process; the memstrata side talks to it via stdlib HTTP (``wedetect_client.py``), so no
GPL code enters the method package. Inference logic mirrors ``infer_wedetect_ref_sdpa.py``.

Launch via ``serve_wedetect.sh`` (sets the env + repo/weights paths).
"""

from __future__ import annotations

import argparse
import copy
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import torch
from PIL import Image


# Populated by _load_models() and read by the request handler.
_STATE: dict = {}
_LOCK = threading.Lock()  # GPU inference is serialized (memstrata acquires one crop at a time).


def _remap_uni_checkpoint(checkpoint: dict) -> dict:
    """Key remap for the Uni detector checkpoint (verbatim from the upstream infer script)."""
    keys = list(checkpoint.keys())
    for key in keys:
        if "backbone" in key:
            new_key = key.replace("backbone.image_model.model.", "backbone.")
            checkpoint[new_key] = checkpoint.pop(key)
    keys = list(checkpoint.keys())
    for key in keys:
        if "bbox_head" in key:
            new_key = key.replace("bbox_head.head_module.", "bbox_head.")
            new_key = new_key.replace("0.2.", "0.6.")
            new_key = new_key.replace("1.2.", "1.6.")
            new_key = new_key.replace("2.2.", "2.6.")
            new_key = new_key.replace("1.bn", "4")
            new_key = new_key.replace("1.conv", "3")
            new_key = new_key.replace("0.bn", "1")
            new_key = new_key.replace("0.conv", "0")
            checkpoint[new_key] = checkpoint.pop(key)
    return checkpoint


def _load_models(repo: str, ref_ckpt: str, uni_ckpt: str, attn: str) -> None:
    if repo not in sys.path:
        sys.path.insert(0, repo)
    from wedetect_ref.models.qwen3vl_referring import (  # type: ignore
        Qwen3VLGroundingForConditionalGeneration,
    )
    from wedetect_ref.models.vision_process import process_vision_info  # type: ignore
    from transformers import AutoProcessor
    from generate_proposal import SimpleYOLOWorldDetector  # type: ignore

    model_size = "base" if "base" in uni_ckpt else "large"
    det = SimpleYOLOWorldDetector(
        backbone_size=model_size, prompt_dim=768, num_prompts=256, num_proposals=100
    )
    checkpoint = _remap_uni_checkpoint(torch.load(uni_ckpt, map_location="cpu"))
    det = det.cuda().eval()
    det.load_state_dict(checkpoint, strict=False)

    model = Qwen3VLGroundingForConditionalGeneration.from_pretrained(
        ref_ckpt, torch_dtype=torch.bfloat16, attn_implementation=attn
    )
    processor = AutoProcessor.from_pretrained(ref_ckpt)
    object_token_index = processor.tokenizer.convert_tokens_to_ids("<object>")
    model.model.object_token_id = object_token_index
    model = model.cuda().eval()

    _STATE.update(
        det=det,
        model=model,
        processor=processor,
        process_vision_info=process_vision_info,
        object_token_index=object_token_index,
    )


def _ground(image_path: str, query: str, score_thre: float, topk: int) -> tuple[list, list]:
    det = _STATE["det"]
    model = _STATE["model"]
    processor = _STATE["processor"]
    process_vision_info = _STATE["process_vision_info"]
    object_token_index = _STATE["object_token_index"]

    image = Image.open(image_path).convert("RGB")
    width, height = image.size

    with torch.no_grad():
        outputs = det([image_path])
    proposals_px = outputs[0]["bboxes"].float()
    if proposals_px.numel() == 0:
        return [], []
    num_proposals = int(proposals_px.shape[0])
    proposal_str = "<object>" * num_proposals

    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "image": copy.deepcopy(image)},
                {"type": "text", "text": 'Please detect the "%s" in the image' % query},
            ],
        },
        {"role": "assistant", "content": [{"type": "text", "text": proposal_str}]},
    ]
    image_inputs, video_inputs = process_vision_info(messages, image_patch_size=16)
    texts = [processor.apply_chat_template(messages, tokenize=False)]
    model_inputs = processor(
        text=texts, images=image_inputs, videos=video_inputs,
        return_tensors="pt", padding=True, do_resize=False,
    ).to(model.device)
    proposals = [proposals_px.cuda().to(model.dtype)]

    with torch.inference_mode():
        pred = model(
            **model_inputs,
            bboxes=copy.deepcopy(proposals),
            ori_shapes=[image.size],
            bboxes_id=object_token_index,
            image_inputs=image_inputs,
        )
    proposal_positions = model_inputs["input_ids"] == object_token_index
    scores = pred.logits.sigmoid()[proposal_positions].view(-1)
    boxes_px = proposals_px.clone().float()

    if score_thre is not None and score_thre >= 0:
        mask = scores > score_thre
        sel_scores = scores[mask]
        sel_boxes = boxes_px[mask]
        if topk and int(sel_scores.numel()) > topk:
            vals, idx = torch.topk(sel_scores, topk)
            sel_scores, sel_boxes = vals, sel_boxes[idx]
        else:
            order = torch.argsort(sel_scores, descending=True)
            sel_scores, sel_boxes = sel_scores[order], sel_boxes[order]
    else:
        k = max(1, topk or 1)
        vals, idx = torch.topk(scores, min(k, int(scores.numel())))
        sel_scores, sel_boxes = vals, boxes_px[idx]

    out_boxes: list[list[int]] = []
    for box in sel_boxes.cpu().tolist():
        x0, y0, x1, y1 = box
        out_boxes.append([
            max(0, min(1000, round(y0 / height * 1000))),
            max(0, min(1000, round(x0 / width * 1000))),
            max(0, min(1000, round(y1 / height * 1000))),
            max(0, min(1000, round(x1 / width * 1000))),
        ])
    return out_boxes, [float(s) for s in sel_scores.cpu().tolist()]


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args) -> None:  # keep stderr quiet
        return

    def _send(self, code: int, payload: dict | str) -> None:
        body = (payload if isinstance(payload, str) else json.dumps(payload)).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self) -> None:
        if self.path.rstrip("/") == "/health":
            self._send(200, {"status": "ok"})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self) -> None:
        if self.path.rstrip("/") != "/ground":
            self._send(404, {"error": "not found"})
            return
        try:
            length = int(self.headers.get("Content-Length", 0))
            req = json.loads(self.rfile.read(length).decode("utf-8"))
        except Exception as exc:  # noqa: BLE001
            self._send(400, {"error": f"bad request: {exc}"})
            return
        image_path = str(req.get("image_path", ""))
        query = str(req.get("query", "")).strip()
        score_thre = req.get("score_thre", 0.25)
        topk = int(req.get("topk", 5) or 5)
        if not image_path or not query:
            self._send(400, {"error": "image_path and query required"})
            return
        try:
            with _LOCK:
                boxes, scores = _ground(image_path, query, float(score_thre), topk)
            self._send(200, {"boxes": boxes, "scores": scores})
        except Exception as exc:  # noqa: BLE001
            self._send(500, {"error": f"{type(exc).__name__}: {exc}"})


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--repo", required=True, help="path to models/vendor/WeDetect")
    parser.add_argument("--ref_checkpoint", required=True)
    parser.add_argument("--uni_checkpoint", required=True)
    parser.add_argument("--attn", default="sdpa", choices=["sdpa", "flash_attention_2", "eager"])
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8710)
    args = parser.parse_args()

    print(f"[serve_wedetect] loading models (attn={args.attn}) ...", flush=True)
    _load_models(args.repo, args.ref_checkpoint, args.uni_checkpoint, args.attn)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[serve_wedetect] ready at http://{args.host}:{args.port} (/health, /ground)", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

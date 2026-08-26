import os
import urllib.request
from pathlib import Path
import cv2
import numpy as np
import torch

from memstrata.lib.weights import weights_root
from .transnetv2_pytorch import TransNetV2 as PyTorchTransNetV2


class TransNetV2:
    """Wrapper class for TransNetV2 model loading, inference, and auto-downloading."""

    def __init__(
        self,
        weights_path: str | Path | None = None,
        device: str = "cuda",
        threshold: float = 0.5,
    ) -> None:
        if weights_path is None:
            self.weights_path = (
                weights_root()
                / "shot_boundary_detection"
                / "TransNetV2"
                / "transnetv2-pytorch-weights.pth"
            )
        else:
            self.weights_path = Path(weights_path)

        self.device_str = device
        self.threshold = threshold

        if self.device_str == "cuda" and torch.cuda.is_available():
            self.device = torch.device("cuda")
        else:
            self.device = torch.device("cpu")

        self._ensure_weights_exist()

        self.model = PyTorchTransNetV2()
        state_dict = torch.load(self.weights_path, map_location=self.device)
        self.model.load_state_dict(state_dict)
        self.model.eval()
        self.model.to(self.device)

    def _ensure_weights_exist(self) -> None:
        """Check if weights exist locally; if not, download them from Hugging Face."""
        if not self.weights_path.exists():
            self.weights_path.parent.mkdir(parents=True, exist_ok=True)
            url = "https://huggingface.co/Sn4kehead/TransNetV2/resolve/main/transnetv2-pytorch-weights.pth"
            print(f"Downloading TransNetV2 weights from {url} to {self.weights_path}...")
            try:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64)"}
                )
                with urllib.request.urlopen(req) as response, open(self.weights_path, "wb") as out_file:
                    meta = response.info()
                    file_size = int(meta.get("Content-Length", 0))
                    print(f"File size: {file_size / (1024 * 1024):.2f} MB")

                    block_size = 8192
                    downloaded = 0
                    while True:
                        buffer = response.read(block_size)
                        if not buffer:
                            break
                        downloaded += len(buffer)
                        out_file.write(buffer)
                        if file_size > 0:
                            percent = downloaded * 100 / file_size
                            if downloaded % (block_size * 100) == 0 or downloaded == file_size:
                                print(f"Downloaded: {percent:.1f}% ({downloaded / (1024 * 1024):.2f} MB)", end="\r")
                    print("\nDownload complete.")
            except Exception as e:
                if self.weights_path.exists():
                    self.weights_path.unlink()
                raise RuntimeError(f"Failed to download TransNetV2 weights: {e}") from e

    def predict_video(self, video_path: str | Path) -> tuple[np.ndarray, list[tuple[float, float]]]:
        """Predict scenes for a video."""
        video_path = str(video_path)
        frames, fps, total_frames, frame_size = self._extract_frames(video_path)
        if len(frames) == 0:
            return np.array([]), []

        predictions = self._run_inference(frames)
        scenes_frames = self._predictions_to_scenes(predictions)

        scenes_seconds = []
        for start_frame, end_frame in scenes_frames:
            start_sec = start_frame / fps
            end_sec = end_frame / fps
            scenes_seconds.append((start_sec, end_sec))

        return predictions, scenes_seconds

    def _extract_frames(self, video_path: str, target_height: int = 27, target_width: int = 48):
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")

        frames = []
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

        orig_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        orig_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_size = (orig_width, orig_height)

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            resized = cv2.resize(frame, (target_width, target_height))
            frames.append(resized)

        cap.release()
        return np.array(frames, dtype=np.uint8), fps, total_frames, frame_size

    def _run_inference(self, frames: np.ndarray) -> np.ndarray:
        predictions = []
        # Batch inference
        batch_size = 100
        for i in range(0, len(frames), batch_size):
            batch = frames[i : i + batch_size]
            # Ensure we have at least 100 frames for lookup window if needed, or pad
            # TransNetV2 expects [B, T, H, W, C]
            # Here we just pass the batch directly as a single sequence of length T
            # Actually, TransNetV2 takes [1, T, H, W, 3]
            tensor = torch.from_numpy(batch).unsqueeze(0).to(self.device)
            with torch.no_grad():
                logits = self.model(tensor)
                if isinstance(logits, tuple):
                    logits = logits[0]
                probs = torch.sigmoid(logits).squeeze(0).cpu().numpy()
                predictions.extend(probs)
        return np.array(predictions)

    def _predictions_to_scenes(self, predictions: np.ndarray) -> list[tuple[int, int]]:
        predictions = (predictions > self.threshold).astype(np.uint8)
        scenes = []
        start = 0
        for i in range(len(predictions)):
            if predictions[i] == 1:
                scenes.append((start, i))
                start = i + 1
        if start < len(predictions):
            scenes.append((start, len(predictions) - 1))
        return scenes

"""On-device leaf classifier: ONNX Runtime inference (CPU, no internet, no PyTorch needed)."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

MEAN = np.array([0.485, 0.456, 0.406], dtype=np.float32)
STD = np.array([0.229, 0.224, 0.225], dtype=np.float32)
IMG = 224
MODEL_DIR = Path(__file__).parent / "models"


def preprocess(img: Image.Image):
    """Resize shorter side to 256, centre-crop 224, normalise (identical to training eval transform)."""
    img = ImageOps.exif_transpose(img).convert("RGB")
    w, h = img.size
    s = 256 / min(w, h)
    img = img.resize((max(IMG, round(w * s)), max(IMG, round(h * s))), Image.BILINEAR)
    w, h = img.size
    l, t = (w - IMG) // 2, (h - IMG) // 2
    crop = img.crop((l, t, l + IMG, t + IMG))
    x = (np.asarray(crop, dtype=np.float32) / 255.0 - MEAN) / STD
    return x.transpose(2, 0, 1)[None].astype(np.float32), crop


def softmax(z):
    z = z - z.max(axis=-1, keepdims=True)
    e = np.exp(z)
    return e / e.sum(axis=-1, keepdims=True)


class DiseaseClassifier:
    def __init__(self, model_dir=MODEL_DIR, conf_threshold=0.70, margin_threshold=0.15):
        import onnxruntime as ort
        model_dir = Path(model_dir)
        self.sess = ort.InferenceSession(str(model_dir / "leaf_model.onnx"), providers=["CPUExecutionProvider"])
        self.input_name = self.sess.get_inputs()[0].name
        self.classes = json.loads((model_dir / "labels.json").read_text())["classes"]
        self.conf_threshold, self.margin_threshold = conf_threshold, margin_threshold

    def _logits(self, x):
        return self.sess.run(None, {self.input_name: x})[0]

    def predict(self, img: Image.Image, top_k=3):
        x, crop = preprocess(img)
        p = softmax(self._logits(x)[0])
        order = np.argsort(-p)[:top_k]
        top = float(p[order[0]])
        margin = top - float(p[order[1]]) if len(order) > 1 else top
        uncertain = top < self.conf_threshold or margin < self.margin_threshold
        return {"x": x, "crop": crop, "top": [(self.classes[i], float(p[i])) for i in order],
                "confidence": top, "uncertain": bool(uncertain), "class_index": int(order[0])}

    def explain(self, x, class_index, patch=56, stride=28):
        """Occlusion sensitivity: how much does the probability drop when a region is hidden? (model-agnostic, ONNX only)"""
        base = softmax(self._logits(x)[0])[class_index]
        pos = list(range(0, IMG - patch + 1, stride))
        batch, coords = [], []
        for i in pos:
            for j in pos:
                xo = x.copy()
                xo[:, :, i:i + patch, j:j + patch] = 0.0  # 0 == dataset mean after normalisation
                batch.append(xo[0])
                coords.append((i, j))
        probs = softmax(self._logits(np.stack(batch).astype(np.float32)))[:, class_index]
        heat, cnt = np.zeros((IMG, IMG), np.float32), np.zeros((IMG, IMG), np.float32)
        for (i, j), pr in zip(coords, probs):
            heat[i:i + patch, j:j + patch] += base - pr
            cnt[i:i + patch, j:j + patch] += 1
        heat = np.clip(heat / np.maximum(cnt, 1), 0, None)
        return heat / (heat.max() + 1e-8)


def overlay(crop: Image.Image, heat: np.ndarray, alpha=0.55) -> Image.Image:
    base = np.asarray(crop.convert("RGB"), dtype=np.float32)
    red = np.zeros_like(base)
    red[..., 0], red[..., 1] = 255.0, 60.0 * (1 - heat)
    a = (alpha * heat)[..., None]
    return Image.fromarray((base * (1 - a) + red * a).astype(np.uint8))

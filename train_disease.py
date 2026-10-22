"""Train + evaluate + export the edge leaf-disease model (MobileNetV3-Small -> ONNX).

Usage
  python train_disease.py --data data/PlantVillage --epochs 8              # full / GPU
  python train_disease.py --data data/PlantVillage --epochs 5 --max-per-class 300   # fast on CPU laptop
  python train_disease.py --smoke-test                                      # 1-minute pipeline check, no dataset needed
Folder layout expected (ImageFolder):  data/PlantVillage/<class_name>/*.jpg
"""
import argparse
import json
import random
import shutil
import tempfile
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, models, transforms

IMG, MEAN, STD = 224, [0.485, 0.456, 0.406], [0.229, 0.224, 0.225]
ROOT = Path(__file__).parent
OUT = ROOT / "models"
REPORTS = ROOT / "reports"


def get_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    if getattr(torch.backends, "mps", None) and torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


def make_loaders(data_dir, bs, workers, max_per_class, val_frac=0.15, seed=42):
    # Strong "field-like" augmentation so the model copes with real phone photos, not just clean lab images
    train_tf = transforms.Compose([
        transforms.RandomResizedCrop(IMG, scale=(0.45, 1.0)),
        transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ColorJitter(0.45, 0.45, 0.45, 0.05),
        transforms.RandomApply([transforms.GaussianBlur(5, (0.1, 2.0))], p=0.3),
        transforms.ToTensor(), transforms.Normalize(MEAN, STD),
        transforms.RandomErasing(p=0.25, scale=(0.02, 0.15)),
    ])
    eval_tf = transforms.Compose([transforms.Resize(256), transforms.CenterCrop(IMG),
                                  transforms.ToTensor(), transforms.Normalize(MEAN, STD)])
    ds_tr, ds_ev = datasets.ImageFolder(data_dir, train_tf), datasets.ImageFolder(data_dir, eval_tf)
    rng = random.Random(seed)
    by_class = {}
    for i, t in enumerate(ds_tr.targets):
        by_class.setdefault(t, []).append(i)
    tr_idx, va_idx = [], []
    for t, idxs in by_class.items():
        rng.shuffle(idxs)
        if max_per_class:
            idxs = idxs[:max_per_class]
        nv = max(1, int(len(idxs) * val_frac))
        va_idx += idxs[:nv]
        tr_idx += idxs[nv:]
    kw = dict(batch_size=bs, num_workers=workers, pin_memory=torch.cuda.is_available())
    return (DataLoader(Subset(ds_tr, tr_idx), shuffle=True, **kw),
            DataLoader(Subset(ds_ev, va_idx), shuffle=False, **kw), ds_tr.classes)


def build_model(n_classes, pretrained=True):
    try:
        w = models.MobileNet_V3_Small_Weights.DEFAULT if pretrained else None
        m = models.mobilenet_v3_small(weights=w)
    except Exception as e:  # no internet for pretrained weights
        print(f"[warn] could not load pretrained weights ({e}); training from scratch (lower accuracy).")
        m = models.mobilenet_v3_small(weights=None)
    m.classifier[3] = nn.Linear(m.classifier[3].in_features, n_classes)
    return m


def run_epoch(model, loader, device, loss_fn, opt=None):
    train = opt is not None
    model.train(train)
    tot, correct, loss_sum = 0, 0, 0.0
    with torch.set_grad_enabled(train):
        for x, y in loader:
            x, y = x.to(device), y.to(device)
            out = model(x)
            loss = loss_fn(out, y)
            if train:
                opt.zero_grad()
                loss.backward()
                opt.step()
            loss_sum += loss.item() * len(y)
            correct += (out.argmax(1) == y).sum().item()
            tot += len(y)
    return loss_sum / max(tot, 1), correct / max(tot, 1)


def evaluate(model, loader, device, classes):
    from sklearn.metrics import classification_report, confusion_matrix
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    model.eval()
    ys, ps = [], []
    with torch.no_grad():
        for x, y in loader:
            ps += model(x.to(device)).argmax(1).cpu().tolist()
            ys += y.tolist()
    labels = list(range(len(classes)))
    rep = classification_report(ys, ps, labels=labels, target_names=classes, digits=3, zero_division=0)
    REPORTS.mkdir(exist_ok=True)
    (REPORTS / "classification_report.txt").write_text(rep)
    cm = confusion_matrix(ys, ps, labels=labels)
    fig, ax = plt.subplots(figsize=(10, 9))
    ax.imshow(cm, cmap="Blues")
    ax.set_title("Confusion matrix (validation)")
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    fig.tight_layout()
    fig.savefig(REPORTS / "confusion_matrix.png", dpi=120)
    plt.close(fig)
    print(rep)


def export_onnx(model, classes, val_acc):
    import onnxruntime as ort
    model.eval().cpu()
    OUT.mkdir(exist_ok=True)
    path = OUT / "leaf_model.onnx"
    dummy = torch.randn(1, 3, IMG, IMG)
    kw = dict(input_names=["input"], output_names=["logits"], opset_version=17,
              dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}})
    try:
        torch.onnx.export(model, dummy, str(path), dynamo=False, **kw)  # torch >= 2.5
    except TypeError:
        torch.onnx.export(model, dummy, str(path), **kw)
    sess = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    with torch.no_grad():
        ref = model(dummy).numpy()
    got = sess.run(None, {"input": dummy.numpy()})[0]
    diff = float(np.abs(ref - got).max())
    t0 = time.time()
    for _ in range(20):
        sess.run(None, {"input": dummy.numpy()})
    ms = (time.time() - t0) / 20 * 1000
    size_mb = path.stat().st_size / 1e6
    (OUT / "labels.json").write_text(json.dumps({"classes": classes, "val_accuracy": val_acc,
                                                 "size_mb": round(size_mb, 2), "cpu_ms_per_image": round(ms, 1)},
                                                indent=2))
    torch.save(model.state_dict(), OUT / "leaf_model.pt")
    print(f"Exported {path}  size={size_mb:.1f} MB  CPU latency={ms:.0f} ms/img  torch-vs-onnx max diff={diff:.2e}")


def make_smoke_dataset(root: Path):
    from PIL import Image
    rng = np.random.default_rng(0)
    for cls, base in {"Tomato___healthy": (40, 160, 60), "Tomato___Late_blight": (120, 90, 50),
                      "Potato___Early_blight": (150, 120, 40)}.items():
        (root / cls).mkdir(parents=True)
        for i in range(24):
            arr = np.clip(np.array(base) + rng.normal(0, 25, (128, 128, 3)), 0, 255).astype(np.uint8)
            Image.fromarray(arr).save(root / cls / f"{i}.jpg")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/PlantVillage")
    ap.add_argument("--epochs", type=int, default=8)
    ap.add_argument("--warmup", type=int, default=2, help="epochs with frozen backbone")
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--workers", type=int, default=2)
    ap.add_argument("--max-per-class", type=int, default=0, help="0 = use all images")
    ap.add_argument("--smoke-test", action="store_true")
    a = ap.parse_args()

    tmp = None
    if a.smoke_test:
        tmp = Path(tempfile.mkdtemp())
        make_smoke_dataset(tmp)
        a.data, a.epochs, a.warmup, a.workers, a.batch = str(tmp), 1, 1, 0, 8
    if not Path(a.data).exists():
        raise SystemExit(f"Dataset folder not found: {a.data}  (see README step 3)")

    torch.manual_seed(42)
    dev = get_device()
    print("Device:", dev)
    tr, va, classes = make_loaders(a.data, a.batch, a.workers, a.max_per_class)
    print(f"{len(classes)} classes | train batches={len(tr)} val batches={len(va)}")
    model = build_model(len(classes), pretrained=not a.smoke_test).to(dev)
    loss_fn = nn.CrossEntropyLoss(label_smoothing=0.1)

    best, best_state = 0.0, None
    for ep in range(a.epochs):
        frozen = ep < a.warmup
        for p in model.features.parameters():
            p.requires_grad = not frozen
        params = [p for p in model.parameters() if p.requires_grad]
        opt = torch.optim.AdamW(params, lr=1e-3 if frozen else 3e-4, weight_decay=1e-4)
        t0 = time.time()
        tl, ta = run_epoch(model, tr, dev, loss_fn, opt)
        vl, vacc = run_epoch(model, va, dev, loss_fn)
        print(f"epoch {ep + 1}/{a.epochs} {'(frozen)' if frozen else ''} train acc {ta:.3f} | "
              f"val loss {vl:.3f} acc {vacc:.3f} | {time.time() - t0:.0f}s")
        if vacc >= best:
            best, best_state = vacc, {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
    model.load_state_dict(best_state)
    model.to(dev)
    evaluate(model, va, dev, classes)
    export_onnx(model, classes, round(best, 4))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)
        print("SMOKE TEST PASSED (this model is untrained toy data - retrain on PlantVillage before demo!)")


if __name__ == "__main__":
    main()

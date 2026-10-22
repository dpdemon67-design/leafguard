"""Test on REAL field photos (not clean dataset images).
Put photos in  data/field_test/<class_name>/*.jpg  using the same class names as training.
Reports accuracy, how often the model said 'uncertain', and accuracy on confident predictions."""
import sys
from pathlib import Path

from PIL import Image

from disease_model import DiseaseClassifier

root = Path(sys.argv[1] if len(sys.argv) > 1 else "data/field_test")
clf = DiseaseClassifier()
n = correct = unc = conf_n = conf_ok = 0
for cdir in sorted(p for p in root.iterdir() if p.is_dir()):
    for f in list(cdir.glob("*.jpg")) + list(cdir.glob("*.jpeg")) + list(cdir.glob("*.png")):
        r = clf.predict(Image.open(f))
        n += 1
        ok = r["top"][0][0] == cdir.name
        correct += ok
        if r["uncertain"]:
            unc += 1
        else:
            conf_n += 1
            conf_ok += ok
if n == 0:
    raise SystemExit(f"No images found under {root}")
print(f"Images: {n} | accuracy (all): {correct / n:.1%} | flagged uncertain: {unc / n:.1%}")
if conf_n:
    print(f"Accuracy on confident predictions: {conf_ok / conf_n:.1%}  ({conf_n} images)")

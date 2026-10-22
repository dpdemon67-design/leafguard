# LeafGuard – Edge AI Leaf Disease & Localized Pest Forecaster (AI Idea Forge 2026, PS-4)

Two **separate** pathways, exactly as the problem statement requires:
- **A – What is on this leaf?** MobileNetV3-Small → ONNX, runs fully offline on CPU, with an "Uncertain" state and an occlusion heat-map explanation.
- **B – What may happen locally?** Crop + 7-day weather + 3-day forecast → Low/Moderate/High pest risk with exact factor contributions, provenance (source, time, location) and a conservative rule.
Guidance is in English / Gujarati / Hindi and never prescribes chemicals; it always defers to KVK / agriculture officers.

## Setup (Python 3.10–3.12)
```
python -m venv venv
venv\Scripts\activate          # Windows   |   source venv/bin/activate  (Mac/Linux)
pip install -r requirements.txt
```

## Step 1 – Check the pipeline in 1 minute (no dataset needed)
```
python train_disease.py --smoke-test
python pest_forecast.py train
streamlit run app.py
```
(The smoke-test model is a toy – retrain properly in Step 3.)

## Step 2 – Get the dataset
Download **PlantVillage** from Kaggle ("abdallahalidev/plantvillage-dataset"), and use its `color` folder.
Final layout:
```
data/PlantVillage/Tomato___Late_blight/*.jpg
data/PlantVillage/Potato___Early_blight/*.jpg ...
```
Tip: keep only crops relevant to Gujarat (tomato, potato, cotton-like leaf classes if available, etc.) by deleting other class folders – fewer classes = better accuracy and a stronger story.

## Step 3 – Train (≈10–20 min on CPU with the cap, ≈5 min on GPU)
```
python train_disease.py --data data/PlantVillage --epochs 6 --max-per-class 300     # laptop CPU
python train_disease.py --data data/PlantVillage --epochs 10                        # GPU / Colab
```
Outputs: `models/leaf_model.onnx`, `models/labels.json` (size, latency, accuracy), `reports/classification_report.txt`, `reports/confusion_matrix.png`.

## Step 4 – Test on REAL field photos (judges will ask!)
Take 5–10 phone photos per disease from real plants/Google Images with proper source, put them in `data/field_test/<class_name>/`, then:
```
python evaluate_field.py data/field_test
```
Quote both "accuracy" and "flagged uncertain". If field accuracy is lower than PlantVillage, say so – and show that the Uncertain state protects the farmer. This honesty scores well.

## Step 5 – Real pest data (optional but a big win)
Create `data/pest_observations.csv` with columns:
`crop,temp_suitability,mean_rh,rain_total,humid_warm_days,hot_dry_days,fc_mean_rh,fc_rain_total,risk`
(risk ∈ Low / Moderate / High, from KVK / ICAR-NCIPM / state agri-dept scouting reports). Then `python pest_forecast.py train`.
Without this file the app trains on **synthetic agronomy-rule data** and says so on screen – never claim it is field-validated.

## Step 6 – Offline demo
`streamlit run app.py`, then turn Wi-Fi OFF. Tab 1 works fully; Tab 2 works with "Offline – enter values" or a cached location (fetch once online first).
For phone demo: laptop hotspot → open `http://<laptop-ip>:8501` on the phone, use the camera input.

## 3-minute demo script
1. Problem (late detection, weak internet). 2. Wi-Fi off → upload a leaf → disease + confidence + heat-map + Gujarati advice.
3. Upload a blurry/non-leaf image → **Uncertain** state. 4. Tab 2 → Ahmedabad + Tomato → risk level, factors, data source/time.
5. Show metrics: model size, ms/image, confusion matrix, field-test result, forecaster vs. humidity-only baseline.
6. Limitations + next steps (ONNX Runtime Web / TFLite Android app, real scouting data, more crops).

## Requirement checklist (from the PS)
| PS point | Where |
|---|---|
| Edge-capable disease model | MobileNetV3-Small ONNX, size + latency in sidebar |
| Works with weak connectivity | ONNX Runtime on CPU; manual/cached weather |
| Separate pest forecast | `pest_forecast.py`, Tab 2 |
| Explain the risk | Factor table + leaf heat-map |
| Advise cautiously / defer when unsure | Uncertain state, conservative bump, KVK note, no chemicals named |
| Gujarati / Hindi guidance | `guidance.py` (have a native speaker proofread!) |
| Test on real field photos | `evaluate_field.py` |

## Troubleshooting
- `onnxruntime` / `onnx` install error → use Python 3.11 or 3.12.
- Pretrained weights download fails → connect to internet once; otherwise it trains from scratch (lower accuracy).
- scikit-learn version warning on pest model → it auto-retrains.
- Windows DataLoader error → add `--workers 0`.

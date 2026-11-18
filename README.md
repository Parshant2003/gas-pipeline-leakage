# Gas Pipeline Leakage Detection (beginner-friendly)

Upload a photo of a pipeline -> the app draws red boxes around gas / vapour leaks and shows a confidence %.

| Part | Technology |
|---|---|
| Model training | Python + YOLOv8n (Ultralytics) |
| Backend | Node.js + Express, model run with **ONNX Runtime** (`onnxruntime-node`) |
| Frontend | plain HTML, CSS, JavaScript |

No database, no login, no Docker. Everything runs on your laptop.

## Folder structure

```
gas-leak-detection/
├── data/gas_grouped/      dataset (train / valid / test) + data.yaml
├── ml/                    Python: train.py, evaluate.py, export_onnx.py, common.py, requirements.txt
├── models/                best.pt, best.onnx, model_meta.json, metrics.json, plots/
├── backend/               Node.js: server.js (API), detector.js (ONNX inference), package.json
├── frontend/              index.html, style.css, app.js
└── dataset_tools/         (optional) scripts that built the leak-free train/valid/test split
```

## Run the app (Windows 11) - the trained model is already included

1. Install **Node.js LTS** (18 or newer) from https://nodejs.org
2. Open **PowerShell** in the project folder and run:

```powershell
cd backend
npm install
npm start
```

3. Open **http://localhost:3000** in your browser, choose an image, click **Detect gas**.

If port 3000 is busy: `$env:PORT=3001; npm start` (PowerShell) and open http://localhost:3001.

## Re-train the model yourself (optional)

Needs **Python 3.10 - 3.12** (https://www.python.org, tick "Add Python to PATH") and the dataset in `data/gas_grouped/`
(train / valid / test folders, see `data/README.txt`).

```powershell
# in the project folder
python -m venv venv
venv\Scripts\Activate.ps1          # if PowerShell blocks this:  Set-ExecutionPolicy -Scope CurrentUser RemoteSigned
pip install -r ml\requirements.txt

python ml\train.py                 # 1) train YOLOv8n  -> models\best.pt   (30 epochs; ~1 hour on a laptop CPU)
python ml\evaluate.py             # 2) metrics on valid + test -> models\metrics.json
python ml\export_onnx.py          # 3) convert to ONNX -> models\best.onnx + models\model_meta.json
```

Then restart the backend (`npm start`). Training is not bit-for-bit repeatable on different hardware, so your numbers can differ slightly from the ones below.

## What each step does

* **train.py** - fine-tunes the pretrained YOLOv8n on the *train* images (416x416, 30 epochs). The *valid* images pick the best epoch. Augmentation (flip, small rotation/shift/zoom, colour change, mosaic) is done on the fly and only on training images. The *test* images are not used.
* **evaluate.py** - runs the best model on valid and test and saves Precision, Recall, F1, mAP@0.5 and mAP@0.5:0.95.
* **export_onnx.py** - converts `best.pt` to `best.onnx` and checks that ONNX gives the same output as PyTorch.
* **backend/detector.js** - resizes the image to 416x416 (grey padding), runs the ONNX model, keeps boxes with confidence >= 25 %, removes overlapping duplicates (NMS) and converts the boxes back to the original image size.
* **backend/server.js** - `POST /api/detect` (upload), `GET /api/metrics`, serves the web page.
* **frontend/app.js** - previews the image, sends it to the API, draws the boxes on a canvas next to the original.

## API

```powershell
curl.exe -F "image=@C:\path\to\photo.jpg" http://localhost:3000/api/detect
```

```json
{
  "status": "Gas detected",
  "count": 1,
  "width": 640, "height": 640,
  "detections": [ { "label": "Gas", "x1": 105, "y1": 425, "x2": 627, "y2": 640, "confidence": 0.2561, "percent": 25.6 } ],
  "time_ms": 65
}
```
Errors return `{ "error": "..." }` (no file -> 400, not an image -> 400, bigger than 10 MB -> 413).

## Dataset

Roboflow "Gas" v3 (CC BY 4.0), re-split by *source photograph* so that the same photo never appears in two splits (the original split had this leakage).

| Split | Images | Boxes |
|---|---|---|
| train | 482 | 489 |
| valid | 103 | 103 |
| test | 103 | 103 |

One class: `Gas`. See `data/gas_grouped/dataset_report.md` for the checks. Each source photo exists in about 3 augmented copies, so the 103 test images are only 40 different photos.

## Results (measured, not estimated)

Model: YOLOv8n, 416 px, 30 epochs, best epoch chosen on valid. Numbers come from `ml/evaluate.py` (`models/metrics.json`).

| Split | Precision | Recall | F1 | mAP@0.5 | mAP@0.5:0.95 |
|---|---|---|---|---|---|
| valid | 0.764 | 0.689 | 0.725 | 0.749 | 0.333 |
| **test** | 0.661 | 0.748 | 0.702 | 0.735 | 0.350 |

(Precision / recall are at the confidence that gives the best F1.)

Checks on the final app (Node.js + ONNX Runtime, confidence 25 %, a box counts as correct if IoU >= 0.5), run on all 103 test images:

* 71 correct boxes, 41 wrong boxes, 32 missed -> precision 0.634, recall 0.689, F1 0.660; a correct gas box was found in 71 of 103 images.
* Compared with PyTorch on the same images: the same number of boxes in 101 of 103 images; the main boxes overlap with a median IoU of 0.986. In the other 2 images Node showed one extra weak box (confidence about 26 %), just above the threshold, caused by tiny resize differences.
* Speed: about 55 - 65 ms per image on a 2-core cloud CPU.

### Honest limits

* mAP@0.5:0.95 is only about 0.35: gas/steam clouds have fuzzy edges, so the box position is often not exact.
* The images are mostly stock photos (some with watermarks), and there are only about 310 different photos in total. Do not expect the same accuracy on real camera footage.
* The dataset has no "clean" pipeline pictures without gas, so false alarms on normal images were **not measured**. Steam, smoke, fog or dust may be reported as gas. A leak can also be missed (recall about 0.7).
* The 3 images that had unusable polygon labels (a water leak) are kept in train as "no gas" examples.
* This is a learning / demonstration project, not a safety system.

## Troubleshooting

| Problem | Fix |
|---|---|
| `Could not load the model` | `models/best.onnx` and `models/model_meta.json` must exist (run `ml\export_onnx.py`) |
| `npm install` fails on `sharp` / `onnxruntime-node` | use Node.js LTS 64-bit, delete `backend\node_modules`, run `npm install` again |
| Port already in use | `$env:PORT=3001; npm start` |
| `FileNotFoundError ... data.yaml` | the dataset must be in `data\gas_grouped\` |
| Training is slow | normal on CPU; use a GPU PC or lower `EPOCHS` in `ml/train.py` |

"""Step 2 - evaluate models/best.pt on the valid and test splits and save models/metrics.json.

Run from the project folder:   python ml/evaluate.py
All numbers come straight from Ultralytics' validation code - nothing is typed in by hand.
"""
import json
import shutil

from ultralytics import YOLO

from common import DATA_DIR, MODELS_DIR, RUNS_DIR, runtime_data_yaml

def main():
    model = YOLO(MODELS_DIR / "best.pt")
    results = {"model": "YOLOv8n", "image_size": 416, "classes": list(model.names.values()), "splits": {}}

    for split, name in (("val", "valid"), ("test", "test")):
        m = model.val(data=runtime_data_yaml(), split=split, imgsz=416, conf=0.001, plots=True,
                      project=str(RUNS_DIR), name=f"eval_{name}", exist_ok=True)
        p, r = float(m.box.mp), float(m.box.mr)          # precision / recall (at the best-F1 confidence)
        results["splits"][name] = {
            "images": len(list((DATA_DIR / name / "images").iterdir())),
            "precision": round(p, 4),
            "recall": round(r, 4),
            "f1": round(2 * p * r / (p + r), 4) if p + r else 0.0,
            "mAP50": round(float(m.box.map50), 4),          # mAP at IoU 0.5
            "mAP50_95": round(float(m.box.map), 4),         # mAP averaged over IoU 0.5 ... 0.95
        }
        print(name, results["splits"][name])

    (MODELS_DIR / "metrics.json").write_text(json.dumps(results, indent=2))

    # keep the useful plots next to the model
    (MODELS_DIR / "plots").mkdir(exist_ok=True)
    for name in ("valid", "test"):
        for f in ("BoxPR_curve.png", "confusion_matrix.png"):
            src = RUNS_DIR / f"eval_{name}" / f
            if src.exists():
                shutil.copy2(src, MODELS_DIR / "plots" / f"{name}_{f}")
    print("Saved", MODELS_DIR / "metrics.json")


if __name__ == "__main__":   # required on Windows
    main()

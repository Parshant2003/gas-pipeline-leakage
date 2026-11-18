"""Step 1 - train YOLOv8n on the gas dataset.

Run from the project folder:   python ml/train.py
The train split is used for learning and the valid split to pick the best epoch.
The test split is NOT touched here (it is used once, in evaluate.py).
"""
import shutil

from ultralytics import YOLO

from common import MODELS_DIR, RUNS_DIR, runtime_data_yaml

EPOCHS = 30      # enough for a small dataset with a pretrained model
IMG_SIZE = 416   # small size = fast on a laptop CPU (gas boxes are large, so this is fine)
BATCH = 16


def main():
    model = YOLO("yolov8n.pt")   # YOLOv8 nano, pretrained on COCO (downloaded automatically the first time)

    model.train(
        data=runtime_data_yaml(),         # points to data/gas_grouped/data.yaml
        epochs=EPOCHS, imgsz=IMG_SIZE, batch=BATCH,
        patience=10,                      # stop early if valid mAP does not improve for 10 epochs
        workers=2, seed=42,
        project=str(RUNS_DIR), name="gas_yolov8n", exist_ok=True,
        cache="ram", close_mosaic=5,
        # data augmentation - Ultralytics applies it to the TRAIN images only
        fliplr=0.5, degrees=5, translate=0.1, scale=0.5,
        hsv_h=0.015, hsv_s=0.6, hsv_v=0.4, mosaic=1.0,
    )

    # save the best model (chosen on the valid split) to models/best.pt
    MODELS_DIR.mkdir(exist_ok=True)
    shutil.copy2(RUNS_DIR / "gas_yolov8n" / "weights" / "best.pt", MODELS_DIR / "best.pt")
    print("Saved", MODELS_DIR / "best.pt")


if __name__ == "__main__":   # this guard is required on Windows (data loader workers)
    main()

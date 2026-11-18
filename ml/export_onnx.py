"""Step 3 - convert models/best.pt to models/best.onnx for the Node.js backend.

Run from the project folder:   python ml/export_onnx.py
"""
import json

import numpy as np
import onnxruntime as ort
import torch
from ultralytics import YOLO

from common import MODELS_DIR

IMG_SIZE = 416   # must be the same size that was used for training

model = YOLO(MODELS_DIR / "best.pt")
model.export(format="onnx", imgsz=IMG_SIZE, opset=12, simplify=True, dynamic=False)   # writes models/best.onnx

# safety check: ONNX Runtime must give the same numbers as PyTorch
x = np.random.RandomState(0).rand(1, 3, IMG_SIZE, IMG_SIZE).astype(np.float32)
session = ort.InferenceSession(str(MODELS_DIR / "best.onnx"), providers=["CPUExecutionProvider"])
onnx_out = session.run(None, {session.get_inputs()[0].name: x})[0]
with torch.no_grad():
    torch_out = model.model.float().eval()(torch.from_numpy(x))
torch_out = (torch_out[0] if isinstance(torch_out, (list, tuple)) else torch_out).numpy()
diff = float(np.abs(onnx_out - torch_out).max())
print("output shape:", onnx_out.shape, "| max difference vs PyTorch:", diff)
assert diff < 0.01, "ONNX output does not match PyTorch!"

# the backend reads the image size and class names from this small file
(MODELS_DIR / "model_meta.json").write_text(json.dumps({"imgsz": IMG_SIZE, "names": list(model.names.values())}, indent=2))
print("Saved models/best.onnx and models/model_meta.json")

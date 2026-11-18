// Runs the YOLOv8n ONNX model. Three steps: prepare the image -> run the model -> read the boxes.
const fs = require('fs');
const path = require('path');
const ort = require('onnxruntime-node');
const sharp = require('sharp');

const MODEL_DIR = path.join(__dirname, '..', 'models');
let session, size, names;                       // filled in by loadModel()

async function loadModel() {
  const meta = JSON.parse(fs.readFileSync(path.join(MODEL_DIR, 'model_meta.json'), 'utf8'));
  size = meta.imgsz;                            // e.g. 416
  names = meta.names;                           // e.g. ['Gas']
  session = await ort.InferenceSession.create(path.join(MODEL_DIR, 'best.onnx'));
  return { size, names };
}

// Step 1: the model needs a square size x size picture. We shrink the image to fit and fill the
// leftover space with grey (this is called "letterbox", the same thing used during training).
async function prepareImage(buffer) {
  const { data, info } = await sharp(buffer).rotate().removeAlpha().raw().toBuffer({ resolveWithObject: true });
  const scale = Math.min(size / info.width, size / info.height);
  const w = Math.round(info.width * scale);
  const h = Math.round(info.height * scale);
  const left = Math.floor((size - w) / 2);
  const top = Math.floor((size - h) / 2);

  const pixels = await sharp(data, { raw: { width: info.width, height: info.height, channels: 3 } })
    .resize(w, h, { fit: 'fill' })
    .extend({ top, bottom: size - h - top, left, right: size - w - left, background: { r: 114, g: 114, b: 114 } })
    .raw()
    .toBuffer();

  // pixels are R,G,B,R,G,B,... (0-255). The model wants all R, then all G, then all B, scaled to 0-1.
  const input = new Float32Array(3 * size * size);
  const area = size * size;
  for (let i = 0; i < area; i++) {
    input[i] = pixels[i * 3] / 255;
    input[area + i] = pixels[i * 3 + 1] / 255;
    input[2 * area + i] = pixels[i * 3 + 2] / 255;
  }
  return { input, width: info.width, height: info.height, scale, left, top };
}

// overlap between two boxes (0 = none, 1 = identical)
function iou(a, b) {
  const w = Math.min(a.x2, b.x2) - Math.max(a.x1, b.x1);
  const h = Math.min(a.y2, b.y2) - Math.max(a.y1, b.y1);
  if (w <= 0 || h <= 0) return 0;
  const inter = w * h;
  return inter / ((a.x2 - a.x1) * (a.y2 - a.y1) + (b.x2 - b.x1) * (b.y2 - b.y1) - inter);
}

// Step 3 helper: the model often draws several boxes on the same gas cloud.
// Non-Maximum Suppression keeps the most confident one and removes boxes that overlap it too much.
function removeOverlaps(boxes, maxOverlap) {
  boxes.sort((a, b) => b.confidence - a.confidence);
  const kept = [];
  for (const box of boxes) {
    if (kept.every((k) => iou(k, box) < maxOverlap)) kept.push(box);
  }
  return kept;
}

async function detect(buffer, minConfidence = 0.25) {
  const start = Date.now();
  const img = await prepareImage(buffer);

  // Step 2: run the model. Output shape is [1, 5, N]: for each of the N candidate boxes it gives
  // centre-x, centre-y, width, height and the "Gas" score (all in the size x size picture).
  const tensor = new ort.Tensor('float32', img.input, [1, 3, size, size]);
  const result = await session.run({ [session.inputNames[0]]: tensor });
  const output = result[session.outputNames[0]];
  const n = output.dims[2];
  const d = output.data;

  const boxes = [];
  for (let i = 0; i < n; i++) {
    const score = d[4 * n + i];
    if (score < minConfidence) continue;                  // ignore weak guesses
    const cx = d[i], cy = d[n + i], w = d[2 * n + i], h = d[3 * n + i];
    // convert from the grey-padded square back to the pixels of the original image
    const clamp = (v, max) => Math.min(Math.max(v, 0), max);
    boxes.push({
      x1: clamp((cx - w / 2 - img.left) / img.scale, img.width),
      y1: clamp((cy - h / 2 - img.top) / img.scale, img.height),
      x2: clamp((cx + w / 2 - img.left) / img.scale, img.width),
      y2: clamp((cy + h / 2 - img.top) / img.scale, img.height),
      confidence: score,
    });
  }

  const detections = removeOverlaps(boxes, 0.45).map((b) => ({
    label: names[0],
    x1: Math.round(b.x1), y1: Math.round(b.y1), x2: Math.round(b.x2), y2: Math.round(b.y2),
    confidence: Number(b.confidence.toFixed(4)),
    percent: Number((b.confidence * 100).toFixed(1)),
  }));

  return {
    status: detections.length > 0 ? 'Gas detected' : 'No gas detected',
    count: detections.length,
    width: img.width,
    height: img.height,
    detections,
    time_ms: Date.now() - start,
  };
}

module.exports = { loadModel, detect };

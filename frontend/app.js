// Frontend logic: pick an image -> send it to /api/detect -> draw the boxes.
const fileInput = document.getElementById('fileInput');
const detectBtn = document.getElementById('detectBtn');
const message = document.getElementById('message');
const resultCard = document.getElementById('resultCard');
const statusEl = document.getElementById('status');
const listEl = document.getElementById('detectionList');
const timingEl = document.getElementById('timing');
const originalImg = document.getElementById('original');
const canvas = document.getElementById('processed');

// 1) show a preview as soon as a file is chosen
fileInput.addEventListener('change', () => {
  const file = fileInput.files[0];
  message.textContent = '';
  resultCard.hidden = true;
  canvas.width = canvas.height = 0;
  if (!file) { detectBtn.disabled = true; return; }
  originalImg.src = URL.createObjectURL(file);
  detectBtn.disabled = false;
});

// 2) send the image to the backend
detectBtn.addEventListener('click', async () => {
  const file = fileInput.files[0];
  const form = new FormData();
  form.append('image', file);                      // the backend expects the field name "image"

  detectBtn.disabled = true;
  detectBtn.textContent = 'Detecting...';
  message.textContent = '';
  try {
    const response = await fetch('/api/detect', { method: 'POST', body: form });
    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');
    showResult(data);
  } catch (err) {
    message.textContent = 'Error: ' + err.message;
  } finally {
    detectBtn.disabled = false;
    detectBtn.textContent = 'Detect gas';
  }
});

// 3) show status, confidence list and the processed image
function showResult(data) {
  resultCard.hidden = false;
  statusEl.textContent = data.status;
  statusEl.className = 'badge ' + (data.count > 0 ? 'gas' : 'clear');
  listEl.innerHTML = '';
  data.detections.forEach((d, i) => {
    const li = document.createElement('li');
    li.textContent = `Region ${i + 1}: ${d.label} - confidence ${d.percent}%`;
    listEl.appendChild(li);
  });
  timingEl.textContent = `Image ${data.width}x${data.height}px, processed in ${data.time_ms} ms`;
  drawBoxes(data);
}

// the boxes come back in the pixels of the original image, so we draw them on a copy of it
function drawBoxes(data) {
  const ctx = canvas.getContext('2d');
  canvas.width = data.width;
  canvas.height = data.height;
  ctx.drawImage(originalImg, 0, 0, data.width, data.height);

  const line = Math.max(2, Math.round(Math.min(data.width, data.height) / 150));
  const font = Math.max(14, Math.round(Math.min(data.width, data.height) / 28));
  ctx.lineWidth = line;
  ctx.font = `bold ${font}px Arial`;
  ctx.textBaseline = 'top';

  data.detections.forEach((d) => {
    ctx.strokeStyle = '#ff2d20';
    ctx.strokeRect(d.x1, d.y1, d.x2 - d.x1, d.y2 - d.y1);
    const text = `${d.label} ${d.percent}%`;
    const w = ctx.measureText(text).width + 10;
    const labelY = d.y1 - font - 6 >= 0 ? d.y1 - font - 6 : d.y1;   // keep the label inside the picture
    ctx.fillStyle = '#ff2d20';
    ctx.fillRect(d.x1, labelY, w, font + 6);
    ctx.fillStyle = '#fff';
    ctx.fillText(text, d.x1 + 5, labelY + 3);
  });
}

// optional: show the real evaluation numbers if ml/evaluate.py has been run
fetch('/api/metrics')
  .then((r) => (r.ok ? r.json() : null))
  .then((m) => {
    if (!m || !m.splits.test) return;
    const t = m.splits.test;
    document.getElementById('metrics').innerHTML =
      `<table><tr><th>Precision</th><th>Recall</th><th>F1</th><th>mAP@0.5</th><th>mAP@0.5:0.95</th><th>Test images</th></tr>
       <tr><td>${t.precision}</td><td>${t.recall}</td><td>${t.f1}</td><td>${t.mAP50}</td><td>${t.mAP50_95}</td><td>${t.images}</td></tr></table>`;
    document.getElementById('metricsCard').hidden = false;
  })
  .catch(() => {});

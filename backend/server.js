// Simple Express server:  npm start  ->  http://localhost:3000
const fs = require('fs');
const path = require('path');
const express = require('express');
const multer = require('multer');
const { loadModel, detect } = require('./detector');

const PORT = process.env.PORT || 3000;
const CONFIDENCE = 0.25;                                   // boxes below 25% confidence are ignored

const app = express();
const upload = multer({
  storage: multer.memoryStorage(),                         // keep the uploaded file in memory (nothing is saved to disk)
  limits: { fileSize: 10 * 1024 * 1024 },                  // max 10 MB
});

// the web page (frontend folder)
app.use(express.static(path.join(__dirname, '..', 'frontend')));

// POST /api/detect  - form field "image" -> JSON with the detected gas boxes
app.post('/api/detect', upload.single('image'), async (req, res) => {
  if (!req.file) {
    return res.status(400).json({ error: 'Please upload an image (form field name: "image").' });
  }
  try {
    const result = await detect(req.file.buffer, CONFIDENCE);
    res.json(result);
  } catch (err) {
    console.error(err.message);
    res.status(400).json({ error: 'Could not read this file as an image.' });
  }
});

// GET /api/metrics - the evaluation numbers written by ml/evaluate.py (if it has been run)
app.get('/api/metrics', (req, res) => {
  const file = path.join(__dirname, '..', 'models', 'metrics.json');
  if (!fs.existsSync(file)) return res.status(404).json({ error: 'Run ml/evaluate.py first.' });
  res.type('json').send(fs.readFileSync(file));
});

// upload errors (file too big, ...)
app.use((err, req, res, next) => {
  res.status(err.code === 'LIMIT_FILE_SIZE' ? 413 : 500).json({ error: err.message });
});

loadModel()
  .then(({ size, names }) => {
    app.listen(PORT, () => console.log(`Model loaded (${size}px, classes: ${names}).\nOpen http://localhost:${PORT}`));
  })
  .catch((err) => {
    console.error('Could not load the model. Did you run the Python steps first?\n', err.message);
    process.exit(1);
  });

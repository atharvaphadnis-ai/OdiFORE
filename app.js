const CLASS_CONFIG = [
  { key: 'normal', label: 'Normal' },
  { key: 'running', label: 'Running' },
  { key: 'rapid_approach', label: 'Rapid Approach' },
  { key: 'loitering', label: 'Loitering' },
  { key: 'sudden_direction_change', label: 'Sudden Direction Change' }
];

const CLASS_KEYS = CLASS_CONFIG.map((item) => item.key);
const CLASS_LABELS = CLASS_CONFIG.map((item) => item.label);
const DEFAULT_SETTINGS = {
  framesPerVideo: 16,
  frameSamplingInterval: 1,
  imageSize: 224,
  epochs: 12,
  batchSize: 8,
  learningRate: 0.001,
  validationSplit: 0.2
};

const state = {
  datasetFilesByClass: {},
  model: null,
  featureExtractor: null,
  activeBackend: 'cpu',
  chartData: {
    trainLoss: [],
    valLoss: [],
    trainAcc: [],
    valAcc: []
  },
  cameraStream: null,
  cameraFrameTimer: null,
  cameraFps: 0,
  cameraLastTimestamp: 0,
  modelStatus: 'not trained',
  lastPrediction: null,
  currentConfig: { ...DEFAULT_SETTINGS }
};

const backendStatusEl = document.getElementById('backendStatus');
const datasetInput = document.getElementById('datasetInput');
const datasetGridEl = document.getElementById('datasetGrid');
const datasetValidationEl = document.getElementById('datasetValidation');
const trainingProgressBarEl = document.getElementById('trainingProgressBar');
const estimatedProgressEl = document.getElementById('estimatedProgress');
const currentEpochEl = document.getElementById('currentEpoch');
const trainLossEl = document.getElementById('trainLoss');
const valLossEl = document.getElementById('valLoss');
const trainAccuracyEl = document.getElementById('trainAccuracy');
const valAccuracyEl = document.getElementById('valAccuracy');
const modelStatusEl = document.getElementById('modelStatus');
const cameraPreviewEl = document.getElementById('cameraPreview');
const cameraPredictionEl = document.getElementById('cameraPrediction');
const cameraConfidenceEl = document.getElementById('cameraConfidence');
const cameraFpsEl = document.getElementById('cameraFps');
const cameraStatusEl = document.getElementById('cameraStatus');
const toastEl = document.getElementById('toast');
const chartCanvas = document.getElementById('trainingChart');
const chartCtx = chartCanvas.getContext('2d');

function showToast(message) {
  toastEl.textContent = message;
  toastEl.classList.add('show');
  clearTimeout(showToast.timer);
  showToast.timer = setTimeout(() => {
    toastEl.classList.remove('show');
  }, 2200);
}

function getSettings() {
  return {
    framesPerVideo: Number(document.getElementById('framesPerVideo').value || DEFAULT_SETTINGS.framesPerVideo),
    frameSamplingInterval: Number(document.getElementById('frameSamplingInterval').value || DEFAULT_SETTINGS.frameSamplingInterval),
    imageSize: Number(document.getElementById('imageSize').value || DEFAULT_SETTINGS.imageSize),
    epochs: Number(document.getElementById('epochs').value || DEFAULT_SETTINGS.epochs),
    batchSize: Number(document.getElementById('batchSize').value || DEFAULT_SETTINGS.batchSize),
    learningRate: Number(document.getElementById('learningRate').value || DEFAULT_SETTINGS.learningRate),
    validationSplit: Number(document.getElementById('validationSplit').value || DEFAULT_SETTINGS.validationSplit)
  };
}

function updateBackendStatus() {
  backendStatusEl.textContent = `Backend: ${state.activeBackend.toUpperCase()}`;
}

async function initializeBackend() {
  const preferred = ['webgpu', 'webgl', 'cpu'];
  let resolved = 'cpu';

  for (const backend of preferred) {
    try {
      await tf.setBackend(backend);
      resolved = backend;
      break;
    } catch (error) {
      console.warn(`Backend ${backend} unavailable:`, error.message || error);
    }
  }

  try {
    await tf.ready();
  } catch (error) {
    console.warn('tf.ready failed:', error);
  }

  state.activeBackend = tf.getBackend ? tf.getBackend() : 'cpu';
  updateBackendStatus();
}

async function ensureFeatureExtractor() {
  if (!state.featureExtractor) {
    try {
      state.featureExtractor = await mobilenet.load({ version: 1, alpha: 0.75 });
    } catch (error) {
      throw new Error('Unable to load MobileNet feature extractor. Check the TensorFlow.js CDN and browser compatibility.');
    }
  }
  return state.featureExtractor;
}

function getClassKeyFromPath(filePath) {
  const normalized = (filePath || '').toLowerCase();

  if (!normalized) return null;

  for (const item of CLASS_CONFIG) {
    const variants = [
      item.key,
      item.key.replace(/_/g, ' '),
      item.key.replace(/_/g, '-'),
      item.label.toLowerCase()
    ];

    if (variants.some((variant) => normalized.includes(variant))) {
      return item.key;
    }
  }

  return null;
}

function updateDatasetValidation() {
  const validationItems = [];
  let totalVideos = 0;

  for (const item of CLASS_CONFIG) {
    const count = state.datasetFilesByClass[item.key]?.length || 0;
    totalVideos += count;
    const valid = count > 0;
    validationItems.push(`
      <div class="validation-item ${valid ? 'valid' : 'invalid'}">
        ${valid ? '✓' : '✗'} ${item.label}: ${count} video${count === 1 ? '' : 's'}
      </div>
    `);
  }

  datasetValidationEl.innerHTML = validationItems.join('') + `
    <div class="validation-item">
      Total videos: ${totalVideos}
    </div>
  `;
}

function renderDatasetRows() {
  datasetGridEl.innerHTML = CLASS_CONFIG.map((item) => {
    const files = state.datasetFilesByClass[item.key] || [];
    const count = files.length;
    const statusText = count > 0 ? 'Ready' : 'Missing';
    const statusClass = count > 0 ? 'ready' : 'empty';
    const estimatedFrames = count * Math.max(1, getSettings().framesPerVideo);

    return `
      <div class="class-card">
        <h3>${item.label}</h3>
        <div class="class-meta">
          <span>Videos: ${count}</span>
          <span>Frames estimated: ${estimatedFrames}</span>
          <span>Dataset status: ${statusText}</span>
        </div>
        <span class="status-indicator ${statusClass}">${statusText}</span>
      </div>
    `;
  }).join('');
}

function handleDatasetSelection(event) {
  const files = Array.from(event.target.files || []);
  const grouped = {};

  for (const key of CLASS_KEYS) {
    grouped[key] = [];
  }

  for (const file of files) {
    const validExt = ['.mp4', '.webm', '.mov'];
    const lowerName = file.name.toLowerCase();
    const hasValidExt = validExt.some((ext) => lowerName.endsWith(ext));

    if (!hasValidExt) continue;

    const classKey = getClassKeyFromPath(file.webkitRelativePath || file.name || '');
    if (!classKey) continue;
    grouped[classKey].push(file);
  }

  state.datasetFilesByClass = grouped;
  renderDatasetRows();
  updateDatasetValidation();
  showToast('Dataset loaded. Review class counts before training.');
}

function validateDataset() {
  const missing = [];

  for (const item of CLASS_CONFIG) {
    const count = state.datasetFilesByClass[item.key]?.length || 0;
    if (count === 0) missing.push(item.label);
  }

  if (missing.length > 0) {
    return {
      ok: false,
      message: `Missing classes: ${missing.join(', ')}. All five classes are required before training.`
    };
  }

  return { ok: true, message: 'Dataset ready' };
}

function aggregateFeatureVectors(frameVectors) {
  if (!frameVectors.length) return new Float32Array();

  const featureDim = frameVectors[0].length;
  const mean = new Float32Array(featureDim);
  const max = new Float32Array(featureDim);
  const delta = new Float32Array(featureDim);

  for (let i = 0; i < featureDim; i += 1) {
    max[i] = -Infinity;
  }

  for (let i = 0; i < frameVectors.length; i += 1) {
    const vector = frameVectors[i];
    for (let j = 0; j < featureDim; j += 1) {
      mean[j] += vector[j];
      if (vector[j] > max[j]) max[j] = vector[j];
      if (i > 0) {
        delta[j] += Math.abs(vector[j] - frameVectors[i - 1][j]);
      }
    }
  }

  for (let i = 0; i < featureDim; i += 1) {
    mean[i] /= frameVectors.length;
    if (frameVectors.length > 1) {
      delta[i] /= (frameVectors.length - 1);
    }
  }

  return Float32Array.from([...mean, ...max, ...delta]);
}

function waitForHandle(resolve, reject, video, targetTime) {
  const onSeeked = () => {
    video.removeEventListener('seeked', onSeeked);
    video.removeEventListener('error', onError);
    resolve();
  };

  const onError = () => {
    video.removeEventListener('seeked', onSeeked);
    video.removeEventListener('error', onError);
    reject(new Error('Video seek failed.'));
  };

  video.addEventListener('seeked', onSeeked, { once: true });
  video.addEventListener('error', onError, { once: true });
  video.currentTime = Math.max(0.01, targetTime);
}

async function extractVideoRepresentation(file, settings) {
  const video = document.createElement('video');
  video.muted = true;
  video.playsInline = true;
  video.preload = 'auto';
  video.crossOrigin = 'anonymous';

  const objectUrl = URL.createObjectURL(file);
  video.src = objectUrl;

  await new Promise((resolve, reject) => {
    video.onloadedmetadata = () => resolve();
    video.onerror = () => reject(new Error(`Unsupported or corrupted video: ${file.name}`));
  });

  const duration = Number.isFinite(video.duration) ? video.duration : 0;
  if (duration <= 0) {
    URL.revokeObjectURL(objectUrl);
    throw new Error(`Video has no usable duration: ${file.name}`);
  }

  const frameVectors = [];
  const sampleCount = Math.max(1, settings.framesPerVideo);

  for (let i = 0; i < sampleCount; i += 1) {
    const targetTime = Math.min(duration - 0.01, (i + 0.5) * (duration / sampleCount));

    await new Promise((resolve, reject) => {
      const alreadyReady = Math.abs(video.currentTime - targetTime) < 0.01;
      if (alreadyReady) {
        resolve();
        return;
      }
      waitForHandle(resolve, reject, video, targetTime);
    });

    const canvas = document.createElement('canvas');
    canvas.width = settings.imageSize;
    canvas.height = settings.imageSize;
    const ctx = canvas.getContext('2d');

    try {
      ctx.drawImage(video, 0, 0, canvas.width, canvas.height);
    } catch (drawError) {
      URL.revokeObjectURL(objectUrl);
      throw new Error(`Failed to decode frames in ${file.name}: ${drawError.message}`);
    }

    const frameTensor = tf.browser.fromPixels(canvas).toFloat().div(255);
    const featureTensor = await state.featureExtractor.infer(frameTensor.expandDims());
    const values = Array.from(featureTensor.dataSync());

    frameVectors.push(new Float32Array(values));

    tf.dispose([frameTensor, featureTensor]);
    if (i % 2 === 0) {
      await tf.nextFrame();
    }
  }

  URL.revokeObjectURL(objectUrl);
  video.pause();
  video.remove();

  const aggregated = aggregateFeatureVectors(frameVectors);
  return Array.from(aggregated);
}

function buildModel(featureSize) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [featureSize], units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: CLASS_KEYS.length, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(0.001),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

function buildTrainingModel(featureSize, learningRate) {
  const model = tf.sequential();
  model.add(tf.layers.dense({ inputShape: [featureSize], units: 128, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.3 }));
  model.add(tf.layers.dense({ units: 64, activation: 'relu' }));
  model.add(tf.layers.dropout({ rate: 0.2 }));
  model.add(tf.layers.dense({ units: CLASS_KEYS.length, activation: 'softmax' }));

  model.compile({
    optimizer: tf.train.adam(learningRate),
    loss: 'categoricalCrossentropy',
    metrics: ['accuracy']
  });

  return model;
}

function splitSamples(samples, validationSplit) {
  if (samples.length < 2) {
    throw new Error('Dataset is too small for validation splitting. Add more video examples per class.');
  }

  const shuffled = [...samples].sort(() => Math.random() - 0.5);
  const validationCount = Math.max(1, Math.floor(shuffled.length * validationSplit));
  const splitIndex = Math.max(1, shuffled.length - validationCount);

  const train = shuffled.slice(0, splitIndex);
  const val = shuffled.slice(splitIndex);

  return { train, val };
}

async function prepareDatasetSamples(settings) {
  const samples = [];
  const classNames = CLASS_KEYS;

  for (const key of classNames) {
    const files = state.datasetFilesByClass[key] || [];
    for (const file of files) {
      const featureVector = await extractVideoRepresentation(file, settings);
      samples.push({ input: featureVector, label: CLASS_KEYS.indexOf(key) });
    }
  }

  if (!samples.length) {
    throw new Error('No valid videos were found in the selected dataset.');
  }

  return samples;
}

function createTensorData(samples) {
  const x = tf.tensor2d(samples.map((sample) => sample.input));
  const y = tf.oneHot(samples.map((sample) => sample.label), CLASS_KEYS.length);

  return { x, y };
}

function updateProgressDisplay(epoch, totalEpochs, startTimestamp) {
  const progress = totalEpochs ? (epoch / totalEpochs) * 100 : 0;
  trainingProgressBarEl.style.width = `${progress}%`;

  const elapsedMs = performance.now() - startTimestamp;
  const remainingEpochs = Math.max(0, totalEpochs - epoch);
  const etaMs = remainingEpochs > 0 ? (elapsedMs / Math.max(1, epoch)) * remainingEpochs : 0;
  const etaSeconds = Math.max(0, Math.round(etaMs / 1000));

  estimatedProgressEl.textContent = `${Math.round(progress)}% complete • ETA ${etaSeconds}s`;
}

function drawTrainingChart() {
  const width = chartCanvas.width;
  const height = chartCanvas.height;
  const padding = 26;

  chartCtx.clearRect(0, 0, width, height);

  chartCtx.strokeStyle = 'rgba(255,255,255,0.08)';
  chartCtx.lineWidth = 1;
  for (let i = 0; i <= 4; i += 1) {
    const y = padding + ((height - padding * 2) / 4) * i;
    chartCtx.beginPath();
    chartCtx.moveTo(padding, y);
    chartCtx.lineTo(width - padding, y);
    chartCtx.stroke();
  }

  const maxLoss = 5;
  const maxAcc = 1;
  const series = {
    trainLoss: state.chartData.trainLoss,
    valLoss: state.chartData.valLoss,
    trainAcc: state.chartData.trainAcc,
    valAcc: state.chartData.valAcc
  };

  const drawSeries = (key, color) => {
    if (!series[key].length) return;
    chartCtx.beginPath();
    chartCtx.lineWidth = 2;
    chartCtx.strokeStyle = color;
    series[key].forEach((value, index) => {
      const x = padding + (index / Math.max(1, series[key].length - 1)) * (width - padding * 2);
      const yMap = (v, maxValue) => height - padding - ((v / maxValue) * (height - padding * 2));
      const y = yMap(key.includes('Loss') ? value : value, key.includes('Loss') ? maxLoss : maxAcc);
      if (index === 0) chartCtx.moveTo(x, y);
      else chartCtx.lineTo(x, y);
    });
    chartCtx.stroke();
  };

  drawSeries('trainLoss', '#52c1ff');
  drawSeries('valLoss', '#ffbf69');
  drawSeries('trainAcc', '#3ddc97');
  drawSeries('valAcc', '#ff6b7d');

  chartCtx.fillStyle = '#a8bfd6';
  chartCtx.font = '12px sans-serif';
  chartCtx.fillText('Loss', 12, 18);
  chartCtx.fillText('Accuracy', 12, 36);
}

function setModelStatus(message) {
  modelStatusEl.textContent = `Model status: ${message}`;
}

async function trainModel() {
  const validation = validateDataset();
  if (!validation.ok) {
    showToast(validation.message);
    return;
  }

  try {
    const settings = getSettings();
    state.currentConfig = settings;
    document.getElementById('framesPerVideo').value = settings.framesPerVideo;
    document.getElementById('imageSize').value = settings.imageSize;

    if (!state.featureExtractor) {
      await ensureFeatureExtractor();
    }

    showToast('Preprocessing video dataset...');
    const samples = await prepareDatasetSamples(settings);

    const { x, y } = createTensorData(samples);
    const split = splitSamples(samples, settings.validationSplit);
    const trainX = tf.tensor2d(split.train.map((item) => item.input));
    const trainY = tf.oneHot(split.train.map((item) => item.label), CLASS_KEYS.length);
    const valX = tf.tensor2d(split.val.map((item) => item.input));
    const valY = tf.oneHot(split.val.map((item) => item.label), CLASS_KEYS.length);

    const model = buildTrainingModel(x.shape[1], settings.learningRate);
    state.model = model;
    state.chartData = { trainLoss: [], valLoss: [], trainAcc: [], valAcc: [] };

    const startTimestamp = performance.now();
    const totalEpochs = settings.epochs;
    currentEpochEl.textContent = `0 / ${totalEpochs}`;

    await model.fit(trainX, trainY, {
      epochs: totalEpochs,
      batchSize: settings.batchSize,
      validationData: [valX, valY],
      verbose: 0,
      callbacks: {
        onEpochEnd: (epoch, logs) => {
          const currentEpoch = epoch + 1;
          currentEpochEl.textContent = `${currentEpoch} / ${totalEpochs}`;

          const trainLoss = Number(logs.loss || 0).toFixed(4);
          const valLoss = Number(logs.val_loss || 0).toFixed(4);
          const trainAcc = Number(logs.acc || 0).toFixed(4);
          const valAcc = Number(logs.val_acc || 0).toFixed(4);

          trainLossEl.textContent = trainLoss;
          valLossEl.textContent = valLoss;
          trainAccuracyEl.textContent = `${trainAcc}`;
          valAccuracyEl.textContent = `${valAcc}`;

          state.chartData.trainLoss.push(Number(trainLoss));
          state.chartData.valLoss.push(Number(valLoss));
          state.chartData.trainAcc.push(Number(trainAcc));
          state.chartData.valAcc.push(Number(valAcc));
          drawTrainingChart();
          updateProgressDisplay(currentEpoch, totalEpochs, startTimestamp);
        }
      }
    });

    setModelStatus('trained and ready');
    showToast('Model training complete.');
    drawTrainingChart();

    tf.dispose([x, y, trainX, trainY, valX, valY]);
  } catch (error) {
    console.error(error);
    showToast(`Training failed: ${error.message}`);
    setModelStatus('training failed');
  }
}

async function exportModel() {
  if (!state.model) {
    showToast('Train a model before exporting.');
    return;
  }

  try {
    await state.model.save('downloads://odifore-model');

    const config = {
      classNames: CLASS_KEYS,
      labels: CLASS_LABELS,
      settings: state.currentConfig,
      featureExtractor: 'MobileNetV1',
      architecture: [
        'Input: aggregated visual features',
        'Dense(128, ReLU)',
        'Dropout(0.3)',
        'Dense(64, ReLU)',
        'Dropout(0.2)',
        'Dense(5, Softmax)'
      ],
      createdAt: new Date().toISOString()
    };

    const blob = new Blob([JSON.stringify(config, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = 'odifore-config.json';
    a.click();
    URL.revokeObjectURL(url);
    showToast('Model and configuration exported.');
  } catch (error) {
    console.error(error);
    showToast(`Export failed: ${error.message}`);
  }
}

async function saveModelToIndexedDB() {
  if (!state.model) {
    showToast('No trained model to save.');
    return;
  }

  try {
    await state.model.save('indexeddb://odifore-model');
    showToast('Model saved to browser storage.');
  } catch (error) {
    console.error(error);
    showToast(`Unable to save model: ${error.message}`);
  }
}

async function loadSavedModelFromIndexedDB() {
  try {
    state.model = await tf.loadLayersModel('indexeddb://odifore-model');
    setModelStatus('loaded from saved browser model');
    showToast('Saved model loaded.');
  } catch (error) {
    console.error(error);
    showToast('No saved browser model was found.');
  }
}

async function deleteSavedModelFromIndexedDB() {
  try {
    await tf.io.removeModel('indexeddb://odifore-model');
    state.model = null;
    setModelStatus('deleted');
    showToast('Saved model deleted.');
  } catch (error) {
    console.error(error);
    showToast('Could not remove saved model.');
  }
}

async function handleModelImport(event) {
  const [file] = Array.from(event.target.files || []);
  if (!file) return;

  try {
    const isConfig = file.name.toLowerCase().includes('config');
    if (isConfig) {
      const config = JSON.parse(await file.text());
      state.currentConfig = config.settings || state.currentConfig;
      document.getElementById('framesPerVideo').value = state.currentConfig.framesPerVideo || DEFAULT_SETTINGS.framesPerVideo;
      document.getElementById('imageSize').value = state.currentConfig.imageSize || DEFAULT_SETTINGS.imageSize;
      document.getElementById('epochs').value = state.currentConfig.epochs || DEFAULT_SETTINGS.epochs;
      document.getElementById('batchSize').value = state.currentConfig.batchSize || DEFAULT_SETTINGS.batchSize;
      document.getElementById('learningRate').value = state.currentConfig.learningRate || DEFAULT_SETTINGS.learningRate;
      document.getElementById('validationSplit').value = state.currentConfig.validationSplit || DEFAULT_SETTINGS.validationSplit;
      showToast('Configuration restored.');
      return;
    }

    if (file.name.toLowerCase().endsWith('.json')) {
      const objectUrl = URL.createObjectURL(file);
      state.model = await tf.loadLayersModel(objectUrl);
      setModelStatus('loaded from JSON export');
      showToast('Model JSON loaded.');
      URL.revokeObjectURL(objectUrl);
      return;
    }

    showToast('Unsupported model file type.');
  } catch (error) {
    console.error(error);
    showToast('Model loading failed. Use LOAD SAVED MODEL or export a valid config file.');
  }
}

async function predictSingleVideoFile(file) {
  if (!state.model) {
    showToast('Train or load a model before testing.');
    return null;
  }

  await ensureFeatureExtractor();
  const sampleSettings = getSettings();
  const vector = await extractVideoRepresentation(file, sampleSettings);
  const tensor = tf.tensor2d([vector]);
  const probs = state.model.predict(tensor);
  const values = Array.from(await probs.data());
  const predictionIndex = values.indexOf(Math.max(...values));
  const probability = values[predictionIndex] * 100;

  const results = CLASS_CONFIG.map((item, index) => ({
    label: item.label,
    probability: values[index] * 100,
    index
  }));

  tf.dispose([tensor, probs]);
  return { predictionIndex, probability, results };
}

function renderPredictionList(results, highlightIndex) {
  const maxProb = Math.max(...results.map((item) => item.probability));

  const html = results.map((item) => {
    const width = (item.probability / maxProb) * 100;
    const highlight = item.index === highlightIndex ? 'highlight' : '';
    return `
      <div class="prediction-item ${highlight}">
        <strong>${item.label}</strong>
        <div class="prediction-bar"><div class="prediction-bar-fill" style="width:${width}%"></div></div>
        <span>${item.probability.toFixed(1)}%</span>
      </div>
    `;
  }).join('');

  document.getElementById('testPredictionResult').innerHTML = html;
}

async function predictSelectedVideo() {
  const input = document.getElementById('testVideoInput');
  const [file] = Array.from(input.files || []);
  if (!file) {
    showToast('Choose a .mp4, .webm, or .mov video file to test.');
    return;
  }

  try {
    const prediction = await predictSingleVideoFile(file);
    if (!prediction) return;

    renderPredictionList(prediction.results, prediction.predictionIndex);
    showToast(`Prediction: ${CLASS_CONFIG[prediction.predictionIndex].label}`);
  } catch (error) {
    console.error(error);
    showToast(`Prediction failed: ${error.message}`);
  }
}

function updateCameraStatus(text) {
  cameraStatusEl.textContent = text;
}

async function startCamera() {
  if (state.cameraStream) {
    stopCamera();
    return;
  }

  try {
    const stream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: 'user' },
      audio: false
    });

    state.cameraStream = stream;
    cameraPreviewEl.srcObject = stream;
    cameraPreviewEl.play();
    state.cameraLastTimestamp = performance.now();
    updateCameraStatus('Running');
    cameraPredictionEl.textContent = 'Analyzing';
    cameraConfidenceEl.textContent = '0%';
    state.cameraFrameTimer = setInterval(async () => {
      try {
        await processCameraFrame();
      } catch (error) {
        console.error(error);
        updateCameraStatus('Error');
      }
    }, 350);
  } catch (error) {
    console.error(error);
    showToast('Camera access failed: ' + error.message);
    updateCameraStatus('Blocked');
  }
}

function stopCamera() {
  if (state.cameraStream) {
    state.cameraStream.getTracks().forEach((track) => track.stop());
    state.cameraStream = null;
  }

  if (state.cameraFrameTimer) {
    clearInterval(state.cameraFrameTimer);
    state.cameraFrameTimer = null;
  }

  cameraPreviewEl.srcObject = null;
  cameraPredictionEl.textContent = 'Idle';
  cameraConfidenceEl.textContent = '0%';
  cameraFpsEl.textContent = '0';
  updateCameraStatus('Stopped');
}

async function processCameraFrame() {
  if (!state.model || !state.cameraStream) return;

  const now = performance.now();
  const delta = now - state.cameraLastTimestamp || 16;
  state.cameraFps = Math.round(1000 / delta);
  cameraFpsEl.textContent = String(state.cameraFps);
  state.cameraLastTimestamp = now;

  const video = cameraPreviewEl;
  if (video.readyState < 2) return;

  updateCameraStatus('Processing');

  const canvas = document.createElement('canvas');
  canvas.width = getSettings().imageSize;
  canvas.height = getSettings().imageSize;
  const ctx = canvas.getContext('2d');
  ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

  const tensor = tf.browser.fromPixels(canvas).toFloat().div(255);
  const featureTensor = await ensureFeatureExtractor().then((extractor) => extractor.infer(tensor.expandDims()));
  const vector = Array.from(featureTensor.dataSync());
  const input = tf.tensor2d([vector]);
  const result = state.model.predict(input);
  const values = Array.from(await result.data());
  const bestIndex = values.indexOf(Math.max(...values));
  const confidence = (values[bestIndex] * 100).toFixed(1);

  cameraPredictionEl.textContent = CLASS_CONFIG[bestIndex].label;
  cameraConfidenceEl.textContent = `${confidence}%`;
  updateCameraStatus('Ready');

  tf.dispose([tensor, featureTensor, input, result]);
}

function initializeUI() {
  datasetInput.addEventListener('change', handleDatasetSelection);
  document.getElementById('trainButton').addEventListener('click', trainModel);
  document.getElementById('exportModelButton').addEventListener('click', exportModel);
  document.getElementById('saveModelButton').addEventListener('click', saveModelToIndexedDB);
  document.getElementById('loadSavedModelButton').addEventListener('click', loadSavedModelFromIndexedDB);
  document.getElementById('deleteSavedModelButton').addEventListener('click', deleteSavedModelFromIndexedDB);
  document.getElementById('loadModelButton').addEventListener('click', () => document.getElementById('modelImportInput').click());
  document.getElementById('modelImportInput').addEventListener('change', handleModelImport);
  document.getElementById('predictVideoButton').addEventListener('click', predictSelectedVideo);
  document.getElementById('startCameraButton').addEventListener('click', () => {
    if (!state.model) {
      showToast('Train or load a model before starting the camera.');
      return;
    }
    if (state.cameraStream) {
      stopCamera();
      return;
    }
    startCamera();
  });

  document.getElementById('testVideoInput').addEventListener('change', () => {
    const [file] = Array.from(document.getElementById('testVideoInput').files || []);
    if (file) showToast(`Video ready: ${file.name}`);
  });

  document.getElementById('framesPerVideo').addEventListener('input', () => renderDatasetRows());
  document.getElementById('imageSize').addEventListener('input', () => renderDatasetRows());

  setModelStatus('not trained');
  renderDatasetRows();
  updateDatasetValidation();
  drawTrainingChart();
}

async function init() {
  await initializeBackend();
  await ensureFeatureExtractor();
  initializeUI();
  updateBackendStatus();
}

init();

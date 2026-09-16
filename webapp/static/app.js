const $ = (id) => document.getElementById(id);

const fileInput = $('fileInput');
const dropzone = $('dropzone');
const preview = $('preview');
const runBtn = $('runBtn');
const statusEl = $('status');
const video = $('video');

let selectedFile = null;   // File chosen via upload tab
let capturedData = null;   // data URI grabbed from the webcam
let stream = null;
let activeTab = 'upload';

/* ---------- tabs ---------- */
document.querySelectorAll('.tab').forEach((tab) => {
  tab.addEventListener('click', () => {
    activeTab = tab.dataset.tab;
    document.querySelectorAll('.tab').forEach((t) => t.classList.toggle('active', t === tab));
    $('tab-upload').classList.toggle('hidden', activeTab !== 'upload');
    $('tab-camera').classList.toggle('hidden', activeTab !== 'camera');
    refreshRunButton();
  });
});

function refreshRunButton() {
  runBtn.disabled = activeTab === 'upload' ? !selectedFile : !capturedData;
}

/* ---------- upload ---------- */
dropzone.addEventListener('click', () => fileInput.click());

['dragenter', 'dragover'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.add('hover');
  })
);

['dragleave', 'drop'].forEach((evt) =>
  dropzone.addEventListener(evt, (e) => {
    e.preventDefault();
    dropzone.classList.remove('hover');
  })
);

dropzone.addEventListener('drop', (e) => {
  if (e.dataTransfer.files.length) setFile(e.dataTransfer.files[0]);
});

fileInput.addEventListener('change', () => {
  if (fileInput.files.length) setFile(fileInput.files[0]);
});

function setFile(file) {
  if (!file.type.startsWith('image/')) {
    statusEl.textContent = 'Please choose an image file.';
    return;
  }
  selectedFile = file;
  preview.src = URL.createObjectURL(file);
  preview.classList.remove('hidden');
  statusEl.textContent = file.name;
  refreshRunButton();
}

/* ---------- camera ---------- */
$('startCam').addEventListener('click', async () => {
  try {
    stream = await navigator.mediaDevices.getUserMedia({ video: { width: 1280, height: 720 } });
    video.srcObject = stream;
    $('snap').disabled = false;
    $('camHint').textContent = 'Camera live. Frame the number plate, then capture.';
  } catch (err) {
    $('camHint').textContent = 'Could not open camera: ' + err.message;
  }
});

$('snap').addEventListener('click', () => {
  const canvas = document.createElement('canvas');
  canvas.width = video.videoWidth;
  canvas.height = video.videoHeight;
  canvas.getContext('2d').drawImage(video, 0, 0);
  capturedData = canvas.toDataURL('image/jpeg', 0.92);
  $('camHint').textContent = 'Frame captured. Ready to check.';
  refreshRunButton();
});

/* ---------- run ---------- */
runBtn.addEventListener('click', async () => {
  runBtn.disabled = true;
  statusEl.textContent = 'Detecting plate and reading text…';

  try {
    let response;
    if (activeTab === 'upload') {
      const body = new FormData();
      body.append('image', selectedFile);
      response = await fetch('/api/detect', { method: 'POST', body });
    } else {
      response = await fetch('/api/detect', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ image_data: capturedData, source: 'camera' }),
      });
    }

    const data = await response.json();
    if (!response.ok) throw new Error(data.error || 'Request failed');

    render(data);
    statusEl.textContent = `Done in ${data.timestamp} · ${data.detections.length} plate(s) found.`;
    loadLog();
  } catch (err) {
    statusEl.textContent = 'Error: ' + err.message;
  } finally {
    refreshRunButton();
  }
});

/* ---------- rendering ---------- */
const VERDICT = {
  authorized: 'ACCESS GRANTED',
  review: 'MANUAL REVIEW',
  denied: 'ACCESS DENIED',
};

function render(data) {
  $('empty').classList.add('hidden');
  $('resultBody').classList.remove('hidden');
  $('annotated').src = data.annotated_image;

  const cards = $('cards');
  cards.innerHTML = '';

  if (!data.detections.length) {
    cards.innerHTML =
      '<div class="card denied"><div class="verdict">NO PLATE DETECTED</div>' +
      '<p class="hint">YOLOv11 found no number plate in this image. Try a closer or sharper shot.</p></div>';
    return;
  }

  data.detections.forEach((d) => {
    const card = document.createElement('div');
    card.className = 'card ' + d.status;

    let note = '';
    if (d.status === 'authorized' && d.matched_plate) {
      note = `Matched guest record <b>${d.matched_plate}</b>.`;
    } else if (d.status === 'review') {
      note = `Closest record <b>${d.closest_plate}</b> (${Math.round(d.match_score * 100)}% similar). Verify manually.`;
    } else {
      note = d.closest_plate
        ? `Not in database. Closest record: <b>${d.closest_plate}</b>.`
        : 'Not in database.';
    }

    card.innerHTML = `
      <div class="verdict">${VERDICT[d.status]}</div>
      <div class="plate">${d.plate_text || '—'}</div>
      <p class="hint" style="margin-top:0">${note}</p>
      <div class="meta">
        <span>detection <b>${Math.round(d.detection_confidence * 100)}%</b></span>
        <span>OCR <b>${Math.round(d.ocr_confidence * 100)}%</b></span>
        <span>raw <b>${d.raw_text || '—'}</b></span>
      </div>
      ${d.crop ? `<img class="crop" src="${d.crop}" alt="Plate crop">` : ''}
    `;
    cards.appendChild(card);
  });
}

/* ---------- side data ---------- */
async function loadDatabase() {
  try {
    const data = await (await fetch('/api/database')).json();
    $('dbChip').textContent = `database: ${data.count} authorized plates`;
  } catch {
    $('dbChip').textContent = 'database: unavailable';
  }
}

async function loadLog() {
  const data = await (await fetch('/api/log')).json();
  const body = document.querySelector('#logTable tbody');
  body.innerHTML = '';
  $('logHint').classList.toggle('hidden', data.entries.length > 0);

  data.entries.forEach((row) => {
    const tr = document.createElement('tr');
    tr.innerHTML = `
      <td>${row.timestamp}</td>
      <td>${row.plate_text || '—'}</td>
      <td class="${row.status}">${VERDICT[row.status] || row.status}</td>
      <td>${row.matched_plate || '—'}</td>
    `;
    body.appendChild(tr);
  });
}

loadDatabase();
loadLog();

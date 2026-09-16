# VisionPass — ANPR-based Vehicle Entry Authorization

**Automatic Number Plate Recognition for controlled-access premises.**
A camera image goes in; a gate decision comes out.

---

## The problem

Entry to high-security premises — government offices, defence installations,
research campuses, gated corporate parks — is still largely manual. A guard
reads the number plate, compares it against a printed or digital register, and
raises the barrier by hand. That process is slow at peak hours, inconsistent
between shifts, and leaves no reliable audit trail of who entered when.

VisionPass automates the recognition and the lookup, and keeps the human in the
loop exactly where a human is still needed.

## What it does

Given a photograph of an approaching vehicle, the system:

1. **Locates** the number plate using a YOLOv11 detector trained on Indian plates
2. **Reads** the plate characters with EasyOCR
3. **Normalises** the text and matches it against a register of authorized vehicles
4. **Decides** — and records the decision in a timestamped audit log

| Decision | Meaning |
| --- | --- |
| 🟢 **ACCESS GRANTED** | Plate matches a registered vehicle — raise the barrier |
| 🟡 **MANUAL REVIEW** | Close but not certain — escalate to the guard, show the closest record |
| 🔴 **ACCESS DENIED** | Not in the register |

The middle state is the point of the design. A security checkpoint should never
open on a guess, but it also should not turn away a registered guest because the
camera read a `0` as a `G`. Uncertain reads go to a person, with the evidence
shown.

Both **still images** and **video** are supported — the notebook runs the same
pipeline frame-by-frame over dashcam footage and writes an annotated video plus
a spreadsheet of every plate it read.

---

## Screenshots

> _Add screenshots of the running web app here._

### Upload and detection
<!-- ![Upload view](docs/screenshot-upload.png) -->

### Access granted
<!-- ![Access granted](docs/screenshot-granted.png) -->

### Access denied
<!-- ![Access denied](docs/screenshot-denied.png) -->

### Entry log
<!-- ![Entry log](docs/screenshot-log.png) -->

_To add these: run the app, take screenshots, save them into a `docs/` folder
with the filenames above, and uncomment the image lines._

---

## Dataset

Training used the **[Indian Vehicle Dataset](https://www.kaggle.com/datasets/saisirishan/indian-vehicle-dataset)**
by *saisirishan*, downloaded from Kaggle via the Kaggle API:

```bash
kaggle datasets download -d saisirishan/indian-vehicle-dataset
```

The dataset ships images in two collections, both annotated in **Pascal VOC XML**:

| Collection | Contents |
| --- | --- |
| `google_images/` | General Indian vehicle photographs |
| `State-wise_OLX/` | Vehicle listings grouped by state registration code |

Indian plates are a good stress test for ANPR: formats vary by state, spacing is
inconsistent, and plates carry an embossed **IND** hologram strip that OCR
happily reads as part of the plate text.

---

## Workflow

### 1. Data preparation

Pascal VOC XML annotations are parsed and converted to YOLO format — bounding
boxes normalised to `class x_center y_center width height`, all under a single
class:

```yaml
train: /content/data_images/train
val:   /content/data_images/test
nc: 1
names: ['license_plate']
```

### 2. Detector training

YOLOv11-nano, fine-tuned from `yolo11n.pt`:

```bash
yolo task=detect mode=train epochs=50 batch=16 plots=True \
     model=yolo11n.pt data=data.yaml
```

**Results after 50 epochs** (`YOLOV11_EPOCH_50/`):

| Metric | Value |
| --- | --- |
| Precision | 0.988 |
| Recall | 0.994 |
| mAP@50 | 0.994 |
| mAP@50-95 | 0.857 |

Training curves, the confusion matrix, and validation predictions are in
`YOLOV11_EPOCH_50/RESULTS_METRICS_TRAIN/`.

### 3. Plate detection (inference)

The trained model runs on the input image at a confidence threshold of 0.25 and
returns bounding boxes for every plate it finds.

### 4. Text extraction

Each detected box is cropped, upscaled if smaller than 100 px tall (EasyOCR
degrades badly on small crops), converted to greyscale, and passed to EasyOCR.
The recognised fragments are joined and stripped to `A–Z0–9`, with the leading
`IND` hologram text removed.

### 5. Database matching

The cleaned plate is compared against `LICENSEPLATE_GUESTS_DATABASE.csv` in
three passes:

| Pass | Test | Result |
| --- | --- | --- |
| 1 | Exact match after normalisation | **granted** |
| 2 | Match allowing confusable characters | **granted** |
| 3 | Similarity ≥ 0.85 to a registered plate | **review** |
| — | Anything lower | **denied** |

Pass 2 exists because OCR rarely reproduces a plate character-for-character.
Characters are treated as interchangeable within these groups:

```
0 O Q D G    1 I L    2 Z    4 A    5 S    6 G    7 T    8 B
```

These are *groups*, not a substitution map, because a glyph can be ambiguous in
more than one direction — `G` is misread for both `0` and `6`, so a one-way
`G→6` rule would miss the `G→0` case. Two plates match if they are the same
length and every character pair shares a group.

> **Worked example.** The plate `KA 02MP 9657` is read by OCR as `KA G2MP 9657`.
> An exact comparison rejects a legitimately registered vehicle. VisionPass sees
> that `G` and `0` share a group, matches the record, and grants access.

### 6. Decision and audit

The verdict is drawn onto the image (colour-coded box and plate text), returned
to the browser alongside the cropped plate and confidence scores, and appended
to `webapp/entry_log.csv` with a timestamp.

---

## Tech stack

| Component | Technology |
| --- | --- |
| Plate detection | YOLOv11-nano (Ultralytics) |
| Text recognition | EasyOCR |
| Image processing | OpenCV |
| Web backend | Flask |
| Frontend | Vanilla HTML / CSS / JavaScript |
| Data handling | pandas |

---

## Repository layout

| Path | What it is |
| --- | --- |
| `webapp/` | Flask web application |
| `webapp/anpr.py` | Detection, OCR and matching logic |
| `webapp/app.py` | HTTP routes and audit logging |
| `ANPR_Indian_Dataset_with_Yolov11_and_EasyOCR_final.ipynb` | Data prep and training |
| `DEMO/ANPR_YOLOv11_DEMO.ipynb` | Image and video inference demo |
| `YOLOV11_EPOCH_50/` | Final training run — weights, metrics, curves |
| `DEMO/best.pt` | Deployed weights (identical to the 50-epoch checkpoint) |
| `LICENSEPLATE_GUESTS_DATABASE.csv` | Registered vehicles (29 plates) |
| `Metrics.docx`, `results.csv` | Training metrics |

---

## Quick start

```bash
python -m venv .venv
.venv\Scripts\python.exe -m pip install -r webapp\requirements.txt
cd webapp
..\.venv\Scripts\python.exe app.py
```

Open <http://127.0.0.1:5000>, drop in a vehicle photo, and press
**Check authorization**.

First launch takes about a minute — YOLOv11 and EasyOCR load into memory, and
EasyOCR downloads its recognition models (~80 MB) once. Runs on CPU; no GPU
required.

See [`webapp/README.md`](webapp/README.md) for configuration options.

### Registering a vehicle

Append a row to `LICENSEPLATE_GUESTS_DATABASE.csv` under the `License Plate`
column. The file is re-read on every request, so no restart is needed.

---

## Limitations

- **OCR accuracy drops** on motion-blurred, heavily angled, or poorly lit plates.
  The confidence scores are surfaced in the UI so a guard can judge for themselves.
- **CPU inference** takes a few seconds per image. A GPU build of PyTorch would
  bring this well under a second for real-time gate use.
- **The register is a flat CSV**, suitable for a demonstration. A production
  deployment would want a proper database with per-vehicle validity windows,
  visitor passes, and revocation.
- **Single class detector** — it finds plates, not vehicle type, colour or make.

---

## Not in this repository

Large demo videos, the virtual environment, and duplicate checkpoints are
excluded to keep clones small (see `.gitignore`). `DEMO/best.pt` — the weights
the application needs — **is** included, so the app runs straight after a clone.

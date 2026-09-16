# VisionPass — web app

Web front end for the ANPR pipeline built in `DEMO/ANPR_YOLOv11_DEMO.ipynb`.
Upload a vehicle photo (or grab a frame from a webcam), and the app will:

1. **Detect** the number plate with the trained YOLOv11 model (`DEMO/best.pt`).
2. **Read** the plate with EasyOCR and normalise the text to `A–Z0–9`.
3. **Match** it against `LICENSEPLATE_GUESTS_DATABASE.csv`.
4. **Decide** — `ACCESS GRANTED`, `MANUAL REVIEW`, or `ACCESS DENIED` — and log it.

## Running

```
webapp\run.bat
```

or manually:

```
..\.venv\Scripts\python.exe app.py
```

Then open <http://127.0.0.1:5000>.

The first start takes a minute: YOLOv11 and EasyOCR load into memory, and EasyOCR
downloads its detection/recognition models (~80 MB) to `~/.EasyOCR` once.

## Matching logic

Raw OCR rarely reproduces a plate character-for-character, so matching runs in
three passes (`anpr.py:match_plate`):

| Pass | Result |
| --- | --- |
| Exact match after normalisation | `authorized` |
| Match after collapsing confusable characters (`O/0`, `I/1`, `S/5`, `B/8`, `Z/2`, `G/6`, `D/0`, `L/1`, `Q/0`) | `authorized` |
| Similarity ≥ 0.85 to a database plate | `review` — closest record shown, guard verifies by hand |
| Anything lower | `denied` |

`review` exists on purpose: a security gate should not auto-open on a fuzzy read,
but it also should not silently reject a guest over one misread character.

## Configuration

Set before launching to override the defaults:

| Variable | Default |
| --- | --- |
| `VISIONPASS_WEIGHTS` | `..\DEMO\best.pt` |
| `VISIONPASS_DB` | `..\LICENSEPLATE_GUESTS_DATABASE.csv` (`.xlsx` also accepted) |
| `VISIONPASS_CONF` | `0.25` — YOLO detection confidence threshold |
| `VISIONPASS_FUZZY` | `0.85` — similarity needed for `review` instead of `denied` |

## Files

| Path | Purpose |
| --- | --- |
| `app.py` | Flask routes: `/`, `/api/detect`, `/api/database`, `/api/log` |
| `anpr.py` | Detection, OCR, normalisation and matching — no web code |
| `templates/index.html` | Single-page UI |
| `static/style.css`, `static/app.js` | Styling and client logic |
| `entry_log.csv` | Audit trail, appended on every check (created on first run) |

## Adding authorized vehicles

Append a row to `LICENSEPLATE_GUESTS_DATABASE.csv` under the `License Plate`
column. The file is re-read on every request, so no restart is needed.

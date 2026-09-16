"""VisionPass ANPR engine.

Wraps the pipeline from ANPR_YOLOv11_DEMO.ipynb: YOLOv11 detects the plate,
EasyOCR reads it, and the text is matched against the guest database.
"""

import base64
import os
import re
import threading
from difflib import SequenceMatcher

import cv2
import numpy as np
import pandas as pd

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

WEIGHTS_PATH = os.environ.get(
    "VISIONPASS_WEIGHTS", os.path.join(PROJECT_ROOT, "DEMO", "best.pt")
)
DATABASE_PATH = os.environ.get(
    "VISIONPASS_DB", os.path.join(PROJECT_ROOT, "LICENSEPLATE_GUESTS_DATABASE.csv")
)
PLATE_COLUMN = "License Plate"

CONF_THRESHOLD = float(os.environ.get("VISIONPASS_CONF", "0.25"))
# Below this similarity a plate is treated as unknown rather than a near-miss.
FUZZY_THRESHOLD = float(os.environ.get("VISIONPASS_FUZZY", "0.85"))

# Characters EasyOCR routinely confuses on Indian plates. These are groups, not
# a one-way map, because a glyph can be ambiguous in more than one direction:
# "G" is misread for both "0" and "6", so collapsing G->6 would lose the G->0
# case. Two plates match if they are the same length and every character pair is
# either identical or shares a group.
_CONFUSION_GROUPS = [
    frozenset("0OQDG"),
    frozenset("1IL"),
    frozenset("2Z"),
    frozenset("4A"),
    frozenset("5S"),
    frozenset("6G"),
    frozenset("7T"),
    frozenset("8B"),
]

# Indian plates carry an "IND" hologram strip that OCR picks up as plate text.
_IND_PREFIX = re.compile(r"^IND")

_model = None
_reader = None
_load_lock = threading.Lock()


def normalise(text):
    """Strip everything that is not A-Z or 0-9, uppercase, drop the IND strip."""
    cleaned = re.sub(r"[^A-Z0-9]", "", str(text).upper())
    return _IND_PREFIX.sub("", cleaned)


def _chars_confusable(a, b):
    if a == b:
        return True
    return any(a in group and b in group for group in _CONFUSION_GROUPS)


def _confusable_equal(a, b):
    """True if two normalised plates differ only by confusable characters."""
    return len(a) == len(b) and all(_chars_confusable(x, y) for x, y in zip(a, b))


def get_model():
    """Load the YOLOv11 weights once, lazily (import is slow)."""
    global _model
    if _model is None:
        with _load_lock:
            if _model is None:
                from ultralytics import YOLO

                _model = YOLO(WEIGHTS_PATH)
    return _model


def get_reader():
    """Load the EasyOCR reader once, lazily."""
    global _reader
    if _reader is None:
        with _load_lock:
            if _reader is None:
                import easyocr

                _reader = easyocr.Reader(["en"], gpu=False)
    return _reader


def warm_up():
    """Force both models into memory so the first upload is not slow."""
    get_model()
    get_reader()


def load_database(path=None):
    """Return the guest list as a list of {'plate', 'key', 'canonical'} dicts."""
    path = path or DATABASE_PATH
    if not os.path.exists(path):
        return []

    if path.lower().endswith((".xlsx", ".xls")):
        df = pd.read_excel(path)
    else:
        df = pd.read_csv(path)

    column = PLATE_COLUMN if PLATE_COLUMN in df.columns else df.columns[0]

    entries = []
    seen = set()
    for value in df[column].dropna():
        plate = str(value).strip().upper()
        key = normalise(plate)
        if not key or key in seen:
            continue
        seen.add(key)
        entries.append({"plate": plate, "key": key})
    return entries


def match_plate(text, entries):
    """Compare OCR text against the guest list.

    Returns (status, matched_entry_or_None, score) where status is one of
    'authorized', 'review' or 'denied'.
    """
    key = normalise(text)
    if not key:
        return "denied", None, 0.0

    for entry in entries:
        if entry["key"] == key:
            return "authorized", entry, 1.0

    # Same plate, read with a confusable character (0/O/G, 1/I, 5/S ...).
    for entry in entries:
        if _confusable_equal(key, entry["key"]):
            return "authorized", entry, 1.0

    best, best_score = None, 0.0
    for entry in entries:
        score = SequenceMatcher(None, key, entry["key"]).ratio()
        if score > best_score:
            best, best_score = entry, score

    if best_score >= FUZZY_THRESHOLD:
        return "review", best, best_score
    return "denied", best, best_score


def _ocr_plate(crop):
    """Read a cropped plate, returning (raw_text, normalised_text, confidence)."""
    reader = get_reader()

    # Upscale small crops; EasyOCR is noticeably better above ~100px height.
    height, width = crop.shape[:2]
    if height and width and height < 100:
        scale = 100.0 / height
        crop = cv2.resize(crop, (int(width * scale), 100), interpolation=cv2.INTER_CUBIC)

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    results = reader.readtext(gray)

    raw = " ".join(item[1] for item in results).strip().upper()
    confidences = [float(item[2]) for item in results]
    confidence = sum(confidences) / len(confidences) if confidences else 0.0
    return raw, normalise(raw), confidence


def _draw(image, box, label, colour):
    x1, y1, x2, y2 = box
    cv2.rectangle(image, (x1, y1), (x2, y2), colour, 3)
    if not label:
        return
    (text_w, text_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    top = max(y1 - text_h - 12, 0)
    cv2.rectangle(image, (x1, top), (x1 + text_w + 12, top + text_h + 12), colour, -1)
    cv2.putText(
        image,
        label,
        (x1 + 6, top + text_h + 4),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
    )


_STATUS_COLOURS = {
    "authorized": (76, 175, 80),   # green
    "review": (0, 193, 255),       # amber
    "denied": (60, 60, 220),       # red
}


def process_image(image_bgr, entries=None):
    """Run the full pipeline on one BGR frame.

    Returns a dict with the annotated image (base64 JPEG) and one record per
    detected plate.
    """
    entries = load_database() if entries is None else entries
    model = get_model()

    annotated = image_bgr.copy()
    detections = []

    results = model(image_bgr, conf=CONF_THRESHOLD, verbose=False)
    for result in results:
        for box in result.boxes:
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            x1, y1 = max(x1, 0), max(y1, 0)
            x2 = min(x2, image_bgr.shape[1])
            y2 = min(y2, image_bgr.shape[0])
            if x2 <= x1 or y2 <= y1:
                continue

            crop = image_bgr[y1:y2, x1:x2]
            raw_text, plate_text, ocr_conf = _ocr_plate(crop)
            status, entry, score = match_plate(plate_text, entries)

            detections.append(
                {
                    "box": [x1, y1, x2, y2],
                    "detection_confidence": round(float(box.conf[0]), 3),
                    "ocr_confidence": round(ocr_conf, 3),
                    "raw_text": raw_text,
                    "plate_text": plate_text,
                    "status": status,
                    "matched_plate": entry["plate"] if entry and status != "denied" else None,
                    "closest_plate": entry["plate"] if entry else None,
                    "match_score": round(score, 3),
                    "crop": _encode(crop),
                }
            )

            label = plate_text or "NO TEXT"
            _draw(annotated, (x1, y1, x2, y2), label, _STATUS_COLOURS[status])

    return {
        "annotated_image": _encode(annotated),
        "detections": detections,
        "database_size": len(entries),
    }


def _encode(image_bgr, quality=85):
    """Encode a BGR image as a base64 JPEG data URI."""
    ok, buffer = cv2.imencode(".jpg", image_bgr, [int(cv2.IMWRITE_JPEG_QUALITY), quality])
    if not ok:
        return None
    return "data:image/jpeg;base64," + base64.b64encode(buffer).decode("ascii")


def decode_upload(file_bytes):
    """Decode uploaded bytes into a BGR image, or None if unreadable."""
    array = np.frombuffer(file_bytes, np.uint8)
    return cv2.imdecode(array, cv2.IMREAD_COLOR)

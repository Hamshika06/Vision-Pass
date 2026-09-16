"""VisionPass - web front end for the YOLOv11 + EasyOCR ANPR pipeline."""

import base64
import csv
import os
import re
from datetime import datetime

from flask import Flask, jsonify, render_template, request

import anpr

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 16 * 1024 * 1024  # 16 MB uploads

LOG_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "entry_log.csv")
LOG_FIELDS = ["timestamp", "plate_text", "status", "matched_plate", "match_score", "source"]


def append_log(rows):
    """Append entry decisions to the audit log, creating it if needed."""
    is_new = not os.path.exists(LOG_PATH)
    with open(LOG_PATH, "a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=LOG_FIELDS)
        if is_new:
            writer.writeheader()
        writer.writerows(rows)


def read_log(limit=50):
    if not os.path.exists(LOG_PATH):
        return []
    with open(LOG_PATH, newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    return list(reversed(rows))[:limit]


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/api/database")
def api_database():
    entries = anpr.load_database()
    return jsonify(
        {
            "path": anpr.DATABASE_PATH,
            "count": len(entries),
            "plates": [entry["plate"] for entry in entries],
        }
    )


@app.route("/api/log")
def api_log():
    return jsonify({"entries": read_log()})


@app.route("/api/detect", methods=["POST"])
def api_detect():
    """Accept an uploaded file or a base64 webcam frame and run the pipeline."""
    image_bytes = None
    source = "upload"

    if "image" in request.files and request.files["image"].filename:
        image_bytes = request.files["image"].read()
    else:
        payload = request.get_json(silent=True) or {}
        data_uri = payload.get("image_data")
        if data_uri:
            source = payload.get("source", "camera")
            image_bytes = base64.b64decode(re.sub(r"^data:image/\w+;base64,", "", data_uri))

    if not image_bytes:
        return jsonify({"error": "No image supplied."}), 400

    image = anpr.decode_upload(image_bytes)
    if image is None:
        return jsonify({"error": "Could not read that file as an image."}), 400

    entries = anpr.load_database()
    if not entries:
        return jsonify({"error": f"Guest database is empty or missing: {anpr.DATABASE_PATH}"}), 500

    result = anpr.process_image(image, entries)

    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    if result["detections"]:
        append_log(
            [
                {
                    "timestamp": timestamp,
                    "plate_text": d["plate_text"],
                    "status": d["status"],
                    "matched_plate": d["matched_plate"] or "",
                    "match_score": d["match_score"],
                    "source": source,
                }
                for d in result["detections"]
            ]
        )

    result["timestamp"] = timestamp
    result["source"] = source
    return jsonify(result)


if __name__ == "__main__":
    print("VisionPass starting...")
    print(f"  weights : {anpr.WEIGHTS_PATH}")
    print(f"  database: {anpr.DATABASE_PATH}")
    print("  loading YOLOv11 + EasyOCR (first run downloads OCR models)...")
    anpr.warm_up()
    print("  ready -> http://127.0.0.1:5000")
    app.run(host="127.0.0.1", port=5000, debug=False)

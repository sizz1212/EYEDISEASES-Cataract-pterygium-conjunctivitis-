from flask import Flask, render_template, request, jsonify, Response
from ultralytics import YOLO
import cv2
import numpy as np
import threading
from flask_cors import CORS

# ---------------------------------------------------------
# FLASK APP
# ---------------------------------------------------------
app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------
# LOAD YOLO MODEL
# ---------------------------------------------------------
try:
    model = YOLO("best.pt")
    print("✅ YOLO model loaded successfully")
except Exception as e:
    print("❌ ERROR loading model:", e)


# ---------------------------------------------------------
# ROUTES
# ---------------------------------------------------------

# HOME PAGE
@app.route("/")
def index():
    return render_template("index.html")


# PRIVACY PAGE (optional)
@app.route("/privacy")
def privacy():
    return render_template("privacy.html")


# AI PAGE (SCAN USING AI)
@app.route("/ai")
def ai_page():
    return render_template("AI_PAGE.html")


# -----------------------------------------
# NEW PAGES (replaced pop-ups)
# -----------------------------------------

@app.route("/developers")
def developers():
    return render_template("developers.html")


@app.route("/user_guide")
def user_guide():
    return render_template("user_guide.html")


@app.route("/data_security")
def data_security():
    return render_template("data_security.html")


@app.route("/about_page")
def about_page():
    return render_template("about_page.html")


# ---------------------------------------------------------
# AI PREDICTION ENDPOINT
# ---------------------------------------------------------
@app.route("/predict", methods=["POST"])
def predict():
    file = request.files.get("image")

    if file is None:
        return jsonify({"prediction": "No image received"}), 400

    try:
        img_bytes = np.frombuffer(file.read(), np.uint8)
        img = cv2.imdecode(img_bytes, cv2.IMREAD_COLOR)

        if img is None:
            return jsonify({"prediction": "Invalid image format"}), 400

        results = model(img)[0]

        # DETECTION
        if results.boxes is not None and len(results.boxes) > 0:
            cls_id = int(results.boxes[0].cls)
            label = results.names[cls_id]
            return jsonify({"prediction": label})

        # CLASSIFICATION
        if hasattr(results, "probs") and results.probs is not None:
            cls_id = int(results.probs.top1)
            label = results.names[cls_id]
            return jsonify({"prediction": label})

        return jsonify({"prediction": "No disease detected"})

    except Exception as e:
        print("❌ Prediction ERROR:", e)
        return jsonify({"prediction": "Error analyzing image"}), 500


# ---------------------------------------------------------
# CAMERA / VIDEO STREAMING (real-time detection page)
# ---------------------------------------------------------
camera = None
camera_lock = threading.Lock()


def get_camera():
    global camera
    with camera_lock:
        if camera is None:
            camera = cv2.VideoCapture(0)  # 0 = default webcam; change index if needed
    return camera


def generate_frames():
    cam = get_camera()
    while True:
        success, frame = cam.read()
        if not success:
            break

        results = model(frame, verbose=False)[0]

        # Draw detection boxes + labels directly on the frame
        if results.boxes is not None and len(results.boxes) > 0:
            for box in results.boxes:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                cls_id = int(box.cls)
                label = results.names[cls_id]
                conf = float(box.conf)

                cv2.rectangle(frame, (x1, y1), (x2, y2), (11, 138, 131), 2)
                cv2.putText(
                    frame,
                    f"{label} {conf:.0%}",
                    (x1, max(y1 - 10, 20)),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    (11, 138, 131),
                    2,
                )

        ok, buffer = cv2.imencode(".jpg", frame)
        if not ok:
            continue

        frame_bytes = buffer.tobytes()
        yield (
            b"--frame\r\n"
            b"Content-Type: image/jpeg\r\n\r\n" + frame_bytes + b"\r\n"
        )


@app.route("/camera")
def camera_page():
    return render_template("camera.html")


@app.route("/video_feed")
def video_feed():
    return Response(
        generate_frames(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------------------------------------------------
# START SERVER
# ---------------------------------------------------------
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=10000)
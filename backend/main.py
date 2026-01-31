from fastapi import FastAPI
from fastapi.responses import StreamingResponse, HTMLResponse
import cv2
import numpy as np
from vision.style_engine import StyleEngine
import tensorflow as tf
import sounddevice as sd
import threading
import time

app = FastAPI()

# Load models
guitar_model = tf.keras.models.load_model("models/guitar_model.h5")
chord_model = tf.keras.models.load_model("models/chord_model.h5")

# Style models
styles = {
    "starry": StyleEngine("models/style/starry-night.model"),
    "muse": StyleEngine("models/style/la_muse.model"),
    "vii": StyleEngine("models/style/composition_vii.model")
}

current_style = None

# Camera
cap = cv2.VideoCapture(0)

latest_frame = None
lock = threading.Lock()

SR = 44100
DUR = 2


# Camera thread
def camera_loop():
    global latest_frame

    while True:
        ret, frame = cap.read()
        if not ret:
            continue

        with lock:
            latest_frame = frame.copy()

        time.sleep(0.03)  # ~30 FPS


threading.Thread(target=camera_loop, daemon=True).start()


def generate_frames():
    global latest_frame, current_style

    while True:

        with lock:
            if latest_frame is None:
                continue

            frame = latest_frame.copy()

        # Apply style if enabled
        if current_style:
            frame = styles[current_style].stylize(frame)

        _, jpeg = cv2.imencode('.jpg', frame)

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')



@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.get("/guitar")
def guitar_type():

    with lock:
        frame = latest_frame.copy()

    img = cv2.resize(frame,(256,256))
    img = img/255.0
    img = np.expand_dims(img,0)

    pred = guitar_model.predict(img, verbose=0)

    return {
        "class": int(np.argmax(pred)),
        "confidence": float(np.max(pred))
    }


@app.get("/chord")
def chord():

    audio = sd.rec(int(SR*DUR),
                   samplerate=SR,
                   channels=1,
                   dtype='float32')
    sd.wait()

    audio = audio.flatten()
    audio = np.expand_dims(audio,0)

    pred = chord_model.predict(audio, verbose=0)

    return {
        "class": int(np.argmax(pred)),
        "confidence": float(np.max(pred))
    }

@app.post("/style/{name}")
def set_style(name: str):

    global current_style

    if name in styles:
        current_style = name
        return {"status":"on", "style":name}

    current_style = None
    return {"status":"off"}

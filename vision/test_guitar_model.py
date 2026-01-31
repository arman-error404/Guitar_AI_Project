import cv2
import numpy as np
import tensorflow as tf

model = tf.keras.models.load_model("models/guitar_model.h5")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.resize(frame, (256,256))
    img = img / 255.0
    img = np.expand_dims(img, 0)

    pred = model.predict(img, verbose=0)

    cid = np.argmax(pred)
    conf = np.max(pred)

    txt = f"Class {cid} | {conf:.2f}"

    cv2.putText(frame, txt,
                (20,40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,(0,255,0),2)

    cv2.imshow("Guitar AI", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


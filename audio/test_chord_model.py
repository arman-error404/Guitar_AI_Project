import sounddevice as sd
import numpy as np
import tensorflow as tf

SR = 44100
DUR = 2

model = tf.keras.models.load_model("models/chord_model.h5")

def record():
    print("Recording...")
    audio = sd.rec(int(SR*DUR),
                   samplerate=SR,
                   channels=1,
                   dtype='float32')
    sd.wait()
    print("Done")
    return audio.flatten()

while True:
    input("Press Enter to record chord")

    data = record()
    data = np.expand_dims(data,0)

    pred = model.predict(data, verbose=0)

    print("Chord:", np.argmax(pred),
          "Conf:", np.max(pred))


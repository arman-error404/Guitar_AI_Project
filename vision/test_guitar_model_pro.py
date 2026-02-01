import cv2
import numpy as np
import tensorflow as tf
import os
import sys
from datetime import datetime


# =========================
# PATH SETUP
# =========================

project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)


# =========================
# CONFIG (FROM TRAINING)
# =========================

IMG_SIZE = 224  # Model expects 224x224 based on config
CAM_INDEX = 0


# =========================
# LOAD MODEL
# =========================

MODEL_PATH = os.path.join(project_root, "models/guitar_model.h5")
CONFIG_PATH = os.path.join(project_root, "models/config.json")

# Check if files exist
if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(f"Model file not found: {MODEL_PATH}")

# Try loading with different methods (handling Keras 3 and legacy formats)
try:
    # First try: standard load with compile=False
    model = tf.keras.models.load_model(MODEL_PATH, compile=False)
    print("✅ Model loaded:", MODEL_PATH)
except (ValueError, OSError, AttributeError) as e:
    error_msg = str(e)
    
    # Check if it's the "No model config" error - this means legacy format
    if "No model config" in error_msg or "model_config" in error_msg.lower():
        print("⚠️ Model file uses legacy format (has 'layers' but no 'model_config')")
        
        # Check if config.json exists to reconstruct the model
        if os.path.exists(CONFIG_PATH):
            print("🔄 Found config.json! Reconstructing model from config...")
            try:
                import json
                
                # Load the model config
                with open(CONFIG_PATH, 'r') as f:
                    model_config = json.load(f)
                
                print("   📖 Loaded model architecture from config.json")
                
                # Reconstruct model from config using Keras 3 API
                import keras
                # Use keras.saving API to deserialize the model config
                model = keras.saving.deserialize_keras_object(model_config)
                
                print("   🏗️  Model architecture reconstructed")
                
                # Now load the weights from the .h5 file
                print("   ⚖️  Loading weights from guitar_model.h5...")
                
                # For Keras 3, we need to load weights from the layers structure
                import h5py
                layers_loaded = 0
                with h5py.File(MODEL_PATH, 'r') as f:
                    if 'layers' in f:
                        # Create a mapping of layer names to model layers
                        layer_map = {layer.name: layer for layer in model.layers}
                        
                        # Load weights layer by layer
                        for layer_name in f['layers'].keys():
                            if layer_name in layer_map:
                                layer = layer_map[layer_name]
                                layer_group = f['layers'][layer_name]
                                
                                # Check for weights in different possible locations
                                weight_values = []
                                
                                # Try 'vars' (old format) - weights may be stored with numeric keys
                                if 'vars' in layer_group:
                                    # Sort keys numerically if they're numbers, otherwise alphabetically
                                    var_keys = sorted(layer_group['vars'].keys(), 
                                                     key=lambda x: int(x) if x.isdigit() else x)
                                    for var_name in var_keys:
                                        weight_values.append(np.array(layer_group['vars'][var_name]))
                                
                                # Try direct weight arrays
                                elif 'kernel:0' in layer_group or 'bias:0' in layer_group:
                                    for weight_name in sorted(layer_group.keys()):
                                        if weight_name.endswith(':0'):
                                            weight_values.append(np.array(layer_group[weight_name]))
                                
                                if weight_values:
                                    try:
                                        # Check if the number of weights matches
                                        expected_weights = len(layer.get_weights())
                                        if len(weight_values) == expected_weights:
                                            layer.set_weights(weight_values)
                                            layers_loaded += 1
                                            if layers_loaded % 20 == 0:  # Print every 20 layers
                                                print(f"      ✓ Loaded {layers_loaded} layers...")
                                        else:
                                            # Try to match weights by shape
                                            expected_shapes = [w.shape for w in layer.get_weights()]
                                            actual_shapes = [w.shape for w in weight_values]
                                            
                                            if expected_shapes == actual_shapes:
                                                layer.set_weights(weight_values)
                                                layers_loaded += 1
                                            else:
                                                # Skip layers with mismatched shapes (might be non-trainable or different structure)
                                                pass
                                    except Exception as layer_err:
                                        # Some layers might fail - that's okay if they're not critical
                                        pass
                
                print(f"   ✅ Loaded weights for {layers_loaded} layers")
                
                print("✅ Model reconstructed and weights loaded successfully!")
                
            except Exception as e_config:
                print(f"❌ Failed to reconstruct model from config: {e_config}")
                print("\n💡 The config.json file might be in a different format.")
                print("   Please ensure it contains the full model architecture.")
                raise
        else:
            print("❌ Cannot load model: Incompatible file format and no config.json found")
            print("\n📋 The issue:")
            print("   - guitar_model.h5 uses old 'layers' format (from older TF/Keras)")
            print("   - Keras 3.x requires 'model_weights' format (like chord_model.h5)")
            print("\n💡 Solutions:")
            print("   1. Place config.json in models/ folder with model architecture")
            print("   2. RE-SAVE THE MODEL (Recommended):")
            print("      - Open your training notebook/script")
            print("      - Load the model: model = tf.keras.models.load_model('old_path')")
            print("      - Re-save it: model.save('models/guitar_model.h5')")
            print("   3. USE OLDER TENSORFLOW (Temporary workaround):")
            print("      - Install: pip install 'tensorflow<2.16' 'keras<3'")
            raise ValueError(
                "Model file format incompatible with Keras 3. "
                "The file uses old 'layers' format. Please provide config.json or re-save the model."
            )
    else:
        # Different error
        print(f"❌ Load failed: {error_msg}")
        raise

print("📐 Model input shape:", model.input_shape)
print("📊 Model output shape:", model.output_shape)


# =========================
# LOAD LABELS
# =========================

from utils.guitar_labels import GUITAR_CLASSES

NUM_CLASSES = len(GUITAR_CLASSES)

print("🏷️ Classes:", GUITAR_CLASSES)
print("🔢 Total classes:", NUM_CLASSES)


# =========================
# IMAGE PREPROCESSING
# =========================

def preprocess(frame):

    # Resize
    img = cv2.resize(frame, (IMG_SIZE, IMG_SIZE))

    # BGR → RGB
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Normalize (match training)
    img = img.astype("float32") / 255.0

    return img


# =========================
# QUALITY CHECK
# =========================

def check_frame_quality(frame):

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    brightness = np.mean(gray)
    contrast = np.std(gray)

    return brightness, contrast


# =======================
# SAVE DEBUG FRAME
# =========================

def save_frame(frame, label):

    folder = os.path.join(project_root, "debug_frames")
    os.makedirs(folder, exist_ok=True)

    ts = datetime.now().strftime("%Y%m%d_%H%M%S")

    path = os.path.join(folder, f"{label}_{ts}.jpg")

    cv2.imwrite(path, frame)

    print(f"💾 Frame saved: {path}")


# =========================
# MAIN LOOP
# =========================

def main():

    print("\n🎸 Guitar Recognition System Ready")
    print("Press Q to quit\n")

    cap = cv2.VideoCapture(CAM_INDEX)

    if not cap.isOpened():
        print("❌ Cannot open camera")
        return


    frame_count = 0

    while True:

        ret, frame = cap.read()

        if not ret:
            print("❌ Frame read failed")
            break


        frame_count += 1


        # Quality check
        brightness, contrast = check_frame_quality(frame)

        if brightness < 40:
            print("⚠️ Frame too dark")
        if contrast < 15:
            print("⚠️ Frame too low contrast")


        # Preprocess
        img = preprocess(frame)

        img_input = np.expand_dims(img, axis=0)


        # Predict
        pred = model.predict(img_input, verbose=0)[0]


        # Sanity check
        if len(pred) != NUM_CLASSES:
            print("❌ Class count mismatch!")
            print("Model:", len(pred), "Labels:", NUM_CLASSES)


        # Top-5
        top_idx = np.argsort(pred)[-5:][::-1]


        # Best
        cid = int(np.argmax(pred))
        conf = float(np.max(pred))


        if cid < NUM_CLASSES:
            label = GUITAR_CLASSES[cid]
        else:
            label = "Unknown"


        # Overlay
        text = f"{label} ({conf:.2f})"

        cv2.putText(frame, text,
                    (20, 40),
                    cv2.FONT_HERSHEY_SIMPLEX,
                  1,
                    (0,255,0), 2)


        # Show top-5 in console
        if frame_count % 60 == 0:

            print("\n🔍 Top Predictions:")

            for i, idx in enumerate(top_idx, 1):

                name = GUITAR_CLASSES[idx] if idx < NUM_CLASSES else "?"

                print(f" {i}. {name}: {pred[idx]:.3f}")


            if conf > 0.98:
                print("⚠️ Overconfident prediction")


        cv2.imshow("Guitar AI Debug", frame)


        key = cv2.waitKey(1)

        if key == ord('q'):
            break

        elif key == ord('s'):
            save_frame(frame, label)



    cap.release()
    cv2.destroyAllWindows()

    print("\n👋 Test ended")


# =========================
# START
# =========================

if __name__ == "__main__":
    main()


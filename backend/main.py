import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from fastapi import FastAPI, UploadFile, File
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from fastapi.responses import StreamingResponse, HTMLResponse, JSONResponse
from fastapi import HTTPException
from pydantic import BaseModel
import cv2
import numpy as np
import tensorflow as tf
import sounddevice as sd
import threading
import time
import joblib
import json
from datetime import datetime
from typing import Optional

app = FastAPI()

# Create directories first before mounting
GUITAR_IMAGES_DIR = "saved_guitars"
DIFFUSION_INPUT_DIR = "diffusion_inputs"
DIFFUSION_OUTPUT_DIR = "diffusion_outputs"
os.makedirs(GUITAR_IMAGES_DIR, exist_ok=True)
os.makedirs(DIFFUSION_INPUT_DIR, exist_ok=True)
os.makedirs(DIFFUSION_OUTPUT_DIR, exist_ok=True)

app.mount("/static", StaticFiles(directory="frontend"), name="static")

# Directories are created above before app initialization

@app.get("/")
def home():
    return FileResponse("frontend/index.html")


@app.get("/health")
def health_check():
    """Health check endpoint"""
    return {"status": "ok", "endpoints": {
        "generate_image": "/generate/image (POST)",
        "get_image": "/generate/image/{filename} (GET)"
    }}

# Load models
# Try to load guitar model - handle weights-only file
guitar_model = None
guitar_class_names = None

# First, try to load class names from notebook (class_names.npy)
try:
    class_names_path = "models/class_names.npy"
    if os.path.exists(class_names_path):
        guitar_class_names = np.load(class_names_path).tolist()
        print(f"Loaded guitar class names from notebook: {guitar_class_names}")
    else:
        # Fallback to guitar_labels.py
        from utils.guitar_labels import GUITAR_CLASSES
        guitar_class_names = GUITAR_CLASSES
        print(f"Using guitar class names from utils: {guitar_class_names}")
except Exception as e:
    print(f"Error loading class names: {e}")
    from utils.guitar_labels import GUITAR_CLASSES
    guitar_class_names = GUITAR_CLASSES

# Try multiple model paths
model_paths = [
    "models/best_guitar_body_style_model.keras",  # From notebook
    "models/guitar_model.h5",  # Alternative path
    "models/model.weights.h5"  # Current path
]

for model_path in model_paths:
    if os.path.exists(model_path):
        try:
            guitar_model = tf.keras.models.load_model(model_path, compile=False)
            print(f"Guitar model loaded successfully from: {model_path}")
            break
        except Exception as e:
            print(f"Error loading model from {model_path}: {e}")
            continue

# If still not loaded, try config.json approach
if guitar_model is None:
    config_path = "models/config.json"
    weights_path = "models/model.weights.h5"
    
    if os.path.exists(config_path) and os.path.exists(weights_path):
        try:
            print("Loading guitar model from config.json and weights...")
            with open(config_path, 'r') as f:
                model_config = json.load(f)
            
            # Reconstruct model from config
            import keras
            guitar_model = keras.saving.deserialize_keras_object(model_config)
            
            # Load weights
            guitar_model.load_weights(weights_path)
            print("Guitar model loaded successfully from config and weights")
        except Exception as e2:
            print(f"Error loading from config: {e2}")
            guitar_model = None

if guitar_model is None:
    print("WARNING: Guitar model not loaded. Guitar detection will not work.")
else:
    print(f"Guitar model input shape: {guitar_model.input_shape}")
    print(f"Guitar model output shape: {guitar_model.output_shape}")

chord_model = None
try:
    chord_model = tf.keras.models.load_model("models/chord_model.h5")
    print("Chord model loaded successfully")
except Exception as e:
    print(f"Error loading chord model: {e}")
    chord_model = None

# Load chord labels
from utils.chord_labels import CHORDS
# Guitar class names are loaded above

# Load style transfer
from utils.style_transfer import StyleTransferProcessor, AVAILABLE_STYLES

# Load audio enhancement
from utils.audio_gan import AudioEnhancementProcessor

# Load diffusion model
from utils.diffusion_model import get_generator

# Initialize style transfer processor
# Set use_gan=True to enable GAN-based style transfer (requires trained models)
# Set use_gan=False to use filter-based approach (works immediately, no training needed)
USE_GAN = os.getenv("USE_GAN_STYLE_TRANSFER", "false").lower() == "true"
style_processor = StyleTransferProcessor(use_gan=USE_GAN)
current_style = "none"  # Default: no style

# Initialize audio enhancement processor
# Set use_gan=True to enable GAN-based audio enhancement (requires trained models)
# Set use_gan=False to use filter-based approach (works immediately, no training needed)
USE_AUDIO_GAN = os.getenv("USE_AUDIO_GAN", "false").lower() == "true"
audio_enhancer = AudioEnhancementProcessor(use_gan=USE_AUDIO_GAN)

# Initialize diffusion generator (lazy loaded on first use)
diffusion_generator = None

# Chord input amplifier settings (auto gain + gentle saturation)
CHORD_AMP_ENABLED = os.getenv("CHORD_AMP", "true").lower() == "true"
CHORD_AMP_TARGET_RMS = float(os.getenv("CHORD_AMP_TARGET_RMS", "0.08"))
CHORD_AMP_MAX_GAIN = float(os.getenv("CHORD_AMP_MAX_GAIN", "8.0"))
CHORD_AMP_SOFT_CLIP = float(os.getenv("CHORD_AMP_SOFT_CLIP", "1.5"))
CHORD_AMP_SILENCE_RMS = float(os.getenv("CHORD_AMP_SILENCE_RMS", "1e-4"))


def amplify_chord_audio(audio: np.ndarray) -> tuple[np.ndarray, dict]:
    """
    Stabilize chord input loudness with automatic gain and soft clipping.
    Returns the processed audio and stats for debugging.
    """
    if not CHORD_AMP_ENABLED:
        return audio, {"enabled": False}

    if audio.size == 0:
        return audio, {"enabled": True, "status": "empty"}

    rms = float(np.sqrt(np.mean(audio ** 2)))
    if rms < CHORD_AMP_SILENCE_RMS:
        return audio, {"enabled": True, "status": "silent", "rms": rms, "gain": 1.0}

    gain = min(CHORD_AMP_MAX_GAIN, CHORD_AMP_TARGET_RMS / (rms + 1e-9))
    amplified = audio * gain

    if CHORD_AMP_SOFT_CLIP > 0:
        # Gentle saturation to tame peaks without hard clipping
        amplified = np.tanh(amplified * CHORD_AMP_SOFT_CLIP) / np.tanh(CHORD_AMP_SOFT_CLIP)

    peak = float(np.max(np.abs(amplified)))
    return amplified.astype(np.float32), {
        "enabled": True,
        "rms": rms,
        "gain": float(gain),
        "peak": peak
    }

# Camera
cap = cv2.VideoCapture(0)

latest_frame = None
lock = threading.Lock()
style_lock = threading.Lock()

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
        
        # Apply style transfer if enabled
        with style_lock:
            style = current_style
        
        if style != "none":
            try:
                frame = style_processor.apply_style(frame, style)
            except Exception as e:
                print(f"Style transfer error: {e}")
                # Continue with original frame if style transfer fails

        _, jpeg = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 85])

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' +
               jpeg.tobytes() + b'\r\n')



@app.get("/video")
def video_feed():
    return StreamingResponse(
        generate_frames(),
        media_type="multipart/x-mixed-replace; boundary=frame"
    )


@app.post("/guitar/detect")
async def guitar_detect_from_image(file: UploadFile = File(...)):
    """Detect guitar type from uploaded image"""
    try:
        # Check if model is loaded
        if guitar_model is None:
            return {
                "error": "Guitar model not loaded",
                "class": -1,
                "class_name": "Model Not Available",
                "type": "Unknown",
                "brand": "Unknown",
                "confidence": 0.0
            }
        
        # Read image file
        contents = await file.read()
        nparr = np.frombuffer(contents, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if frame is None:
            return {
                "error": "Failed to decode image",
                "class": -1,
                "class_name": "Invalid Image",
                "type": "Unknown",
                "brand": "Unknown",
                "confidence": 0.0
            }

        # Preprocess image - use 224x224 to match notebook model
        img = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB (model expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        # Predict
        pred = guitar_model.predict(img, verbose=0)
        class_idx = int(np.argmax(pred[0]))
        confidence = float(np.max(pred[0]))
        
        # Get guitar type name from loaded class names
        if guitar_class_names and class_idx < len(guitar_class_names):
            class_name = guitar_class_names[class_idx]
        else:
            class_name = f"Unknown (index {class_idx})"
        
        # Parse class name to extract type and brand
        type_name = class_name
        brand_name = "Unknown"
        
        # Map class names to more readable types and brands
        type_mapping = {
            "electric_solid": {"type": "Electric Solid Body", "brand": "Various"},
            "electric_offset": {"type": "Electric Offset", "brand": "Fender"},
            "electric_hollow": {"type": "Electric Hollow Body", "brand": "Gibson"},
            "acoustic": {"type": "Acoustic", "brand": "Various"},
            "classical": {"type": "Classical", "brand": "Various"}
        }
        
        # Try to find mapping
        class_name_lower = class_name.lower()
        if class_name_lower in type_mapping:
            type_name = type_mapping[class_name_lower]["type"]
            brand_name = type_mapping[class_name_lower]["brand"]
        else:
            # Format the class name nicely
            type_name = class_name.replace("_", " ").title()

        # Save the captured image
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"guitar_{class_name}_{confidence:.2f}_{timestamp}.jpg"
            filepath = os.path.join(GUITAR_IMAGES_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"✅ Saved guitar image: {filepath}")
        except Exception as save_error:
            print(f"⚠️ Failed to save guitar image: {save_error}")

        return {
            "class": class_idx,
            "class_name": class_name,
            "type": type_name,
            "brand": brand_name,
            "confidence": confidence,
            "image_saved": True,
            "image_path": filepath if 'filepath' in locals() else None
        }
    except Exception as e:
        print(f"Error in guitar detection from image: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "class": -1,
            "class_name": "Error",
            "type": "Unknown",
            "brand": "Unknown",
            "confidence": 0.0
        }


@app.get("/guitar")
def guitar_type():
    try:
        # Check if model is loaded
        if guitar_model is None:
            return {
                "error": "Guitar model not loaded",
                "class": -1,
                "class_name": "Model Not Available",
                "type": "Unknown",
                "brand": "Unknown",
                "confidence": 0.0
            }
        
        # Check if frame is available
        with lock:
            if latest_frame is None:
                return {
                    "error": "Camera frame not available",
                    "class": -1,
                    "class_name": "No Frame",
                    "type": "Unknown",
                    "brand": "Unknown",
                    "confidence": 0.0
                }
            frame = latest_frame.copy()

        # Preprocess image - use 224x224 to match notebook model
        img = cv2.resize(frame, (224, 224))
        # Convert BGR to RGB (model expects RGB)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        # Normalize to [0, 1]
        img = img.astype(np.float32) / 255.0
        img = np.expand_dims(img, 0)

        # Predict
        pred = guitar_model.predict(img, verbose=0)
        class_idx = int(np.argmax(pred[0]))
        confidence = float(np.max(pred[0]))
        
        # Get guitar type name from loaded class names
        if guitar_class_names and class_idx < len(guitar_class_names):
            class_name = guitar_class_names[class_idx]
        else:
            class_name = f"Unknown (index {class_idx})"
        
        # Parse class name to extract type and brand
        # Class names are like: "electric_solid", "acoustic", "classical", etc.
        type_name = class_name
        brand_name = "Unknown"
        
        # Map class names to more readable types and brands
        type_mapping = {
            "electric_solid": {"type": "Electric Solid Body", "brand": "Various"},
            "electric_offset": {"type": "Electric Offset", "brand": "Fender"},
            "electric_hollow": {"type": "Electric Hollow Body", "brand": "Gibson"},
            "acoustic": {"type": "Acoustic", "brand": "Various"},
            "classical": {"type": "Classical", "brand": "Various"}
        }
        
        # Try to find mapping
        class_name_lower = class_name.lower()
        if class_name_lower in type_mapping:
            type_name = type_mapping[class_name_lower]["type"]
            brand_name = type_mapping[class_name_lower]["brand"]
        else:
            # Format the class name nicely
            type_name = class_name.replace("_", " ").title()

        # Save the captured image
        filepath = None
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"guitar_{class_name}_{confidence:.2f}_{timestamp}.jpg"
            filepath = os.path.join(GUITAR_IMAGES_DIR, filename)
            cv2.imwrite(filepath, frame)
            print(f"✅ Saved guitar image: {filepath}")
        except Exception as save_error:
            print(f"⚠️ Failed to save guitar image: {save_error}")

        return {
            "class": class_idx,
            "class_name": class_name,
            "type": type_name,
            "brand": brand_name,
            "confidence": confidence,
            "image_saved": filepath is not None,
            "image_path": filepath
        }
    except Exception as e:
        print(f"Error in guitar detection: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "class": -1,
            "class_name": "Error",
            "type": "Unknown",
            "brand": "Unknown",
            "confidence": 0.0
        }


@app.get("/chord")
def chord():
    try:
        # Check if model is loaded
        if chord_model is None:
            return {
                "error": "Chord model not loaded",
                "class": "Model Not Available",
                "confidence": 0.0
            }
        
        import librosa
        import soundfile as sf

        # Use correct sample rate for chord model (matches training/test)
        SAMPLE_RATE = 22050  # Match test_chord_model.py
        DURATION = 2.0
        
        audio = sd.rec(int(SAMPLE_RATE * DURATION),
                       samplerate=SAMPLE_RATE,
                       channels=1,
                       dtype='float32')
        sd.wait()
        audio = audio.flatten()

        # Stabilize input loudness for more consistent chord predictions
        audio, amp_stats = amplify_chord_audio(audio)
        if amp_stats.get("enabled"):
            print(f"Chord amp stats: {amp_stats}")

        # Save audio to file for later use
        sf.write('last_audio.wav', audio, SAMPLE_RATE)

        # Convert audio to spectrogram (match test_chord_model.py exactly)
        BINS_PER_OCTAVE = 36
        N_BINS = 7 * BINS_PER_OCTAVE  # 252
        MAX_LEN = 256
        
        # Compute CQT spectrogram
        cqt = np.abs(librosa.cqt(
            y=audio,
            sr=SAMPLE_RATE,
            bins_per_octave=BINS_PER_OCTAVE,
            n_bins=N_BINS
        ))
        
        # Convert to dB scale (MATCH TRAINING: use ref=np.max)
        spec = librosa.amplitude_to_db(cqt, ref=np.max)
        
        # Pad or truncate to expected size (NO RESIZING - preserves frequency structure)
        if spec.shape[1] < MAX_LEN:
            spec = np.pad(spec, ((0, 0), (0, MAX_LEN - spec.shape[1])))
        else:
            spec = spec[:, :MAX_LEN]
        
        # Apply GAN-based audio enhancement if available
        # GAN preserves dB scale, so output is still in dB (raw values)
        # The GAN normalizes internally but denormalizes back to original dB scale
        try:
            enhanced_spec = audio_enhancer.enhance_spectrogram(spec, spec_type="cqt")
            # Validate that enhanced spec has same shape and reasonable dB range
            if enhanced_spec is not None and enhanced_spec.shape == spec.shape:
                # Check if values are in reasonable dB range (typically -80 to 0 dB)
                if enhanced_spec.min() > -100 and enhanced_spec.max() < 10:
                    spec = enhanced_spec
                else:
                    print(f"Warning: GAN output has unusual dB range [{enhanced_spec.min():.2f}, {enhanced_spec.max():.2f}], using original")
            else:
                print(f"Warning: GAN output shape mismatch or None, using original spectrogram")
        except Exception as e:
            print(f"Audio enhancement error: {e}, using original spectrogram")
        
        # IMPORTANT: Model expects RAW dB values (NOT normalized)
        # DO NOT normalize - match test_chord_model.py line 98-100
        # Just ensure shape is correct: (252, 256, 1)
        spec_final = np.expand_dims(spec, axis=-1)  # (252, 256, 1)
        spec_final = np.expand_dims(spec_final, axis=0)  # (1, 252, 256, 1)

        pred = chord_model.predict(spec_final, verbose=0)
        class_idx = int(np.argmax(pred))
        
        # Use CHORDS list directly instead of encoder
        if class_idx < len(CHORDS):
            chord_name = CHORDS[class_idx]
        else:
            chord_name = f"Unknown (index {class_idx})"

        return {
            "class": chord_name,
            "confidence": float(np.max(pred))
        }
    except Exception as e:
        print(f"Error in chord detection: {e}")
        import traceback
        traceback.print_exc()
        return {
            "error": str(e),
            "class": "Error",
            "confidence": 0.0
        }


@app.get("/styles")
def get_styles():
    """Get available style options"""
    return {
        "styles": AVAILABLE_STYLES,
        "current": current_style
    }


@app.post("/style/{style_name}")
def set_style(style_name: str):
    """Set the current style for video feed"""
    global current_style
    
    if style_name not in AVAILABLE_STYLES:
        return {
            "error": f"Style '{style_name}' not found",
            "available_styles": list(AVAILABLE_STYLES.keys())
        }
    
    with style_lock:
        current_style = style_name
    
    return {
        "success": True,
        "style": style_name,
        "style_display_name": AVAILABLE_STYLES[style_name]
    }


@app.get("/style/current")
def get_current_style():
    """Get the current active style"""
    return {
        "style": current_style,
        "style_display_name": AVAILABLE_STYLES.get(current_style, "Unknown")
    }


@app.get("/chord/diagram/{chord_name}")
def get_chord_diagram(chord_name: str):
    """Get chord diagram data for a given chord name"""
    # Parse chord name to extract base note and type
    chord_name_clean = chord_name.strip()
    
    # Basic chord diagram data structure
    # This is a simplified version - you can expand this with actual chord fingerings
    chord_data = {
        "name": chord_name_clean,
        "fingering": get_chord_fingering(chord_name_clean),
        "notes": get_chord_notes(chord_name_clean)
    }
    
    return chord_data


def get_chord_fingering(chord_name: str):
    """Get finger positions for a chord (simplified - returns common fingerings)"""
    # Common chord fingerings - this is a simplified mapping
    # In a real app, you'd have a comprehensive database
    common_chords = {
        "F": {"strings": [1, 3, 3, 2, 1, 1], "frets": [1, 3, 3, 2, 1, 1], "capo": 0},
        "F-major": {"strings": [1, 3, 3, 2, 1, 1], "frets": [1, 3, 3, 2, 1, 1], "capo": 0},
        "F1": {"strings": [1, 3, 3, 2, 1, 1], "frets": [1, 3, 3, 2, 1, 1], "capo": 0},
        "F2": {"strings": [1, 3, 3, 2, 1, 1], "frets": [1, 3, 3, 2, 1, 1], "capo": 0},
        "F3": {"strings": [1, 3, 3, 2, 1, 1], "frets": [1, 3, 3, 2, 1, 1], "capo": 0},
        "F4": {"strings": [1, 3, 3, 2, 1, 1], "frets": [1, 3, 3, 2, 1, 1], "capo": 0},
        "F5": {"strings": [1, 3, 3, 2, 1, 1], "frets": [1, 3, 3, 2, 1, 1], "capo": 0},
        "C": {"strings": [0, 1, 0, 2, 1, 0], "frets": [0, 1, 0, 2, 1, 0], "capo": 0},
        "C-major": {"strings": [0, 1, 0, 2, 1, 0], "frets": [0, 1, 0, 2, 1, 0], "capo": 0},
        "G": {"strings": [3, 0, 0, 0, 2, 3], "frets": [3, 0, 0, 0, 2, 3], "capo": 0},
        "G-major": {"strings": [3, 0, 0, 0, 2, 3], "frets": [3, 0, 0, 0, 2, 3], "capo": 0},
        "A": {"strings": [0, 0, 2, 2, 2, 0], "frets": [0, 0, 2, 2, 2, 0], "capo": 0},
        "A-major": {"strings": [0, 0, 2, 2, 2, 0], "frets": [0, 0, 2, 2, 2, 0], "capo": 0},
        "D": {"strings": [0, 0, 0, 2, 3, 2], "frets": [0, 0, 0, 2, 3, 2], "capo": 0},
        "D-major": {"strings": [0, 0, 0, 2, 3, 2], "frets": [0, 0, 0, 2, 3, 2], "capo": 0},
        "E": {"strings": [0, 2, 2, 1, 0, 0], "frets": [0, 2, 2, 1, 0, 0], "capo": 0},
        "E-major": {"strings": [0, 2, 2, 1, 0, 0], "frets": [0, 2, 2, 1, 0, 0], "capo": 0},
    }
    
    # Try to find exact match or partial match
    chord_key = chord_name_clean
    if chord_key not in common_chords:
        # Try to extract base note
        base_note = chord_name_clean[0] if len(chord_name_clean) > 0 else "C"
        if base_note in common_chords:
            chord_key = base_note
        else:
            # Default to C major
            chord_key = "C"
    
    return common_chords.get(chord_key, common_chords["C"])


def get_chord_notes(chord_name: str):
    """Get musical notes for a chord"""
    # Simplified note mapping - in a real app, you'd parse the chord name properly
    chord_notes_map = {
        "F": ["F", "A", "C"],
        "F-major": ["F", "A", "C"],
        "F1": ["F", "A", "C"],
        "F2": ["F", "A", "C"],
        "F3": ["F", "A", "C"],
        "F4": ["F", "A", "C"],
        "F5": ["F", "A", "C"],
        "C": ["C", "E", "G"],
        "C-major": ["C", "E", "G"],
        "G": ["G", "B", "D"],
        "G-major": ["G", "B", "D"],
        "A": ["A", "C#", "E"],
        "A-major": ["A", "C#", "E"],
        "D": ["D", "F#", "A"],
        "D-major": ["D", "F#", "A"],
        "E": ["E", "G#", "B"],
        "E-major": ["E", "G#", "B"],
    }
    
    chord_name_clean = chord_name.strip()
    chord_key = chord_name_clean
    if chord_key not in chord_notes_map:
        base_note = chord_name_clean[0] if len(chord_name_clean) > 0 else "C"
        if base_note in chord_notes_map:
            chord_key = base_note
        else:
            chord_key = "C"
    
    return chord_notes_map.get(chord_key, ["C", "E", "G"])


# Diffusion model endpoints
class GenerationRequest(BaseModel):
    query: str
    use_random_prompt: bool = True
    num_inference_steps: int = 50
    guidance_scale: float = 7.5
    negative_prompt: str = ""


@app.post("/generate/image")
async def generate_image(request: GenerationRequest):
    """Generate an image using Stable Diffusion 3 (Diffusers)."""
    global diffusion_generator
    
    try:
        # Lazy load the generator on first use
        if diffusion_generator is None:
            hf_token = os.getenv("HF_TOKEN") or os.getenv("HUGGINGFACE_TOKEN")
            device = "auto"  # Prefer CUDA if available, else MPS, else CPU
            diffusion_generator = get_generator(hf_token=hf_token, device=device)
        
        # Generate the image
        image = diffusion_generator.generate_image(
            query=request.query,
            use_random_prompt=request.use_random_prompt,
            num_inference_steps=request.num_inference_steps,
            guidance_scale=request.guidance_scale,
            negative_prompt=request.negative_prompt,
        )
        
        # Save the generated image
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"generated_{timestamp}.png"
        output_path = diffusion_generator.save_image(
            image=image,
            output_dir=DIFFUSION_OUTPUT_DIR,
            filename=filename
        )
        
        # Also save as JPEG for web display
        jpeg_filename = filename.replace(".png", ".jpg")
        jpeg_path = os.path.join(DIFFUSION_OUTPUT_DIR, jpeg_filename)
        image_rgb = image.convert("RGB")
        image_rgb.save(jpeg_path, "JPEG", quality=95)
        
        # Return the image path and serve it
        return {
            "success": True,
            "image_path": output_path,
            "image_url": f"/generate/image/{jpeg_filename}",
            "filename": jpeg_filename,
            "prompt_used": request.query
        }
        
    except Exception as e:
        print(f"Error generating image: {e}")
        import traceback
        traceback.print_exc()
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/generate/image/{filename}")
async def get_generated_image(filename: str):
    """Serve generated images"""
    # Use absolute path to ensure we're looking in the right place
    filepath = os.path.abspath(os.path.join(DIFFUSION_OUTPUT_DIR, filename))
    
    # Security: ensure the file is within the output directory
    output_dir_abs = os.path.abspath(DIFFUSION_OUTPUT_DIR)
    if not filepath.startswith(output_dir_abs):
        raise HTTPException(status_code=403, detail="Access denied")
    
    if os.path.exists(filepath):
        return FileResponse(filepath)
    else:
        raise HTTPException(
            status_code=404, 
            detail=f"Image not found: {filename}. Checked: {filepath}"
        )

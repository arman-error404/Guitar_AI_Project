import sounddevice as sd
import numpy as np
import tensorflow as tf
import librosa
import cv2
import sys
import os
import soundfile as sf
from datetime import datetime

# Add parent directory to path to allow imports from utils
project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, project_root)

# Match training parameters exactly
SAMPLE_RATE = 22050     
DURATION = 3.0
BINS_PER_OCTAVE = 36
N_BINS = 7 * BINS_PER_OCTAVE  # 252
MAX_LEN = 256  # Time dimension


model = tf.keras.models.load_model(os.path.join(project_root, "models/chord_model.h5"))


from utils.chord_labels import CHORDS


def record_audio():
    print("\n🎸 Recording... Play chord now!")
    
    audio = sd.rec(
        int(SAMPLE_RATE * DURATION),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    
    sd.wait()
    
    print("✅ Done recording\n")
    
    audio_flat = audio.flatten()
    
    # Check audio quality
    print(f"📊 Audio stats: min={audio_flat.min():.4f}, max={audio_flat.max():.4f}, mean={audio_flat.mean():.4f}, std={audio_flat.std():.4f}")
    print(f"   RMS: {np.sqrt(np.mean(audio_flat**2)):.4f}")
    
    if np.abs(audio_flat).max() < 0.01:
        print("⚠️  WARNING: Audio seems very quiet! Check microphone.")
    
    return audio_flat


def save_audio(audio, filename=None):
    """Save audio to file"""
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = os.path.join(project_root, f"recorded_chord_{timestamp}.wav")
    
    sf.write(filename, audio, SAMPLE_RATE)
    print(f"💾 Audio saved to: {filename}")
    return filename


def load_audio(filename):
    """Load audio from file"""
    audio, sr = librosa.load(filename, sr=SAMPLE_RATE, duration=DURATION)
    print(f"📂 Loaded audio from: {filename}")
    print(f"   Sample rate: {sr}, Duration: {len(audio)/sr:.2f}s, Shape: {audio.shape}")
    return audio


def make_cqt(audio):
    # Compute CQT - MATCH TRAINING EXACTLY
    cqt = np.abs(
        librosa.cqt(
            y=audio,
            sr=SAMPLE_RATE,
            bins_per_octave=BINS_PER_OCTAVE,
            n_bins=N_BINS
        )
    )
    
    print(f"📊 Raw CQT shape: {cqt.shape}, min: {cqt.min():.4f}, max: {cqt.max():.4f}, mean: {cqt.mean():.4f}")
    
    # Log scale - MATCH TRAINING: use ref=np.max
    cqt = librosa.amplitude_to_db(cqt, ref=np.max)
    
    print(f"📊 After dB conversion: min: {cqt.min():.4f}, max: {cqt.max():.4f}, mean: {cqt.mean():.4f}")
    
    # Pad or truncate time dimension to MAX_LEN (like training)
    if cqt.shape[1] < MAX_LEN:
        cqt = np.pad(cqt, ((0, 0), (0, MAX_LEN - cqt.shape[1])))
    else:
        cqt = cqt[:, :MAX_LEN]
    
    # IMPORTANT: Training does NOT normalize! The model expects raw dB values
    # Remove normalization - model was trained on raw dB values
    # cqt = (cqt - cqt.min()) / (cqt.max() - cqt.min() + 1e-8)  # REMOVED
    
    # Add channel dimension
    cqt = np.expand_dims(cqt, axis=-1)
    
    print(f"📊 Final CQT shape: {cqt.shape}, min: {cqt.min():.4f}, max: {cqt.max():.4f}, mean: {cqt.mean():.4f}")
    print(f"   Expected shape: (252, 256, 1)")
    
    return cqt


def main():
    print("🎵 Guitar Chord Recognition Ready")
    print("Press ENTER to record, CTRL+C to exit")
    
    while True:
        input("\n▶ Press Enter to record chord...")
        
        # Record audio
        audio = record_audio()
        
        # Save audio to file
        audio_file = save_audio(audio)
        
        # Load audio from file (to simulate the full pipeline)
        audio_loaded = load_audio(audio_file)
        
        # Process audio
        spec = make_cqt(audio_loaded)
        
        spec = np.expand_dims(spec, axis=0)
        print(f"🔢 Model input shape: {spec.shape}")
        
        # Predict
        pred = model.predict(spec, verbose=0)
        pred_flat = pred[0]  # Flatten prediction
        
        # Check if predictions are all the same (model might be broken)
        unique_preds = len(np.unique(pred_flat))
        print(f"🔍 Unique prediction values: {unique_preds} (should be {len(CHORDS)})")
        
        # Get top predictions
        top_indices = np.argsort(pred_flat)[-5:][::-1]  # Top 5
        top_probs = pred_flat[top_indices]
        
        cid = np.argmax(pred_flat)
        conf = float(np.max(pred_flat))
        
        # Debug: Show top 5 predictions
        print("\n🔍 Top 5 Predictions:")
        for i, (idx, prob) in enumerate(zip(top_indices, top_probs), 1):
            chord_name = CHORDS[idx] if idx < len(CHORDS) else f"Index {idx}"
            print(f"   {i}. {chord_name} (index {idx}): {prob:.4f}")
        
        if cid < len(CHORDS):
            chord = CHORDS[cid]
        else:
            chord = f"Unknown (index {cid})"
        
        print(f"\n🎼 Detected Chord:")
        print(f"   ➜ {chord} (index: {cid})")
        print(f"   ➜ Confidence: {conf:.4f}")
        print(f"   ➜ Total classes: {len(CHORDS)}, Model output size: {len(pred_flat)}")
        
        # Additional debug: check if model output is suspicious
        if conf > 0.99:
            print("⚠️  WARNING: Very high confidence - model might be overconfident or broken")
        if unique_preds < 10:
            print("⚠️  WARNING: Model predictions are too uniform - model might not be working correctly")


if __name__ == "__main__":
    main()
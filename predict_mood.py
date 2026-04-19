import numpy as np
import pandas as pd
import librosa
import joblib
import sys
import os

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
OUTPUT_DIR = os.path.join(BASE_DIR, 'outputs')


def _load_artifact(file_name):
    path = os.path.join(OUTPUT_DIR, file_name)
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing artifact: {path}. Run train_mood_model.py to generate outputs first."
        )
    return joblib.load(path)

model           = _load_artifact('best_model.pkl')
scaler          = _load_artifact('scaler.pkl')
le              = _load_artifact('label_encoder.pkl')
feature_columns = _load_artifact('feature_columns.pkl')


def extract_features(file_path, offset=0.0):
    y, sr = librosa.load(file_path, offset=offset, duration=30, sr=22050)

    y_harmonic, y_percussive = librosa.effects.hpss(y)

    chroma    = librosa.feature.chroma_stft(y=y, sr=sr)
    rms       = librosa.feature.rms(y=y)
    spec_cent = librosa.feature.spectral_centroid(y=y, sr=sr)
    spec_bw   = librosa.feature.spectral_bandwidth(y=y, sr=sr)
    rolloff   = librosa.feature.spectral_rolloff(y=y, sr=sr)
    zcr       = librosa.feature.zero_crossing_rate(y)
    mel       = librosa.feature.melspectrogram(y=y, sr=sr)
    tonnetz   = librosa.feature.tonnetz(y=y_harmonic, sr=sr)
    poly      = librosa.feature.poly_features(y=y, sr=sr)
    tempogram = librosa.feature.tempogram(y=y, sr=sr)
    tempo     = librosa.beat.beat_track(y=y, sr=sr)[0]
    mfcc      = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=20)

    tempo_val = float(np.array(tempo).flatten()[0])

    features = {}

    features['chroma_stft_mean']         = float(np.mean(chroma))
    features['chroma_stft_var']          = float(np.var(chroma))
    features['rms_mean']                 = float(np.mean(rms))
    features['rms_var']                  = float(np.var(rms))
    features['spectral_centroid_mean']   = float(np.mean(spec_cent))
    features['spectral_centroid_var']    = float(np.var(spec_cent))
    features['spectral_bandwidth_mean']  = float(np.mean(spec_bw))
    features['spectral_bandwidth_var']   = float(np.var(spec_bw))
    features['rolloff_mean']             = float(np.mean(rolloff))
    features['rolloff_var']              = float(np.var(rolloff))
    features['zero_crossing_rate_mean']  = float(np.mean(zcr))
    features['zero_crossing_rate_var']   = float(np.var(zcr))
    features['harmony_mean']             = float(np.mean(y_harmonic))
    features['harmony_var']              = float(np.var(y_harmonic))
    features['perceptr_mean']            = float(np.mean(y_percussive))
    features['perceptr_var']             = float(np.var(y_percussive))
    features['melspectrogram_mean']      = float(np.mean(mel))
    features['melspectrogram_var']       = float(np.var(mel))
    features['tempogram_mean']           = float(np.mean(tempogram))
    features['tempogram_var']            = float(np.var(tempogram))
    features['tonnetz_mean']             = float(np.mean(tonnetz))
    features['tonnetz_var']              = float(np.var(tonnetz))
    features['poly_features_mean']       = float(np.mean(poly))
    features['poly_features_var']        = float(np.var(poly))
    features['tempo']                    = tempo_val

    for i in range(1, 21):
        features[f'mfcc{i}_mean'] = float(np.mean(mfcc[i-1]))
        features[f'mfcc{i}_var']  = float(np.var(mfcc[i-1]))

    cst_mean = features['chroma_stft_mean']
    rms_mean = features['rms_mean']
    sc_mean  = features['spectral_centroid_mean']
    cst_var  = features['chroma_stft_var']
    sbw_mean = features['spectral_bandwidth_mean']
    sbw_var  = features['spectral_bandwidth_var']
    mel_mean = features['melspectrogram_mean']
    mel_var  = features['melspectrogram_var']

    features['energy_tempo_ratio']      = rms_mean / (cst_mean + 1e-6)
    features['spectral_energy']         = sc_mean  * rms_mean
    features['tonal_brightness']        = cst_mean * sc_mean
    features['chroma_variance_ratio']   = cst_var  / (cst_mean + 1e-6)
    features['brightness_energy_ratio'] = sc_mean  / (rms_mean + 1e-6)
    features['bandwidth_stability']     = sbw_mean / (sbw_var  + 1e-6)
    features['mel_energy_ratio']        = mel_mean / (mel_var  + 1e-6)
    features['harmonic_richness']       = cst_mean * sbw_mean

    return features


def predict_mood(file_path, verbose=True):
    if not os.path.exists(file_path):
        raise FileNotFoundError(f"File not found: {file_path}")

    print("test", file=sys.stderr, flush=True)
    duration = librosa.get_duration(path=file_path)

    # sample 3 segments — start, 40% through, 70% through
    offsets = [
        0,
        duration * 0.4,
        duration * 0.7
    ]

    if verbose:
        print(f"Song duration  : {duration/60:.1f} mins", flush=True)
        print(f"Sampling 3 segments ({', '.join(f'{o:.0f}s' for o in offsets)})\n", flush=True)

    votes     = []
    all_probs = []
    segment_labels = []

    for offset in offsets:
        if verbose:
            print(f"  Extracting at {offset:.0f}s...", flush=True)
        features       = extract_features(file_path, offset=offset)
        row            = {col: features.get(col, 0.0) for col in feature_columns}
        feature_df     = pd.DataFrame([row], columns=feature_columns)
        feature_scaled = scaler.transform(feature_df)

        prediction = model.predict(feature_scaled)[0]
        votes.append(prediction)
        label = le.inverse_transform([prediction])[0]
        segment_labels.append(label)
        if verbose:
            print(f"  → {label}", flush=True)

        if hasattr(model, 'predict_proba'):
            all_probs.append(model.predict_proba(feature_scaled)[0])

    # majority vote across 3 segments
    final_pred = max(set(votes), key=votes.count)
    mood       = le.inverse_transform([final_pred])[0]

    if verbose:
        print(f"\nMajority vote  : {votes.count(final_pred)}/3 segments agree", flush=True)

    result = {
        'mood': mood,
        'votes': votes.count(final_pred),
        'segment_predictions': segment_labels,
        'duration_seconds': float(duration),
    }

    if all_probs:
        avg_probs  = np.mean(all_probs, axis=0)
        confidence = dict(zip(le.classes_, (avg_probs * 100).round(1)))
        confidence = dict(sorted(confidence.items(), key=lambda x: x[1], reverse=True))
        result['confidence'] = confidence

        if verbose:
            print(f"\nPredicted mood : {mood.upper()}", flush=True)
            print("\nConfidence scores (averaged across 3 segments):", flush=True)
            for m, score in confidence.items():
                bar   = '█' * int(score / 5)
                arrow = ' <- predicted' if m == mood else ''
                print(f"  {m:12s} {score:5.1f}%  {bar}{arrow}", flush=True)
    else:
        result['confidence'] = {}
        if verbose:
            print(f"\nPredicted mood : {mood.upper()}", flush=True)

    return result


if __name__ == "__main__":
    if len(sys.argv) < 2:
        file_path = input("Enter path to audio file (.wav or .mp3): ").strip()
    else:
        file_path = sys.argv[1]

    outcome = predict_mood(file_path, verbose=True)
    print(f"\nFinal mood prediction: {outcome['mood'].upper()}")
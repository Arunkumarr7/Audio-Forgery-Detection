import os
import pandas as pd
import librosa
import numpy as np

def extract_chroma(file_path):
    try:
        # Load up to 5 seconds of audio at 16kHz for quick feature extraction
        y, sr = librosa.load(file_path, sr=16000, duration=5.0)
        chroma = librosa.feature.chroma_stft(y=y, sr=sr)
        return np.mean(chroma.T, axis=0)
    except Exception:
        return None

def prepare_dataset(csv_path, audio_dir):
    features = []
    labels = []
    
    # Read the dataset csv
    df = pd.read_csv(csv_path)
    print(f"Total rows found in CSV: {len(df)}")
    
    processed_count = 0
    for _, row in df.iterrows():
        # Get filename from 'file' column
        filename = str(row['file'])
        if not filename.endswith('.wav'):
            filename += '.wav'
            
        # Map labels: bona-fide -> 0 (Real), spoof -> 1 (Fake)
        label_val = str(row['label']).strip().lower()
        if label_val == 'bona-fide':
            label = 0
        elif label_val == 'spoof':
            label = 1
        else:
            continue # Skip row if it doesn't match standard labels
            
        file_path = os.path.join(audio_dir, filename)
        
        # Extract features if the file exists in your data/all_audio folder
        if os.path.exists(file_path):
            feat = extract_chroma(file_path)
            if feat is not None:
                features.append(feat)
                labels.append(label)
                processed_count += 1
                
        if processed_count % 500 == 0 and processed_count > 0:
            print(f"Successfully processed {processed_count} files...")
            
    return np.array(features), np.array(labels)

if __name__ == "__main__":
    CSV_FILE = "data/meta.csv"
    AUDIO_DIR = "data/all_audio"
    
    print("Launching Feature Extraction Pipeline...")
    X, y = prepare_dataset(CSV_FILE, AUDIO_DIR)
    
    if len(X) > 0:
        np.save('X_features.npy', X)
        np.save('y_labels.npy', y)
        print(f"Extraction fully complete! Saved feature matrices shape: {X.shape}")
    else:
        print("Error: No features extracted. Please double-check that your .wav files are inside the 'data/all_audio' folder.")
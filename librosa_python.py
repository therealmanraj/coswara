import glob
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import os
from scipy.io import wavfile

# Function to load and preprocess audio files
def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.size == 0:
            print(f"Warning: Empty audio file {file_path}")
            return None, None
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y)
        return y, sr
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")
        return None, None

# Function to extract features from audio
def extract_features(y, sr):
    features = {
        'mfcc': librosa.feature.mfcc(y=y, sr=sr, n_mfcc=13).mean(axis=1),
        'chroma': librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1),
        'spectral_contrast': librosa.feature.spectral_contrast(y=y, sr=sr).mean(axis=1),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(y).mean(),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        'rms': librosa.feature.rms(y=y).mean()
    }
    return features

# Function to flatten feature dictionary
def flatten_features(features):
    flattened = {}
    for key, values in features.items():
        if isinstance(values, np.ndarray):
            for i, value in enumerate(values):
                flattened[f'{key}_{i}'] = value
        else:
            flattened[key] = values
    return flattened

# Function to plot waveform and save it
def plot_and_save_waveform(y, sr, category, identifier, output_dir):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title('Waveform')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.savefig(os.path.join(output_dir, f'{category}-{identifier}-waveform.png'))
    plt.close()

# Function to plot spectrogram and save it
def plot_and_save_spectrogram(y, sr, category, identifier, output_dir):
    D = np.abs(librosa.stft(y))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Spectrogram')
    plt.savefig(os.path.join(output_dir, f'{category}-{identifier}-spectrogram.png'))
    plt.close()

# Function to plot mel-spectrogram and save it
def plot_and_save_mel_spectrogram(y, sr, category, identifier, output_dir):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title('Mel-Spectrogram')
    plt.savefig(os.path.join(output_dir, f'{category}-{identifier}-mel_spectrogram.png'))
    plt.close()

# Function to plot MFCC and save it
def plot_and_save_mfcc(y, sr, category, identifier, output_dir):
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title('MFCC')
    plt.savefig(os.path.join(output_dir, f'{category}-{identifier}-mfcc.png'))
    plt.close()

# Collect audio file paths
audio_files = {
    'cough_heavy': glob.glob('/Users/manraj/Documents/GitHub/coswara/Extracted_data/202*/*/cough-heavy.wav'),
    'cough_shallow': glob.glob('/Users/manraj/Documents/GitHub/coswara/Extracted_data/202*/*/cough-shallow.wav'),
    'counting_fast': glob.glob('/Users/manraj/Documents/GitHub/coswara/Extracted_data/202*/*/counting-fast.wav'),
    'counting_normal': glob.glob('/Users/manraj/Documents/GitHub/coswara/Extracted_data/202*/*/counting-normal.wav'),
    'breathing_deep': glob.glob('/Users/manraj/Documents/GitHub/coswara/Extracted_data/202*/*/breathing-deep.wav'),
    'breathing_shallow': glob.glob('/Users/manraj/Documents/GitHub/coswara/Extracted_data/202*/*/breathing-shallow.wav'),
    'vowel_a': glob.glob('/Users/manraj/Documents/GitHub/coswara/Extracted_data/202*/*/vowel-a.wav'),
    'vowel_e': glob.glob('/Users/manraj/Documents/GitHub/coswara/Extracted_data/202*/*/vowel-e.wav'),
    'vowel_o': glob.glob('/Users/manraj/Documents/GitHub/coswara/Extracted_data/202*/*/vowel-o.wav')
}

# Extract unique identifiers
identifiers = {key: [os.path.basename(os.path.dirname(file)) for file in files] for key, files in audio_files.items()}

# Process and extract features for each category
feature_dfs = {}
for category, files in audio_files.items():
    features_list = []
    for file in files:
        y, sr = preprocess_audio(file)
        if y is not None and sr is not None:
            features = extract_features(y, sr)
            identifier = os.path.basename(os.path.dirname(file))
            features['identifier'] = identifier
            features_list.append(flatten_features(features))

            # Directory to save plots
            output_dir = os.path.dirname(file)
            print(output_dir)
            
            # Create the directory if it does not exist
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # Plot and save each type of plot
            plot_and_save_waveform(y, sr, category, identifier, output_dir)
            plot_and_save_spectrogram(y, sr, category, identifier, output_dir)
            plot_and_save_mel_spectrogram(y, sr, category, identifier, output_dir)
            # plot_and_save_mfcc(y, sr, category, identifier, output_dir)
    
    feature_dfs[category] = pd.DataFrame(features_list)

# Save the DataFrames to CSV files
for category, df in feature_dfs.items():
    df.set_index('identifier', inplace=True)
    output_dir = '/Users/manraj/Documents/GitHub/coswara/Extracted Librosa Features'
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    df.to_csv(os.path.join(output_dir, f'{category}_features.csv'))

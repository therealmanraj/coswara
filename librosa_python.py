import glob
import pandas as pd
import numpy as np
import librosa
import librosa.display
import matplotlib.pyplot as plt
import seaborn as sns
import os

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

# Collect audio file paths
audio_files = {
    'cough_heavy': glob.glob('/Users/manraj/Desktop/Coswara-Data/Extracted_data/202*/*/cough-heavy.wav'),
    # 'cough_shallow': glob.glob('/Users/manraj/Desktop/Coswara-Data/Extracted_data/202*/*/cough-shallow.wav'),
    # 'counting_fast': glob.glob('/Users/manraj/Desktop/Coswara-Data/Extracted_data/202*/*/counting-fast.wav'),
    # 'counting_normal': glob.glob('/Users/manraj/Desktop/Coswara-Data/Extracted_data/202*/*/counting-normal.wav'),
    # 'breathing_deep': glob.glob('/Users/manraj/Desktop/Coswara-Data/Extracted_data/202*/*/breathing-deep.wav'),
    # 'breathing_shallow': glob.glob('/Users/manraj/Desktop/Coswara-Data/Extracted_data/202*/*/breathing-shallow.wav'),
    # 'vowel_a': glob.glob('/Users/manraj/Desktop/Coswara-Data/Extracted_data/202*/*/vowel-a.wav'),
    # 'vowel_e': glob.glob('/Users/manraj/Desktop/Coswara-Data/Extracted_data/202*/*/vowel-e.wav'),
    # 'vowel_o': glob.glob('/Users/manraj/Desktop/Coswara-Data/Extracted_data/202*/*/vowel-o.wav')
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
            features['identifier'] = os.path.basename(os.path.dirname(file))
            features_list.append(flatten_features(features))
    feature_dfs[category] = pd.DataFrame(features_list)

# Save the DataFrames to CSV files and visualize features
for category, df in feature_dfs.items():
    df.set_index('identifier', inplace=True)
    df.to_csv(f'{category}_features.csv')
    print(f"{category.capitalize()} Features DataFrame:")
    print(df.head())

# Example of visualizing the MFCCs for one category
df = feature_dfs['cough_heavy']
plt.figure(figsize=(14, 8))
sns.boxplot(data=df.filter(like='mfcc'))
plt.title('MFCCs for Cough Heavy')
plt.xlabel('MFCC Coefficients')
plt.ylabel('Values')
plt.show()

# Function to plot waveform
def plot_waveform(y, sr, title='Waveform'):
    plt.figure(figsize=(14, 5))
    librosa.display.waveshow(y, sr=sr)
    plt.title(title)
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.show()

# Function to plot spectrogram
def plot_spectrogram(y, sr, title='Spectrogram'):
    D = np.abs(librosa.stft(y))
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(librosa.amplitude_to_db(D, ref=np.max), sr=sr, x_axis='time', y_axis='log')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

# Function to plot mel-spectrogram
def plot_mel_spectrogram(y, sr, title='Mel-Spectrogram'):
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.amplitude_to_db(S, ref=np.max)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(S_DB, sr=sr, x_axis='time', y_axis='mel')
    plt.colorbar(format='%+2.0f dB')
    plt.title(title)
    plt.show()

# Function to plot MFCC
def plot_mfcc(y, sr, title='MFCC'):
    mfccs = librosa.feature.mfcc(y, sr=sr, n_mfcc=13)
    plt.figure(figsize=(14, 5))
    librosa.display.specshow(mfccs, sr=sr, x_axis='time')
    plt.colorbar()
    plt.title(title)
    plt.tight_layout()
    plt.show()

# Example of plotting for a sample file
file_path = '/Users/manraj/Desktop/Coswara-Data/Extracted_data/20200413/0Rlzhiz6bybk77wdLjxwy7yLDhg1/breathing-deep.wav' # Change to actual file path
y, sr = preprocess_audio(file_path)
if y is not None and sr is not None:
    plot_waveform(y, sr, title='Waveform for Sample File')
    plot_spectrogram(y, sr, title='Spectrogram for Sample File')
    plot_mel_spectrogram(y, sr, title='Mel-Spectrogram for Sample File')
    plot_mfcc(y, sr, title='MFCC for Sample File')

# # Iterate through each category and its corresponding DataFrame
# for category, df in feature_dfs.items():
#     print(f"Visualizations for {category.capitalize()}:")
#     for identifier, row in df.iterrows():
#         file_paths = glob.glob(f'/Users/manraj/Desktop/Coswara-Data/Extracted_data/202*/{identifier}/{category}.wav')
#         for file_path in file_paths:
#             y, sr = preprocess_audio(file_path)
#             if y is not None and sr is not None:
#                 print(f"Identifier: {identifier}")
#                 plt.figure(figsize=(14, 12))
                
#                 plt.subplot(2, 2, 1)
#                 plot_waveform(y, sr, title=f'Waveform for {identifier}')
                
#                 plt.subplot(2, 2, 2)
#                 plot_spectrogram(y, sr, title=f'Spectrogram for {identifier}')
                
#                 plt.subplot(2, 2, 3)
#                 plot_mel_spectrogram(y, sr, title=f'Mel-Spectrogram for {identifier}')
                
#                 plt.subplot(2, 2, 4)
#                 plot_mfcc(y, sr, title=f'MFCC for {identifier}')
                
#                 plt.tight_layout()
#                 plt.show()

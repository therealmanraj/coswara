import glob
import pandas as pd
import numpy as np
from scipy.io import wavfile

# Define audio categories and corresponding file paths
audio_categories = {
    "cough_heavy": "cough-heavy.wav",
    "cough_shallow": "cough-shallow.wav",
    "counting_fast": "counting-fast.wav",
    "counting_normal": "counting-normal.wav",
    "breathing_deep": "breathing-deep.wav",
    "breathing_shallow": "breathing-shallow.wav",
    "vowel_a": "vowel-a.wav",
    "vowel_e": "vowel-e.wav",
    "vowel_o": "vowel-o.wav"
}

# Load the main dataframe
dataframe = pd.read_csv('/Users/manraj/Desktop/Coswara-Data/combined_script_data.csv')
print("Initial DataFrame:")
print(dataframe.head())

# Process each audio category
for category in audio_categories:
    # Get file paths for the category
    audio_csv_files = glob.glob(f'/Users/manraj/Desktop/Coswara-Data/Extracted_Audio_CSV/audio_{category}_data.csv')
    
    # Loop through each file, read the data, and merge with the main dataframe
    for file in audio_csv_files:
        try:
            print(f"Processing file: {file}")
            dataframe_to_combine = pd.read_csv(file)
            dataframe_to_combine.set_index('identifier', inplace=True)
            dataframe = dataframe.merge(dataframe_to_combine, left_on='id', right_on='identifier', how='left')
        except ValueError as e:
            print(f"Error reading {file}: {e}")
        except Exception as e:
            print(f"Unexpected error reading {file}: {e}")

print("DataFrame after merging audio data:")
print(dataframe.head())

# Function to extract the identifier from FILENAME
def extract_id(filename):
    return filename.split('_')[0]

# Get file paths for the annotation files
annotations_csv_files = glob.glob('/Users/manraj/Desktop/Coswara-Data/annotations/*.csv')

# Loop through each annotation file, read the data, and merge with the main dataframe
for file in annotations_csv_files:
    try:
        print(f"Processing file: {file}")
        dataframe_to_combine = pd.read_csv(file)
        dataframe_to_combine['id'] = dataframe_to_combine['FILENAME'].apply(extract_id)
        suffix = file.split("/")[-1].split(".")[0]  # Generate a unique suffix based on the filename
        dataframe = dataframe.merge(dataframe_to_combine, on='id', how='left', suffixes=('', f'_{suffix}'))
    except ValueError as e:
        print(f"Error reading {file}: {e}")
    except Exception as e:
        print(f"Unexpected error reading {file}: {e}")

print("Final DataFrame after merging annotation data:")
print(dataframe.head())

# Save the combined dataframe to a new CSV file
dataframe.to_csv('/Users/manraj/Desktop/Coswara-Data/combined_audio_data_final_annotation.csv', index=True)
print("Final DataFrame saved to '/Users/manraj/Desktop/Coswara-Data/combined_audio_data_final_annotation.csv'")

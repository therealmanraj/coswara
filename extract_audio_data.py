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

# Initialize DataFrames to store the audio data
audio_dfs = {category: pd.DataFrame() for category in audio_categories}

# Process each audio category
for category, audio_file in audio_categories.items():
    # Get file paths for the category
    audio_files = glob.glob(f'/Users/manraj/Documents/GitHub/coswara/Extracted_data/202*/*/{audio_file}')
    
    # Extract identifiers from file paths
    identifiers = [file_path.split('/')[-2] for file_path in audio_files]
    
    # Initialize DataFrames for min and max values
    min_df = pd.DataFrame(index=identifiers, columns=[f"min_{category}"])
    max_df = pd.DataFrame(index=identifiers, columns=[f"max_{category}"])
    
    # Loop through each file, read the data, and store min and max values
    for file in audio_files:
        try:
            print(f"Processing file: {file}")
            samplerate, data = wavfile.read(file)
            id = file.split('/')[-2]
            if len(data) > 0:
                min_value = np.min(data)
                max_value = np.max(data)
                min_df.at[id, f"min_{category}"] = min_value
                max_df.at[id, f"max_{category}"] = max_value
        except ValueError as e:
            print(f"Error reading {file}: {e}")
        except Exception as e:
            print(f"Unexpected error reading {file}: {e}")
    
    # Concatenate min and max DataFrames for the category
    category_df = pd.concat([min_df, max_df], axis=1)
    category_df['identifier'] = category_df.index
    
    # Store the DataFrame in the dictionary
    audio_dfs[category] = category_df

    # Save DataFrame to CSV
    category_df.to_csv(f'/Users/manraj/Documents/GitHub/coswara/Extracted_Audio_CSV/audio_{category}_data.csv', index=False)

    # Print DataFrame for verification
    print(f"DataFrame for {category}:")
    print(category_df.head())

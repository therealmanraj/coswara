import os
import glob
import pandas as pd
import librosa
import numpy as np
import subprocess
from tqdm import tqdm

# Set directory paths
coswara_data_dir = os.path.abspath('.')
extracted_data_dir = os.path.join(coswara_data_dir, 'Extracted_data')

# Check and create directories if needed
if not os.path.exists(extracted_data_dir):
    os.makedirs(extracted_data_dir)

# Extract .tar.gz files
dirs_extracted = set(map(os.path.basename, glob.glob(f'{extracted_data_dir}/202*')))
dirs_all = set(map(os.path.basename, glob.glob(f'{coswara_data_dir}/202*')))
dirs_to_extract = list(dirs_all - dirs_extracted)

for d in tqdm(dirs_to_extract, desc="Extracting .tar.gz files"):
    p = subprocess.Popen(f'cat {coswara_data_dir}/{d}/*.tar.gz.* | tar -xvz -C {extracted_data_dir}/', shell=True)
    p.wait()

print("Extraction process complete!")

# Combine CSV files
csv_files = glob.glob(os.path.join(coswara_data_dir, '202*', '*.csv'))
combined_df = pd.concat([pd.read_csv(csv_file) for csv_file in tqdm(csv_files, desc="Combining CSV files")])
combined_df.to_csv('combinedScriptData.csv', index=False)

# Load the main dataframe
dataframe = pd.read_csv('combinedScriptData.csv')

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

# Load and merge audio CSV files with the main dataframe
for category in tqdm(audio_categories, desc="Merging audio CSV files"):
    audio_csv_file = os.path.join(coswara_data_dir, 'Extracted_Audio_CSV', f'audio_{category}_data.csv')
    if os.path.exists(audio_csv_file):
        audio_df = pd.read_csv(audio_csv_file)
        audio_df.set_index('identifier', inplace=True)
        dataframe = dataframe.merge(audio_df, left_on='id', right_on='identifier', how='left')

# Load and merge annotation CSV files with the main dataframe
annotation_files = glob.glob(os.path.join(coswara_data_dir, 'annotations', '*.csv'))

for file in tqdm(annotation_files, desc="Merging annotation CSV files"):
    annotation_df = pd.read_csv(file)
    annotation_df['id'] = annotation_df['FILENAME'].apply(lambda x: x.split('_')[0])
    suffix = os.path.basename(file).split('.')[0]
    dataframe = dataframe.merge(annotation_df, on='id', how='left', suffixes=('', f'_{suffix}'))

dataframe.to_csv('combined_audio_data_final_annotation.csv', index=False)

# Function to extract features from audio
def extract_features(y, sr):
    return {
        'chroma': librosa.feature.chroma_stft(y=y, sr=sr).mean(axis=1),
        'zero_crossing_rate': librosa.feature.zero_crossing_rate(y).mean(),
        'spectral_rolloff': librosa.feature.spectral_rolloff(y=y, sr=sr).mean(),
        'spectral_centroid': librosa.feature.spectral_centroid(y=y, sr=sr).mean(),
        'rms': librosa.feature.rms(y=y).mean(),
        'duration': librosa.get_duration(y=y)
    }

# Function to load and preprocess audio files
def preprocess_audio(file_path):
    try:
        y, sr = librosa.load(file_path, sr=None)
        if y.size == 0:
            return None, None
        y = librosa.util.normalize(y)
        y, _ = librosa.effects.trim(y)
        return y, sr
    except Exception:
        return None, None

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

# Process and extract features for each category
feature_dfs = {}
for category, filename in tqdm(audio_categories.items(), desc="Extracting audio features"):
    audio_files = glob.glob(os.path.join(coswara_data_dir, 'Extracted_data/202*/*', filename))
    features_list = []
    for file in tqdm(audio_files, desc=f"Processing {category} audio files"):
        y, sr = preprocess_audio(file)
        if y is not None and sr is not None:
            features = extract_features(y, sr)
            features['identifier'] = os.path.basename(os.path.dirname(file))
            features_list.append(flatten_features(features))
    feature_dfs[category] = pd.DataFrame(features_list)

# Load and merge feature DataFrames with the main dataframe
for category, df in tqdm(feature_dfs.items(), desc="Merging feature DataFrames"):
    # df.to_csv(f'{category}_features.csv')
    df = df.add_suffix(f'_{category}')
    df.rename(columns={f'identifier_{category}': 'identifier'}, inplace=True)
    dataframe = dataframe.merge(df, left_on='id', right_on='identifier', how='left')
    dataframe.drop(columns=['identifier'], inplace=True)

# Column mapping
column_mapping = {
    "id": "User ID",
    "a": "Age (number)",
    "covid_status": "Health status (e.g. : positive_mild, healthy,etc.)",
    "record_date": "Date when the user recorded and submitted the samples",
    "ep": "Proficient in English (y/n)",
    "g": "Gender (male/female/other)",
    "l_c": "Country",
    "l_l": "Locality",
    "l_s": "State",
    "rU": "Returning User (y/n)",
    "asthma": "Asthma (True/False)",
    "cough": "Cough (True/False)",
    "smoker": "Smoker (True/False)",
    "test_status": "Status of COVID Test (p->Positive, n->Negative, na-> Not taken Test)",
    "ht": "Hypertension  (True/False)",
    "cold": "Cold  (True/False)",
    "diabetes": "Diabetes  (True/False)",
    "diarrhoea": "Diarrheoa (True/False)",
    "um": "Using Mask (y/n)",
    "ihd": "Ischemic Heart Disease (True/False)",
    "bd": "Breathing Difficulties (True/False)",
    "st": "Sore Throat (True/False)",
    "fever": "Fever (True/False)",
    "ftg": "Fatigue (True/False)",
    "mp": "Muscle Pain (True/False)",
    "loss_of_smell": "Loss of Smell & Taste (True/False)",
    "cld": "Chronic Lung Disease (True/False)",
    "pneumonia": "Pneumonia (True/False)",
    "ctScan": "CT-Scan (y/n if the user has taken a test)",
    "testType": "Type of test (RAT/RT-PCR)",
    "test_date": "Date of COVID Test (if taken)",
    "vacc": "Vaccination status (y->both doses, p->one dose(partially vaccinated), n->no doses)",
    "ctDate": "Date of CT-Scan",
    "ctScore": "CT-Score",
    "others_resp": "Respiratory illnesses other than the listed ones (True/False)",
    "others_preexist": "Pre-existing conditions other than the listed ones (True/False)"
}

# Update true_false_columns with mapped names
true_false_columns = [
    'Proficient in English (y/n)',
    'Returning User (y/n)',
    'Smoker (True/False)',
    'Using Mask (y/n)',
    'Cough (True/False)',
    'Cold  (True/False)',
    'Diarrheoa (True/False)',
    'Breathing Difficulties (True/False)',
    'Sore Throat (True/False)',
    'Fever (True/False)',
    'Fatigue (True/False)',
    'Muscle Pain (True/False)',
    'Asthma (True/False)',
    'Loss of Smell & Taste (True/False)',
    'Chronic Lung Disease (True/False)',
    'Pneumonia (True/False)',
    'Respiratory illnesses other than the listed ones (True/False)',
    'Hypertension  (True/False)',
    'Diabetes  (True/False)',
    'Pre-existing conditions other than the listed ones (True/False)',
    'CT-Scan (y/n if the user has taken a test)',
    'Ischemic Heart Disease (True/False)'
]

# Rename columns in the DataFrame
dataframe = dataframe.rename(columns=column_mapping)

# Fill missing values in true_false_columns with False
for col in true_false_columns:
    if col in dataframe.columns:
        dataframe[col] = dataframe[col].fillna(False)

# Display the DataFrame after transformation
print("DataFrame after renaming and replacing null values with False:")
print(dataframe.head())

# Fill missing values with False for specified columns
columns_to_fill_false = [
    'Date of CT-Scan', 'CT-Score', 'dT', 'Type of test (RAT/RT-PCR)',
    'Date of COVID Test (if taken)', 'Status of COVID Test (p->Positive, n->Negative, na-> Not taken Test)',
    'Vaccination status (y->both doses, p->one dose(partially vaccinated), n->no doses)', 'Locality',
    'iF', 'date', 'test',
]
dataframe[columns_to_fill_false] = dataframe[columns_to_fill_false].fillna(False)

# Update categorical column mappings
replacement_mappings = {
    'Proficient in English (y/n)': {'y': 'Yes', 'n': 'No'},
    'Returning User (y/n)': {'y': 'Yes', 'n': 'No'},
    'Smoker (True/False)': {'True': 'True', 'False': 'False', 'Yes': 'True', 'No': 'False'},
    'Using Mask (y/n)': {'y': 'Yes', 'n': 'No'},
    'CT-Scan (y/n if the user has taken a test)': {'y': 'Yes', 'n': 'No'},
    'Status of COVID Test (p->Positive, n->Negative, na-> Not taken Test)': {'p': 'Positive', 'n': 'Negative', 'na': 'Not Taken Test'},
    'Vaccination status (y->both doses, p->one dose(partially vaccinated), n->no doses)': {'y': 'Both Doses', 'p': 'One Dose', 'n': 'No Doses'},
    'Gender (male/female/other)': {'male': 'Male', 'female': 'Female', 'other': 'Other'},
    'Type of test (RAT/RT-PCR)': {'rtpcr': 'RT-PCR', 'rat': 'RAT'},
    'Health status (e.g. : positive_mild, healthy,etc.)': {
        'healthy': 'Negative', 'positive_mild': 'Positive', 'no_resp_illness_exposed': 'Negative',
        'positive_moderate': 'Positive', 'resp_illness_not_identified': 'Illness Not Identified',
        'recovered_full': 'Negative', 'positive_asymp': 'Positive', 'under_validation': 'Illness Not Identified'
    }
}
# Apply the replacements
for col, mapping in replacement_mappings.items():
    if col in dataframe.columns:
        dataframe[col] = dataframe[col].replace(mapping)

dataframe.to_csv('librosa_features.csv', index=False)
print("Final DataFrame saved to 'librosa_features.csv'")
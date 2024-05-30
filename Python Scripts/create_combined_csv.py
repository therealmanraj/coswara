import glob
import pandas as pd

# Get a list of all CSV files in a directory
csv_files = glob.glob('/Users/manraj/Desktop/Coswara-Data/Extracted_data/202*/*/*.csv')
print(csv_files)

# Create an empty dataframe to store the combined data
combined_df = pd.DataFrame()

# Loop through each CSV file and append its contents to the combined dataframe
for csv_file in csv_files:
    df = pd.read_csv(csv_file)
    combined_df = pd.concat([combined_df, df])

print(combined_df)
combined_df.to_csv('combinedScriptData.csv', index=False)
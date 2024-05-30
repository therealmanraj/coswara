# coswara
Steps: 1: 
	run extract_data.py file
2: 
	change in extract_audio_data.py
	change line 25 to the extracted data folder 	audio_files = glob.glob(f'/Users/manraj/Documents/GitHub/coswara/Extracted_data/202*/*/{audio_file}')
	Change till the extracted folder using copy path leave /202*/*/{audio_file}
	change line 58 as well
	run extract_audio_data.py 3:
	change in create_combined_csv.py
	change line 5
	run create_combined_csv.py
4:
	change in combine_data_final.py 	change line 20
	change line 27
	change line 49
	run the combine_data_final.py

CSV combined_audio_data_final_annotation.csv is created The data is less because of choosing only one folder. To get more data just copy the folders directly into the directory.

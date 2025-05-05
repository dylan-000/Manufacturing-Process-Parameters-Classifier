
"""
Created on Tue Jan 14 12:42:10 2025

@author: Dylan Fisher

This program will: 
    
    - take the top-n microphones in each test file (already calculated and accesesible via a .csv file)
    - for each microphone file, segment it into (user-entered length in seconds) divisions of the recording. 
    - for each section, autocorrelation and fast fourier transformation may be optionally applied to the data
    - finally, labels will be applied based on the parameters present in each test (accessible via an excel file containing info on each test)
    
"""

import pandas as pd
import numpy as np 
from pathlib import Path
from utils import export_dataset, fft, autocorrelate, parse_string, sort_test_data, rpm_labeler, feed_rate_labeler, doc_labeler
from scipy.signal import resample

#--------------------- Establishing the options for the creation of the dataset ---------------------#

# Get the top-n mics to incorporate into the dataset
included_mics = int(input("Enter an Integer number for the number of top mic files to use from each test: "))
segment_size_sec = float(input('Enter a floating point value for the segment size in seconds: '))

autocorrelation_choice = input("Would you like to apply autocorrelation on the data? [y/n]: ")
fourier_transform_choice = input("Would you like to apply a fourier transformation on the data? [y/n]: ")

resample_choice = input("Would you like to resample the data? Not doing this will result in a non-uniform segment size. [y/n]: ")

# Convert inputs to boolean
if autocorrelation_choice == 'y':
    autocorrelation_choice = True
else:
    autocorrelation_choice = False
    
if fourier_transform_choice == 'y':
    fourier_transform_choice = True
else:
    fourier_transform_choice = False

if resample_choice == 'y':
    resample_choice = True
else:
    resample_choice = False
    
#-----------------------  Obtaining the Mic rankings -------------------------#

# Open the mic rankings and gather the top microphones as specified by the user
mic_rankings_path = "top32_absolute_value_sum_ranked_mic_scores_per_test_new.csv"
mic_rankings = pd.read_csv(mic_rankings_path)

# Obtain the paths of the mic files in each test file. Format is Test Number: [mic1.txt, mic32.txt] 
test_file_dict = {} 

grouped_tests = mic_rankings.groupby('Test File')

for test_file, group in grouped_tests:
    sorted_group = group.sort_values(by='Absolute Sum', ascending=False)
    top_n_mics = sorted_group['Mic#'].head(included_mics).tolist()
    test_file_dict[test_file] = top_n_mics

# ----------------------- Parsing Data into Dataframe ------------------------#

# Supply the root directory of the test files. read in a dataframe that holds the process parameters for each test file
test_files_root = Path("C:/Users/fish2/Desktop/Research/Test Files (Time Domain)") 
test_file_labels = pd.read_csv('test_labels_guide.csv')

# Create the dataframe outline and create a temporary list to store the row data that we accumulate
dataset = pd.DataFrame(columns=['Test_Number', 'Mic_Number', 'Segment_Number', 'Sample_Data', 'Sample_Rate', 'DOC', 'RPM', 'Feed_Rate'])
rows = [] # Holds the row data

# Iterate through each file in the root directory and then get to the test files within each submodule. Structure is root -> submodule -> test file -> mic file
for submodule in test_files_root.iterdir():
    for test_file in submodule.iterdir():
        for mic_file in test_file.iterdir():
            
            if mic_file.name in test_file_dict.get(test_file.name): # find the correct mic files

                # Open the file, read in the data, and parse the header information.
                with open(mic_file, 'r') as file: 
                    file_lines = file.readlines()

                # Header Information relevant to the number of samples, sample rate   
                header_lines = file_lines[:4][1].split() 
                lines = file_lines[4:]
                sample_rate = int(header_lines[4]) 
                total_samples = int(header_lines[5])

                # Defining the number of samples that go into each segment. We take the user-inputted segment size and multiply it by the sample rate to get the number of samples per segment
                samples_per_segment = int(segment_size_sec * sample_rate) 
                
                # Collect the segmented data
                numbers = list([float(num) for line in lines for num in line.split()])
                segments = [numbers[i:i + samples_per_segment] for i in range(0, len(numbers), samples_per_segment)]
                
                # Process each segment and add it to the DataFrame
                for segment_idx, segment in enumerate(segments):
                    if len(segment) < samples_per_segment:  # Skip incomplete segments
                        continue
                    
                    # Apply transformations if specified
                    if resample_choice:
                        segment = resample(x=segment, num=1200) # convert to 12000 Hz becuase it is the most common rate in the recorded samples
                        sample_rate = 12000 # change the sample rate to 12000 Hz
                    if autocorrelation_choice:
                        segment = autocorrelate(segment)
                    if fourier_transform_choice:
                        segment = fft(segment)

                    # Prepare the row data
                    test_number = int(parse_string(test_file.name))
                    row_data = {
                        'Test_Number': test_number,
                        'Mic_Number': parse_string(mic_file.name),
                        'Segment_Number': segment_idx + 1,
                        'Sample_Data': np.array(segment, dtype=float),
                        'DOC': doc_labeler(float(test_file_labels.iloc[test_number - 1]['Depth of Cut (in)'])),
                        'RPM': rpm_labeler(int(test_file_labels.iloc[test_number - 1]['speed (rpm)'])),
                        'Feed_Rate': feed_rate_labeler(float(test_file_labels.iloc[test_number - 1]['Feed Rate (in/rev)'])),
                        'Sample_Rate': sample_rate
                    }

                    # Append to the dataset DataFrame
                    rows.append(row_data)  
        print(f"Processed {test_file.name}")

# Export the data. Default to 15 tests per file. This means multiple folders are created.
dataset = sort_test_data(pd.DataFrame(rows))
export_dataset(dataset, top_mics=included_mics, tests_per_set=15, fft=fourier_transform_choice, autocorr=autocorrelation_choice)

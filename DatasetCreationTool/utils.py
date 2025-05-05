import pandas as pd
import numpy as np 
import os
import re
import matplotlib.pyplot as plt
from pathlib import Path
from scipy.signal import resample

'''

Various Utility functions used in this project

'''

# Labeling functions for the process parameters. It is important that they are zero-indexed -- the keras models prefer this format.
def doc_labeler(DOC: float) -> int:
    if DOC == .005:
        return 0
    elif DOC == .01:
        return 1
    elif DOC == .02:
        return 2
    elif DOC == .06:
        return 3
    else: 
        return -1
    
def feed_rate_labeler(feed_rate: float) -> int:
    if feed_rate == 0.0050:
        return 0
    elif feed_rate == 0.0058:
        return 1
    elif feed_rate == 0.0074:
        return 2
    elif feed_rate == 0.0101:
        return 3
    elif feed_rate == 0.0130:
        return 4
    elif feed_rate == 0.0152:
        return 5
    elif feed_rate == 0.0087:
        return 6
    else:
        return -1  # Return -1 if the feed_rate does not match any of the specified values

def rpm_labeler(rpm: int) -> int:
    if rpm == 800:
        return 0
    elif rpm == 1200:
        return 1
    elif rpm == 1400:
        return 2
    else: 
        return -1  # Return -1 if the rpm does not match any of the specified values
    
# Super helpful parsing function for parsing some of the string names of the files and returning their values as integers
def parse_string(test_num_string) -> int: 
    match = re.search(r'\d+', test_num_string)
    if match:
        return int(match.group())  # Convert to integer if needed
    return None  # Return None if no number is found

# Autocorrelates the data from a segment of time-domain data
def autocorrelate(signal_data_segment):
    signal_data_segment = np.array(signal_data_segment) # set the data to numpy array
    corrs = np.correlate(signal_data_segment, signal_data_segment, mode='same') 
    return corrs

# Performs FFT on a segment of data
def fft(signal_data_segment):
    signal_data_segment = np.array(signal_data_segment)
    return np.fft.fft(signal_data_segment)

# Sorts the test data in increasing order from: 'Test_Number', 'Mic Number', 'Segment Number'
def sort_test_data(dataset):
    return dataset.sort_values(by=['Test_Number', 'Mic_Number', 'Segment_Number'], ascending=[True, True, True])

# exports a dataframe given some parameters relating to the transformations applied to the dataset
def export_dataset(dataset, top_mics: int, tests_per_set: int, fft: bool, autocorr: bool ):
    
    # For every n tests, create a .csv file
    max_test = dataset['Test_Number'].nunique()
    tests_per_segment = tests_per_set
    
    file_name = f'Top{top_mics}mics_fft={fft}_autocorr={autocorr}'
    os.mkdir(file_name)
    os.chdir(file_name)
    
    # Group by test numbers and save segment files
    for start_test in range(0, max_test + 1, tests_per_segment):
        end_test = min(start_test + tests_per_segment - 1, max_test)
        
        # Filter the DataFrame for the current 15-test segment
        segment_tests = dataset['Test_Number'].unique()[start_test:end_test + 1]
        segment_df = dataset[dataset['Test_Number'].isin(segment_tests)]
        
        # Save to CSV
        segment_filename = f"processed_dataset_tests_{start_test}_to_{end_test}.parquet"
        segment_df.to_parquet(segment_filename, engine='pyarrow', index=False) # by setting the index to false, we ensure that unnecessary indexes are not added to the data file
        print(f"Saved {segment_filename}")

    print("Dataset processing and segmentation complete!")    

# plots the FFT of a segment of data
def plot_fft(fft_data, sample_rate, sample_duration, test_num, mic_num, segment_num):   
    
    # Create the title of the plot
    fft_title = f"FFT of Test {test_num}, Mic {mic_num}, Segment {segment_num}"
    
    # Calculate the quantity of samples per channel
    samples_per_channel = int(round(sample_rate * sample_duration, 0))
    
    # Freq bin calculation and retirving the magnitude of the FFT data
    freqs = np.fft.fftfreq(samples_per_channel, 1 / sample_rate)[:samples_per_channel // 2]
    magnitude = np.abs(fft_data) / samples_per_channel
    magnitude = magnitude[:len(freqs)]
    
    # Cut off data past 11,000 Hz
    cutoff = 11000
    valid_indices = freqs <= cutoff
    freqs = freqs[valid_indices]
    magnitude = magnitude[valid_indices]

    # Create the figure and subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6), sharey=True)
    
    # --- Plot Window 1: 0–3000 Hz ---
    window_1_indices = freqs <= 3000
    ax1.plot(freqs[window_1_indices], magnitude[window_1_indices], color='purple')
    ax1.set_title(fft_title + ' (0–3000 Hz) ')
    ax1.set_xlabel("Frequency (Hz)")
    ax1.set_ylabel("Magnitude")
    ax1.grid(which='both', linestyle='--', linewidth=0.5)
    
    # Major and minor ticks for 0–3000 Hz
    major_ticks_1 = np.arange(0, 3001, 1000)
    minor_ticks_1 = np.arange(0, 3001, 100)
    ax1.set_xticks(major_ticks_1)
    ax1.set_xticks(minor_ticks_1, minor=True)
    ax1.set_xticklabels([f"{int(tick)} Hz" for tick in major_ticks_1])

    # --- Plot Window 2: 3000–11,000 Hz ---
    window_2_indices = (freqs > 3000) & (freqs <= 11000)
    ax2.plot(freqs[window_2_indices], magnitude[window_2_indices], color='green')
    ax2.set_title(fft_title + ' (3000–11,000 Hz) ')
    ax2.set_xlabel("Frequency (Hz)")
    ax2.grid(which='both', linestyle='--', linewidth=0.5)
    
    # Auto-spaced ticks for 3000–11,000 Hz
    major_ticks_2 = np.linspace(3000, 11000, 5)  # 5 evenly spaced major ticks
    ax2.set_xticks(major_ticks_2)
    ax2.minorticks_on()  # Enable minor ticks (automatic spacing)

    # Adjust layout and display
    plt.tight_layout()
    plt.show()
    

# Exploring the data for demoing purposes
if __name__ == '__main__': 

    root = Path("C:/Users/fish2/ResearchWork/DatasetCreationTool/Top3mics_fft=True_autocorr=True")

    for file in root.iterdir():

        df = pd.read_parquet(file)
        print(df.columns)
        
        
import h5py
import os
import numpy as np
from tqdm import tqdm

def count_timestamps(file_path):
    """count the number of timestamps in a single hdf5 file"""
    try:
        with h5py.File(file_path, 'r') as f:
            # assume the timestamps are the direct sub-groups of data group
            if 'data' in f:
                return len(f['data'].keys())
            else:
                # try to find the top level timestamps
                return len(f.keys())
    except Exception as e:
        print(f"error when process file {file_path}: {str(e)}")
        return 0

def main():
    # the directory of the files, modify as needed
    directory = "."  # current directory, modify as needed
    
    total_timestamps = 0
    file_count = 0
    timestamp_counts = []
    
    # traverse data_100.hdf5 to data_500.hdf5
    for i in tqdm(range(100, 215)):
        file_name = f"data_{i}.hdf5"
        file_path = os.path.join(directory, file_name)
        
        if os.path.exists(file_path):
            file_count += 1
            timestamps = count_timestamps(file_path)
            total_timestamps += timestamps
            timestamp_counts.append(timestamps)
            # print(f"{file_name}: {timestamps} 个时间戳")
    
    # statistics and output results    
    print(f"\nprocessed {file_count} files")
    print(f"total found {total_timestamps} timestamps")
    
    if timestamp_counts:
        print(f"average {total_timestamps / file_count:.2f} timestamps per file")
        print(f"least timestamps: {min(timestamp_counts)}")
        print(f"most timestamps: {max(timestamp_counts)}")
        
        # calculate the timestamp distribution
        unique_counts = np.unique(timestamp_counts, return_counts=True)
        for count, frequency in zip(unique_counts[0], unique_counts[1]):
            print(f"{count} timestamps: {frequency} files")

if __name__ == "__main__":
    main()
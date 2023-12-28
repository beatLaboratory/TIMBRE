"""
A set of helper functions for downloading and preprocessing hippocampal data.

@author: Gautam Agarwal
"""
from scipy.signal import decimate
import numpy as np
import math
import os
from scipy import io
from scipy.stats import mode
    
def get_behav(mat_file, fs = 25):
    """
    Organizes information about rat behavior into a matrix
    
    Input:
    mat_file: file containing behavioral data
    fs = sampling frequency

    Output:
    lapID = Data stored in the format listed below

    lapID format:
    Column 0: Trial Number
    Column 1: Maze arm (-1/0/1/2) (-1 = not in maze arm)
    Column 2: Correct (0/1)
    Column 3: other/first approach/port/last departure (0/1/2/3)
    Column 4: position in mm
    """    
    dec = int(1250/fs) #decimation factor
    mat = io.loadmat(mat_file, variable_names = ['Track'])
    lapID = np.array([ np.squeeze(mat['Track']["lapID"][0][0])[::dec] ],dtype='float32') - 1
    lapID = np.append(lapID, [ np.squeeze(mat['Track']["mazeSect"][0][0])[::dec] ], axis = 0)
    lapID = np.append(lapID, [ np.squeeze(mat['Track']["corrChoice"][0][0])[::dec] ], axis = 0)
    lapID = np.append(lapID, np.zeros((1, len(lapID[0]))), axis = 0)
    lapID = np.append(lapID, decimate(mat['Track']["xMM"][0][0].T,dec), axis =0)
    lapID = np.append(lapID, decimate(mat['Track']["yMM"][0][0].T,dec), axis =0)
    lapID = lapID.T
    
    # Filter values and construct column 3
    in_arm = np.in1d(lapID[:,1], np.array(range(4, 10))) #rat is occupying a maze arm
    in_end = np.in1d(lapID[:,1], np.array(range(7, 10)))
    #lapID[np.in1d(lapID[:,1], np.array(range(4, 10)), invert = True), 1] = -1
    lapID[in_arm, 1] = (lapID[in_arm, 1] - 1) % 3
    lapID[~in_arm,1] = -1
    #lapID[lapID[:, 1] == 0, :] = 0

    for i in range(int(np.max(lapID[:,0]))):
        r = np.logical_and(lapID[:, 0] == i, in_end)#lapID[:, 3] == 2)
        inds = np.where(np.logical_and(lapID[:, 0] == i, in_arm))[0]
        all_end = np.where(r)[0]
        #if all_end.size > 0: #valid trial where rat goes to end of arm
        lapID[inds[inds < all_end[0]], 3] = 1
        lapID[inds[inds > all_end[-1]], 3] = 3
        lapID[longest_stretch(r),3] = 2

    # Return structured data
    return lapID

def longest_stretch(bool_array):
    """
    Finds longest contiguous stretch of True values

    Input:
    bool_array = boolean vector

    Output:
    bool_most_common = boolean vector, True only for longest stretch of 'True' in bool_array
    """
    bool_array_diff = np.append(bool_array[0],bool_array)
    bool_array_diff = np.cumsum(np.abs(np.diff(bool_array_diff)))
    bool_most_common = bool_array_diff == mode(bool_array_diff[bool_array])[0]
    return bool_most_common
    

def get_spikes(mat_file,fs = 25):
    """
    Counts spikes for each neuron in each time bin
    
    Input:
    mat_file = file containing spike data
    fs = sampling rate
    
    Output:
    sp = binned spikes
    """
    mat = io.loadmat(mat_file, variable_names = ['Spike', 'Clu','xml'])
    n_channels = mat['xml']['nChannels'][0][0][0][0]
    dec = int(1250/fs)
    max_spike_res = np.ceil(np.max(mat['Spike']['res'][0][0])/dec) + 1
    max_spike_clu = np.max(mat['Spike']['totclu'][0][0]) + 1# Precompute the bins
    bins_res = np.arange(max_spike_res)
    bins_clu = np.arange(max_spike_clu)
    spike_res = np.squeeze(mat['Spike']['res'][0][0]) // dec
    spike_clu = np.squeeze(mat['Spike']['totclu'][0][0]) - 1

    # Bin both dimensions using histogram2d.
    sp, _,_ = np.histogram2d(spike_res, spike_clu, bins = (bins_res, bins_clu) )
    sp = sp.astype(np.uint8)
    
    mask = mat['Clu']['shank'][0][0][0] <= math.ceil(n_channels / 8)
    sp = sp[:, mask]
    
    return sp

def get_LFP(lfp_file, n_channels, fs = 25):
    """
    Decimates LFPs to desired sampling rate
    
    Input:
    lfp_file = raw lfp data file of type .lfp
    fs = sampling rate

    Output:
    X = formatted lfp data
    """
    dec = int(1250/fs)
    file_size = os.path.getsize(lfp_file)
    data_size = np.dtype('int16').itemsize
    total_elements = file_size // data_size
    n_samples = total_elements // n_channels

    # Clip the rows to remove electrodes implanted in mPFC.
    if n_channels > 256: #sessions 1 and 2
        n_keep = 255
    else:                #sessions 3 and 4
        n_keep = 192

    #Load and decimate the data (takes more memory!)
    #slice_data = np.memmap(lfp_file, dtype='int16', mode='r', shape=(n_samples, n_channels))
    #X = decimate(slice_data[:, :n_keep], dec, axis=0)
    
    
    # Process each channel individually and store in the pre-allocated array (takes less memory)
    final_length = math.ceil(n_samples / dec)
    X = np.zeros((final_length, n_keep), dtype=np.float32)
    for channel in range(n_keep):
        # Load the channel data using memmap
        channel_data = np.memmap(lfp_file, dtype='int16', mode='r', shape=(n_samples, n_channels))[:, channel]
        # Decimate the channel data
        X[:, channel] = decimate(channel_data, dec, axis=0)
        print(channel)

    return X
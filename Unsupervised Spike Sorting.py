import scipy.io as spio
import numpy as np
from scipy.signal import find_peaks
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

# Load the .mat file (adjust the loop for D2-D6 paths)
for ds in ['D2','D3','D4','D5','D6']:
    mat = spio.loadmat(f'CI/CW C/Coursework C Datasets/{ds}.mat', squeeze_me=True)
    d = mat['d']

    # 1. Bandpass filter (300-3000 Hz recommended)
    from scipy.signal import butter, filtfilt
    def bandpass(data, fs=25000, low=300, high=3000, order=3):
        b, a = butter(order, [low/(fs/2), high/(fs/2)], btype='band')
        return filtfilt(b, a, data)
    d_filt = bandpass(d)

    # 2. Spike detection using threshold (e.g. 4x std)
    threshold = 4 * np.std(d_filt)
    peak_indices, _ = find_peaks(np.abs(d_filt), height=threshold, distance=100) # enforce minimum spacing

    # 3. Extract spike segments
    window_size = 250
    spike_segments = []
    valid_indices = []
    for idx in peak_indices:
        if idx-window_size >= 0 and idx+window_size < len(d_filt):
            segment = d_filt[idx-window_size:idx+window_size]
            spike_segments.append(segment)
            valid_indices.append(idx)
    spike_segments = np.array(spike_segments)

    # 4. PCA Feature extraction (reduce dimensionality)
    pca = PCA(n_components=10)
    spike_features = pca.fit_transform(spike_segments)

    # 5. KMeans clustering to assign classes
    n_classes = 5
    km = KMeans(n_clusters=n_classes, random_state=42)
    class_labels = km.fit_predict(spike_features)

    # 6. Output vectors
    Index = np.array(valid_indices)
    Class = np.array(class_labels)

    # After computing Index and Class for each dataset D2-D6:
    output_dict = {'Index': Index, 'Class': Class}
    spio.savemat(f'Output_{ds}.mat', output_dict)

    print(f"{ds}: Detected {len(Index)} spikes. Class distribution: {np.bincount(Class)}")
    # Now Index and Class are available for each dataset
    # You can save or use them for downstream CNN classification


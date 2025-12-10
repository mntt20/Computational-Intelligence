import scipy.io as spio
import numpy as np
from scipy.signal import butter, filtfilt, find_peaks
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

fs = 25000
window_size = 250      # samples before/after spike peak (adjust if needed)
bands = [(300,700), (500,1500), (1500,3000)]  # Multi-frequency bands

def multi_band_filter(data, fs, band_limits):
    filt_segments = []
    for low, high in band_limits:
        b, a = butter(3, [low/(fs/2), high/(fs/2)], btype='band')
        filt_data = filtfilt(b, a, data)
        filt_segments.append(filt_data)
    return filt_segments

for ds in ['D2','D3','D4','D5','D6']:
    # Load
    mat = spio.loadmat(f'CI/CW C/Coursework C Datasets/{ds}.mat', squeeze_me=True)
    d = mat['d']
    
    # Multi-band filtering
    filtered_signals = multi_band_filter(d, fs, bands)
    
    # Combine to a single envelope for robust spike detection
    envelope = np.sum([np.abs(sig) for sig in filtered_signals], axis=0)
    
    # Spike detection (simple thresholding)
    thr = 4 * np.std(envelope)
    indices, _ = find_peaks(envelope, height=thr, distance=100)
    
    spike_features = []
    valid_indices = []
    for idx in indices:
        # Boundary check
        if idx - window_size >= 0 and idx + window_size < len(d):
            combined = []
            # For each filtered band, extract waveform segment
            for sig in filtered_signals:
                seg = sig[idx-window_size : idx+window_size]
                combined.extend(seg)
            spike_features.append(combined)
            valid_indices.append(idx)
    
    spike_features = np.array(spike_features)
    
    # PCA feature reduction (to 10 components)
    pca = PCA(n_components=10)
    spike_pca = pca.fit_transform(spike_features)
    
    # K-means clustering (5 clusters)
    kmeans = KMeans(n_clusters=5, random_state=42)
    spike_classes = kmeans.fit_predict(spike_pca) + 1   # use 1-based classes
    
    # Save results to .mat
    output_dict = {'Index': np.array(valid_indices), 'Class': spike_classes}
    spio.savemat(f'Multiband_KMeans_{ds}.mat', output_dict)
    print(f"Processed {ds}: {len(valid_indices)} spikes clustered.")

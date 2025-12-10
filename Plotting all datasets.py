import scipy.io as spio
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt, find_peaks

fs = 25000
window_size = 250
b, a = butter(3, [300/(fs/2), 3000/(fs/2)], btype='band')

for ds in ['D1','D2','D3','D4','D5','D6']:
    mat = spio.loadmat(f'CI\\CW C\\Coursework C Datasets\\{ds}.mat', squeeze_me=True)
    d = mat['d']
    d_filt = filtfilt(b, a, d)

    if ds == 'D1':
        Index = mat['Index']
        spike_idx = Index[0]
    else:
        thr = 4*np.std(d_filt)
        peaks, _ = find_peaks(np.abs(d_filt), height=thr, distance=100)
        if len(peaks) == 0:
            print(f'No spikes found in {ds}')
            continue
        spike_idx = peaks[0]

    start = spike_idx - window_size
    end = spike_idx + window_size
    if start < 0 or end > len(d_filt):
        continue

    seg = d_filt[start:end]
    t = np.linspace(-window_size/fs, window_size/fs, len(seg))

    plt.figure()
    plt.plot(t, seg)
    plt.title(f'Example spike from {ds} at sample {spike_idx}')
    plt.xlabel('Time (s)')
    plt.ylabel('Amplitude')
    plt.tight_layout()
    plt.show()
